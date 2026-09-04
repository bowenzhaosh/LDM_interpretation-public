import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from .generative import (
    ContextGroup,
    generate_group,
    sample_context,
    sample_valid_sigma,
    sigma_to_params,
)
from .model import (
    configure_determinism,
    load_registered_checkpoint,
    predict_probabilities,
)
from .oracle import GridOracle
from .provenance import (
    current_head,
    derive_seed,
    enforce_cost_gate,
    evaluation_root,
    load_locked_query_banks,
    require_scientific_run_path,
    repository_root,
    validate_locked_validations,
    verify_panel_lock,
    verify_run_lock,
)
from .registry import (
    expanded_checkpoint_record,
    load_checkpoint_registry,
    sha256_file,
)
from .storage import load_numeric_npz, write_json_atomic, write_numeric_npz_atomic

FAILED_SCIENTIFIC_COMMITS = {
    "d0b049d6241845e55443f4950e52b70644b2b1ab",
}


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _require_current_panel_commit(commit_sha: str, root: Path) -> None:
    if commit_sha in FAILED_SCIENTIFIC_COMMITS:
        raise ValueError("scientific panel belongs to an immutable failed stream")
    if commit_sha != current_head(root):
        raise ValueError("scientific panel commit is not the current repository HEAD")


def _select_interior_groups(
    *,
    commit_sha: str,
    query_banks: np.ndarray,
    n_groups: int,
    n_continuations: int,
    max_core_candidates: int,
    max_blocks_per_core: int,
    min_within_group_sd: float,
    interior_weight_min: float = 0.05,
    interior_weight_max: float = 0.95,
    minimum_endpoint_js: float = 0.10,
    seed_namespace: str | None = None,
    seed_root: str | None = None,
):
    if not 0.0 < interior_weight_min < interior_weight_max < 1.0:
        raise ValueError("interior-weight thresholds must lie strictly inside (0,1)")
    if not 0.0 <= minimum_endpoint_js <= 1.0:
        raise ValueError("minimum endpoint JS must lie in [0,1]")
    seed_root = commit_sha if seed_root is None else seed_root
    if not isinstance(seed_root, str) or len(seed_root) != 40:
        raise ValueError("selection seed root must be a full commit SHA")
    if seed_namespace is not None:
        if (
            not isinstance(seed_namespace, str)
            or not seed_namespace
            or seed_namespace.startswith(":")
            or seed_namespace.endswith(":")
        ):
            raise ValueError("seed namespace must be a nonempty colon-free-boundary string")
        seed_prefix = f"{seed_namespace}:"
    else:
        seed_prefix = ""
    primary_oracle = GridOracle(query_banks[0], quadrature=15)
    alternate_oracle = GridOracle(query_banks[1], quadrature=15)
    required_blocks = n_continuations + 1
    selected = []
    core_sigma = []
    core_graph = []
    core_params = []
    core_context = []
    core_seed = []
    core_reason = []  # 0 accepted, 1 insufficient blocks, 2 insufficient ell variation
    core_acceptance_rank = []
    block_context = []
    block_core_index = []
    block_index = []
    block_seed = []
    block_ell = []
    block_js_primary = []
    block_js_alternate = []
    block_reason = []  # 0 eligible, 1 noninterior, 2 primary JS, 3 alternate JS
    block_eligible_rank = []

    for core_index in range(max_core_candidates):
        this_core_seed = derive_seed(
            seed_root, f"{seed_prefix}core-candidate:{core_index}"
        )
        core_rng = np.random.default_rng(this_core_seed)
        covariance = sample_valid_sigma(core_rng)
        graph = int(core_rng.integers(0, 2))
        parameters = sigma_to_params(covariance, graph)
        core = sample_context(core_rng, graph, parameters, 20)
        core_sigma.append(covariance)
        core_graph.append(graph)
        core_params.append([parameters.beta, parameters.b_root, parameters.b_effect])
        core_context.append(core)
        core_seed.append(this_core_seed)
        passing_blocks = []
        passing_ells = []

        for candidate_index in range(max_blocks_per_core):
            this_block_seed = derive_seed(
                seed_root,
                f"{seed_prefix}core-candidate:{core_index}:block:{candidate_index}",
            )
            block_rng = np.random.default_rng(this_block_seed)
            block = sample_context(block_rng, graph, parameters, 10)
            context = np.concatenate([core, block], axis=0)
            ell = primary_oracle.log_evidence(context)
            weight = float(expit(ell))
            js_primary = -1.0
            js_alternate = -1.0
            reason = 1
            if interior_weight_min <= weight <= interior_weight_max:
                primary_bundle = primary_oracle.evaluate(context)
                if abs(primary_bundle.ell - ell) > 1e-10:
                    raise AssertionError("selection evidence changed during endpoint evaluation")
                js_primary = primary_bundle.js
                if js_primary >= minimum_endpoint_js:
                    alternate_bundle = alternate_oracle.evaluate(context)
                    if abs(alternate_bundle.ell - ell) > 1e-10:
                        raise AssertionError("query banks disagree on selection evidence")
                    js_alternate = alternate_bundle.js
                    reason = 0 if js_alternate >= minimum_endpoint_js else 3
                else:
                    reason = 2
            block_context.append(block)
            block_core_index.append(core_index)
            block_index.append(candidate_index)
            block_seed.append(this_block_seed)
            block_ell.append(ell)
            block_js_primary.append(js_primary)
            block_js_alternate.append(js_alternate)
            block_reason.append(reason)
            if reason == 0:
                eligible_rank = len(passing_blocks)
                block_eligible_rank.append(eligible_rank)
                passing_blocks.append(block)
                passing_ells.append(ell)
            else:
                block_eligible_rank.append(-1)
            if len(passing_blocks) == required_blocks:
                break

        if len(passing_blocks) < required_blocks:
            reason = 1
            rank = -1
        elif float(np.std(passing_ells[1:])) < min_within_group_sd:
            reason = 2
            rank = -1
        else:
            reason = 0
            rank = len(selected)
            selected.append(
                ContextGroup(
                    sigma=covariance,
                    graph=graph,
                    params=parameters,
                    core=core,
                    reference=passing_blocks[0],
                    continuations=np.stack(passing_blocks[1:], axis=0),
                )
            )
        core_reason.append(reason)
        core_acceptance_rank.append(rank)
        if len(selected) == n_groups:
            break

    if len(selected) != n_groups:
        raise RuntimeError(
            f"INCONCLUSIVE_IDENTIFIABILITY: accepted {len(selected)} of {n_groups} "
            f"groups within {max_core_candidates} core candidates"
        )
    accepted_ids = np.flatnonzero(np.asarray(core_reason) == 0)
    accepted_ranks = np.asarray(core_acceptance_rank)[accepted_ids]
    if not np.array_equal(accepted_ranks, np.arange(n_groups)):
        raise AssertionError("accepted cores are not the first ranked passing cores")
    if len(np.unique(core_seed)) != len(core_seed) or len(np.unique(block_seed)) != len(block_seed):
        raise AssertionError("candidate seed collision")
    selection = {
        "selection_mode": np.asarray(1, dtype=np.int8),
        "selection_max_core_candidates": np.asarray(max_core_candidates, dtype=np.int32),
        "selection_max_blocks_per_core": np.asarray(max_blocks_per_core, dtype=np.int32),
        "selection_min_within_group_sd": np.asarray(min_within_group_sd, dtype=np.float64),
        "selection_interior_weight_min": np.asarray(
            interior_weight_min, dtype=np.float64
        ),
        "selection_interior_weight_max": np.asarray(
            interior_weight_max, dtype=np.float64
        ),
        "selection_minimum_endpoint_js": np.asarray(
            minimum_endpoint_js, dtype=np.float64
        ),
        "selection_seed_namespace": np.frombuffer(
            (seed_namespace or "").encode("utf-8"), dtype=np.uint8
        ),
        "selection_seed_root": np.frombuffer(seed_root.encode("ascii"), dtype=np.uint8),
        "candidate_core_sigma": np.stack(core_sigma),
        "candidate_core_graph": np.asarray(core_graph, dtype=np.int8),
        "candidate_core_params": np.asarray(core_params, dtype=np.float64),
        "candidate_core_context": np.stack(core_context),
        "candidate_core_seed": np.asarray(core_seed, dtype=np.uint64),
        "candidate_core_reason": np.asarray(core_reason, dtype=np.int8),
        "candidate_core_acceptance_rank": np.asarray(core_acceptance_rank, dtype=np.int32),
        "candidate_block_context": np.stack(block_context),
        "candidate_block_core_index": np.asarray(block_core_index, dtype=np.int32),
        "candidate_block_index": np.asarray(block_index, dtype=np.int16),
        "candidate_block_seed": np.asarray(block_seed, dtype=np.uint64),
        "candidate_block_ell": np.asarray(block_ell, dtype=np.float64),
        "candidate_block_js_primary": np.asarray(block_js_primary, dtype=np.float64),
        "candidate_block_js_alternate": np.asarray(block_js_alternate, dtype=np.float64),
        "candidate_block_reason": np.asarray(block_reason, dtype=np.int8),
        "candidate_block_eligible_rank": np.asarray(block_eligible_rank, dtype=np.int16),
    }
    return selected, selection


def generate_panel(
    *,
    commit_sha: str,
    query_banks: np.ndarray,
    n_groups: int,
    n_continuations: int,
    out_path: Path,
    interior_selected: bool = False,
    max_core_candidates: int = 2_000,
    max_blocks_per_core: int = 512,
    min_within_group_sd: float = 0.25,
    interior_weight_min: float = 0.05,
    interior_weight_max: float = 0.95,
    minimum_endpoint_js: float = 0.10,
    scientific: bool = False,
    run_lock_sha256: str | None = None,
    seed_namespace: str | None = None,
    seed_root: str | None = None,
):
    started = time.perf_counter()
    out_path = Path(out_path).resolve()
    if scientific:
        configure_determinism(0)
        root = repository_root()
        _require_current_panel_commit(commit_sha, root)
        locked_commit, locked_hash, lock = verify_run_lock(root)
        settings = lock["settings"]
        if commit_sha != locked_commit or run_lock_sha256 != locked_hash:
            raise ValueError("scientific panel identity does not match the verified run lock")
        if not np.array_equal(query_banks, load_locked_query_banks(root)):
            raise ValueError("scientific panel query banks are not the locked banks")
        expected_design = {
            "n_groups": settings["groups"],
            "n_continuations": settings["continuations"],
            "max_core_candidates": settings["max_core_candidates"],
            "max_blocks_per_core": settings["max_blocks_per_core"],
            "min_within_group_sd": settings["min_within_group_sd"],
            "interior_weight_min": 0.05,
            "interior_weight_max": 0.95,
            "minimum_endpoint_js": 0.10,
        }
        observed_design = {
            "n_groups": n_groups,
            "n_continuations": n_continuations,
            "max_core_candidates": max_core_candidates,
            "max_blocks_per_core": max_blocks_per_core,
            "min_within_group_sd": min_within_group_sd,
            "interior_weight_min": interior_weight_min,
            "interior_weight_max": interior_weight_max,
            "minimum_endpoint_js": minimum_endpoint_js,
        }
        if not interior_selected or observed_design != expected_design:
            raise ValueError("scientific panel design does not match the verified run lock")
        if seed_namespace is not None or seed_root is not None:
            raise ValueError("locked scientific panel does not accept a seed override")
        require_scientific_run_path(
            out_path,
            commit_sha=commit_sha,
            relative="panel.npz",
        )
        metadata_path = out_path.with_suffix(".json")
        if out_path.exists() or metadata_path.exists():
            raise FileExistsError("scientific panel output already exists")
        if out_path.parent.exists() and any(out_path.parent.iterdir()):
            raise FileExistsError("scientific run directory is not empty")
    if interior_selected:
        groups, selection = _select_interior_groups(
            commit_sha=commit_sha,
            query_banks=query_banks,
            n_groups=n_groups,
            n_continuations=n_continuations,
            max_core_candidates=max_core_candidates,
            max_blocks_per_core=max_blocks_per_core,
            min_within_group_sd=min_within_group_sd,
            interior_weight_min=interior_weight_min,
            interior_weight_max=interior_weight_max,
            minimum_endpoint_js=minimum_endpoint_js,
            seed_namespace=seed_namespace,
            seed_root=seed_root,
        )
    else:
        rng = np.random.default_rng(derive_seed(commit_sha, "groups"))
        groups = [generate_group(rng, n_continuations=n_continuations) for _ in range(n_groups)]
        selection = {
            "selection_mode": np.asarray(0, dtype=np.int8),
            "selection_max_core_candidates": np.asarray(0, dtype=np.int32),
            "selection_max_blocks_per_core": np.asarray(0, dtype=np.int32),
            "selection_min_within_group_sd": np.asarray(0.0, dtype=np.float64),
            "selection_interior_weight_min": np.asarray(
                interior_weight_min, dtype=np.float64
            ),
            "selection_interior_weight_max": np.asarray(
                interior_weight_max, dtype=np.float64
            ),
            "selection_minimum_endpoint_js": np.asarray(
                minimum_endpoint_js, dtype=np.float64
            ),
            "selection_seed_namespace": np.empty(0, dtype=np.uint8),
            "selection_seed_root": np.frombuffer(commit_sha.encode("ascii"), dtype=np.uint8),
            "candidate_core_sigma": np.empty((0, 2, 2), dtype=np.float64),
            "candidate_core_graph": np.empty(0, dtype=np.int8),
            "candidate_core_params": np.empty((0, 3), dtype=np.float64),
            "candidate_core_context": np.empty((0, 20, 2), dtype=np.float64),
            "candidate_core_seed": np.empty(0, dtype=np.uint64),
            "candidate_core_reason": np.empty(0, dtype=np.int8),
            "candidate_core_acceptance_rank": np.empty(0, dtype=np.int32),
            "candidate_block_context": np.empty((0, 10, 2), dtype=np.float64),
            "candidate_block_core_index": np.empty(0, dtype=np.int32),
            "candidate_block_index": np.empty(0, dtype=np.int16),
            "candidate_block_seed": np.empty(0, dtype=np.uint64),
            "candidate_block_ell": np.empty(0, dtype=np.float64),
            "candidate_block_js_primary": np.empty(0, dtype=np.float64),
            "candidate_block_js_alternate": np.empty(0, dtype=np.float64),
            "candidate_block_reason": np.empty(0, dtype=np.int8),
            "candidate_block_eligible_rank": np.empty(0, dtype=np.int16),
        }
    sigma = np.stack([group.sigma for group in groups])
    graph = np.asarray([group.graph for group in groups], dtype=np.int8)
    params = np.asarray(
        [[group.params.beta, group.params.b_root, group.params.b_effect] for group in groups],
        dtype=np.float64,
    )
    core = np.stack([group.core for group in groups])
    reference = np.stack([group.reference for group in groups])
    continuations = np.stack([group.continuations for group in groups])
    base_contexts = np.concatenate([core, reference], axis=1)
    target_contexts = np.concatenate(
        [
            np.broadcast_to(core[:, None, :, :], (n_groups, n_continuations, 20, 2)),
            continuations,
        ],
        axis=2,
    )

    bank_count, query_count = query_banks.shape
    shape_core = (bank_count, n_groups, query_count, 100)
    shape_target = (bank_count, n_groups, n_continuations, query_count, 100)
    f0_core = np.empty(shape_core, dtype=np.float64)
    f1_core = np.empty_like(f0_core)
    f0_base = np.empty_like(f0_core)
    f1_base = np.empty_like(f0_core)
    f0_target = np.empty(shape_target, dtype=np.float64)
    f1_target = np.empty_like(f0_target)
    js_core = np.empty((bank_count, n_groups), dtype=np.float64)
    js_base = np.empty_like(js_core)
    js_target = np.empty((bank_count, n_groups, n_continuations), dtype=np.float64)
    ell_core_by_bank = np.empty_like(js_core)
    ell_base_by_bank = np.empty_like(js_core)
    ell_target_by_bank = np.empty_like(js_target)

    for bank_index, bank in enumerate(query_banks):
        oracle = GridOracle(bank, quadrature=15)
        for group_index in range(n_groups):
            core_bundle = oracle.evaluate(core[group_index])
            base_bundle = oracle.evaluate(base_contexts[group_index])
            ell_core_by_bank[bank_index, group_index] = core_bundle.ell
            ell_base_by_bank[bank_index, group_index] = base_bundle.ell
            f0_core[bank_index, group_index] = core_bundle.f0
            f1_core[bank_index, group_index] = core_bundle.f1
            f0_base[bank_index, group_index] = base_bundle.f0
            f1_base[bank_index, group_index] = base_bundle.f1
            js_core[bank_index, group_index] = core_bundle.js
            js_base[bank_index, group_index] = base_bundle.js
            for continuation_index in range(n_continuations):
                target_bundle = oracle.evaluate(target_contexts[group_index, continuation_index])
                ell_target_by_bank[bank_index, group_index, continuation_index] = target_bundle.ell
                f0_target[bank_index, group_index, continuation_index] = target_bundle.f0
                f1_target[bank_index, group_index, continuation_index] = target_bundle.f1
                js_target[bank_index, group_index, continuation_index] = target_bundle.js
        del oracle

    if np.max(np.abs(ell_core_by_bank[0] - ell_core_by_bank[1])) > 1e-10:
        raise AssertionError("query bank changed core log evidence")
    if np.max(np.abs(ell_base_by_bank[0] - ell_base_by_bank[1])) > 1e-10:
        raise AssertionError("query bank changed base log evidence")
    if np.max(np.abs(ell_target_by_bank[0] - ell_target_by_bank[1])) > 1e-10:
        raise AssertionError("query bank changed target log evidence")
    ell_core = ell_core_by_bank[0]
    ell_base = ell_base_by_bank[0]
    ell_target = ell_target_by_bank[0]
    w_core = expit(ell_core)
    w_base = expit(ell_base)
    w_target = expit(ell_target)
    interior_core = (w_core >= interior_weight_min) & (w_core <= interior_weight_max)
    interior_base = (w_base >= interior_weight_min) & (w_base <= interior_weight_max)
    interior_target = (w_target >= interior_weight_min) & (
        w_target <= interior_weight_max
    )
    js_core_both = np.all(js_core >= minimum_endpoint_js, axis=0)
    js_base_both = np.all(js_base >= minimum_endpoint_js, axis=0)
    js_target_both = np.all(js_target >= minimum_endpoint_js, axis=0)
    eligible_replace = (
        interior_base[:, None]
        & interior_target
        & js_base_both[:, None]
        & js_target_both
    )
    eligible_append = (
        interior_core[:, None]
        & interior_target
        & js_core_both[:, None]
        & js_target_both
    )
    eligible_replace_groups = int(np.sum(np.sum(eligible_replace, axis=1) >= 4))
    eligible_append_groups = int(np.sum(np.sum(eligible_append, axis=1) >= 4))
    if scientific:
        if not interior_selected or run_lock_sha256 is None:
            raise ValueError("scientific panels require the locked selected-interior design")
        if eligible_replace.shape != (256, 8) or not eligible_replace.all():
            raise AssertionError("scientific replace cohort is not fully eligible")

    write_numeric_npz_atomic(
        out_path,
        sigma=sigma,
        graph=graph,
        params=params,
        core=core,
        reference=reference,
        continuations=continuations,
        query_banks=query_banks,
        ell_core=ell_core,
        ell_base=ell_base,
        ell_target=ell_target,
        f0_core=f0_core,
        f1_core=f1_core,
        f0_base=f0_base,
        f1_base=f1_base,
        f0_target=f0_target,
        f1_target=f1_target,
        js_core=js_core,
        js_base=js_base,
        js_target=js_target,
        eligible_replace=eligible_replace.astype(np.uint8),
        eligible_append=eligible_append.astype(np.uint8),
        commit_sha=np.frombuffer(commit_sha.encode(), dtype=np.uint8),
        evaluation_root=np.asarray(evaluation_root(commit_sha), dtype=np.uint64),
        scientific=np.asarray(int(scientific), dtype=np.int8),
        run_lock_sha256=np.frombuffer(
            bytes.fromhex(run_lock_sha256) if run_lock_sha256 is not None else bytes(32),
            dtype=np.uint8,
        ),
        **selection,
    )
    metadata = {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__)),
        "commit_sha": commit_sha,
        "n_groups": n_groups,
        "n_continuations": n_continuations,
        "eligible_replace_groups": eligible_replace_groups,
        "eligible_append_groups": eligible_append_groups,
        "interior_selected": interior_selected,
        "scientific": scientific,
        "run_lock_sha256": run_lock_sha256,
        "evaluation_root": evaluation_root(commit_sha),
        "seed_namespace": seed_namespace,
        "seed_root": commit_sha if seed_root is None else seed_root,
        "candidate_core_count": int(len(selection["candidate_core_seed"])),
        "candidate_block_count": int(len(selection["candidate_block_seed"])),
        "panel_sha256": sha256_file(out_path),
        "panel_bytes": out_path.stat().st_size,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    write_json_atomic(out_path.with_suffix(".json"), metadata)
    return metadata


def _contexts_from_panel(panel):
    core = panel["core"]
    base = np.concatenate([panel["core"], panel["reference"]], axis=1)
    target = np.concatenate(
        [
            np.broadcast_to(
                panel["core"][:, None, :, :],
                (
                    panel["core"].shape[0],
                    panel["continuations"].shape[1],
                    panel["core"].shape[1],
                    2,
                ),
            ),
            panel["continuations"],
        ],
        axis=2,
    )
    return core, base, target


def production_shape_diagnostics(
    model,
    sample: np.ndarray,
    companion_sample: np.ndarray,
    queries: np.ndarray,
    *,
    batch_permutation: np.ndarray,
    row_permutation: np.ndarray,
) -> dict:
    """Check invariants of the fixed physical batch shape used in production.

    Singleton inference is measured only as a portability diagnostic. It is not
    part of the fixed-batch-64 scientific estimand and cannot fail this guard.
    """

    sample = np.asarray(sample, dtype=np.float64)
    companion_sample = np.asarray(companion_sample, dtype=np.float64)
    batch_permutation = np.asarray(batch_permutation, dtype=np.int64)
    row_permutation = np.asarray(row_permutation, dtype=np.int64)
    if sample.ndim != 3 or sample.shape[0] != 64 or sample.shape[2] != 2:
        raise ValueError("production diagnostics require 64 contexts with two columns")
    if companion_sample.shape != sample.shape:
        raise ValueError("companion contexts must match the production sample shape")
    if sorted(batch_permutation.tolist()) != list(range(64)):
        raise ValueError("batch_permutation is not a permutation of 64 contexts")
    if sorted(row_permutation.tolist()) != list(range(sample.shape[1])):
        raise ValueError("row_permutation does not cover every context row")
    if np.array_equal(batch_permutation, np.arange(64)):
        raise ValueError("batch_permutation must move at least one context")
    if np.array_equal(row_permutation, np.arange(sample.shape[1])):
        raise ValueError("row_permutation must move at least one row")

    production = predict_probabilities(model, sample, queries, batch_size=64)
    replay = predict_probabilities(model, sample, queries, batch_size=64)
    singleton = predict_probabilities(model, sample, queries, batch_size=1)

    permuted = predict_probabilities(
        model, sample[batch_permutation], queries, batch_size=64
    )
    restored = permuted[np.argsort(batch_permutation)]

    focal_groups = np.arange(64, dtype=np.int64).reshape(16, 4)
    companion_exact = True
    companion_error = 0.0
    companion_group_errors = []
    for focal_indices in focal_groups:
        companion_variant = companion_sample.copy()
        companion_variant[focal_indices] = sample[focal_indices]
        nonfocal = np.setdiff1d(np.arange(64), focal_indices)
        if np.array_equal(companion_variant[nonfocal], sample[nonfocal]):
            raise ValueError("companion replacement did not change nonfocal contexts")
        companion_predictions = predict_probabilities(
            model, companion_variant, queries, batch_size=64
        )
        group_error = float(
            np.max(
                np.abs(
                    production[focal_indices]
                    - companion_predictions[focal_indices]
                )
            )
        )
        companion_group_errors.append(group_error)
        companion_error = max(companion_error, group_error)
        companion_exact = companion_exact and bool(
            np.array_equal(
                production[focal_indices], companion_predictions[focal_indices]
            )
        )

    row_permuted = predict_probabilities(
        model, sample[:, row_permutation], queries, batch_size=64
    )
    replay_exact = bool(np.array_equal(production, replay))
    batch_axis_exact = bool(np.array_equal(production, restored))
    batch_axis_error = float(np.max(np.abs(production - restored)))
    row_error = float(np.max(np.abs(production - row_permuted)))
    descriptive_batch_error = float(np.max(np.abs(singleton - production)))
    passed = bool(
        replay_exact
        and batch_axis_exact
        and companion_exact
        and row_error <= 1e-6
    )
    return {
        "pass": passed,
        "production_replay_byte_identical": replay_exact,
        "batch_axis_permutation_byte_identical": batch_axis_exact,
        "companion_replacement_byte_identical": companion_exact,
        "max_batch_axis_permutation_error": batch_axis_error,
        "max_companion_replacement_error": companion_error,
        "max_row_permutation_error": row_error,
        "descriptive_max_batch_1_vs_64_error": descriptive_batch_error,
        "focal_contexts_checked": 64,
        "companion_variants": 16,
        "companion_group_max_errors": companion_group_errors,
    }


def golden_replay(panel: dict, registry: dict, *, fleet_guard: bool = True):
    core, base, target = _contexts_from_panel(panel)
    target_flat = target.reshape(-1, target.shape[2], 2)
    commit = bytes(panel["commit_sha"].tolist()).decode("ascii")
    if fleet_guard:
        if core.shape[0] < 64 or target_flat.shape[0] < 64:
            raise ValueError("golden replay needs complete core and target batches")
        sample_indices = {
            "core20": np.random.default_rng(
                derive_seed(commit, "guard:core-context-sample")
            ).choice(core.shape[0], size=64, replace=False),
            "length30": np.random.default_rng(
                derive_seed(commit, "guard:target-context-sample")
            ).choice(target_flat.shape[0], size=64, replace=False),
        }
        samples = {
            "core20": core[sample_indices["core20"]],
            "length30": target_flat[sample_indices["length30"]],
        }
        sample_source = "scientific-panel"
    else:
        smoke_groups = [
            generate_group(
                np.random.default_rng(derive_seed(commit, f"smoke-guard:{index}")),
                n_continuations=2,
            )
            for index in range(64)
        ]
        samples = {
            "core20": np.stack([group.core for group in smoke_groups]),
            "length30": np.stack(
                [
                    np.concatenate([group.core, group.reference], axis=0)
                    for group in smoke_groups
                ]
            ),
        }
        sample_indices = {
            "core20": np.empty(0, dtype=np.int64),
            "length30": np.empty(0, dtype=np.int64),
        }
        sample_source = "dedicated-deterministic-smoke"
    row_permutations = {
        kind: np.random.default_rng(
            derive_seed(commit, f"guard:row-permutation:{kind}")
        ).permutation(sample.shape[1])
        for kind, sample in samples.items()
    }
    batch_permutations = {
        kind: np.random.default_rng(
            derive_seed(commit, f"guard:batch-permutation:{kind}")
        ).permutation(sample.shape[0])
        for kind, sample in samples.items()
    }
    companion_samples = {
        kind: np.roll(sample, shift=7, axis=0) for kind, sample in samples.items()
    }
    identities = (
        [(seed, step) for seed in range(16) for step in (0, 12_000)]
        if fleet_guard
        else [(0, 0)]
    )
    records = []
    for seed, step in identities:
        record = expanded_checkpoint_record(registry, seed, step)
        model = load_registered_checkpoint(record)
        for bank_index, queries in enumerate(panel["query_banks"]):
            for context_kind, sample in samples.items():
                diagnostic = production_shape_diagnostics(
                    model,
                    sample,
                    companion_samples[context_kind],
                    queries,
                    batch_permutation=batch_permutations[context_kind],
                    row_permutation=row_permutations[context_kind],
                )
                records.append(
                    {
                        "seed": seed,
                        "step": step,
                        "bank_index": bank_index,
                        "context_kind": context_kind,
                        **diagnostic,
                    }
                )
        del model
    passed = all(record["pass"] for record in records)
    return {
        "pass": passed,
        "production_replay_byte_identical": all(
            record["production_replay_byte_identical"] for record in records
        ),
        "batch_axis_permutation_byte_identical": all(
            record["batch_axis_permutation_byte_identical"] for record in records
        ),
        "companion_replacement_byte_identical": all(
            record["companion_replacement_byte_identical"] for record in records
        ),
        "max_batch_axis_permutation_error": max(
            record["max_batch_axis_permutation_error"] for record in records
        ),
        "max_companion_replacement_error": max(
            record["max_companion_replacement_error"] for record in records
        ),
        "max_row_permutation_error": max(
            record["max_row_permutation_error"] for record in records
        ),
        "descriptive_max_batch_1_vs_64_error": max(
            record["descriptive_max_batch_1_vs_64_error"] for record in records
        ),
        "identities_checked": len(identities),
        "banks_checked": 2,
        "context_kinds": list(samples),
        "contexts_checked_per_kind": 64,
        "sample_flat_indices": {
            kind: indices.tolist() for kind, indices in sample_indices.items()
        },
        "sample_source": sample_source,
        "records": records,
        "unused_core_shape": list(core.shape),
        "unused_base_shape": list(base.shape),
    }


def _validate_probabilities(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape or value.dtype != np.float32:
        raise ValueError(f"{name} has the wrong shape or dtype")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError(f"{name} is not a finite probability tensor")
    if not np.allclose(value.sum(axis=-1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} does not normalize")
    if not np.any(value > 0):
        raise ValueError(f"{name} is an all-zero arm")


def _validate_prediction_shard(
    shard: dict,
    *,
    core_count: int,
    continuation_count: int,
    queries: np.ndarray,
    seed: int,
    step: int,
    bank_index: int,
    checkpoint_sha256: str,
    panel_sha256: str,
    scientific: bool,
) -> None:
    required = {
        "p_core",
        "p_base",
        "p_target",
        "query_bank",
        "checkpoint_sha256",
        "panel_sha256",
        "seed",
        "step",
        "bank_index",
        "scientific",
    }
    if set(shard) != required:
        raise ValueError(f"prediction shard schema mismatch: {sorted(set(shard) ^ required)}")
    base_shape = (core_count, len(queries), 100)
    _validate_probabilities("p_core", shard["p_core"], base_shape)
    _validate_probabilities("p_base", shard["p_base"], base_shape)
    _validate_probabilities(
        "p_target",
        shard["p_target"],
        (core_count, continuation_count, len(queries), 100),
    )
    if not np.array_equal(shard["query_bank"], queries):
        raise ValueError("prediction query bank mismatch")
    if bytes(shard["checkpoint_sha256"].tolist()).hex() != checkpoint_sha256:
        raise ValueError("prediction checkpoint hash mismatch")
    if bytes(shard["panel_sha256"].tolist()).hex() != panel_sha256:
        raise ValueError("prediction panel hash mismatch")
    if (
        int(shard["seed"]) != seed
        or int(shard["step"]) != step
        or int(shard["bank_index"]) != bank_index
        or int(shard["scientific"]) != int(scientific)
    ):
        raise ValueError("prediction shard identity mismatch")


def _validate_completed_pre_score_guard(
    guard: dict,
    *,
    panel: dict,
    panel_path: Path,
    registry_path: Path,
) -> tuple[dict, float]:
    expected_keys = {
        "schema_version",
        "scientific",
        "status",
        "pass",
        "commit_sha",
        "panel",
        "panel_sha256",
        "run_lock_sha256",
        "registry_sha256",
        "implementation_sha256s",
        "diagnostics",
        "guard_wall_seconds",
        "guard_peak_rss_bytes",
        "golden_wall_seconds",
    }
    commit = bytes(panel["commit_sha"].tolist()).decode("ascii")
    expected_implementations = {
        "evaluation.py": sha256_file(Path(__file__)),
        "model.py": sha256_file(Path(__file__).with_name("model.py")),
    }
    diagnostics = guard.get("diagnostics", {})
    records = diagnostics.get("records", []) if isinstance(diagnostics, dict) else []
    expected_identities = {
        (seed, step, bank, kind)
        for seed in range(16)
        for step in (0, 12_000)
        for bank in (0, 1)
        for kind in ("core20", "length30")
    }
    observed_identities = {
        (
            int(record.get("seed", -1)),
            int(record.get("step", -1)),
            int(record.get("bank_index", -1)),
            record.get("context_kind"),
        )
        for record in records
    }
    expected_samples = {
        "core20": np.random.default_rng(
            derive_seed(commit, "guard:core-context-sample")
        ).choice(int(panel["core"].shape[0]), size=64, replace=False).tolist(),
        "length30": np.random.default_rng(
            derive_seed(commit, "guard:target-context-sample")
        ).choice(
            int(panel["core"].shape[0] * panel["continuations"].shape[1]),
            size=64,
            replace=False,
        ).tolist(),
    }
    numeric_records = bool(records) and all(
        record.get("pass") is True
        and record.get("production_replay_byte_identical") is True
        and record.get("batch_axis_permutation_byte_identical") is True
        and record.get("companion_replacement_byte_identical") is True
        and float(record.get("max_batch_axis_permutation_error", np.inf)) == 0.0
        and float(record.get("max_companion_replacement_error", np.inf)) == 0.0
        and 0.0 <= float(record.get("max_row_permutation_error", np.inf)) <= 1e-6
        and 0.0
        <= float(record.get("descriptive_max_batch_1_vs_64_error", np.inf))
        < np.inf
        and record.get("focal_contexts_checked") == 64
        and record.get("companion_variants") == 16
        and record.get("companion_group_max_errors") == [0.0] * 16
        for record in records
    )
    maxima = {
        key: max(float(record[key]) for record in records)
        for key in (
            "max_batch_axis_permutation_error",
            "max_companion_replacement_error",
            "max_row_permutation_error",
            "descriptive_max_batch_1_vs_64_error",
        )
    } if records else {}
    golden_wall = float(guard.get("golden_wall_seconds", np.nan))
    guard_wall = float(guard.get("guard_wall_seconds", np.nan))
    guard_peak = int(guard.get("guard_peak_rss_bytes", -1))
    valid = bool(
        set(guard) == expected_keys
        and guard.get("schema_version") == 1
        and guard.get("scientific") is True
        and guard.get("status") == "COMPLETE"
        and guard.get("pass") is True
        and guard.get("commit_sha") == commit
        and guard.get("panel") == panel_path.name
        and guard.get("panel_sha256") == sha256_file(panel_path)
        and guard.get("run_lock_sha256")
        == bytes(panel["run_lock_sha256"].tolist()).hex()
        and guard.get("registry_sha256") == sha256_file(registry_path)
        and guard.get("implementation_sha256s") == expected_implementations
        and np.isfinite(golden_wall)
        and golden_wall >= 0.0
        and np.isfinite(guard_wall)
        and guard_wall >= golden_wall
        and guard_peak >= 0
        and diagnostics.get("pass") is True
        and diagnostics.get("identities_checked") == 32
        and diagnostics.get("banks_checked") == 2
        and diagnostics.get("context_kinds") == ["core20", "length30"]
        and diagnostics.get("contexts_checked_per_kind") == 64
        and diagnostics.get("sample_source") == "scientific-panel"
        and diagnostics.get("sample_flat_indices") == expected_samples
        and len(records) == 128
        and observed_identities == expected_identities
        and numeric_records
        and diagnostics.get("production_replay_byte_identical") is True
        and diagnostics.get("batch_axis_permutation_byte_identical") is True
        and diagnostics.get("companion_replacement_byte_identical") is True
        and all(diagnostics.get(key) == value for key, value in maxima.items())
    )
    if not valid:
        raise ValueError("existing scientific pre-score guard is not resumable")
    return diagnostics, golden_wall


def _start_score_progress(
    path: Path,
    *,
    scientific: bool,
    commit_sha: str,
    panel_sha256: str,
    pre_score_guard_sha256: str,
    pre_score_guard_wall_seconds: float,
    pre_score_guard_peak_rss_bytes: int,
) -> dict:
    if (
        not np.isfinite(float(pre_score_guard_wall_seconds))
        or float(pre_score_guard_wall_seconds) < 0.0
        or int(pre_score_guard_peak_rss_bytes) < 0
    ):
        raise ValueError("pre-score guard resources are invalid")
    identity = {
        "schema_version": 1,
        "scientific": scientific,
        "commit_sha": commit_sha,
        "panel_sha256": panel_sha256,
        "pre_score_guard_sha256": pre_score_guard_sha256,
        "pre_score_guard_wall_seconds": pre_score_guard_wall_seconds,
        "pre_score_guard_peak_rss_bytes": pre_score_guard_peak_rss_bytes,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    if path.exists():
        progress = json.loads(path.read_text())
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("score progress identity mismatch")
        attempts = progress.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("score progress has no prior attempt")
        attempts_valid = all(
            set(attempt)
            == {
                "attempt_index",
                "status",
                "wall_seconds",
                "peak_rss_bytes",
                "completed_shards",
            }
            and attempt.get("attempt_index") == index
            and attempt.get("status") in {"RUNNING", "INTERRUPTED", "COMPLETE"}
            and np.isfinite(float(attempt.get("wall_seconds", np.nan)))
            and float(attempt.get("wall_seconds", -1.0)) >= 0.0
            and int(attempt.get("peak_rss_bytes", -1)) >= 0
            and 0 <= int(attempt.get("completed_shards", -1)) <= 64
            for index, attempt in enumerate(attempts)
        )
        if (
            not attempts_valid
            or not np.isclose(
                float(progress.get("cumulative_wall_seconds", np.nan)),
                pre_score_guard_wall_seconds
                + sum(float(attempt["wall_seconds"]) for attempt in attempts),
                rtol=1e-12,
                atol=1e-9,
            )
            or int(progress.get("peak_rss_bytes", -1))
            != max(
                pre_score_guard_peak_rss_bytes,
                *(int(attempt["peak_rss_bytes"]) for attempt in attempts),
            )
            or not isinstance(progress.get("validated_shard_identities"), list)
        ):
            raise ValueError("score progress resource history mismatch")
        if attempts[-1].get("status") == "RUNNING":
            attempts[-1]["status"] = "INTERRUPTED"
        elif attempts[-1].get("status") != "COMPLETE":
            raise ValueError("score progress is not safely resumable")
    else:
        progress = {
            **identity,
            "attempts": [],
            "cumulative_wall_seconds": pre_score_guard_wall_seconds,
            "peak_rss_bytes": pre_score_guard_peak_rss_bytes,
            "validated_shard_identities": [],
        }
    progress["attempts"].append(
        {
            "attempt_index": len(progress["attempts"]),
            "status": "RUNNING",
            "wall_seconds": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
            "completed_shards": 0,
        }
    )
    write_json_atomic(path, progress)
    return progress


def _update_score_progress(
    path: Path,
    progress: dict,
    *,
    attempt_started: float,
    records: list[dict],
    status: str,
) -> dict:
    if status not in {"RUNNING", "COMPLETE"}:
        raise ValueError("unsupported score progress status")
    attempt = progress["attempts"][-1]
    attempt.update(
        {
            "status": status,
            "wall_seconds": time.perf_counter() - attempt_started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "completed_shards": len(records),
        }
    )
    progress["cumulative_wall_seconds"] = float(
        progress["pre_score_guard_wall_seconds"]
        + sum(float(value["wall_seconds"]) for value in progress["attempts"])
    )
    progress["peak_rss_bytes"] = int(
        max(
            int(progress["pre_score_guard_peak_rss_bytes"]),
            *(int(value["peak_rss_bytes"]) for value in progress["attempts"]),
        )
    )
    progress["validated_shard_identities"] = sorted(
        [
            [int(record["seed"]), int(record["step"]), int(record["bank_index"])]
            for record in records
        ]
    )
    write_json_atomic(path, progress)
    return progress


def _score_checkpoint_pair(
    *,
    registry: dict,
    seed: int,
    step: int,
    panel: dict,
    out_dir: Path,
    core: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    target_flat: np.ndarray,
    panel_hash: str,
    scientific: bool,
) -> tuple[list[dict], int]:
    checkpoint = expanded_checkpoint_record(registry, seed, step)
    model = load_registered_checkpoint(checkpoint)
    records = []
    computed = 0
    try:
        for bank_index, queries in enumerate(panel["query_banks"]):
            path = out_dir / f"pred_s{seed:02d}_step{step:05d}_bank{bank_index}.npz"
            if path.exists():
                existing = load_numeric_npz(path)
                _validate_prediction_shard(
                    existing,
                    core_count=len(core),
                    continuation_count=target.shape[1],
                    queries=queries,
                    seed=seed,
                    step=step,
                    bank_index=bank_index,
                    checkpoint_sha256=checkpoint["sha256"],
                    panel_sha256=panel_hash,
                    scientific=scientific,
                )
            else:
                p_core = predict_probabilities(model, core, queries, batch_size=64)
                p_base = predict_probabilities(model, base, queries, batch_size=64)
                p_target = predict_probabilities(
                    model, target_flat, queries, batch_size=64
                ).reshape(len(core), target.shape[1], len(queries), 100)
                write_numeric_npz_atomic(
                    path,
                    p_core=p_core,
                    p_base=p_base,
                    p_target=p_target,
                    query_bank=np.asarray(queries, dtype=np.float64),
                    checkpoint_sha256=np.frombuffer(
                        bytes.fromhex(checkpoint["sha256"]), dtype=np.uint8
                    ),
                    panel_sha256=np.frombuffer(
                        bytes.fromhex(panel_hash), dtype=np.uint8
                    ),
                    seed=np.asarray(seed, dtype=np.int16),
                    step=np.asarray(step, dtype=np.int32),
                    bank_index=np.asarray(bank_index, dtype=np.int8),
                    scientific=np.asarray(int(scientific), dtype=np.int8),
                )
                _validate_prediction_shard(
                    load_numeric_npz(path),
                    core_count=len(core),
                    continuation_count=target.shape[1],
                    queries=queries,
                    seed=seed,
                    step=step,
                    bank_index=bank_index,
                    checkpoint_sha256=checkpoint["sha256"],
                    panel_sha256=panel_hash,
                    scientific=scientific,
                )
                computed += 1
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "bank_index": bank_index,
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    finally:
        del model
    return records, computed


def score_checkpoints(
    *,
    panel_path: Path,
    registry_path: Path | None,
    out_dir: Path,
    seeds: list[int] | None = None,
    steps: list[int] | None = None,
    scientific: bool = True,
):
    started = time.perf_counter()
    configure_determinism(0)
    panel = load_numeric_npz(panel_path)
    root = repository_root()
    if scientific:
        panel_commit = bytes(panel["commit_sha"].tolist()).decode("ascii")
        _require_current_panel_commit(panel_commit, root)
        require_scientific_run_path(
            panel_path,
            commit_sha=panel_commit,
            relative="panel.npz",
        )
        require_scientific_run_path(
            out_dir,
            commit_sha=panel_commit,
            relative="predictions",
        )
        locked_registry = (root / "config" / "checkpoint_registry.json").resolve()
        if registry_path is not None and registry_path.resolve() != locked_registry:
            raise ValueError("scientific scoring must use the locked checkpoint registry")
        registry_path = locked_registry
        seeds = list(range(16))
        steps = [0, 12_000]
    elif registry_path is None or seeds is None or steps is None:
        raise ValueError("smoke scoring requires an explicit registry, seeds, and steps")
    assert registry_path is not None and seeds is not None and steps is not None
    panel_hash = sha256_file(panel_path)
    pre_score_guard_path = out_dir.parent / "pre_score_guard.json"
    resume_guard = False
    if scientific and pre_score_guard_path.exists():
        pre_score_guard = json.loads(pre_score_guard_path.read_text())
        if (
            pre_score_guard.get("status") != "COMPLETE"
            or pre_score_guard.get("pass") is not True
        ):
            raise RuntimeError(
                "scientific stream has a failed or interrupted pre-score record; "
                "use a new commit stream"
            )
        resume_guard = True
    else:
        pre_score_guard = {
            "schema_version": 1,
            "scientific": scientific,
            "status": "RUNNING",
            "pass": False,
            "commit_sha": bytes(panel["commit_sha"].tolist()).decode("ascii"),
            "panel": panel_path.name,
            "panel_sha256": panel_hash,
            "run_lock_sha256": bytes(panel["run_lock_sha256"].tolist()).hex(),
            "registry_sha256": (
                sha256_file(registry_path) if registry_path.is_file() else None
            ),
            "implementation_sha256s": {
                "evaluation.py": sha256_file(Path(__file__)),
                "model.py": sha256_file(Path(__file__).with_name("model.py")),
            },
            "diagnostics": None,
        }
        write_json_atomic(pre_score_guard_path, pre_score_guard)
    guard_started = started
    try:
        if scientific:
            verified_commit, _ = verify_panel_lock(panel)
            if verified_commit != panel_commit:
                raise ValueError("panel commit changed during lock verification")
            validations = validate_locked_validations(
                root, query_banks=panel["query_banks"]
            )
            enforce_cost_gate(validations["artifacts/validation/smoke_budget.json"])
        registry = load_checkpoint_registry(registry_path)
        if resume_guard:
            golden, golden_wall_seconds = _validate_completed_pre_score_guard(
                pre_score_guard,
                panel=panel,
                panel_path=panel_path,
                registry_path=registry_path,
            )
        else:
            golden_started = time.perf_counter()
            golden = golden_replay(panel, registry, fleet_guard=scientific)
            golden_wall_seconds = time.perf_counter() - golden_started
    except BaseException as error:
        if not resume_guard:
            pre_score_guard.update(
                {
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "guard_wall_seconds": time.perf_counter() - guard_started,
                    "guard_peak_rss_bytes": _peak_rss_bytes(),
                }
            )
            write_json_atomic(pre_score_guard_path, pre_score_guard)
        raise
    if not resume_guard:
        pre_score_guard.update(
            {
                "status": "COMPLETE",
                "pass": golden["pass"],
                "guard_wall_seconds": time.perf_counter() - guard_started,
                "guard_peak_rss_bytes": _peak_rss_bytes(),
                "golden_wall_seconds": golden_wall_seconds,
                "diagnostics": golden,
            }
        )
        write_json_atomic(pre_score_guard_path, pre_score_guard)
    if not golden["pass"]:
        raise AssertionError("production-shape pre-score guard failed")
    core, base, target = _contexts_from_panel(panel)
    target_flat = target.reshape(-1, target.shape[2], 2)
    if scientific and any(len(value) % 64 for value in (core, base, target_flat)):
        raise AssertionError("scientific inference arrays must use complete batches of 64")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir.parent / "score_progress.json"
    if (out_dir / "prediction_ledger.json").exists():
        raise FileExistsError("prediction ledger already exists; scoring is complete")
    attempt_started = started if resume_guard else time.perf_counter()
    progress = _start_score_progress(
        progress_path,
        scientific=scientific,
        commit_sha=bytes(panel["commit_sha"].tolist()).decode("ascii"),
        panel_sha256=panel_hash,
        pre_score_guard_sha256=sha256_file(pre_score_guard_path),
        pre_score_guard_wall_seconds=float(pre_score_guard["guard_wall_seconds"]),
        pre_score_guard_peak_rss_bytes=int(pre_score_guard["guard_peak_rss_bytes"]),
    )
    records = []
    progress = _update_score_progress(
        progress_path,
        progress,
        attempt_started=attempt_started,
        records=records,
        status="RUNNING",
    )
    computed = 0
    for seed in seeds:
        for step in steps:
            try:
                checkpoint_records, checkpoint_computed = _score_checkpoint_pair(
                    registry=registry,
                    seed=seed,
                    step=step,
                    panel=panel,
                    out_dir=out_dir,
                    core=core,
                    base=base,
                    target=target,
                    target_flat=target_flat,
                    panel_hash=panel_hash,
                    scientific=scientific,
                )
            except BaseException:
                progress = _update_score_progress(
                    progress_path,
                    progress,
                    attempt_started=attempt_started,
                    records=records,
                    status="RUNNING",
                )
                raise
            records.extend(checkpoint_records)
            computed += checkpoint_computed
            progress = _update_score_progress(
                progress_path,
                progress,
                attempt_started=attempt_started,
                records=records,
                status="RUNNING",
            )
    expected_identities = {
        (seed, step, bank) for seed in seeds for step in steps for bank in (0, 1)
    }
    observed_identities = {
        (record["seed"], record["step"], record["bank_index"]) for record in records
    }
    if observed_identities != expected_identities or len(records) != len(expected_identities):
        raise AssertionError("prediction fleet is incomplete or duplicated")
    if scientific and len(records) != 64:
        raise AssertionError("scientific scoring must produce exactly 64 shards")
    expected_names = {record["path"] for record in records}
    actual_names = {path.name for path in out_dir.glob("pred_*.npz")}
    if actual_names != expected_names:
        raise ValueError("prediction directory contains a stale or missing shard")
    progress = _update_score_progress(
        progress_path,
        progress,
        attempt_started=attempt_started,
        records=records,
        status="COMPLETE",
    )
    ledger = {
        "schema_version": 1,
        "scientific": scientific,
        "commit_sha": bytes(panel["commit_sha"].tolist()).decode("ascii"),
        "panel": panel_path.name,
        "panel_sha256": panel_hash,
        "run_lock_sha256": bytes(panel["run_lock_sha256"].tolist()).hex(),
        "registry_sha256": sha256_file(registry_path),
        "pre_score_guard_sha256": sha256_file(pre_score_guard_path),
        "score_progress_sha256": sha256_file(progress_path),
        "golden": golden,
        "golden_wall_seconds": golden_wall_seconds,
        "records": records,
        "computed_shards": computed,
        "wall_seconds": progress["cumulative_wall_seconds"],
        "peak_rss_bytes": progress["peak_rss_bytes"],
    }
    write_json_atomic(out_dir / "prediction_ledger.json", ledger)
    return ledger


def main(argv=None):
    configure_determinism(0)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    panel_parser = subparsers.add_parser("panel")
    panel_parser.add_argument("--out", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--panel", type=Path, required=True)
    score_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "panel":
        commit, lock_hash, lock = verify_run_lock()
        validations = validate_locked_validations()
        enforce_cost_gate(validations["artifacts/validation/smoke_budget.json"])
        settings = lock["settings"]
        metadata = generate_panel(
            commit_sha=commit,
            query_banks=load_locked_query_banks(),
            n_groups=settings["groups"],
            n_continuations=settings["continuations"],
            out_path=args.out,
            interior_selected=True,
            max_core_candidates=settings["max_core_candidates"],
            max_blocks_per_core=settings["max_blocks_per_core"],
            min_within_group_sd=settings["min_within_group_sd"],
            scientific=True,
            run_lock_sha256=lock_hash,
        )
        verify_panel_lock(load_numeric_npz(args.out))
        print(json.dumps(metadata, sort_keys=True))
    else:
        ledger = score_checkpoints(
            panel_path=args.panel,
            registry_path=None,
            out_dir=args.out,
            scientific=True,
        )
        print(
            json.dumps(
                {
                    "computed_shards": ledger["computed_shards"],
                    "records": len(ledger["records"]),
                    "wall_seconds": ledger["wall_seconds"],
                    "peak_rss_bytes": ledger["peak_rss_bytes"],
                    "golden": ledger["golden"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
