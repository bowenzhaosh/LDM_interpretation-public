"""Independent raw-array verifier for the Phase-1 confirmation join.

This module intentionally imports only the Python standard library and NumPy.
It does not import the panel, PFN, oracle, join, metric, or configuration
implementation modules. It verifies the sealed joined raw arrays and
reconstructs the registered decision from those arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


ROW_KEYS = (
    "row_id",
    "prior_code",
    "draw_index",
    "evaluation_seed",
    "stream_index",
    "atom_bank_index",
    "atom_seed",
    "shard_local_index",
)
RAW_ARRAYS = (
    "schema_version",
    "attempt_identity_sha256",
    *ROW_KEYS,
    "row_key_sha256",
    "input_row_sha256",
    "outcome_bin",
    "model_seeds",
    "checkpoint_steps",
    "oracle_full_nll",
    "oracle_ablated_nll",
    "ordering_value",
    "keep_full",
    "keep_ablated",
    "ess_full_atoms",
    "ess_ablated_atoms",
    "pfn_nll",
    "deficit",
    "gap",
)
HALF_ARRAYS = (
    "schema_version",
    "attempt_identity_sha256",
    *ROW_KEYS,
    "row_key_sha256",
    "input_row_sha256",
    "oracle_half_full_nll",
    "oracle_half_ablated_nll",
    "oracle_full_nll",
    "oracle_ablated_nll",
    "pfn_final_nll",
)
BOOTSTRAP_ARRAYS = (
    "ordering_value",
    "deficit",
    "gap_final",
    "delta",
    "deficit_change_final_minus_early",
    "delta_change_final_minus_early",
)
MODEL_SEEDS = np.array([0, 1, 2], dtype=np.int64)
CHECKPOINT_STEPS = np.array([20_000, 60_000, 120_000], dtype=np.int64)
CANONICAL_SOURCES = (
    "PHASE1_ORDERING_PREREG.md",
    "PHASE1_ORDERING_CONFIRMATION_AMENDMENT.md",
    "PHASE1_ORDERING_CONFIRMATION_FIX1.md",
    "PHASE1_ORDERING_CONFIRMATION_FIX2.md",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/ARTIFACT_SHA256SUMS",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/TERMINAL_STATUS.json",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/diagnostics/pfn-replay-sweep-runtime-v2-2157aa0.py",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/diagnostics/pfn-replay-full-panel-v3-2157aa0.py",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/slurm/p1-replay-sweep-160038.out",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/slurm/p1-replay-sweep-160040.out",
    "campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/slurm/p1-replay-sweep-160041.out",
    "config/phase1_ordering_confirmation.json",
    "config/phase1_checkpoint_registry.json",
    "artifacts/phase1/d4_generator.py",
    "artifacts/phase1/d4_train_fleet.py",
    "environment/phase1-washu-runtime.json",
    "environment/phase1-washu-binary-inventory.json",
    "environment/phase1-washu-requirements-lock.txt",
    "cluster/phase1_confirmation.sbatch",
    "cluster/submit_phase1_confirmation.sh",
    "cluster/submit_phase1_confirmation.py",
    "src/pfn_dag_verify/phase1_confirm_common.py",
    "src/pfn_dag_verify/phase1_ordering.py",
    "src/pfn_dag_verify/phase1_panel.py",
    "src/pfn_dag_verify/phase1_pfn.py",
    "src/pfn_dag_verify/phase1_oracle_confirm.py",
    "src/pfn_dag_verify/phase1_join.py",
    "src/pfn_dag_verify/phase1_confirmation_verify.py",
    "src/pfn_dag_verify/storage.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _load_npz(path: Path, schema: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != schema:
            raise RuntimeError(f"raw-array schema mismatch: {path}")
        return {name: archive[name].copy() for name in archive.files}


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _verify_annotated_tag(repo: Path, tag: str, commit: str) -> None:
    tag_ref = f"refs/tags/{tag}"
    if _git(repo, "cat-file", "-t", tag_ref) != "tag":
        raise RuntimeError(f"required attempt tag is not annotated: {tag}")
    if _git(repo, "rev-parse", "--verify", f"{tag_ref}^{{commit}}") != commit:
        raise RuntimeError(f"required attempt tag does not resolve to {commit}: {tag}")


def _verify_source(
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    identity: dict[str, Any],
    commit: str,
    tag: str,
) -> None:
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("independent verification requires a clean source checkout")
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise RuntimeError(
            "independent verifier source checkout is not at the attempt commit"
        )
    _verify_annotated_tag(repo, tag, commit)
    if config.get("required_attempt_tag") != tag:
        raise RuntimeError("attempt tag differs from the frozen config")
    if config.get("independent_verifier") != (
        "src/pfn_dag_verify/phase1_confirmation_verify.py"
    ):
        raise RuntimeError("independent verifier path drifted")
    if _sha256_file(repo / str(config["independent_verifier"])) != config.get(
        "independent_verifier_sha256"
    ):
        raise RuntimeError("independent verifier source hash mismatch")
    if identity.get("git_commit") != commit:
        raise RuntimeError("joined identity commit mismatch")
    if identity.get("config_sha256") != _sha256_file(config_path):
        raise RuntimeError("joined identity config hash mismatch")
    observed = identity.get("source_inventory")
    if not isinstance(observed, dict) or set(observed) != set(CANONICAL_SOURCES):
        raise RuntimeError("joined source inventory mismatch")
    for relative in CANONICAL_SOURCES:
        path = repo / relative
        if not path.is_file() or _sha256_file(path) != observed[relative]:
            raise RuntimeError(f"joined source inventory hash mismatch: {relative}")


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"independent raw-array algebra mismatch: {label}")


def _canonical_rows(
    raw: dict[str, np.ndarray], config: dict[str, Any], identity_sha256: str
) -> dict[str, np.ndarray]:
    if raw["schema_version"].tolist() != [1]:
        raise RuntimeError("joined raw schema version mismatch")
    expected_identity = np.frombuffer(bytes.fromhex(identity_sha256), dtype=np.uint8)
    if not np.array_equal(raw["attempt_identity_sha256"], expected_identity):
        raise RuntimeError("joined raw identity mismatch")
    if raw["model_seeds"].dtype != np.int64 or not np.array_equal(
        raw["model_seeds"], MODEL_SEEDS
    ):
        raise RuntimeError("joined model-seed registry mismatch")
    if raw["checkpoint_steps"].dtype != np.int64 or not np.array_equal(
        raw["checkpoint_steps"], CHECKPOINT_STEPS
    ):
        raise RuntimeError("joined checkpoint registry mismatch")
    n = 2 * 3 * int(config["contexts_per_prior_draw"])
    if n != 6402 or raw["row_id"].shape != (n,):
        raise RuntimeError("joined row count mismatch")
    if raw["row_id"].dtype != np.int64:
        raise RuntimeError("joined row-id dtype mismatch")
    if not np.array_equal(np.sort(raw["row_id"]), np.arange(n, dtype=np.int64)):
        raise RuntimeError("joined row IDs are not unique and complete")
    order = np.argsort(raw["row_id"], kind="stable")
    rows = {
        name: (
            raw[name][order]
            if raw[name].ndim > 0 and raw[name].shape[0] == n
            else raw[name].copy()
        )
        for name in raw
    }
    row_id = rows["row_id"]
    prior_code = row_id // (3 * 1067)
    within_prior = row_id % (3 * 1067)
    draw = within_prior // 1067
    stream = within_prior % 1067
    bank = stream % 3
    local = stream // 3
    expected_metadata = {
        "prior_code": prior_code,
        "draw_index": draw,
        "stream_index": stream,
        "atom_bank_index": bank,
        "shard_local_index": local,
    }
    evaluation_seeds = config["evaluation_seeds"]
    expected_metadata["evaluation_seed"] = np.where(
        prior_code == 0,
        np.asarray(evaluation_seeds["C"], dtype=np.int64)[draw],
        np.asarray(evaluation_seeds["N"], dtype=np.int64)[draw],
    )
    atom_seeds = np.asarray(
        [int(row["seed"]) for row in config["atom_banks"]], dtype=np.int64
    )
    expected_metadata["atom_seed"] = atom_seeds[bank]
    for name, expected in expected_metadata.items():
        if rows[name].dtype != np.int64 or not np.array_equal(rows[name], expected):
            raise RuntimeError(f"joined row metadata mismatch: {name}")
    for name in ("row_key_sha256", "input_row_sha256"):
        if rows[name].dtype != np.uint8 or rows[name].shape != (n, 32):
            raise RuntimeError(f"joined row digest shape mismatch: {name}")
    if rows["outcome_bin"].dtype != np.int64 or np.any(
        (rows["outcome_bin"] < 0) | (rows["outcome_bin"] >= 100)
    ):
        raise RuntimeError("joined observed-bin array mismatch")
    for name in (
        "oracle_full_nll",
        "oracle_ablated_nll",
        "ordering_value",
        "keep_full",
        "keep_ablated",
        "ess_full_atoms",
        "ess_ablated_atoms",
        "pfn_nll",
        "deficit",
        "gap",
    ):
        if rows[name].dtype != np.float64 or not np.isfinite(rows[name]).all():
            raise RuntimeError(f"joined numeric array mismatch: {name}")
    if rows["pfn_nll"].shape != (n, 3, 3) or rows["deficit"].shape != (n, 3, 3):
        raise RuntimeError("joined fleet tensor shape mismatch")
    if rows["gap"].shape != (n, 3, 3):
        raise RuntimeError("joined gap tensor shape mismatch")
    if np.any(rows["keep_full"] <= 0.0) or np.any(rows["keep_full"] > 1.0 + 1e-6):
        raise RuntimeError("joined full retained-mass range mismatch")
    if np.any(rows["keep_ablated"] <= 0.0) or np.any(rows["keep_ablated"] > 1.0 + 1e-6):
        raise RuntimeError("joined ablated retained-mass range mismatch")
    for name in ("ess_full_atoms", "ess_ablated_atoms"):
        if np.any(rows[name] <= 0.0) or np.any(
            rows[name] > int(config["atom_count"]) + 1e-6
        ):
            raise RuntimeError(f"joined ESS range mismatch: {name}")
    _assert_close(
        rows["ordering_value"],
        rows["oracle_ablated_nll"] - rows["oracle_full_nll"],
        "ordering_value",
    )
    _assert_close(
        rows["deficit"],
        rows["pfn_nll"] - rows["oracle_ablated_nll"][:, None, None],
        "deficit",
    )
    _assert_close(
        rows["gap"],
        rows["pfn_nll"] - rows["oracle_full_nll"][:, None, None],
        "gap",
    )
    return rows


def _canonical_half(
    half: dict[str, np.ndarray], rows: dict[str, np.ndarray], identity_sha256: str
) -> dict[str, np.ndarray]:
    expected_identity = np.frombuffer(bytes.fromhex(identity_sha256), dtype=np.uint8)
    if half["schema_version"].tolist() != [1] or not np.array_equal(
        half["attempt_identity_sha256"], expected_identity
    ):
        raise RuntimeError("nested-half identity mismatch")
    ids = half["row_id"]
    if ids.dtype != np.int64:
        raise RuntimeError("nested-half row-id dtype mismatch")
    expected_ids = np.concatenate(
        [np.arange(start, start + 200, dtype=np.int64) for start in (0, 3201)]
    )
    if not np.array_equal(np.sort(ids), expected_ids):
        raise RuntimeError("nested-half row IDs mismatch")
    order = np.argsort(ids, kind="stable")
    value = {
        name: (
            half[name][order]
            if half[name].ndim > 0 and half[name].shape[0] == len(ids)
            else half[name].copy()
        )
        for name in half
    }
    for name in ROW_KEYS + ("row_key_sha256", "input_row_sha256"):
        if not np.array_equal(value[name], rows[name][expected_ids]):
            raise RuntimeError(f"nested-half metadata mismatch: {name}")
    for half_name, full_name in (
        ("oracle_full_nll", "oracle_full_nll"),
        ("oracle_ablated_nll", "oracle_ablated_nll"),
    ):
        _assert_close(value[half_name], rows[full_name][expected_ids], half_name)
    _assert_close(
        value["pfn_final_nll"], rows["pfn_nll"][expected_ids, :, 2], "pfn_final_nll"
    )
    for name in (
        "oracle_half_full_nll",
        "oracle_half_ablated_nll",
        "oracle_full_nll",
        "oracle_ablated_nll",
        "pfn_final_nll",
    ):
        if value[name].dtype != np.float64 or not np.isfinite(value[name]).all():
            raise RuntimeError(f"nested-half numeric array mismatch: {name}")
    return value


def _point_estimates(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    deficit = np.empty((2, 3), dtype=np.float64)
    ordering = np.empty(2, dtype=np.float64)
    gap_final = np.empty(2, dtype=np.float64)
    for prior in (0, 1):
        mask = rows["prior_code"] == prior
        if int(mask.sum()) != 3201:
            raise RuntimeError("prior arm does not contain exactly 3201 contexts")
        deficit[prior] = rows["deficit"][mask].mean(axis=0).mean(axis=0)
        ordering[prior] = rows["ordering_value"][mask].mean()
        gap_final[prior] = rows["gap"][mask, :, 2].mean(axis=0).mean()
    return {
        "deficit": deficit,
        "ordering_value": ordering,
        "gap_final": gap_final,
        "delta": deficit[0] - deficit[1],
        "deficit_change_final_minus_early": deficit[:, 2] - deficit[:, 0],
        "delta_change_final_minus_early": np.array(
            deficit[0, 2] - deficit[1, 2] - deficit[0, 0] + deficit[1, 0]
        ),
    }


def _bootstrap(
    rows: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    specification = config["bootstrap"]
    replicates = int(specification["replicates"])
    chunk_size = int(specification["chunk_size"])
    if (
        specification["bit_generator"] != "PCG64"
        or replicates != 50_000
        or chunk_size != 256
    ):
        raise RuntimeError("bootstrap specification drift")
    master_seed = int(specification["master_seed"])
    boot_d = np.zeros((2, 3, replicates), dtype=np.float64)
    boot_v = np.zeros((2, replicates), dtype=np.float64)
    boot_gap = np.zeros((2, replicates), dtype=np.float64)
    hashes: dict[str, str] = {}
    for prior_code in (0, 1):
        prior_name = "C" if prior_code == 0 else "N"
        prior_count = int(np.count_nonzero(rows["prior_code"] == prior_code))
        for evaluation_seed in config["evaluation_seeds"][prior_name]:
            for atom_record in config["atom_banks"]:
                atom_seed = int(atom_record["seed"])
                mask = (
                    (rows["prior_code"] == prior_code)
                    & (rows["evaluation_seed"] == int(evaluation_seed))
                    & (rows["atom_seed"] == atom_seed)
                )
                indices = np.flatnonzero(mask)
                indices = indices[np.argsort(rows["row_id"][indices])]
                n = len(indices)
                expected_n = 356 if int(atom_record["bank_index"]) < 2 else 355
                if n != expected_n:
                    raise RuntimeError("bootstrap stratum size mismatch")
                local_v = rows["ordering_value"][indices]
                local_d = rows["deficit"][indices]
                local_gap = rows["gap"][indices, :, 2]
                rng = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence(
                            [master_seed, prior_code, int(evaluation_seed), atom_seed]
                        )
                    )
                )
                digest = hashlib.sha256()
                weight = n / prior_count
                for start in range(0, replicates, chunk_size):
                    stop = min(replicates, start + chunk_size)
                    draw = rng.integers(0, n, size=(stop - start, n), dtype=np.int64)
                    digest.update(draw.astype("<i8", copy=False).tobytes(order="C"))
                    boot_v[prior_code, start:stop] += weight * local_v[draw].mean(
                        axis=1
                    )
                    sampled_d = local_d[draw].mean(axis=1).mean(axis=1)
                    boot_d[prior_code, :, start:stop] += weight * sampled_d.T
                    sampled_gap = local_gap[draw].mean(axis=1).mean(axis=1)
                    boot_gap[prior_code, start:stop] += weight * sampled_gap
                hashes[f"{prior_name}_eval{int(evaluation_seed)}_atom{atom_seed}"] = (
                    digest.hexdigest()
                )
    delta = boot_d[0] - boot_d[1]
    return {
        "ordering_value": boot_v,
        "deficit": boot_d,
        "gap_final": boot_gap,
        "delta": delta,
        "deficit_change_final_minus_early": boot_d[:, 2] - boot_d[:, 0],
        "delta_change_final_minus_early": delta[2] - delta[0],
    }, hashes


def _interval(value: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(value, (0.025, 0.975), method="linear")]


def _bootstrap_half(
    half: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    bootstrap_config = config["bootstrap"]
    replicates = int(bootstrap_config["replicates"])
    chunk_size = int(bootstrap_config["chunk_size"])
    master_seed = int(bootstrap_config["master_seed"])
    namespace = int(bootstrap_config["nested_half_namespace"])
    boot_d = np.zeros((2, replicates), dtype=np.float64)
    boot_e = np.zeros((2, replicates), dtype=np.float64)
    boot_g = np.zeros((2, replicates), dtype=np.float64)
    hashes: dict[str, str] = {}
    for prior_code, prior in enumerate(("C", "N")):
        for atom_record in config["atom_banks"]:
            atom_seed = int(atom_record["seed"])
            bank_index = int(atom_record["bank_index"])
            mask = (half["prior_code"] == prior_code) & (half["atom_seed"] == atom_seed)
            row_indices = np.flatnonzero(mask)
            row_indices = row_indices[np.argsort(half["row_id"][row_indices])]
            expected_n = 67 if bank_index < 2 else 66
            if len(row_indices) != expected_n:
                raise RuntimeError("nested-half bootstrap stratum size mismatch")
            local_d = (
                half["oracle_ablated_nll"][row_indices]
                - half["oracle_half_ablated_nll"][row_indices]
            )
            local_e = (
                half["oracle_full_nll"][row_indices]
                - half["oracle_half_full_nll"][row_indices]
            )
            local_g = (
                np.mean(half["pfn_final_nll"][row_indices], axis=1)
                - half["oracle_full_nll"][row_indices]
            )
            generator = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence(
                        [master_seed, namespace, prior_code, atom_seed]
                    )
                )
            )
            digest = hashlib.sha256()
            weight = expected_n / 200.0
            for start in range(0, replicates, chunk_size):
                stop = min(replicates, start + chunk_size)
                draw = generator.integers(
                    0,
                    expected_n,
                    size=(stop - start, expected_n),
                    dtype=np.int64,
                )
                digest.update(draw.astype("<i8", copy=False).tobytes(order="C"))
                boot_d[prior_code, start:stop] += weight * local_d[draw].mean(axis=1)
                boot_e[prior_code, start:stop] += weight * local_e[draw].mean(axis=1)
                boot_g[prior_code, start:stop] += weight * local_g[draw].mean(axis=1)
            hashes[f"{prior}_atom{atom_seed}"] = digest.hexdigest()
    return {"d": boot_d, "e": boot_e, "g": boot_g}, hashes


def _half_gate(half: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    bootstrap, index_hashes = _bootstrap_half(half, config)
    reports: dict[str, Any] = {}
    d_values: dict[str, float] = {}
    d_limit = float(config["gates"]["nested_half_causal_ablated_abs_max"])
    full_limit = float(config["gates"]["nested_half_full_abs_max"])
    fraction_limit = float(
        config["gates"]["full_oracle_half_change_fraction_of_positive_gap_max"]
    )
    replay_bound = float(config["pfn_combined_context_batch_logp_atol"])
    for prior_code, prior in enumerate(("C", "N")):
        mask = half["prior_code"] == prior_code
        if int(mask.sum()) != 200:
            raise RuntimeError("nested-half prior subset is not exactly 200")
        d = float(
            np.mean(
                half["oracle_ablated_nll"][mask] - half["oracle_half_ablated_nll"][mask]
            )
        )
        e = float(
            np.mean(half["oracle_full_nll"][mask] - half["oracle_half_full_nll"][mask])
        )
        g = float(
            np.mean(half["pfn_final_nll"][mask] - half["oracle_full_nll"][mask, None])
        )
        d_ci = _interval(bootstrap["d"][prior_code])
        e_ci = _interval(bootstrap["e"][prior_code])
        g_lower = float(np.quantile(bootstrap["g"][prior_code], 0.05, method="linear"))
        replay_robust_g_lower = g_lower - replay_bound
        e_ci_abs_max = max(abs(e_ci[0]), abs(e_ci[1]))
        d_values[prior] = d
        reports[prior] = {
            "ablated_full_minus_half_nll": d,
            "ablated_full_minus_half_ci95": d_ci,
            "full_full_minus_half_nll": e,
            "full_full_minus_half_ci95": e_ci,
            "fixed_fleet_final_gap": g,
            "fixed_fleet_final_gap_one_sided_95_lower": g_lower,
            "pfn_single_nll_replay_bound": replay_bound,
            "replay_robust_gap_one_sided_95_lower": replay_robust_g_lower,
            "positive_gap": replay_robust_g_lower > 0.0,
            "full_change_fraction": abs(e) / g if g > 0.0 else None,
            "full_predictive_pass": bool(
                replay_robust_g_lower > 0.0
                and e_ci_abs_max < full_limit
                and e_ci_abs_max < fraction_limit * replay_robust_g_lower
            ),
        }
    difference = abs(d_values["C"] - d_values["N"])
    causal_change = abs(d_values["C"])
    causal_ci = _interval(bootstrap["d"][0])
    difference_ci = _interval(bootstrap["d"][0] - bootstrap["d"][1])
    difference_limit = float(
        config["gates"]["nested_half_control_subtracted_ablated_abs_max"]
    )
    causal_pass = bool(causal_ci[0] > -d_limit and causal_ci[1] < d_limit)
    difference_pass = bool(
        difference_ci[0] > -difference_limit and difference_ci[1] < difference_limit
    )
    return {
        "priors": reports,
        "abs_causal_ablated_change": causal_change,
        "causal_ablated_change_ci95": causal_ci,
        "causal_ablated_change_pass": causal_pass,
        "abs_ablated_change_difference": difference,
        "ablated_change_difference_ci95": difference_ci,
        "ablated_change_difference_pass": difference_pass,
        "bootstrap_index_stream_sha256": index_hashes,
        "pass": bool(
            causal_pass
            and difference_pass
            and all(reports[p]["full_predictive_pass"] for p in ("C", "N"))
        ),
    }


def _decide(
    points: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    half_gate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    effect_floor = float(gates["primary_effect_floor"])
    clearance = float(gates["primary_numerical_clearance"])
    positive_boundary = effect_floor - clearance
    rejection_boundary = effect_floor + clearance
    replay_epsilon = float(config["pfn_combined_context_batch_logp_atol"])
    replay_bounds = {
        "single_pfn_nll": replay_epsilon,
        "direct_deficit_or_gap": replay_epsilon,
        "causal_minus_control_delta": 2.0 * replay_epsilon,
        "direct_checkpoint_change": 2.0 * replay_epsilon,
        "delta_difference_in_differences": 4.0 * replay_epsilon,
    }
    if clearance <= 0.0:
        raise RuntimeError("primary numerical clearance must be positive")
    c_lower = float(
        np.quantile(
            bootstrap["ordering_value"][0],
            gates["value_c_one_sided_quantile"],
            method="linear",
        )
    )
    n_ci = _interval(bootstrap["ordering_value"][1])
    ordering_gate = bool(
        c_lower > clearance
        and n_ci[0] >= float(gates["value_n_equivalence_lower"])
        and n_ci[1] <= float(gates["value_n_equivalence_upper"])
    )
    kl_alarm: dict[str, Any] = {}
    kl_clear = True
    kl_threshold = float(gates["kl_alarm_ci_upper_below"])
    kl_alarm_boundary = kl_threshold - clearance
    kl_clear_boundary = kl_threshold + clearance
    for prior_code, prior in enumerate(("C", "N")):
        ci = _interval(bootstrap["gap_final"][prior_code])
        alarm = kl_alarm_boundary - ci[1] > replay_bounds["direct_deficit_or_gap"]
        clear = ci[1] - kl_clear_boundary > replay_bounds["direct_deficit_or_gap"]
        kl_alarm[prior] = {
            "point": float(points["gap_final"][prior_code]),
            "ci95": ci,
            "alarm": alarm,
            "numerically_borderline": bool(not alarm and not clear),
            "clear": clear,
            "alarm_boundary": kl_alarm_boundary,
            "clear_boundary": kl_clear_boundary,
            "pfn_replay_bound": replay_bounds["direct_deficit_or_gap"],
        }
        kl_clear = kl_clear and clear
    validity = {
        "completeness_and_provenance": True,
        "inference_guards": True,
        "predictive_truncation": True,
        "monte_carlo_diagnostics_reported": True,
        "fixed_fleet_completeness": True,
        "ordering_value": ordering_gate,
        "oracle_convergence": bool(half_gate["pass"]),
        "kl_alarm_clear": bool(kl_clear),
    }
    all_valid = all(validity.values())
    deficits: dict[str, dict[str, Any]] = {"C": {}, "N": {}}
    changes: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for index, step in enumerate(CHECKPOINT_STEPS.tolist()):
        for prior_code, prior in enumerate(("C", "N")):
            point = float(points["deficit"][prior_code, index])
            ci = _interval(bootstrap["deficit"][prior_code, index])
            decision_endpoint = max(point, ci[1])
            endpoint_replay_bound = replay_bounds["direct_deficit_or_gap"]
            effect = bool(
                prior == "C" and point < effect_floor and ci[1] < effect_floor
            )
            direct = bool(
                prior == "C"
                and positive_boundary - decision_endpoint > endpoint_replay_bound
            )
            clearly_fails = bool(
                prior == "C"
                and decision_endpoint - rejection_boundary > endpoint_replay_bound
            )
            deficits[prior][str(step)] = {
                "point": point,
                "ci95": ci,
                "passes_effect_floor": effect,
                "passes_direct_rule": direct,
                "pfn_replay_bound": endpoint_replay_bound,
                "distance_beyond_positive_boundary": (
                    positive_boundary - decision_endpoint
                ),
                "distance_beyond_rejection_boundary": (
                    decision_endpoint - rejection_boundary
                ),
                "numerically_borderline": bool(
                    prior == "C" and not direct and not clearly_fails
                ),
                "clearly_fails_effect_rule": clearly_fails,
            }
        point = float(points["delta"][index])
        ci = _interval(bootstrap["delta"][index])
        decision_endpoint = max(point, ci[1])
        endpoint_replay_bound = replay_bounds["causal_minus_control_delta"]
        effect = bool(point < effect_floor and ci[1] < effect_floor)
        direct = bool(positive_boundary - decision_endpoint > endpoint_replay_bound)
        clearly_fails = bool(
            decision_endpoint - rejection_boundary > endpoint_replay_bound
        )
        deltas[str(step)] = {
            "point": point,
            "ci95": ci,
            "passes_effect_floor": effect,
            "passes_rule": direct,
            "pfn_replay_bound": endpoint_replay_bound,
            "distance_beyond_positive_boundary": (
                positive_boundary - decision_endpoint
            ),
            "distance_beyond_rejection_boundary": (
                decision_endpoint - rejection_boundary
            ),
            "numerically_borderline": bool(not direct and not clearly_fails),
            "clearly_fails_effect_rule": clearly_fails,
        }
    for prior_code, prior in enumerate(("C", "N")):
        point = float(points["deficit_change_final_minus_early"][prior_code])
        ci = _interval(bootstrap["deficit_change_final_minus_early"][prior_code])
        decision_endpoint = max(point, ci[1])
        endpoint_replay_bound = replay_bounds["direct_checkpoint_change"]
        effect = bool(prior == "C" and point < effect_floor and ci[1] < effect_floor)
        replay_pass = bool(
            prior == "C" and effect_floor - decision_endpoint > endpoint_replay_bound
        )
        changes[prior] = {
            "point": point,
            "ci95": ci,
            "passes_effect_floor": effect,
            "passes_direct_change_rule": replay_pass,
            "pfn_replay_bound": endpoint_replay_bound,
            "distance_beyond_effect_floor": effect_floor - decision_endpoint,
            "replay_sensitive": bool(prior == "C" and effect and not replay_pass),
        }
    change_point = float(points["delta_change_final_minus_early"])
    change_ci = _interval(bootstrap["delta_change_final_minus_early"])
    change_effect = bool(change_point < effect_floor and change_ci[1] < effect_floor)
    change_endpoint = max(change_point, change_ci[1])
    change_replay_bound = replay_bounds["delta_difference_in_differences"]
    change_replay_pass = bool(effect_floor - change_endpoint > change_replay_bound)
    if not all_valid:
        primary = secondary = "NOT_EVALUATED"
        decision = "INCONCLUSIVE_PHASE1_INSTRUMENT"
    else:
        final_pass = bool(
            deficits["C"]["120000"]["passes_direct_rule"]
            and deltas["120000"]["passes_rule"]
        )
        final_clear_failure = bool(
            deficits["C"]["120000"]["clearly_fails_effect_rule"]
            or deltas["120000"]["clearly_fails_effect_rule"]
        )
        if final_pass:
            primary = "REPLICATED_ORDERING_USE"
        elif final_clear_failure:
            primary = "NOT_REPLICATED_ORDERING_USE"
        else:
            primary = "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"
        change_pass = bool(
            changes["C"]["passes_direct_change_rule"] and change_replay_pass
        )
        early_clear_failure = bool(
            deficits["C"]["20000"]["clearly_fails_effect_rule"]
            or deltas["20000"]["clearly_fails_effect_rule"]
        )
        secondary_pass = bool(final_pass and early_clear_failure and change_pass)
        early_nominal_clear_failure = bool(
            deficits["C"]["20000"]["distance_beyond_rejection_boundary"] > 0.0
            or deltas["20000"]["distance_beyond_rejection_boundary"] > 0.0
        )
        nominal_secondary_components = bool(
            final_pass
            and early_nominal_clear_failure
            and changes["C"]["passes_effect_floor"]
            and change_effect
        )
        if primary == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE":
            secondary = "NOT_EVALUATED_PRIMARY_NUMERICALLY_UNCLEARED"
        elif secondary_pass:
            secondary = "SUPPORTED_UNDERTRAINING_CAN_OBSCURE_ORDERING_ADVANTAGE"
        elif nominal_secondary_components:
            secondary = "INCONCLUSIVE_UNDERTRAINING_REPLAY_SENSITIVITY"
        else:
            secondary = "NOT_SUPPORTED_UNDERTRAINING_CLAIM"
        decision = primary
    return {
        "decision": decision,
        "primary": primary,
        "secondary": secondary,
        "validity_gates": validity,
        "all_validity_gates_pass": all_valid,
        "ordering_value": {
            "C_point": float(points["ordering_value"][0]),
            "C_one_sided_95_lower": c_lower,
            "C_numerical_clearance": clearance,
            "N_point": float(points["ordering_value"][1]),
            "N_ci95": n_ci,
        },
        "oracle_convergence": half_gate,
        "kl_alarm": kl_alarm,
        "deficit_by_prior_and_checkpoint": deficits,
        "deficit_change_final_minus_early": changes,
        "delta_by_checkpoint": deltas,
        "delta_change_final_minus_early": {
            "point": change_point,
            "ci95": change_ci,
            "passes_effect_floor": change_effect,
            "passes_rule": change_replay_pass,
            "pfn_replay_bound": change_replay_bound,
            "distance_beyond_effect_floor": effect_floor - change_endpoint,
            "replay_sensitive": bool(change_effect and not change_replay_pass),
        },
        "numerical_clearance": {
            "effect_floor": effect_floor,
            "oracle_clearance_nats": clearance,
            "clearance_boundary": positive_boundary,
            "rejection_boundary": rejection_boundary,
            "pfn_replay_bounds": replay_bounds,
            "borderline_action": (
                "stop_without_claim_and_increase_numeric_fidelity_or_replay_stability"
            ),
        },
    }


def _same_json(expected: Any, observed: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise RuntimeError(f"independent result structure mismatch: {label}")
        for key in expected:
            _same_json(expected[key], observed[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise RuntimeError(f"independent result list mismatch: {label}")
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _same_json(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected != observed:
            raise RuntimeError(f"independent result mismatch: {label}")
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(
            float(expected), float(observed), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"independent result numeric mismatch: {label}")
        return
    if expected != observed:
        raise RuntimeError(f"independent result mismatch: {label}")


def verify_confirmation(
    root: Path,
    repo: Path,
    commit: str,
    tag: str,
    out: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    repo = repo.resolve()
    config_path = (
        (repo / "config/phase1_ordering_confirmation.json").resolve()
        if config_path is None
        else config_path.resolve()
    )
    if root == repo or root.is_relative_to(repo):
        raise RuntimeError("joined output must be outside the source checkout")
    if out.resolve().is_relative_to(repo):
        raise RuntimeError(
            "independent verification output must be outside the source checkout"
        )
    config = _load_json(config_path)
    complete = _load_json(root / "COMPLETE.json")
    summary = _load_json(root / "confirmation_summary.json")
    identity = summary.get("identity")
    identity_sha256 = summary.get("identity_sha256")
    if not isinstance(identity, dict) or not isinstance(identity_sha256, str):
        raise RuntimeError("joined identity is missing")
    if _sha256_json(identity) != identity_sha256:
        raise RuntimeError("joined identity digest mismatch")
    _verify_source(repo, config_path, config, identity, commit, tag)
    if (
        complete.get("identity") != identity
        or complete.get("identity_sha256") != identity_sha256
    ):
        raise RuntimeError("completion marker identity mismatch")
    expected_artifacts = {
        "RUNNING.json",
        "confirmatory_raw.npz",
        "nested_half_raw.npz",
        "bootstrap_raw.npz",
        "confirmation_summary.json",
    }
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise RuntimeError("completion artifact inventory mismatch")
    for name, record in artifacts.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"completion artifact bytes mismatch: {name}")
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"completion artifact hash mismatch: {name}")
    raw = _load_npz(root / "confirmatory_raw.npz", RAW_ARRAYS)
    rows = _canonical_rows(raw, config, identity_sha256)
    half = _canonical_half(
        _load_npz(root / "nested_half_raw.npz", HALF_ARRAYS), rows, identity_sha256
    )
    bootstrap, index_hashes = _bootstrap(rows, config)
    stored_bootstrap = _load_npz(root / "bootstrap_raw.npz", BOOTSTRAP_ARRAYS)
    for name in BOOTSTRAP_ARRAYS:
        _assert_close(stored_bootstrap[name], bootstrap[name], f"bootstrap:{name}")
    points = _point_estimates(rows)
    half_gate = _half_gate(half, config)
    decision = _decide(points, bootstrap, half_gate, config)
    if complete.get("decision") != decision["decision"]:
        raise RuntimeError("completion decision does not reproduce independently")
    _same_json(decision, summary.get("result"), "result")
    summary_bootstrap = summary.get("bootstrap")
    if not isinstance(summary_bootstrap, dict):
        raise RuntimeError("bootstrap summary is missing")
    for key, value in config["bootstrap"].items():
        if summary_bootstrap.get(key) != value:
            raise RuntimeError(f"bootstrap summary drift: {key}")
    if summary_bootstrap.get("index_stream_sha256") != index_hashes:
        raise RuntimeError("bootstrap index stream digest mismatch")
    if summary.get("rows") != 6402 or summary.get("fixed_models_per_prior") != 3:
        raise RuntimeError("joined summary dimensions mismatch")
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"independent verification output already exists: {out}")
    result = {
        "schema_version": 1,
        "verification": "INDEPENDENT_CONFIRMATION_RAW_RECOMPUTATION_PASS",
        "decision": decision["decision"],
        "identity_sha256": identity_sha256,
        "source_commit": commit,
        "source_tag": tag,
        "joined_raw_sha256": _sha256_file(root / "confirmatory_raw.npz"),
        "joined_summary_sha256": _sha256_file(root / "confirmation_summary.json"),
        "runtime": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(out)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    result = verify_confirmation(
        args.root, args.repo, args.commit, args.attempt_tag, args.out, args.config
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
