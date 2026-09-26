"""Prospective native-output mapping qualification for the frozen PFN fleet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from .constants import FIXED_PRIOR_SEEDS
from .evaluation import generate_panel
from .instrument import (
    project_coordinate_batch,
    project_kl_batch,
)
from .model import configure_determinism, load_registered_checkpoint, predict_probabilities
from .provenance import (
    clean_head,
    derive_seed,
    load_locked_query_banks,
    repository_root,
    verify_runtime,
)
from .registry import expanded_checkpoint_record, load_checkpoint_registry, sha256_file
from .storage import load_numeric_npz, write_json_atomic, write_numeric_npz_atomic


STREAM_NAMESPACE = "mapping-calibration-v1"
SEED_ROOT_SHA = "dd749cfa8bf75ab34a0f5c757c500516adc354d2"
SEALED_QUERY_BANK_SHA256 = "6e6fdda9741f1db3bbe14c8251f74b17e783ccd541fe19b26b5bb6bfa869af33"
SEALED_CHECKPOINT_REGISTRY_SHA256 = "5f7c77e033a86e9217e892eeb9001c6928b46340c83cb2710e5fdfa1ebe25bb8"
ATTEMPT_ATTESTATION = "config/mapping_qualification_attempt.json"
ATTEMPT_TAG = "mapping-qualification-attempt-v1"
EXPECTED_QUALIFICATION_PROTOCOL_FILES = frozenset(
    {
        "MAPPING_QUALIFICATION_PREREG.md",
        "README.md",
        "pyproject.toml",
        "config/checkpoint_registry.json",
        "config/query_bank.json",
        "environment/installed-distributions.json",
        "environment/requirements-lock.txt",
        "environment/runtime.json",
        "src/pfn_dag_verify/__init__.py",
        "src/pfn_dag_verify/calibration.py",
        "src/pfn_dag_verify/constants.py",
        "src/pfn_dag_verify/evaluation.py",
        "src/pfn_dag_verify/generative.py",
        "src/pfn_dag_verify/instrument.py",
        "src/pfn_dag_verify/legacy_compare.py",
        "src/pfn_dag_verify/mapping_qualification.py",
        "src/pfn_dag_verify/model.py",
        "src/pfn_dag_verify/oracle.py",
        "src/pfn_dag_verify/provenance.py",
        "src/pfn_dag_verify/qualification_seal.py",
        "src/pfn_dag_verify/query_bank.py",
        "src/pfn_dag_verify/registry.py",
        "src/pfn_dag_verify/storage.py",
        "src/pfn_dag_verify/validation.py",
        "tests/test_integrity_guards.py",
        "tests/test_mapping_qualification.py",
        "tests/test_storage_and_registry.py",
    }
)
QUALIFICATION_SETTINGS = {
    "panel": {
        "groups": 64,
        "core_rows": 20,
        "reference_rows": 10,
        "targets_per_group": 2,
        "target_rows": 10,
    },
    "instrument": {
        "banks": 2,
        "queries_per_bank": 8,
        "bins": 100,
        "interior_weight_min": 0.05,
        "interior_weight_max": 0.95,
        "minimum_endpoint_js": 0.10,
    },
    "fleet": {"checkpoint_count": 16, "trained_step": 12_000},
    "inference": {
        "physical_batch_size": 64,
        "row_permutation_atol": 1e-6,
    },
    "mapping_gates": {
        "boundary_rate_max": 0.0,
        "coordinate_kl_abs_g_median_max": 0.10,
        "coordinate_kl_abs_g_p95_max": 0.30,
        "mixture_residual_median_max": 0.10,
        "mixture_residual_p95_max": 0.30,
        "cross_bank_abs_g_median_max": 0.10,
        "cross_bank_abs_g_p95_max": 0.30,
    },
}
N_GROUPS = QUALIFICATION_SETTINGS["panel"]["groups"]
N_TARGETS = QUALIFICATION_SETTINGS["panel"]["targets_per_group"]
PHYSICAL_BATCH_SIZE = QUALIFICATION_SETTINGS["inference"]["physical_batch_size"]
TRAINED_STEP = QUALIFICATION_SETTINGS["fleet"]["trained_step"]
EXPECTED_BANK_BLOCKS = (
    QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"]
    * QUALIFICATION_SETTINGS["instrument"]["banks"]
)
EXPECTED_CROSS_BANK_BLOCKS = QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"]
ROW_PERMUTATION_ATOL = QUALIFICATION_SETTINGS["inference"]["row_permutation_atol"]
CONTEXT_ROWS = (
    QUALIFICATION_SETTINGS["panel"]["core_rows"]
    + QUALIFICATION_SETTINGS["panel"]["reference_rows"]
)


def seed_label(namespace: str, suffix: str) -> str:
    if namespace != STREAM_NAMESPACE or namespace.endswith(":"):
        raise ValueError(f"seed namespace must be exactly {STREAM_NAMESPACE!r}")
    if not suffix or suffix.startswith(":"):
        raise ValueError("seed-label suffix must be nonempty and cannot start with a colon")
    return f"{namespace}:{suffix}"


def _pair(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("mapping statistic input must be a nonempty finite vector")
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _validate_mapping_shapes(
    p_base,
    p_target,
    f0_base,
    f1_base,
    f0_target,
    f1_target,
):
    p_base = np.asarray(p_base, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)
    f0_base = np.asarray(f0_base, dtype=np.float64)
    f1_base = np.asarray(f1_base, dtype=np.float64)
    f0_target = np.asarray(f0_target, dtype=np.float64)
    f1_target = np.asarray(f1_target, dtype=np.float64)
    if p_base.ndim != 3 or p_target.ndim != 4:
        raise ValueError("base and target predictions must have context-query-bin shapes")
    if not (p_base.shape == f0_base.shape == f1_base.shape):
        raise ValueError("base prediction and endpoint shapes differ")
    if not (p_target.shape == f0_target.shape == f1_target.shape):
        raise ValueError("target prediction and endpoint shapes differ")
    if p_target.shape[0] != p_base.shape[0] or p_target.shape[2:] != p_base.shape[1:]:
        raise ValueError("base and target panel dimensions differ")
    arrays = (p_base, p_target, f0_base, f1_base, f0_target, f1_target)
    if any(not value.size or not np.isfinite(value).all() for value in arrays):
        raise ValueError("mapping inputs must be nonempty and finite")
    return arrays


def mapping_block(
    p_base,
    p_target,
    f0_base,
    f1_base,
    f0_target,
    f1_target,
):
    (
        p_base,
        p_target,
        f0_base,
        f1_base,
        f0_target,
        f1_target,
    ) = _validate_mapping_shapes(
        p_base,
        p_target,
        f0_base,
        f1_base,
        f0_target,
        f1_target,
    )
    n_groups, n_targets, n_queries, n_bins = p_target.shape
    flat_shape = (n_groups * n_targets, n_queries, n_bins)
    coord_base = project_coordinate_batch(p_base, f0_base, f1_base)
    kl_base = project_kl_batch(p_base, f0_base, f1_base)
    coord_target = project_coordinate_batch(
        p_target.reshape(flat_shape),
        f0_target.reshape(flat_shape),
        f1_target.reshape(flat_shape),
    )
    kl_target = project_kl_batch(
        p_target.reshape(flat_shape),
        f0_target.reshape(flat_shape),
        f1_target.reshape(flat_shape),
    )
    g_difference = np.concatenate(
        [np.abs(coord_base.g - kl_base.g), np.abs(coord_target.g - kl_target.g)]
    )
    boundary = np.concatenate(
        [
            coord_base.boundary | kl_base.boundary,
            coord_target.boundary | kl_target.boundary,
        ]
    )
    block = {
        "rows_base": int(n_groups),
        "rows_target": int(n_groups * n_targets),
        "boundary_rate": float(np.mean(boundary)),
        "coordinate_kl_abs_g": _pair(g_difference),
        "mixture_residual_base": _pair(coord_base.normalized_residual),
        "mixture_residual_target": _pair(coord_target.normalized_residual),
    }
    gates = QUALIFICATION_SETTINGS["mapping_gates"]
    block["pass"] = bool(
        block["boundary_rate"] <= gates["boundary_rate_max"]
        and block["coordinate_kl_abs_g"]["median"]
        <= gates["coordinate_kl_abs_g_median_max"]
        and block["coordinate_kl_abs_g"]["p95"]
        <= gates["coordinate_kl_abs_g_p95_max"]
        and block["mixture_residual_base"]["median"]
        <= gates["mixture_residual_median_max"]
        and block["mixture_residual_base"]["p95"]
        <= gates["mixture_residual_p95_max"]
        and block["mixture_residual_target"]["median"]
        <= gates["mixture_residual_median_max"]
        and block["mixture_residual_target"]["p95"]
        <= gates["mixture_residual_p95_max"]
    )
    arrays = {
        "g_base": coord_base.g,
        "g_target": coord_target.g.reshape(n_groups, n_targets),
        "kl_g_base": kl_base.g,
        "kl_g_target": kl_target.g.reshape(n_groups, n_targets),
        "boundary_base": coord_base.boundary.astype(np.uint8),
        "boundary_target": coord_target.boundary.reshape(n_groups, n_targets).astype(np.uint8),
        "kl_boundary_base": kl_base.boundary.astype(np.uint8),
        "kl_boundary_target": kl_target.boundary.reshape(n_groups, n_targets).astype(np.uint8),
        "mixture_residual_base": coord_base.normalized_residual,
        "mixture_residual_target": coord_target.normalized_residual.reshape(
            n_groups, n_targets
        ),
    }
    return block, arrays


def cross_bank_block(first: dict[str, np.ndarray], second: dict[str, np.ndarray]):
    if first["g_base"].shape != second["g_base"].shape:
        raise ValueError("cross-bank base coordinates have different shapes")
    if first["g_target"].shape != second["g_target"].shape:
        raise ValueError("cross-bank target coordinates have different shapes")
    difference = np.concatenate(
        [
            np.abs(first["g_base"] - second["g_base"]),
            np.abs(first["g_target"] - second["g_target"]).reshape(-1),
        ]
    )
    result = {"absolute_g_disagreement": _pair(difference)}
    gates = QUALIFICATION_SETTINGS["mapping_gates"]
    result["pass"] = bool(
        result["absolute_g_disagreement"]["median"]
        <= gates["cross_bank_abs_g_median_max"]
        and result["absolute_g_disagreement"]["p95"]
        <= gates["cross_bank_abs_g_p95_max"]
    )
    return result


def inference_guard_from_predictions(
    reference: np.ndarray,
    replay: np.ndarray,
    batch_restored: np.ndarray,
    row_permuted: np.ndarray,
) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    replay = np.asarray(replay, dtype=np.float32)
    batch_restored = np.asarray(batch_restored, dtype=np.float32)
    row_permuted = np.asarray(row_permuted, dtype=np.float32)
    if not (
        reference.shape == replay.shape == batch_restored.shape == row_permuted.shape
    ):
        raise ValueError("qualification inference-guard prediction shapes differ")
    if any(
        not value.size or not np.isfinite(value).all()
        for value in (reference, replay, batch_restored, row_permuted)
    ):
        raise ValueError("qualification inference-guard predictions must be finite")
    replay_exact = bool(np.array_equal(reference, replay))
    batch_exact = bool(np.array_equal(reference, batch_restored))
    row_error = float(np.max(np.abs(reference.astype(np.float64) - row_permuted)))
    result = {
        "production_replay_byte_identical": replay_exact,
        "batch_permutation_byte_identical": batch_exact,
        "max_row_permutation_error": row_error,
    }
    result["pass"] = bool(
        replay_exact and batch_exact and row_error <= ROW_PERMUTATION_ATOL
    )
    return result


def inference_guard(model, contexts, queries, reference: np.ndarray) -> tuple[dict, dict]:
    contexts = np.asarray(contexts, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    replay = predict_probabilities(
        model, contexts, queries, batch_size=PHYSICAL_BATCH_SIZE
    )
    batch_restored = predict_probabilities(
        model, contexts[::-1].copy(), queries, batch_size=PHYSICAL_BATCH_SIZE
    )[::-1]
    row_permuted = predict_probabilities(
        model, contexts[:, ::-1, :].copy(), queries, batch_size=PHYSICAL_BATCH_SIZE
    )
    result = inference_guard_from_predictions(
        reference, replay, batch_restored, row_permuted
    )
    return result, {
        "replay": replay,
        "batch_restored": batch_restored,
        "row_permuted": row_permuted,
    }


def combine_inference_guards(base: dict, target: dict) -> dict:
    result = {
        "production_replay_byte_identical": bool(
            base["production_replay_byte_identical"]
            and target["production_replay_byte_identical"]
        ),
        "batch_permutation_byte_identical": bool(
            base["batch_permutation_byte_identical"]
            and target["batch_permutation_byte_identical"]
        ),
        "max_row_permutation_error": float(
            max(base["max_row_permutation_error"], target["max_row_permutation_error"])
        ),
    }
    result["pass"] = bool(
        result["production_replay_byte_identical"]
        and result["batch_permutation_byte_identical"]
        and result["max_row_permutation_error"] <= ROW_PERMUTATION_ATOL
    )
    return result


def qualification_decision(bank_blocks: list[dict], cross_bank_blocks: list[dict]) -> str:
    if len(bank_blocks) != EXPECTED_BANK_BLOCKS:
        raise ValueError(f"qualification requires exactly {EXPECTED_BANK_BLOCKS} bank blocks")
    if len(cross_bank_blocks) != EXPECTED_CROSS_BANK_BLOCKS:
        raise ValueError(
            f"qualification requires exactly {EXPECTED_CROSS_BANK_BLOCKS} cross-bank blocks"
        )
    if any(
        "pass" not in block or not isinstance(block["pass"], (bool, np.bool_))
        for block in bank_blocks + cross_bank_blocks
    ):
        raise ValueError("qualification blocks must contain boolean pass fields")
    _assert_information_barrier(bank_blocks)
    _assert_information_barrier(cross_bank_blocks)
    bank_identities = {
        (int(block.get("seed", -1)), int(block.get("bank_index", -1)))
        for block in bank_blocks
    }
    expected_bank_identities = {
        (seed, bank)
        for seed in range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"])
        for bank in range(QUALIFICATION_SETTINGS["instrument"]["banks"])
    }
    if bank_identities != expected_bank_identities:
        raise ValueError("qualification bank blocks are incomplete or duplicated")
    cross_identities = {int(block.get("seed", -1)) for block in cross_bank_blocks}
    if cross_identities != set(
        range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"])
    ):
        raise ValueError("qualification cross-bank blocks are incomplete or duplicated")
    if any(int(block.get("step", -1)) != TRAINED_STEP for block in bank_blocks + cross_bank_blocks):
        raise ValueError("qualification includes a wrong training step")
    return (
        "QUALIFIED"
        if all(bool(block["pass"]) for block in bank_blocks + cross_bank_blocks)
        else "FAILED_NATIVE_MAPPING"
    )


def _assert_information_barrier(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "slope" in normalized or "regress" in normalized:
                raise ValueError(f"forbidden qualification output key: {key}")
            _assert_information_barrier(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_information_barrier(child)


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def qualification_run_directory(commit_sha: str, root: Path | None = None) -> Path:
    root = repository_root() if root is None else Path(root).resolve()
    if len(commit_sha) != 40:
        raise ValueError("qualification run requires a full commit SHA")
    return root / "runs" / f"mapping-qualification-{commit_sha[:7]}"


def _load_attempt_attestation(root: Path) -> dict:
    path = root / ATTEMPT_ATTESTATION
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("tracked mapping-qualification attempt attestation is missing")
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ATTEMPT_ATTESTATION],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError("mapping-qualification attempt attestation is not tracked")
    attestation = json.loads(path.read_text())
    required = {
        "schema_version",
        "status",
        "attempt_id",
        "attempt_tag",
        "stream_namespace",
        "seed_root_sha",
        "query_bank_sha256",
        "checkpoint_registry_sha256",
        "protocol_sha256s",
    }
    if set(attestation) != required:
        raise ValueError("mapping-qualification attempt attestation schema mismatch")
    expected_header = {
        "schema_version": 1,
        "status": "PRECOMMITTED_ONE_SHOT",
        "attempt_id": "native-mapping-v1",
        "attempt_tag": ATTEMPT_TAG,
        "stream_namespace": STREAM_NAMESPACE,
        "seed_root_sha": SEED_ROOT_SHA,
        "query_bank_sha256": SEALED_QUERY_BANK_SHA256,
        "checkpoint_registry_sha256": SEALED_CHECKPOINT_REGISTRY_SHA256,
    }
    for key, expected in expected_header.items():
        if attestation.get(key) != expected:
            raise ValueError(f"mapping-qualification attempt attestation mismatch: {key}")
    hashes = attestation["protocol_sha256s"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("mapping-qualification protocol hash inventory is empty")
    if set(hashes) != EXPECTED_QUALIFICATION_PROTOCOL_FILES:
        raise ValueError("mapping-qualification protocol hash inventory is incomplete")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("mapping-qualification protocol hash entry is malformed")
        target = root / relative
        if target.is_symlink() or not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"mapping-qualification protocol hash mismatch: {relative}")
    return attestation


def _attempt_tag_payload(attestation: dict, commit_sha: str) -> str:
    return json.dumps(
        {
            "attempt_attestation_sha256": hashlib.sha256(
                json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "attempt_id": attestation["attempt_id"],
            "commit_sha": commit_sha,
            "seed_root_sha": SEED_ROOT_SHA,
            "stream_namespace": STREAM_NAMESPACE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _verify_attempt_tag(root: Path, commit_sha: str, attestation: dict) -> str:
    tag_ref = f"refs/tags/{ATTEMPT_TAG}"
    target = subprocess.run(
        ["git", "rev-parse", f"{tag_ref}^{{}}"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if target.returncode != 0 or target.stdout.strip() != commit_sha:
        raise ValueError("mapping-qualification attempt tag target mismatch")
    object_result = subprocess.run(
        ["git", "rev-parse", tag_ref],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tag_object = object_result.stdout.strip()
    contents = subprocess.run(
        ["git", "for-each-ref", "--format=%(contents)", tag_ref],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if contents != _attempt_tag_payload(attestation, commit_sha):
        raise ValueError("mapping-qualification attempt tag payload mismatch")
    return tag_object


def _register_attempt(root: Path, commit_sha: str, attestation: dict) -> str:
    tag_ref = f"refs/tags/{ATTEMPT_TAG}"
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", tag_ref],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if exists.returncode == 0:
        raise RuntimeError("mapping qualification is one-shot; attempt tag already exists")
    identity = subprocess.run(
        ["git", "show", "-s", "--format=%an%n%ae", commit_sha],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    if len(identity) != 2 or not identity[0] or not identity[1]:
        raise ValueError("cannot derive an attempt-tagger identity from the protocol commit")
    subprocess.run(
        [
            "git",
            "-c",
            f"user.name={identity[0]}",
            "-c",
            f"user.email={identity[1]}",
            "tag",
            "--annotate",
            ATTEMPT_TAG,
            commit_sha,
            "--message",
            _attempt_tag_payload(attestation, commit_sha),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _verify_attempt_tag(root, commit_sha, attestation)


def _require_run_path(path: Path, commit_sha: str, root: Path) -> Path:
    observed = Path(os.path.abspath(path))
    expected = qualification_run_directory(commit_sha, root)
    if observed != expected:
        raise ValueError(f"qualification path must be commit-named: {expected}")
    relative = observed.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"qualification path cannot traverse a symlink: {cursor}")
    return observed


def _qualification_lock(root: Path, commit_sha: str, query_banks: np.ndarray) -> dict:
    query_path = root / "config" / "query_bank.json"
    registry_path = root / "config" / "checkpoint_registry.json"
    prereg_path = root / "MAPPING_QUALIFICATION_PREREG.md"
    attestation_path = root / ATTEMPT_ATTESTATION
    attestation = _load_attempt_attestation(root)
    tag_object = _verify_attempt_tag(root, commit_sha, attestation)
    query_hash = sha256_file(query_path)
    registry_hash = sha256_file(registry_path)
    if query_hash != SEALED_QUERY_BANK_SHA256:
        raise ValueError("query bank differs from the sealed-v3 input")
    if registry_hash != SEALED_CHECKPOINT_REGISTRY_SHA256:
        raise ValueError("checkpoint registry differs from the sealed-v3 input")
    return {
        "schema_version": 1,
        "commit_sha": commit_sha,
        "stream_namespace": STREAM_NAMESPACE,
        "seed_root_sha": SEED_ROOT_SHA,
        "settings": QUALIFICATION_SETTINGS,
        "query_banks": query_banks.tolist(),
        "query_bank_sha256": query_hash,
        "checkpoint_registry_sha256": registry_hash,
        "prereg_sha256": sha256_file(prereg_path),
        "attempt_attestation_sha256": sha256_file(attestation_path),
        "attempt_tag": ATTEMPT_TAG,
        "attempt_tag_object": tag_object,
        "information_barrier": "no composition regression or slope is computed",
    }


def _context_digests(panel: dict) -> set[str]:
    if "contexts" in panel:
        contexts = np.asarray(panel["contexts"], dtype=np.float64)
    else:
        core = np.asarray(panel["core"], dtype=np.float64)
        reference = np.asarray(panel["reference"], dtype=np.float64)
        continuations = np.asarray(panel["continuations"], dtype=np.float64)
        base = np.concatenate([core, reference], axis=1)
        target = np.concatenate(
            [
                np.broadcast_to(
                    core[:, None, :, :],
                    (len(core), continuations.shape[1], core.shape[1], 2),
                ),
                continuations,
            ],
            axis=2,
        )
        contexts = np.concatenate([base[:, None], target], axis=1).reshape(
            -1, CONTEXT_ROWS, 2
        )
    return {
        hashlib.sha256(np.ascontiguousarray(context).tobytes()).hexdigest()
        for context in contexts
    }


def _verify_seed_and_context_disjointness(root: Path, panel: dict) -> None:
    core_seeds = np.asarray(panel["candidate_core_seed"], dtype=np.uint64)
    block_seeds = np.asarray(panel["candidate_block_seed"], dtype=np.uint64)
    expected_core = np.asarray(
        [
            derive_seed(
                SEED_ROOT_SHA,
                seed_label(STREAM_NAMESPACE, f"core-candidate:{index}"),
            )
            for index in range(len(core_seeds))
        ],
        dtype=np.uint64,
    )
    expected_block = np.asarray(
        [
            derive_seed(
                SEED_ROOT_SHA,
                seed_label(
                    STREAM_NAMESPACE,
                    f"core-candidate:{int(core_index)}:block:{int(block_index)}",
                ),
            )
            for core_index, block_index in zip(
                panel["candidate_block_core_index"], panel["candidate_block_index"]
            )
        ],
        dtype=np.uint64,
    )
    if not np.array_equal(core_seeds, expected_core):
        raise ValueError("qualification core seeds do not reproduce from locked labels")
    if not np.array_equal(block_seeds, expected_block):
        raise ValueError("qualification block seeds do not reproduce from locked labels")
    new_seeds = {int(value) for value in np.concatenate([core_seeds, block_seeds])}
    if len(new_seeds) != len(core_seeds) + len(block_seeds):
        raise ValueError("qualification candidate seeds are not unique")
    if new_seeds & FIXED_PRIOR_SEEDS:
        raise ValueError("qualification seed overlaps a fixed calibration or validation seed")

    prior_panel_paths = [
        root / "runs" / "scientific-d0b049d" / "panel.npz",
        root / "runs" / "scientific-dd749cf" / "panel.npz",
    ]
    prior_context_paths = [
        root / "artifacts" / "calibration" / "calibration_panel.npz",
        root / "runs" / "smoke-v2" / "panel.npz",
        *prior_panel_paths,
    ]
    for path in prior_context_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required prior-stream panel is missing: {path}")
    for path in prior_panel_paths:
        prior = load_numeric_npz(path)
        prior_seeds = {
            int(value)
            for value in np.concatenate(
                [prior["candidate_core_seed"], prior["candidate_block_seed"]]
            )
        }
        if new_seeds & prior_seeds:
            raise ValueError(f"qualification seed overlaps prior stream: {path}")

    new_contexts = _context_digests(panel)
    for path in prior_context_paths:
        prior_contexts = _context_digests(load_numeric_npz(path))
        if new_contexts & prior_contexts:
            raise ValueError(f"qualification context overlaps prior stream: {path}")


def build_qualification_panel(*, root: Path | None = None) -> dict:
    root = repository_root() if root is None else Path(root).resolve()
    configure_determinism(0)
    verify_runtime(root)
    commit_sha = clean_head(root)
    prior_attempts = sorted((root / "runs").glob("mapping-qualification-*"))
    prior_attempts += sorted((root / "runs").glob(".mapping-qualification-*"))
    if prior_attempts:
        raise RuntimeError(
            "mapping qualification is one-shot; an attempt path already exists: "
            + ", ".join(str(path) for path in prior_attempts)
        )
    attestation = _load_attempt_attestation(root)
    _register_attempt(root, commit_sha, attestation)
    run_dir = _require_run_path(qualification_run_directory(commit_sha, root), commit_sha, root)
    if run_dir.exists():
        raise FileExistsError(f"qualification run directory already exists: {run_dir}")
    building = run_dir.with_name(f".{run_dir.name}-building")
    if building.exists():
        raise FileExistsError(f"qualification build directory already exists: {building}")
    building.mkdir(parents=True)
    query_banks = load_locked_query_banks(root)
    metadata = generate_panel(
        commit_sha=commit_sha,
        query_banks=query_banks,
        n_groups=N_GROUPS,
        n_continuations=N_TARGETS,
        out_path=building / "panel.npz",
        interior_selected=True,
        max_core_candidates=2_000,
        max_blocks_per_core=512,
        min_within_group_sd=0.0,
        interior_weight_min=QUALIFICATION_SETTINGS["instrument"][
            "interior_weight_min"
        ],
        interior_weight_max=QUALIFICATION_SETTINGS["instrument"][
            "interior_weight_max"
        ],
        minimum_endpoint_js=QUALIFICATION_SETTINGS["instrument"][
            "minimum_endpoint_js"
        ],
        scientific=False,
        seed_namespace=STREAM_NAMESPACE,
        seed_root=SEED_ROOT_SHA,
    )
    panel = load_numeric_npz(building / "panel.npz")
    if panel["eligible_replace"].shape != (N_GROUPS, N_TARGETS):
        raise AssertionError("qualification panel has the wrong eligibility shape")
    if not panel["eligible_replace"].astype(bool).all():
        raise AssertionError("qualification panel contains an ineligible base-target pair")
    _verify_seed_and_context_disjointness(root, panel)
    lock = _qualification_lock(root, commit_sha, query_banks)
    lock["panel_sha256"] = metadata["panel_sha256"]
    write_json_atomic(building / "qualification_lock.json", lock)
    building.rename(run_dir)
    return {**metadata, "run_dir": str(run_dir), "qualification_lock": lock}


def _validate_qualification_panel(
    *, root: Path, run_dir: Path, commit_sha: str, query_banks: np.ndarray
) -> tuple[dict, dict]:
    panel_path = run_dir / "panel.npz"
    metadata_path = run_dir / "panel.json"
    lock_path = run_dir / "qualification_lock.json"
    if not (panel_path.is_file() and metadata_path.is_file() and lock_path.is_file()):
        raise FileNotFoundError("qualification panel, metadata, or lock is missing")
    panel = load_numeric_npz(panel_path)
    metadata = json.loads(metadata_path.read_text())
    lock = json.loads(lock_path.read_text())
    expected_lock = _qualification_lock(root, commit_sha, query_banks)
    for key in (
        "schema_version",
        "commit_sha",
        "stream_namespace",
        "seed_root_sha",
        "settings",
        "query_banks",
        "query_bank_sha256",
        "checkpoint_registry_sha256",
        "prereg_sha256",
        "attempt_attestation_sha256",
        "attempt_tag",
        "attempt_tag_object",
        "information_barrier",
    ):
        if lock.get(key) != expected_lock[key]:
            raise ValueError(f"qualification lock mismatch: {key}")
    if lock.get("panel_sha256") != sha256_file(panel_path):
        raise ValueError("qualification panel hash mismatch")
    if metadata.get("commit_sha") != commit_sha:
        raise ValueError("qualification panel commit mismatch")
    if metadata.get("seed_namespace") != STREAM_NAMESPACE:
        raise ValueError("qualification panel seed namespace mismatch")
    if metadata.get("n_groups") != N_GROUPS or metadata.get("n_continuations") != N_TARGETS:
        raise ValueError("qualification panel dimensions mismatch")
    expected_shapes = {
        "core": (N_GROUPS, QUALIFICATION_SETTINGS["panel"]["core_rows"], 2),
        "reference": (
            N_GROUPS,
            QUALIFICATION_SETTINGS["panel"]["reference_rows"],
            2,
        ),
        "continuations": (
            N_GROUPS,
            N_TARGETS,
            QUALIFICATION_SETTINGS["panel"]["target_rows"],
            2,
        ),
        "query_banks": (
            QUALIFICATION_SETTINGS["instrument"]["banks"],
            QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
        ),
        "f0_base": (
            QUALIFICATION_SETTINGS["instrument"]["banks"],
            N_GROUPS,
            QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
            QUALIFICATION_SETTINGS["instrument"]["bins"],
        ),
        "f1_base": (
            QUALIFICATION_SETTINGS["instrument"]["banks"],
            N_GROUPS,
            QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
            QUALIFICATION_SETTINGS["instrument"]["bins"],
        ),
        "f0_target": (
            QUALIFICATION_SETTINGS["instrument"]["banks"],
            N_GROUPS,
            N_TARGETS,
            QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
            QUALIFICATION_SETTINGS["instrument"]["bins"],
        ),
        "f1_target": (
            QUALIFICATION_SETTINGS["instrument"]["banks"],
            N_GROUPS,
            N_TARGETS,
            QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
            QUALIFICATION_SETTINGS["instrument"]["bins"],
        ),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in panel or panel[key].shape != expected_shape:
            raise ValueError(f"qualification panel shape mismatch: {key}")
    if not np.array_equal(panel["query_banks"], query_banks):
        raise ValueError("qualification query banks mismatch")
    locked_instrument = QUALIFICATION_SETTINGS["instrument"]
    selection_thresholds = {
        "selection_interior_weight_min": locked_instrument["interior_weight_min"],
        "selection_interior_weight_max": locked_instrument["interior_weight_max"],
        "selection_minimum_endpoint_js": locked_instrument["minimum_endpoint_js"],
    }
    for key, expected in selection_thresholds.items():
        if key not in panel or float(panel[key]) != expected:
            raise ValueError(f"qualification selection threshold mismatch: {key}")
    encoded_namespace = bytes(panel["selection_seed_namespace"].tolist()).decode("utf-8")
    if encoded_namespace != STREAM_NAMESPACE:
        raise ValueError("qualification panel seed namespace bytes mismatch")
    encoded_root = bytes(panel["selection_seed_root"].tolist()).decode("ascii")
    if encoded_root != SEED_ROOT_SHA or metadata.get("seed_root") != SEED_ROOT_SHA:
        raise ValueError("qualification panel seed root mismatch")
    if panel["eligible_replace"].shape != (N_GROUPS, N_TARGETS):
        raise ValueError("qualification eligibility shape mismatch")
    expected_oracle_shapes = {
        "ell_base": (N_GROUPS,),
        "ell_target": (N_GROUPS, N_TARGETS),
        "js_base": (locked_instrument["banks"], N_GROUPS),
        "js_target": (
            locked_instrument["banks"],
            N_GROUPS,
            N_TARGETS,
        ),
    }
    for key, expected_shape in expected_oracle_shapes.items():
        if key not in panel or panel[key].shape != expected_shape:
            raise ValueError(f"qualification eligibility source shape mismatch: {key}")
    base_weight = expit(np.asarray(panel["ell_base"], dtype=np.float64))
    target_weight = expit(np.asarray(panel["ell_target"], dtype=np.float64))
    interior_base = (
        base_weight >= locked_instrument["interior_weight_min"]
    ) & (base_weight <= locked_instrument["interior_weight_max"])
    interior_target = (
        target_weight >= locked_instrument["interior_weight_min"]
    ) & (target_weight <= locked_instrument["interior_weight_max"])
    js_base = np.all(
        np.asarray(panel["js_base"], dtype=np.float64)
        >= locked_instrument["minimum_endpoint_js"],
        axis=0,
    )
    js_target = np.all(
        np.asarray(panel["js_target"], dtype=np.float64)
        >= locked_instrument["minimum_endpoint_js"],
        axis=0,
    )
    recomputed_eligible = (
        interior_base[:, None] & interior_target & js_base[:, None] & js_target
    )
    if not np.array_equal(panel["eligible_replace"].astype(bool), recomputed_eligible):
        raise ValueError("qualification eligibility flags do not match locked thresholds")
    if not recomputed_eligible.all():
        raise ValueError("qualification panel includes ineligible rows")
    _verify_seed_and_context_disjointness(root, panel)
    return panel, lock


def _prediction_contexts(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    base = np.concatenate([panel["core"], panel["reference"]], axis=1)
    target = np.concatenate(
        [
            np.broadcast_to(
                panel["core"][:, None, :, :],
                (N_GROUPS, N_TARGETS, panel["core"].shape[1], 2),
            ),
            panel["continuations"],
        ],
        axis=2,
    )
    if base.shape != (N_GROUPS, CONTEXT_ROWS, 2) or target.shape != (
        N_GROUPS,
        N_TARGETS,
        CONTEXT_ROWS,
        2,
    ):
        raise AssertionError("qualification prediction contexts have the wrong shapes")
    return base, target


def _write_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("qualification manifest already exists")
    records = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"qualification artifact cannot be a symlink: {path}")
        if path.is_file() and path != manifest_path:
            records.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {"schema_version": 1, "files": records}
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_qualification_artifact(
    *, root: Path | None = None, commit_sha: str | None = None
) -> dict:
    root = repository_root() if root is None else Path(root).resolve()
    configure_determinism(0)
    commit_sha = clean_head(root) if commit_sha is None else commit_sha
    verify_runtime(root)
    run_dir = _require_run_path(qualification_run_directory(commit_sha, root), commit_sha, root)
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    if not (summary_path.is_file() and manifest_path.is_file()):
        raise RuntimeError("completed mapping qualification is missing")
    query_banks = load_locked_query_banks(root)
    panel, lock = _validate_qualification_panel(
        root=root, run_dir=run_dir, commit_sha=commit_sha, query_banks=query_banks
    )
    summary = json.loads(summary_path.read_text())
    required_summary_keys = {
        "schema_version",
        "status",
        "decision",
        "commit_sha",
        "stream_namespace",
        "panel_sha256",
        "prereg_sha256",
        "checkpoint_registry_sha256",
        "query_bank_sha256",
        "attempt_attestation_sha256",
        "attempt_tag",
        "attempt_tag_object",
        "settings",
        "groups",
        "targets_per_group",
        "checkpoint_count",
        "bank_block_count",
        "cross_bank_block_count",
        "bank_blocks",
        "cross_bank_blocks",
        "information_barrier",
        "wall_seconds",
        "peak_rss_bytes",
        "manifest_files",
        "manifest_sha256",
    }
    if set(summary) != required_summary_keys:
        raise ValueError("qualification summary schema mismatch")
    expected_header = {
        "schema_version": 1,
        "status": "COMPLETE",
        "commit_sha": commit_sha,
        "stream_namespace": STREAM_NAMESPACE,
        "panel_sha256": lock["panel_sha256"],
        "prereg_sha256": lock["prereg_sha256"],
        "checkpoint_registry_sha256": lock["checkpoint_registry_sha256"],
        "query_bank_sha256": lock["query_bank_sha256"],
        "attempt_attestation_sha256": lock["attempt_attestation_sha256"],
        "attempt_tag": lock["attempt_tag"],
        "attempt_tag_object": lock["attempt_tag_object"],
        "settings": QUALIFICATION_SETTINGS,
        "groups": N_GROUPS,
        "targets_per_group": N_TARGETS,
        "checkpoint_count": QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"],
        "bank_block_count": EXPECTED_BANK_BLOCKS,
        "cross_bank_block_count": EXPECTED_CROSS_BANK_BLOCKS,
        "information_barrier": {
            "composition_regression_computed": False,
            "composition_slope_computed": False,
        },
    }
    for key, expected in expected_header.items():
        if summary.get(key) != expected:
            raise ValueError(f"qualification summary mismatch: {key}")
    if not np.isfinite(summary.get("wall_seconds", np.nan)) or summary["wall_seconds"] < 0:
        raise ValueError("qualification wall time is invalid")
    if not isinstance(summary.get("peak_rss_bytes"), int) or summary["peak_rss_bytes"] <= 0:
        raise ValueError("qualification peak RSS is invalid")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("qualification manifest schema mismatch")
    if summary.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("qualification manifest hash mismatch")
    records = manifest["files"]
    if summary.get("manifest_files") != len(records):
        raise ValueError("qualification manifest count mismatch")
    record_paths = [record.get("path") for record in records]
    if len(record_paths) != len(set(record_paths)):
        raise ValueError("qualification manifest has duplicate paths")
    observed_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path not in {summary_path, manifest_path}
    }
    if set(record_paths) != observed_files:
        raise ValueError("qualification manifest inventory mismatch")
    for record in records:
        path = run_dir / record["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid qualification manifest path: {path}")
        if path.stat().st_size != record.get("size") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"qualification artifact mismatch: {path}")

    registry = load_checkpoint_registry(root / "config" / "checkpoint_registry.json")
    base_contexts, target_contexts = _prediction_contexts(panel)
    del base_contexts, target_contexts
    recomputed_bank_blocks = []
    recomputed_cross_blocks = []
    seen_checkpoint_hashes = set()
    prediction_dir = run_dir / "predictions"
    expected_names = {
        f"mapping_s{seed:02d}_bank{bank}.npz"
        for seed in range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"])
        for bank in range(QUALIFICATION_SETTINGS["instrument"]["banks"])
    }
    if {path.name for path in prediction_dir.iterdir() if path.is_file()} != expected_names:
        raise ValueError("qualification prediction shard inventory mismatch")
    for seed in range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"]):
        record = expanded_checkpoint_record(registry, seed, TRAINED_STEP)
        seen_checkpoint_hashes.add(record["sha256"])
        coordinate_arrays = []
        for bank_index, queries in enumerate(query_banks):
            shard = load_numeric_npz(
                prediction_dir / f"mapping_s{seed:02d}_bank{bank_index}.npz"
            )
            if (
                int(shard["seed"]) != seed
                or int(shard["step"]) != TRAINED_STEP
                or int(shard["bank_index"]) != bank_index
                or bytes(shard["checkpoint_sha256"].tolist()).hex() != record["sha256"]
                or not np.array_equal(shard["query_bank"], queries)
            ):
                raise ValueError("qualification prediction identity mismatch")
            block, arrays = mapping_block(
                shard["p_base"],
                shard["p_target"],
                panel["f0_base"][bank_index],
                panel["f1_base"][bank_index],
                panel["f0_target"][bank_index],
                panel["f1_target"][bank_index],
            )
            for key, expected in arrays.items():
                if key not in shard or not np.array_equal(shard[key], expected):
                    raise ValueError(f"qualification derived array mismatch: {key}")
            required_guard_arrays = {
                "guard_base_replay",
                "guard_base_batch_restored",
                "guard_base_row_permuted",
                "guard_target_replay",
                "guard_target_batch_restored",
                "guard_target_row_permuted",
            }
            if not required_guard_arrays <= set(shard):
                raise ValueError("qualification raw inference-guard arrays are missing")
            base_guard = inference_guard_from_predictions(
                shard["p_base"],
                shard["guard_base_replay"],
                shard["guard_base_batch_restored"],
                shard["guard_base_row_permuted"],
            )
            target_guard = inference_guard_from_predictions(
                shard["p_target"].reshape(
                    N_GROUPS * N_TARGETS,
                    QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"],
                    QUALIFICATION_SETTINGS["instrument"]["bins"],
                ),
                shard["guard_target_replay"],
                shard["guard_target_batch_restored"],
                shard["guard_target_row_permuted"],
            )
            guard = combine_inference_guards(base_guard, target_guard)
            block["inference_guard"] = guard
            block["pass"] = bool(block["pass"] and guard["pass"])
            block.update(
                {
                    "seed": seed,
                    "step": TRAINED_STEP,
                    "bank_index": bank_index,
                    "checkpoint_sha256": record["sha256"],
                }
            )
            recomputed_bank_blocks.append(block)
            coordinate_arrays.append(arrays)
        cross = cross_bank_block(coordinate_arrays[0], coordinate_arrays[1])
        cross.update(
            {
                "seed": seed,
                "step": TRAINED_STEP,
                "checkpoint_sha256": record["sha256"],
            }
        )
        recomputed_cross_blocks.append(cross)
    if len(seen_checkpoint_hashes) != QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"]:
        raise ValueError("qualification checkpoint hashes are not distinct")
    decision = qualification_decision(recomputed_bank_blocks, recomputed_cross_blocks)
    if summary["bank_blocks"] != recomputed_bank_blocks:
        raise ValueError("qualification bank summary does not match raw predictions")
    if summary["cross_bank_blocks"] != recomputed_cross_blocks:
        raise ValueError("qualification cross-bank summary does not match raw predictions")
    if summary.get("decision") != decision:
        raise ValueError("qualification decision does not match recomputed gates")
    return summary


def verify_completed_qualification(
    *, root: Path | None = None, commit_sha: str | None = None
) -> dict:
    summary = verify_qualification_artifact(root=root, commit_sha=commit_sha)
    if summary["decision"] != "QUALIFIED":
        raise RuntimeError("mapping qualification did not pass")
    return summary


def score_qualification(*, root: Path | None = None) -> dict:
    started = time.perf_counter()
    root = repository_root() if root is None else Path(root).resolve()
    configure_determinism(0)
    verify_runtime(root)
    commit_sha = clean_head(root)
    run_dir = _require_run_path(qualification_run_directory(commit_sha, root), commit_sha, root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"qualification run directory is missing: {run_dir}")
    if (run_dir / "summary.json").exists() or (run_dir / "manifest.json").exists():
        raise FileExistsError("qualification scoring is already complete")
    building = run_dir / ".score-building"
    if building.exists() or (run_dir / "predictions").exists():
        raise FileExistsError("qualification scoring has a prior partial attempt")
    query_banks = load_locked_query_banks(root)
    panel, lock = _validate_qualification_panel(
        root=root, run_dir=run_dir, commit_sha=commit_sha, query_banks=query_banks
    )
    registry = load_checkpoint_registry(root / "config" / "checkpoint_registry.json")
    base_contexts, target_contexts = _prediction_contexts(panel)
    building.mkdir()
    prediction_dir = building / "predictions"
    prediction_dir.mkdir()
    bank_blocks = []
    cross_blocks = []
    for seed in range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"]):
        record = expanded_checkpoint_record(registry, seed, TRAINED_STEP)
        model = load_registered_checkpoint(record)
        coordinate_arrays = []
        for bank_index, queries in enumerate(query_banks):
            p_base = predict_probabilities(
                model, base_contexts, queries, batch_size=PHYSICAL_BATCH_SIZE
            )
            flat_target_contexts = target_contexts.reshape(
                N_GROUPS * N_TARGETS, CONTEXT_ROWS, 2
            )
            flat_target = predict_probabilities(
                model,
                flat_target_contexts,
                queries,
                batch_size=PHYSICAL_BATCH_SIZE,
            )
            p_target = flat_target.reshape(N_GROUPS, N_TARGETS, len(queries), -1)
            block, arrays = mapping_block(
                p_base,
                p_target,
                panel["f0_base"][bank_index],
                panel["f1_base"][bank_index],
                panel["f0_target"][bank_index],
                panel["f1_target"][bank_index],
            )
            base_guard, base_guard_arrays = inference_guard(
                model, base_contexts, queries, p_base
            )
            target_guard, target_guard_arrays = inference_guard(
                model, flat_target_contexts, queries, flat_target
            )
            guard = combine_inference_guards(base_guard, target_guard)
            block["inference_guard"] = guard
            block["pass"] = bool(block["pass"] and guard["pass"])
            block.update(
                {
                    "seed": seed,
                    "step": TRAINED_STEP,
                    "bank_index": bank_index,
                    "checkpoint_sha256": record["sha256"],
                }
            )
            bank_blocks.append(block)
            coordinate_arrays.append(arrays)
            write_numeric_npz_atomic(
                prediction_dir / f"mapping_s{seed:02d}_bank{bank_index}.npz",
                p_base=p_base,
                p_target=p_target,
                query_bank=queries,
                seed=np.asarray(seed, dtype=np.int16),
                step=np.asarray(TRAINED_STEP, dtype=np.int32),
                bank_index=np.asarray(bank_index, dtype=np.int8),
                checkpoint_sha256=np.frombuffer(bytes.fromhex(record["sha256"]), dtype=np.uint8),
                guard_base_replay=base_guard_arrays["replay"],
                guard_base_batch_restored=base_guard_arrays["batch_restored"],
                guard_base_row_permuted=base_guard_arrays["row_permuted"],
                guard_target_replay=target_guard_arrays["replay"],
                guard_target_batch_restored=target_guard_arrays["batch_restored"],
                guard_target_row_permuted=target_guard_arrays["row_permuted"],
                **arrays,
            )
        cross = cross_bank_block(coordinate_arrays[0], coordinate_arrays[1])
        cross.update(
            {
                "seed": seed,
                "step": TRAINED_STEP,
                "checkpoint_sha256": record["sha256"],
            }
        )
        cross_blocks.append(cross)
        del model
    expected_names = {
        f"mapping_s{seed:02d}_bank{bank}.npz"
        for seed in range(QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"])
        for bank in range(QUALIFICATION_SETTINGS["instrument"]["banks"])
    }
    observed_names = {path.name for path in prediction_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise AssertionError("qualification prediction fleet is incomplete")
    decision = qualification_decision(bank_blocks, cross_blocks)
    prediction_dir.rename(run_dir / "predictions")
    building.rmdir()
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": decision,
        "commit_sha": commit_sha,
        "stream_namespace": STREAM_NAMESPACE,
        "panel_sha256": lock["panel_sha256"],
        "prereg_sha256": lock["prereg_sha256"],
        "checkpoint_registry_sha256": lock["checkpoint_registry_sha256"],
        "query_bank_sha256": lock["query_bank_sha256"],
        "attempt_attestation_sha256": lock["attempt_attestation_sha256"],
        "attempt_tag": lock["attempt_tag"],
        "attempt_tag_object": lock["attempt_tag_object"],
        "settings": QUALIFICATION_SETTINGS,
        "groups": N_GROUPS,
        "targets_per_group": N_TARGETS,
        "checkpoint_count": QUALIFICATION_SETTINGS["fleet"]["checkpoint_count"],
        "bank_block_count": len(bank_blocks),
        "cross_bank_block_count": len(cross_blocks),
        "bank_blocks": bank_blocks,
        "cross_bank_blocks": cross_blocks,
        "information_barrier": {
            "composition_regression_computed": False,
            "composition_slope_computed": False,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    manifest = _write_manifest(run_dir)
    summary["manifest_files"] = len(manifest["files"])
    summary["manifest_sha256"] = sha256_file(run_dir / "manifest.json")
    _assert_information_barrier(bank_blocks)
    _assert_information_barrier(cross_blocks)
    write_json_atomic(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("panel")
    subparsers.add_parser("score")
    subparsers.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "panel":
        result = build_qualification_panel()
    elif args.command == "score":
        result = score_qualification()
    else:
        result = verify_qualification_artifact()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "score" and result.get("decision") != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
