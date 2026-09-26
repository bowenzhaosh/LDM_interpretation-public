"""Shared fail-closed contracts for the Phase-1 confirmatory replication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch

from .phase1_ordering import (
    _acquire_attempt_lease,
    configure_runtime,
    verify_runtime_binary_inventory,
)


CONFIRMATION_CONFIG = "config/phase1_ordering_confirmation.json"
INDEPENDENT_QUALIFICATION_NAME = "independent_verification.json"
QUALIFICATION_PROTOCOL_VERSION = 3
QUALIFICATION_SOURCE_TAG = "phase1-ordering-qualification-v3"
QUALIFICATION_VERIFIER_TAG = "phase1-ordering-qualification-v3-verifier-fix1"
CHECKPOINT_REGISTRY_SHA256 = (
    "8824f1fdf2a6dc24977402b4426f18a0c037a8c3f900e64aaa0bf80499314194"
)
CHECKPOINT_REMOTE_ROOT = "/engrfs/project/class/zhao.b/pfn-dag-e18b/nets4_xlong"
QUALIFICATION_V3_CONTEXT_SEEDS = frozenset(
    (*range(880_943_000, 880_943_003), *range(880_953_000, 880_953_003))
)
CANONICAL_CONFIRMATION_SOURCES = (
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sha256_named_arrays(
    arrays: dict[str, np.ndarray], names: Iterable[str] | None = None
) -> str:
    ordered = sorted(arrays) if names is None else list(names)
    digest = hashlib.sha256()
    for name in ordered:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _checkpoint_filename(prior: str, seed: int, step: int) -> str:
    prefix = f"M4_{prior}_s{seed}_st120000"
    return f"{prefix}.pt" if step == 120_000 else f"{prefix}_ck{step}.pt"


def _validate_checkpoint_registry_contents(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    verify_remote_files: bool,
) -> None:
    if set(registry) != {
        "schema_version",
        "scope",
        "remote_root",
        "model_definition",
        "model_definition_sha256",
        "training_sidecars",
        "records",
    }:
        raise RuntimeError("checkpoint registry key inventory drifted")
    if (
        registry.get("schema_version") != 1
        or registry.get("scope") != "fixed archived d=4 base 120k fleet"
        or registry.get("remote_root") != CHECKPOINT_REMOTE_ROOT
        or registry.get("model_definition") != config.get("model_definition")
        or registry.get("model_definition_sha256")
        != config.get("model_definition_sha256")
    ):
        raise RuntimeError("checkpoint registry identity drifted")
    records = registry.get("records")
    expected_tuples = {
        (prior, seed, step)
        for prior in ("C", "N")
        for seed in range(3)
        for step in (20_000, 60_000, 120_000)
    }
    if not isinstance(records, list) or len(records) != len(expected_tuples):
        raise RuntimeError("checkpoint registry does not contain exactly 18 records")
    observed_tuples: set[tuple[str, int, int]] = set()
    filenames: set[str] = set()
    hashes: set[str] = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {
            "prior",
            "seed",
            "checkpoint_step",
            "planned_total_steps",
            "filename",
            "bytes",
            "sha256",
        }:
            raise RuntimeError("checkpoint registry record schema drifted")
        key = (str(row["prior"]), int(row["seed"]), int(row["checkpoint_step"]))
        if key not in expected_tuples or key in observed_tuples:
            raise RuntimeError("checkpoint registry tuple inventory drifted")
        observed_tuples.add(key)
        expected_name = _checkpoint_filename(*key)
        expected_bytes = 4_343_220 if key[2] == 120_000 else 4_344_012
        if (
            row["planned_total_steps"] != 120_000
            or row["filename"] != expected_name
            or row["bytes"] != expected_bytes
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise RuntimeError("checkpoint registry record semantics drifted")
        filenames.add(str(row["filename"]))
        hashes.add(str(row["sha256"]))
    if observed_tuples != expected_tuples or len(filenames) != 18 or len(hashes) != 18:
        raise RuntimeError("checkpoint registry records are not unique and complete")
    sidecars = registry.get("training_sidecars")
    expected_sidecars = {(prior, seed) for prior in ("C", "N") for seed in range(3)}
    if not isinstance(sidecars, list) or len(sidecars) != len(expected_sidecars):
        raise RuntimeError("checkpoint training-sidecar inventory drifted")
    observed_sidecars: set[tuple[str, int]] = set()
    for row in sidecars:
        if not isinstance(row, dict) or set(row) != {
            "prior",
            "seed",
            "filename",
            "bytes",
            "sha256",
        }:
            raise RuntimeError("checkpoint training-sidecar schema drifted")
        key = (str(row["prior"]), int(row["seed"]))
        if key not in expected_sidecars or key in observed_sidecars:
            raise RuntimeError("checkpoint training-sidecar tuple drifted")
        observed_sidecars.add(key)
        if (
            row["filename"] != f"M4_{key[0]}_s{key[1]}_st120000.json"
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise RuntimeError("checkpoint training-sidecar semantics drifted")
    if observed_sidecars != expected_sidecars:
        raise RuntimeError("checkpoint training-sidecar inventory incomplete")
    if not verify_remote_files:
        return
    root = Path(CHECKPOINT_REMOTE_ROOT).resolve(strict=True)
    for row in records:
        path = (root / str(row["filename"])).resolve(strict=True)
        if root not in path.parents:
            raise RuntimeError("checkpoint escapes the registered root")
        payload = path.read_bytes()
        if (
            len(payload) != row["bytes"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise RuntimeError(f"checkpoint bytes do not match registry: {path}")
    for row in sidecars:
        path = (root / str(row["filename"])).resolve(strict=True)
        if root not in path.parents:
            raise RuntimeError("training sidecar escapes the registered root")
        payload = path.read_bytes()
        if (
            len(payload) != row["bytes"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise RuntimeError(f"training sidecar bytes do not match registry: {path}")
        metadata = json.loads(payload)
        if not isinstance(metadata, dict) or set(metadata) != {
            "d",
            "final_loss",
            "n_params",
            "prior",
            "scale",
            "seed",
            "steps",
            "wallclock_s",
        }:
            raise RuntimeError("training sidecar metadata schema drifted")
        if (
            metadata["d"] != 4
            or metadata["prior"] != row["prior"]
            or metadata["seed"] != row["seed"]
            or metadata["steps"] != 120_000
            or metadata["scale"] != "base"
            or metadata["n_params"] != 1_082_980
            or not np.isfinite(float(metadata["final_loss"]))
            or not np.isfinite(float(metadata["wallclock_s"]))
        ):
            raise RuntimeError("training sidecar metadata does not match its arm")


def validate_checkpoint_registry(
    config: dict[str, Any], *, verify_remote_files: bool
) -> dict[str, Any]:
    path = resolve_repo_path(str(config["checkpoint_registry"]))
    if (
        config.get("checkpoint_registry_sha256") != CHECKPOINT_REGISTRY_SHA256
        or sha256_file(path) != CHECKPOINT_REGISTRY_SHA256
    ):
        raise RuntimeError("checkpoint registry hash mismatch")
    registry = load_json(path)
    _validate_checkpoint_registry_contents(
        config, registry, verify_remote_files=verify_remote_files
    )
    return registry


def _qualification_artifact_hashes(
    config: dict[str, Any], verification: dict[str, Any]
) -> dict[str, str]:
    if config.get("qualification_protocol_version") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("confirmation requires qualification protocol v3")
    if config.get("qualification_source_tag") != QUALIFICATION_SOURCE_TAG:
        raise RuntimeError(
            "confirmation requires the frozen qualification v3 source tag"
        )
    if (
        verification.get("schema_version") != QUALIFICATION_PROTOCOL_VERSION
        or verification.get("qualification_protocol_version")
        != QUALIFICATION_PROTOCOL_VERSION
    ):
        raise RuntimeError("independent qualification is not protocol v3")
    expected_fields = {
        "verification": "INDEPENDENT_RAW_RECOMPUTATION_PASS",
        "decision": "QUALIFICATION_PASS",
        "source_commit": config.get("qualification_source_commit"),
        "source_tag": config.get("qualification_source_tag"),
        "verifier_commit": config.get("qualification_verifier_commit"),
        "verifier_source_tag": config.get("qualification_verifier_tag"),
        "verifier_source_sha256": config.get("qualification_verifier_source_sha256"),
        "selected_truncation": config.get("selected_truncation"),
        "reference_truncation": config.get("qualification_reference_truncation"),
        "bank_atom_sha256": [row["sha256"] for row in config.get("atom_banks", [])],
        "determinism_canary_sha256": config.get("atom_determinism_canary_sha256"),
    }
    if any(verification.get(name) != value for name, value in expected_fields.items()):
        raise RuntimeError(
            "independent qualification verdict does not unlock confirmation"
        )
    optional_canary_fields = {
        "atom_determinism_canary_seed": "atom_determinism_canary_seed",
        "determinism_canary_seed": "atom_determinism_canary_seed",
        "atom_determinism_canary_count": "atom_determinism_canary_count",
        "determinism_canary_count": "atom_determinism_canary_count",
    }
    for verification_key, config_key in optional_canary_fields.items():
        if verification_key in verification and verification[
            verification_key
        ] != config.get(config_key):
            raise RuntimeError("independent qualification canary identity mismatch")
    artifact_fields = {
        "COMPLETE.json": (
            "qualification_complete_sha256",
            "joined_complete_sha256",
        ),
        "qualification_raw.npz": (
            "qualification_raw_sha256",
            "joined_raw_sha256",
        ),
        "qualification_summary.json": (
            "qualification_summary_sha256",
            "joined_summary_sha256",
        ),
    }
    expected: dict[str, str] = {}
    for filename, (config_key, verification_key) in artifact_fields.items():
        configured_hash = config.get(config_key)
        if not isinstance(configured_hash, str) or len(configured_hash) != 64:
            raise RuntimeError(f"invalid configured qualification hash: {config_key}")
        if verification.get(verification_key) != configured_hash:
            raise RuntimeError(
                f"independent qualification artifact hash mismatch: {filename}"
            )
        expected[filename] = configured_hash
    return expected


def _verify_qualification_artifacts(
    qualification_root: Path, expected_hashes: dict[str, str], *, label: str
) -> None:
    joined = qualification_root / "joined"
    for filename, expected_hash in expected_hashes.items():
        path = joined / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"{label} qualification artifact mismatch: {path}")


def git_provenance(
    require_clean: bool, required_tag: str | None = None
) -> dict[str, Any]:
    root = repo_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if require_clean and status:
        raise RuntimeError("confirmatory execution refuses a dirty source tree")
    if required_tag:
        tag_ref = f"refs/tags/{required_tag}"
        tag_type = subprocess.run(
            ["git", "cat-file", "-t", tag_ref],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if tag_type != "tag":
            raise RuntimeError(f"required attempt tag is not annotated: {required_tag}")
        tag_commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if tag_commit != commit:
            raise RuntimeError(
                f"required attempt tag {required_tag} does not resolve to HEAD"
            )
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def validate_confirmation_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    canonical_config = (repo_root() / CONFIRMATION_CONFIG).resolve()
    if config_path != canonical_config:
        raise RuntimeError("confirmation requires the canonical repository config path")
    config = load_json(config_path)
    if config.get("qualification_protocol_version") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("confirmation requires qualification protocol v3")
    fixed = {
        "schema_version": 1,
        "status": "confirmatory_locked",
        "scope": "fixed archived d4 base fleet ordering-use replication",
        "required_attempt_tag": "phase1-ordering-confirmation-v1-fix2",
        "protocol_amendment": "PHASE1_ORDERING_CONFIRMATION_FIX2.md",
        "dimension": 4,
        "context_size": 30,
        "native_output_bins": 100,
        "priors": ["C", "N"],
        "evaluation_draws": 3,
        "contexts_per_prior_draw": 1067,
        "shard_assignment": "full-stream-then-modulo-v1",
        "atom_count": 3_000_000,
        "atom_hash_scheme": "sha256-c-contiguous-raw-bytes-v1",
        "atom_determinism_canary_seed": 881_103_999,
        "atom_determinism_canary_count": 4096,
        "selected_truncation": 16_384,
        "qualification_reference_truncation": 32_768,
        "nested_half_atom_count": 1_500_000,
        "pfn_batch_size": 64,
        "pfn_replay_rows_per_stratum": 8,
        "pfn_batch_logp_atol": 5e-5,
        "pfn_combined_context_batch_logp_atol": 8e-5,
        "pfn_replay_permutation_seeds": [1_212_240_001, 1_212_240_002],
        "pfn_replay_probability_atol": 1e-6,
        "pfn_replay_total_variation_atol": 3e-6,
        "pfn_context_permutation_roll": 7,
        "pfn_context_permutation_logp_atol": 3e-5,
        "quadrature_interior_nodes": 32,
        "quadrature_tail_nodes": 128,
        "query_grid_chunk": 16,
        "context_atom_batch": 250_000,
        "oracle_compute_dtype": "float64",
        "probability_sum_atol": 1e-8,
        "qualification_protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "qualification_source_tag": QUALIFICATION_SOURCE_TAG,
        "generator_module": "artifacts/phase1/d4_generator.py",
        "model_definition": "artifacts/phase1/d4_train_fleet.py",
        "checkpoint_registry": "config/phase1_checkpoint_registry.json",
        "checkpoint_registry_sha256": CHECKPOINT_REGISTRY_SHA256,
        "environment_contract": "environment/phase1-washu-runtime.json",
        "runtime_binary_inventory": "environment/phase1-washu-binary-inventory.json",
        "requirements_lock": "environment/phase1-washu-requirements-lock.txt",
        "cluster_wrapper": "cluster/phase1_confirmation.sbatch",
        "cluster_launcher": "cluster/submit_phase1_confirmation.sh",
        "independent_verifier": "src/pfn_dag_verify/phase1_confirmation_verify.py",
        "require_clean_git": True,
    }
    if any(config.get(name) != value for name, value in fixed.items()):
        raise RuntimeError("confirmation config differs from its frozen core")
    if config.get("evaluation_seeds") != {
        "C": [881_003_000, 881_003_001, 881_003_002],
        "N": [881_013_000, 881_013_001, 881_013_002],
    }:
        raise RuntimeError("confirmation evaluation seeds drifted")
    evaluation_seeds = {
        int(seed) for values in config["evaluation_seeds"].values() for seed in values
    }
    if not evaluation_seeds.isdisjoint(QUALIFICATION_V3_CONTEXT_SEEDS):
        raise RuntimeError("confirmation reuses a qualification-v3 context seed")
    expected_banks = [
        (
            0,
            881_003_101,
            "827c9ec7b2f36720c62b4a4c89ccd82a27252238a943b0762a6a5133b25ad0d7",
        ),
        (
            1,
            881_003_102,
            "a848e61b929c18decaadd9b633f60f22e524f17a2b1ec24d6c7642dbba764920",
        ),
        (
            2,
            881_003_103,
            "08351198797063f55fcbf37781a0aef1f891a4b58d1183f7d69e94876167a9b3",
        ),
    ]
    observed_banks = [
        (int(row["bank_index"]), int(row["seed"]), str(row["sha256"]))
        for row in config.get("atom_banks", [])
    ]
    if observed_banks != expected_banks:
        raise RuntimeError("confirmation atom-bank registry drifted")
    if config.get("nested_half_subset") != {
        "draw_index": 0,
        "stream_index_stop_exclusive": 200,
    }:
        raise RuntimeError("confirmation nested-half subset drifted")
    if config.get("bootstrap") != {
        "replicates": 50_000,
        "master_seed": 881_003_900,
        "bit_generator": "PCG64",
        "child_seed_fields": [
            "master_seed",
            "prior_code",
            "evaluation_seed",
            "atom_seed",
        ],
        "chunk_size": 256,
        "nested_half_namespace": 1_212_238_918,
        "quantile_method": "linear",
        "index_stream_encoding": "little-endian-int64-c-order-v1",
    }:
        raise RuntimeError("confirmation bootstrap contract drifted")
    if config.get("gates") != {
        "value_c_one_sided_quantile": 0.05,
        "value_n_equivalence_lower": -1e-5,
        "value_n_equivalence_upper": 1e-5,
        "nested_half_causal_ablated_abs_max": 0.0005,
        "nested_half_control_subtracted_ablated_abs_max": 0.0005,
        "nested_half_full_abs_max": 0.0005,
        "full_oracle_half_change_fraction_of_positive_gap_max": 0.2,
        "kl_alarm_ci_upper_below": -0.004,
        "primary_effect_floor": -0.008,
        "qualified_topk_quadrature_clearance": 0.0005,
        "primary_numerical_clearance": 0.001,
    }:
        raise RuntimeError("confirmation scientific gates drifted")
    expected_top_level = {
        "schema_version",
        "status",
        "scope",
        "required_attempt_tag",
        "protocol_amendment",
        "protocol_amendment_sha256",
        "dimension",
        "context_size",
        "native_output_bins",
        "priors",
        "evaluation_draws",
        "contexts_per_prior_draw",
        "evaluation_seeds",
        "shard_assignment",
        "atom_count",
        "atom_hash_scheme",
        "atom_banks",
        "atom_determinism_canary_seed",
        "atom_determinism_canary_count",
        "atom_determinism_canary_sha256",
        "selected_truncation",
        "qualification_reference_truncation",
        "qualification_protocol_version",
        "qualification_source_commit",
        "qualification_source_tag",
        "qualification_verifier_commit",
        "qualification_verifier_tag",
        "qualification_verifier_source_sha256",
        "qualification_local_root",
        "qualification_remote_root",
        "qualification_complete_sha256",
        "qualification_raw_sha256",
        "qualification_summary_sha256",
        "qualification_independent_verification_sha256",
        "generator_module",
        "generator_sha256",
        "model_definition",
        "model_definition_sha256",
        "checkpoint_registry",
        "checkpoint_registry_sha256",
        "environment_contract",
        "runtime_binary_inventory",
        "runtime_binary_fingerprint",
        "requirements_lock",
        "cluster_wrapper",
        "cluster_launcher",
        "independent_verifier",
        "independent_verifier_sha256",
        "require_clean_git",
        "quadrature_interior_nodes",
        "quadrature_tail_nodes",
        "query_grid_chunk",
        "context_atom_batch",
        "oracle_compute_dtype",
        "probability_sum_atol",
        "nested_half_atom_count",
        "nested_half_subset",
        "pfn_batch_size",
        "pfn_replay_rows_per_stratum",
        "pfn_batch_logp_atol",
        "pfn_combined_context_batch_logp_atol",
        "pfn_replay_permutation_seeds",
        "pfn_replay_probability_atol",
        "pfn_replay_total_variation_atol",
        "pfn_context_permutation_roll",
        "pfn_context_permutation_logp_atol",
        "bootstrap",
        "gates",
    }
    if set(config) != expected_top_level:
        raise RuntimeError("confirmation config key inventory drifted")
    expected_files = {
        "generator_module": "generator_sha256",
        "model_definition": "model_definition_sha256",
        "checkpoint_registry": "checkpoint_registry_sha256",
        "independent_verifier": "independent_verifier_sha256",
    }
    for path_key, hash_key in expected_files.items():
        path = resolve_repo_path(str(config[path_key]))
        if sha256_file(path) != config[hash_key]:
            raise RuntimeError(f"confirmation file hash mismatch: {path_key}")
    validate_checkpoint_registry(config, verify_remote_files=False)
    amendment = resolve_repo_path(str(config["protocol_amendment"]))
    if sha256_file(amendment) != config["protocol_amendment_sha256"]:
        raise RuntimeError("confirmation protocol-amendment hash mismatch")
    wrapper = resolve_repo_path(str(config["cluster_wrapper"]))
    if wrapper != (repo_root() / "cluster/phase1_confirmation.sbatch").resolve():
        raise RuntimeError("confirmation cluster-wrapper path drifted")
    launcher = resolve_repo_path(str(config["cluster_launcher"]))
    if launcher != (repo_root() / "cluster/submit_phase1_confirmation.sh").resolve():
        raise RuntimeError("confirmation cluster-launcher path drifted")
    local_root = resolve_repo_path(str(config["qualification_local_root"]))
    verification_path = local_root / INDEPENDENT_QUALIFICATION_NAME
    if (
        sha256_file(verification_path)
        != config["qualification_independent_verification_sha256"]
    ):
        raise RuntimeError("independent qualification verification hash mismatch")
    verification = load_json(verification_path)
    qualification_hashes = _qualification_artifact_hashes(config, verification)
    remote_root = Path(str(config["qualification_remote_root"])).resolve()
    for label, qualification_root in (("local", local_root), ("remote", remote_root)):
        if label == "remote" and not qualification_root.exists():
            continue
        _verify_qualification_artifacts(
            qualification_root, qualification_hashes, label=label
        )
    tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", str(config["qualification_source_tag"])],
        cwd=repo_root(),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tag_commit != config["qualification_source_commit"]:
        raise RuntimeError("qualification source tag/commit mismatch")
    if config["qualification_verifier_tag"] != QUALIFICATION_VERIFIER_TAG:
        raise RuntimeError("qualification verifier tag drifted")
    verifier_tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", str(config["qualification_verifier_tag"])],
        cwd=repo_root(),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if verifier_tag_commit != config["qualification_verifier_commit"]:
        raise RuntimeError("qualification verifier tag/commit mismatch")
    verifier_source = repo_root() / "src/pfn_dag_verify/phase1_qualification_verify.py"
    if sha256_file(verifier_source) != config["qualification_verifier_source_sha256"]:
        raise RuntimeError("qualification verifier source hash mismatch")
    return config


def verify_locked_runtime(
    config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    configure_runtime(0)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production confirmation requires its locked CUDA runtime")
    contract = load_json(resolve_repo_path(str(config["environment_contract"])))
    expected_inventory = load_json(
        resolve_repo_path(str(config["runtime_binary_inventory"]))
    )
    observed_inventory = verify_runtime_binary_inventory(expected_inventory, device)
    observed = {
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if any(observed.get(name) != contract.get(name) for name in observed):
        raise RuntimeError("confirmation runtime contract mismatch")
    if (
        observed_inventory["runtime_binary_fingerprint"]
        != config["runtime_binary_fingerprint"]
    ):
        raise RuntimeError("confirmation runtime fingerprint mismatch")
    return observed_inventory


def expected_attempt_identity(
    config_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    config = validate_confirmation_config(config_path)
    git = git_provenance(
        bool(config["require_clean_git"]), str(config["required_attempt_tag"])
    )
    root = repo_root()
    paths = [(root / relative).resolve() for relative in CANONICAL_CONFIRMATION_SOURCES]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise RuntimeError(f"canonical confirmation source is missing: {missing}")
    sources = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    identity = {
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path.resolve()),
        "source_inventory": sources,
        "runtime_binary_fingerprint": config["runtime_binary_fingerprint"],
        "qualification_complete_sha256": config["qualification_complete_sha256"],
        "qualification_independent_verification_sha256": config[
            "qualification_independent_verification_sha256"
        ],
        "checkpoint_registry_sha256": config["checkpoint_registry_sha256"],
    }
    return identity, sha256_json(identity), config, git


def attempt_identity(
    config_path: Path,
    executable_paths: Iterable[Path] = (),
    *,
    device: torch.device,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    identity, identity_sha256, config, git = expected_attempt_identity(config_path)
    # Every production stage, including panel generation, must fail before doing
    # scientific work when the archived fleet or its training sidecars are not
    # mounted exactly as registered. Static config validation opts out explicitly.
    validate_checkpoint_registry(config, verify_remote_files=True)
    root = repo_root()
    canonical = {(root / value).resolve() for value in CANONICAL_CONFIRMATION_SOURCES}
    supplied = {path.resolve() for path in executable_paths}
    if not supplied <= canonical:
        raise RuntimeError(
            "stage supplied a source outside the canonical attempt inventory"
        )
    runtime = verify_locked_runtime(config, device)
    if runtime["runtime_binary_fingerprint"] != identity["runtime_binary_fingerprint"]:
        raise RuntimeError("observed runtime differs from canonical attempt identity")
    return identity, identity_sha256, config, git


def acquire_empty_output(output_dir: Path, identity_sha256: str) -> Path:
    output_dir = output_dir.resolve()
    complete = output_dir / "COMPLETE.json"
    if complete.exists() or (output_dir / "PANEL_COMPLETE.json").exists():
        raise FileExistsError(
            f"completed confirmatory output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    lease = output_dir / "ATTEMPT.lock"
    _acquire_attempt_lease(lease, identity_sha256, False)
    if any(path != lease for path in output_dir.iterdir()):
        raise FileExistsError(f"nonempty confirmatory output directory: {output_dir}")
    return lease
