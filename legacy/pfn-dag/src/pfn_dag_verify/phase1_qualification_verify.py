"""Independent verifier for the frozen Phase-1 oracle qualification artifacts.

This module deliberately does not import the qualification runner or joiner.
It recomputes the scientific gate from sealed raw arrays and verifies the
artifact source and verifier source against separately recorded Git commits.
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


COMMIT = "cdd541c2ac7038b5cb8c7c6d3f1f6ac1811e4b88"
V3_SOURCE_TAG = "phase1-ordering-qualification-v3"
V3_VERIFIER_FIX_TAG = "phase1-ordering-qualification-v3-verifier-fix1"
V3_VERIFIER_FIX1_JOINED_SHA256 = {
    "COMPLETE.json": "dff6b9a4eaf826744911c49cd1d391597e5fead34a74f45bac2d8fe27f83bcdd",
    "qualification_raw.npz": "a4bc89e5f22f41764ddefb310bc2fb67dc291c0d0156949c1840841549915317",
    "qualification_summary.json": "e0d30a980de7f1e520e1fc3292edb6a028093f5443e269a3c96d92d5b2abeb30",
}
CANDIDATES = (8192, 16384)
REFERENCE = 32768
PRIORS = ("C", "N")
PREDICTORS = ("full", "ablated")
EXPECTED_SHARD_ARTIFACTS = {
    "ATOM_BANK.json",
    "RUNNING.json",
    "calibration_raw.npz",
    "calibration_summary.json",
    "partial_C.npz",
    "partial_N.npz",
}
EXPECTED_RAW_V1 = {
    "prior_codes": ((2,), np.dtype(np.int64)),
    "candidates": ((2,), np.dtype(np.int64)),
    "reference_truncation": ((1,), np.dtype(np.int64)),
    "attempt_identity_sha256": ((2, 32), np.dtype(np.uint8)),
    "completed": ((2, 160), np.dtype(np.int8)),
    "keep_full": ((2, 3, 160), np.dtype(np.float64)),
    "keep_ablated": ((2, 3, 160), np.dtype(np.float64)),
    "ess_full_atoms": ((2, 160), np.dtype(np.float64)),
    "ess_ablated_atoms": ((2, 160), np.dtype(np.float64)),
    "full_probability_sum_error": ((2, 3, 160), np.dtype(np.float64)),
    "ablated_probability_sum_error": ((2, 3, 160), np.dtype(np.float64)),
    "full_probability_minimum": ((2, 3, 160), np.dtype(np.float64)),
    "ablated_probability_minimum": ((2, 3, 160), np.dtype(np.float64)),
    "js_full": ((2, 2, 160), np.dtype(np.float64)),
    "js_ablated": ((2, 2, 160), np.dtype(np.float64)),
    "abs_logp_change_full": ((2, 2, 160), np.dtype(np.float64)),
    "abs_logp_change_ablated": ((2, 2, 160), np.dtype(np.float64)),
}


def _expected_raw(protocol_version: int) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    expected = dict(EXPECTED_RAW_V1)
    if protocol_version >= 2:
        expected.update(
            {
                "outcome_bins": ((2, 160), np.dtype(np.int64)),
                "full_probability": ((2, 3, 160, 100), np.dtype(np.float64)),
                "ablated_probability": ((2, 3, 160, 100), np.dtype(np.float64)),
            }
        )
    if protocol_version == 3:
        expected.update(
            {
                "quadrature_grid_interior_nodes": ((2,), np.dtype(np.int64)),
                "quadrature_grid_tail_nodes": ((2,), np.dtype(np.int64)),
                "quadrature_truncation_levels": ((3,), np.dtype(np.int64)),
                "quadrature_grid_full_probability": (
                    (2, 2, 3, 160, 100),
                    np.dtype(np.float64),
                ),
                "quadrature_grid_ablated_probability": (
                    (2, 2, 3, 160, 100),
                    np.dtype(np.float64),
                ),
                "quadrature_js_full": ((2, 3, 160), np.dtype(np.float64)),
                "quadrature_js_ablated": ((2, 3, 160), np.dtype(np.float64)),
                "quadrature_max_bin_abs_logp_change_full": (
                    (2, 3, 160),
                    np.dtype(np.float64),
                ),
                "quadrature_max_bin_abs_logp_change_ablated": (
                    (2, 3, 160),
                    np.dtype(np.float64),
                ),
                "quadrature_reference_weighted_abs_logp_change_full": (
                    (2, 3, 160),
                    np.dtype(np.float64),
                ),
                "quadrature_reference_weighted_abs_logp_change_ablated": (
                    (2, 3, 160),
                    np.dtype(np.float64),
                ),
                "quadrature_max_bin_abs_ordering_value_change": (
                    (2, 3, 160),
                    np.dtype(np.float64),
                ),
            }
        )
    elif protocol_version not in {1, 2}:
        raise ValueError(f"unsupported qualification protocol version: {protocol_version}")
    return expected
THRESHOLDS = {
    "median_js": 9e-5,
    "p95_js": 9e-4,
    "median_abs_logp_change": 0.0018,
    "p95_abs_logp_change": 0.009,
}
BASE_THRESHOLDS = {
    "median_js": 1e-4,
    "p95_js": 1e-3,
    "median_abs_logp_change": 0.002,
    "p95_abs_logp_change": 0.01,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(payload.encode())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _verify_annotated_tag(repo: Path, tag: str, commit: str) -> None:
    tag_ref = f"refs/tags/{tag}"
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", tag_ref],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tag_type != "tag":
        raise RuntimeError(f"required source tag is not annotated: {tag}")
    tagged_commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tagged_commit != commit:
        raise RuntimeError(
            f"required source tag {tag} resolves to {tagged_commit}, not {commit}"
        )


def _relative_source_path(
    absolute: str, expected_relative_paths: set[str]
) -> str:
    absolute_path = Path(absolute)
    if not absolute_path.is_absolute():
        raise RuntimeError(f"source inventory path is not absolute: {absolute}")
    matches = []
    for relative in expected_relative_paths:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or ".." in relative_path.parts
        ):
            raise RuntimeError(f"unsafe expected source path: {relative}")
        if (
            len(absolute_path.parts) >= len(relative_path.parts)
            and absolute_path.parts[-len(relative_path.parts) :] == relative_path.parts
        ):
            matches.append(relative)
    if len(matches) != 1:
        raise RuntimeError(
            f"source inventory path does not identify exactly one expected source: "
            f"{absolute}; matches={sorted(matches)}"
        )
    return matches[0]


def _verify_source_inventory(
    inventory: dict[str, Any],
    repo: Path,
    commit: str,
    expected_relative_paths: set[str],
) -> None:
    if not isinstance(inventory, dict) or not inventory:
        raise RuntimeError("missing source inventory")
    relative_inventory = {}
    for absolute, expected in inventory.items():
        relative = _relative_source_path(str(absolute), expected_relative_paths)
        if relative in relative_inventory:
            raise RuntimeError(f"duplicate source inventory path: {relative}")
        relative_inventory[relative] = expected
    if set(relative_inventory) != expected_relative_paths:
        missing = sorted(expected_relative_paths - set(relative_inventory))
        extra = sorted(set(relative_inventory) - expected_relative_paths)
        raise RuntimeError(
            f"source inventory path-set mismatch: missing={missing}, extra={extra}"
        )
    for relative, expected in relative_inventory.items():
        observed = _sha256_bytes(_git_blob(repo, commit, relative))
        if observed != expected:
            raise RuntimeError(f"source inventory mismatch: {relative}")


def _verify_frozen_joined_artifacts(
    root: Path, expected_sha256: dict[str, str]
) -> None:
    joined = root / "joined"
    if set(expected_sha256) != {
        "COMPLETE.json",
        "qualification_raw.npz",
        "qualification_summary.json",
    }:
        raise RuntimeError("invalid frozen joined-artifact hash set")
    for name, expected in expected_sha256.items():
        path = joined / name
        if not path.is_file():
            raise RuntimeError(f"missing frozen joined artifact: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen joined artifact hash mismatch: {name}: "
                f"{observed} != {expected}"
            )


def _verify_artifact_inventory(
    directory: Path, complete: dict[str, Any], expected_names: set[str]
) -> None:
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise RuntimeError(f"artifact inventory mismatch: {directory}")
    for name, record in artifacts.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"missing artifact: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"artifact size mismatch: {path}")
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {path}")


def _load_raw(path: Path, protocol_version: int) -> dict[str, np.ndarray]:
    expected_raw = _expected_raw(protocol_version)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_raw):
            raise RuntimeError(f"raw inventory mismatch: {path}")
        raw = {name: archive[name].copy() for name in archive.files}
    for name, (shape, dtype) in expected_raw.items():
        if raw[name].shape != shape or raw[name].dtype != dtype:
            raise RuntimeError(f"raw shape/dtype mismatch: {path}:{name}")
    return raw


def _js_rows(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    middle = 0.5 * (candidate + reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(
            candidate > 0.0,
            candidate * (np.log(candidate) - np.log(middle)),
            0.0,
        )
        right = np.where(
            reference > 0.0,
            reference * (np.log(reference) - np.log(middle)),
            0.0,
        )
    return 0.5 * (left.sum(axis=-1) + right.sum(axis=-1))


def _observed_logp_rows(
    candidate: np.ndarray, reference: np.ndarray, outcome_bins: np.ndarray
) -> np.ndarray:
    rows = np.arange(candidate.shape[0], dtype=np.int64)
    candidate_observed = candidate[rows, outcome_bins]
    reference_observed = reference[rows, outcome_bins]
    if np.any(candidate_observed <= 0.0) or np.any(reference_observed <= 0.0):
        raise RuntimeError("observed-bin probabilities must be positive")
    return np.abs(np.log(candidate_observed) - np.log(reference_observed))


def _metric_row(js: np.ndarray, logp: np.ndarray) -> dict[str, Any]:
    row = {
        "median_js": float(np.median(js)),
        "p95_js": float(np.quantile(js, 0.95)),
        "median_abs_logp_change": float(np.median(logp)),
        "p95_abs_logp_change": float(np.quantile(logp, 0.95)),
    }
    row["pass"] = bool(all(row[name] <= limit for name, limit in THRESHOLDS.items()))
    row["borderline"] = bool(
        any(
            THRESHOLDS[name] < row[name] <= 1.1 * BASE_THRESHOLDS[name]
            for name in THRESHOLDS
        )
    )
    return row


def _quadrature_report(raw: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    limits = config["quadrature_qualification"]
    prior_rows: dict[str, Any] = {}
    pass_all = True
    for prior_index, prior in enumerate(PRIORS):
        predictor_rows: dict[str, Any] = {}
        for predictor in PREDICTORS:
            max_js = float(np.max(raw[f"quadrature_js_{predictor}"][prior_index]))
            max_bin_logp = float(
                np.max(
                    raw[f"quadrature_max_bin_abs_logp_change_{predictor}"][
                        prior_index
                    ]
                )
            )
            max_weighted_logp = float(
                np.max(
                    raw[
                        "quadrature_reference_weighted_abs_logp_change_"
                        f"{predictor}"
                    ][prior_index]
                )
            )
            passed = bool(
                max_js <= float(limits["js_max"])
                and max_bin_logp
                <= float(limits["max_bin_abs_logp_change_max"])
                and max_weighted_logp
                <= float(limits["reference_weighted_abs_logp_change_max"])
            )
            predictor_rows[predictor] = {
                "max_js": max_js,
                "max_bin_abs_logp_change": max_bin_logp,
                "max_reference_weighted_abs_logp_change": max_weighted_logp,
                "pass": passed,
            }
            pass_all = pass_all and passed
        max_value_change = float(
            np.max(
                raw["quadrature_max_bin_abs_ordering_value_change"][prior_index]
            )
        )
        value_pass = bool(
            max_value_change
            <= float(limits["max_bin_abs_ordering_value_change_max"])
        )
        pass_all = pass_all and value_pass
        prior_rows[prior] = {
            "predictors": predictor_rows,
            "max_bin_abs_ordering_value_change": max_value_change,
            "ordering_value_pass": value_pass,
            "pass": bool(value_pass and all(row["pass"] for row in predictor_rows.values())),
        }
    return {
        "production_grid": {
            "interior_nodes": int(config["quadrature_interior_nodes"]),
            "tail_nodes": int(config["quadrature_tail_nodes"]),
        },
        "reference_grid": {
            "interior_nodes": int(config["quadrature_reference_interior_nodes"]),
            "tail_nodes": int(config["quadrature_reference_tail_nodes"]),
        },
        "limits": limits,
        "priors": prior_rows,
        "pass": pass_all,
    }


def _verify_shard(
    run_dir: Path,
    bank: int,
    repo: Path,
    commit: str,
    protocol_version: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    complete = _load_json(run_dir / "COMPLETE.json")
    _verify_artifact_inventory(run_dir, complete, EXPECTED_SHARD_ARTIFACTS)
    running = _load_json(run_dir / "RUNNING.json")
    summary = _load_json(run_dir / "calibration_summary.json")
    identity = complete.get("identity")
    if not isinstance(identity, dict) or running.get("identity") != identity:
        raise RuntimeError(f"identity object mismatch: bank {bank}")
    digest = _sha256_json(identity)
    if (
        complete.get("identity_sha256") != digest
        or running.get("identity_sha256") != digest
        or summary.get("attempt_identity_sha256") != digest
    ):
        raise RuntimeError(f"identity digest mismatch: bank {bank}")
    if identity.get("git_commit") != commit:
        raise RuntimeError(f"identity commit mismatch: bank {bank}")
    shard_sources = {
        "src/pfn_dag_verify/phase1_ordering.py",
        "src/pfn_dag_verify/storage.py",
        "PHASE1_ORDERING_PREREG.md",
        f"config/phase1_ordering_qualification{('_v' + str(protocol_version)) if protocol_version >= 2 else ''}_bank{bank}.json",
        "artifacts/phase1/d4_generator.py",
        "environment/phase1-washu-runtime.json",
        "environment/phase1-washu-requirements-lock.txt",
        "cluster/phase1_calibration.sbatch",
        "environment/phase1-washu-binary-inventory.json",
    }
    if protocol_version >= 2:
        shard_sources.add(
            f"PHASE1_ORDERING_QUALIFICATION_V{protocol_version}_PREREG.md"
        )
    _verify_source_inventory(
        identity.get("source_inventory"), repo, commit, shard_sources
    )
    if summary.get("git") != {"commit": commit, "dirty": False, "status": []}:
        raise RuntimeError(f"summary Git provenance mismatch: bank {bank}")
    expected_stage = "phase1_ordering_cross_bank_qualification"
    if protocol_version >= 2:
        expected_stage += f"_v{protocol_version}"
    if summary.get("stage") != expected_stage:
        raise RuntimeError(f"stage mismatch: bank {bank}")
    if summary.get("scientific_endpoints_computed") is not False:
        raise RuntimeError(f"scientific endpoint barrier violated: bank {bank}")
    if summary.get("completion_state") != "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER":
        raise RuntimeError(f"completion-state mismatch: bank {bank}")
    if complete.get("decision") != summary.get("decision"):
        raise RuntimeError(f"decision marker mismatch: bank {bank}")

    config_suffix = "" if protocol_version == 1 else f"_v{protocol_version}"
    config_relative = (
        f"config/phase1_ordering_qualification{config_suffix}_bank{bank}.json"
    )
    config_blob = _git_blob(repo, commit, config_relative)
    config = json.loads(config_blob)
    if summary.get("config_sha256") != _sha256_bytes(config_blob):
        raise RuntimeError(f"config hash mismatch: bank {bank}")
    expected_config = {
        "qualification_bank_index": bank,
        "calibration_contexts_per_prior": 160,
        "calibration_seed_root": {
            1: 880_903_000 + bank,
            2: 880_923_000 + bank,
            3: 880_943_000 + bank,
        }[protocol_version],
        "atom_count": 3_000_000,
        "atom_seed": 881_003_101 + bank,
        "atom_determinism_canary_seed": 881_103_999,
        "atom_determinism_canary_count": 4096,
        "truncation_candidates": list(CANDIDATES),
        "reference_truncation": REFERENCE,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise RuntimeError(f"frozen config field mismatch: bank {bank}")
    if protocol_version >= 2:
        expected_versioned = {
            "qualification_protocol_version": protocol_version,
            "archive_predictive_arrays": True,
            "ablation_weighting": (
                "uniform-over-orderings-after-within-order-topk-normalization-v1"
            ),
            "qualification_preregistration": (
                f"PHASE1_ORDERING_QUALIFICATION_V{protocol_version}_PREREG.md"
            ),
        }
        if protocol_version == 3:
            expected_versioned.update(
                {
                    "quadrature_interior_nodes": 32,
                    "quadrature_tail_nodes": 128,
                    "quadrature_reference_interior_nodes": 64,
                    "quadrature_reference_tail_nodes": 256,
                    "quadrature_qualification": {
                        "js_max": 1e-7,
                        "reference_probability_floor": 1e-8,
                        "max_bin_abs_logp_change_max": 5e-4,
                        "reference_weighted_abs_logp_change_max": 1e-4,
                        "max_bin_abs_ordering_value_change_max": 5e-4,
                    },
                    "required_source_tag": "phase1-ordering-qualification-v3",
                    "oracle_internal_dtype": "float64",
                }
            )
        if any(config.get(name) != value for name, value in expected_versioned.items()):
            raise RuntimeError(
                f"frozen v{protocol_version} config field mismatch: bank {bank}"
            )
    if protocol_version == 3 and summary.get("oracle_internal_dtype") != "float64":
        raise RuntimeError(f"qualification v3 oracle dtype mismatch: bank {bank}")
    limits = config.get("qualification_zero_exceedance")
    if limits != {
        "familywise_alpha": 0.05,
        "families": 48,
        "contexts_per_family": 160,
        "js_strict_p95_max": 0.0009,
        "abs_logp_change_strict_p95_max": 0.009,
        "bonferroni_clopper_pearson_upper": 0.04201037701571053,
    }:
        raise RuntimeError(f"qualification limits mismatch: bank {bank}")

    raw_path = run_dir / "calibration_raw.npz"
    if summary.get("raw_sha256") != _sha256_file(raw_path):
        raise RuntimeError(f"raw summary hash mismatch: bank {bank}")
    raw = _load_raw(raw_path, protocol_version)
    if raw["prior_codes"].tolist() != [0, 1]:
        raise RuntimeError(f"prior codes mismatch: bank {bank}")
    if raw["candidates"].tolist() != list(CANDIDATES):
        raise RuntimeError(f"candidates mismatch: bank {bank}")
    if raw["reference_truncation"].tolist() != [REFERENCE]:
        raise RuntimeError(f"reference mismatch: bank {bank}")
    if not np.all(raw["completed"] == 1):
        raise RuntimeError(f"incomplete raw contexts: bank {bank}")
    identity_bytes = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)
    if not np.all(raw["attempt_identity_sha256"] == identity_bytes[None, :]):
        raise RuntimeError(f"raw identity mismatch: bank {bank}")
    numeric = [name for name in raw if name != "attempt_identity_sha256"]
    if not all(np.isfinite(raw[name]).all() for name in numeric):
        raise RuntimeError(f"non-finite raw diagnostics: bank {bank}")
    if max(
        float(raw["full_probability_sum_error"].max()),
        float(raw["ablated_probability_sum_error"].max()),
    ) > 1e-8:
        raise RuntimeError(f"normalization failure: bank {bank}")
    if min(
        float(raw["full_probability_minimum"].min()),
        float(raw["ablated_probability_minimum"].min()),
    ) < 0.0:
        raise RuntimeError(f"negative probability: bank {bank}")
    if protocol_version >= 2:
        outcome_bins = raw["outcome_bins"]
        if np.any(outcome_bins < 0) or np.any(outcome_bins >= 100):
            raise RuntimeError(f"outcome bin out of range: bank {bank}")
        for predictor in PREDICTORS:
            probabilities = raw[f"{predictor}_probability"]
            if np.any(probabilities < 0.0) or not np.allclose(
                probabilities.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0
            ):
                raise RuntimeError(f"archived probability invalid: bank {bank} {predictor}")
            for prior_index in range(2):
                reference_probability = probabilities[prior_index, -1]
                for candidate_index in range(2):
                    candidate_probability = probabilities[prior_index, candidate_index]
                    recomputed_js = _js_rows(
                        candidate_probability, reference_probability
                    )
                    recomputed_logp = _observed_logp_rows(
                        candidate_probability,
                        reference_probability,
                        outcome_bins[prior_index],
                    )
                    if not np.allclose(
                        recomputed_js,
                        raw[f"js_{predictor}"][prior_index, candidate_index],
                        atol=2e-15,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"independent JS recomputation mismatch: bank {bank} {predictor}"
                        )
                    if not np.allclose(
                        recomputed_logp,
                        raw[f"abs_logp_change_{predictor}"][
                            prior_index, candidate_index
                        ],
                        atol=2e-15,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"independent logp recomputation mismatch: bank {bank} {predictor}"
                        )
    if protocol_version == 3:
        if raw["quadrature_grid_interior_nodes"].tolist() != [32, 64]:
            raise RuntimeError(f"quadrature interior-grid mismatch: bank {bank}")
        if raw["quadrature_grid_tail_nodes"].tolist() != [128, 256]:
            raise RuntimeError(f"quadrature tail-grid mismatch: bank {bank}")
        if raw["quadrature_truncation_levels"].tolist() != [8192, 16384, 32768]:
            raise RuntimeError(f"quadrature truncation-axis mismatch: bank {bank}")
        probability_floor = float(
            config["quadrature_qualification"]["reference_probability_floor"]
        )
        for predictor in PREDICTORS:
            grid = raw[f"quadrature_grid_{predictor}_probability"]
            if np.any(grid <= 0.0) or not np.allclose(
                grid.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0
            ):
                raise RuntimeError(
                    f"quadrature-grid probability invalid: bank {bank} {predictor}"
                )
            if not np.array_equal(grid[:, 0], raw[f"{predictor}_probability"]):
                raise RuntimeError(
                    f"quadrature production-axis mismatch: bank {bank} {predictor}"
                )
            for prior_index in range(2):
                for level_index in range(3):
                    production_probability = grid[prior_index, 0, level_index]
                    reference_probability = grid[prior_index, 1, level_index]
                    quadrature_js = _js_rows(
                        production_probability, reference_probability
                    )
                    log_change = np.abs(
                        np.log(production_probability)
                        - np.log(reference_probability)
                    )
                    active = reference_probability >= probability_floor
                    if not np.all(np.any(active, axis=1)):
                        raise RuntimeError(
                            f"quadrature floor masks every bin: bank {bank} {predictor}"
                        )
                    max_log_change = np.max(
                        np.where(active, log_change, -np.inf), axis=1
                    )
                    weighted_log_change = np.sum(
                        reference_probability * log_change, axis=1
                    )
                    if not np.allclose(
                        quadrature_js,
                        raw[f"quadrature_js_{predictor}"][
                            prior_index, level_index
                        ],
                        atol=2e-15,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"independent quadrature JS mismatch: bank {bank} {predictor}"
                        )
                    if not np.allclose(
                        max_log_change,
                        raw[f"quadrature_max_bin_abs_logp_change_{predictor}"][
                            prior_index, level_index
                        ],
                        atol=2e-15,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"independent quadrature max-logp mismatch: "
                            f"bank {bank} {predictor}"
                        )
                    if not np.allclose(
                        weighted_log_change,
                        raw[
                            "quadrature_reference_weighted_abs_logp_change_"
                            f"{predictor}"
                        ][prior_index, level_index],
                        atol=2e-15,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"independent quadrature weighted-logp mismatch: "
                            f"bank {bank} {predictor}"
                        )
        full_grid = raw["quadrature_grid_full_probability"]
        ablated_grid = raw["quadrature_grid_ablated_probability"]
        production_value = np.log(full_grid[:, 0]) - np.log(ablated_grid[:, 0])
        reference_value = np.log(full_grid[:, 1]) - np.log(ablated_grid[:, 1])
        value_active = (full_grid[:, 1] >= probability_floor) & (
            ablated_grid[:, 1] >= probability_floor
        )
        if not np.all(np.any(value_active, axis=-1)):
            raise RuntimeError(f"quadrature value floor masks every bin: bank {bank}")
        value_change = np.max(
            np.where(
                value_active,
                np.abs(production_value - reference_value),
                -np.inf,
            ),
            axis=-1,
        )
        if not np.allclose(
            value_change,
            raw["quadrature_max_bin_abs_ordering_value_change"],
            atol=2e-15,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"independent quadrature ordering-value mismatch: bank {bank}"
            )
        if summary.get("quadrature_qualification") != _quadrature_report(raw, config):
            raise RuntimeError(f"quadrature summary mismatch: bank {bank}")

    metadata_names = {"prior_codes", "candidates", "reference_truncation"}
    if protocol_version == 3:
        metadata_names.update(
            {
                "quadrature_grid_interior_nodes",
                "quadrature_grid_tail_nodes",
                "quadrature_truncation_levels",
            }
        )
    partial_names = set(raw) - metadata_names
    for prior_index, prior in enumerate(PRIORS):
        with np.load(run_dir / f"partial_{prior}.npz", allow_pickle=False) as partial:
            if set(partial.files) != partial_names:
                raise RuntimeError(f"partial inventory mismatch: bank {bank} {prior}")
            for name in partial.files:
                if not np.array_equal(partial[name], raw[name][prior_index]):
                    raise RuntimeError(f"partial/raw mismatch: bank {bank} {prior} {name}")

    atom_bank = _load_json(run_dir / "ATOM_BANK.json")
    if atom_bank != summary.get("atom_bank"):
        raise RuntimeError(f"atom marker/summary mismatch: bank {bank}")
    if (
        atom_bank.get("seed") != 881_003_101 + bank
        or atom_bank.get("count") != 3_000_000
        or atom_bank.get("shape") != [3_000_000, 4, 4]
        or atom_bank.get("dtype") != np.dtype(np.float64).str
    ):
        raise RuntimeError(f"atom metadata mismatch: bank {bank}")
    canary = atom_bank.get("determinism_canary", {})
    if (
        canary.get("seed") != 881_103_999
        or canary.get("count") != 4096
        or canary.get("shape") != [4096, 4, 4]
        or canary.get("dtype") != np.dtype(np.float64).str
    ):
        raise RuntimeError(f"canary metadata mismatch: bank {bank}")
    for label, value in (("atom", atom_bank.get("sha256")), ("canary", canary.get("sha256"))):
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"invalid {label} hash: bank {bank}")
        bytes.fromhex(value)
    return summary, raw, config


def verify_qualification(
    root: Path,
    repo: Path,
    commit: str = COMMIT,
    protocol_version: int = 1,
    verifier_commit: str | None = None,
    verifier_tag: str | None = None,
) -> dict[str, Any]:
    if protocol_version == 2:
        raise RuntimeError(
            "qualification v2 is blocked-before-execution and superseded by v3"
        )
    root = root.resolve()
    repo = repo.resolve()
    resolved_verifier_commit = verifier_commit or commit
    resolved_verifier_tag: str | None = None
    if protocol_version >= 2:
        verifier_relative = "src/pfn_dag_verify/phase1_qualification_verify.py"
        committed_verifier_sha256 = _sha256_bytes(
            _git_blob(repo, resolved_verifier_commit, verifier_relative)
        )
        if _sha256_file(Path(__file__).resolve()) != committed_verifier_sha256:
            raise RuntimeError(
                "independent verifier source differs from verifier commit"
            )
    if protocol_version == 3:
        _verify_annotated_tag(repo, V3_SOURCE_TAG, commit)
        if resolved_verifier_commit == commit:
            if verifier_tag not in {None, V3_SOURCE_TAG}:
                raise RuntimeError("same-commit verifier tag mismatch")
            resolved_verifier_tag = V3_SOURCE_TAG
        else:
            if verifier_tag != V3_VERIFIER_FIX_TAG:
                raise RuntimeError("qualification v3 verifier-fix tag mismatch")
            _verify_annotated_tag(repo, V3_VERIFIER_FIX_TAG, resolved_verifier_commit)
            resolved_verifier_tag = V3_VERIFIER_FIX_TAG
            _verify_frozen_joined_artifacts(root, V3_VERIFIER_FIX1_JOINED_SHA256)
    verified = [
        _verify_shard(
            root / f"bank{bank}" / "run",
            bank,
            repo,
            commit,
            protocol_version,
        )
        for bank in range(3)
    ]
    atom_hashes = [summary["atom_bank"]["sha256"] for summary, _, _ in verified]
    canary_hashes = {
        summary["atom_bank"]["determinism_canary"]["sha256"]
        for summary, _, _ in verified
    }
    if len(set(atom_hashes)) != 3 or len(canary_hashes) != 1:
        raise RuntimeError("atom-bank distinctness or determinism canary failed")

    reports: list[dict[str, Any]] = []
    selected: int | None = None
    for candidate_index, candidate in enumerate(CANDIDATES):
        candidate_pass = True
        bank_reports: list[dict[str, Any]] = []
        for bank, (_, raw, _) in enumerate(verified):
            prior_reports: dict[str, Any] = {}
            for prior_index, prior in enumerate(PRIORS):
                predictor_reports: dict[str, Any] = {}
                for predictor in PREDICTORS:
                    js = raw[f"js_{predictor}"][prior_index, candidate_index]
                    logp = raw[f"abs_logp_change_{predictor}"][prior_index, candidate_index]
                    aggregate = _metric_row(js, logp)
                    js_exceedances = int(np.count_nonzero(js > 0.0009))
                    logp_exceedances = int(np.count_nonzero(logp > 0.009))
                    passed = bool(
                        aggregate["pass"] and js_exceedances == 0 and logp_exceedances == 0
                    )
                    candidate_pass = candidate_pass and passed
                    predictor_reports[predictor] = {
                        "aggregate": aggregate,
                        "js_exceedances": js_exceedances,
                        "abs_logp_change_exceedances": logp_exceedances,
                        "pass": passed,
                    }
                prior_reports[prior] = predictor_reports
            bank_reports.append({"bank_index": bank, "priors": prior_reports})
        reports.append(
            {
                "candidate": candidate,
                "compared_with": REFERENCE,
                "banks": bank_reports,
                "pass_all_banks": candidate_pass,
            }
        )
        if selected is None and candidate_pass:
            selected = candidate

    quadrature_reports: list[dict[str, Any]] | None = None
    if protocol_version == 3:
        quadrature_reports = []
        quadrature_pass_all = True
        for bank, (_, raw, config) in enumerate(verified):
            report = _quadrature_report(raw, config)
            quadrature_pass_all = quadrature_pass_all and bool(report["pass"])
            quadrature_reports.append({"bank_index": bank, **report})
        if not quadrature_pass_all:
            selected = None

    joined = root / "joined"
    joined_complete = _load_json(joined / "COMPLETE.json")
    _verify_artifact_inventory(
        joined,
        joined_complete,
        {"qualification_raw.npz", "qualification_summary.json"},
    )
    joined_summary = _load_json(joined / "qualification_summary.json")
    expected_joined_stage = "phase1_ordering_cross_bank_qualification"
    if protocol_version >= 2:
        expected_joined_stage += f"_v{protocol_version}"
    expected_joined_stage += "_join"
    protocol_field_valid = (
        joined_summary.get("qualification_protocol_version") == protocol_version
        if protocol_version >= 2
        else joined_summary.get("qualification_protocol_version", 1) == 1
    )
    if (
        joined_summary.get("schema_version") != protocol_version
        or not protocol_field_valid
        or joined_summary.get("stage") != expected_joined_stage
    ):
        observed = {
            "schema_version": joined_summary.get("schema_version"),
            "qualification_protocol_version": joined_summary.get(
                "qualification_protocol_version"
            ),
            "stage": joined_summary.get("stage"),
        }
        raise RuntimeError(f"joined protocol identity mismatch: {observed}")
    identity = joined_complete.get("identity")
    if not isinstance(identity, dict) or joined_summary.get("identity") != identity:
        raise RuntimeError("joined identity object mismatch")
    digest = _sha256_json(identity)
    if (
        joined_complete.get("identity_sha256") != digest
        or joined_summary.get("identity_sha256") != digest
    ):
        raise RuntimeError("joined identity digest mismatch")
    if identity.get("git_commit") != commit or joined_summary.get("git", {}).get("commit") != commit:
        raise RuntimeError("joined commit mismatch")
    joined_sources = {
        "src/pfn_dag_verify/phase1_qualification.py",
        "src/pfn_dag_verify/phase1_ordering.py",
        "src/pfn_dag_verify/storage.py",
        "PHASE1_ORDERING_PREREG.md",
        *{
            f"config/phase1_ordering_qualification{('_v' + str(protocol_version)) if protocol_version >= 2 else ''}_bank{bank}.json"
            for bank in range(3)
        },
    }
    if protocol_version >= 2:
        joined_sources.update(
            {
                "src/pfn_dag_verify/phase1_qualification_verify.py",
                f"PHASE1_ORDERING_QUALIFICATION_V{protocol_version}_PREREG.md",
            }
        )
    _verify_source_inventory(
        identity.get("source_inventory"), repo, commit, joined_sources
    )
    for bank in range(3):
        suffix = f"/bank{bank}/run"
        entries = [
            (path, value)
            for path, value in identity.get("run_marker_hashes", {}).items()
            if path.endswith(suffix)
        ]
        if len(entries) != 1:
            raise RuntimeError(f"joined run-marker identity mismatch: bank {bank}")
        if entries[0][1] != _sha256_file(root / f"bank{bank}" / "run" / "COMPLETE.json"):
            raise RuntimeError(f"joined run-marker hash mismatch: bank {bank}")
    if identity.get("atom_bank_sha256") != atom_hashes:
        raise RuntimeError("joined atom-bank hashes mismatch")
    if identity.get("atom_determinism_canary_sha256") != next(iter(canary_hashes)):
        raise RuntimeError("joined canary hash mismatch")
    expected_decision = "QUALIFICATION_PASS" if selected is not None else "QUALIFICATION_FAIL"
    if (
        joined_summary.get("decision") != expected_decision
        or joined_complete.get("decision") != expected_decision
        or joined_summary.get("selected_truncation") != selected
        or joined_summary.get("candidate_reports") != reports
        or joined_summary.get("quadrature_reports") != quadrature_reports
    ):
        raise RuntimeError("joined scientific decision does not reproduce")
    cp_upper = 1.0 - (0.05 / 48.0) ** (1.0 / 160.0)
    if not math.isclose(
        joined_summary.get("zero_exceedance_clopper_pearson_upper"),
        cp_upper,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("joined Clopper-Pearson value mismatch")
    with np.load(joined / "qualification_raw.npz", allow_pickle=False) as archive:
        expected_joined_names = {
            "bank_indices",
            "candidates",
            "js_full",
            "js_ablated",
            "abs_logp_change_full",
            "abs_logp_change_ablated",
        }
        if protocol_version >= 2:
            expected_joined_names.update(
                {
                    "outcome_bins",
                    "full_probability",
                    "ablated_probability",
                }
            )
        quadrature_metadata: set[str] = set()
        if protocol_version == 3:
            quadrature_metadata = {
                "quadrature_grid_interior_nodes",
                "quadrature_grid_tail_nodes",
                "quadrature_truncation_levels",
            }
            expected_joined_names.update(
                quadrature_metadata
                | {
                    "quadrature_grid_full_probability",
                    "quadrature_grid_ablated_probability",
                    "quadrature_js_full",
                    "quadrature_js_ablated",
                    "quadrature_max_bin_abs_logp_change_full",
                    "quadrature_max_bin_abs_logp_change_ablated",
                    "quadrature_reference_weighted_abs_logp_change_full",
                    "quadrature_reference_weighted_abs_logp_change_ablated",
                    "quadrature_max_bin_abs_ordering_value_change",
                }
            )
        if set(archive.files) != expected_joined_names:
            raise RuntimeError("joined raw inventory mismatch")
        if archive["bank_indices"].tolist() != [0, 1, 2]:
            raise RuntimeError("joined bank indices mismatch")
        if archive["candidates"].tolist() != list(CANDIDATES):
            raise RuntimeError("joined candidates mismatch")
        for name in quadrature_metadata:
            if not np.array_equal(archive[name], verified[0][1][name]):
                raise RuntimeError(f"joined quadrature metadata mismatch: {name}")
        for name in expected_joined_names - {
            "bank_indices",
            "candidates",
            *quadrature_metadata,
        }:
            expected = np.stack([raw[name] for _, raw, _ in verified])
            if not np.array_equal(archive[name], expected):
                raise RuntimeError(f"joined raw stack mismatch: {name}")

    worst: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        cells = [
            cell
            for bank in reports[candidate_index]["banks"]
            for prior in bank["priors"].values()
            for cell in prior.values()
        ]
        worst[str(candidate)] = {
            "total_js_exceedances": sum(cell["js_exceedances"] for cell in cells),
            "total_abs_logp_change_exceedances": sum(
                cell["abs_logp_change_exceedances"] for cell in cells
            ),
            "worst_p95_js": max(cell["aggregate"]["p95_js"] for cell in cells),
            "worst_p95_abs_logp_change": max(
                cell["aggregate"]["p95_abs_logp_change"] for cell in cells
            ),
        }
    return {
        "schema_version": protocol_version,
        "qualification_protocol_version": protocol_version,
        "verification": "INDEPENDENT_RAW_RECOMPUTATION_PASS",
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verifier_runtime": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_commit": commit,
        "source_tag": V3_SOURCE_TAG if protocol_version == 3 else None,
        "verifier_commit": resolved_verifier_commit,
        "verifier_source_tag": resolved_verifier_tag,
        "decision": expected_decision,
        "selected_truncation": selected,
        "reference_truncation": REFERENCE,
        "bank_atom_sha256": atom_hashes,
        "determinism_canary_sha256": next(iter(canary_hashes)),
        "zero_exceedance_clopper_pearson_upper": cp_upper,
        "candidate_summary": worst,
        "joined_complete_sha256": _sha256_file(joined / "COMPLETE.json"),
        "joined_raw_sha256": _sha256_file(joined / "qualification_raw.npz"),
        "joined_summary_sha256": _sha256_file(joined / "qualification_summary.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=COMMIT)
    parser.add_argument("--verifier-commit")
    parser.add_argument("--verifier-tag")
    parser.add_argument("--protocol-version", type=int, choices=(1, 2, 3), default=1)
    arguments = parser.parse_args(argv)
    if arguments.protocol_version >= 2 and arguments.commit == COMMIT:
        raise RuntimeError(
            f"qualification v{arguments.protocol_version} requires its explicit source commit"
        )
    if arguments.protocol_version >= 2 and arguments.verifier_commit is None:
        raise RuntimeError(
            f"qualification v{arguments.protocol_version} requires its explicit verifier commit"
        )
    result = verify_qualification(
        arguments.root,
        arguments.repo,
        arguments.commit,
        protocol_version=arguments.protocol_version,
        verifier_commit=arguments.verifier_commit,
        verifier_tag=arguments.verifier_tag,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.out.with_suffix(arguments.out.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(arguments.out)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
