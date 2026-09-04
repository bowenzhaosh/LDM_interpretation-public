"""Fail-closed join and decision for the Phase-1 ordering-use confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch

from .phase1_confirm_common import (
    acquire_empty_output,
    attempt_identity,
    expected_attempt_identity,
    load_json,
    repo_root,
    sha256_file,
)
from .phase1_oracle_confirm import ORACLE_ARRAYS
from .phase1_ordering import load_fleet_module
from .phase1_panel import (
    INPUT_ARRAYS,
    LABEL_ARRAYS,
    ROW_KEYS,
    _covariance_symmetry_diagnostics,
    _digest_rows,
)
from .phase1_pfn import (
    PREDICTION_ARRAYS,
    REPLAY_BATCH_COMPARISONS,
    REPLAY_COMPARISONS,
    REPLAY_COMBINED_COMPARISONS,
    REPLAY_CONTEXT_COMPARISONS,
    REPLAY_EXACT_COMPARISONS,
)
from .storage import write_json_atomic, write_numeric_npz_atomic


MODEL_SEEDS = np.array([0, 1, 2], dtype=np.int64)
CHECKPOINT_STEPS = np.array([20_000, 60_000, 120_000], dtype=np.int64)


def _validate_panel_covariances(fleet: Any, sigmas: np.ndarray, name: str) -> None:
    _covariance_symmetry_diagnostics(sigmas)
    if not np.all(fleet.validity_keep(sigmas)):
        raise RuntimeError(f"panel covariance validity failure: {name}")


def _load_npz(path: Path, schema: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != schema:
            raise RuntimeError(f"NPZ schema mismatch: {path}")
        return {name: archive[name].copy() for name in archive.files}


def _verify_artifacts(directory: Path, artifacts: Any, expected: set[str]) -> None:
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise RuntimeError(f"artifact inventory mismatch: {directory}")
    for relative, record in artifacts.items():
        path = directory / relative
        if not path.is_file():
            raise RuntimeError(f"missing artifact: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"artifact byte-count mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {path}")


def _assert_identity(
    value: dict[str, Any], identity: dict[str, Any], identity_sha256: str, label: str
) -> None:
    if (
        value.get("identity") != identity
        or value.get("identity_sha256") != identity_sha256
    ):
        raise RuntimeError(f"attempt identity mismatch: {label}")


def _verify_panel(
    panel_dir: Path, identity: dict[str, Any], identity_sha256: str
) -> None:
    complete = load_json(panel_dir / "PANEL_COMPLETE.json")
    _assert_identity(complete, identity, identity_sha256, "panel")
    if complete.get("decision") != "PANEL_COMPLETE":
        raise RuntimeError("panel completion decision mismatch")
    expected = {"RUNNING.json", "INPUTS_COMPLETE.json", "panel_manifest.json"}
    expected |= {
        f"{kind}/{prior}_d{draw}_b{bank}.npz"
        for kind in ("inputs", "labels")
        for prior in ("C", "N")
        for draw in range(3)
        for bank in range(3)
    }
    _verify_artifacts(panel_dir, complete.get("artifacts"), expected)
    input_marker = load_json(panel_dir / "INPUTS_COMPLETE.json")
    _assert_identity(input_marker, identity, identity_sha256, "panel inputs")
    if input_marker.get("decision") != "PANEL_INPUTS_COMPLETE":
        raise RuntimeError("panel-input completion decision mismatch")
    _verify_artifacts(
        panel_dir / "inputs",
        input_marker.get("artifacts"),
        {
            f"{prior}_d{draw}_b{bank}.npz"
            for prior in ("C", "N")
            for draw in range(3)
            for bank in range(3)
        },
    )


def _verify_pfn_root(
    pfn_dir: Path,
    identity: dict[str, Any],
    identity_sha256: str,
    config: dict[str, Any],
) -> None:
    complete = load_json(pfn_dir / "COMPLETE.json")
    _assert_identity(complete, identity, identity_sha256, "PFN fleet")
    if complete.get("decision") != "PFN_FLEET_COMPLETE":
        raise RuntimeError("PFN fleet completion decision mismatch")
    _verify_artifacts(
        pfn_dir,
        complete.get("artifacts"),
        {"RUNNING.json", "pfn_fleet_summary.json"},
    )
    expected_markers = {
        f"{prior}_s{seed}_t{step}/d{draw}_b{bank}/COMPLETE.json"
        for prior in ("C", "N")
        for seed in MODEL_SEEDS.tolist()
        for step in CHECKPOINT_STEPS.tolist()
        for draw in range(3)
        for bank in range(3)
    }
    markers = complete.get("leaf_marker_hashes")
    if not isinstance(markers, dict) or set(markers) != expected_markers:
        raise RuntimeError("PFN leaf-marker inventory mismatch")
    discovered_markers = {
        path.relative_to(pfn_dir).as_posix()
        for path in pfn_dir.rglob("COMPLETE.json")
        if path != pfn_dir / "COMPLETE.json"
    }
    if discovered_markers != expected_markers:
        raise RuntimeError("unexpected or missing PFN completion shard")
    for relative, expected_hash in markers.items():
        if sha256_file(pfn_dir / relative) != expected_hash:
            raise RuntimeError(f"PFN leaf-marker hash mismatch: {relative}")
    summary = load_json(pfn_dir / "pfn_fleet_summary.json")
    _assert_identity(summary, identity, identity_sha256, "PFN summary")
    if summary.get("prediction_shards") != 162 or summary.get("checkpoint_count") != 18:
        raise RuntimeError("PFN fleet count mismatch")
    replay = summary.get("replay")
    expected_replay = {
        f"{prior}_s{seed}_t{step}"
        for prior in ("C", "N")
        for seed in MODEL_SEEDS.tolist()
        for step in CHECKPOINT_STEPS.tolist()
    }
    if not isinstance(replay, dict) or set(replay) != expected_replay:
        raise RuntimeError("PFN replay inventory mismatch")
    metric_names = {
        "bit_identical",
        "max_abs_logp_error",
        "max_abs_probability_error",
        "max_total_variation",
    }
    for row in replay.values():
        if set(row) != {
            "stress_rows",
            "full_panel_rows",
            "full_panel_shards",
            "comparisons",
            "exact_controls_pass",
            "batch_max_abs_logp_error",
            "context_max_abs_logp_error",
            "combined_max_abs_logp_error",
            "approximate_max_abs_probability_error",
            "approximate_max_total_variation",
            "pass",
        }:
            raise RuntimeError("PFN replay record schema mismatch")
        if (
            row.get("pass") is not True
            or row.get("exact_controls_pass") is not True
            or row.get("stress_rows") != 72
            or row.get("full_panel_rows") != 3201
        ):
            raise RuntimeError("PFN replay guard did not pass for every checkpoint")
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, dict) or set(comparisons) != set(
            REPLAY_COMPARISONS
        ):
            raise RuntimeError("PFN replay comparison inventory mismatch")
        for name, comparison in comparisons.items():
            if not isinstance(comparison, dict) or set(comparison) != metric_names:
                raise RuntimeError("PFN replay comparison metric schema mismatch")
            if not all(
                np.isfinite(float(comparison[metric]))
                and float(comparison[metric]) >= 0.0
                for metric in metric_names - {"bit_identical"}
            ):
                raise RuntimeError("PFN replay comparison metric is invalid")
            if name in REPLAY_EXACT_COMPARISONS and (
                comparison["bit_identical"] is not True
                or any(
                    float(comparison[metric]) != 0.0
                    for metric in metric_names - {"bit_identical"}
                )
            ):
                raise RuntimeError("PFN exact replay control is not bit-identical")
        full_panel_names = {
            name.removeprefix("full_panel_")
            for name in REPLAY_COMPARISONS
            if name.startswith("full_panel_")
        }
        full_panel_shards = row.get("full_panel_shards")
        if not isinstance(full_panel_shards, list) or len(full_panel_shards) != 9:
            raise RuntimeError("PFN full-panel replay shard inventory mismatch")
        if (
            {int(shard.get("shard_index", -1)) for shard in full_panel_shards}
            != set(range(9))
            or sum(int(shard.get("rows", -1)) for shard in full_panel_shards) != 3201
            or {int(shard.get("rows", -1)) for shard in full_panel_shards} != {355, 356}
        ):
            raise RuntimeError("PFN full-panel replay row inventory mismatch")
        for name in full_panel_names:
            shard_metrics = []
            for shard in full_panel_shards:
                shard_comparisons = shard.get("comparisons")
                if (
                    not isinstance(shard_comparisons, dict)
                    or set(shard_comparisons) != full_panel_names
                ):
                    raise RuntimeError("PFN full-panel comparison inventory mismatch")
                metric = shard_comparisons[name]
                if not isinstance(metric, dict) or set(metric) != metric_names:
                    raise RuntimeError("PFN full-panel comparison schema mismatch")
                shard_metrics.append(metric)
            aggregate = comparisons[f"full_panel_{name}"]
            if bool(aggregate["bit_identical"]) != all(
                bool(metric["bit_identical"]) for metric in shard_metrics
            ) or any(
                float(aggregate[metric_name])
                != max(float(metric[metric_name]) for metric in shard_metrics)
                for metric_name in metric_names - {"bit_identical"}
            ):
                raise RuntimeError("PFN full-panel replay aggregate mismatch")
        batch_max = max(
            float(comparisons[name]["max_abs_logp_error"])
            for name in REPLAY_BATCH_COMPARISONS
        )
        context_max = max(
            float(comparisons[name]["max_abs_logp_error"])
            for name in REPLAY_CONTEXT_COMPARISONS
        )
        combined_max = max(
            float(comparisons[name]["max_abs_logp_error"])
            for name in REPLAY_COMBINED_COMPARISONS
        )
        approximate = (
            *REPLAY_BATCH_COMPARISONS,
            *REPLAY_CONTEXT_COMPARISONS,
            *REPLAY_COMBINED_COMPARISONS,
        )
        probability_max = max(
            float(comparisons[name]["max_abs_probability_error"])
            for name in approximate
        )
        total_variation_max = max(
            float(comparisons[name]["max_total_variation"]) for name in approximate
        )
        if not all(
            (
                float(row["batch_max_abs_logp_error"]) == batch_max,
                float(row["context_max_abs_logp_error"]) == context_max,
                float(row["combined_max_abs_logp_error"]) == combined_max,
                float(row["approximate_max_abs_probability_error"]) == probability_max,
                float(row["approximate_max_total_variation"]) == total_variation_max,
            )
        ):
            raise RuntimeError("PFN replay aggregate does not match its comparisons")
        if (
            batch_max > float(config["pfn_batch_logp_atol"])
            or context_max > float(config["pfn_context_permutation_logp_atol"])
            or combined_max
            > float(config["pfn_combined_context_batch_logp_atol"])
            or probability_max > float(config["pfn_replay_probability_atol"])
            or total_variation_max > float(config["pfn_replay_total_variation_atol"])
        ):
            raise RuntimeError("PFN replay summary exceeds a frozen tolerance")
    registry_path = (repo_root() / str(config["checkpoint_registry"])).resolve()
    registry = load_json(registry_path)
    records = registry.get("records")
    if not isinstance(records, list) or len(records) != 18:
        raise RuntimeError("checkpoint registry does not contain exactly 18 records")
    expected_tuples = {
        (prior, seed, step)
        for prior in ("C", "N")
        for seed in MODEL_SEEDS.tolist()
        for step in CHECKPOINT_STEPS.tolist()
    }
    observed_tuples = {
        (str(row["prior"]), int(row["seed"]), int(row["checkpoint_step"]))
        for row in records
    }
    if observed_tuples != expected_tuples:
        raise RuntimeError("checkpoint registry tuple inventory mismatch")


def _verify_oracle_root(
    oracle_dir: Path,
    bank: int,
    identity: dict[str, Any],
    identity_sha256: str,
    config: dict[str, Any],
) -> None:
    complete = load_json(oracle_dir / "COMPLETE.json")
    _assert_identity(complete, identity, identity_sha256, f"oracle bank {bank}")
    if complete.get("decision") != "ORACLE_BANK_COMPLETE":
        raise RuntimeError(f"oracle completion decision mismatch: bank {bank}")
    _verify_artifacts(
        oracle_dir,
        complete.get("artifacts"),
        {"RUNNING.json", "ATOM_BANK.json", "oracle_bank_summary.json"},
    )
    expected_markers = {
        f"{prior}_d{draw}/COMPLETE.json" for prior in ("C", "N") for draw in range(3)
    }
    markers = complete.get("leaf_complete_sha256")
    if not isinstance(markers, dict) or set(markers) != expected_markers:
        raise RuntimeError(f"oracle leaf-marker inventory mismatch: bank {bank}")
    discovered_markers = {
        path.relative_to(oracle_dir).as_posix()
        for path in oracle_dir.rglob("COMPLETE.json")
        if path != oracle_dir / "COMPLETE.json"
    }
    if discovered_markers != expected_markers:
        raise RuntimeError(
            f"unexpected or missing oracle completion shard: bank {bank}"
        )
    for relative, expected_hash in markers.items():
        if sha256_file(oracle_dir / relative) != expected_hash:
            raise RuntimeError(f"oracle leaf-marker hash mismatch: {relative}")
    atom = load_json(oracle_dir / "ATOM_BANK.json")
    expected_atom = config["atom_banks"][bank]
    if (
        atom.get("bank_index") != bank
        or atom.get("seed") != expected_atom["seed"]
        or atom.get("count") != config["atom_count"]
        or atom.get("sha256") != expected_atom["sha256"]
        or atom.get("determinism_canary", {}).get("sha256")
        != config["atom_determinism_canary_sha256"]
    ):
        raise RuntimeError(f"oracle atom-bank marker mismatch: bank {bank}")


def _check_attempt_bytes(
    arrays: dict[str, np.ndarray], identity_sha256: str, path: Path
) -> None:
    expected = np.frombuffer(bytes.fromhex(identity_sha256), dtype=np.uint8)
    if not np.array_equal(arrays["attempt_identity_sha256"], expected):
        raise RuntimeError(f"raw attempt identity mismatch: {path}")


def _check_row_alignment(
    arrays: dict[str, np.ndarray], expected: dict[str, np.ndarray], path: Path
) -> None:
    for name in (*ROW_KEYS, "row_key_sha256", "input_row_sha256"):
        if not np.array_equal(arrays[name], expected[name]):
            raise RuntimeError(f"row alignment mismatch: {path}:{name}")


def _observed_nll(probability: np.ndarray, bins: np.ndarray, label: str) -> np.ndarray:
    if probability.dtype != np.float64 or probability.shape != (len(bins), 100):
        raise RuntimeError(f"predictive shape mismatch: {label}")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise RuntimeError(f"invalid predictive probability: {label}")
    if float(np.max(np.abs(probability.sum(axis=1) - 1.0))) > 1e-8:
        raise RuntimeError(f"predictive normalization mismatch: {label}")
    selected = probability[np.arange(len(bins)), bins]
    if not np.isfinite(selected).all() or np.any(selected <= 0.0):
        raise RuntimeError(f"observed-bin probability is nonpositive: {label}")
    return -np.log(selected)


def _observed_native_nll(
    log_probability: np.ndarray, bins: np.ndarray, label: str
) -> np.ndarray:
    if (
        log_probability.shape != (len(bins), 100)
        or not np.isfinite(log_probability).all()
    ):
        raise RuntimeError(f"invalid native log-probability: {label}")
    if float(np.max(np.abs(np.exp(log_probability).sum(axis=1) - 1.0))) > 1e-6:
        raise RuntimeError(f"native log-probability normalization mismatch: {label}")
    return -log_probability[np.arange(len(bins)), bins]


def _load_joined_rows(
    panel_dir: Path,
    pfn_dir: Path,
    oracle_dirs: list[Path],
    identity_sha256: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    fleet = load_fleet_module((repo_root() / str(config["generator_module"])).resolve())
    registry = load_json((repo_root() / str(config["checkpoint_registry"])).resolve())
    registry_by_tuple = {
        (str(row["prior"]), int(row["seed"]), int(row["checkpoint_step"])): row
        for row in registry["records"]
    }
    checkpoint_root = Path(str(registry["remote_root"])).resolve()
    n_total = 2 * 3 * int(config["contexts_per_prior_draw"])
    metadata = {name: np.empty(n_total, dtype=np.int64) for name in ROW_KEYS}
    metadata["row_key_sha256"] = np.empty((n_total, 32), dtype=np.uint8)
    metadata["input_row_sha256"] = np.empty((n_total, 32), dtype=np.uint8)
    outcome_bin = np.empty(n_total, dtype=np.int64)
    oracle_full_nll = np.empty(n_total, dtype=np.float64)
    oracle_ablated_nll = np.empty(n_total, dtype=np.float64)
    ordering_value = np.empty(n_total, dtype=np.float64)
    keep_full = np.empty(n_total, dtype=np.float64)
    keep_ablated = np.empty(n_total, dtype=np.float64)
    ess_full = np.empty(n_total, dtype=np.float64)
    ess_ablated = np.empty(n_total, dtype=np.float64)
    pfn_nll = np.empty((n_total, 3, 3), dtype=np.float64)
    filled = np.zeros(n_total, dtype=np.int8)
    half_parts: list[dict[str, np.ndarray]] = []

    for prior_code, prior in enumerate(("C", "N")):
        for draw in range(3):
            for bank in range(3):
                name = f"{prior}_d{draw}_b{bank}.npz"
                input_path = panel_dir / "inputs" / name
                label_path = panel_dir / "labels" / name
                inputs = _load_npz(input_path, INPUT_ARRAYS)
                labels = _load_npz(label_path, LABEL_ARRAYS)
                _check_attempt_bytes(inputs, identity_sha256, input_path)
                _check_attempt_bytes(labels, identity_sha256, label_path)
                for key in (*ROW_KEYS, "row_key_sha256"):
                    if not np.array_equal(inputs[key], labels[key]):
                        raise RuntimeError(f"panel input/label mismatch: {name}:{key}")
                if not np.array_equal(
                    inputs["row_key_sha256"], _digest_rows(inputs, ROW_KEYS)
                ):
                    raise RuntimeError(f"panel row-key digest mismatch: {name}")
                if not np.array_equal(
                    inputs["input_row_sha256"],
                    _digest_rows(inputs, (*ROW_KEYS, "contexts", "queries")),
                ):
                    raise RuntimeError(f"panel input-row digest mismatch: {name}")
                row_ids = inputs["row_id"].astype(np.int64)
                n = len(row_ids)
                if (
                    inputs["contexts"].dtype != np.float64
                    or inputs["contexts"].shape != (n, 30, 4)
                    or inputs["queries"].dtype != np.float64
                    or inputs["queries"].shape != (n, 3)
                    or labels["outcomes"].dtype != np.float64
                    or labels["outcomes"].shape != (n,)
                    or labels["sigmas"].dtype != np.float64
                    or labels["sigmas"].shape != (n, 4, 4)
                    or labels["true_orderings"].dtype != np.int64
                    or labels["true_orderings"].shape != (n,)
                ):
                    raise RuntimeError(
                        f"panel input/label shape or dtype mismatch: {name}"
                    )
                if not all(
                    np.isfinite(value).all()
                    for value in (
                        inputs["contexts"],
                        inputs["queries"],
                        labels["outcomes"],
                        labels["sigmas"],
                    )
                ):
                    raise RuntimeError(f"panel contains non-finite values: {name}")
                if np.any(
                    (labels["true_orderings"] < 0) | (labels["true_orderings"] >= 24)
                ):
                    raise RuntimeError(f"panel ordering label is invalid: {name}")
                _validate_panel_covariances(fleet, labels["sigmas"], name)
                if np.any(filled[row_ids]):
                    raise RuntimeError("duplicate panel row ID")
                filled[row_ids] = 1
                for key in ROW_KEYS:
                    metadata[key][row_ids] = inputs[key]
                metadata["row_key_sha256"][row_ids] = inputs["row_key_sha256"]
                metadata["input_row_sha256"][row_ids] = inputs["input_row_sha256"]
                bins = labels["outcome_bins"].astype(np.int64)
                if labels["outcome_bins"].dtype != np.int64:
                    raise RuntimeError(f"outcome-bin dtype mismatch: {name}")
                if np.any((bins < 0) | (bins >= 100)):
                    raise RuntimeError(f"observed bin outside native head: {name}")
                if not np.array_equal(
                    bins, np.asarray(fleet.bin_y(labels["outcomes"]), dtype=np.int64)
                ):
                    raise RuntimeError(f"panel outcome/bin mismatch: {name}")
                outcome_bin[row_ids] = bins

                oracle_leaf = oracle_dirs[bank] / f"{prior}_d{draw}"
                oracle_complete = load_json(oracle_leaf / "COMPLETE.json")
                if oracle_complete.get(
                    "identity_sha256"
                ) != identity_sha256 or oracle_complete.get("identity_tuple") != [
                    prior,
                    draw,
                    bank,
                ]:
                    raise RuntimeError(f"oracle leaf identity mismatch: {oracle_leaf}")
                oracle_expected = {"oracle_raw.npz", "oracle_summary.json"}
                if draw == 0:
                    oracle_expected.add("oracle_half_raw.npz")
                _verify_artifacts(
                    oracle_leaf, oracle_complete.get("artifacts"), oracle_expected
                )
                oracle_path = oracle_leaf / "oracle_raw.npz"
                oracle_summary = load_json(oracle_leaf / "oracle_summary.json")
                if (
                    oracle_summary.get("identity_sha256") != identity_sha256
                    or oracle_summary.get("prior") != prior
                    or oracle_summary.get("draw_index") != draw
                    or oracle_summary.get("atom_bank_index") != bank
                    or oracle_summary.get("input_sha256") != sha256_file(input_path)
                    or oracle_summary.get("selected_truncation")
                    != int(config["selected_truncation"])
                ):
                    raise RuntimeError(f"oracle leaf summary mismatch: {oracle_leaf}")
                oracle = _load_npz(oracle_path, ORACLE_ARRAYS)
                _check_attempt_bytes(oracle, identity_sha256, oracle_path)
                _check_row_alignment(oracle, inputs, oracle_path)
                oracle_full_nll[row_ids] = _observed_nll(
                    oracle["full_probability"], bins, f"full {name}"
                )
                oracle_ablated_nll[row_ids] = _observed_nll(
                    oracle["ablated_probability"], bins, f"ablated {name}"
                )
                posterior = oracle["ordering_posterior"]
                if (
                    posterior.dtype != np.float64
                    or posterior.shape != (len(row_ids), 24)
                    or not np.isfinite(posterior).all()
                    or np.any(posterior < 0.0)
                ):
                    raise RuntimeError(
                        f"ordering-posterior shape/finite failure: {name}"
                    )
                if float(np.max(np.abs(posterior.sum(axis=1) - 1.0))) > 1e-8:
                    raise RuntimeError(
                        f"ordering-posterior normalization failure: {name}"
                    )
                keep_full[row_ids] = oracle["keep_full"]
                keep_ablated[row_ids] = oracle["keep_ablated"]
                ess_full[row_ids] = oracle["ess_full_atoms"]
                ess_ablated[row_ids] = oracle["ess_ablated_atoms"]
                ordering_value[row_ids] = (
                    oracle_ablated_nll[row_ids] - oracle_full_nll[row_ids]
                )

                for model_index, seed in enumerate(MODEL_SEEDS.tolist()):
                    for step_index, step in enumerate(CHECKPOINT_STEPS.tolist()):
                        prediction_leaf = (
                            pfn_dir / f"{prior}_s{seed}_t{step}" / f"d{draw}_b{bank}"
                        )
                        prediction_complete = load_json(
                            prediction_leaf / "COMPLETE.json"
                        )
                        if prediction_complete.get(
                            "identity_sha256"
                        ) != identity_sha256 or prediction_complete.get(
                            "identity_tuple"
                        ) != [prior, seed, step, draw, bank]:
                            raise RuntimeError(
                                f"PFN leaf identity mismatch: {prediction_leaf}"
                            )
                        _verify_artifacts(
                            prediction_leaf,
                            prediction_complete.get("artifacts"),
                            {"prediction_raw.npz", "prediction_summary.json"},
                        )
                        prediction_path = prediction_leaf / "prediction_raw.npz"
                        prediction_summary = load_json(
                            prediction_leaf / "prediction_summary.json"
                        )
                        if (
                            prediction_summary.get("identity_sha256") != identity_sha256
                            or prediction_summary.get("prior") != prior
                            or prediction_summary.get("model_seed") != seed
                            or prediction_summary.get("checkpoint_step") != step
                            or prediction_summary.get("draw_index") != draw
                            or prediction_summary.get("atom_bank_index") != bank
                            or prediction_summary.get("input_sha256")
                            != sha256_file(input_path)
                            or prediction_summary.get("raw_sha256")
                            != sha256_file(prediction_path)
                        ):
                            raise RuntimeError(
                                f"PFN leaf summary mismatch: {prediction_leaf}"
                            )
                        expected_checkpoint = registry_by_tuple[(prior, seed, step)]
                        observed_checkpoint = prediction_summary.get("checkpoint")
                        if not isinstance(observed_checkpoint, dict) or any(
                            observed_checkpoint.get(key) != value
                            for key, value in expected_checkpoint.items()
                        ):
                            raise RuntimeError(
                                f"PFN leaf checkpoint record mismatch: {prediction_leaf}"
                            )
                        expected_path = (
                            checkpoint_root / str(expected_checkpoint["filename"])
                        ).resolve()
                        if observed_checkpoint.get("resolved_path") != str(
                            expected_path
                        ):
                            raise RuntimeError(
                                f"PFN leaf checkpoint path mismatch: {prediction_leaf}"
                            )
                        prediction = _load_npz(prediction_path, PREDICTION_ARRAYS)
                        _check_attempt_bytes(
                            prediction, identity_sha256, prediction_path
                        )
                        _check_row_alignment(prediction, inputs, prediction_path)
                        if not np.all(prediction["model_seed"] == seed) or not np.all(
                            prediction["checkpoint_step"] == step
                        ):
                            raise RuntimeError(
                                f"PFN identity tuple mismatch: {prediction_path}"
                            )
                        if prediction["log_probability"].dtype != np.float64:
                            raise RuntimeError(
                                f"PFN log-probability dtype mismatch: {prediction_path}"
                            )
                        pfn_nll[row_ids, model_index, step_index] = (
                            _observed_native_nll(
                                prediction["log_probability"],
                                bins,
                                str(prediction_path),
                            )
                        )

                if np.any(inputs["nested_half_mask"]):
                    half_path = (
                        oracle_dirs[bank] / f"{prior}_d{draw}" / "oracle_half_raw.npz"
                    )
                    half_schema = tuple(
                        name for name in ORACLE_ARRAYS if not name.startswith("ess_")
                    )
                    half = _load_npz(half_path, half_schema)
                    _check_attempt_bytes(half, identity_sha256, half_path)
                    mask = inputs["nested_half_mask"] == 1
                    expected_half = {
                        key: inputs[key][mask]
                        for key in (*ROW_KEYS, "row_key_sha256", "input_row_sha256")
                    }
                    _check_row_alignment(half, expected_half, half_path)
                    half_bins = bins[mask]
                    half_parts.append(
                        {
                            "row_id": half["row_id"].copy(),
                            "prior_code": half["prior_code"].copy(),
                            "draw_index": half["draw_index"].copy(),
                            "evaluation_seed": half["evaluation_seed"].copy(),
                            "stream_index": half["stream_index"].copy(),
                            "atom_bank_index": half["atom_bank_index"].copy(),
                            "atom_seed": half["atom_seed"].copy(),
                            "shard_local_index": half["shard_local_index"].copy(),
                            "row_key_sha256": half["row_key_sha256"].copy(),
                            "input_row_sha256": half["input_row_sha256"].copy(),
                            "oracle_half_full_nll": _observed_nll(
                                half["full_probability"], half_bins, f"half full {name}"
                            ),
                            "oracle_half_ablated_nll": _observed_nll(
                                half["ablated_probability"],
                                half_bins,
                                f"half ablated {name}",
                            ),
                        }
                    )
    if not np.all(filled == 1) or not np.array_equal(
        metadata["row_id"], np.arange(n_total)
    ):
        raise RuntimeError("joined panel is not exactly row IDs 0..6401")
    if not np.array_equal(metadata["atom_bank_index"], metadata["stream_index"] % 3):
        raise RuntimeError("joined atom-bank assignment mismatch")
    if not np.array_equal(metadata["shard_local_index"], metadata["stream_index"] // 3):
        raise RuntimeError("joined shard-local index mismatch")
    for name, value in (("keep_full", keep_full), ("keep_ablated", keep_ablated)):
        if np.any(value <= 0.0) or np.any(value > 1.0 + 1e-6):
            raise RuntimeError(f"joined retained-mass range failure: {name}")
    for name, value in (
        ("ess_full_atoms", ess_full),
        ("ess_ablated_atoms", ess_ablated),
    ):
        if np.any(value <= 0.0) or np.any(value > int(config["atom_count"]) + 1e-6):
            raise RuntimeError(f"joined collapsed-ESS range failure: {name}")
    deficit = pfn_nll - oracle_ablated_nll[:, None, None]
    gap = pfn_nll - oracle_full_nll[:, None, None]
    algebra_error = float(
        np.max(np.abs(deficit - (gap - ordering_value[:, None, None])))
    )
    if algebra_error > 1e-12:
        raise RuntimeError(f"row algebra identity failure: {algebra_error}")
    numeric = (
        oracle_full_nll,
        oracle_ablated_nll,
        ordering_value,
        keep_full,
        keep_ablated,
        ess_full,
        ess_ablated,
        pfn_nll,
        deficit,
        gap,
    )
    if not all(np.isfinite(value).all() for value in numeric):
        raise RuntimeError("joined raw tensor contains non-finite values")
    rows = {
        "schema_version": np.array([1], dtype=np.int64),
        "attempt_identity_sha256": np.frombuffer(
            bytes.fromhex(identity_sha256), dtype=np.uint8
        ).copy(),
        **metadata,
        "outcome_bin": outcome_bin,
        "model_seeds": MODEL_SEEDS.copy(),
        "checkpoint_steps": CHECKPOINT_STEPS.copy(),
        "oracle_full_nll": oracle_full_nll,
        "oracle_ablated_nll": oracle_ablated_nll,
        "ordering_value": ordering_value,
        "keep_full": keep_full,
        "keep_ablated": keep_ablated,
        "ess_full_atoms": ess_full,
        "ess_ablated_atoms": ess_ablated,
        "pfn_nll": pfn_nll,
        "deficit": deficit,
        "gap": gap,
    }
    half_rows = {
        name: np.concatenate([part[name] for part in half_parts])
        for name in half_parts[0]
    }
    order = np.argsort(half_rows["row_id"])
    half_rows = {name: value[order] for name, value in half_rows.items()}
    if len(half_rows["row_id"]) != 400 or len(np.unique(half_rows["row_id"])) != 400:
        raise RuntimeError("nested-half subset is not exactly 400 unique rows")
    expected_half_ids = np.concatenate(
        [
            np.arange(prior_code * 3 * 1067, prior_code * 3 * 1067 + 200)
            for prior_code in (0, 1)
        ]
    )
    if not np.array_equal(half_rows["row_id"], expected_half_ids):
        raise RuntimeError(
            "nested-half subset differs from draw 0, stream indices 0..199"
        )
    half_ids = half_rows["row_id"]
    half_rows["oracle_full_nll"] = oracle_full_nll[half_ids]
    half_rows["oracle_ablated_nll"] = oracle_ablated_nll[half_ids]
    half_rows["pfn_final_nll"] = pfn_nll[half_ids, :, 2]
    half_rows = {
        "schema_version": np.array([1], dtype=np.int64),
        "attempt_identity_sha256": np.frombuffer(
            bytes.fromhex(identity_sha256), dtype=np.uint8
        ).copy(),
        **half_rows,
    }
    return rows, half_rows, {"row_algebra_max_abs_error": algebra_error}


def _point_estimates(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    d = np.empty((2, 3), dtype=np.float64)
    v = np.empty(2, dtype=np.float64)
    gap_final = np.empty(2, dtype=np.float64)
    for prior_code in (0, 1):
        mask = rows["prior_code"] == prior_code
        if int(mask.sum()) != 3201:
            raise RuntimeError("prior arm does not contain exactly 3201 contexts")
        d[prior_code] = rows["deficit"][mask].mean(axis=0).mean(axis=0)
        v[prior_code] = rows["ordering_value"][mask].mean()
        gap_final[prior_code] = rows["gap"][mask, :, 2].mean(axis=0).mean()
    return {
        "deficit": d,
        "ordering_value": v,
        "gap_final": gap_final,
        "delta": d[0] - d[1],
        "deficit_change_final_minus_early": d[:, 2] - d[:, 0],
        "delta_change_final_minus_early": np.array(
            d[0, 2] - d[1, 2] - d[0, 0] + d[1, 0]
        ),
    }


def _bootstrap(
    rows: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    specification = config["bootstrap"]
    replicates = int(specification["replicates"])
    chunk_size = int(specification["chunk_size"])
    master_seed = int(specification["master_seed"])
    if (
        specification["bit_generator"] != "PCG64"
        or replicates != 50_000
        or chunk_size != 256
    ):
        raise RuntimeError("bootstrap specification drift")
    boot_d = np.zeros((2, 3, replicates), dtype=np.float64)
    boot_v = np.zeros((2, replicates), dtype=np.float64)
    boot_gap_final = np.zeros((2, replicates), dtype=np.float64)
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
                indices_in_rows = np.flatnonzero(mask)
                indices_in_rows = indices_in_rows[
                    np.argsort(rows["row_id"][indices_in_rows])
                ]
                n = len(indices_in_rows)
                expected_n = 356 if int(atom_record["bank_index"]) < 2 else 355
                if n != expected_n:
                    raise RuntimeError("bootstrap stratum size mismatch")
                local_v = rows["ordering_value"][indices_in_rows]
                local_d = rows["deficit"][indices_in_rows]
                local_gap = rows["gap"][indices_in_rows, :, 2]
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
                    boot_gap_final[prior_code, start:stop] += weight * sampled_gap
                key = f"{prior_name}_eval{int(evaluation_seed)}_atom{atom_seed}"
                hashes[key] = digest.hexdigest()
    delta = boot_d[0] - boot_d[1]
    change = delta[2] - delta[0]
    return {
        "ordering_value": boot_v,
        "deficit": boot_d,
        "gap_final": boot_gap_final,
        "delta": delta,
        "deficit_change_final_minus_early": boot_d[:, 2] - boot_d[:, 0],
        "delta_change_final_minus_early": change,
    }, hashes


def _interval(value: np.ndarray, quantiles=(0.025, 0.975)) -> list[float]:
    result = np.quantile(value, quantiles, method="linear")
    return [float(item) for item in np.atleast_1d(result)]


def _bootstrap_nested_half(
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
            indices_in_rows = np.flatnonzero(mask)
            indices_in_rows = indices_in_rows[
                np.argsort(half["row_id"][indices_in_rows])
            ]
            expected_n = 67 if bank_index < 2 else 66
            if len(indices_in_rows) != expected_n:
                raise RuntimeError("nested-half bootstrap stratum size mismatch")
            local_d = (
                half["oracle_ablated_nll"][indices_in_rows]
                - half["oracle_half_ablated_nll"][indices_in_rows]
            )
            local_e = (
                half["oracle_full_nll"][indices_in_rows]
                - half["oracle_half_full_nll"][indices_in_rows]
            )
            local_g = (
                np.mean(half["pfn_final_nll"][indices_in_rows], axis=1)
                - half["oracle_full_nll"][indices_in_rows]
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


def _nested_half_gate(
    half: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    bootstrap, index_hashes = _bootstrap_nested_half(half, config)
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
    difference_bootstrap = bootstrap["d"][0] - bootstrap["d"][1]
    difference_ci = _interval(difference_bootstrap)
    difference_limit = float(
        config["gates"]["nested_half_control_subtracted_ablated_abs_max"]
    )
    causal_pass = bool(causal_ci[0] > -d_limit and causal_ci[1] < d_limit)
    difference_pass = bool(
        difference_ci[0] > -difference_limit and difference_ci[1] < difference_limit
    )
    passed = bool(
        causal_pass
        and difference_pass
        and all(reports[prior]["full_predictive_pass"] for prior in ("C", "N"))
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
        "pass": passed,
    }


def _decide(
    points: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    half_gate: dict[str, Any],
    config: dict[str, Any],
    mechanical_gates: dict[str, bool],
) -> dict[str, Any]:
    gates = config["gates"]
    effect_floor = float(gates["primary_effect_floor"])
    numerical_clearance = float(gates["primary_numerical_clearance"])
    clearance_boundary = effect_floor - numerical_clearance
    rejection_boundary = effect_floor + numerical_clearance
    replay_epsilon = float(config["pfn_combined_context_batch_logp_atol"])
    replay_bounds = {
        "single_pfn_nll": replay_epsilon,
        "direct_deficit_or_gap": replay_epsilon,
        "causal_minus_control_delta": 2.0 * replay_epsilon,
        "direct_checkpoint_change": 2.0 * replay_epsilon,
        "delta_difference_in_differences": 4.0 * replay_epsilon,
    }
    if numerical_clearance <= 0.0:
        raise RuntimeError("primary numerical clearance must be positive")
    v_c_lower = float(
        np.quantile(
            bootstrap["ordering_value"][0],
            float(gates["value_c_one_sided_quantile"]),
            method="linear",
        )
    )
    v_n_ci = _interval(bootstrap["ordering_value"][1])
    ordering_gate = bool(
        v_c_lower > numerical_clearance
        and v_n_ci[0] >= float(gates["value_n_equivalence_lower"])
        and v_n_ci[1] <= float(gates["value_n_equivalence_upper"])
    )
    gap_reports: dict[str, Any] = {}
    kl_gate = True
    kl_threshold = float(gates["kl_alarm_ci_upper_below"])
    kl_alarm_boundary = kl_threshold - numerical_clearance
    kl_clear_boundary = kl_threshold + numerical_clearance
    for prior_code, prior in enumerate(("C", "N")):
        ci = _interval(bootstrap["gap_final"][prior_code])
        alarm = kl_alarm_boundary - ci[1] > replay_bounds["direct_deficit_or_gap"]
        clear = ci[1] - kl_clear_boundary > replay_bounds["direct_deficit_or_gap"]
        gap_reports[prior] = {
            "point": float(points["gap_final"][prior_code]),
            "ci95": ci,
            "alarm": alarm,
            "numerically_borderline": bool(not alarm and not clear),
            "clear": clear,
            "alarm_boundary": kl_alarm_boundary,
            "clear_boundary": kl_clear_boundary,
            "pfn_replay_bound": replay_bounds["direct_deficit_or_gap"],
        }
        kl_gate = kl_gate and clear
    validity_gates = {
        **mechanical_gates,
        "ordering_value": ordering_gate,
        "oracle_convergence": bool(half_gate["pass"]),
        "kl_alarm_clear": bool(kl_gate),
    }
    all_valid = all(validity_gates.values())
    deficit_reports: dict[str, dict[str, Any]] = {"C": {}, "N": {}}
    deficit_change_reports: dict[str, Any] = {}
    delta_reports: dict[str, Any] = {}
    for index, step in enumerate(CHECKPOINT_STEPS.tolist()):
        for prior_code, prior in enumerate(("C", "N")):
            deficit_point = float(points["deficit"][prior_code, index])
            deficit_ci = _interval(bootstrap["deficit"][prior_code, index])
            decision_endpoint = max(deficit_point, deficit_ci[1])
            endpoint_replay_bound = replay_bounds["direct_deficit_or_gap"]
            passes_effect_floor = bool(
                prior == "C"
                and deficit_point < effect_floor
                and deficit_ci[1] < effect_floor
            )
            passes_clearance = bool(
                prior == "C"
                and clearance_boundary - decision_endpoint > endpoint_replay_bound
            )
            clearly_fails = bool(
                prior == "C"
                and decision_endpoint - rejection_boundary > endpoint_replay_bound
            )
            deficit_reports[prior][str(step)] = {
                "point": deficit_point,
                "ci95": deficit_ci,
                "passes_effect_floor": passes_effect_floor,
                "passes_direct_rule": passes_clearance,
                "pfn_replay_bound": endpoint_replay_bound,
                "distance_beyond_positive_boundary": (
                    clearance_boundary - decision_endpoint
                ),
                "distance_beyond_rejection_boundary": (
                    decision_endpoint - rejection_boundary
                ),
                "numerically_borderline": bool(
                    prior == "C" and not passes_clearance and not clearly_fails
                ),
                "clearly_fails_effect_rule": clearly_fails,
            }
        point = float(points["delta"][index])
        ci = _interval(bootstrap["delta"][index])
        decision_endpoint = max(point, ci[1])
        endpoint_replay_bound = replay_bounds["causal_minus_control_delta"]
        passes_effect_floor = bool(point < effect_floor and ci[1] < effect_floor)
        passes_clearance = bool(
            clearance_boundary - decision_endpoint > endpoint_replay_bound
        )
        clearly_fails = bool(
            decision_endpoint - rejection_boundary > endpoint_replay_bound
        )
        delta_reports[str(step)] = {
            "point": point,
            "ci95": ci,
            "passes_effect_floor": passes_effect_floor,
            "passes_rule": passes_clearance,
            "pfn_replay_bound": endpoint_replay_bound,
            "distance_beyond_positive_boundary": (
                clearance_boundary - decision_endpoint
            ),
            "distance_beyond_rejection_boundary": (
                decision_endpoint - rejection_boundary
            ),
            "numerically_borderline": bool(not passes_clearance and not clearly_fails),
            "clearly_fails_effect_rule": clearly_fails,
        }
    for prior_code, prior in enumerate(("C", "N")):
        point = float(points["deficit_change_final_minus_early"][prior_code])
        ci = _interval(bootstrap["deficit_change_final_minus_early"][prior_code])
        decision_endpoint = max(point, ci[1])
        endpoint_replay_bound = replay_bounds["direct_checkpoint_change"]
        passes_effect_floor = bool(
            prior == "C" and point < effect_floor and ci[1] < effect_floor
        )
        passes_replay_margin = bool(
            prior == "C" and effect_floor - decision_endpoint > endpoint_replay_bound
        )
        deficit_change_reports[prior] = {
            "point": point,
            "ci95": ci,
            "passes_effect_floor": passes_effect_floor,
            # The identical ablated-oracle row cancels between checkpoints.
            "passes_direct_change_rule": passes_replay_margin,
            "pfn_replay_bound": endpoint_replay_bound,
            "distance_beyond_effect_floor": effect_floor - decision_endpoint,
            "replay_sensitive": bool(
                prior == "C" and passes_effect_floor and not passes_replay_margin
            ),
        }
    change_point = float(points["delta_change_final_minus_early"])
    change_ci = _interval(bootstrap["delta_change_final_minus_early"])
    change_passes_effect_floor = bool(
        change_point < effect_floor and change_ci[1] < effect_floor
    )
    change_endpoint = max(change_point, change_ci[1])
    change_replay_bound = replay_bounds["delta_difference_in_differences"]
    change_passes_replay_margin = bool(
        effect_floor - change_endpoint > change_replay_bound
    )
    if not all_valid:
        primary = "NOT_EVALUATED"
        secondary = "NOT_EVALUATED"
        decision = "INCONCLUSIVE_PHASE1_INSTRUMENT"
    else:
        final_pass = bool(
            deficit_reports["C"]["120000"]["passes_direct_rule"]
            and delta_reports["120000"]["passes_rule"]
        )
        final_clear_failure = bool(
            deficit_reports["C"]["120000"]["clearly_fails_effect_rule"]
            or delta_reports["120000"]["clearly_fails_effect_rule"]
        )
        if final_pass:
            primary = "REPLICATED_ORDERING_USE"
        elif final_clear_failure:
            primary = "NOT_REPLICATED_ORDERING_USE"
        else:
            primary = "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"
        change_pass = bool(
            deficit_change_reports["C"]["passes_direct_change_rule"]
            and change_passes_replay_margin
        )
        early_clear_failure = bool(
            deficit_reports["C"]["20000"]["clearly_fails_effect_rule"]
            or delta_reports["20000"]["clearly_fails_effect_rule"]
        )
        secondary_pass = bool(final_pass and early_clear_failure and change_pass)
        early_nominal_clear_failure = bool(
            deficit_reports["C"]["20000"]["distance_beyond_rejection_boundary"] > 0.0
            or delta_reports["20000"]["distance_beyond_rejection_boundary"] > 0.0
        )
        nominal_secondary_components = bool(
            final_pass
            and early_nominal_clear_failure
            and deficit_change_reports["C"]["passes_effect_floor"]
            and change_passes_effect_floor
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
        "validity_gates": validity_gates,
        "all_validity_gates_pass": all_valid,
        "ordering_value": {
            "C_point": float(points["ordering_value"][0]),
            "C_one_sided_95_lower": v_c_lower,
            "C_numerical_clearance": numerical_clearance,
            "N_point": float(points["ordering_value"][1]),
            "N_ci95": v_n_ci,
        },
        "oracle_convergence": half_gate,
        "kl_alarm": gap_reports,
        "deficit_by_prior_and_checkpoint": deficit_reports,
        "deficit_change_final_minus_early": deficit_change_reports,
        "delta_by_checkpoint": delta_reports,
        "delta_change_final_minus_early": {
            "point": change_point,
            "ci95": change_ci,
            "passes_effect_floor": change_passes_effect_floor,
            # The same full/ablated oracle rows cancel in this checkpoint change.
            "passes_rule": change_passes_replay_margin,
            "pfn_replay_bound": change_replay_bound,
            "distance_beyond_effect_floor": effect_floor - change_endpoint,
            "replay_sensitive": bool(
                change_passes_effect_floor and not change_passes_replay_margin
            ),
        },
        "numerical_clearance": {
            "effect_floor": effect_floor,
            "oracle_clearance_nats": numerical_clearance,
            "clearance_boundary": clearance_boundary,
            "rejection_boundary": rejection_boundary,
            "pfn_replay_bounds": replay_bounds,
            "borderline_action": (
                "stop_without_claim_and_increase_numeric_fidelity_or_replay_stability"
            ),
        },
    }


def join_confirmation(
    config_path: Path,
    panel_dir: Path,
    pfn_dir: Path,
    oracle_dirs: list[Path],
    output_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if len(oracle_dirs) != 3:
        raise ValueError("exactly three oracle bank directories are required")
    identity, identity_sha256, config, git = attempt_identity(
        config_path.resolve(), [Path(__file__)], device=torch.device(device_name)
    )
    panel_dir = panel_dir.resolve()
    pfn_dir = pfn_dir.resolve()
    oracle_dirs = [path.resolve() for path in oracle_dirs]
    _verify_panel(panel_dir, identity, identity_sha256)
    _verify_pfn_root(pfn_dir, identity, identity_sha256, config)
    for bank, directory in enumerate(oracle_dirs):
        _verify_oracle_root(directory, bank, identity, identity_sha256, config)
    upstream_markers = {
        "panel": sha256_file(panel_dir / "PANEL_COMPLETE.json"),
        "pfn": sha256_file(pfn_dir / "COMPLETE.json"),
        "oracle_banks": [
            sha256_file(directory / "COMPLETE.json") for directory in oracle_dirs
        ],
    }
    lease = acquire_empty_output(output_dir, identity_sha256)
    output_dir = output_dir.resolve()
    running_path = output_dir / "RUNNING.json"
    write_json_atomic(
        running_path, {"identity": identity, "identity_sha256": identity_sha256}
    )
    rows, half_rows, integrity = _load_joined_rows(
        panel_dir, pfn_dir, oracle_dirs, identity_sha256, config
    )
    points = _point_estimates(rows)
    bootstrap, index_hashes = _bootstrap(rows, config)
    half_gate = _nested_half_gate(half_rows, config)
    mechanical_gates = {
        "completeness_and_provenance": True,
        "inference_guards": True,
        "predictive_truncation": True,
        "monte_carlo_diagnostics_reported": True,
        "fixed_fleet_completeness": True,
    }
    decision = _decide(points, bootstrap, half_gate, config, mechanical_gates)
    raw_path = output_dir / "confirmatory_raw.npz"
    write_numeric_npz_atomic(raw_path, **rows)
    half_path = output_dir / "nested_half_raw.npz"
    write_numeric_npz_atomic(half_path, **half_rows)
    bootstrap_path = output_dir / "bootstrap_raw.npz"
    write_numeric_npz_atomic(bootstrap_path, **bootstrap)
    diagnostics = {
        prior: {
            "keep_full": {
                "minimum": float(
                    rows["keep_full"][rows["prior_code"] == prior_code].min()
                ),
                "median": float(
                    np.median(rows["keep_full"][rows["prior_code"] == prior_code])
                ),
            },
            "keep_ablated": {
                "minimum": float(
                    rows["keep_ablated"][rows["prior_code"] == prior_code].min()
                ),
                "median": float(
                    np.median(rows["keep_ablated"][rows["prior_code"] == prior_code])
                ),
            },
            "ess_full_atoms": {
                "minimum": float(
                    rows["ess_full_atoms"][rows["prior_code"] == prior_code].min()
                ),
                "median": float(
                    np.median(rows["ess_full_atoms"][rows["prior_code"] == prior_code])
                ),
            },
            "ess_ablated_atoms": {
                "minimum": float(
                    rows["ess_ablated_atoms"][rows["prior_code"] == prior_code].min()
                ),
                "median": float(
                    np.median(
                        rows["ess_ablated_atoms"][rows["prior_code"] == prior_code]
                    )
                ),
            },
        }
        for prior_code, prior in enumerate(("C", "N"))
    }
    end_identity, end_sha256, _, _ = expected_attempt_identity(config_path.resolve())
    if end_identity != identity or end_sha256 != identity_sha256:
        raise RuntimeError("confirmation attempt identity changed during join")
    summary = {
        "schema_version": 1,
        "stage": "phase1_ordering_confirmation_join",
        "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "git": git,
        "join_runtime": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "validated_runtime_binary_fingerprint": identity[
                "runtime_binary_fingerprint"
            ],
        },
        "scope": config["scope"],
        "rows": len(rows["row_id"]),
        "fixed_models_per_prior": 3,
        "checkpoint_steps": CHECKPOINT_STEPS.tolist(),
        "upstream_complete_sha256": upstream_markers,
        "bootstrap": {
            **config["bootstrap"],
            "index_stream_encoding": "little-endian-int64-c-order-v1",
            "index_stream_sha256": index_hashes,
        },
        "integrity": integrity,
        "diagnostics": diagnostics,
        "result": decision,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (raw_path, half_path, bootstrap_path)
        },
    }
    summary_path = output_dir / "confirmation_summary.json"
    write_json_atomic(summary_path, summary)
    complete_path = output_dir / "COMPLETE.json"
    write_json_atomic(
        complete_path,
        {
            "identity": identity,
            "identity_sha256": identity_sha256,
            "decision": decision["decision"],
            "upstream_complete_sha256": upstream_markers,
            "artifacts": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in (
                    running_path,
                    raw_path,
                    half_path,
                    bootstrap_path,
                    summary_path,
                )
            },
        },
    )
    lease.unlink()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--pfn", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    result = join_confirmation(
        arguments.config,
        arguments.panel,
        arguments.pfn,
        arguments.oracle,
        arguments.out,
        arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
