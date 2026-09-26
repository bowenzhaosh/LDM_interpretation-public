"""Outcome-blind exact-oracle scorer for one Phase-1 atom bank.

This process accepts panel inputs only. It does not accept or open outcomes,
observed bins, PFN predictions, or scientific endpoints. The join is the only
process allowed to combine these predictions with labels and PFN outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .phase1_confirm_common import (
    acquire_empty_output,
    attempt_identity,
    load_json,
    repo_root,
    sha256_file,
)
from .phase1_ordering import (
    OrderingOracle,
    load_fleet_module,
    quadrature_grid,
    sample_sigmas_exact,
)
from .phase1_panel import INPUT_ARRAYS, ROW_KEYS, _digest_rows
from .storage import write_json_atomic, write_numeric_npz_atomic


ORACLE_ARRAYS = (
    "schema_version",
    "attempt_identity_sha256",
    *ROW_KEYS,
    "row_key_sha256",
    "input_row_sha256",
    "full_probability",
    "ablated_probability",
    "ordering_posterior",
    "keep_full",
    "keep_ablated",
    "ess_full_atoms",
    "ess_ablated_atoms",
)


def _make_oracle(
    fleet: Any,
    atoms: np.ndarray,
    device: torch.device,
    config: dict[str, Any],
) -> OrderingOracle:
    if config.get("oracle_compute_dtype") != "float64":
        raise RuntimeError("confirmation oracle compute dtype must be float64")
    oracle = OrderingOracle(
        fleet,
        atoms,
        device=device,
        context_atom_batch=int(config["context_atom_batch"]),
        compute_dtype=torch.float64,
    )
    if oracle.compute_dtype != torch.float64:
        raise RuntimeError("confirmation oracle did not retain float64 compute")
    return oracle


def _legacy_raw_array_sha256(value: np.ndarray) -> str:
    """Match the byte-only digest used by the frozen qualification runner."""

    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _verify_inputs_marker(
    panel_dir: Path, identity: dict[str, Any], identity_sha256: str
) -> None:
    marker = load_json(panel_dir / "INPUTS_COMPLETE.json")
    if (
        marker.get("identity") != identity
        or marker.get("identity_sha256") != identity_sha256
    ):
        raise RuntimeError("panel-input attempt identity mismatch")
    artifacts = marker.get("artifacts")
    expected = {
        f"{prior}_d{draw}_b{bank}.npz"
        for prior in ("C", "N")
        for draw in range(3)
        for bank in range(3)
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise RuntimeError("panel-input artifact inventory mismatch")
    for name, record in artifacts.items():
        path = panel_dir / "inputs" / name
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"panel-input artifact mismatch: {name}")


def _load_input_shard(
    path: Path,
    identity_sha256: str,
    prior: str,
    draw: int,
    bank: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != INPUT_ARRAYS:
            raise RuntimeError(f"panel-input schema mismatch: {path}")
        arrays = {name: archive[name].copy() for name in archive.files}
    n = 356 if bank < 2 else 355
    if arrays["contexts"].shape != (n, 30, 4) or arrays["queries"].shape != (n, 3):
        raise RuntimeError(f"panel-input shape mismatch: {path}")
    expected_identity = np.frombuffer(bytes.fromhex(identity_sha256), dtype=np.uint8)
    if not np.array_equal(arrays["attempt_identity_sha256"], expected_identity):
        raise RuntimeError(f"panel-input identity mismatch: {path}")
    prior_code = 0 if prior == "C" else 1
    checks = {
        "prior_code": prior_code,
        "draw_index": draw,
        "evaluation_seed": int(config["evaluation_seeds"][prior][draw]),
        "atom_bank_index": bank,
        "atom_seed": int(config["atom_banks"][bank]["seed"]),
    }
    if any(not np.all(arrays[name] == value) for name, value in checks.items()):
        raise RuntimeError(f"panel-input metadata mismatch: {path}")
    expected_indices = np.arange(bank, 1067, 3, dtype=np.int64)
    if not np.array_equal(arrays["stream_index"], expected_indices):
        raise RuntimeError(f"panel-input stream-index mismatch: {path}")
    if not np.array_equal(arrays["shard_local_index"], np.arange(n)):
        raise RuntimeError(f"panel-input local-index mismatch: {path}")
    if not np.array_equal(arrays["row_key_sha256"], _digest_rows(arrays, ROW_KEYS)):
        raise RuntimeError(f"panel-input row-key hash mismatch: {path}")
    if not np.array_equal(
        arrays["input_row_sha256"],
        _digest_rows(arrays, (*ROW_KEYS, "contexts", "queries")),
    ):
        raise RuntimeError(f"panel-input row-content hash mismatch: {path}")
    expected_half = (
        (draw == int(config["nested_half_subset"]["draw_index"]))
        & (
            expected_indices
            < int(config["nested_half_subset"]["stream_index_stop_exclusive"])
        )
    ).astype(np.int8)
    if not np.array_equal(arrays["nested_half_mask"], expected_half):
        raise RuntimeError(f"nested-half mask mismatch: {path}")
    forbidden = {"outcomes", "outcome_bins", "sigmas", "true_orderings"}
    if forbidden & set(arrays):
        raise RuntimeError("oracle scorer received forbidden label arrays")
    return arrays


def _prior_parameters(fleet: Any, prior: str) -> tuple[float, bool]:
    if prior == "C":
        return float(fleet.R_OF["C"]), False
    if prior == "N":
        return 2.0, True
    raise RuntimeError(f"unsupported prior: {prior}")


def _empty_oracle_arrays(
    shard: dict[str, np.ndarray], identity_sha256: str
) -> dict[str, np.ndarray]:
    n = len(shard["row_id"])
    return {
        "schema_version": np.array([1], dtype=np.int64),
        "attempt_identity_sha256": np.frombuffer(
            bytes.fromhex(identity_sha256), dtype=np.uint8
        ).copy(),
        **{name: shard[name].copy() for name in ROW_KEYS},
        "row_key_sha256": shard["row_key_sha256"].copy(),
        "input_row_sha256": shard["input_row_sha256"].copy(),
        "full_probability": np.empty((n, 100), dtype=np.float64),
        "ablated_probability": np.empty((n, 100), dtype=np.float64),
        "ordering_posterior": np.empty((n, 24), dtype=np.float64),
        "keep_full": np.empty(n, dtype=np.float64),
        "keep_ablated": np.empty(n, dtype=np.float64),
        "ess_full_atoms": np.empty(n, dtype=np.float64),
        "ess_ablated_atoms": np.empty(n, dtype=np.float64),
    }


def _score_shard(
    oracle: OrderingOracle,
    shard: dict[str, np.ndarray],
    prior: str,
    config: dict[str, Any],
    quadrature: tuple[np.ndarray, np.ndarray, np.ndarray],
    identity_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    values, bins, log_weights = quadrature
    r, gaussian = _prior_parameters(oracle.fleet, prior)
    selected = int(config["selected_truncation"])
    probability_atol = float(config["probability_sum_atol"])
    result = _empty_oracle_arrays(shard, identity_sha256)
    half_indices = np.flatnonzero(shard["nested_half_mask"] == 1)
    draw = int(shard["draw_index"][0])
    bank = int(shard["atom_bank_index"][0])
    expected_half_rows = (67, 67, 66)[bank] if draw == 0 else 0
    if len(half_indices) != expected_half_rows:
        raise RuntimeError("oracle nested-half row count mismatch")
    half_result: dict[str, np.ndarray] | None = None
    if len(half_indices):
        half_result = {
            "schema_version": np.array([1], dtype=np.int64),
            "attempt_identity_sha256": result["attempt_identity_sha256"].copy(),
            **{name: shard[name][half_indices].copy() for name in ROW_KEYS},
            "row_key_sha256": shard["row_key_sha256"][half_indices].copy(),
            "input_row_sha256": shard["input_row_sha256"][half_indices].copy(),
            "full_probability": np.empty((len(half_indices), 100), dtype=np.float64),
            "ablated_probability": np.empty((len(half_indices), 100), dtype=np.float64),
            "ordering_posterior": np.empty((len(half_indices), 24), dtype=np.float64),
            "keep_full": np.empty(len(half_indices), dtype=np.float64),
            "keep_ablated": np.empty(len(half_indices), dtype=np.float64),
        }
    half_lookup = {int(index): position for position, index in enumerate(half_indices)}
    for index in range(len(shard["row_id"])):
        likelihood = oracle.context_log_likelihood(
            shard["contexts"][index], r, gaussian
        )
        if (
            tuple(likelihood.shape) != (24, int(config["atom_count"]))
            or likelihood.dtype != torch.float64
            or not torch.isfinite(likelihood).all()
        ):
            raise RuntimeError("oracle context likelihood is invalid")
        ess_full, ess_ablated = oracle.collapsed_atom_ess(likelihood)
        prediction = oracle.predict_from_log_likelihood(
            likelihood,
            shard["queries"][index],
            r,
            gaussian,
            selected,
            values,
            bins,
            log_weights,
            int(config["query_grid_chunk"]),
            probability_atol,
        )
        result["full_probability"][index] = prediction.full
        result["ablated_probability"][index] = prediction.ablated
        result["ordering_posterior"][index] = prediction.ordering_posterior
        result["keep_full"][index] = prediction.keep_full
        result["keep_ablated"][index] = prediction.keep_ablated
        result["ess_full_atoms"][index] = ess_full
        result["ess_ablated_atoms"][index] = ess_ablated
        if index in half_lookup:
            half_prediction = oracle.predict_from_log_likelihood(
                likelihood,
                shard["queries"][index],
                r,
                gaussian,
                selected,
                values,
                bins,
                log_weights,
                int(config["query_grid_chunk"]),
                probability_atol,
                atom_limit=int(config["nested_half_atom_count"]),
            )
            position = half_lookup[index]
            assert half_result is not None
            half_result["full_probability"][position] = half_prediction.full
            half_result["ablated_probability"][position] = half_prediction.ablated
            half_result["ordering_posterior"][position] = (
                half_prediction.ordering_posterior
            )
            half_result["keep_full"][position] = half_prediction.keep_full
            half_result["keep_ablated"][position] = half_prediction.keep_ablated
        del likelihood
    if tuple(result) != ORACLE_ARRAYS:
        raise RuntimeError("oracle output schema drift")
    if half_result is not None and tuple(half_result) != tuple(
        name for name in ORACLE_ARRAYS if not name.startswith("ess_")
    ):
        raise RuntimeError("half-oracle output schema drift")
    for name in ("full_probability", "ablated_probability"):
        for arrays in (result, half_result):
            if arrays is None:
                continue
            probability = arrays[name]
            if not np.isfinite(probability).all() or np.any(probability < 0.0):
                raise RuntimeError(f"invalid oracle probability: {name}")
            if float(np.max(np.abs(probability.sum(axis=1) - 1.0))) > probability_atol:
                raise RuntimeError(f"oracle probability normalization failure: {name}")
    for arrays in (result, half_result):
        if arrays is None:
            continue
        posterior = arrays["ordering_posterior"]
        if (
            not np.isfinite(posterior).all()
            or np.any(posterior < 0.0)
            or float(np.max(np.abs(posterior.sum(axis=1) - 1.0))) > probability_atol
        ):
            raise RuntimeError("oracle ordering posterior is invalid")
        for name in ("keep_full", "keep_ablated"):
            if (
                not np.isfinite(arrays[name]).all()
                or np.any(arrays[name] <= 0.0)
                or np.any(arrays[name] > 1.0 + 1e-6)
            ):
                raise RuntimeError(f"oracle retained mass is invalid: {name}")
    for name in ("ess_full_atoms", "ess_ablated_atoms"):
        if np.any(result[name] <= 0.0) or np.any(
            result[name] > int(config["atom_count"]) + 1e-6
        ):
            raise RuntimeError(f"oracle collapsed ESS is invalid: {name}")
    if not all(
        np.isfinite(result[name]).all()
        for name in ORACLE_ARRAYS
        if name
        not in {
            "schema_version",
            "attempt_identity_sha256",
            *ROW_KEYS,
            "row_key_sha256",
            "input_row_sha256",
        }
    ):
        raise RuntimeError("non-finite oracle diagnostics")
    return result, half_result


def score_oracle_bank(
    config_path: Path,
    panel_dir: Path,
    output_dir: Path,
    bank_index: int,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if bank_index not in {0, 1, 2}:
        raise ValueError("bank_index must be 0, 1, or 2")
    started = time.time()
    device = torch.device(device_name)
    identity, identity_sha256, config, git = attempt_identity(
        config_path.resolve(), [Path(__file__)], device=device
    )
    _verify_inputs_marker(panel_dir.resolve(), identity, identity_sha256)
    lease = acquire_empty_output(output_dir, identity_sha256)
    output_dir = output_dir.resolve()
    running_path = output_dir / "RUNNING.json"
    write_json_atomic(
        running_path, {"identity": identity, "identity_sha256": identity_sha256}
    )
    fleet_path = (repo_root() / str(config["generator_module"])).resolve()
    fleet = load_fleet_module(fleet_path)
    atom_record = config["atom_banks"][bank_index]
    atoms = sample_sigmas_exact(
        fleet,
        np.random.default_rng(int(atom_record["seed"])),
        int(config["atom_count"]),
    )
    atom_sha256 = _legacy_raw_array_sha256(atoms)
    if atom_sha256 != atom_record["sha256"]:
        raise RuntimeError(
            "regenerated confirmation atom bank differs from qualification"
        )
    half_atom_sha256 = _legacy_raw_array_sha256(
        atoms[: int(config["nested_half_atom_count"])]
    )
    canary = sample_sigmas_exact(
        fleet,
        np.random.default_rng(int(config["atom_determinism_canary_seed"])),
        int(config["atom_determinism_canary_count"]),
    )
    canary_sha256 = _legacy_raw_array_sha256(canary)
    if canary_sha256 != config["atom_determinism_canary_sha256"]:
        raise RuntimeError("confirmation atom determinism canary mismatch")
    atom_marker_path = output_dir / "ATOM_BANK.json"
    write_json_atomic(
        atom_marker_path,
        {
            "bank_index": bank_index,
            "seed": int(atom_record["seed"]),
            "count": int(config["atom_count"]),
            "shape": list(atoms.shape),
            "dtype": atoms.dtype.str,
            "hash_scheme": config["atom_hash_scheme"],
            "sha256": atom_sha256,
            "nested_half_prefix": {
                "count": int(config["nested_half_atom_count"]),
                "shape": [int(config["nested_half_atom_count"]), 4, 4],
                "dtype": atoms.dtype.str,
                "sha256": half_atom_sha256,
            },
            "determinism_canary": {
                "seed": int(config["atom_determinism_canary_seed"]),
                "count": int(config["atom_determinism_canary_count"]),
                "shape": list(canary.shape),
                "dtype": canary.dtype.str,
                "sha256": canary_sha256,
            },
        },
    )
    del canary
    oracle = _make_oracle(fleet, atoms, device, config)
    del atoms
    quadrature = quadrature_grid(fleet, config)
    leaf_markers: dict[str, str] = {}
    diagnostic_rows: dict[str, Any] = {}
    for prior in ("C", "N"):
        for draw in range(3):
            input_path = panel_dir / "inputs" / f"{prior}_d{draw}_b{bank_index}.npz"
            shard = _load_input_shard(
                input_path, identity_sha256, prior, draw, bank_index, config
            )
            raw, half = _score_shard(
                oracle, shard, prior, config, quadrature, identity_sha256
            )
            shard_dir = output_dir / f"{prior}_d{draw}"
            raw_path = shard_dir / "oracle_raw.npz"
            write_numeric_npz_atomic(raw_path, **raw)
            artifact_paths = [raw_path]
            half_path: Path | None = None
            if half is not None:
                half_path = shard_dir / "oracle_half_raw.npz"
                write_numeric_npz_atomic(half_path, **half)
                artifact_paths.append(half_path)
            diagnostic_rows[f"{prior}_d{draw}"] = {
                "rows": len(raw["row_id"]),
                "half_rows": 0 if half is None else len(half["row_id"]),
                "minimum_keep_full": float(raw["keep_full"].min()),
                "minimum_keep_ablated": float(raw["keep_ablated"].min()),
                "minimum_ess_full_atoms": float(raw["ess_full_atoms"].min()),
                "minimum_ess_ablated_atoms": float(raw["ess_ablated_atoms"].min()),
            }
            summary_path = shard_dir / "oracle_summary.json"
            write_json_atomic(
                summary_path,
                {
                    "schema_version": 1,
                    "stage": "phase1_confirmation_oracle_prediction",
                    "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER",
                    "identity_sha256": identity_sha256,
                    "git": git,
                    "prior": prior,
                    "draw_index": draw,
                    "atom_bank_index": bank_index,
                    "atom_count": int(config["atom_count"]),
                    "nested_half_atom_count": int(config["nested_half_atom_count"]),
                    "selected_truncation": int(config["selected_truncation"]),
                    "input_sha256": sha256_file(input_path),
                    "artifacts": {
                        path.name: {
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in artifact_paths
                    },
                    "diagnostics": diagnostic_rows[f"{prior}_d{draw}"],
                },
            )
            artifact_paths.append(summary_path)
            complete_path = shard_dir / "COMPLETE.json"
            write_json_atomic(
                complete_path,
                {
                    "identity_sha256": identity_sha256,
                    "identity_tuple": [prior, draw, bank_index],
                    "artifacts": {
                        path.name: {
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in artifact_paths
                    },
                },
            )
            leaf_markers[str(complete_path.relative_to(output_dir))] = sha256_file(
                complete_path
            )
    end_identity, end_sha256, _, _ = attempt_identity(
        config_path.resolve(), [Path(__file__)], device=device
    )
    if end_identity != identity or end_sha256 != identity_sha256:
        raise RuntimeError(
            "confirmation attempt identity changed during oracle scoring"
        )
    summary = {
        "schema_version": 1,
        "stage": "phase1_confirmation_oracle_bank",
        "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "git": git,
        "bank_index": bank_index,
        "atom_bank_sha256": atom_sha256,
        "atom_determinism_canary_sha256": canary_sha256,
        "selected_truncation": int(config["selected_truncation"]),
        "rows": int(sum(row["rows"] for row in diagnostic_rows.values())),
        "leaf_complete_sha256": leaf_markers,
        "diagnostics": diagnostic_rows,
        "wall_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    summary_path = output_dir / "oracle_bank_summary.json"
    write_json_atomic(summary_path, summary)
    complete_path = output_dir / "COMPLETE.json"
    write_json_atomic(
        complete_path,
        {
            "identity": identity,
            "identity_sha256": identity_sha256,
            "decision": "ORACLE_BANK_COMPLETE",
            "artifacts": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in (running_path, atom_marker_path, summary_path)
            },
            "leaf_complete_sha256": leaf_markers,
        },
    )
    lease.unlink()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    result = score_oracle_bank(
        arguments.config,
        arguments.panel,
        arguments.out,
        arguments.bank,
        arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
