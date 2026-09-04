"""Generate and seal the outcome-separated Phase-1 confirmation panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .phase1_confirm_common import (
    acquire_empty_output,
    attempt_identity,
    repo_root,
    sha256_file,
    sha256_named_arrays,
)
from .phase1_ordering import generate_evaluation_stream, load_fleet_module
from .storage import write_json_atomic, write_numeric_npz_atomic


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
INPUT_ARRAYS = (
    "schema_version",
    "attempt_identity_sha256",
    *ROW_KEYS,
    "row_key_sha256",
    "input_row_sha256",
    "nested_half_mask",
    "contexts",
    "queries",
)
LABEL_ARRAYS = (
    "schema_version",
    "attempt_identity_sha256",
    *ROW_KEYS,
    "row_key_sha256",
    "outcomes",
    "outcome_bins",
    "sigmas",
    "true_orderings",
)

# The frozen generator constructs Sigma_ij and Sigma_ji through the same
# floating-point products in opposite operand order. IEEE-754 roundoff can
# therefore differ by a few ulps even though the analytical matrix is
# symmetric. This bound accepts only that construction-level roundoff and
# leaves the generated arrays unchanged.
COVARIANCE_SYMMETRY_RTOL = 4.0 * np.finfo(np.float64).eps


def _covariance_symmetry_diagnostics(sigmas: np.ndarray) -> dict[str, float]:
    if sigmas.ndim != 3 or sigmas.shape[1:] != (4, 4):
        raise RuntimeError("covariance symmetry input shape mismatch")
    transposed = sigmas.transpose(0, 2, 1)
    asymmetry = np.abs(sigmas - transposed)
    symmetry_scale = np.maximum(np.abs(sigmas), np.abs(transposed))
    relative_asymmetry = np.divide(
        asymmetry,
        symmetry_scale,
        out=np.zeros_like(asymmetry),
        where=symmetry_scale > 0.0,
    )
    if np.any(asymmetry > COVARIANCE_SYMMETRY_RTOL * symmetry_scale):
        raise RuntimeError(
            "generated covariance exceeds the float64 symmetry roundoff bound"
        )
    return {
        "covariance_symmetry_rtol": float(COVARIANCE_SYMMETRY_RTOL),
        "covariance_max_abs_asymmetry": float(np.max(asymmetry)),
        "covariance_max_relative_asymmetry": float(np.max(relative_asymmetry)),
    }


def _digest_rows(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    count = len(arrays[names[0]])
    result = np.empty((count, 32), dtype=np.uint8)
    for index in range(count):
        digest = sha256_named_arrays(
            {name: np.asarray(arrays[name][index]) for name in names}, names
        )
        result[index] = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)
    return result


def split_stream(
    stream: dict[str, np.ndarray],
    *,
    prior_code: int,
    draw_index: int,
    evaluation_seed: int,
    atom_seeds: list[int],
    identity_sha256: str,
    nested_half_draw: int,
    nested_half_stop: int,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    count = len(stream["contexts"])
    expected = {
        "contexts": (count, 30, 4),
        "queries": (count, 3),
        "outcomes": (count,),
        "outcome_bins": (count,),
        "sigmas": (count, 4, 4),
        "true_orderings": (count,),
    }
    if set(stream) != set(expected):
        raise RuntimeError("evaluation stream inventory mismatch")
    for name, shape in expected.items():
        if stream[name].shape != shape:
            raise RuntimeError(f"evaluation stream shape mismatch: {name}")
    if len(atom_seeds) != 3:
        raise ValueError("confirmation requires exactly three atom seeds")
    identity_bytes = np.frombuffer(bytes.fromhex(identity_sha256), dtype=np.uint8)
    outputs: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for bank_index, atom_seed in enumerate(atom_seeds):
        indices = np.arange(bank_index, count, 3, dtype=np.int64)
        n = len(indices)
        keys = {
            "row_id": ((prior_code * 3 + draw_index) * count + indices).astype(
                np.int64
            ),
            "prior_code": np.full(n, prior_code, dtype=np.int64),
            "draw_index": np.full(n, draw_index, dtype=np.int64),
            "evaluation_seed": np.full(n, evaluation_seed, dtype=np.int64),
            "stream_index": indices,
            "atom_bank_index": np.full(n, bank_index, dtype=np.int64),
            "atom_seed": np.full(n, atom_seed, dtype=np.int64),
            "shard_local_index": np.arange(n, dtype=np.int64),
        }
        row_key_sha256 = _digest_rows(keys, ROW_KEYS)
        contexts = np.ascontiguousarray(stream["contexts"][indices], dtype=np.float64)
        queries = np.ascontiguousarray(stream["queries"][indices], dtype=np.float64)
        input_row_sha256 = _digest_rows(
            {**keys, "contexts": contexts, "queries": queries},
            (*ROW_KEYS, "contexts", "queries"),
        )
        inputs = {
            "schema_version": np.array([1], dtype=np.int64),
            "attempt_identity_sha256": identity_bytes.copy(),
            **keys,
            "row_key_sha256": row_key_sha256,
            "input_row_sha256": input_row_sha256,
            "nested_half_mask": (
                (draw_index == nested_half_draw) & (indices < nested_half_stop)
            ).astype(np.int8),
            "contexts": contexts,
            "queries": queries,
        }
        labels = {
            "schema_version": np.array([1], dtype=np.int64),
            "attempt_identity_sha256": identity_bytes.copy(),
            **{name: value.copy() for name, value in keys.items()},
            "row_key_sha256": row_key_sha256.copy(),
            "outcomes": np.ascontiguousarray(
                stream["outcomes"][indices], dtype=np.float64
            ),
            "outcome_bins": np.ascontiguousarray(
                stream["outcome_bins"][indices], dtype=np.int64
            ),
            "sigmas": np.ascontiguousarray(stream["sigmas"][indices], dtype=np.float64),
            "true_orderings": np.ascontiguousarray(
                stream["true_orderings"][indices], dtype=np.int64
            ),
        }
        if tuple(inputs) != INPUT_ARRAYS or tuple(labels) != LABEL_ARRAYS:
            raise RuntimeError("panel schema construction drift")
        outputs.append((inputs, labels))
    return outputs


def _validate_stream(
    fleet: Any, stream: dict[str, np.ndarray], count: int
) -> dict[str, float]:
    for name in ("contexts", "queries", "outcomes", "sigmas"):
        if not np.isfinite(stream[name]).all():
            raise RuntimeError(f"non-finite generated stream array: {name}")
    sigmas = stream["sigmas"]
    covariance_diagnostics = _covariance_symmetry_diagnostics(sigmas)
    if np.any(np.linalg.eigvalsh(stream["sigmas"])[:, 0] <= 0.0):
        raise RuntimeError("generated covariance is not positive definite")
    if not np.all(fleet.validity_keep(stream["sigmas"])):
        raise RuntimeError("generated covariance violates the fleet validity region")
    if np.any((stream["true_orderings"] < 0) | (stream["true_orderings"] >= 24)):
        raise RuntimeError("generated ordering label is outside [0,23]")
    expected_bins = np.asarray(fleet.bin_y(stream["outcomes"]), dtype=np.int64)
    if not np.array_equal(stream["outcome_bins"], expected_bins):
        raise RuntimeError("generated outcome-bin mismatch")
    if len(expected_bins) != count or np.any(
        (expected_bins < 0) | (expected_bins >= 100)
    ):
        raise RuntimeError("generated native bin is outside [0,99]")
    return covariance_diagnostics


def build_panel(
    config_path: Path, output_dir: Path, device_name: str = "cuda"
) -> dict[str, Any]:
    device = torch.device(device_name)
    identity, identity_sha256, config, git = attempt_identity(
        config_path.resolve(),
        [Path(__file__), Path(__file__).with_name("phase1_ordering.py")],
        device=device,
    )
    lease = acquire_empty_output(output_dir, identity_sha256)
    output_dir = output_dir.resolve()
    running_path = output_dir / "RUNNING.json"
    write_json_atomic(
        running_path, {"identity": identity, "identity_sha256": identity_sha256}
    )
    fleet_path = (repo_root() / str(config["generator_module"])).resolve()
    fleet = load_fleet_module(fleet_path)
    count = int(config["contexts_per_prior_draw"])
    atom_seeds = [int(row["seed"]) for row in config["atom_banks"]]
    half = config["nested_half_subset"]
    stream_records: dict[str, Any] = {}
    artifact_paths: list[Path] = [running_path]
    global_row_ids: list[np.ndarray] = []
    for prior_code, prior in enumerate(config["priors"]):
        for draw_index, seed in enumerate(config["evaluation_seeds"][prior]):
            stream = generate_evaluation_stream(fleet, prior, count, 30, int(seed))
            validation = _validate_stream(fleet, stream, count)
            stream_name = f"{prior}_d{draw_index}"
            stream_records[stream_name] = {
                "prior": prior,
                "draw_index": draw_index,
                "evaluation_seed": int(seed),
                "rows": count,
                "content_sha256": sha256_named_arrays(stream),
                "validation": validation,
            }
            shards = split_stream(
                stream,
                prior_code=prior_code,
                draw_index=draw_index,
                evaluation_seed=int(seed),
                atom_seeds=atom_seeds,
                identity_sha256=identity_sha256,
                nested_half_draw=int(half["draw_index"]),
                nested_half_stop=int(half["stream_index_stop_exclusive"]),
            )
            if [len(inputs["row_id"]) for inputs, _ in shards] != [356, 356, 355]:
                raise RuntimeError("confirmation shard-size invariant failed")
            reconstructed = np.concatenate(
                [inputs["stream_index"] for inputs, _ in shards]
            )
            if not np.array_equal(np.sort(reconstructed), np.arange(count)):
                raise RuntimeError(
                    "confirmation shards do not reconstruct the complete stream"
                )
            for bank_index, (inputs, labels) in enumerate(shards):
                if any(
                    not np.array_equal(inputs[name], labels[name])
                    for name in (*ROW_KEYS, "row_key_sha256")
                ):
                    raise RuntimeError("panel input/label row-key mismatch")
                if not np.all(inputs["atom_bank_index"] == inputs["stream_index"] % 3):
                    raise RuntimeError("panel modulo assignment mismatch")
                if not np.all(
                    inputs["shard_local_index"] == inputs["stream_index"] // 3
                ):
                    raise RuntimeError("panel local-index mismatch")
                name = f"{prior}_d{draw_index}_b{bank_index}.npz"
                input_path = output_dir / "inputs" / name
                label_path = output_dir / "labels" / name
                write_numeric_npz_atomic(input_path, **inputs)
                write_numeric_npz_atomic(label_path, **labels)
                artifact_paths.extend((input_path, label_path))
                global_row_ids.append(inputs["row_id"])
    joined_rows = np.sort(np.concatenate(global_row_ids))
    if not np.array_equal(joined_rows, np.arange(2 * 3 * count)):
        raise RuntimeError("global confirmation row IDs are not exactly contiguous")

    input_paths = [path for path in artifact_paths if path.parent.name == "inputs"]
    inputs_complete_path = output_dir / "INPUTS_COMPLETE.json"
    write_json_atomic(
        inputs_complete_path,
        {
            "identity": identity,
            "identity_sha256": identity_sha256,
            "decision": "PANEL_INPUTS_COMPLETE",
            "artifacts": {
                path.name: {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in input_paths
            },
        },
    )
    artifact_paths.append(inputs_complete_path)

    manifest = {
        "schema_version": 1,
        "stage": "phase1_confirmation_panel",
        "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_PANEL_COMPLETE_MARKER",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "git": git,
        "algorithm": "full-stream-then-modulo-v1",
        "rows": int(len(joined_rows)),
        "shards": 18,
        "stream_records": stream_records,
        "artifacts": {
            str(path.relative_to(output_dir)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
            if path != running_path
        },
    }
    manifest_path = output_dir / "panel_manifest.json"
    write_json_atomic(manifest_path, manifest)
    artifact_paths.append(manifest_path)
    end_identity, end_sha256, _, _ = attempt_identity(
        config_path.resolve(),
        [Path(__file__), Path(__file__).with_name("phase1_ordering.py")],
        device=device,
    )
    if end_identity != identity or end_sha256 != identity_sha256:
        raise RuntimeError(
            "confirmation attempt identity changed during panel generation"
        )
    complete = {
        "identity": identity,
        "identity_sha256": identity_sha256,
        "decision": "PANEL_COMPLETE",
        "artifacts": {
            str(path.relative_to(output_dir)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    complete_path = output_dir / "PANEL_COMPLETE.json"
    write_json_atomic(complete_path, complete)
    lease.unlink()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    result = build_panel(arguments.config, arguments.out, arguments.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
