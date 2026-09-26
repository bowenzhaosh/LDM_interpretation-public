"""Independent verifier for a completed Phase-1 oracle calibration artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ARTIFACTS = {
    "RUNNING.json",
    "calibration_raw.npz",
    "calibration_summary.json",
    "partial_C.npz",
    "partial_N.npz",
}

FORBIDDEN = {
    "contexts",
    "queries",
    "outcomes",
    "outcome_bins",
    "true_orderings",
    "full_probability",
    "ablated_probability",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(js: np.ndarray, logp: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    values = {
        "median_js": float(np.median(js)),
        "p95_js": float(np.quantile(js, 0.95)),
        "median_abs_logp_change": float(np.median(logp)),
        "p95_abs_logp_change": float(np.quantile(logp, 0.95)),
    }
    limits = {
        "median_js": float(thresholds["median_js_max"]),
        "p95_js": float(thresholds["p95_js_max"]),
        "median_abs_logp_change": float(thresholds["median_abs_logp_change_max"]),
        "p95_abs_logp_change": float(thresholds["p95_abs_logp_change_max"]),
    }
    margin = float(thresholds["numerical_indifference_fraction"])
    values["pass"] = bool(
        all(float(values[name]) <= limit * (1.0 - margin) for name, limit in limits.items())
    )
    values["borderline"] = bool(
        any(
            limit * (1.0 - margin) < float(values[name]) <= limit * (1.0 + margin)
            for name, limit in limits.items()
        )
    )
    return values


def verify(
    run_dir: Path, config_path: Path, output_dir: Path | None = None, job_id: str | None = None
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    complete = _load(run_dir / "COMPLETE.json")
    summary = _load(run_dir / "calibration_summary.json")
    running = _load(run_dir / "RUNNING.json")
    config = _load(config_path)
    artifacts = complete.get("artifacts", {})
    if set(artifacts) != EXPECTED_ARTIFACTS:
        raise RuntimeError("COMPLETE artifact inventory mismatch")
    artifact_checks: dict[str, Any] = {}
    for name, metadata in artifacts.items():
        path = run_dir / name
        artifact_checks[name] = {
            "exists": path.is_file(),
            "bytes_match": path.is_file() and path.stat().st_size == int(metadata["bytes"]),
            "sha_match": path.is_file() and _sha(path) == str(metadata["sha256"]),
        }
    if not all(all(row.values()) for row in artifact_checks.values()):
        raise RuntimeError("COMPLETE artifact verification failed")
    identity_match = (
        complete.get("identity_sha256")
        == running.get("identity_sha256")
        == summary.get("attempt_identity_sha256")
    )
    if not identity_match:
        raise RuntimeError("attempt identity mismatch")
    if summary.get("config_sha256") != _sha(config_path):
        raise RuntimeError("config hash mismatch")
    with np.load(run_dir / "calibration_raw.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    if FORBIDDEN & set(raw):
        raise RuntimeError("forbidden arrays present in calibration raw artifact")
    candidate_metrics: list[dict[str, Any]] = []
    selected: int | None = None
    for candidate_index, candidate in enumerate(raw["candidates"].tolist()):
        pass_all = True
        for prior_index, prior in enumerate(("C", "N")):
            for predictor in ("full", "ablated"):
                row = _row(
                    raw[f"js_{predictor}"][prior_index, candidate_index],
                    raw[f"abs_logp_change_{predictor}"][prior_index, candidate_index],
                    config["thresholds"],
                )
                pass_all = pass_all and bool(row["pass"])
                candidate_metrics.append(
                    {
                        "candidate": candidate,
                        "prior": prior,
                        "predictor": predictor,
                        **row,
                    }
                )
        if selected is None and pass_all:
            selected = candidate
    numeric_names = [name for name in raw if name != "attempt_identity_sha256"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "source_commit": summary["git"]["commit"],
        "source_dirty": summary["git"]["dirty"],
        "verified_complete_marker": True,
        "artifact_checks": artifact_checks,
        "all_marker_artifact_hashes_and_sizes_match": True,
        "attempt_identity_matches_running_summary_and_marker": identity_match,
        "all_contexts_completed": bool(np.all(raw["completed"] == 1)),
        "all_diagnostics_finite": bool(
            all(np.isfinite(raw[name]).all() for name in numeric_names)
        ),
        "forbidden_calibration_arrays_present": sorted(FORBIDDEN & set(raw)),
        "maximum_probability_sum_error": float(
            max(
                raw["full_probability_sum_error"].max(),
                raw["ablated_probability_sum_error"].max(),
            )
        ),
        "minimum_probability": float(
            min(
                raw["full_probability_minimum"].min(),
                raw["ablated_probability_minimum"].min(),
            )
        ),
        "reference_truncation": int(raw["reference_truncation"][0]),
        "all_candidates_compared_directly_to_reference": all(
            row.get("compared_with") == int(raw["reference_truncation"][0])
            for row in summary["candidate_results"]
        ),
        "candidate_metrics": candidate_metrics,
        "recomputed_decision": "CALIBRATION_PASS" if selected is not None else "CALIBRATION_FAIL",
        "recomputed_selected_truncation": selected,
        "reported_selected_truncation": summary["selected_truncation"],
        "raw_sha256": _sha(run_dir / "calibration_raw.npz"),
        "cuda_peak_allocated_bytes": summary["cuda_peak_allocated_bytes"],
        "wall_seconds": summary["wall_seconds"],
        "scope": (
            "Artifact-level verification of the locked finite calibration panel; "
            "not an independent A100 replay or a PFN/scientific-endpoint result."
        ),
    }
    if output_dir is not None:
        output_dir = output_dir.resolve()
        if job_id is None:
            raise ValueError("job_id is required when an output directory is supplied")
        slurm = output_dir / f"slurm-{job_id}.out"
        gpu_trace = output_dir / "gpu_utilization.csv"
        gpu_identity = output_dir / "gpu_identity.csv"
        result.update(
            {
                "slurm_log_sha256": _sha(slurm),
                "gpu_trace_sha256": _sha(gpu_trace),
                "gpu_identity_sha256": _sha(gpu_identity),
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = verify(arguments.run, arguments.config, arguments.output_dir, arguments.job_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
