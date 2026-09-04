import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from .analysis import _derive_one
from .evaluation import (
    _peak_rss_bytes,
    configure_determinism,
    generate_panel,
    score_checkpoints,
)
from .provenance import (
    MEMORY_CAP_BYTES,
    RAW_CAP_BYTES,
    WALL_CAP_SECONDS,
    load_locked_query_banks,
    repository_root,
    verify_runtime,
)
from .storage import load_numeric_npz, write_json_atomic, write_numeric_npz_atomic
from .registry import package_source_hashes, sha256_file, validation_input_hashes


SMOKE_COMMIT = "5b0c0e" * 6 + "5b0c"


def run_smoke(out_path: Path) -> dict:
    configure_determinism(0)
    root = repository_root()
    verify_runtime(root)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pfn-dag-smoke-v3-") as temporary:
        run_dir = Path(temporary)
        panel_path = run_dir / "panel.npz"
        panel_metadata = generate_panel(
            commit_sha=SMOKE_COMMIT,
            query_banks=load_locked_query_banks(root),
            n_groups=8,
            n_continuations=8,
            out_path=panel_path,
            interior_selected=True,
            max_core_candidates=2_000,
            max_blocks_per_core=512,
            min_within_group_sd=0.25,
            scientific=False,
        )
        panel = load_numeric_npz(panel_path)
        if panel["eligible_replace"].shape != (8, 8) or not panel[
            "eligible_replace"
        ].astype(bool).all():
            raise AssertionError("smoke selected-interior panel is not fully eligible")

        prediction_dir = run_dir / "predictions"
        prediction_ledger = score_checkpoints(
            panel_path=panel_path,
            registry_path=root / "config" / "checkpoint_registry.json",
            out_dir=prediction_dir,
            seeds=[0],
            steps=[0, 12_000],
            scientific=False,
        )
        derive_started = time.perf_counter()
        derived_dir = run_dir / "derived"
        derived_dir.mkdir()
        derived_bytes = 0
        for record in prediction_ledger["records"]:
            prediction = load_numeric_npz(prediction_dir / record["path"])
            derived = _derive_one(panel, prediction, int(prediction["bank_index"]))
            path = derived_dir / record["path"].replace("pred_", "derived_")
            write_numeric_npz_atomic(path, **derived)
            derived_bytes += path.stat().st_size
        derive_wall = time.perf_counter() - derive_started

        panel_bytes = panel_path.stat().st_size
        panel_bytes += panel_path.with_suffix(".json").stat().st_size
        prediction_shard_bytes = sum(
            (prediction_dir / record["path"]).stat().st_size
            for record in prediction_ledger["records"]
        )
        prediction_metadata_bytes = (
            (prediction_dir / "prediction_ledger.json").stat().st_size
            + (run_dir / "pre_score_guard.json").stat().st_size
            + (run_dir / "score_progress.json").stat().st_size
        )
        prediction_bytes = prediction_shard_bytes + prediction_metadata_bytes
        replay_bundle_path = run_dir / "projected-source.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(replay_bundle_path), "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        replay_bundle_bytes = replay_bundle_path.stat().st_size
        tracked_output = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_paths = [
            item.decode("utf-8") for item in tracked_output.split(b"\0") if item
        ]
        tracked_repository_bytes = sum(
            (root / relative).stat().st_size for relative in tracked_paths
        )

        group_ratio = 256 / 8
        shard_ratio = 64 / len(prediction_ledger["records"])
        smoke_guard_records = len(prediction_ledger["golden"]["records"])
        guard_ratio = 128 / smoke_guard_records
        inference_ratio = group_ratio * shard_ratio
        projected_panel_wall = panel_metadata["wall_seconds"] * group_ratio * 1.25
        guard_wall = float(prediction_ledger["golden_wall_seconds"])
        non_guard_score_wall = prediction_ledger["wall_seconds"] - guard_wall
        if guard_wall <= 0 or non_guard_score_wall <= 0:
            raise AssertionError("smoke score timing components must be positive")
        projected_guard_wall = guard_wall * guard_ratio * 1.25
        projected_inference_wall = non_guard_score_wall * inference_ratio * 1.25
        projected_score_wall = projected_guard_wall + projected_inference_wall
        projected_derive_wall = derive_wall * inference_ratio * 1.25
        projected_wall = (
            projected_panel_wall + projected_score_wall + projected_derive_wall
        )
        projected_raw = int(
            1.25
            * (
                panel_bytes * group_ratio
                + prediction_shard_bytes * inference_ratio
                + prediction_metadata_bytes * guard_ratio
                + derived_bytes * inference_ratio
                + replay_bundle_bytes
                + tracked_repository_bytes
                + 10 * 2**20
            )
        )
        projected_peak = int(
            max(
                _peak_rss_bytes() * 4,
                panel_bytes * group_ratio * 3,
                512 * 2**20,
            )
        )
        result = {
            "schema_version": 1,
            "producer_sha256": sha256_file(Path(__file__)),
            "implementation_sha256s": package_source_hashes(),
            "input_sha256s": validation_input_hashes(),
            "smoke_stream_label": "smoke-v3",
            "smoke_kind": "interior-selected-end-to-end-v3",
            "smoke_commit": SMOKE_COMMIT,
            "smoke_groups": 8,
            "smoke_continuations": 8,
            "max_core_candidates": 2000,
            "max_blocks_per_core": 512,
            "min_within_group_sd": 0.25,
            "candidate_core_count": panel_metadata["candidate_core_count"],
            "candidate_block_count": panel_metadata["candidate_block_count"],
            "eligible_replace_rows": int(panel["eligible_replace"].sum()),
            "prediction_shards": len(prediction_ledger["records"]),
            "guard_records": smoke_guard_records,
            "guard_sample_source": prediction_ledger["golden"]["sample_source"],
            "projected_scientific_guard_records": 128,
            "components": {
                "panel_wall_seconds": panel_metadata["wall_seconds"],
                "score_wall_seconds": prediction_ledger["wall_seconds"],
                "guard_wall_seconds": guard_wall,
                "non_guard_score_wall_seconds": non_guard_score_wall,
                "derive_wall_seconds": derive_wall,
                "panel_bytes": panel_bytes,
                "prediction_shard_bytes": prediction_shard_bytes,
                "prediction_metadata_bytes": prediction_metadata_bytes,
                "prediction_bytes": prediction_bytes,
                "derived_bytes": derived_bytes,
                "replay_bundle_bytes": replay_bundle_bytes,
                "tracked_repository_bytes": tracked_repository_bytes,
                "measured_peak_rss_bytes": _peak_rss_bytes(),
            },
            "projection_multipliers": {
                "group_ratio": group_ratio,
                "shard_ratio": shard_ratio,
                "guard_ratio": guard_ratio,
                "safety_factor": 1.25,
            },
            "projected_wall_seconds": projected_wall,
            "projected_peak_rss_bytes": projected_peak,
            "projected_raw_bytes": projected_raw,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "raw_cap_bytes": RAW_CAP_BYTES,
            "measurement_wall_seconds": time.perf_counter() - started,
        }
        result["pass"] = bool(
            result["eligible_replace_rows"] == 64
            and result["prediction_shards"] == 4
            and projected_wall <= WALL_CAP_SECONDS
            and projected_peak <= MEMORY_CAP_BYTES
            and projected_raw <= RAW_CAP_BYTES
        )
    write_json_atomic(out_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_smoke(args.out)
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
