import argparse
import json
from pathlib import Path

import torch

from .constants import BASE_MODEL_CONFIG
from .model import state_schema
from .registry import sha256_file


ORIGINAL_SOURCES = (
    "e21_fleet.py",
    "experiment_v3bump.py",
    "d5c_gate0.py",
    "d5c_analyze.py",
)


def _record_path(path: Path, repository: Path | None) -> tuple[str, str | None]:
    if repository is None:
        return str(path.resolve()), None
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    return relative, f"repo://{relative}"


def build_registry(
    checkpoint_dir: Path, source_dir: Path, repository: Path | None = None
):
    checkpoints = []
    common_schema = None
    for seed in range(16):
        for step in (0, 12_000):
            path = checkpoint_dir / f"M_base_AL40_s{seed}_dose{step}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            schema = state_schema(state)
            if common_schema is None:
                common_schema = schema
            elif schema != common_schema:
                raise ValueError(f"state schema differs for {path}")
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f"non-finite checkpoint tensor in {path}")
            registered_path, durable_uri = _record_path(path, repository)
            checkpoints.append(
                {
                    "seed": seed,
                    "step": step,
                    "filename": path.name,
                    "path": registered_path,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "durable_uri": durable_uri,
                }
            )
    sources = []
    for filename in ORIGINAL_SOURCES:
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        registered_path, durable_uri = _record_path(path, repository)
        sources.append(
            {
                "filename": filename,
                "path": registered_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "durable_uri": durable_uri,
            }
        )
    return {
        "schema_version": 1,
        "fleet": "base_AL40_dose",
        "model_config": BASE_MODEL_CONFIG,
        "training_config": {
            "prior": "AL40",
            "context_rows": 30,
            "query_rows": 7,
            "peak_learning_rate": 0.001,
            "warmup_steps": 800,
            "max_step": 12000,
            "training_data": "online synthetic Fix-B AL40 generator; exact stream lineage unavailable",
        },
        "state_schema": common_schema,
        "original_sources": sources,
        "checkpoints": checkpoints,
        "provenance_limit": (
            "The files are content-addressed, but no immutable upstream release or complete "
            "training-stream ledger was available. Claims apply to these exact hashes."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = build_registry(args.checkpoint_dir, args.source_dir, args.repository)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checkpoints": len(registry["checkpoints"]),
                "schema_keys": len(registry["state_schema"]),
                "total_bytes": sum(item["size"] for item in registry["checkpoints"]),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
