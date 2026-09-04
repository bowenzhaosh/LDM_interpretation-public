#!/usr/bin/env python3
"""Submit the sharded oracle-precision pilot to the WashU cluster.

Syncs the frozen repository to the cluster (excluding .git and bulky artifacts)
and submits one Slurm job per row shard on the owned condo-cse5100 partition.
Records every job ID in a SUBMISSION.json under the attempt root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import os


CLUSTER = "washu"
REMOTE_ROOT = "/engrfs/project/class/zhao.b/pfn-dag-oracle-precision-pilot-v1"
CONDO = "condo-cse5100"
ACCOUNT = "engr-acad-cse5100"
N_SHARDS = 40
N_ROWS = 400
SBATCH = "cluster/oracle_precision_pilot.sbatch"


def _ssh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", CLUSTER, *args],
                          check=check, capture_output=True, text=True)


def _rsync(local: Path, remote: str) -> None:
    cmd = [
        "rsync", "-az", "--delete", "--exclude=.git", "--exclude=.pytest_cache",
        "--exclude=.ruff_cache", "--exclude=artifacts/checkpoints",
        "--exclude=campaigns/**/checkpoints",
        "--exclude=run",  # never wipe a live run's outputs on re-sync
        str(local) + "/", f"{CLUSTER}:{remote}/",
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--partition", default=CONDO)
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--validate-only", action="store_true",
                        help="sync and run a 1-row smoke, then stop")
    args = parser.parse_args()
    repo = args.repo.resolve()
    _rsync(repo, REMOTE_ROOT)
    print(f"synced repo to {CLUSTER}:{REMOTE_ROOT}", flush=True)
    if args.validate_only:
        cmd = (f"sbatch --partition={args.partition} --account={ACCOUNT} --gres=gpu:1 --cpus-per-task=8 "
               f"--mem=48G --time={args.time} --parsable {REMOTE_ROOT}/{SBATCH} "
               f"--source-root {REMOTE_ROOT} --out-root {REMOTE_ROOT}/run --row-start 0 --row-count 1")
        out = _ssh(cmd)
        print("validation job:", out.stdout.strip(), flush=True)
        return 0
    rows_per = args.rows // args.n_shards
    records = {"schema_version": 1, "n_shards": args.n_shards,
               "rows_per_shard": rows_per, "job_ids": []}
    for s in range(args.n_shards):
        rs = s * rows_per
        rc = rows_per if s < args.n_shards - 1 else args.rows - rs
        cmd = (f"sbatch --partition={args.partition} --account={ACCOUNT} --gres=gpu:1 --cpus-per-task=8 "
               f"--mem=48G --time={args.time} --parsable {REMOTE_ROOT}/{SBATCH} "
               f"--source-root {REMOTE_ROOT} --out-root {REMOTE_ROOT}/run "
               f"--row-start {rs} --row-count {rc}")
        out = _ssh(cmd)
        jid = out.stdout.strip()
        records["job_ids"].append({"shard": s, "row_start": rs, "row_count": rc, "job_id": jid})
        print(f"submitted shard {s} (rows {rs}..{rs+rc}): {jid}", flush=True)
    _ssh(f"mkdir -p {REMOTE_ROOT}")
    records_path = f"{REMOTE_ROOT}/run/SUBMISSION.json"
    _ssh(f"mkdir -p {REMOTE_ROOT}/run")
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", CLUSTER, f"cat > {records_path}"],
        input=json.dumps(records, indent=2), text=True, check=True)
    print(f"submission record: {CLUSTER}:{records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
