#!/usr/bin/env python3
"""Harvest corrected-campaign results from WashU and report job status.

Usage:
  python3 cluster/harvest_corrected.py [--status] [--fetch]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CLUSTER = "washu"
REMOTE_ROOT = "/engrfs/project/class/zhao.b/pfn-dag-corrected"
LOCAL_RAW = Path(__file__).resolve().parents[1] / "campaigns/corrected_20260812/raw"


def _ssh(*args: str) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", CLUSTER, *args],
                       check=True, capture_output=True, text=True)
    return r.stdout


def status() -> None:
    out = _ssh(f"squeue -u $USER -o '%.8i %.10P %.16j %.2t %.8M' 2>/dev/null")
    print(out)
    print("remote raw dir:")
    print(_ssh(f"ls -la {REMOTE_ROOT}/campaigns/corrected_20260812/raw/ 2>/dev/null"))


def fetch() -> None:
    subprocess.run([
        "rsync", "-az",
        f"{CLUSTER}:{REMOTE_ROOT}/campaigns/corrected_20260812/raw/",
        f"{str(LOCAL_RAW)}/",
    ], check=True)
    print(f"fetched -> {LOCAL_RAW}")
    for f in sorted(LOCAL_RAW.glob("*.json")):
        print(f"  {f.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status", action="store_true")
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    if args.status:
        status()
    if args.fetch:
        fetch()
    if not args.status and not args.fetch:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
