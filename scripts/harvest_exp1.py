"""Harvest exp1 results from WashU without overwriting the committed record.

Written after a plain scp overwrote four committed artifacts. They had never
existed on the cluster, so exp1's --resume did not skip them, the fleet re-scored
them on a different eval seed (101 -> 820), and the sync then replaced the record
with the twin. The values agreed to within noise, which is exactly why it would
have gone unnoticed.

The rule here: a file that is tracked by git and would change is NOT overwritten.
It is fetched to a .incoming twin and reported, so a real disagreement surfaces as
a disagreement instead of as a silent update, and an eval-seed twin surfaces as a
twin instead of as a correction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REMOTE = ("washu:/engrfs/project/class/zhao.b/pfn-dag-e18b/"
          "experiments/binary-duel/results/exp1")
LOCAL = Path("campaigns/binary_duel_ext3_20260821/results/exp1")


def tracked(path: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0


def committed_bytes(path: Path) -> bytes | None:
    r = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default=".harvest_stage",
                    help="scratch directory the remote copy lands in first")
    ap.add_argument("--apply", action="store_true",
                    help="move new files into place (conflicts are never moved)")
    a = ap.parse_args(argv)

    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=250",
                         f"{REMOTE}/*.json", str(stage)]).returncode
    if rc != 0:
        print(f"scp failed with {rc}", file=sys.stderr)
        return rc

    new, same, conflict = [], [], []
    for f in sorted(stage.glob("*.json")):
        dest = LOCAL / f.name
        if not dest.exists():
            new.append(f)
            continue
        if f.read_bytes() == dest.read_bytes():
            same.append(f)
            continue
        # differs. Is the local copy the committed record?
        if tracked(dest) and committed_bytes(dest) == dest.read_bytes():
            conflict.append((f, dest))
        else:
            new.append(f)          # local is itself an un-committed harvest; refresh it

    print(f"{len(new)} new/refreshable, {len(same)} identical, {len(conflict)} CONFLICT")
    for f, dest in conflict:
        try:
            a_ = json.loads(committed_bytes(dest))
            b_ = json.loads(f.read_text())
            print(f"  CONFLICT {dest.name}: committed eval_seed "
                  f"{a_.get('eval_seed')} vs incoming {b_.get('eval_seed')}; "
                  f"theil_slope {a_['e1a_natural']['reg_w']['theil_slope']:.6f} vs "
                  f"{b_['e1a_natural']['reg_w']['theil_slope']:.6f}")
        except Exception:
            print(f"  CONFLICT {dest.name}: differs, could not parse for a summary")
        keep = dest.with_suffix(".json.incoming")
        keep.write_bytes(f.read_bytes())
        print(f"           incoming copy kept at {keep.name}, record untouched")

    if a.apply:
        for f in new:
            (LOCAL / f.name).write_bytes(f.read_bytes())
        print(f"applied {len(new)} files; {len(conflict)} conflicts left for a human")
    else:
        print("dry run. Re-run with --apply to move the new files into place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
