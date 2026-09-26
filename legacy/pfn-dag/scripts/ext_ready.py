"""EXT harvester helper: which (cell, seeds) groups are fully trained and not yet scored.

Reads .claude/ext_jobs.tsv (ts, jid, mode, cell, eps, seeds, tlimit, lane|kind) and a
sacct dump (jid|State per line, from stdin). A training group = all ledger rows with
the same (cell, seeds) and a training mode (anything but 'score'). It is READY when
every one of its jobs is COMPLETED and no 'score' row exists for the same (cell, seeds,
eps). Prints one line per ready group: cell<TAB>eps list<TAB>seeds<TAB>kind.
Groups with a FAILED/TIMEOUT/CANCELLED job are printed to stderr once and skipped.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".claude/ext_jobs.tsv"
TERMINAL_BAD = ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY")


def main() -> int:
    states = {}
    for line in sys.stdin:
        if "|" in line:
            j, s = line.strip().split("|")[:2]
            states[j] = s.split()[0]
    train = defaultdict(list)
    scored = set()
    for line in LEDGER.read_text().splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        ts, jid, mode, cell, eps, seeds, tlimit, extra = f[:8]
        if mode == "score":
            for e in eps.split():
                scored.add((cell, seeds, e))
        else:
            train[(cell, seeds)].append((jid, eps))
    for (cell, seeds), jobs in sorted(train.items()):
        st = [states.get(j, "UNKNOWN") for j, _ in jobs]
        if any(s in TERMINAL_BAD for s in st):
            bad = [(j, s) for (j, _), s in zip(jobs, st) if s in TERMINAL_BAD]
            print(f"SKIP {cell} seeds='{seeds}': {bad}", file=sys.stderr)
            continue
        if not all(s == "COMPLETED" for s in st):
            continue
        eps_todo = sorted({e for _, e in jobs if (cell, seeds, e) not in scored}, key=float)
        if not eps_todo:
            continue
        kind = "n40ext" if "_n40_" in cell else "orig"
        print(f"{cell}\t{' '.join(eps_todo)}\t{seeds}\t{kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
