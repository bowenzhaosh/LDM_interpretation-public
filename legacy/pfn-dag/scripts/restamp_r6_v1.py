"""Re-stamp the round-1 R6 artifacts DIAGNOSTIC. They are stamped live and should not be.

The pre-run audit proved that the build which wrote them published
`substitution_control_max_dev: 0.0` from an expression that returns 0.0 for every
input, including deliberately corrupted ones. The eta values themselves are sound
and the v2 re-run reproduces them (0.6706 against 0.6706 at d=2; 0.19998 against
0.2000 at d=3), so this is not a retraction of the measurement. It is that an
artifact whose published control means nothing must not carry a tag that tells a
fail-closed reader to trust it. F.8's discipline is writers stamp and readers
require; a live tag on a vacuous control is exactly the failure that discipline
exists to prevent.

The files are kept rather than deleted: they are the record of what was run, and
the supersession note travels inside them.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

NOTE = ("Re-stamped diagnostic 2026-08-24. Written by a build whose "
        "substitution_control_max_dev was an algebraic tautology returning 0.0 for "
        "any input, so the control this file publishes is vacuous. The eta value is "
        "reproduced by the v2 build (see v2_* in the same directory); use those. "
        "Superseded, not retracted.")


def main(argv=None) -> int:
    """usage: restamp_r6_v1.py [DIR] [GLOB] [NOTE]

    Two generations of eta artifacts stamped live at once would be pooled by any
    reader that groups by cell, and every model seed would be counted twice. Exactly
    one generation stays live; the rest are stamped diagnostic with a note saying
    what superseded them and why.
    """
    argv = list(argv or [])
    root = Path(argv[0]) if argv else Path("campaigns/binary_duel_ext3_20260821/results/r6")
    pattern = argv[1] if len(argv) > 1 else "r6_d*.json"
    note = argv[2] if len(argv) > 2 else NOTE
    n = 0
    for f in sorted(root.glob(pattern)):
        d = json.loads(f.read_text())
        art = d.get("artifact", {})
        if art.get("status") == "diagnostic":
            continue
        art["status"] = "diagnostic"
        art["superseded_note"] = note
        d["artifact"] = art
        f.write_text(json.dumps(d, indent=2))
        n += 1
        print(f"  diagnostic <- {f.name}")
    print(f"re-stamped {n} artifacts matching {pattern!r} under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
