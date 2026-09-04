"""Amendment E / D2 -- compare the post-fix identified-set ladder across
solvers and across the two execution environments.

Three questions, kept separate because they fail for different reasons:

  1. Does the dial survive D1?  The Q1 ladder must stay strictly monotone in
     eps and the exact set must still cover the truth in every cell.
  2. Solver inhomogeneity (the unresolved finding in the committed audit):
     within one environment, does highs agree with highs-ipm?
  3. Environment invariance: the Mac ran scipy 1.12.0 / numpy 1.26.4 and WashU
     runs scipy 1.15.3 / numpy 2.2.6, which are different HiGHS builds under a
     preregistered LP instrument.  Same cell, same config, two builds -- do
     they agree?

Nothing here decides anything.  It prints the comparison; the ruling on whether
an agreement is close enough is a preregistration question, not a script's.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "campaigns/corrected_20260812/raw/postE"
WASHU = REPO / "campaigns/corrected_20260812/raw/postE_washu"

# AMENDMENT_E.2, recorded before the D1 fix so the regeneration is a comparison
# and not a silent replacement.
PRE_FIX_Q1 = {0.1: 0.4510, 0.25: 0.3232, 0.5: 0.2073}
# Endpoints short-circuit the mixture, so D1 cannot move them; carried from the
# committed ladder purely to show monotonicity across the full dial.
ENDPOINTS = {0.0: 1.0000, 1.0: 0.1255}

CELLS = [(0.1, "0p1"), (0.25, "0p25"), (0.5, "0p5")]
SOLVERS = [("highs", "highs"), ("highs-ipm", "highs_ipm")]


def load(root: Path, tag: str, stag: str) -> dict | None:
    p = root / f"ident_eps{tag}_{stag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def env_of(root: Path, tag: str, stag: str) -> str:
    p = root / f"ident_eps{tag}_{stag}.env.json"
    if p.exists():
        e = json.loads(p.read_text())
        return f"numpy {e['numpy']} / scipy {e['scipy']} @ {e['host'].split('.')[0]}"
    return "numpy 1.26.4 / scipy 1.12.0 @ mac (no sidecar; pre-sbatch run)"


def q1(d: dict) -> float:
    return d["aggregates"]["Q1"]["avg_width_mean"]


def main() -> int:
    print("=" * 78)
    print("Q1 avg_width_mean -- post-fix ladder vs the pre-fix values in AMENDMENT_E.2")
    print("=" * 78)
    print(f"{'eps':>6} {'pre-fix':>10} {'mac highs':>12} {'mac ipm':>12} "
          f"{'washu highs':>13} {'washu ipm':>12}")
    for eps, tag in CELLS:
        row = [f"{eps:>6.2f}", f"{PRE_FIX_Q1[eps]:>10.4f}"]
        for root in (MAC, WASHU):
            for sname, stag in SOLVERS:
                d = load(root, tag, stag)
                row.append(f"{q1(d):>12.6f}" if d else f"{'--':>12}")
        print(" ".join(row))

    print()
    print("=" * 78)
    print("(1) dial survives D1?")
    print("=" * 78)
    best: dict[float, float] = dict(ENDPOINTS)
    for eps, tag in CELLS:
        for root in (WASHU, MAC):           # prefer the cluster ladder when present
            for _, stag in SOLVERS:
                d = load(root, tag, stag)
                if d and eps not in best:
                    best[eps] = q1(d)
    ladder = [(e, best[e]) for e in sorted(best)]
    print("  ladder: " + "  ".join(f"{e:g}:{w:.4f}" for e, w in ladder))
    widths = [w for _, w in ladder]
    mono = all(a > b for a, b in zip(widths, widths[1:]))
    print(f"  strictly monotone decreasing in eps: {mono}")

    cover_fail = []
    for root, lbl in ((MAC, "mac"), (WASHU, "washu")):
        for eps, tag in CELLS:
            for sname, stag in SOLVERS:
                d = load(root, tag, stag)
                if not d:
                    continue
                for regime in ("Q1", "Q2", "Q3"):
                    f = d["aggregates"][regime]["true_in_all_frac"]
                    if f != 1.0:
                        cover_fail.append(f"{lbl}/{sname}/eps{eps}/{regime}={f}")
    print(f"  true_in_all_frac == 1.0 in every cell and regime: {not cover_fail}")
    if cover_fail:
        print("  COVERAGE FAILURES: " + ", ".join(cover_fail))

    print()
    print("=" * 78)
    print("(2) solver inhomogeneity, WITHIN environment")
    print("=" * 78)
    for root, lbl in ((MAC, "mac"), (WASHU, "washu")):
        for eps, tag in CELLS:
            a, b = load(root, tag, "highs"), load(root, tag, "highs_ipm")
            if not (a and b):
                print(f"  {lbl:>5} eps={eps:<5g} incomplete pair")
                continue
            qa, qb = q1(a), q1(b)
            rel = abs(qa - qb) / max(abs(qa), 1e-300)
            print(f"  {lbl:>5} eps={eps:<5g} highs={qa:.8f} ipm={qb:.8f} rel={rel:.2e}")

    print()
    print("=" * 78)
    print("(3) environment invariance, SAME cell and solver, two HiGHS builds")
    print("=" * 78)
    for eps, tag in CELLS:
        for sname, stag in SOLVERS:
            a, b = load(MAC, tag, stag), load(WASHU, tag, stag)
            if not (a and b):
                print(f"  eps={eps:<5g} {sname:<10} incomplete pair")
                continue
            qa, qb = q1(a), q1(b)
            rel = abs(qa - qb) / max(abs(qa), 1e-300)
            print(f"  eps={eps:<5g} {sname:<10} mac={qa:.8f} washu={qb:.8f} rel={rel:.2e}")

    print()
    print("environments:")
    print(f"  mac   : {env_of(MAC, '0p1', 'highs')}")
    print(f"  washu : {env_of(WASHU, '0p1', 'highs')}")
    print()
    print("runtimes (s):")
    for root, lbl in ((MAC, "mac"), (WASHU, "washu")):
        for eps, tag in CELLS:
            for sname, stag in SOLVERS:
                d = load(root, tag, stag)
                if d:
                    print(f"  {lbl:>5} eps={eps:<5g} {sname:<10} {d.get('runtime_s', -1):>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
