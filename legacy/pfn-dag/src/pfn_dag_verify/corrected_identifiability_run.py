"""Track A2 + B3-exact driver — tiny exact identifiability analysis.

For one world (d, K, eps): over fixed exact contexts, computes
  - exact order-posterior entropy,
  - Q1/Q2/Q3 query-operator rank / effective rank / conditioning,
  - identified-set per-order probability ranges (L1 LP) and their widths,
  - whether the true order posterior lies inside each identified set,
and aggregates over contexts. Also records the exact quantities the epsilon
continuum needs (order entropy, structural info, conditioning, widths).

Run:  PYTHONPATH=src python -m pfn_dag_verify.corrected_identifiability_run \
          --d 3 --K 8 --eps 1.0 --n-contexts 20 \
          --out campaigns/corrected_20260812/raw/identifiability_eps1.0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .artifact_status import EST_IDENT_WIDTH, stamp
from .corrected_identifiability import analyze_world, _world_summary
from .corrected_world import make_world


# Amendment F.5, clause 2. The headline ident n_rows is DEFINED as Track B's
# panel n_rows, not merely declared to match it: if the two differ, the width
# appendix's x and the model-side y sit at different information levels and the
# comparison is invalid. The value is duplicated here rather than imported
# because corrected_trackb pulls in torch and this is a CPU-only LP job;
# tests/test_f5_n_rows.py enforces that the two never drift apart.
HEADLINE_N_ROWS = 20


def run(d: int, K: int, eps: float, n_contexts: int, out: Path, tol: float,
        seed: int, solver: str = "highs", n_rows: int = HEADLINE_N_ROWS,
        crossover: str | None = None) -> dict:
    world = make_world(k=K, d=d, eps=eps)
    t0 = time.time()
    res = analyze_world(world, n_contexts=n_contexts, n_rows=n_rows, seed=seed, tol=tol,
                        solver=solver)
    res["world"] = _world_summary(world)
    res["runtime_s"] = float(time.time() - t0)
    res["config"] = {"d": d, "K": K, "eps": eps, "n_contexts": n_contexts,
                     "tol": tol, "seed": seed, "solver": solver,
                     # F.5 clause 1: n_rows is recorded from here on. Its absence
                     # is why nothing on disk distinguished the ident set from
                     # the nrows10 set except the filename.
                     "n_rows": n_rows,
                     "crossover": crossover or "default"}
    # F.8: a diagnostic re-run of a retired configuration must not be readable
    # as a live ladder measurement.
    stamp(res, EST_IDENT_WIDTH, "diagnostic" if crossover == "legacy" else "live")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--n-contexts", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=990_200_000)
    p.add_argument("--n-rows", type=int, default=HEADLINE_N_ROWS,
                   help="context rows; F.5 defines the headline value as Track B's panel n_rows")
    p.add_argument("--crossover", type=str, default=None, choices=[None, "legacy"],
                   help="F.4 attribution experiment only; default reproduces current behaviour")
    p.add_argument("--solver", type=str, default="highs",
                   choices=["highs", "highs-ipm", "highs-ds"],
                   help="scipy LP solver method (highs = dual simplex, highs-ipm = "
                        "interior point). highs-ipm avoids degenerate dual-simplex "
                        "cycling in the mixed-noise regime (eps~0.5).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    run(args.d, args.K, args.eps, args.n_contexts, args.out, args.tol, args.seed,
        args.solver, n_rows=args.n_rows, crossover=args.crossover)
    print(json.dumps({"done": True, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
