"""Amendment F.4 — is the eps=0.5 ladder move D1's effect, or a code-version change?

The committed audit records that the published pre-fix Q1 width at eps=0.5
(0.2073) came from one of two IPM runs, that the config block did not record the
``crossover`` option, and that the stored runtime (689.4 s) matches the run made
BEFORE the code gained ``crossover: off``. So the +0.0535 post-fix move on that
rung may mix the D1 mixture-granularity fix with a solver-configuration change,
and that rung is the one whose movement flattens the top of the ladder.

This runs the full 2x2 rather than assuming the two effects are additive:

                        crossover: current      crossover: legacy
    SEM post-fix (D1)   already measured        cell 2
    SEM pre-fix         cell 3                  cell 4

Cell 4 is the diagnostic: if it reproduces 0.2073, the audit's reading is right
and the published number came from the legacy solver configuration.

The pre-fix sampler is applied as a RUNTIME PATCH, recovered verbatim from
9eca9ac9 (the HEAD pinned in AMENDMENT_E.0). The shipped ``corrected_sem`` is
never edited: the point is to measure the old instrument, not to reintroduce it.

Whichever way this lands, the PAPER cites only the post-fix current-solver
ladder. This feeds the errata narrative and our own calibration, and stays off
the retrain's critical path.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pfn_dag_verify import corrected_sem as SEM
from pfn_dag_verify.corrected_identifiability_run import run


def _prefix_sample_residuals(rng, b, spec, shape):
    """``sample_residuals`` exactly as it stood at 9eca9ac9.

    The ONLY difference from the current implementation is the mask shape:
    ``rng.random(tuple(shape) + (1,))`` shares one Bernoulli draw across the d
    coordinates of a row, so the sampled law is the per-ROW mixture rather than
    the per-COORDINATE mixture that ``residual_logpdf`` scores against. Both
    endpoints short-circuit before the mask, which is why eps=0 and eps=1 were
    never affected.
    """
    b = np.asarray(b, dtype=np.float64)
    d = b.shape[0]
    full_shape = tuple(shape) + (d,)
    eps = spec.eps
    if eps <= 0.0:
        return rng.normal(0.0, math.sqrt(2.0) * b[None, :], full_shape)
    a, c = SEM._al_ac(b, spec.r)
    al = (rng.exponential(a[None, :], full_shape)
          - rng.exponential(c[None, :], full_shape) - (a - c)[None, :])
    if eps >= 1.0:
        return al
    gauss = rng.normal(0.0, math.sqrt(2.0) * b[None, :], full_shape)
    mask = rng.random(tuple(shape) + (1,)) < eps
    return np.where(mask, al, gauss)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sem", choices=["postfix", "prefix"], required=True)
    p.add_argument("--crossover", choices=["current", "legacy"], required=True)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--solver", type=str, default="highs-ipm")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    original = SEM.sample_residuals
    if a.sem == "prefix":
        # generate_observational resolves sample_residuals as a module global at
        # call time, so patching the attribute is sufficient and is undone below.
        SEM.sample_residuals = _prefix_sample_residuals
    try:
        t0 = time.time()
        res = run(d=3, K=8, eps=a.eps, n_contexts=20, out=a.out, tol=1e-3,
                  seed=990_200_000, solver=a.solver,
                  crossover=None if a.crossover == "current" else "legacy")
        wall = time.time() - t0
    finally:
        SEM.sample_residuals = original

    q1 = res["aggregates"]["Q1"]["avg_width_mean"]
    # Re-stamp the artifact with the cell's identity so a reader can never
    # mistake a diagnostic run for a headline ladder measurement.
    doc = json.loads(a.out.read_text())
    doc["config"]["f4_cell"] = {"sem": a.sem, "crossover": a.crossover,
                                "amendment": "F.4",
                                "headline": False,
                                "note": "attribution diagnostic; NOT the cited ladder"}
    a.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps({"sem": a.sem, "crossover": a.crossover, "eps": a.eps,
                      "solver": a.solver, "Q1_avg_width_mean": q1,
                      "wall_s": round(wall, 1), "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
