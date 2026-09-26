"""Amendment F.2b — produce the PRIMARY x-axis artifact.

Writes ``deficit_eps{tag}.json``, the panel mean of the normalized
order-marginal entropy deficit ``1 - H(p(o|D))/log O``, evaluated on the SAME
eval panel the PFN is scored on. ``corrected_verdict.cells_from_raw`` reads
these; without them the primary association has no coordinates and the verdict
is unconditionally NOT_EVALUABLE.

Three things this file is careful about, each of which produces a plausible
wrong number if got wrong:

0. WHICH HALF. F.2d.5 splits the cell's panel in two and this file reads HALF A
   only, so no context here contributes to the readouts that supply y. The
   half index and both hashes are recorded rather than re-derived.

1. WHICH PANEL. ``split_panel.build_split_panel`` is the only supported source. The
   classifier panel (seed 771000000) and the tomography panel (772000000) are
   the same shape and would return a number that looks right: measured at
   eps=0.5 the classifier panel gives a materially different deficit from the
   eval panel. The panel seed and a SHA-256 of the stacked context array go in
   the artifact so a mixed-panel ladder cannot be assembled unnoticed.

2. DOUBLE COUNTING. ``make_eval_panel`` appends the same context object once per
   query, so averaging over ``[q[0] for q in panel]`` counts each context
   n_query_per_context times. The mean survives, the SE does not: it comes out a
   factor sqrt(n_query_per_context) too small, and every near-tie z inflates by
   the same factor. The deduplicating helper is used for exactly this reason.

3. WEIGHTING. The average is over the DISTINCT CONTEXTS, uniformly. This is the
   preregistered choice, on the ground that the E.5 tracking statistic is a
   per-context quantity, so a context-level average is what matches y. A
   query-weighted average over panel entries is identical here (the multiplicity
   is uniform); an average over only the in-task query subset is NOT, and is
   explicitly not used. ``weighting`` records the choice in the artifact.

n_rows is recorded and is required to equal Track B's panel n_rows: F.5 clause 2
defines them to be the same number, and n_rows is the dominant confound for this
estimand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifact_status import EST_ORDER_EVIDENCE, stamp
from .corrected_identifiability_run import HEADLINE_N_ROWS
from .corrected_oracle import panel_order_entropy_deficit
from .corrected_world import make_world
from .split_panel import HALF_A, SPLIT_SEED, build_split_panel

# Track B's eval panel, verbatim (corrected_trackb: N_EVAL_CONTEXTS,
# N_QUERY_PER_CONTEXT and the 770-series seed). Duplicated rather than imported
# because corrected_trackb pulls in torch and this is a CPU-only job;
# tests/test_f5_n_rows.py enforces that they never drift apart.
PANEL_SEED = 770_000_000
N_EVAL_CONTEXTS = 60          # E-era single panel; superseded as the HEADLINE
                              # count by F.2's sizing rule, which is per HALF
N_PER_HALF = 500              # F.2's floor, per half (F.2d.5 draws 2n)
N_QUERY_PER_CONTEXT = 2
WEIGHTING = "uniform_over_distinct_contexts"


def panel_sha256(contexts) -> str:
    arr = np.ascontiguousarray(np.stack(contexts), dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def run(d: int, K: int, eps: float, out: Path,
        n_rows: int = HEADLINE_N_ROWS,
        n_per_half: int = N_PER_HALF,
        panel_seed: int = PANEL_SEED,
        split_seed: int = SPLIT_SEED,
        n_query_per_context: int = N_QUERY_PER_CONTEXT) -> dict:
    if n_rows != HEADLINE_N_ROWS:
        raise ValueError(
            f"n_rows={n_rows} != headline {HEADLINE_N_ROWS}. F.5 clause 2 defines "
            "the deficit panel's n_rows as Track B's panel n_rows; a mismatch "
            "moves a cell a full rung on the dial while looking plausible.")
    world = make_world(k=K, d=d, eps=eps)
    # F.2d.5: HALF A only. The readouts never see these contexts.
    panel = build_split_panel(world, n_per_half, n_rows, panel_seed,
                              split_seed=split_seed,
                              n_query_per_context=n_query_per_context)
    contexts = panel.half(HALF_A)
    res = panel_order_entropy_deficit(world, contexts)
    doc = {
        **res,
        **panel.provenance(HALF_A),
        "eps": float(eps), "d": int(d), "K": int(K), "O": int(world.O),
        "weighting": WEIGHTING,
        "amendment": "F.2b/F.2d.5",
        "axis": "primary",
    }
    stamp(doc, EST_ORDER_EVIDENCE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--n-rows", type=int, default=HEADLINE_N_ROWS)
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--panel-seed", type=int, default=PANEL_SEED)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    doc = run(a.d, a.K, a.eps, a.out, n_rows=a.n_rows, n_per_half=a.n_per_half,
              panel_seed=a.panel_seed, split_seed=a.split_seed)
    print(json.dumps({k: doc[k] for k in
                      ("eps", "deficit_mean", "deficit_se", "n_contexts",
                       "half", "n_rows", "panel_sha256", "half_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
