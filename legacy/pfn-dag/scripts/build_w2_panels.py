"""Amendment F.2d.5 / T6 — build and record the W2 half-panels.

Fourteen half-panels: seven eps, each drawn as 2 x 500 iid contexts from its own
world prior and split into fixed disjoint halves. Per eps the panel is FIXED
across seeds and scales, so every model in a cell is scored on identical points
and the deficit is computed on contexts no readout ever sees.

The contexts themselves are not written to disk. They are a deterministic
function of (eps, d, K, n_rows, n_per_half, panel_seed, split_seed,
n_query_per_context), and a manifest of hashes is both smaller and a stronger
check than a copy of the arrays would be: a copy proves what was stored, a hash
proves what regenerates. Determinism is verified here rather than asserted --
each panel is built twice and the two hashes compared.

The manifest records the half indices in full. F.2d.5 requires the assignment to
be RECORDED rather than re-derived at read time, because a split that has to be
recomputed to be checked is one that can silently change when the code around it
changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pfn_dag_verify.artifact_status import LIVE, stamp                       # noqa: E402
from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED      # noqa: E402
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS     # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag                         # noqa: E402
from pfn_dag_verify.corrected_world import make_world                        # noqa: E402
from pfn_dag_verify.split_panel import (                                     # noqa: E402
    HALF_A, HALF_B, SPLIT_SEED, build_split_panel,
)

# F.6r: 0.35 and 0.75 join the ladder. eps=0 is drawn and scored -- it carries
# the null-type claim -- but F.2c takes it out of the fidelity series by
# construction, so it never enters the association.
EPS_GRID = [0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-rows", type=int, default=HEADLINE_N_ROWS)
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--panel-seed", type=int, default=PANEL_SEED)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--n-query-per-context", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    t0 = time.time()
    panels = []
    for e in EPS_GRID:
        world = make_world(k=a.K, d=a.d, eps=e)
        sp = build_split_panel(world, a.n_per_half, a.n_rows, a.panel_seed,
                               split_seed=a.split_seed,
                               n_query_per_context=a.n_query_per_context)
        again = build_split_panel(world, a.n_per_half, a.n_rows, a.panel_seed,
                                  split_seed=a.split_seed,
                                  n_query_per_context=a.n_query_per_context)
        if (again.panel_sha256, again.sha256_a, again.sha256_b) != (
                sp.panel_sha256, sp.sha256_a, sp.sha256_b):
            raise RuntimeError(f"eps={e}: panel is not reproducible from its seeds")
        if set(sp.index_a) & set(sp.index_b):
            raise RuntimeError(f"eps={e}: halves overlap")
        if len(sp.index_a) + len(sp.index_b) != 2 * a.n_per_half:
            raise RuntimeError(f"eps={e}: halves do not partition the draw")
        panels.append({"eps": e, "tag": eps_tag(e),
                       "A": sp.provenance(HALF_A), "B": sp.provenance(HALF_B)})
        print(f"eps={e:<5g} panel {sp.panel_sha256[:16]} "
              f"A {sp.sha256_a[:16]} B {sp.sha256_b[:16]}")

    # Distinct panels across eps is structural, not luck: the world carries eps,
    # so a panel drawn against one eps is not a panel for another. Checked so
    # that a bug collapsing the worlds to one would be caught here.
    hashes = {q["A"]["panel_sha256"] for q in panels}
    if len(hashes) != len(panels):
        raise RuntimeError("two eps produced the same panel; the worlds are not "
                           "distinct and the ladder would be one cell repeated")

    doc = {"amendment": "F.2d.5", "eps_grid": EPS_GRID,
           "d": a.d, "K": a.K, "n_rows": a.n_rows, "n_per_half": a.n_per_half,
           "n_contexts_drawn_per_cell": 2 * a.n_per_half,
           "panel_seed": a.panel_seed, "split_seed": a.split_seed,
           "n_query_per_context": a.n_query_per_context,
           "n_half_panels": 2 * len(panels),
           "panels": panels, "wallclock_s": float(time.time() - t0)}
    stamp(doc, "w2_split_panel_manifest", LIVE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps({"n_half_panels": doc["n_half_panels"], "out": str(a.out),
                      "wallclock_s": round(doc["wallclock_s"], 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
