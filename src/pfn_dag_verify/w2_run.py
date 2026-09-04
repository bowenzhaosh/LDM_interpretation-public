"""Amendment F — W2, the retrain on the repaired instrument.

One (eps, seed) cell per invocation. Twenty-one cells: seven eps x three seeds,
base scale.

What this produces, and what it deliberately does not.

PRODUCES
  * the trained weights, every requested intermediate checkpoint, and the DOSE-0
    state, all on disk with provenance sidecars. Dose-0 is written before step 1
    (F.2c's operational corollary): F.2c's floor curve does not exist without
    it, and a fleet that trains without persisting it would have to be re-run.
  * the in-task expected regret on HALF B of the cell's split panel, which is
    what gate (a) reads. Half A is the deficit's and no readout here touches it.

DOES NOT PRODUCE
  * the three-curve calibrated fidelity of F.2c. The model, floor and ceiling
    readouts run over the PERSISTED checkpoints in a later pass -- which is the
    entire reason F.2c's corollary is about persistence rather than about W2
    computing the curve. Nothing here is a y value, and nothing here should be
    read as one.

The in-task estimand is the ``target == d-1`` subset. The eval panel samples the
target uniformly with no target index given to the model, so the mixed-panel
mean is dominated by out-of-task queries; E.3 retired that estimand and the pair
is named separately in artifact_status precisely because they have been confused
once already.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifact_status import DIAGNOSTIC, EST_REGRET_INTASK, LIVE, stamp
from .corrected_deficit_run import N_PER_HALF, PANEL_SEED
from .corrected_identifiability_run import HEADLINE_N_ROWS
from .corrected_models import (
    DEV, ModelConfig, evaluate_pfn_checkpoint, evaluate_pfn_state, make_eval_panel,
    train_pfn,
)
from .corrected_trackb import N_QUERY_PER_CONTEXT, SCALES
from .corrected_verdict import eps_tag
from .corrected_world import make_world
from .split_panel import HALF_B, SPLIT_SEED, build_split_panel, half_panel_entries

STEPS = 100_000
CKPT_STEPS = (0, 10_000, 25_000, 50_000, 100_000)   # 0 is the F.2c floor state
SCALE = "base"


def _intask(ev: dict, d: int) -> dict:
    """The target == d-1 slice of a scored panel, with its own n and SE."""
    row = ev["regret_by_target"].get(str(d - 1))
    if row is None or not row["n"]:
        raise ValueError(
            f"no target={d-1} queries on this panel; the in-task estimand is "
            "undefined and the mixed-panel mean is NOT a substitute (E.3)")
    return {"in_task_regret": row["mean"], "in_task_regret_se": row["se"],
            "n_in_task_queries": row["n"]}


def run(d: int, K: int, eps: float, seed: int, out: Path, netdir: Path,
        steps: int = STEPS, n_per_half: int = N_PER_HALF,
        n_rows: int = HEADLINE_N_ROWS, panel_seed: int = PANEL_SEED,
        split_seed: int = SPLIT_SEED,
        ckpt_steps: tuple[int, ...] = CKPT_STEPS,
        diagnostic: bool = False) -> dict[str, Any]:
    world = make_world(k=K, d=d, eps=eps)
    split = build_split_panel(world, n_per_half, n_rows, panel_seed,
                              split_seed=split_seed,
                              n_query_per_context=N_QUERY_PER_CONTEXT)
    full = make_eval_panel(world, 2 * n_per_half, n_rows, seed=panel_seed,
                           n_query_per_context=N_QUERY_PER_CONTEXT)
    panel_b = half_panel_entries(full, split, HALF_B)
    device = torch.device(DEV)
    cfg = ModelConfig(name=SCALE, d=d, **SCALES[SCALE])

    t0 = time.time()
    res = train_pfn(world, steps=steps, seed=seed, cfg=cfg, n_ctx=n_rows,
                    n_query=7, ckpt_steps=ckpt_steps,
                    outdir=netdir, tag_prefix=f"{SCALE}_s{seed}")
    train_s = float(time.time() - t0)

    ev = evaluate_pfn_checkpoint(res["model"], world, panel_b, device)
    doc: dict[str, Any] = {
        "amendment": "F/W2",
        "world": {"d": d, "K": K, "eps": eps, "spec_r": world.spec.r},
        "scale": SCALE, "seed": seed, "steps": steps,
        "n_params": res["n_params"], "final_loss": res["final_loss"],
        "train_wallclock_s": train_s,
        "panel": split.provenance(HALF_B),
        "n_scored_queries": len(panel_b),
        "final": {**_intask(ev, d), "mixed_panel_regret": ev["bayes_regret_mean"],
                  "js_mean": ev["js_mean"],
                  "regret_by_target": ev["regret_by_target"]},
        "checkpoints": {},
        "device": str(DEV), "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu", "numpy": np.__version__,
    }
    # The learning curve off the SAME training run, on the SAME half-B panel.
    # Step 0 is included: it is the untrained model's regret, and it is the only
    # number in this file that belongs to the floor state.
    for s in ckpt_steps:
        if s not in res["checkpoints"]:
            raise ValueError(
                f"checkpoint {s} was requested and is absent. Step 0 in "
                "particular used to be a SILENT no-op, so its absence is "
                "checked here rather than discovered when the floor curve is "
                "built weeks from now.")
        evc = evaluate_pfn_state(res["checkpoints"][s], cfg, world, panel_b, device)
        doc["checkpoints"][str(s)] = {**_intask(evc, d),
                                      "js_mean": evc["js_mean"],
                                      "is_dose0": s == 0}
    # A short plumbing run is stamped DIAGNOSTIC, so the verdict loader's
    # require_live refuses it. A smoke that can be read as the measurement is
    # worse than no smoke: the artifacts share a filename pattern and differ
    # only in step count, which is exactly the kind of difference nobody
    # notices in a directory listing.
    stamp(doc, EST_REGRET_INTASK, DIAGNOSTIC if diagnostic else LIVE)
    doc["diagnostic_run"] = bool(diagnostic)
    # The reader checks these three, so the writer stamps them. They are the
    # difference between an artifact that CAN be refused and one the loader has
    # to take on trust -- and the regret is gate (a)'s only input.
    doc["artifact"]["n_rows"] = int(n_rows)
    doc["artifact"]["half"] = HALF_B
    doc["artifact"]["n_per_half"] = int(n_per_half)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=float))
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--n-rows", type=int, default=HEADLINE_N_ROWS)
    p.add_argument("--ckpt-steps", type=int, nargs="*", default=list(CKPT_STEPS),
                   help="smoke runs only; the production ladder is the default")
    p.add_argument("--diagnostic", action="store_true",
                   help="stamp DIAGNOSTIC: a plumbing smoke, not a W2 cell")
    p.add_argument("--outroot", type=Path, required=True)
    a = p.parse_args(argv)
    tag = eps_tag(a.eps)
    doc = run(a.d, a.K, a.eps, a.seed,
              out=a.outroot / f"w2_eps{tag}_s{a.seed}.json",
              netdir=a.outroot / "nets" / f"eps{tag}",
              steps=a.steps, n_per_half=a.n_per_half, n_rows=a.n_rows,
              ckpt_steps=tuple(a.ckpt_steps), diagnostic=a.diagnostic)
    print(json.dumps({"eps": a.eps, "seed": a.seed,
                      "in_task_regret": doc["final"]["in_task_regret"],
                      "n_in_task_queries": doc["final"]["n_in_task_queries"],
                      "dose0_regret": doc["checkpoints"]["0"]["in_task_regret"],
                      "train_wallclock_s": round(doc["train_wallclock_s"], 1)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
