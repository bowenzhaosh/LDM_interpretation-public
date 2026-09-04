"""Corrected track-B measurement: re-train base-scale PFNs and report
PER-TARGET Bayes regret on the fixed mixed-target panel.

Why this exists: the original trackb regret (0.326-0.661 across eps) is a mean
over a panel where `make_eval_panel` samples the query target uniformly over
{0,1,2}, while `train_pfn` trains the PFN to predict ONLY the last coordinate
(column d-1). The model has no target-index input, so ~2/3 of the panel
queries (target != d-1) are out-of-task: the mixed-panel regret does not
measure the trained task. This module re-trains (weights were never persisted;
`corrected_trackb` never passed outdir) and reports the in-task (target=d-1)
regret, the per-target decomposition, and an in-task learning curve, so the
verdict can be re-derived on honest numbers.

Caveat (recorded, not fixed): training determinism across the original runs is
unrecorded (no torch/CUDA version pinned), so the re-trained models are
"the same recipe and seed" but not bit-identical to the original runs. The
per-target decomposition on a re-trained model is the quantity of interest;
the mixed-panel aggregate should match the original within training noise.

Run (cluster or local):
  PYTHONPATH=src python -m pfn_dag_verify.corrected_intask_run --d 3 --K 8 --eps 0.1 \
      --out campaigns/corrected_20260812/raw/trackb_eps0p1_corrected.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .corrected_models import (
    ModelConfig,
    evaluate_pfn_checkpoint,
    evaluate_pfn_state,
    make_eval_panel,
    train_pfn,
)
from .corrected_world import make_world

DEV = "cuda" if torch.cuda.is_available() else "cpu"

BASE = dict(d_model=128, d_ff=256, n_heads=4, n_layers=2)
STEPS = [10_000, 25_000, 50_000, 100_000]
N_CONTEXT = 20
N_EVAL_CONTEXTS = 60
N_QUERY_PER_CONTEXT = 2
PANEL_SEED = 770_000_000


def run_intask(d: int, K: int, eps: float, out: Path, seeds: list[int]) -> dict[str, Any]:
    world = make_world(k=K, d=d, eps=eps)
    # The SAME fixed mixed-target panel the original trackb used (seed 770M),
    # so per-target decomposition is directly comparable to the stored
    # bayes_regret aggregates.
    panel = make_eval_panel(world, N_EVAL_CONTEXTS, N_CONTEXT, seed=PANEL_SEED,
                            n_query_per_context=N_QUERY_PER_CONTEXT)
    device = torch.device(DEV)
    cfg = ModelConfig(name="base", d=d, **BASE)
    intask_key = str(d - 1)

    results: dict[str, Any] = {
        "world": {"d": d, "K": K, "eps": eps, "spec_r": world.spec.r},
        "panel": {"n_contexts": N_EVAL_CONTEXTS, "n_rows": N_CONTEXT,
                  "n_queries": len(panel), "seed": PANEL_SEED,
                  "targets": "uniform",
                  "note": ("fixed mixed-target panel (original seed); target=d-1 "
                           "subset is the in-task regret")},
        "pfn": {},
        "base_learning_curve": None,       # mixed-panel, for continuity
        "base_in_task_learning_curve": None,
    }

    def _save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))

    _save()

    for i, seed in enumerate(seeds):
        t0 = time.time()
        use_ckpts = (i == 0)
        # Amendment E: persist weights + provenance. Every pre-Amendment-E d=3
        # number came from weights that did not survive the process.
        netdir = Path(out).parent / "models" / f"eps{str(eps).replace('.', 'p')}"
        res = train_pfn(world, steps=max(STEPS), seed=seed, cfg=cfg,
                        n_ctx=N_CONTEXT, n_query=7, ckpt_every=0,
                        ckpt_steps=tuple(STEPS) if use_ckpts else (),
                        outdir=netdir, tag_prefix=f"intask_s{seed}")
        ev = evaluate_pfn_checkpoint(res["model"], world, panel, device)
        rbt = ev["regret_by_target"]
        results["pfn"][f"s{seed}"] = {
            "n_params": res["n_params"],
            "final_loss": res["final_loss"],
            "bayes_regret": ev["bayes_regret_mean"],          # mixed panel (as before)
            "bayes_regret_se": ev["bayes_regret_se"],
            "js": ev["js_mean"],
            "regret_by_target": rbt,
            "in_task_regret": rbt[intask_key]["mean"],        # the honest number
            "in_task_regret_se": rbt[intask_key]["se"],
            "in_task_frac": rbt[intask_key]["n"] / len(panel),
            "wallclock_s": float(time.time() - t0),
        }
        if use_ckpts:
            lc, lc_in = [], []
            for steps in STEPS:
                ev_c = evaluate_pfn_state(res["checkpoints"][steps], cfg,
                                          world, panel, device)
                lc.append({"steps": steps, "bayes_regret": ev_c["bayes_regret_mean"],
                           "bayes_regret_se": ev_c["bayes_regret_se"],
                           "js": ev_c["js_mean"]})
                rbt_c = ev_c["regret_by_target"]
                lc_in.append({"steps": steps,
                              "in_task_regret": rbt_c[intask_key]["mean"],
                              "in_task_regret_se": rbt_c[intask_key]["se"]})
            results["base_learning_curve"] = lc
            results["base_in_task_learning_curve"] = lc_in
        del res["model"]
        del res["checkpoints"]
        torch.cuda.empty_cache()
        _save()

    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=str, default="0,1,2")
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    run_intask(args.d, args.K, args.eps, args.out, seeds)
    print(json.dumps({"done": True, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
