"""Track B driver (B1 vanilla PFN saturation + B2 direct order classifier +
B3 epsilon continuum) — runs one world (d, K, eps) end to end.

For each model scale and training budget, trains the vanilla PFN with the
corrected forward-map generator, evaluates Bayes regret on a FIXED held-out
panel at multiple checkpoints (learning curve, not replicates), trains the
direct order classifier, and evaluates its posterior-fidelity metrics against
the exact p(o | D). Writes machine-readable JSON.

Run (cluster or local):
  PYTHONPATH=src python -m pfn_dag_verify.corrected_trackb --d 3 --K 8 --eps 1.0 \
      --out campaigns/corrected_20260812/raw/trackb_eps1.0.json
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
    evaluate_classifier_checkpoint,
    evaluate_pfn_checkpoint,
    evaluate_pfn_state,
    make_classifier_panel,
    make_eval_panel,
    train_classifier,
    train_pfn,
)
from .corrected_world import make_world

DEV = "cuda" if torch.cuda.is_available() else "cpu"


SCALES = {
    "small": dict(d_model=64, d_ff=128, n_heads=2, n_layers=1),
    "base": dict(d_model=128, d_ff=256, n_heads=4, n_layers=2),
    "large": dict(d_model=256, d_ff=512, n_heads=4, n_layers=4),
}
STEPS = [10_000, 25_000, 50_000, 100_000]
N_CONTEXT = 20
N_EVAL_CONTEXTS = 60
N_QUERY_PER_CONTEXT = 2
CLASSIFIER_STEPS = 50_000


def run_world(d: int, K: int, eps: float, out: Path, seeds: list[int]) -> dict[str, Any]:
    world = make_world(k=K, d=d, eps=eps)
    panel = make_eval_panel(world, N_EVAL_CONTEXTS, N_CONTEXT, seed=770_000_000,
                            n_query_per_context=N_QUERY_PER_CONTEXT)
    cpanel = make_classifier_panel(world, N_EVAL_CONTEXTS, N_CONTEXT, seed=771_000_000)
    device = torch.device(DEV)

    results: dict[str, Any] = {
        "world": {"d": d, "K": K, "eps": eps, "spec_r": world.spec.r},
        "panel": {"n_contexts": N_EVAL_CONTEXTS, "n_rows": N_CONTEXT,
                  "n_queries": len(panel), "seed": 770_000_000},
        "pfn": {},
        "classifier": {},
    }

    def _save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))

    _save()

    base_tom_model = None
    base_cfg = None
    for scale_name, scale in SCALES.items():
        cfg = ModelConfig(name=scale_name, d=d, **scale)
        for seed in seeds:
            t0 = time.time()
            use_ckpts = (scale_name == "base" and seed == seeds[0])
            # Amendment E: persist weights + provenance for every scale x seed.
            netdir = Path(out).parent / "models" / f"eps{str(eps).replace('.', 'p')}"
            res = train_pfn(world, steps=max(STEPS), seed=seed, cfg=cfg,
                            n_ctx=N_CONTEXT, n_query=7, ckpt_every=0,
                            ckpt_steps=tuple(STEPS) if use_ckpts else (),
                            outdir=netdir, tag_prefix=f"{scale_name}_s{seed}")
            ev = evaluate_pfn_checkpoint(res["model"], world, panel, device)
            results["pfn"].setdefault(scale_name, {})
            results["pfn"][scale_name][f"s{seed}"] = {
                "n_params": res["n_params"],
                "final_loss": res["final_loss"],
                "bayes_regret": ev["bayes_regret_mean"],
                "bayes_regret_se": ev["bayes_regret_se"],
                "js": ev["js_mean"],
                "wallclock_s": float(time.time() - t0),
            }
            if use_ckpts:
                # checkpoint-as-learning-curve from the ONE training run
                lc = []
                for steps in STEPS:
                    ev_c = evaluate_pfn_state(res["checkpoints"][steps], cfg,
                                              world, panel, device)
                    lc.append({"steps": steps, "bayes_regret": ev_c["bayes_regret_mean"],
                               "bayes_regret_se": ev_c["bayes_regret_se"],
                               "js": ev_c["js_mean"]})
                results["pfn"]["base_learning_curve"] = lc
                base_tom_model = res["model"]
                base_cfg = cfg
            else:
                del res["model"]
            del res["checkpoints"]
            torch.cuda.empty_cache()
        _save()  # PFN results saved before the classifier/tomography stages

    # B2 — direct order classifier
    for seed in seeds:
        t0 = time.time()
        cfg = ModelConfig(name="classifier", d=d)
        res = train_classifier(world, CLASSIFIER_STEPS, seed, cfg, n_ctx=N_CONTEXT)
        ev = evaluate_classifier_checkpoint(res["model"], world, cpanel, device)
        results["classifier"][f"s{seed}"] = {
            "n_params": res["n_params"],
            "final_loss": res["final_loss"],
            **ev,
            "wallclock_s": float(time.time() - t0),
        }
        del res["model"]
        torch.cuda.empty_cache()
    _save()  # classifier results saved before tomography

    # Tomography (Track A+B connection) — on the base-scale seed-0 final model
    # (reused, not re-trained).
    from .corrected_tomography import run_tomography, aggregate_tomography
    t0 = time.time()
    tom_ctxs = make_classifier_panel(world, 15, N_CONTEXT, seed=772_000_000)
    # tolerance = PFN's typical predictive JS error on the eval panel
    ev_base = evaluate_pfn_checkpoint(base_tom_model, world, panel, device)
    tol = max(ev_base["js_mean"] * 2.0, 1e-3)
    tom = run_tomography(world, base_tom_model, tom_ctxs, tol=tol, device=DEV)
    results["tomography"] = {
        "tol": tol, "n_contexts": len(tom_ctxs),
        "aggregates": aggregate_tomography(tom),
        "wallclock_s": float(time.time() - t0),
    }
    del base_tom_model
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
    run_world(args.d, args.K, args.eps, args.out, seeds)
    print(json.dumps({"done": True, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
