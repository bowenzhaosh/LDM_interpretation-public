"""Re-run the PFN posterior tomography with the FIXED code, per epsilon.

The pre-fix Track B runs produced valid PFN results but a tainted tomography
section (the one-sided L1 LP bug inflated the reported projection residual by
2x and the identified-set tolerance by 2x). This re-trains the base-scale PFN
(identical seed/config to the Track B driver) and recomputes the tomography,
then merges it into the existing trackb JSON.

Run:  PYTHONPATH=src python -m pfn_dag_verify.corrected_tomography_run \
          --d 3 --K 8 --eps 1.0 --trackb-json path/to/trackb_eps1p0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .corrected_models import (
    ModelConfig,
    evaluate_pfn_checkpoint,
    make_classifier_panel,
    make_eval_panel,
    train_pfn,
)
from .corrected_tomography import aggregate_tomography, run_tomography
from .corrected_trackb import N_CONTEXT, N_EVAL_CONTEXTS, N_QUERY_PER_CONTEXT, SCALES
from .corrected_world import make_world

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(d: int, K: int, eps: float, trackb_json: Path, out: Path,
        ckpt: Path | None = None) -> dict:
    world = make_world(k=K, d=d, eps=eps)
    panel = make_eval_panel(world, N_EVAL_CONTEXTS, N_CONTEXT, seed=770_000_000,
                            n_query_per_context=N_QUERY_PER_CONTEXT)
    tom_ctxs = make_classifier_panel(world, 15, N_CONTEXT, seed=772_000_000)
    cfg = ModelConfig(name="base", d=d, **SCALES["base"])
    t0 = time.time()
    # Amendment E: prefer LOADING the fleet's persisted seed-0 weights so the
    # tomography readout sits on the same model as the regret/classifier numbers
    # it is plotted beside in F4. Training here produced a different seed-0 model.
    if ckpt is not None:
        from .corrected_models import PFN
        model = PFN(cfg).to(DEV)
        model.load_state_dict(torch.load(Path(ckpt), map_location=DEV))
        model.eval()
        res = {"model": model, "loaded_from": str(ckpt)}
    else:
        netdir = Path(out).parent / "models" / f"eps{str(eps).replace('.', 'p')}"
        res = train_pfn(world, steps=100_000, seed=0, cfg=cfg,
                        n_ctx=N_CONTEXT, n_query=7,
                        outdir=netdir, tag_prefix="tomography_s0")
    device = torch.device(DEV)
    ev_base = evaluate_pfn_checkpoint(res["model"], world, panel, device)
    tol = max(ev_base["js_mean"] * 2.0, 1e-3)
    tom = run_tomography(world, res["model"], tom_ctxs, tol=tol, device=DEV)
    result = {
        "tol": tol,
        "n_contexts": len(tom_ctxs),
        "aggregates": aggregate_tomography(tom),
        "wallclock_s": float(time.time() - t0),
        "fixed_code": True,
    }
    # merge into the existing trackb JSON (preserve the valid PFN results)
    if trackb_json is not None and trackb_json.is_file():
        full = json.loads(trackb_json.read_text())
        full["tomography"] = result
        trackb_json.write_text(json.dumps(full, indent=2, default=str))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--trackb-json", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="load these persisted weights instead of retraining (Amendment E)")
    args = p.parse_args(argv)
    run(args.d, args.K, args.eps, args.trackb_json, args.out, ckpt=args.ckpt)
    print(json.dumps({"done": True, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
