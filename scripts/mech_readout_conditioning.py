"""Conditioning of the registered F.2c readout (LP projection on the Q2 panel).

The readout maps a predictive on the Q2 obs queries to an order posterior by
L1-projecting onto the hull of latent-state predictives (48 latent weights from
a handful of 100-bin predictives). If tiny predictive errors move the projected
order marginal a long way, the readout measures conditioning, not knowledge.

Test: start from the EXACT panel predictive (projects to r = 0 by construction),
mix in a fraction alpha of the model's predictive (or of matched-L1 noise), and
project. Report JS(w_proj_o, w_exact_o) against the predictive L1 distance.
DIAGNOSTIC; reads no registered number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mech_phase1 as M  # noqa: E402
from pfn_dag_verify.corrected_oracle import exact_joint_posterior  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402
from pfn_dag_verify.corrected_world import make_world  # noqa: E402

OUT = ROOT / "campaigns/mech_20260827/readout_conditioning"
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _strip(preds):
    return {"obs": [{"pred": np.asarray(t["pred"], float)} for t in preds["obs"]],
            "int": [None] * len(preds.get("int", []))}


def _one(job):
    from pfn_dag_verify.corrected_tomography import _lp_project_or_ranges, pfn_panel_predictions
    ci, ctx, seed = job
    world, op, panel, model = M._G["world"], M._G["op"], M._G["panel"], M._G["model"]
    post = exact_joint_posterior(world, ctx)
    ex = _strip(op.exact_predictive(post["w_lo"]))
    md = _strip(pfn_panel_predictions(model, world, panel, [ctx], "cpu")[0])
    rng = np.random.default_rng(seed)
    l1_model = float(np.mean([np.abs(a["pred"] - b["pred"]).sum() for a, b in zip(md["obs"], ex["obs"])]))
    targets = [q.target for q in panel.obs]
    l1_by_target = {}
    for t in sorted(set(targets)):
        l1_by_target[t] = float(np.mean([np.abs(a["pred"] - b["pred"]).sum()
                                         for a, b, tt in zip(md["obs"], ex["obs"], targets) if tt == t]))
    out = {"ci": ci, "l1_model": l1_model, "l1_by_target": l1_by_target, "rows": []}
    # In-task-only readout: the same LP on the panel restricted to target d-1.
    op_in, panel_in = M._G["op_in"], M._G["panel_in"]
    keep = [i for i, t in enumerate(targets) if t == world.d - 1]
    for kind, src in (("exact_intask", ex), ("model_intask", md)):
        tgt = {"obs": [src["obs"][i] for i in keep], "int": []}
        try:
            proj = _lp_project_or_ranges(world, op_in, tgt, "project", 0.0, solver=M.SOLVER)
            out["rows"].append((kind, 1.0, float(np.mean([np.abs(src["obs"][i]["pred"] - ex["obs"][i]["pred"]).sum() for i in keep])),
                                float(proj["residual"]), float(M.js(post["w_o"][None], proj["w_o"][None])[0])))
        except Exception:
            out["rows"].append((kind, 1.0, float("nan"), float("nan"), float("nan")))
    for kind in ("model", "noise"):
        for alpha in ALPHAS:
            obs = []
            for a, b in zip(ex["obs"], md["obs"]):
                if kind == "model":
                    p = (1 - alpha) * a["pred"] + alpha * b["pred"]
                else:
                    # noise with the SAME L1 distance from exact as alpha*model
                    n = rng.normal(size=a["pred"].shape)
                    n -= n.mean()
                    target_l1 = alpha * np.abs(b["pred"] - a["pred"]).sum()
                    n *= target_l1 / max(np.abs(n).sum(), 1e-12)
                    p = np.maximum(a["pred"] + n, 1e-12)
                    p /= p.sum()
                obs.append({"pred": p})
            tgt = {"obs": obs, "int": ex["int"]}
            l1 = float(np.mean([np.abs(o["pred"] - a["pred"]).sum() for o, a in zip(obs, ex["obs"])]))
            try:
                proj = _lp_project_or_ranges(world, op, tgt, "project", 0.0, solver=M.SOLVER)
                jsv = float(M.js(post["w_o"][None], proj["w_o"][None])[0])
                out["rows"].append((kind, alpha, l1, float(proj["residual"]), jsv))
            except Exception as exc:
                out["rows"].append((kind, alpha, l1, float("nan"), float("nan")))
    return out


def _init_cond(eps: float, ckpt: str) -> None:
    from pfn_dag_verify.corrected_identifiability import QueryPanel, build_panel_operator
    M._init(eps, ckpt)
    world, panel = M._G["world"], M._G["panel"]
    panel_in = QueryPanel(name="Q2_intask", obs=tuple(q for q in panel.obs if q.target == world.d - 1))
    M._G.update(panel_in=panel_in, op_in=build_panel_operator(world, panel_in))


def run(eps: float, seed: int, step: int, jobs: int, limit: int | None) -> None:
    tag = eps_tag(eps)
    world = make_world(k=M.K, d=M.D, eps=eps)
    ctxs, _ = M.half_b(world)
    if limit:
        ctxs = ctxs[:limit]
    ck = M.NETS / f"eps{tag}" / f"{M.SCALE}_s{seed}_ck{step}.pt"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_cond, initargs=(eps, str(ck))) as pool:
        res = list(pool.map(_one, [(i, c, 330_000_000 + i) for i, c in enumerate(ctxs)], chunksize=2))
    rows = [r for o in res for r in o["rows"]]
    summ = {"eps": eps, "seed": seed, "step": step, "n_ctx": len(ctxs),
            "l1_model_mean": float(np.mean([o["l1_model"] for o in res])),
            "l1_model_by_target": {str(t): float(np.mean([o["l1_by_target"][t] for o in res]))
                                   for t in res[0]["l1_by_target"]},
            "table": []}
    for kind, alpha in [("exact_intask", 1.0), ("model_intask", 1.0)]:
        sel = [r for r in rows if r[0] == kind]
        summ["table"].append({"kind": kind, "alpha": alpha,
                              "l1_pred": float(np.nanmean([r[2] for r in sel])),
                              "proj_residual": float(np.nanmean([r[3] for r in sel])),
                              "js_to_exact": float(np.nanmean([r[4] for r in sel])),
                              "js_median": float(np.nanmedian([r[4] for r in sel])),
                              "n_fail": int(sum(np.isnan(r[4]) for r in sel))})
    for kind in ("model", "noise"):
        for alpha in ALPHAS:
            sel = [r for r in rows if r[0] == kind and r[1] == alpha]
            summ["table"].append({"kind": kind, "alpha": alpha,
                                  "l1_pred": float(np.nanmean([r[2] for r in sel])),
                                  "proj_residual": float(np.nanmean([r[3] for r in sel])),
                                  "js_to_exact": float(np.nanmean([r[4] for r in sel])),
                                  "js_median": float(np.nanmedian([r[4] for r in sel])),
                                  "n_fail": int(sum(np.isnan(r[4]) for r in sel))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cond_eps{tag}_s{seed}_ck{step}.json").write_text(json.dumps(summ, indent=2))
    print(f"[eps={eps}] {time.time() - t0:.0f}s  model L1 from exact: {summ['l1_model_mean']:.4f}", flush=True)
    for t in summ["table"]:
        print(f"  {t['kind']:5s} alpha={t['alpha']:<5} L1={t['l1_pred']:.4f} resid={t['proj_residual']:.4f} "
              f"JS(w_o)={t['js_to_exact']:.4f} (med {t['js_median']:.4f})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, nargs="*", default=[0.25, 1.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=100_000)
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    for e in a.eps:
        run(e, a.seed, a.step, a.jobs, a.limit)


if __name__ == "__main__":
    main()
