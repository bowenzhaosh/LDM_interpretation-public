"""F.1 / E.3 — rescore the W2 checkpoints under the REGISTERED estimand.

WHY THIS IS A NEW FILE AND NOT AN EDIT
--------------------------------------
F.1 registers: "Every behavioural estimand becomes an exact expectation over
outcomes: the bin-summed expected NLL, not the NLL of a single sampled outcome.
The scorer to be replaced is the single-sampled-outcome path in
`corrected_models.evaluate_pfn_checkpoint`."

It was never replaced. At the LOCKED digest d29e0325...b9d9 that function still
scores one drawn bin (`p_pfn[ob]`), and it produced every number in W2's regret
table. Editing it now would (a) change a file Amendment F's F.0 block digests, so
the lock W2 was fired against would no longer verify, and (b) do so AFTER the
as-built gate failure is known.

So the registered estimand lives here instead, under a distinct name. The locked
tree stays byte-identical, all nine F.0 digests continue to verify, W2's
provenance is untouched, and the two series can be reported side by side without
either being mistaken for the other.

E.3's formula, verbatim (AMENDMENT_E.md, E.3):

    R = sum_b p*(b) [ -log q(b) + log p*(b) ]  =  KL( p* || q )

with p* the exact Bayes predictive and q the model's, summed over the 100 native
bins. This removes outcome-sampling noise entirely, which is exactly why E.3 says
"power planning targets panel count only".

Sign discipline, also E.3: a negative regret on an individual context is
legitimate (the Bayes predictive hedges). Only the prior-averaged quantity is
constrained non-negative, and a prior-averaged negative beyond panel-level SE is
an instrument failure rather than a result. Reported, never repaired.

This script computes NO y and touches no fidelity curve.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pfn_dag_verify.artifact_status import DIAGNOSTIC, LIVE, stamp
from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS
from pfn_dag_verify.corrected_models import DEV, ModelConfig, PFN, make_eval_panel
from pfn_dag_verify.corrected_oracle import exact_joint_posterior, obs_query_operator
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT, SCALES
from pfn_dag_verify.corrected_verdict import eps_tag
from pfn_dag_verify.corrected_world import make_world
from pfn_dag_verify.split_panel import (
    HALF_B, SPLIT_SEED, build_split_panel, half_panel_entries,
)

ESTIMAND = "in_task_expected_regret_bin_summed"   # deliberately NOT the locked name
SCALE = "base"
D, K = 3, 8
EPS_ALL = (0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0)


def kl_star_q(p_star: np.ndarray, q: np.ndarray) -> float:
    """E.3: R = sum_b p*(b) [ -log q(b) + log p*(b) ] = KL(p* || q).

    Clamped exactly as the locked scorer clamps its single-bin logs (1e-300), so
    the two series differ ONLY in bin-summed versus single-draw, not in floor.
    """
    p = np.asarray(p_star, dtype=np.float64)
    qq = np.maximum(np.asarray(q, dtype=np.float64), 1e-300)
    pp = np.maximum(p, 1e-300)
    return float(np.sum(p * (np.log(pp) - np.log(qq))))


def load_model(path: Path, cfg: ModelConfig, device) -> PFN:
    m = PFN(cfg)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.to(device).eval()
    return m


def score(model, world, panel, device) -> dict[str, Any]:
    """Bin-summed expected regret over the panel, grouped by context.

    The SE is a PANEL-LEVEL standard error over CONTEXTS, which E.3 requires.
    Pooling queries as though they were independent would understate it wherever
    one context carries more than one query -- and this panel carries two.

    The in-task estimand is the ``target == d-1`` subset (E.3 keeps it; the
    mixed-panel mean is the one it retired). Both are reported, under names that
    cannot be confused, because they have been confused before.
    """
    rows = []
    for ctx, x_q, target, _ob in panel:
        post = exact_joint_posterior(world, ctx)
        qop = obs_query_operator(world, x_q, target)
        p_star = qop.predictive(post["w_lo"])
        ctx_t = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
        qxy = torch.tensor(x_q, dtype=torch.float32, device=device).reshape(1, 1, -1)
        tok = torch.full((1,), 2, dtype=torch.long, device=device)
        # predict_bin_probs already runs under no_grad, moves to cpu and returns
        # float64 numpy (corrected_models.py:84-88). Wrapping it in no_grad and
        # calling .detach() on the result is how the first run died.
        q = model.predict_bin_probs(ctx_t, qxy, tok)[0, 0]
        rows.append({"ctx_key": hash(np.asarray(ctx, dtype=np.float64).tobytes()),
                     "target": int(target), "r": kl_star_q(p_star, q)})

    def block(sel) -> dict:
        vals = [x["r"] for x in rows if sel(x)]
        if not vals:
            return {"mean": None, "se": None, "n": 0, "n_contexts": 0}
        per_ctx: dict[int, list[float]] = {}
        for x in rows:
            if sel(x):
                per_ctx.setdefault(x["ctx_key"], []).append(x["r"])
        arr = np.array([float(np.mean(v)) for v in per_ctx.values()], dtype=np.float64)
        n = len(arr)
        return {"mean": float(arr.mean()),
                "se": float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                "sd_per_context": float(arr.std(ddof=1)) if n > 1 else 0.0,
                "n": len(vals), "n_contexts": n,
                "min": float(min(vals)), "max": float(max(vals)),
                "frac_negative": float(np.mean([v < 0 for v in vals]))}

    counts: dict[int, int] = {}
    for x in rows:
        counts[x["target"]] = counts.get(x["target"], 0) + 1
    return {"by_target_counts": {str(k): v for k, v in sorted(counts.items())},
            "in_task": block(lambda x: x["target"] == D - 1),
            "mixed_panel_RETIRED_ESTIMAND": block(lambda x: True),
            "n_scored_queries": len(rows)}


def run_cell(eps: float, seed: int, netdir: Path, a) -> dict[str, Any]:
    t0 = time.time()
    world = make_world(k=K, d=D, eps=eps)
    split = build_split_panel(world, a.n_per_half, HEADLINE_N_ROWS, PANEL_SEED,
                              split_seed=SPLIT_SEED,
                              n_query_per_context=N_QUERY_PER_CONTEXT)
    full = make_eval_panel(world, 2 * a.n_per_half, HEADLINE_N_ROWS, seed=PANEL_SEED,
                           n_query_per_context=N_QUERY_PER_CONTEXT)
    panel_b = half_panel_entries(full, split, HALF_B)
    device = torch.device(DEV if a.device == "auto" else a.device)
    cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE])

    # U1 is underdetermined by the locked text: nothing checks that base_sN.pt and
    # base_sN_ck100000.pt agree. Check it here rather than choosing silently.
    final_p = netdir / f"{SCALE}_s{seed}.pt"
    ck_p = netdir / f"{SCALE}_s{seed}_ck100000.pt"
    agree = None
    if final_p.is_file() and ck_p.is_file():
        sa = torch.load(final_p, map_location="cpu")
        sb = torch.load(ck_p, map_location="cpu")
        agree = bool(set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa))
    use = ck_p if ck_p.is_file() else final_p
    model = load_model(use, cfg, device)
    res = score(model, world, panel_b, device)

    doc: dict[str, Any] = {
        "amendment": "F.1 / E.3 (bin-summed expected regret)",
        "formula": "R = sum_b p*(b) [ -log q(b) + log p*(b) ] = KL(p* || q)",
        "NOT_THE_LOCKED_SCORER": (
            "corrected_models.evaluate_pfn_checkpoint, at the digest Amendment F's "
            "F.0 records, scores one DRAWN bin and does not implement F.1. This "
            "artifact is the registered estimand computed by a separate scorer; the "
            "locked tree is unmodified and its digests still verify."),
        "world": {"d": D, "K": K, "eps": eps}, "seed": seed, "scale": SCALE,
        "checkpoint": str(use.name),
        "final_and_ck100000_agree": agree,
        "panel": split.provenance(HALF_B),
        **res,
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "environment": {"host": socket.gethostname(), "platform": platform.platform(),
                        "python": sys.version.split()[0]},
        "wallclock_s": round(time.time() - t0, 2),
    }
    stamp(doc, ESTIMAND, LIVE if a.status == "live" else DIAGNOSTIC)
    doc["artifact"]["n_rows"] = int(HEADLINE_N_ROWS)
    doc["artifact"]["half"] = HALF_B
    doc["artifact"]["n_per_half"] = int(a.n_per_half)
    out = Path(a.outdir) / f"f1regret_eps{eps_tag(eps)}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=float))
    it = res["in_task"]
    print(f"[eps={eps} s={seed}] in-task R = {it['mean']:.6f} +/- {it['se']:.6f} "
          f"(n_ctx {it['n_contexts']}, n_q {it['n']}) ck_agree={agree} "
          f"{doc['wallclock_s']}s -> {out.name}", flush=True)
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", type=float, nargs="*", default=list(EPS_ALL))
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--netroot", type=Path, required=True,
                   help="W2 outroot; nets live at <root>/nets/eps<tag>/")
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--device", default="auto")
    p.add_argument("--status", choices=["live", "diagnostic"], default="live")
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args(argv)
    for e in a.eps:
        nd = Path(a.netroot) / "nets" / f"eps{eps_tag(e)}"
        if not nd.is_dir():
            nd = Path(a.netroot) / f"eps{eps_tag(e)}" / "nets"
        for s in a.seeds:
            run_cell(e, s, nd, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
