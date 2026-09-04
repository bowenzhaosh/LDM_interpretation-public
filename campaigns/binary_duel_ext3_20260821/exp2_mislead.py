#!/usr/bin/env python3
"""EXPERIMENT 2 — deliberately mislead the PFN (PREREG-BINARY-DUEL.md §3).

  E2a  Natural misleading contexts: true-G1 contexts whose residual realization
       produced negative ell (evidence for the wrong order G2). Compare model
       order weight + held-out NLL on fresh A- and B-outcomes against matched
       positive-ell controls (matched on |ell|).
  E2b  Wrong-order evidence injection: replace a delta-fraction of rows with
       their residual-matched G2 mirrors (same Sigma). Same-order control:
       replace with matched fresh G1 rows. Trace logit w and NLL vs delta.

Numbers + gate booleans only. Per-seed result JSONs.
"""
import os
import sys
import json
import argparse
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import core as C
import evidence_core as EC

RESULT_DIR = os.path.join(_HERE, "results", "exp2")
os.makedirs(RESULT_DIR, exist_ok=True)


def _w_and_s(model, D, S, prior, qx_grid=C.Q0):
    p1, p2 = EC.E.sigma_to_params_G1(S), EC.E.sigma_to_params_G2(S)
    lg1, lg2 = C.anchor_logprobs(p1, p2, prior, qx_grid)
    logQ = C.model_predictive(model, D, qx_grid)
    return C.evidence_scores(logQ, lg1, lg2)


def _nll_on(model, D, S, cls_true, prior, M, rng):
    """NLL of Q(D) on M fresh outcomes of the given true process."""
    pT = EC.E.sigma_to_params_G1(S) if cls_true == 1 else EC.E.sigma_to_params_G2(S)
    xs, ys = C.fresh_outcomes(prior, cls_true, pT, M, rng)
    logQ = C.model_predictive(model, D, xs)
    return C.mean_nll_at_outcomes(logQ, ys), xs, ys


def run_e2a(model, prior, rng, K=300, pool=30000, M=20, smoke=False, tag="e2a"):
    K, pool, M = (20, 200, 5) if smoke else (K, pool, M)
    # draw true-G1 contexts with latent
    contexts = []
    while len(contexts) < pool:
        D, S, cls, params = C.gen_context_with_latent(prior, rng)
        if cls != 1:
            continue
        ell = EC.compute_ell(D, prior)
        if not np.isfinite(ell):
            continue
        contexts.append(dict(D=D, S=S, ell=float(ell)))
    ells = np.asarray([c["ell"] for c in contexts])
    # misleading tail (most negative) + positive pool for controls
    neg_idx = np.argsort(ells)[:K]
    pos_idx = np.where(ells > 0.5)[0]
    if len(pos_idx) < K:
        pos_idx = np.argsort(ells)[::-1][:K]
    # greedy match each misleading |ell| to nearest positive |ell|
    neg_ells = ells[neg_idx]
    pos_ells = ells[pos_idx]
    used = np.zeros(len(pos_ells), bool)
    pairs = []
    for n_i in range(len(neg_idx)):
        dist = np.abs(np.abs(pos_ells) - np.abs(neg_ells[n_i]))
        dist[used] = np.inf
        j = int(np.argmin(dist))
        used[j] = True
        pairs.append((int(neg_idx[n_i]), int(pos_idx[j])))

    recs = []
    for a_i, b_i in pairs:
        cm, cc = contexts[a_i], contexts[b_i]
        sm, sc = cm["S"], cc["S"]
        s_m, lw_m = _w_and_s(model, cm["D"], sm, prior)   # (s, logit w)
        s_c, lw_c = _w_and_s(model, cc["D"], sc, prior)
        nll_Am, xs, ys = _nll_on(model, cm["D"], sm, 1, prior, M, rng)
        nll_Bm, _, _ = _nll_on(model, cm["D"], sm, 2, prior, M, rng)
        nll_Ac, _, _ = _nll_on(model, cc["D"], sc, 1, prior, M, rng)
        nll_Bc, _, _ = _nll_on(model, cc["D"], sc, 2, prior, M, rng)
        recs.append(dict(ell_mis=cm["ell"], ell_ctrl=cc["ell"],
                         logit_w_mis=lw_m, logit_w_ctrl=lw_c, s_mis=s_m, s_ctrl=s_c,
                         nll_A_mis=nll_Am, nll_B_mis=nll_Bm,
                         nll_A_ctrl=nll_Ac, nll_B_ctrl=nll_Bc))
    r = recs
    lw_mis = np.array([x["logit_w_mis"] for x in r])
    lw_ctrl = np.array([x["logit_w_ctrl"] for x in r])
    harm_A = np.array([x["nll_A_mis"] - x["nll_A_ctrl"] for x in r])
    benefit_B = np.array([x["nll_B_ctrl"] - x["nll_B_mis"] for x in r])
    ell_mis = np.array([x["ell_mis"] for x in r])
    # scaling of logit w across the misleading tail
    scale_reg = C.fit_evidence_regression(ell_mis, lw_mis, rng_seed=101)
    return dict(tag=tag, K=len(r),
                mean_logit_w_mis=float(np.nanmean(lw_mis)),
                mean_logit_w_ctrl=float(np.nanmean(lw_ctrl)),
                frac_w_mis_pos=float(np.nanmean(lw_mis > 0)),
                frac_w_ctrl_neg=float(np.nanmean(lw_ctrl < 0)),
                harm_A_mean=float(np.nanmean(harm_A)),
                harm_A_wilcox_p=C.wilcoxon_paired(harm_A),
                benefit_B_mean=float(np.nanmean(benefit_B)),
                benefit_B_wilcox_p=C.wilcoxon_paired(benefit_B),
                scale_reg=scale_reg,
                gate_w=bool(np.nanmean(lw_mis) > 0 and np.nanmean(lw_ctrl) < 0),
                gate_harm=bool(C.wilcoxon_paired(harm_A) is not None
                              and C.wilcoxon_paired(harm_A) < 0.01
                              and np.nanmean(harm_A) > 0),
                gate_benefit=bool(C.wilcoxon_paired(benefit_B) is not None
                                  and C.wilcoxon_paired(benefit_B) < 0.01
                                  and np.nanmean(benefit_B) > 0))


def row_features(D):
    """(n, 4) standardized row features: ||row||, x*y, |x|, |y|."""
    x, y = D[:, 0], D[:, 1]
    f = np.stack([np.hypot(x, y), x * y, np.abs(x), np.abs(y)], axis=1)
    mu, sd = f.mean(0), f.std(0) + 1e-9
    return (f - mu) / sd


def nearest_fresh_rows(D_base, S, prior, n_rows, rng):
    """Pick fresh G1(S) rows nearest (in standardized feature space) to the
    given base rows. Returns (n_rows, 2) array."""
    p1 = EC.E.sigma_to_params_G1(S)
    # pool of fresh G1 rows
    n_pool = max(400, 8 * n_rows)
    z1 = C._noise_std_resid(prior, n_pool, rng)
    z2 = C._noise_std_resid(prior, n_pool, rng)
    x = p1[1] * z1
    fresh = np.stack([x, p1[0] * x + p1[2] * z2], axis=1)
    ff = row_features(fresh)
    bf = row_features(D_base)
    out = []
    for row in bf:
        d = np.sum((ff - row) ** 2, axis=1)
        j = int(np.argmin(d))
        out.append(fresh[j])
    return np.asarray(out)


def run_e2b(model, prior, rng, K=400, M=20, deltas=(0.0, 0.10, 0.25, 0.50), smoke=False):
    K, M, deltas = (40, 5, (0.0, 0.10, 0.25)) if smoke else (K, M, deltas)
    # base A-contexts (G1), each with its own fixed Sigma
    bases = []
    while len(bases) < K:
        D, S, cls, params = C.gen_context_with_latent(prior, rng)
        if cls != 1:
            continue
        bases.append(dict(D=D, S=S))
    recs = []
    for bi, b in enumerate(bases):
        D, S = b["D"], b["S"]
        pA = EC.E.sigma_to_params_G1(S)
        pB = EC.E.sigma_to_params_G2(S)
        # row-wise mirrors (same residuals, other order). D was built under G1:
        # x = b1 z1, y = a x + b2 z2  =>  recover z1 = x/b1, z2 = (y - a x)/b2.
        z1 = D[:, 0] / pA[1]
        z2 = (D[:, 1] - pA[0] * D[:, 0]) / pA[2]
        mirror_rows = C.context_from_std(prior, 2, pB, z1, z2)  # G2(S) rows, same z
        fresh_rows = nearest_fresh_rows(D, S, prior, C.N_CONTEXT, rng)
        rng.shuffle(fresh_rows)
        # one fixed outcome set per base (shared by wrong and control branches)
        xs_out, ys_out = C.fresh_outcomes(prior, 1, pA, M, rng)
        rec = dict()
        for dlt in deltas:
            n_swap = int(round(dlt * C.N_CONTEXT))
            idx = rng.choice(C.N_CONTEXT, n_swap, replace=False) if n_swap else []
            D_w = D.copy(); D_c = D.copy()
            if n_swap:
                D_w[idx] = mirror_rows[idx]
                D_c[idx] = fresh_rows[idx]
            s_w, lw_w = _w_and_s(model, D_w, S, prior)   # (s, logit w)
            s_c, lw_c = _w_and_s(model, D_c, S, prior)
            nll_A_w = C.mean_nll_at_outcomes(C.model_predictive(model, D_w, xs_out), ys_out)
            nll_A_c = C.mean_nll_at_outcomes(C.model_predictive(model, D_c, xs_out), ys_out)
            ell_w = EC.compute_ell(D_w, prior)
            rec[f"w_w_{dlt:.2f}"] = lw_w; rec[f"w_c_{dlt:.2f}"] = lw_c
            rec[f"s_w_{dlt:.2f}"] = s_w; rec[f"s_c_{dlt:.2f}"] = s_c
            rec[f"nll_A_w_{dlt:.2f}"] = nll_A_w; rec[f"nll_A_c_{dlt:.2f}"] = nll_A_c
            rec[f"ell_w_{dlt:.2f}"] = ell_w
        recs.append(rec)
    # aggregate per delta
    out = dict(tag="e2b", K=len(recs))
    for dlt in deltas:
        ww = np.array([r[f"w_w_{dlt:.2f}"] for r in recs])
        wc = np.array([r[f"w_c_{dlt:.2f}"] for r in recs])
        na_w = np.array([r[f"nll_A_w_{dlt:.2f}"] for r in recs])
        na_c = np.array([r[f"nll_A_c_{dlt:.2f}"] for r in recs])
        ew = np.array([r[f"ell_w_{dlt:.2f}"] for r in recs])
        out[f"delta_{dlt:.2f}"] = dict(
            mean_logit_w_wrong=float(np.nanmean(ww)),
            mean_logit_w_control=float(np.nanmean(wc)),
            mean_nll_A_wrong=float(np.nanmean(na_w)),
            mean_nll_A_control=float(np.nanmean(na_c)),
            mean_ell_injected=float(np.nanmean(ew)))
    # monotonicity: mean logit w (wrong) increasing and mean NLL (wrong) increasing in delta
    lw = [out[f"delta_{d:.2f}"]["mean_logit_w_wrong"] for d in deltas]
    na = [out[f"delta_{d:.2f}"]["mean_nll_A_wrong"] for d in deltas]
    lc = [out[f"delta_{d:.2f}"]["mean_logit_w_control"] for d in deltas]
    out["monotone_w_wrong"] = bool(all(lw[i] <= lw[i + 1] + 1e-9 for i in range(len(lw) - 1)))
    out["monotone_nll_wrong"] = bool(all(na[i] <= na[i + 1] + 1e-9 for i in range(len(na) - 1)))
    out["spread_logit_w_wrong"] = float(lw[-1] - lw[0])
    out["spread_nll_wrong"] = float(na[-1] - na[0])
    out["control_flat_w"] = float(max(lc) - min(lc))
    out["gate_w_monotone"] = out["monotone_w_wrong"] and out["spread_logit_w_wrong"] > 0.1
    out["gate_nll_monotone"] = out["monotone_nll_wrong"] and out["spread_nll_wrong"] > 0.0
    out["gate_control_flat"] = bool(out["control_flat_w"] < 0.1)
    return out


def run_seed(scale, prior, seed, dose, eval_seed, smoke=False):
    out = dict(scale=scale, prior=prior, model_seed=seed, dose=dose, eval_seed=eval_seed)
    model = C.load_pfn(scale, prior, seed, dose)
    rng = np.random.default_rng(eval_seed)
    out["e2a"] = run_e2a(model, prior, rng, smoke=smoke)
    out["e2b"] = run_e2b(model, prior, rng, smoke=smoke)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scale", default="base")
    ap.add_argument("--prior", default="AL40")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--dose", type=int, default=12000)
    ap.add_argument("--eval-seed-base", type=int, default=201)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()
    for si, seed in enumerate(seeds):
        eval_seed = a.eval_seed_base + si
        fname = f"exp2_{a.prior}_{a.scale}_s{seed}_dose{a.dose}.json"
        fpath = os.path.join(RESULT_DIR, fname)
        if a.resume and os.path.exists(fpath):
            print(f"[{seed}] exists, skip", flush=True)
            continue
        print(f"[{seed}] eval_seed={eval_seed} starting", flush=True)
        res = run_seed(a.scale, a.prior, seed, a.dose, eval_seed, smoke=a.smoke)
        C.save_partial(fpath, res)
    print(f"all seeds done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
