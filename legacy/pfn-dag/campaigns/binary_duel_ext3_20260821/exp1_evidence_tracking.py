#!/usr/bin/env python3
"""EXPERIMENT 1 — behavioral use of order evidence (PREREG-BINARY-DUEL.md §2).

Estimands (primary model: AL40 base, dose12000, seeds 0-3):
  E1a  evidence tracking: logit w(D) ~ ell(D) (Theil-Sen; also OLS, Spearman,
       monotone-fraction, bootstrap CI). s(D) slope as secondary.
  E1b  matched-pair cross-scoring on fresh outcomes: gain_spec vs gain_neut.
  E1c  controls: Gaussian-null, dose0 init, emergence ladder, seed survival,
       mechanism survival splits.

Numbers + gate booleans only. Per-seed result JSONs with resume.
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

RESULT_DIR = os.path.join(_HERE, "results", "exp1")
os.makedirs(RESULT_DIR, exist_ok=True)


def both_param_sets(S):
    return (EC.E.sigma_to_params_G1(S), EC.E.sigma_to_params_G2(S))


def context_scores(model, D, S, prior, qx_grid=C.Q0):
    """s(D) and logit w(D) for one context."""
    p1, p2 = both_param_sets(S)
    logpG1, logpG2 = C.anchor_logprobs(p1, p2, prior, qx_grid)
    logQ = C.model_predictive(model, D, qx_grid)
    return C.evidence_scores(logQ, logpG1, logpG2)


def score_bank(model, bank, prior, qx_grid=C.Q0, bs=256):
    """s and logit w for every context in a bank (batched forward)."""
    Ds = np.asarray([d for d in bank["D"]], np.float32)
    qx = np.repeat(np.asarray(qx_grid, np.float32)[None, :, None], len(Ds), axis=0)  # (n, nq, 1)
    logQ_all = EC.model_predictive_bins(model, Ds, qx)  # (n, nq, 100)
    s_list, lw_list = [], []
    for i, S in enumerate(bank["S"]):
        p1, p2 = both_param_sets(S)
        logpG1, logpG2 = C.anchor_logprobs(p1, p2, prior, qx_grid)
        s, lw = C.evidence_scores(logQ_all[i], logpG1, logpG2)
        s_list.append(s); lw_list.append(lw)
    return np.asarray(s_list), np.asarray(lw_list)


def run_e1a(model, bank, prior, tag, seed_meta):
    """E1a estimands on one bank. Returns dict of regression stats + raw arrays."""
    ell = np.asarray(bank["ell"], float)
    s_vals, lw_vals = score_bank(model, bank, prior)
    reg_w = C.fit_evidence_regression(ell, lw_vals, rng_seed=seed_meta["eval_seed"])
    reg_s = C.fit_evidence_regression(ell, s_vals, rng_seed=seed_meta["eval_seed"])
    return dict(tag=tag, **seed_meta,
                n=len(ell),
                ell_min=float(ell.min()), ell_max=float(ell.max()),
                reg_w=reg_w, reg_s=reg_s,
                nan_w=int(np.isnan(lw_vals).sum()))


def run_e1b(model, prior, n_pairs, M, rng, tag, seed_meta, n_out_clip=2.8):
    """E1b + control 6. Matched pairs; cross-scoring gain on fresh outcomes.

    Returns dict with per-pair deltas + gates.
    """
    # pre-draw all pairs + outcomes (deterministic per rng)
    pairs = [C.matched_pair(prior, rng) for _ in range(n_pairs)]
    res = []
    for pr in pairs:
        cls = pr["cls_true"]
        pT = pr["params_true"]; pM = pr["params_mirror"]
        xA, yA = C.fresh_outcomes(prior, cls, pT, M, rng)
        # query-aligned model predictives
        lq_true = C.model_predictive(model, pr["D_true"], xA)
        lq_mirror = C.model_predictive(model, pr["D_mirror"], xA)
        nll_true = C.mean_nll_at_outcomes(lq_true, yA)
        nll_mirror = C.mean_nll_at_outcomes(lq_mirror, yA)
        gain_spec = nll_mirror - nll_true
        # control 6: independent same-order context (same Sigma, fresh residuals)
        z1 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
        z2 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
        D_same = C.context_from_std(prior, cls, pT, z1, z2)
        lq_same = C.model_predictive(model, D_same, xA)
        nll_same = C.mean_nll_at_outcomes(lq_same, yA)
        gain_neut = nll_same - nll_true
        res.append(dict(cls_true=int(cls), gain_spec=gain_spec, gain_neut=gain_neut,
                        nll_true=nll_true, nll_mirror=nll_mirror, nll_same=nll_same))
    gain_spec = np.array([r["gain_spec"] for r in res])
    gain_neut = np.array([r["gain_neut"] for r in res])
    g1 = np.array([r["gain_spec"] for r in res if r["cls_true"] == 1])
    g2 = np.array([r["gain_spec"] for r in res if r["cls_true"] == 2])
    return dict(tag=tag, **seed_meta, n_pairs=len(res),
                gain_spec_mean=float(np.nanmean(gain_spec)),
                gain_spec_median=float(np.nanmedian(gain_spec)),
                gain_spec_wilcox_p=C.wilcoxon_paired(gain_spec),
                gain_spec_cls1_mean=float(np.nanmean(g1)), gain_spec_cls1_n=int(len(g1)),
                gain_spec_cls2_mean=float(np.nanmean(g2)), gain_spec_cls2_n=int(len(g2)),
                gain_neut_mean=float(np.nanmean(gain_neut)),
                gain_neut_wilcox_p=C.wilcoxon_paired(gain_neut),
                gate_spec = bool(C.wilcoxon_paired(gain_spec) is not None
                                 and C.wilcoxon_paired(gain_spec) < 0.01
                                 and np.nanmean(gain_spec) > 0),
                gate_ctrl6 = bool(np.nanmean(gain_spec) > np.nanmean(gain_neut)))


def positive_control_bayes(ell, S_list, prior, qx_grid=C.Q0, rng_seed=0):
    """Harness check: perfect-Bayes Q = sigmoid(ell) pG1 + sigmoid(-ell) pG2
    must yield logit w = -ell exactly (beta_w == -1)."""
    lw = []
    for e, S in zip(ell, S_list):
        p1, p2 = both_param_sets(S)
        logpG1, logpG2 = C.anchor_logprobs(p1, p2, prior, qx_grid)
        p1p = np.exp(logpG1); p2p = np.exp(logpG2)
        sig = 1.0 / (1.0 + np.exp(-e))
        Q = sig * p1p + (1.0 - sig) * p2p
        _, lwv = C.evidence_scores(np.log(Q + 1e-30), logpG1, logpG2)
        lw.append(lwv)
    lw = np.asarray(lw)
    m = np.isfinite(lw) & np.isfinite(ell)
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    ts = C.fit_evidence_regression(ell[m], lw[m], rng_seed=rng_seed)
    # expected beta_w = -1: report deviation
    return dict(n=int(m.sum()),
                beta_w=ts["theil_slope"],
                beta_dev_from_neg1=float(ts["theil_slope"] - (-1.0)))


def mechanism_splits(ell, s_vals, lw_vals, bank, rng_seed=0):
    """E1c.5: E1a split by |a| and |rho| bands of the context's Sigma."""
    a_vals = np.asarray([p[0] for p in bank["params"]], float)
    rho_vals = np.asarray([S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]) for S in bank["S"]], float)
    out = {}
    a_low = np.abs(a_vals) <= np.median(np.abs(a_vals))
    a_high = ~a_low
    r_low = np.abs(rho_vals) <= np.median(np.abs(rho_vals))
    r_high = ~r_low
    for name, m in [("a_low", a_low), ("a_high", a_high),
                    ("rho_low", r_low), ("rho_high", r_high)]:
        if m.sum() < 20:
            out[name] = dict(n=int(m.sum()))
            continue
        out[name] = dict(n=int(m.sum()),
                         reg_w=C.fit_evidence_regression(ell[m], lw_vals[m], rng_seed=rng_seed))
    return out


def run_seed(scale, prior, seed, dose, eval_seed, prior_n, smoke=False):
    """Run all E1 estimands for one (model seed, dose). Returns result dict."""
    t0 = time.time()
    out = dict(scale=scale, prior=prior, model_seed=seed, dose=dose, eval_seed=eval_seed)
    model = C.load_pfn(scale, prior, seed, dose)
    rng = np.random.default_rng(eval_seed)

    n_nat = 60 if smoke else 1500
    n_bal = 60 if smoke else 1500
    n_null = 60 if smoke else 1000
    n_pairs = 60 if smoke else 800
    M = 5 if smoke else 20

    # E1a — natural + balanced
    nat = C.gen_bank(prior, n_nat, rng)
    bal = C.gen_bank(prior, n_bal, rng, balanced=True)
    out["e1a_natural"] = run_e1a(model, nat, prior, "natural", dict(eval_seed=eval_seed))
    out["e1a_balanced"] = run_e1a(model, bal, prior, "balanced", dict(eval_seed=eval_seed))

    # positive control on the natural bank
    ell = np.asarray(nat["ell"], float)
    out["positive_control"] = positive_control_bayes(ell, nat["S"], prior, rng_seed=eval_seed)

    # mechanism splits (natural bank)
    s_vals, lw_vals = score_bank(model, nat, prior)
    out["mech_splits"] = mechanism_splits(ell, s_vals, lw_vals, nat, rng_seed=eval_seed)

    # E1b + control 6
    out["e1b_crossscore"] = run_e1b(model, prior, n_pairs, M, rng, "matched", dict(eval_seed=eval_seed))

    # E1c.1 — Gaussian-null bank (same model)
    null = C.gen_bank("N", n_null, rng)
    s_n, lw_n = score_bank(model, null, "N")
    out["e1c_gaussian_null"] = dict(n=len(null["D"]),
                                    mean_abs_s=float(np.nanmean(np.abs(s_n))),
                                    max_abs_s=float(np.nanmax(np.abs(s_n))),
                                    frac_finite_w=float(np.isfinite(lw_n).mean()))
    # Gaussian-null cross-scoring: same matched-pair machinery under N
    out["e1c_gaussian_crossscore"] = run_e1b(model, "N", max(60, n_pairs // 4), M, rng,
                                             "gaussian_null", dict(eval_seed=eval_seed))
    out["wallclock_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scale", default="base")
    ap.add_argument("--prior", default="AL40")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--dose", type=int, default=12000)
    ap.add_argument("--eval-seed-base", type=int, default=101)
    ap.add_argument("--resume", action="store_true", help="skip completed per-seed files")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()
    for si, seed in enumerate(seeds):
        eval_seed = a.eval_seed_base + si
        fname = f"exp1_{a.prior}_{a.scale}_s{seed}_dose{a.dose}.json"
        fpath = os.path.join(RESULT_DIR, fname)
        if a.resume and os.path.exists(fpath):
            print(f"[{seed}] exists, skip", flush=True)
            continue
        print(f"[{seed}] eval_seed={eval_seed} starting", flush=True)
        res = run_seed(a.scale, a.prior, seed, a.dose, eval_seed, a.prior, smoke=a.smoke)
        C.save_partial(fpath, res)
    print(f"all seeds done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
