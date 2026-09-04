#!/usr/bin/env python3
"""Shared core for the BINARY-DUEL experiment line (PREREG-BINARY-DUEL.md).

Implements the exact-substrate pieces that all three experiments share:
  - context/latent generation (keeps Sigma, cls, params alongside each panel)
  - the order-conditioned predictives p_G1, p_G2 (single-Sigma analytic anchors)
  - the model's prediction-space order evidence s(D) and KL-mixture weight w(D)
  - matched-pair mirror construction for cross-scoring
  - regression helpers (Theil-Sen + OLS + Spearman + bootstrap CI)

All numbers are numpy floats; scripts emit numbers + gate booleans only.
"""
import os
import sys
import math
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "infra")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: F401  (path setup)
import evidence_core as EC

# --- substrate constants (identical to e21_fleet / dose_response.py) ---------
N_CONTEXT = 30
N_QUERY = 7
N_BINS = 100
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
Q0 = np.linspace(-3, 3, N_QUERY)       # training query grid
Q1 = np.linspace(-4, 4, 9)             # robustness grid
QUERY_CLIP = 2.8                       # fresh-outcome query rejection bound (inside Q0)
MIX_GRID = np.linspace(0.0, 1.0, 2001) # mixture-weight fit grid

# Fix-B prior bounds (mirror experiment_v3bump)
A_LO, A_HI = 0.4, 1.0
B_LO, B_HI = 0.6, 1.5
RHO_MAG_LO, RHO_MAG_HI = 0.4, 0.8

NULL_TOK = EC.NULL_TOK


# ---------------------------------------------------------------------------
# generation with latent preserved
# ---------------------------------------------------------------------------

def gen_context_with_latent(prior, rng):
    """Draw (Sigma, cls, params) and one (N_CONTEXT, 2) panel.

    Returns (D, S, cls, params). cls in {1,2}; params is (a, b1, b2) for that
    order via sigma_to_params_G{cls}.
    """
    S, *_ = EC.E.sample_valid_Sigma(rng)
    cls = int(rng.integers(1, 3))
    params = EC.E.sigma_to_params_G1(S) if cls == 1 else EC.E.sigma_to_params_G2(S)
    D = EC._sample_context(prior, cls, params, rng)
    return D, S, cls, params


def gen_bank(prior, n, rng, balanced=False, bal_window=(-6, 6), bal_dense=(-3, 3)):
    """Generate n contexts with latent info. Returns dict with lists.

    balanced: stratify ℓ into the informative window (mirrors evidence_core's
    generate_fixed_panel logic but keeps Sigma/cls/params).
    """
    out = {"D": [], "y": [], "ell": [], "S": [], "cls": [], "params": []}
    if not balanced:
        while len(out["D"]) < n:
            D, S, cls, params = gen_context_with_latent(prior, rng)
            ell = EC.compute_ell(D, prior)
            if not np.isfinite(ell):
                continue
            out["D"].append(D); out["y"].append(1 if cls == 1 else 0)
            out["ell"].append(float(ell)); out["S"].append(S)
            out["cls"].append(cls); out["params"].append(params)
        return out

    # balanced: generate a pool, stratify on |ℓ| inside the informative window
    pool = {"D": [], "y": [], "ell": [], "S": [], "cls": [], "params": []}
    n_pool = int(n * 3)
    while len(pool["D"]) < n_pool:
        D, S, cls, params = gen_context_with_latent(prior, rng)
        ell = EC.compute_ell(D, prior)
        if not np.isfinite(ell):
            continue
        pool["D"].append(D); pool["y"].append(1 if cls == 1 else 0)
        pool["ell"].append(float(ell)); pool["S"].append(S)
        pool["cls"].append(cls); pool["params"].append(params)

    ell = np.asarray(pool["ell"], float)
    inner = (ell >= bal_window[0]) & (ell <= bal_window[1])
    dense = (ell >= bal_dense[0]) & (ell <= bal_dense[1])
    n_inner = min(int(n * 0.7), int(inner.sum()))
    n_dense = min(int(n * 0.35), int(dense.sum()))
    n_outer = n - n_inner
    inner_idx = np.where(inner)[0]; dense_idx = np.where(dense)[0]; outer_idx = np.where(~inner)[0]
    rng.shuffle(inner_idx); rng.shuffle(dense_idx); rng.shuffle(outer_idx)
    sel = list(dense_idx[:n_dense])
    rem = [i for i in inner_idx[:n_inner] if i not in sel]
    sel.extend(rem[: n_inner - len(sel)])
    need = n - len(sel)
    if need > 0:
        sel.extend(outer_idx[:need])
    sel = np.asarray(sorted(sel))
    for k in out:
        out[k] = [pool[k][i] for i in sel]
    return out


def _noise_std_resid(prior, n, rng):
    """Standardized residuals (mean 0, Var 2) from the family, scale 1."""
    if prior == "N":
        return rng.normal(0.0, math.sqrt(2.0), n)
    r = 1.0 if prior == "L" else int(prior[2:]) / 10.0
    c = math.sqrt(2.0 * 1.0 * 1.0 / (1.0 + r * r))
    a_ = r * c
    return rng.exponential(a_, n) - rng.exponential(c, n) - (a_ - c)


def context_from_std(prior, cls, params, z1, z2):
    """Build a panel from standardized residuals z1,z2 under the given order.

    cls==1 (G1): x = b1 z1, y = a x + b2 z2.
    cls==2 (G2): y = b1 z1, x = a y + b2 z2.
    """
    a, b1, b2 = params
    if cls == 1:
        x = b1 * z1
        return np.stack([x, a * x + b2 * z2], axis=1)
    y = b1 * z1
    return np.stack([a * y + b2 * z2, y], axis=1)


def matched_pair(prior, rng, n_points=N_CONTEXT):
    """Draw one matched pair: D_true (true order) + D_mirror (other order).

    Both use the SAME Sigma and the SAME standardized residual draws z1,z2, so
    they are two order-symmetrised realizations of one covariance matrix.

    Returns dict: D_true, D_mirror, S, cls_true (1 or 2), params_true,
    params_mirror, z1, z2.
    """
    S, *_ = EC.E.sample_valid_Sigma(rng)
    cls = int(rng.integers(1, 3))
    pT = EC.E.sigma_to_params_G1(S) if cls == 1 else EC.E.sigma_to_params_G2(S)
    pM = EC.E.sigma_to_params_G2(S) if cls == 1 else EC.E.sigma_to_params_G1(S)
    z1 = _noise_std_resid(prior, n_points, rng)
    z2 = _noise_std_resid(prior, n_points, rng)
    D_true = context_from_std(prior, cls, pT, z1, z2)
    D_mirror = context_from_std(prior, 3 - cls, pM, z1, z2)
    return dict(D_true=D_true, D_mirror=D_mirror, S=S, cls_true=cls,
                params_true=pT, params_mirror=pM, z1=z1, z2=z2)


def fresh_outcomes(prior, cls, params, M, rng, x_clip=QUERY_CLIP):
    """Draw M fresh outcomes from the true process, queries kept inside |x*|<=clip.

    G1: x* = b1 z1*, y* = a x* + b2 z2*;  G2 mirrored. Returns (x_star, y_star).
    """
    a, b1, b2 = params
    xs, ys = [], []
    while len(xs) < M:
        z1 = _noise_std_resid(prior, 1, rng)[0]
        if cls == 1:
            x_star = b1 * z1
        else:
            y_exo = b1 * z1
            x_star = a * y_exo + b2 * _noise_std_resid(prior, 1, rng)[0]
            y_star = y_exo
        if abs(x_star) > x_clip:
            continue
        if cls == 1:
            y_star = a * x_star + b2 * _noise_std_resid(prior, 1, rng)[0]
        xs.append(x_star); ys.append(y_star)
    return np.asarray(xs), np.asarray(ys)


# ---------------------------------------------------------------------------
# order-conditioned predictives (analytic anchors)
# ---------------------------------------------------------------------------

def _predictive_logpdf_centered(y, loc, scale, prior, bin_width=0.16):
    """Log probability mass of y in a bin under the CENTERED family density.

    The generative process centers the AL (mean 0, Var 2*scale^2) by subtracting
    (a_-c) from the raw AL draw (e21_fleet.al_noise / evidence_core.resid_logpdf_vec).
    EC._predictive_logpdf_scalar OMITS this centering shift (evaluates the raw AL
    at (y-loc)/scale), so its mode sits +(a_-c) above the true one. This function
    applies the shift exactly as the verified oracle does.
    """
    z = (y - loc) / (scale * math.sqrt(2) + 1e-9)
    if prior.startswith("AL") or prior == "L":
        r = 1.0 if prior == "L" else int(prior[2:]) / 10.0
        c = math.sqrt(2.0 / (1.0 + r * r))
        a_ = r * c
        # centered residual in standardized units, matching resid_logpdf_vec:
        z_centered = z * math.sqrt(2.0) + (a_ - c)   # (y-loc)/scale + (a_-c)
        if z_centered >= 0:
            lp = -z_centered / a_ - math.log(a_ + c)
        else:
            lp = z_centered / c - math.log(a_ + c)
        return lp - math.log(scale) + math.log(bin_width)
    return EC._predictive_logpdf_scalar(y, loc, scale, prior, bin_width=bin_width)


def _bin_logp(y_values, loc, scale, prior):
    """log probability mass of the given values under the CENTERED family density.

    loc may be a scalar or array parallel to y_values (broadcast pairwise).
    """
    y_values = np.asarray(y_values, float)
    loc_arr = np.asarray(loc, float)
    if loc_arr.ndim == 0:
        loc_arr = np.full_like(y_values, float(loc_arr))
    assert loc_arr.shape == y_values.shape, (loc_arr.shape, y_values.shape)
    return np.asarray([_predictive_logpdf_centered(float(y), float(l), float(scale), prior)
                       for y, l in zip(y_values, loc_arr)], float)


def anchor_logprobs(params_G1, params_G2, prior, qx_values):
    """Normalized (log) order-conditioned predictives for query values qx_values.

    p_G1(y*|x*)  = f(y* - a1 x*; b12)                        (G1 conditional)
    p_G2(y*|x*)  = f(x* - a2 y*; b22) * f(y*; b21)           (G2 joint, normalized over y*)

    Returns (logpG1, logpG2) each (len(qx_values), N_BINS), log-normalized rows.
    """
    a1, b11, b12 = params_G1
    a2, b21, b22 = params_G2
    nq = len(qx_values)
    logpG1 = np.zeros((nq, N_BINS))
    logpG2 = np.zeros((nq, N_BINS))
    for qi, x in enumerate(qx_values):
        logpG1[qi] = _bin_logp(BIN_CENTERS, a1 * x, b12, prior)
        # G2 joint over y* at fixed x*:
        logpG2[qi] = _bin_logp(BIN_CENTERS, 0.0, b21, prior) + _bin_logp(
            np.full(N_BINS, float(x)), a2 * BIN_CENTERS, b22, prior)
        # note: _bin_logp(BIN_CENTERS, 0.0, b21) = f(y*; b21); the second term is
        # f(x* - a2 y*; b22) via loc=a2*y_bin evaluated at observation x*.
        logpG1[qi] -= logsumexp(logpG1[qi])
        logpG2[qi] -= logsumexp(logpG2[qi])
    return logpG1, logpG2


def logsumexp(a):
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def anchor_kl(logpG1, logpG2):
    """Mean over queries of KL(p_G1 || p_G2) — 0 if anchors coincide."""
    p1 = np.exp(logpG1); p2 = np.exp(logpG2)
    return float(np.mean(np.sum(p1 * (logpG1 - logpG2), axis=1)))


# ---------------------------------------------------------------------------
# model prediction-space order evidence
# ---------------------------------------------------------------------------

def model_predictive(model, D, qx_values):
    """(nq, N_BINS) log-probabilities of the model for one context."""
    D = np.asarray(D, np.float32)[None, :, :]
    qx = np.asarray(qx_values, np.float32)[None, :, None]
    return EC.model_predictive_bins(model, D, qx)[0]


def evidence_scores(logQ, logpG1, logpG2, min_kl=1e-6):
    """s(D) and logit w(D) for a context.

    s(D)      = mean over queries of E_Q[log p_G1 - log p_G2]
    w(D)      = global KL-mixture weight toward G2 over the query grid
    Returns (s, logit_w). logit_w = nan if the anchors coincide (unidentifiable).
    """
    p1 = np.exp(logpG1); p2 = np.exp(logpG2)
    s_per_q = np.sum(np.exp(logQ) * (logpG1 - logpG2), axis=1)
    s = float(np.mean(s_per_q))
    if anchor_kl(logpG1, logpG2) < min_kl:
        return s, float("nan")
    Q = np.exp(logQ)  # (nq, N_BINS)
    obj = np.array([np.sum(Q * np.log((1 - w) * p1 + w * p2 + 1e-30)) for w in MIX_GRID])
    w = float(MIX_GRID[int(np.argmax(obj))])
    return s, _logit(np.clip(w, 1e-6, 1 - 1e-6))


def _logit(p):
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# regression helpers
# ---------------------------------------------------------------------------

def fit_evidence_regression(x, y, n_boot=500, rng_seed=0):
    """Theil-Sen + OLS + Spearman + monotone fraction + bootstrap CI of Theil-Sen slope.

    x, y: 1-D arrays (ℓ, and logit w or s). Returns a dict of floats.
    monotone_frac = fraction of pairs (i<j, x-sorted ascending) with y_j > y_i.
    Sign-informative: ~1 for a monotone increasing relation, ~0 for monotone
    decreasing (e.g. logit w vs ℓ tracks decreasing, so mono_w near 0 is correct).
    Bootstrap CI: resamples of size min(n, 600) keep Theil-Sen O(n^2) cheap at
    n=1500 while the point estimate stays the exact full-n slope.
    """
    from scipy import stats as st
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return dict(n=0)
    ts = st.theilslopes(y, x)
    ols = np.polyfit(x, y, 1)
    rho, pval = st.spearmanr(x, y)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # concordant pairs (sign of Δy matches sign of Δx), Kendall-style
    dx = xs[None, :] - xs[:, None]   # (i<j) triangle
    dy = ys[None, :] - ys[:, None]
    iu = np.triu_indices(len(xs), k=1)
    if len(iu[0]):
        sig = np.sign(dx[iu]) * np.sign(dy[iu])
        sig = sig[sig != 0]
        mono = float(np.mean(sig > 0)) if len(sig) else float("nan")
    else:
        mono = float("nan")
    rng = np.random.default_rng(rng_seed)
    n_bs = min(len(x), 600)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), n_bs)
        t = st.theilslopes(y[idx], x[idx])
        bs.append(t[0])
    bs = np.asarray(bs)
    return dict(n=len(x),
                theil_slope=float(ts[0]), theil_int=float(ts[1]),
                ols_slope=float(ols[0]), ols_int=float(ols[1]),
                spearman_rho=float(rho), spearman_p=float(pval),
                monotone_frac=mono,
                slope_ci_lo=float(np.percentile(bs, 2.5)),
                slope_ci_hi=float(np.percentile(bs, 97.5)),
                slope_ci_incl0=bool((np.percentile(bs, 2.5) <= 0) and (np.percentile(bs, 97.5) >= 0)))


def wilcoxon_paired(delta):
    """One-sided (positive) signed-rank p for paired deltas; None if degenerate."""
    from scipy import stats as st
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    if len(d) < 8 or np.allclose(d, 0):
        return None
    stat, p = st.wilcoxon(d, alternative="greater")
    return float(p)


def mean_nll_at_outcomes(logQ, y_star):
    """Mean negative log-likelihood of the model predictive at the given outcomes.

    logQ: (M, N_BINS) log-probs, one row per outcome (query x* aligned row-wise).
    Bins via bin_y(y*).
    """
    bins = np.searchsorted(BIN_EDGES[1:-1], np.asarray(y_star)).clip(0, N_BINS - 1)
    idx = (np.arange(len(bins)), bins)
    return float(-np.mean(logQ[idx]))


# ---------------------------------------------------------------------------
# provenance + result helpers
# ---------------------------------------------------------------------------

def checkpoint_path(scale, prior, seed, dose):
    return config.DOSE_NETS / f"M_{scale}_{prior}_s{seed}_dose{dose}.pt"


def load_pfn(scale, prior, seed, dose):
    p = checkpoint_path(scale, prior, seed, dose)
    if not p.exists():
        raise FileNotFoundError(p)
    return EC.load_model(str(p), scale=scale)


def save_partial(path, data):
    path = str(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)
    print(f"  saved {path}", flush=True)


def load_partial(path):
    path = str(path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    D, S, cls, params = gen_context_with_latent("AL40", rng)
    print("context:", D.shape, "cls", cls, "params", tuple(round(v, 3) for v in params),
          "ell %.3f" % EC.compute_ell(D, "AL40"))
    p1 = EC.E.sigma_to_params_G1(S); p2 = EC.E.sigma_to_params_G2(S)
    lg1, lg2 = anchor_logprobs(p1, p2, "AL40", Q0)
    print("anchors:", lg1.shape, "KL(pG1||pG2) per-query mean %.3f" % anchor_kl(lg1, lg2))
    mp = matched_pair("AL40", rng)
    print("matched pair: true cls", mp["cls_true"],
          "ell_true %.2f" % EC.compute_ell(mp["D_true"], "AL40"),
          "ell_mirror %.2f" % EC.compute_ell(mp["D_mirror"], "AL40"))
    xs, ys = fresh_outcomes("AL40", 1, p1, 5, rng)
    print("fresh outcomes:", xs, ys)
    bank = gen_bank("AL40", 60, rng)
    print("bank: n=%d ell∈[%.2f, %.2f]" % (len(bank["D"]), min(bank["ell"]), max(bank["ell"])))
    bankb = gen_bank("AL40", 60, rng, balanced=True)
    print("balanced bank: n=%d ell∈[%.2f, %.2f]" % (len(bankb["D"]), min(bankb["ell"]), max(bankb["ell"])))
    print("core.py OK")
