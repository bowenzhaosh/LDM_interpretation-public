#!/usr/bin/env python3
"""EXPERIMENT 4 (PREREG-EXT3 §2) — d=3 port of the binary-duel evidence readout.

d=3 substrate: 3-variable complete-DAG SEM, permuted-LDL construction, AL skew r,
6 orderings. Duel: pi_A=(0,1,2) [x2 = sink] vs pi_B=(0,2,1) [x2 = middle] — a single
adjacent transposition (x1 <-> x2) holding x0 as root; the d=3 analog of the d=2
G1<->G2 swap. Model PFN4 predicts canonical x2 | (x0,x1) + context D.

Estimands (identical form to d=2 E1a/E1b):
  E1a: regress logit w(D) ~ ell(D)   (ideal Bayes beta_w = -1)
  E1b: matched-pair cross-scoring on fresh outcomes (gain_spec vs gain_neut)
  controls: positive control (perfect-Bayes), Gaussian-null, dose ladder.

Construction is loaded VERBATIM from d4plus_oracle.py (importlib, argv ["..","3"]).
Numbers + gate booleans only; per-(seed,dose) result JSONs with resume.
"""
import os
import sys
import math
import json
import argparse
import time
import importlib.util

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# substrate constants (d=3)
# ---------------------------------------------------------------------------
D_DIM = 3
N_CONTEXT = 30
N_QUERY = 9
N_BINS = 100
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
BIN_WIDTH = float(BIN_EDGES[1] - BIN_EDGES[0])
LOG_BW = math.log(BIN_WIDTH)
MIX_GRID = np.linspace(0.0, 1.0, 2001)
QUERY_CLIP = 2.8

PI_A = (0, 1, 2)   # x0 root, x1 middle, x2 = sink
PI_B = (0, 2, 1)   # x0 root, x2 = middle, x1 = sink
R_OF = {"A": 2.0, "C": 4.0}

# base model config (nets3 M3_* base nets)
D_MODEL, D_FF, N_HEADS, N_LAYERS = 256, 512, 4, 2
NULL_TOK = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _grid(lo, hi, n):
    xs = np.linspace(lo, hi, n)
    return np.array([(x0, x1) for x0 in xs for x1 in xs], dtype=float)


Q0 = _grid(-2, 2, 3)   # 9 queries
Q1 = _grid(-3, 3, 4)   # 16 robustness queries


# ---------------------------------------------------------------------------
# construction: load d4plus_oracle.py VERBATIM (D_DIM=3)
# ---------------------------------------------------------------------------
def load_orc(path=None):
    path = path or os.environ.get("D4ORC", os.path.join(_HERE, "d4plus_oracle.py"))
    argv_bak = sys.argv[:]
    sys.argv = ["d4plus_oracle.py", "3"]
    spec = importlib.util.spec_from_file_location("orc3", path)
    orc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orc)
    sys.argv = argv_bak
    return orc


# ---------------------------------------------------------------------------
# model (PFN4 base, verbatim from d3_train_fleet.cluster.py)
# ---------------------------------------------------------------------------
class PFN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_embed = nn.Linear(D_DIM, D_MODEL)
        self.query_embed = nn.Linear(D_DIM - 1, D_MODEL)
        self.token_embed = nn.Embedding(3, D_MODEL)
        enc = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
                                         batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.out_head = nn.Linear(D_MODEL, N_BINS)

    def forward(self, ctx, qxy, tok):
        ce = self.point_embed(ctx)
        te = self.token_embed(tok).unsqueeze(1)
        outs = []
        for q in range(qxy.shape[1]):
            qe = self.query_embed(qxy[:, q, :]).unsqueeze(1)
            outs.append(self.out_head(self.transformer(torch.cat([te, ce, qe], 1))[:, -1, :]))
        return torch.stack(outs, 1)


def ckpt_path(prior, seed, dose, nets_dir):
    if dose == 20000:
        return os.path.join(nets_dir, f"M3_{prior}_s{seed}_st20000.pt")
    return os.path.join(nets_dir, f"M3_{prior}_s{seed}_st20000_ck{dose}.pt")


def load_model(prior, seed, dose, nets_dir):
    p = ckpt_path(prior, seed, dose, nets_dir)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    model = PFN4()
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.to(DEV).eval()
    return model


# ---------------------------------------------------------------------------
# math helpers (verbatim from d=2 core.py)
# ---------------------------------------------------------------------------
def logsumexp(a):
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def _logit(p):
    return math.log(p / (1.0 - p))


def anchor_kl(logp1, logp2):
    p1 = np.exp(logp1)
    p2 = np.exp(logp2)
    return float(np.mean(np.sum(p1 * (logp1 - logp2), axis=1)))


def evidence_scores(logQ, logp1, logp2, min_kl=1e-6):
    """s(D) and logit w(D). w toward pi_B. Returns (s, logit_w); logit_w=nan if
    anchors coincide (unidentifiable)."""
    p1 = np.exp(logp1)
    p2 = np.exp(logp2)
    s_per_q = np.sum(np.exp(logQ) * (logp1 - logp2), axis=1)
    s = float(np.mean(s_per_q))
    if anchor_kl(logp1, logp2) < min_kl:
        return s, float("nan")
    Q = np.exp(logQ)
    obj = np.array([np.sum(Q * np.log((1 - w) * p1 + w * p2 + 1e-30)) for w in MIX_GRID])
    w = float(MIX_GRID[int(np.argmax(obj))])
    return s, _logit(np.clip(w, 1e-6, 1 - 1e-6))


def fit_evidence_regression(x, y, n_boot=500, rng_seed=0):
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
    dx = xs[None, :] - xs[:, None]
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
        bs.append(st.theilslopes(y[idx], x[idx])[0])
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
    from scipy import stats as st
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    if len(d) < 8 or np.allclose(d, 0):
        return None
    return float(st.wilcoxon(d, alternative="greater")[1])


def mean_nll_per_row(logQ, y):
    """Per-row mean NLL. logQ: (n, M, N_BINS) log-probs, y: (n, M) outcomes."""
    bins = np.searchsorted(BIN_EDGES[1:-1], np.asarray(y)).clip(0, N_BINS - 1)
    n, M = bins.shape
    rows = np.repeat(np.arange(n), M)
    cols = bins.ravel()
    vals = logQ.reshape(-1, N_BINS)[rows, cols]
    return -vals.reshape(n, M).mean(axis=1)


# ---------------------------------------------------------------------------
# generation (d=3)
# ---------------------------------------------------------------------------
def gen_context(prior, rng, orc, oracle):
    """Draw one context under pi_A (cls=1) or pi_B (cls=2), with ell(D)."""
    S = orc.sample_Sigmas(rng, 1)[0]
    cls = int(rng.integers(0, 2)) + 1
    pi = PI_A if cls == 1 else PI_B
    r = R_OF.get(prior, 2.0)
    D = orc.gen_data(S, orc.ORDERINGS.index(pi), r, N_CONTEXT, rng)
    post = oracle.posterior(D, r)
    ia = orc.ORDERINGS.index(PI_A)
    ib = orc.ORDERINGS.index(PI_B)
    ell = math.log(post[ia]) - math.log(post[ib])
    return D, S, cls, ell


def gen_bank(prior, n, rng, orc, oracle, balanced=False):
    Ds, ells, Ss, clss = [], [], [], []
    if not balanced:
        while len(Ds) < n:
            D, S, cls, ell = gen_context(prior, rng, orc, oracle)
            if not np.isfinite(ell):
                continue
            Ds.append(D); ells.append(float(ell)); Ss.append(S); clss.append(cls)
        return dict(D=Ds, ell=ells, S=Ss, cls=clss)
    # balanced: pool 3n, then spread ell uniformly in rank (scale-free stratification)
    pool = []
    while len(pool) < 3 * n:
        D, S, cls, ell = gen_context(prior, rng, orc, oracle)
        if not np.isfinite(ell):
            continue
        pool.append(dict(D=D, ell=float(ell), S=S, cls=cls))
    order = np.argsort([p["ell"] for p in pool])
    idx = np.linspace(0, len(order) - 1, n).astype(int)
    for i in idx:
        p = pool[order[i]]
        Ds.append(p["D"]); ells.append(p["ell"]); Ss.append(p["S"]); clss.append(p["cls"])
    return dict(D=Ds, ell=ells, S=Ss, cls=clss)


def context_from_std3(S, pi, z, orc):
    """x from standardized residuals z (K,3) assigned to positions root/mid/sink."""
    Lunit, U, b = orc.params_for(S[None], pi)
    Lunit, b = Lunit[0], b[0]
    e = z * b[None, :]
    xpi = e @ Lunit.T
    x = np.empty_like(xpi)
    x[:, list(pi)] = xpi
    return x


def matched_pair(prior, rng, orc, n_points=N_CONTEXT):
    S = orc.sample_Sigmas(rng, 1)[0]
    cls = int(rng.integers(0, 2)) + 1
    piT = PI_A if cls == 1 else PI_B
    piM = PI_B if cls == 1 else PI_A
    r = R_OF.get(prior, 2.0)
    z = orc.al_sample(1.0, r, (n_points, D_DIM), rng)
    D_true = context_from_std3(S, piT, z, orc)
    D_mirror = context_from_std3(S, piM, z, orc)
    return dict(D_true=D_true, D_mirror=D_mirror, S=S, cls_true=cls)


def fresh_outcomes(prior, cls, S, M, rng, orc, clip=QUERY_CLIP):
    r = R_OF.get(prior, 2.0)
    pi = PI_A if cls == 1 else PI_B
    Lunit, U, b = orc.params_for(S[None], pi)
    Lunit, b = Lunit[0], b[0]
    xs0, xs1, ys = [], [], []
    while len(xs0) < M:
        z = orc.al_sample(1.0, r, (1, D_DIM), rng)
        e = z * b[None, :]
        xpi = e @ Lunit.T
        x = np.empty_like(xpi)
        x[:, list(pi)] = xpi
        x0, x1, x2 = x[0]
        if abs(x0) > clip or abs(x1) > clip:
            continue
        xs0.append(x0); xs1.append(x1); ys.append(x2)
    return np.asarray(xs0), np.asarray(xs1), np.asarray(ys)


# ---------------------------------------------------------------------------
# order-conditioned predictives (analytic anchors)
# ---------------------------------------------------------------------------
def anchor_logprobs(S, prior, qx_grid, orc):
    """p_piA (x2 sink: single AL) and p_piB (x2 middle: product of two ALs),
    each (nq, N_BINS) log-normalized."""
    La, Ua, ba = orc.params_for(S[None], PI_A)
    Lb, Ub, bb = orc.params_for(S[None], PI_B)
    Ua, ba = Ua[0], ba[0]
    Lb, bb = Lb[0], bb[0]
    r = R_OF.get(prior, 2.0)
    nq = len(qx_grid)
    logpA = np.zeros((nq, N_BINS))
    logpB = np.zeros((nq, N_BINS))
    for qi, (x0, x1) in enumerate(qx_grid):
        locA = -Ua[2, 0] * x0 - Ua[2, 1] * x1
        logpA[qi] = orc.al_logpdf(BIN_CENTERS - locA, ba[2], r) + LOG_BW
        e1 = BIN_CENTERS - Lb[1, 0] * x0
        e2 = x1 - Lb[2, 0] * x0 - Lb[2, 1] * e1
        logpB[qi] = orc.al_logpdf(e1, bb[1], r) + orc.al_logpdf(e2, bb[2], r) + 2 * LOG_BW
        logpA[qi] -= logsumexp(logpA[qi])
        logpB[qi] -= logsumexp(logpB[qi])
    return logpA, logpB


def _gauss_logpdf(x, scale):
    return -0.5 * (x * x) / (2 * scale * scale + 1e-12) - 0.5 * math.log(4 * math.pi * scale * scale)


def anchor_logprobs_gauss(S, qx_grid, orc):
    """Gaussian-null anchors (coincide -> s=0, w=nan)."""
    La, Ua, ba = orc.params_for(S[None], PI_A)
    Lb, Ub, bb = orc.params_for(S[None], PI_B)
    Ua, ba = Ua[0], ba[0]
    Lb, bb = Lb[0], bb[0]
    nq = len(qx_grid)
    logpA = np.zeros((nq, N_BINS))
    logpB = np.zeros((nq, N_BINS))
    for qi, (x0, x1) in enumerate(qx_grid):
        locA = -Ua[2, 0] * x0 - Ua[2, 1] * x1
        logpA[qi] = _gauss_logpdf(BIN_CENTERS - locA, ba[2]) + LOG_BW
        e1 = BIN_CENTERS - Lb[1, 0] * x0
        e2 = x1 - Lb[2, 0] * x0 - Lb[2, 1] * e1
        logpB[qi] = _gauss_logpdf(e1, bb[1]) + _gauss_logpdf(e2, bb[2]) + 2 * LOG_BW
        logpA[qi] -= logsumexp(logpA[qi])
        logpB[qi] -= logsumexp(logpB[qi])
    return logpA, logpB


# ---------------------------------------------------------------------------
# model predictives
# ---------------------------------------------------------------------------
def model_predictive_batch(model, D, qxy):
    """(n, nq, N_BINS) log-probs. D: (n, N_CONTEXT, 3), qxy: (n, nq, 2)."""
    Dt = torch.tensor(np.asarray(D, np.float32), device=DEV)
    qx = torch.tensor(np.asarray(qxy, np.float32), device=DEV)
    n = Dt.shape[0]
    tok = torch.full((n,), NULL_TOK, dtype=torch.long, device=DEV)
    with torch.no_grad():
        logits = model(Dt, qx, tok)
    return torch.log_softmax(logits, dim=-1).cpu().numpy()


def score_bank(model, bank, prior, orc, qx_grid=Q0, bs=256):
    Ds = np.asarray([d for d in bank["D"]], np.float32)
    n = len(Ds)
    s_list, lw_list, kl_list = [], [], []
    qx = torch.tensor(np.asarray(qx_grid, np.float32)[None, :, :], device=DEV)
    tok = torch.full((1,), NULL_TOK, dtype=torch.long, device=DEV)
    for i0 in range(0, n, bs):
        batch = torch.tensor(Ds[i0:i0 + bs], device=DEV)
        with torch.no_grad():
            logits = model(batch, qx.repeat(len(batch), 1, 1), tok.repeat(len(batch)))
        logQ = torch.log_softmax(logits, dim=-1).cpu().numpy()
        for j, S in enumerate(bank["S"][i0:i0 + bs]):
            logpA, logpB = anchor_logprobs(S, prior, qx_grid, orc)
            s, lw = evidence_scores(logQ[j], logpA, logpB)
            s_list.append(s); lw_list.append(lw); kl_list.append(anchor_kl(logpA, logpB))
    return np.asarray(s_list), np.asarray(lw_list), np.asarray(kl_list)


# ---------------------------------------------------------------------------
# estimands
# ---------------------------------------------------------------------------
def run_e1a(model, bank, prior, orc, seed_meta, qx_grid=Q0):
    ell = np.asarray(bank["ell"], float)
    s_vals, lw_vals, kl_vals = score_bank(model, bank, prior, orc, qx_grid)
    reg_w = fit_evidence_regression(ell, lw_vals, rng_seed=seed_meta["eval_seed"])
    reg_s = fit_evidence_regression(ell, s_vals, rng_seed=seed_meta["eval_seed"])
    return dict(**seed_meta, n=len(ell),
                ell_min=float(ell.min()), ell_max=float(ell.max()),
                mean_anchor_kl=float(np.mean(kl_vals)),
                reg_w=reg_w, reg_s=reg_s, nan_w=int(np.isnan(lw_vals).sum()))


def run_e1b(model, prior, n_pairs, M, rng, orc, seed_meta):
    pairs = [matched_pair(prior, rng, orc) for _ in range(n_pairs)]
    X0, X1, Y = [], [], []
    for pr in pairs:
        x0, x1, y = fresh_outcomes(prior, pr["cls_true"], pr["S"], M, rng, orc)
        X0.append(x0); X1.append(x1); Y.append(y)
    qxy = np.stack([np.asarray(X0), np.asarray(X1)], axis=-1)          # (n_pairs, M, 2)
    Y = np.asarray(Y)
    D_true = np.asarray([p["D_true"] for p in pairs], np.float32)
    D_mirror = np.asarray([p["D_mirror"] for p in pairs], np.float32)
    D_same = []
    for pr in pairs:
        pi = PI_A if pr["cls_true"] == 1 else PI_B
        r = R_OF.get(prior, 2.0)
        z = orc.al_sample(1.0, r, (N_CONTEXT, D_DIM), rng)
        D_same.append(context_from_std3(pr["S"], pi, z, orc))
    D_same = np.asarray(D_same, np.float32)
    lq_true = model_predictive_batch(model, D_true, qxy)
    lq_mirror = model_predictive_batch(model, D_mirror, qxy)
    lq_same = model_predictive_batch(model, D_same, qxy)
    nll_true = mean_nll_per_row(lq_true, Y)
    nll_mirror = mean_nll_per_row(lq_mirror, Y)
    nll_same = mean_nll_per_row(lq_same, Y)
    gain_spec = nll_mirror - nll_true
    gain_neut = nll_same - nll_true
    cls_arr = np.array([p["cls_true"] for p in pairs])
    g1 = gain_spec[cls_arr == 1]
    g2 = gain_spec[cls_arr == 2]
    return dict(**seed_meta, n_pairs=n_pairs,
                gain_spec_mean=float(np.nanmean(gain_spec)),
                gain_spec_median=float(np.nanmedian(gain_spec)),
                gain_spec_wilcox_p=wilcoxon_paired(gain_spec),
                gain_spec_cls1_mean=float(np.nanmean(g1)), gain_spec_cls1_n=int(len(g1)),
                gain_spec_cls2_mean=float(np.nanmean(g2)), gain_spec_cls2_n=int(len(g2)),
                gain_neut_mean=float(np.nanmean(gain_neut)),
                gain_neut_wilcox_p=wilcoxon_paired(gain_neut),
                gate_spec=bool(wilcoxon_paired(gain_spec) is not None
                               and wilcoxon_paired(gain_spec) < 0.01
                               and np.nanmean(gain_spec) > 0),
                gate_ctrl6=bool(np.nanmean(gain_spec) > np.nanmean(gain_neut)))


def positive_control_bayes(ell, S_list, prior, orc, qx_grid=Q0, rng_seed=0):
    lw = []
    for e, S in zip(ell, S_list):
        logpA, logpB = anchor_logprobs(S, prior, qx_grid, orc)
        pA = np.exp(logpA); pB = np.exp(logpB)
        sig = 1.0 / (1.0 + np.exp(-e))
        Q = sig * pA + (1.0 - sig) * pB
        _, lwv = evidence_scores(np.log(Q + 1e-30), logpA, logpB)
        lw.append(lwv)
    lw = np.asarray(lw)
    ell = np.asarray(ell, float)
    m = np.isfinite(lw) & np.isfinite(ell)
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    ts = fit_evidence_regression(ell[m], lw[m], rng_seed=rng_seed)
    return dict(n=int(m.sum()), beta_w=ts["theil_slope"],
                beta_dev_from_neg1=float(ts["theil_slope"] - (-1.0)))


# ---------------------------------------------------------------------------
# per-seed run
# ---------------------------------------------------------------------------
def run_seed(prior, seed, dose, eval_seed, nets_dir, orc, oracle, smoke=False, qx_grid=Q0):
    t0 = time.time()
    out = dict(prior=prior, model_seed=seed, dose=dose, eval_seed=eval_seed)
    model = load_model(prior, seed, dose, nets_dir)
    rng = np.random.default_rng(eval_seed)

    n_nat = 60 if smoke else 1500
    n_bal = 60 if smoke else 1500
    n_pairs = 60 if smoke else 800
    M = 5 if smoke else 20

    nat = gen_bank(prior, n_nat, rng, orc, oracle)
    bal = gen_bank(prior, n_bal, rng, orc, oracle, balanced=True)
    out["e1a_natural"] = run_e1a(model, nat, prior, orc, dict(eval_seed=eval_seed), qx_grid)
    out["e1a_balanced"] = run_e1a(model, bal, prior, orc, dict(eval_seed=eval_seed), qx_grid)

    ell = np.asarray(nat["ell"], float)
    out["positive_control"] = positive_control_bayes(ell, nat["S"], prior, orc, qx_grid, rng_seed=eval_seed)

    out["e1b_crossscore"] = run_e1b(model, prior, n_pairs, M, rng, orc, dict(eval_seed=eval_seed))

    # Gaussian-null bank (same model, anchors under Gaussian family)
    null = gen_bank("N", n_nat, rng, orc, oracle)
    Ds = np.asarray([d for d in null["D"]], np.float32)
    s_n, lw_n = [], []
    qx = torch.tensor(np.asarray(qx_grid, np.float32)[None, :, :], device=DEV)
    tok = torch.full((1,), NULL_TOK, dtype=torch.long, device=DEV)
    for i0 in range(0, len(Ds), 256):
        batch = torch.tensor(Ds[i0:i0 + 256], device=DEV)
        with torch.no_grad():
            logits = model(batch, qx.repeat(len(batch), 1, 1), tok.repeat(len(batch)))
        logQ = torch.log_softmax(logits, dim=-1).cpu().numpy()
        for j, S in enumerate(null["S"][i0:i0 + 256]):
            logpA, logpB = anchor_logprobs_gauss(S, qx_grid, orc)
            s, lw = evidence_scores(logQ[j], logpA, logpB)
            s_n.append(s); lw_n.append(lw)
    s_n = np.asarray(s_n); lw_n = np.asarray(lw_n)
    out["gaussian_null"] = dict(n=len(null["D"]),
                                mean_abs_s=float(np.nanmean(np.abs(s_n))),
                                max_abs_s=float(np.nanmax(np.abs(s_n))) if len(s_n) else float("nan"),
                                frac_finite_w=float(np.isfinite(lw_n).mean()))
    out["wallclock_s"] = time.time() - t0
    return out


def mc_convergence_check(oracle, prior, orc, r, n_check=10):
    """Posterior from two M/2 halves; correlation + max-diff on n_check contexts."""
    M = len(oracle.atoms)
    half = M // 2
    rng = np.random.default_rng(7777)
    corr, dp = [], []
    for _ in range(n_check):
        S = orc.sample_Sigmas(rng, 1)[0]
        cls = int(rng.integers(0, 2)) + 1
        pi = PI_A if cls == 1 else PI_B
        D = orc.gen_data(S, orc.ORDERINGS.index(pi), r, N_CONTEXT, rng)
        p1 = oracle.posterior(D, r, slice(0, half))
        p2 = oracle.posterior(D, r, slice(half, M))
        corr.append(float(np.corrcoef(p1, p2)[0, 1]))
        dp.append(float(np.abs(p1 - p2).max()))
    return dict(n_atoms=M, corr_median=float(np.median(corr)),
                maxdp_median=float(np.median(dp)),
                ok=bool(np.median(corr) >= 0.98 and np.median(dp) < 0.02))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prior", default="C", choices=["A", "C", "N"])
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--dose", type=int, default=20000)
    ap.add_argument("--eval-seed-base", type=int, default=601)
    ap.add_argument("--nets-dir", default=os.environ.get("NETS3", os.path.join(_HERE, "..", "nets3")))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "results", "exp4"))
    ap.add_argument("--m-atoms", type=int, default=20000)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    orc = load_orc()
    log = lambda m: print(f"[{time.time():.0f}s] {m}", flush=True)
    log(f"d3 readout: prior={a.prior} seeds={a.seeds} dose={a.dose} dev={DEV}")

    # atom set + oracle (built once per process; dose-independent)
    M_atoms = 2000 if a.smoke else a.m_atoms
    atoms = orc.sample_Sigmas(np.random.default_rng(7005), M_atoms)
    oracle = orc.Oracle(atoms)
    r = R_OF.get(a.prior, 2.0)
    conv = mc_convergence_check(oracle, a.prior, orc, r)
    log(f"atom set {len(atoms)} | MC conv corr {conv['corr_median']:.4f} maxdp {conv['maxdp_median']:.4f} ok={conv['ok']}")

    seeds = [int(x) for x in a.seeds.split(",")]
    for si, seed in enumerate(seeds):
        eval_seed = a.eval_seed_base + si
        fname = f"exp4_{a.prior}_s{seed}_dose{a.dose}.json"
        fpath = os.path.join(a.outdir, fname)
        if a.resume and os.path.exists(fpath):
            log(f"[{seed}] exists, skip")
            continue
        log(f"[{seed}] eval_seed={eval_seed}")
        res = run_seed(a.prior, seed, a.dose, eval_seed, a.nets_dir, orc, oracle, smoke=a.smoke)
        res["mc_conv"] = conv
        with open(fpath, "w") as f:
            json.dump(res, f, indent=1, default=str)
        log(f"[{seed}] done {res['wallclock_s']:.1f}s")


if __name__ == "__main__":
    main()
