#!/usr/bin/env python3
"""EXPERIMENT 3 — internal causal-order interchange (PREREG-BINARY-DUEL.md §4).

Phase A — component screen. Four-context grid {mech1, mech2} x {orderA, orderB}
(shared residual draws):
  donor   = (Sigma1, order B)     base    = (Sigma2, order A)
  desired = (Sigma2, order B)     control = (Sigma1, order A)

At each site (layer in {0,1} x component in {ctx-resid, query-resid, attn, mlp,
head0..3}), run the donor forward to capture, then run the base forward with the
donor activation on that site. Metrics per site:
  transfer   = (KL(Q_base||Q_desired) - KL(Q_patch||Q_desired)) / KL(Q_base||Q_desired)
  anti_donor = KL(Q_patch||Q_desired) < KL(Q_patch||Q_donor)
Placebos: no-patch (Q_base, transfer=0) and random-subspace donor patch (must not transfer).

GATE: some site shows mean transfer > 0 (signed-rank p<0.05) AND anti_donor rate
> 0.5, with placebo transfer not significant.

Numbers + gate booleans only. Per-seed result JSONs with resume.
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import core as C
import evidence_core as EC

RESULT_DIR = os.path.join(_HERE, "results", "exp3")
os.makedirs(RESULT_DIR, exist_ok=True)

N_HEAD = 4
CTX_SL = slice(1, 1 + C.N_CONTEXT)   # context tokens
QRY_SL = slice(1 + C.N_CONTEXT, 2 + C.N_CONTEXT)


# ---------------------------------------------------------------------------
# faithful per-head attention (replicates nn.MultiheadAttention, dropout=0)
# ---------------------------------------------------------------------------

def _attn_heads(attn, x):
    """Packed self-attention, returns (merged_output, per_head_outputs).

    x: (B, T, E)  -> out (B, T, E), ho (B, H, T, hd). Exact at dropout=0.
    """
    B, T, E = x.shape
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    H, hd = attn.num_heads, E // attn.num_heads
    q = q.reshape(B, T, H, hd).transpose(1, 2)
    k = k.reshape(B, T, H, hd).transpose(1, 2)
    v = v.reshape(B, T, H, hd).transpose(1, 2)
    att = torch.softmax((q @ k.transpose(-2, -1)) * (hd ** -0.5), dim=-1)
    ho = att @ v                      # (B, H, T, hd)
    merged = ho.transpose(1, 2).reshape(B, T, E)
    return attn.out_proj(merged), ho


# site keys: layer x component
COMPONENTS = ["ctx_resid", "query_resid", "attn", "mlp"] + [f"head{i}" for i in range(N_HEAD)]


def site_keys(n_layers=2):
    return [f"L{li}.{comp}" for li in range(n_layers) for comp in COMPONENTS]


class Harness:
    """Capture-and-replace forward over EF.PFNModel, per query."""

    def __init__(self, model):
        self.model = model
        self.layers = model.transformer.layers
        self.store = {}

    @torch.no_grad()
    def _run(self, ctx_np, qx_np, patch=None, capture=False):
        """One context. ctx_np: (K,2); qx_np: (nq,). Returns (nq,100) log-probs.

        Forward is POST-LN (norm_first=False), matching torch's
        TransformerEncoderLayer: h = norm1(h + attn(h)), h = norm2(h + ff(h)).
        """
        K = ctx_np.shape[0]
        ce = self.model.point_embed(torch.tensor(ctx_np, dtype=torch.float32)[None])
        te = self.model.token_embed(torch.tensor([C.NULL_TOK], dtype=torch.long)).unsqueeze(1)
        outs = []
        qx = torch.tensor(qx_np, dtype=torch.float32)
        for qi, xq in enumerate(qx):
            qe = self.model.query_embed(xq.reshape(1, 1)).unsqueeze(1)
            h = torch.cat([te, ce, qe], 1)          # (1, 1+K+1, E)
            for li, lay in enumerate(self.layers):
                a_out, ho = _attn_heads(lay.self_attn, h)   # attn on raw h (pre-resid)
                if patch is not None:
                    pk, psite = patch
                    if psite[0] == li and psite[1] == "attn":
                        a_out = self.store["attn"][li][qi]
                    elif psite[0] == li and psite[1].startswith("head"):
                        hi = int(psite[1][4:])
                        ho = ho.clone()
                        ho[:, hi] = self.store["head"][li][hi][qi]
                        a_out = lay.self_attn.out_proj(ho.transpose(1, 2).reshape(1, h.shape[1], -1))
                if capture:
                    self.store["attn"][li][qi] = a_out.detach().clone()
                    for hi in range(N_HEAD):
                        self.store["head"][li][hi][qi] = ho[:, hi].detach().clone()
                h = lay.norm1(h + a_out)
                m = lay.linear2(lay.activation(lay.linear1(h)))  # ff on post-norm1 h
                if patch is not None:
                    pk, psite = patch
                    if psite == (li, "mlp"):
                        m = self.store["mlp"][li][qi]
                if capture:
                    self.store["mlp"][li][qi] = m.detach().clone()
                h = lay.norm2(h + m)
                if patch is not None:
                    pk, psite = patch
                    if psite == (li, "ctx_resid"):
                        h[:, CTX_SL] = self.store["resid"][li][qi][:, CTX_SL]
                    elif psite == (li, "query_resid"):
                        h[:, QRY_SL] = self.store["resid"][li][qi][:, QRY_SL]
                    elif psite == (li, "resid_all"):
                        h = self.store["resid"][li][qi]
                if capture:
                    self.store["resid"][li][qi] = h.detach().clone()
            outs.append(self.model.out_head(h[:, -1, :]))
        return torch.log_softmax(torch.stack(outs, 1), -1)[0].numpy()  # (nq, 100)

    def _init_store(self, nq):
        self.store = {
            "resid": {li: [None] * nq for li in range(len(self.layers))},
            "attn":  {li: [None] * nq for li in range(len(self.layers))},
            "mlp":   {li: [None] * nq for li in range(len(self.layers))},
            "head":  {li: {hi: [None] * nq for hi in range(N_HEAD)} for li in range(len(self.layers))},
        }

    def capture(self, ctx_np, qx_np):
        self._init_store(len(qx_np))
        logQ = self._run(ctx_np, qx_np, patch=None, capture=True)
        return logQ

    def patched(self, ctx_np, qx_np, site):
        li = int(site[1])               # "L0.ctx_resid" -> layer 0
        comp = site.split(".", 1)[1]    # component string, e.g. "head2"
        return self._run(ctx_np, qx_np, patch=("x", (li, comp)))

    def clean(self, ctx_np, qx_np):
        return self._run(ctx_np, qx_np, patch=None, capture=False)

    @torch.no_grad()
    def random_direction(self, ctx_np, qx_np, li, seed):
        """Placebo: perturb the resid stream at layer li with a matched-magnitude
        random direction, no donor content. At layer li (post-norm2), add
        eta * R/||R|| with ||eta|| = ||h_donor - h_base||, R ~ N(0,I) per position.

        This is the clean null: same perturbation scale as the true donor swap,
        zero donor signal. A donor-specific site must transfer MORE than this."""
        rng = np.random.default_rng(seed)
        ce = self.model.point_embed(torch.tensor(ctx_np, dtype=torch.float32)[None])
        te = self.model.token_embed(torch.tensor([C.NULL_TOK], dtype=torch.long)).unsqueeze(1)
        outs = []
        for qi, xq in enumerate(torch.tensor(qx_np, dtype=torch.float32)):
            qe = self.model.query_embed(xq.reshape(1, 1)).unsqueeze(1)
            h = torch.cat([te, ce, qe], 1)
            for li2, lay in enumerate(self.layers):
                a_out, ho = _attn_heads(lay.self_attn, h)   # attn on raw h (post-LN model)
                h = lay.norm1(h + a_out)
                m = lay.linear2(lay.activation(lay.linear1(h)))
                h = lay.norm2(h + m)
                if li2 == li:
                    d = self.store["resid"][li][qi] - h    # (1, T, E) donor diff
                    eta = float(torch.linalg.norm(d))
                    R = torch.randn(1, d.shape[1], d.shape[2], dtype=torch.float32)
                    h = h + eta * R / (torch.linalg.norm(R) + 1e-9)
            outs.append(self.model.out_head(h[:, -1, :]))
        return torch.log_softmax(torch.stack(outs, 1), -1)[0].numpy()


# ---------------------------------------------------------------------------
# quadruple construction
# ---------------------------------------------------------------------------

def make_quadruple(prior, rng):
    """Draw Sigma1, Sigma2, orderA/B and shared residuals; build 4 contexts.

    Returns dict with the (N_CONTEXT,2) panels, Sigmas and order labels.
    """
    S1, *_ = EC.E.sample_valid_Sigma(rng)
    S2, *_ = EC.E.sample_valid_Sigma(rng)
    orderA = int(rng.integers(1, 3))          # 1=G1, 2=G2
    orderB = 3 - orderA
    z1 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
    z2 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
    p1A = EC.E.sigma_to_params_G1(S1); p1B = EC.E.sigma_to_params_G2(S1)
    p2A = EC.E.sigma_to_params_G1(S2); p2B = EC.E.sigma_to_params_G2(S2)
    donor   = C.context_from_std(prior, orderB, p1B, z1, z2)   # (S1, B)
    base    = C.context_from_std(prior, orderA, p2A, z1, z2)   # (S2, A)
    desired = C.context_from_std(prior, orderB, p2B, z1, z2)   # (S2, B)
    control = C.context_from_std(prior, orderA, p1A, z1, z2)   # (S1, A)
    return dict(donor=donor, base=base, desired=desired, control=control,
                S1=S1, S2=S2, orderA=orderA)


def kl_bins(loga, logb):
    """KL(a||b) over (nq,100) log-probs: sum_q sum_b a(b)(log a - log b)."""
    a = np.exp(loga)
    return float(np.sum(a * (loga - logb)))


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------

def run_phaseA(model, prior, rng, n_pairs, qx_grid, rng_placebo_seed=777, smoke=False):
    n_pairs = max(10, n_pairs) if smoke else n_pairs
    sites = site_keys(len(model.transformer.layers))
    har = Harness(model)
    # accumulators
    n_run = 0
    trans = {k: [] for k in sites}
    adon = {k: [] for k in sites}
    ctrl = {k: [] for k in sites}
    pla = []
    n_degen = 0
    for i in range(n_pairs):
        q = make_quadruple(prior, rng)
        logQ_base = har.clean(q["base"], qx_grid)
        logQ_des  = har.clean(q["desired"], qx_grid)
        logQ_ctrl = har.clean(q["control"], qx_grid)
        logQ_don  = har.capture(q["donor"], qx_grid)
        kl_bd = kl_bins(logQ_base, logQ_des)
        if kl_bd < 1e-6:
            n_degen += 1
            continue
        # per-site patched predictives
        for site in sites:
            logQ_p = har.patched(q["base"], qx_grid, site)
            kl_pd = kl_bins(logQ_p, logQ_des)
            kl_pn = kl_bins(logQ_p, logQ_don)
            tr = (kl_bd - kl_pd) / kl_bd
            trans[site].append(tr)
            adon[site].append(float(kl_pd < kl_pn))
            ctrl[site].append(kl_bins(logQ_p, logQ_ctrl))
        # placebo: matched-magnitude random-direction perturbation at L1 resid
        li_p = min(1, len(model.transformer.layers) - 1)
        logQ_pl = har.random_direction(q["base"], qx_grid, li_p, seed=rng_placebo_seed + i)
        pla.append((kl_bd - kl_bins(logQ_pl, logQ_des)) / kl_bd)
        n_run += 1
        if i % 20 == 0:
            print(f"    pair {i}/{n_pairs} (done {n_run}, degen {n_degen})", flush=True)
    out = {"n_pairs_done": n_run, "n_degen": n_degen, "sites": sites}
    for site in sites:
        t = np.asarray(trans[site]); a = np.asarray(adon[site]); c = np.asarray(ctrl[site])
        w_p = C.wilcoxon_paired(t)
        out[site] = dict(transfer_mean=float(np.mean(t)),
                         transfer_med=float(np.median(t)),
                         transfer_p=w_p,
                         anti_donor_rate=float(np.mean(a)),
                         kl_to_ctrl_mean=float(np.mean(c)),
                         n=int(len(t)))
    pla = np.asarray(pla)
    out["placebo_rs"] = dict(transfer_mean=float(np.mean(pla)),
                             transfer_p=C.wilcoxon_paired(pla), n=int(len(pla)))
    # gate: some site transfer>0 (p<0.05, mean>0.01) AND anti_donor_rate>0.5 AND placebo not significant
    any_site = any(out[s]["transfer_p"] is not None and out[s]["transfer_p"] < 0.05
                   and out[s]["transfer_mean"] > 0.01 and out[s]["anti_donor_rate"] > 0.5
                   for s in sites)
    pla_ok = out["placebo_rs"]["transfer_p"] is None or out["placebo_rs"]["transfer_p"] >= 0.05
    out["gate_phaseA"] = bool(any_site and pla_ok)
    out["best_site"] = max(sites, key=lambda s: out[s]["transfer_mean"])
    return out


def run_seed(scale, prior, seed, dose, eval_seed, smoke=False):
    model = C.load_pfn(scale, prior, seed, dose)
    rng = np.random.default_rng(eval_seed)
    n_pairs = 40 if smoke else 200
    qx = C.Q0 if not smoke else np.linspace(-2, 2, 5)
    t0 = time.time()
    out = dict(scale=scale, prior=prior, model_seed=seed, dose=dose, eval_seed=eval_seed)
    out["phaseA"] = run_phaseA(model, prior, rng, n_pairs, qx, smoke=smoke)
    out["wallclock_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scale", default="base")
    ap.add_argument("--prior", default="AL40")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--dose", type=int, default=12000)
    ap.add_argument("--eval-seed-base", type=int, default=301)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()
    for si, seed in enumerate(seeds):
        eval_seed = a.eval_seed_base + si
        fname = f"exp3_{a.prior}_{a.scale}_s{seed}_dose{a.dose}.json"
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
