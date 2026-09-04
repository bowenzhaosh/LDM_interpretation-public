#!/usr/bin/env python3
"""EXPERIMENT 3B — patching Phase B: DAS-lite distributed intervention subspace
(PREREG-EXT3 §3, original PREREG-BINARY-DUEL §4).

Phase A (exp3) found the component screen weak and seed-inconsistent (best head
site varies: L0.head2 / L1.head1 / L1.head3 / L1.head0; transfer +0.02..+0.07;
resid/attn/mlp negative). Phase B asks the DAS-lite question directly: is there a
LOW-RANK orthogonal subspace U (d_model x r) of the residual stream such that

    patch = h_base + U U^T (h_donor - h_base)

moves the base predictive toward the desired (order-swapped) predictive?

Sites (both resid-stream, corrected from the first-draft "L1 ctx+query" which is
dead because the post-LN output reads only the query position):
  resid0  = residual stream at LAYER 0 output (post-norm2), ALL positions
            (NULL marker + context + query, 0..31). The distributed
            representation L1 reads via attention; rank ladder is meaningful.
            Identity patch (U=I) reproduces the donor EXACTLY (no ceiling).
  query1  = residual stream at LAYER 1 output (post-norm2), query position only.
            The final readout representation (out_head reads exactly this).

Ranks r in {1,2,4,8,16}. Soft orthogonality penalty on U. Train on 400 fresh
quadruples, evaluate on 200 held-out (unseen Sigma/context/residual, both
directions, unseen query grid Q1). Placebo = random orthogonal U of the same rank
(applied to the same donor diff): the learned U must transfer MORE than random.

Numbers + gate booleans only; per-(seed,site) result JSON with resume.
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import core as C
import evidence_core as EC
import exp3_patch as exp3  # reuse _attn_heads, make_quadruple (identical substrate + quadruple)

RESULT_DIR = os.path.join(_HERE, "results", "exp3b")
os.makedirs(RESULT_DIR, exist_ok=True)

RANKS = [1, 2, 4, 8, 16]
ORTH_W = 0.1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# faithful differentiable forward (post-LN, batched over queries)
# ---------------------------------------------------------------------------
def build_seq(model, ctx_np, qx_np):
    nq = len(qx_np)
    ce = model.point_embed(torch.tensor(ctx_np, dtype=torch.float32, device=DEV)[None]).repeat(nq, 1, 1)
    te = model.token_embed(torch.tensor([C.NULL_TOK], dtype=torch.long, device=DEV)).unsqueeze(0).repeat(nq, 1, 1)
    qe = model.query_embed(torch.tensor(qx_np, dtype=torch.float32, device=DEV).reshape(nq, 1, 1))
    return torch.cat([te, ce, qe], 1)   # (nq, 1+K+1, E)


def run_layer(lay, h):
    a_out, ho = exp3._attn_heads(lay.self_attn, h)
    h = lay.norm1(h + a_out)
    m = lay.linear2(lay.activation(lay.linear1(h)))
    return lay.norm2(h + m)


def full_forward(model, ctx_np, qx_np):
    """(logQ, h0, h1): logQ (nq,N_BINS); h0/h1 (nq,T,E) post-norm2 resid per layer."""
    h = build_seq(model, ctx_np, qx_np)
    h0 = run_layer(model.transformer.layers[0], h)
    h1 = run_layer(model.transformer.layers[1], h0)
    logQ = torch.log_softmax(model.out_head(h1[:, -1, :]), -1)
    return logQ, h0, h1


def forward_kl(loga, logb):
    """KL(a||b) summed over queries+bins (matches exp3.kl_bins)."""
    a = torch.exp(loga)
    return torch.sum(a * (loga - logb))


def patch_forward(model, q, U, site):
    """Differentiable patched predictive logQ (nq,N_BINS)."""
    if site == "resid0":
        diff = q["h0_don"] - q["h0_base"]
        # patch ALL positions (NULL marker + context + query) so the identity
        # control (U=I) reproduces the donor exactly — no ceiling reduction.
        h = q["h0_base"] + (diff @ U) @ U.T
        h1p = run_layer(model.transformer.layers[1], h)
        return torch.log_softmax(model.out_head(h1p[:, -1, :]), -1)
    if site == "query1":
        diff = q["h1_don"] - q["h1_base"]
        d = diff[:, -1, :]                       # (nq, E)
        proj = (d @ U) @ U.T
        h1p = q["h1_base"].clone()
        h1p[:, -1, :] = h1p[:, -1, :] + proj
        return torch.log_softmax(model.out_head(h1p[:, -1, :]), -1)
    raise ValueError(site)


# ---------------------------------------------------------------------------
# quadruple precompute
# ---------------------------------------------------------------------------
def make_quad(model, prior, rng, qx_grid):
    """One quadruple -> dict of DEV tensors for the site(s). control unused here."""
    q = exp3.make_quadruple(prior, rng)
    logQ_base, h0_base, h1_base = full_forward(model, q["base"], qx_grid)
    logQ_don, h0_don, h1_don = full_forward(model, q["donor"], qx_grid)
    logQ_des, _, _ = full_forward(model, q["desired"], qx_grid)
    return dict(logQ_base=logQ_base.detach(), logQ_don=logQ_don.detach(),
                logQ_des=logQ_des.detach(),
                h0_base=h0_base.detach(), h0_don=h0_don.detach(),
                h1_base=h1_base.detach(), h1_don=h1_don.detach())


def precompute(model, prior, rng, n, qx_grid):
    quads = []
    n_deg = 0
    while len(quads) < n:
        q = make_quad(model, prior, rng, qx_grid)
        kl_bd = float(forward_kl(q["logQ_base"], q["logQ_des"]).item())
        if kl_bd < 1e-6:
            n_deg += 1
            continue
        quads.append(q)
    return quads, n_deg


# ---------------------------------------------------------------------------
# training + eval
# ---------------------------------------------------------------------------
def orth_init(E, r, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(E, r, generator=g))
    return Q


def train_U(model, quads, site, rank, n_steps, lr, seed):
    E = model.point_embed.out_features
    U = nn.Parameter(orth_init(E, rank, seed).to(DEV))
    opt = torch.optim.Adam([U], lr=lr)
    I = torch.eye(rank, device=DEV)
    kl_hist = []
    for step in range(n_steps):
        q = quads[step % len(quads)]
        logQ_p = patch_forward(model, q, U, site)
        kl = forward_kl(logQ_p, q["logQ_des"])
        orth = torch.sum((U.T @ U - I) ** 2)
        loss = kl + ORTH_W * orth
        opt.zero_grad()
        loss.backward()
        opt.step()
        kl_hist.append(float(kl.item()))
    return U.detach(), kl_hist


def evaluate(model, quads, U, site):
    transfers, anti_donors, kl_pd_mean = [], [], []
    for q in quads:
        logQ_p = patch_forward(model, q, U, site)
        kl_bd = float(forward_kl(q["logQ_base"], q["logQ_des"]).item())
        kl_pd = float(forward_kl(logQ_p, q["logQ_des"]).item())
        kl_pn = float(forward_kl(logQ_p, q["logQ_don"]).item())
        transfers.append((kl_bd - kl_pd) / kl_bd)
        anti_donors.append(float(kl_pd < kl_pn))
    transfers = np.asarray(transfers)
    return dict(transfer_mean=float(np.mean(transfers)),
                transfer_med=float(np.median(transfers)),
                transfer_p=C.wilcoxon_paired(transfers),
                anti_donor_rate=float(np.mean(anti_donors)),
                n=int(len(transfers)))


def placebo_transfer(model, quads, site, rank, seed):
    """Random orthogonal U of the same rank applied to the same donor diff."""
    E = model.point_embed.out_features
    U = orth_init(E, rank, seed).to(DEV)
    return evaluate(model, quads, U, site)


def run_seed(prior, seed, dose, eval_seed, site, smoke=False):
    t0 = time.time()
    model = C.load_pfn("base", prior, seed, dose)
    model.to(DEV).eval()
    model.requires_grad_(False)
    rng = np.random.default_rng(eval_seed)
    qx_grid = C.Q0 if not smoke else np.linspace(-2, 2, 5)
    qx_eval = C.Q1 if not smoke else np.linspace(-3, 3, 7)

    n_train = 20 if smoke else 400
    n_eval = 10 if smoke else 200
    n_steps = 30 if smoke else 2000
    lr = 1e-2

    train_quads, nd1 = precompute(model, prior, rng, n_train, qx_grid)
    eval_quads, nd2 = precompute(model, prior, rng, n_eval, qx_eval)

    out = dict(prior=prior, model_seed=seed, dose=dose, eval_seed=eval_seed, site=site,
               n_train=len(train_quads), n_eval=len(eval_quads), n_degen=nd1 + nd2)
    for ri, rank in enumerate(RANKS):
        U, kl_hist = train_U(model, train_quads, site, rank, n_steps, lr, seed=1000 + rank)
        ev = evaluate(model, eval_quads, U, site)
        pl = placebo_transfer(model, eval_quads, site, rank, seed=2000 + rank)
        ev["kl_train_final"] = float(np.mean(kl_hist[-50:]))
        ev["placebo_transfer_mean"] = pl["transfer_mean"]
        ev["placebo_transfer_p"] = pl["transfer_p"]
        ev["gate"] = bool(ev["transfer_p"] is not None and ev["transfer_p"] < 0.05
                          and ev["anti_donor_rate"] > 0.5
                          and ev["transfer_mean"] > pl["transfer_mean"])
        out[f"rank{rank}"] = ev
        print(f"  [{site} r={rank}] transfer {ev['transfer_mean']:+.4f} "
              f"(p={ev['transfer_p']}), anti_donor {ev['anti_donor_rate']:.2f}, "
              f"placebo {pl['transfer_mean']:+.4f}, gate={ev['gate']}", flush=True)
    out["gate_phaseB"] = bool(any(out[f"rank{r}"]["gate"] for r in RANKS))
    out["wallclock_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# harness self-checks (smoke only)
# ---------------------------------------------------------------------------
def harness_checks(model, prior, rng):
    """(a) full_forward parity vs model.forward; (b) identity patch -> donor."""
    qx = C.Q0
    q = exp3.make_quadruple(prior, rng)
    logQ_mine, h0, h1 = full_forward(model, q["base"], qx)
    with torch.no_grad():
        ct = torch.tensor(q["base"], dtype=torch.float32, device=DEV)[None]
        qxt = torch.tensor(qx, dtype=torch.float32, device=DEV)[None, :, None]
        tok = torch.full((1,), C.NULL_TOK, dtype=torch.long, device=DEV)
        logQ_ref = torch.log_softmax(model(ct, qxt, tok), -1)[0]
    parity = float((logQ_mine - logQ_ref).abs().max().item())
    # identity patch (full-rank U) must reproduce donor
    E = model.point_embed.out_features
    U = torch.eye(E, device=DEV)
    quad = make_quad(model, prior, rng, qx)
    logQ_patch = patch_forward(model, quad, U, "resid0")
    logQ_patch_q1 = patch_forward(model, quad, U, "query1")
    id_err0 = float((logQ_patch - quad["logQ_don"]).abs().max().item())
    id_err1 = float((logQ_patch_q1 - quad["logQ_don"]).abs().max().item())
    return dict(parity_maxdiff=parity, identity_patch_err_resid0=id_err0,
                identity_patch_err_query1=id_err1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prior", default="AL40")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--dose", type=int, default=12000)
    ap.add_argument("--site", default="resid0", choices=["resid0", "query1"])
    ap.add_argument("--eval-seed-base", type=int, default=701)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()
    for si, seed in enumerate(seeds):
        eval_seed = a.eval_seed_base + si
        fname = f"exp3b_{a.prior}_s{seed}_dose{a.dose}_{a.site}.json"
        fpath = os.path.join(RESULT_DIR, fname)
        if a.resume and os.path.exists(fpath):
            print(f"[{seed}] exists, skip", flush=True)
            continue
        print(f"[{seed}] eval_seed={eval_seed} site={a.site}", flush=True)
        res = run_seed(a.prior, seed, a.dose, eval_seed, a.site, smoke=a.smoke)
        if a.smoke:
            model = C.load_pfn("base", a.prior, seed, a.dose)
            model.to(DEV).eval()
            res["harness"] = harness_checks(model, a.prior, np.random.default_rng(eval_seed + 999))
        C.save_partial(fpath, res)
    print(f"all seeds done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
