#!/usr/bin/env python3
"""E21 — the full paper fleet: a FINE prior ladder x a SCALE ladder (d=2 Fix-B), warmup+cosine recipe.
Delivers in one campaign: (C2) information law with many points; (RF2) payoff/I-tracking with enough priors
to be significant; (C5) clean saturation curve + 8.5M fixed; and payoff-vs-scale. Data construction VERBATIM
from arch_fleet/e19 (apples-to-apples). GPU. Manifest split across a Slurm array (--task/--ntask).

PRIOR ENCODING (filename-safe): AL<rr> = asymmetric-Laplace skew r=rr/10 (AL15..AL50); T<df> = Student-t (T5,T3);
L = symmetric Laplace; N = Gaussian (I=0 anchor). I_pi spans ~0.1 (heavy-tail/symmetric) to ~0.65 (high skew)."""
import os, sys, json, math, time, argparse, zlib
import numpy as np
import scipy.stats as st
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = os.environ.get("E21_OUT", ".")
SIGMA_LO, SIGMA_HI = 0.6, 1.5; RHO_MAG_LO, RHO_MAG_HI = 0.4, 0.8
A_HI = 1.5; B_LO, B_HI = 0.30, 1.30
N_CONTEXT, N_QUERY, N_BINS = 30, 7, 100
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1); BATCH, NULL_TOK = 32, 2
SCALES = {  # warmup recipe validated by e19 smoke (large@6e-4 hit 2.814; xl@3e-4 stable)
    "base":  dict(d_model=256, d_ff=512,  n_heads=4, n_layers=2, peak_lr=1e-3, steps=12000, warmup=800),   # ~1.1M
    "mid":   dict(d_model=384, d_ff=768,  n_heads=6, n_layers=3, peak_lr=8e-4, steps=12000, warmup=1000),  # ~4M
    "large": dict(d_model=512, d_ff=1024, n_heads=8, n_layers=4, peak_lr=6e-4, steps=12000, warmup=1500),  # ~8.5M
    "xl":    dict(d_model=768, d_ff=1536, n_heads=8, n_layers=4, peak_lr=3e-4, steps=15000, warmup=2000),  # ~19M
    "xxl":   dict(d_model=1024,d_ff=2048, n_heads=8, n_layers=4, peak_lr=2e-4, steps=18000, warmup=3000),  # ~34M
    "xxxl":  dict(d_model=1024,d_ff=2048, n_heads=8, n_layers=6, peak_lr=1.5e-4,steps=20000, warmup=4000), # ~50M
}
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)
def pseed(prior): return zlib.crc32(prior.encode()) % 100000

def sample_params(rng, n):
    rows = []
    while sum(len(r) for r in rows) < n:
        M = 4096
        s1 = np.exp(rng.uniform(math.log(SIGMA_LO), math.log(SIGMA_HI), M)); s2 = np.exp(rng.uniform(math.log(SIGMA_LO), math.log(SIGMA_HI), M))
        rho = rng.choice([-1.0, 1.0], M) * rng.uniform(RHO_MAG_LO, RHO_MAG_HI, M); omr = np.sqrt((1 - rho**2)/2.0)
        a1 = rho*s2/s1; b11 = s1/math.sqrt(2); b12 = s2*omr; a2 = rho*s1/s2; b21 = s2/math.sqrt(2); b22 = s1*omr
        ok = ((np.abs(a1)<=A_HI)&(b11>=B_LO)&(b11<=B_HI)&(b12>=B_LO)&(b12<=B_HI)&(np.abs(a2)<=A_HI)&(b21>=B_LO)&(b21<=B_HI)&(b22>=B_LO)&(b22<=B_HI))
        rows.append(np.stack([a1[ok], b11[ok], b12[ok], a2[ok], b21[ok], b22[ok]], axis=1))
    P = np.concatenate(rows)[:n]; return P, rng.integers(1, 3, n)
def sample_errors(prior, b, n_total, rng):
    k = len(b)
    if prior == "L": return rng.laplace(0, b[:, None], (k, n_total))
    if prior == "N": return rng.normal(0, b[:, None]*math.sqrt(2.0), (k, n_total))
    if prior.startswith("T"):
        nu = float(prior[1:]); s = b[:, None]*math.sqrt(2*(nu-2)/nu); return rng.standard_t(nu, (k, n_total))*s
    fam = prior[:2]                                            # MORE-FAMILIES (standardized -> scaled to Var=2b^2)
    if fam in ("SN", "GN", "GA", "GM"):
        v = float(prior[2:]) / (10.0 if fam in ("GN", "GM") else 1.0)
        if fam == "GM":                                        # symmetric 2-Gaussian mixture, mode sep v
            sd = math.sqrt(1.0 + v*v); z = (rng.choice([-v, v], (k, n_total)) + rng.normal(0, 1, (k, n_total))) / sd
        else:
            d = st.skewnorm(a=v) if fam == "SN" else (st.gennorm(beta=v) if fam == "GN" else st.gamma(a=v))
            mu, sdv = float(d.mean()), float(d.std()); z = (d.rvs(size=(k, n_total), random_state=rng) - mu) / sdv
        return z * (b[:, None] * math.sqrt(2.0))
    r = int(prior[2:]) / 10.0                                  # AL<rr>
    c = np.sqrt(2*b*b/(1+r*r)); a = r*c
    return rng.exponential(a[:, None], (k, n_total)) - rng.exponential(c[:, None], (k, n_total)) - (a-c)[:, None]
def gen_batch(prior, rng, n_pts):
    P, cls = sample_params(rng, BATCH)
    b1 = np.where(cls==1, P[:,1], P[:,4]); b2 = np.where(cls==1, P[:,2], P[:,5]); a = np.where(cls==1, P[:,0], P[:,3])
    e1 = sample_errors(prior, b1, n_pts, rng); e2 = sample_errors(prior, b2, n_pts, rng)
    x = np.where((cls==1)[:, None], e1, a[:, None]*e1+e2); y = np.where((cls==1)[:, None], a[:, None]*e1+e2, e1)
    return np.stack([x, y], axis=2)
def bin_y(v): return np.searchsorted(BIN_EDGES[1:-1], v)

class PFNModel(nn.Module):
    def __init__(s, d_model, d_ff, n_heads, n_layers, **_):
        super().__init__(); s.point_embed = nn.Linear(2, d_model); s.query_embed = nn.Linear(1, d_model)
        s.token_embed = nn.Embedding(3, d_model)
        s.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, dropout=0.0), n_layers)
        s.out_head = nn.Linear(d_model, N_BINS)
    def forward(s, ctx, qx, tok):
        ce = s.point_embed(ctx); te = s.token_embed(tok).unsqueeze(1); outs = []
        for q in range(qx.shape[1]):
            o = s.transformer(torch.cat([te, ce, s.query_embed(qx[:, q, :]).unsqueeze(1)], 1)); outs.append(s.out_head(o[:, -1, :]))
        return torch.stack(outs, 1)
def lr_at(step, peak, total, warm): return peak*(step+1)/warm if step < warm else peak*0.5*(1+math.cos(math.pi*min(1.0,(step-warm)/max(1,total-warm))))

def train_one(scale, prior, seed):
    cfg = SCALES[scale]; steps = cfg["steps"]; tag = f"M_{scale}_{prior}_s{seed}_st{steps}"
    if os.path.exists(f"{OUTDIR}/{tag}.pt"): log(f"{tag} exists, skip"); return
    torch.manual_seed(1000*seed + pseed(prior)); rng = np.random.default_rng(7000 + 100*seed + pseed(prior))
    model = PFNModel(**cfg).to(DEV); npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["peak_lr"]); model.train(); ll = []
    tok = torch.full((BATCH,), NULL_TOK, dtype=torch.long, device=DEV)
    for step in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(step, cfg["peak_lr"], steps, cfg["warmup"])
        blk = gen_batch(prior, rng, N_CONTEXT + N_QUERY)
        ctx = torch.tensor(blk[:, :N_CONTEXT, :], dtype=torch.float32, device=DEV)
        qx = torch.tensor(blk[:, N_CONTEXT:, 0:1], dtype=torch.float32, device=DEV)
        yb = torch.tensor(bin_y(blk[:, N_CONTEXT:, 1]), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx, qx, tok).reshape(-1, N_BINS), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); ll.append(loss.item())
    model.eval().cpu(); torch.save(model.state_dict(), f"{OUTDIR}/{tag}.pt")
    json.dump(dict(scale=scale, prior=prior, seed=seed, steps=steps, n_params=int(npar), final_loss=float(np.mean(ll[-200:])),
                   wallclock_s=time.time()-t0), open(f"{OUTDIR}/{tag}.json", "w"))
    log(f"{tag} DONE n={npar:,} loss={np.mean(ll[-200:]):.4f}")

def manifest():
    # fine prior ladder spanning I; dissociation pair AL20/AL40 + anchor N get more seeds + xxl
    ladder = os.environ.get("E21_PRIORS", "AL15,AL20,AL25,AL30,AL35,AL40,AL50,L,T5,T3,N").split(",")
    scales = os.environ.get("E21_SCALES", "base,mid,large,xl").split(",")
    nseed = int(os.environ.get("E21_NSEED", "6")); keyn = int(os.environ.get("E21_KEYSEED", "8"))
    key = {"AL20", "AL40", "N"}   # dissociation pair + zero-anchor get more seeds
    jobs = []
    for sc in scales:
        for p in ladder:
            jobs += [(sc, p, s) for s in range(keyn if p in key else nseed)]
    # heavy tiers only if explicitly requested (QOS-capped at ~6 GPUs -> keep the budget ~10-12h)
    for sc in os.environ.get("E21_HEAVY", "").split(","):
        if sc in ("xxl", "xxxl"):
            for p in ladder: jobs += [(sc, p, s) for s in range(int(os.environ.get("E21_HEAVYSEED", "8")))]
    return jobs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", type=int, default=0); ap.add_argument("--ntask", type=int, default=1)
    a = ap.parse_args(); jobs = manifest(); mine = [j for i, j in enumerate(jobs) if i % a.ntask == a.task]
    log(f"E21 on {DEV} | task {a.task}/{a.ntask} | {len(mine)}/{len(jobs)} trainings | out={OUTDIR}")
    for sc, p, s in mine: train_one(sc, p, s)
    log("task complete")
