#!/usr/bin/env python3
"""d=4 fleet trainer — D=4 generalization of d6_train_fleet.py. Construction (sample_Sigmas / params_for /
al_sample / gen_data) is VERBATIM from d4plus_oracle.py so the trained nets' frozen reps are scored against the
SAME oracle (no construction drift). One-query task (as at d=3): predict canonical x4 | (x1,x2,x3) + context.
Net: point_embed Linear(4,d), query_embed Linear(3,d), d256/512ff/4h/2L, 100 bins on [-8,8], K=30, batch 32.
Priors A=AL(r2), C=AL(r4), N=Gaussian(I=0). Manifest A,C x16 + N x8 @20000 (40 trainings). GPU.
Usage:  python3 d4_train_fleet.py <task_idx> <n_tasks> <outdir>
Smoke:  python3 d4_train_fleet.py --smoke      (CPU; asserts construction-parity vs oracle + loss-decrease)
"""
import sys, os, json, math, time
from itertools import permutations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

D_DIM = 4
ORDERINGS = list(permutations(range(D_DIM)))
N_CONTEXT, N_QUERY, N_BINS = 30, 7, 100
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)
SCALES = {  # warmup+cosine recipe from e21_fleet (validated at d=2); d=4 reuses it across the ladder
    "base":  dict(d_model=256,  d_ff=512,  n_heads=4, n_layers=2, peak_lr=1e-3,  steps=20000, warmup=800),   # ~1.1M (already trained on WashU as M4_<p>_)
    "mid":   dict(d_model=384,  d_ff=768,  n_heads=6, n_layers=3, peak_lr=8e-4,  steps=20000, warmup=1000),  # ~4M
    "large": dict(d_model=512,  d_ff=1024, n_heads=8, n_layers=4, peak_lr=6e-4,  steps=20000, warmup=1500),  # ~8.5M
    "xl":    dict(d_model=768,  d_ff=1536, n_heads=8, n_layers=4, peak_lr=3e-4,  steps=25000, warmup=2000),  # ~19M
    "xxl":   dict(d_model=1024, d_ff=2048, n_heads=8, n_layers=4, peak_lr=2e-4,  steps=28000, warmup=3000),  # ~34M
}
SCALE = os.environ.get("D4_SCALE", "base"); _C = SCALES[SCALE]
D_MODEL, D_FF, N_HEADS, N_LAYERS = _C["d_model"], _C["d_ff"], _C["n_heads"], _C["n_layers"]
PEAK_LR, STEPS, WARMUP = _C["peak_lr"], _C["steps"], _C["warmup"]
STEPS = int(os.environ.get("STEPS_OVERRIDE", STEPS))
BATCH, NULL_TOK = 32, 2
DOSE_CKPTS = sorted(int(x) for x in os.environ.get("D4_DOSE_CKPTS", "").split(",") if x.strip())  # e.g. 0,300,1000,3000,6000,12000
R_OF = {"A": 2.0, "C": 4.0}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = "."
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

# ---------- construction VERBATIM from d4plus_oracle.py (general-d, D_DIM=4) ----------
def params_for(S, pi):
    S = np.asarray(S); n, d = len(S), S.shape[1]
    Spi = S[:, pi][:, :, pi]
    Lch = np.linalg.cholesky(Spi)
    diag = np.diagonal(Lch, axis1=1, axis2=2)
    Lunit = Lch / diag[:, None, :]
    Dv = diag ** 2
    U = np.linalg.inv(Lunit)
    b = np.sqrt(np.maximum(Dv, 1e-12) / 2.0)
    return Lunit, U, b
def validity_keep(S):
    keep = np.ones(len(S), bool)
    for pi in ORDERINGS:
        Lunit, U, b = params_for(S, pi)
        beta = -U
        mask = np.tril(np.ones((D_DIM, D_DIM)), -1).astype(bool)
        amax = np.abs(beta[:, mask]).max(1)
        keep &= (amax <= 1.5) & (b >= 0.3).all(1) & (b <= 1.3).all(1)
    return keep
def sample_Sigmas(rng, n):
    out = []
    while len(out) < n:
        m = 8192
        sd = np.exp(rng.uniform(math.log(0.6), math.log(1.5), (m, D_DIM)))
        R = np.eye(D_DIM)[None].repeat(m, 0)
        iu = np.triu_indices(D_DIM, 1)
        rho = rng.choice([-1., 1.], (m, len(iu[0]))) * rng.uniform(0.3, 0.8, (m, len(iu[0])))
        R[:, iu[0], iu[1]] = rho; R[:, iu[1], iu[0]] = rho
        S = sd[:, :, None] * R * sd[:, None, :]
        ev = np.linalg.eigvalsh(S); S = S[ev[:, 0] > 1e-6]
        S = S[validity_keep(S)]
        out.extend(S)
    return np.array(out[:n])
def al_ac(b, r):
    c = np.sqrt(2.0 * b * b / (1.0 + r * r)); return r * c, c
def al_sample(b, r, size, rng):
    a, c = al_ac(b, r); return rng.exponential(a, size) - rng.exponential(c, size) - (a - c)
def gen_data(S1, fam, r, k, rng, gaussian=False):
    pi = ORDERINGS[fam]; Lunit, U, b = params_for(S1[None], pi); Lunit, b = Lunit[0], b[0]
    if gaussian:
        e = rng.normal(0, np.sqrt(2.0) * b[None, :], (k, D_DIM))
    else:
        e = np.stack([al_sample(np.full(k, b[m]), r, k, rng) for m in range(D_DIM)], 1)
    xpi = e @ Lunit.T
    x = np.empty_like(xpi); x[:, list(pi)] = xpi
    return x

def gen_batch(prior, rng, n_pts):
    S = sample_Sigmas(rng, BATCH)
    fams = rng.integers(0, len(ORDERINGS), BATCH)
    gaussian = (prior == "N"); r = R_OF.get(prior, 2.0)
    X = np.zeros((BATCH, n_pts, D_DIM))
    for bi in range(BATCH):
        X[bi] = gen_data(S[bi], int(fams[bi]), r, n_pts, rng, gaussian=gaussian)
    return X
def bin_y(v): return np.searchsorted(BIN_EDGES[1:-1], v)

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
        ce = self.point_embed(ctx); te = self.token_embed(tok).unsqueeze(1); outs = []
        for q in range(qxy.shape[1]):
            qe = self.query_embed(qxy[:, q, :]).unsqueeze(1)
            outs.append(self.out_head(self.transformer(torch.cat([te, ce, qe], 1))[:, -1, :]))
        return torch.stack(outs, 1)

def lr_at(step, peak, total, warm):
    return peak*(step+1)/warm if step < warm else peak*0.5*(1+math.cos(math.pi*min(1.0, (step-warm)/max(1, total-warm))))

def train_one(prior, seed, steps):
    # base keeps the legacy M4_<p>_ name (backward-compat with the WashU base nets + readout); ladder = M4_<scale>_<p>_
    tag = (f"M4_{prior}_s{seed}_st{steps}" if SCALE == "base" else f"M4_{SCALE}_{prior}_s{seed}_st{steps}")
    if os.path.exists(f"{OUTDIR}/{tag}.pt"): log(f"{tag} exists, skip"); return
    torch.manual_seed(1000 * seed + ord(prior)); rng = np.random.default_rng(9000 + 100 * seed + ord(prior))
    model = PFN4().to(DEV); npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR); model.train(); L = []
    tok = torch.full((BATCH,), NULL_TOK, dtype=torch.long, device=DEV); ckset = set(DOSE_CKPTS)
    for step in range(steps):
        if step in ckset: torch.save(model.state_dict(), f"{OUTDIR}/{tag}_ck{step}.pt"); log(f"  {tag} ckpt@{step}")  # dose checkpoints
        for g in opt.param_groups: g["lr"] = lr_at(step, PEAK_LR, steps, WARMUP)
        blk = gen_batch(prior, rng, N_CONTEXT + N_QUERY)
        ctx = torch.tensor(blk[:, :N_CONTEXT, :], dtype=torch.float32, device=DEV)
        q = blk[:, N_CONTEXT:, :]
        qxy = torch.tensor(q[:, :, :D_DIM - 1], dtype=torch.float32, device=DEV)
        yb = torch.tensor(bin_y(q[:, :, D_DIM - 1]), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx, qxy, tok).reshape(-1, N_BINS), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
        if (step + 1) % 2000 == 0: log(f"  {tag} step {step+1}/{steps} loss={np.mean(L[-200:]):.4f}")
    model.eval().cpu(); torch.save(model.state_dict(), f"{OUTDIR}/{tag}.pt")
    json.dump(dict(prior=prior, seed=seed, scale=SCALE, steps=steps, d=D_DIM, n_params=int(npar),
                   final_loss=float(np.mean(L[-200:])), wallclock_s=time.time() - t0),
              open(f"{OUTDIR}/{tag}.json", "w"))
    log(f"{tag} DONE n={npar:,} final_loss={np.mean(L[-200:]):.4f}")

def manifest():
    nac = int(os.environ.get("D4_NSEED_AC", "16")); nn = int(os.environ.get("D4_NSEED_N", "8"))
    jobs = [(p, s, STEPS) for p in ("A", "C") for s in range(nac)]
    jobs += [("N", s, STEPS) for s in range(nn)]
    return jobs

def smoke():
    import importlib.util
    log("SMOKE 1/2: construction parity vs d4plus_oracle.gen_data (must be byte-identical)")
    here = os.path.dirname(os.path.abspath(__file__))
    argv_bak = sys.argv[:]; sys.argv = ["d4plus_oracle.py", "4"]
    spec = importlib.util.spec_from_file_location("orc4", os.path.join(here, "d4plus_oracle.py"))
    orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc); sys.argv = argv_bak
    S = sample_Sigmas(np.random.default_rng(123), 1)[0]
    for fam, r, gz in [(5, 2.0, False), (11, 4.0, False), (0, 2.0, True)]:
        a = gen_data(S, fam, r, 8, np.random.default_rng(7), gaussian=gz)
        b = orc.gen_data(S, fam, r, 8, np.random.default_rng(7), gaussian=gz)
        assert np.allclose(a, b), f"CONSTRUCTION MISMATCH fam={fam} gauss={gz} max|d|={np.abs(a-b).max():.2e}"
    log("  parity OK on 3 (family, gaussian) cases")
    log("SMOKE 2/2: 150-step tiny train, loss must decrease > 0.05")
    global OUTDIR; OUTDIR = "/tmp/d4smoke"; os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(9000); torch.manual_seed(0)
    model = PFN4().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR); model.train(); L = []
    tok = torch.full((BATCH,), NULL_TOK, dtype=torch.long, device=DEV)
    for step in range(150):
        blk = gen_batch("A", rng, N_CONTEXT + N_QUERY)
        ctx = torch.tensor(blk[:, :N_CONTEXT, :], dtype=torch.float32, device=DEV)
        q = blk[:, N_CONTEXT:, :]
        qxy = torch.tensor(q[:, :, :D_DIM - 1], dtype=torch.float32, device=DEV)
        yb = torch.tensor(bin_y(q[:, :, D_DIM - 1]), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx, qxy, tok).reshape(-1, N_BINS), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
    init, fin = float(np.mean(L[:20])), float(np.mean(L[-20:]))
    npar = sum(p.numel() for p in model.parameters())
    log(f"  params={npar:,}  loss {init:.3f} -> {fin:.3f}")
    assert fin < init - 0.05, f"LOSS DID NOT DECREASE ({init:.3f} -> {fin:.3f})"
    log("SMOKE PASS")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke(); sys.exit(0)
    TASK = int(sys.argv[1]); NTASK = int(sys.argv[2]); OUTDIR = sys.argv[3]
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = manifest(); mine = [j for i, j in enumerate(jobs) if i % NTASK == TASK]
    log(f"d4 train task {TASK}/{NTASK} on {DEV}: {len(mine)}/{len(jobs)} trainings -> {OUTDIR}")
    for p, s, st in mine: train_one(p, s, st)
    log("task complete")
