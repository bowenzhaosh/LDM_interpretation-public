"""Branch B: train PFNs on the exact finite prior.

Sampling: uniformly select one of the K atoms, uniformly select an ordering,
generate a context from that (sigma, ordering, prior). Train the PFN4 (base
scale: 256d/512ff/4h/2L, ~1.1M params) on next-bin prediction. Save dense
checkpoints every 500 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pilot_shared import fleet, D_DIM, N_ORDERINGS, N_BINS

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N_CONTEXT = 30
N_QUERY = 7
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)
D_MODEL = 256
D_FF = 512
N_HEADS = 4
N_LAYERS = 2
PEAK_LR = 1e-3
WARMUP = 400
NULL_TOK = 2
BATCH = 32


def bin_y(v: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_EDGES[1:-1], v)


def gen_batch_from_library(
    sigmas: np.ndarray,
    prior: str,
    rng: np.random.Generator,
    batch: int,
    n_pts: int,
) -> np.ndarray:
    """Generate (batch, n_pts, 4) from the finite atom library."""
    K = len(sigmas)
    gaussian = prior == "N"
    r = 2.0 if gaussian else 4.0
    idx = rng.integers(0, K, batch)
    fams = rng.integers(0, N_ORDERINGS, batch)
    X = np.zeros((batch, n_pts, D_DIM), dtype=np.float32)
    f = fleet()
    for bi in range(batch):
        pi = f.ORDERINGS[int(fams[bi])]
        _, U, b = f.params_for(sigmas[int(idx[bi])][None], pi)
        U, b = U[0], b[0]
        if gaussian:
            e = rng.normal(0, np.sqrt(2.0) * b[None, :], (n_pts, D_DIM))
        else:
            c = np.sqrt(2.0 * b * b / (1.0 + r * r))
            a = r * c
            e = rng.exponential(a, (n_pts, D_DIM)) - rng.exponential(c, (n_pts, D_DIM)) - (a - c)
        xpi = e @ U.T
        x = np.empty((n_pts, D_DIM)); x[:, list(pi)] = xpi
        X[bi] = x
    return X


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


def lr_at(step: int, peak: float, total: int, warm: int) -> float:
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def train_one(
    sigmas: np.ndarray,
    prior: str,
    seed: int,
    steps: int,
    outdir: Path,
    ckpt_every: int = 500,
    tag_prefix: str = "",
) -> dict[str, Any]:
    tag = f"{tag_prefix}bb_{prior}_s{seed}_st{steps}"
    ckpt_path = outdir / f"{tag}.pt"
    sidecar = outdir / f"{tag}.json"
    if ckpt_path.exists():
        existing = json.loads(sidecar.read_text())
        return existing
    torch.manual_seed(10000 * seed + ord(prior))
    rng = np.random.default_rng(90000 + 100 * seed + ord(prior))
    model = PFN4().to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR)
    model.train()
    L = []
    tok = torch.full((BATCH,), NULL_TOK, dtype=torch.long, device=DEV)
    t0 = time.time()
    for step in range(steps):
        if step > 0 and step % ckpt_every == 0:
            torch.save(model.state_dict(), str(outdir / f"{tag}_ck{step}.pt"))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, PEAK_LR, steps, WARMUP)
        blk = gen_batch_from_library(sigmas, prior, rng, BATCH, N_CONTEXT + N_QUERY)
        ctx = torch.tensor(blk[:, :N_CONTEXT, :], dtype=torch.float32, device=DEV)
        q = blk[:, N_CONTEXT:, :]
        qxy = torch.tensor(q[:, :, :D_DIM - 1], dtype=torch.float32, device=DEV)
        yb = torch.tensor(bin_y(q[:, :, D_DIM - 1]), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx, qxy, tok).reshape(-1, N_BINS), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        L.append(loss.item())
    model.eval().cpu()
    torch.save(model.state_dict(), str(ckpt_path))
    result = {
        "prior": prior, "seed": seed, "steps": steps, "d": D_DIM,
        "n_params": int(npar), "final_loss": float(np.mean(L[-200:])),
        "wallclock_s": float(time.time() - t0), "library_sha256": "",
    }
    sidecar.write_text(json.dumps(result, indent=2))
    return result


def train_fleet(
    library_path: Path,
    prior: str,
    seeds: list[int],
    steps: int,
    outdir: Path,
    ckpt_every: int = 500,
) -> list[dict[str, Any]]:
    with np.load(library_path) as z:
        sigmas = z["sigmas"]
    results = []
    for seed in seeds:
        r = train_one(sigmas, prior, seed, steps, outdir, ckpt_every)
        results.append(r)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--prior", type=str, required=True, choices=["C", "N"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--tag-prefix", type=str, default="")
    args = p.parse_args(argv)
    with np.load(args.library) as z:
        sigmas = z["sigmas"]
    result = train_one(sigmas, args.prior, args.seed, args.steps, args.out, args.ckpt_every, args.tag_prefix)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
