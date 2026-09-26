"""Original finite-study model/generator definitions; training CLI omitted."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from .corrected_sem import ResidualSpec, generate_observational, params_for
from .pilot_shared import N_BINS
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)

@dataclass(frozen=True)
class ModelConfig:
    name: str
    d: int
    n_bins: int = N_BINS
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 2
    n_params_target: int | None = None


class PFN(nn.Module):
    """Vanilla PFN: transformer encoder over context tokens + query token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.point_embed = nn.Linear(cfg.d, cfg.d_model)
        self.query_embed = nn.Linear(cfg.d - 1, cfg.d_model)
        self.token_embed = nn.Embedding(3, cfg.d_model)  # 0 ctx, 1 query, 2 null
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=cfg.n_layers)
        self.out_head = nn.Linear(cfg.d_model, cfg.n_bins)

    def forward(self, ctx, qxy, tok):
        ce = self.point_embed(ctx)                     # (B, n_ctx, D)
        te = self.token_embed(tok).unsqueeze(1)
        outs = []
        for q in range(qxy.shape[1]):
            qe = self.query_embed(qxy[:, q, :]).unsqueeze(1)
            out = self.out_head(self.transformer(torch.cat([te, ce, qe], 1))[:, -1, :])
            outs.append(out)
        return torch.stack(outs, 1)

    def predict_bin_probs(self, ctx, qxy, tok) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(ctx, qxy, tok)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def _vectorized_observational(
    world,
    rng: np.random.Generator,
    k_idx: np.ndarray,
    o_idx: np.ndarray,
    n_pts: int,
) -> np.ndarray:
    """Fully vectorized observational generation for a batch.

    Uses the world's precomputed per-(k,o) L_unit / b / permutation cache, so
    a training step avoids per-item Cholesky and residual sampling overhead.
    Returns X (B, n_pts, d) in original coordinates (float32).
    """
    c = world.params_cache()
    L_unit = c["L_unit"][k_idx, o_idx]          # (B, d, d)
    b = c["b"][k_idx, o_idx]                    # (B, d)
    perms = c["perms"][o_idx]                   # (B, d)
    B = len(k_idx)
    d = world.d
    e = np.empty((B, n_pts, d), dtype=np.float64)
    spec = world.spec
    if spec.eps <= 0.0:
        e = rng.normal(0.0, math.sqrt(2.0) * b[:, None, :], (B, n_pts, d))
    else:
        c_al, c_la = _al_ac_batch(b, spec.r)
        al = (rng.exponential(c_al[:, None, :], (B, n_pts, d))
              - rng.exponential(c_la[:, None, :], (B, n_pts, d))
              - (c_al - c_la)[:, None, :])
        if spec.eps >= 1.0:
            e = al
        else:
            g = rng.normal(0.0, math.sqrt(2.0) * b[:, None, :], (B, n_pts, d))
            # Amendment E.1: ELEMENTWISE mask, matching residual_logpdf. The
            # previous (B, 1, 1) mask (introduced with the 3360b99 vectorization)
            # shared one component draw across every row AND every coordinate of a
            # training example, so the training law differed from both the eval
            # sampler and the oracle density at 0 < eps < 1.
            mask = rng.random((B, n_pts, d)) < spec.eps
            e = np.where(mask, al, g)
    # Forward map x_pi = e @ L_unit.T per batch item: sum_d e[b,p,d] L_unit[b,q,d].
    x_pi = np.einsum("bpd,bqd->bpq", e, L_unit)   # (B, n_pts, d) permuted
    X = np.empty((B, n_pts, d), dtype=np.float64)
    for bi in range(B):
        X[bi][:, perms[bi]] = x_pi[bi]
    return X.astype(np.float32, copy=False)


def _al_ac_batch(b: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def gen_batch(
    world,
    rng: np.random.Generator,
    batch: int,
    n_ctx: int,
    n_query: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch of observational (context, query, target-bin) examples.

    Returns (X (B, n_ctx+n_query, d), k_idx (B,), o_idx (B,)). X is raw
    data; the last n_query rows hold the query (d-1 covariates + target).
    """
    K, O = world.K, world.O
    k_idx = rng.integers(0, K, batch)
    o_idx = rng.integers(0, O, batch)
    X = _vectorized_observational(world, rng, k_idx, o_idx, n_ctx + n_query)
    return X, k_idx, o_idx


def lr_at(step: int, peak: float, total: int, warm: int) -> float:
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

