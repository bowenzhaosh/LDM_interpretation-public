"""Track B — corrected model experiments (B1 vanilla PFN saturation, B2 direct
order classifier, B3 epsilon continuum).

Both models train on data generated with the CORRECTED forward map
(corrected_sem.generate_observational), which fixes the Branch B inverse-map
defect. They are evaluated against the corrected exact oracle
(corrected_oracle), which conditions atom weights on the context.

B1 — vanilla PFN: context + query (observe d-1 covariates, predict target bin),
     next-bin cross-entropy. Bayes regret = NLL_PFN - NLL_exact_Bayes on a
     fixed held-out panel.

B2 — direct order classifier: context D -> order posterior q(o | D), trained
     with the sampled latent order as the one-hot label. The population
     cross-entropy optimum is the Bayesian posterior over order.
"""

from __future__ import annotations

import json
from pathlib import Path

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corrected_sem import ResidualSpec, generate_observational, params_for
from .pilot_shared import N_BINS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)


def bin_y(v: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_EDGES[1:-1], v)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

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


class OrderClassifier(nn.Module):
    """Direct order classifier: context D -> order logits (O,)."""

    def __init__(self, cfg: ModelConfig, n_orders: int):
        super().__init__()
        self.cfg = cfg
        self.point_embed = nn.Linear(cfg.d, cfg.d_model)
        self.token_embed = nn.Embedding(2, cfg.d_model)  # 0 ctx, 1 cls
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=cfg.n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.head = nn.Linear(cfg.d_model, n_orders)

    def forward(self, ctx):
        ce = self.point_embed(ctx)                      # (B, n_ctx, D)
        cls = self.cls_token.expand(ctx.shape[0], -1, -1)
        h = self.transformer(torch.cat([cls, ce], 1))
        return self.head(h[:, 0, :])                     # (B, O)

    def predict_order_probs(self, ctx) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(ctx)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Data generators (corrected forward-map; Branch B's inverse-map bug is NOT
# reproduced here).
# ---------------------------------------------------------------------------

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


def gen_classifier_batch(
    world,
    rng: np.random.Generator,
    batch: int,
    n_ctx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch of (context, latent k, latent order) for the direct classifier."""
    K, O = world.K, world.O
    k_idx = rng.integers(0, K, batch)
    o_idx = rng.integers(0, O, batch)
    X = _vectorized_observational(world, rng, k_idx, o_idx, n_ctx)
    return X, k_idx, o_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lr_at(step: int, peak: float, total: int, warm: int) -> float:
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def train_pfn(
    world,
    steps: int,
    seed: int,
    cfg: ModelConfig,
    n_ctx: int = 20,
    n_query: int = 7,
    batch: int = 32,
    peak_lr: float = 1e-3,
    warmup: int = 200,
    ckpt_every: int = 0,
    outdir=None,
    tag_prefix: str = "",
    ckpt_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Train a vanilla PFN. If ckpt_steps is non-empty, the state dict at each
    listed step is returned in `checkpoints` (dict step -> state_dict), giving
    the learning curve from ONE training run (checkpoints-as-curve, not
    replicates)."""
    # A ckpt_step this run cannot reach used to be a SILENT no-op: the capture
    # fires on (step + 1) in ckpt_set, so anything above `steps` simply never
    # happened and the caller got a shorter dict than it asked for. That is how
    # a fleet reports training with dose-0 and produces nothing. Refuse at the
    # point of the mistake instead. Additive: no existing caller passes an
    # unreachable step, so no prior run's behaviour changes.
    bad = sorted(s for s in ckpt_steps if s < 0 or s > steps)
    if bad:
        raise ValueError(
            f"ckpt_steps {bad} cannot be reached in a {steps}-step run; "
            "requesting one would have been silently dropped")
    torch.manual_seed(1000 * seed + 7)
    rng = np.random.default_rng(10000 + seed)
    model = PFN(cfg).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    tok = torch.full((batch,), 2, dtype=torch.long, device=DEV)
    losses = []
    checkpoints: dict[int, dict] = {}
    ckpt_set = set(ckpt_steps)
    model.train()
    # Amendment F.2c: step 0 is the DOSE-0 state, before any optimizer step, and
    # it is the floor curve of the calibrated y-axis. The capture below fires on
    # `(step + 1) in ckpt_set`, so 0 was unreachable and no dose-0 state could
    # ever be persisted. Requesting 0 was previously a silent no-op rather than
    # an error, which is the worst version: a fleet would have "run with dose-0"
    # and produced nothing. Existing callers are unaffected -- 0 was never a
    # valid ckpt_step.
    if 0 in ckpt_set:
        checkpoints[0] = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        # ...and it goes to DISK HERE, before step 1, not in the persistence
        # block at the end. Two reasons, both operational rather than tidy:
        # a run that dies at step 40k still leaves a usable floor curve, and a
        # dose-0 file whose mtime predates every other checkpoint is evidence
        # that it is the untrained state rather than something reconstructed
        # afterwards. Its sidecar cannot carry a final loss, so it carries the
        # pre-training facts and says which it is.
        if outdir is not None:
            _p = Path(outdir)
            _p.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoints[0], _p / f"{tag_prefix}_ck0.pt")
            (_p / f"{tag_prefix}_ck0.provenance.json").write_text(json.dumps({
                "tag_prefix": tag_prefix, "dose0": True, "steps_taken": 0,
                "steps_planned": int(steps), "seed": int(seed),
                "n_ctx": int(n_ctx), "n_query": int(n_query), "batch": int(batch),
                "peak_lr": float(peak_lr), "warmup": int(warmup),
                "ckpt_steps": [int(x) for x in ckpt_steps],
                "n_params": int(npar),
                "world": {"d": int(world.d), "K": int(world.K),
                          "eps": float(world.spec.eps), "r": float(world.spec.r)},
                "cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                            else str(v)) for k, v in vars(cfg).items()},
                "torch": torch.__version__, "cuda": (torch.version.cuda or "cpu"),
                "device": str(DEV), "numpy": np.__version__,
                "amendment": "F.2c",
            }, indent=2))
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, peak_lr, steps, warmup)
        X, _, _ = gen_batch(world, rng, batch, n_ctx, n_query)
        ctx = torch.tensor(X[:, :n_ctx, :], dtype=torch.float32, device=DEV)
        q = X[:, n_ctx:, :]
        qxy = torch.tensor(q[:, :, :world.d - 1], dtype=torch.float32, device=DEV)
        yb = torch.tensor(bin_y(q[:, :, world.d - 1]), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx, qxy, tok).reshape(-1, cfg.n_bins), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) in ckpt_set:
            checkpoints[step + 1] = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
    model.eval()
    # Amendment E: persistence. The old gate required ckpt_every > 0 for the FINAL
    # save, and never wrote the ckpt_steps checkpoints to disk at all, so every
    # d=3 result came from weights that did not survive the process. outdir alone
    # now persists the final weights, each ckpt_steps checkpoint, and a provenance
    # sidecar pinning torch/CUDA/numpy so the run can be re-derived.
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), outdir / f"{tag_prefix}.pt")
        for step, sd in checkpoints.items():
            if step == 0:
                continue            # already on disk, from before step 1
            torch.save(sd, outdir / f"{tag_prefix}_ck{step}.pt")
        (outdir / f"{tag_prefix}.provenance.json").write_text(json.dumps({
            "tag_prefix": tag_prefix,
            "steps": int(steps),
            "seed": int(seed),
            "n_ctx": int(n_ctx),
            "n_query": int(n_query),
            "batch": int(batch),
            "peak_lr": float(peak_lr),
            "warmup": int(warmup),
            "ckpt_steps": [int(x) for x in ckpt_steps],
            "n_params": int(npar),
            "final_loss": float(np.mean(losses[-200:])),
            "world": {"d": int(world.d), "K": int(world.K),
                      "eps": float(world.spec.eps), "r": float(world.spec.r)},
            "cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                    for k, v in vars(cfg).items()},
            "torch": torch.__version__,
            "cuda": (torch.version.cuda or "cpu"),
            "device": str(DEV),
            "numpy": np.__version__,
            "amendment": "E",
        }, indent=2))
    return {"n_params": int(npar), "final_loss": float(np.mean(losses[-200:])),
            "model": model, "losses": losses, "checkpoints": checkpoints}


def train_classifier(
    world,
    steps: int,
    seed: int,
    cfg: ModelConfig,
    n_ctx: int = 20,
    batch: int = 32,
    peak_lr: float = 1e-3,
    warmup: int = 200,
    outdir=None,
    tag_prefix: str = "",
) -> dict[str, Any]:
    torch.manual_seed(1000 * seed + 11)
    rng = np.random.default_rng(20000 + seed)
    model = OrderClassifier(cfg, world.O).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    losses = []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, peak_lr, steps, warmup)
        X, _, o_idx = gen_classifier_batch(world, rng, batch, n_ctx)
        ctx = torch.tensor(X, dtype=torch.float32, device=DEV)
        yb = torch.tensor(o_idx, dtype=torch.long, device=DEV)
        loss = F.cross_entropy(model(ctx), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    if outdir is not None:
        torch.save(model.state_dict(), outdir / f"{tag_prefix}.pt")
    return {"n_params": int(npar), "final_loss": float(np.mean(losses[-200:])),
            "model": model, "losses": losses}


# ---------------------------------------------------------------------------
# Evaluation vs the corrected exact oracle
# ---------------------------------------------------------------------------

def exact_bayes_predictive(world, ctx, x_q, target):
    from .corrected_oracle import exact_joint_posterior, obs_query_operator
    post = exact_joint_posterior(world, ctx)
    qop = obs_query_operator(world, x_q, target)
    return qop.predictive(post["w_lo"]), post


def evaluate_pfn_state(
    state: dict,
    cfg: ModelConfig,
    world,
    contexts,
    device,
) -> dict[str, Any]:
    """Evaluate a checkpoint state_dict on the fixed panel (one curve point)."""
    model = PFN(cfg).to(DEV)
    model.load_state_dict(state)
    return evaluate_pfn_checkpoint(model, world, contexts, device)


def evaluate_pfn_checkpoint(
    model: PFN,
    world,
    contexts: list[tuple[np.ndarray, np.ndarray, int, int]],
    device: torch.device,
) -> dict[str, Any]:
    """Bayes regret and JS on a fixed panel of (ctx, x_q, target, bin).

    contexts: list of (ctx (n_ctx,d), x_q (d-1,), target int, outcome_bin int).
    """
    from .corrected_oracle import exact_joint_posterior, obs_query_operator
    regrets, js_list = [], []
    per_target: dict[int, list[float]] = {}
    for ctx, x_q, target, ob in contexts:
        post = exact_joint_posterior(world, ctx)
        qop = obs_query_operator(world, x_q, target)
        p_bayes = qop.predictive(post["w_lo"])
        ctx_t = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
        qxy = torch.tensor(x_q, dtype=torch.float32, device=device).reshape(1, 1, -1)
        tok = torch.full((1,), 2, dtype=torch.long, device=device)
        p_pfn = model.predict_bin_probs(ctx_t, qxy, tok)[0, 0]
        nll_pfn = -math.log(max(p_pfn[ob], 1e-300))
        nll_bayes = -math.log(max(p_bayes[ob], 1e-300))
        regrets.append(nll_pfn - nll_bayes)
        per_target.setdefault(int(target), []).append(nll_pfn - nll_bayes)
        m = 0.5 * (p_pfn + p_bayes)
        js_list.append(0.5 * np.sum(p_pfn * np.log(p_pfn / np.maximum(m, 1e-300)))
                       + 0.5 * np.sum(p_bayes * np.log(p_bayes / np.maximum(m, 1e-300))))
    return {
        "bayes_regret_mean": float(np.mean(regrets)),
        "bayes_regret_se": float(np.std(regrets) / math.sqrt(len(regrets))),
        "js_mean": float(np.mean(js_list)),
        # Per-target decomposition: the PFN is trained to predict ONLY the last
        # coordinate (column d-1), but the eval panel samples the target
        # uniformly over {0..d-1} with no target-index input to the model. The
        # mixed-panel mean is therefore dominated by out-of-task queries; the
        # target=d-1 subset is the in-task regret and the honest measure of the
        # trained predictive.
        "regret_by_target": {
            str(t): {"n": len(v), "mean": float(np.mean(v)),
                     "se": float(np.std(v) / math.sqrt(len(v)))}
            for t, v in sorted(per_target.items())
        },
    }


def evaluate_classifier_checkpoint(
    model: OrderClassifier,
    world,
    contexts: list[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """KL/JS/TV/Brier/entropy-error/MAP-accuracy vs exact p(o | D)."""
    from .corrected_oracle import exact_joint_posterior
    O = world.O
    kls, jss, tvs, briers, ent_errs, map_acc = [], [], [], [], [], []
    for ctx in contexts:
        post = exact_joint_posterior(world, ctx)
        p_exact = post["w_o"]
        ctx_t = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
        p_pred = model.predict_order_probs(ctx_t)[0]
        p_pred = np.maximum(p_pred, 1e-300)
        p_exact_c = np.maximum(p_exact, 1e-300)
        kls.append(np.sum(p_exact_c * (np.log(p_exact_c) - np.log(p_pred))))
        m = 0.5 * (p_pred + p_exact_c)
        jss.append(0.5 * np.sum(p_pred * np.log(p_pred / m))
                   + 0.5 * np.sum(p_exact_c * np.log(p_exact_c / m)))
        tvs.append(0.5 * np.sum(np.abs(p_pred - p_exact_c)))
        briers.append(np.sum((p_pred - p_exact) ** 2))
        ent_pred = -np.sum(p_pred * np.log(p_pred))
        ent_exact = -np.sum(p_exact_c * np.log(p_exact_c))
        ent_errs.append(abs(ent_pred - ent_exact))
        map_acc.append(int(np.argmax(p_pred) == np.argmax(p_exact)))
    n = len(contexts)
    return {"kl_mean": float(np.mean(kls)), "js_mean": float(np.mean(jss)),
            "tv_mean": float(np.mean(tvs)), "brier_mean": float(np.mean(briers)),
            "entropy_error_mean": float(np.mean(ent_errs)),
            "map_accuracy": float(np.mean(map_acc)),
            "n_contexts": int(n)}


# ---------------------------------------------------------------------------
# Fixed evaluation panels
# ---------------------------------------------------------------------------

def make_eval_panel(world, n_contexts: int, n_rows: int, seed: int,
                    n_query_per_context: int = 1):
    """Fixed held-out panel of (ctx, x_q, target-bin) triples.

    The panel is deterministic for a given seed and used for ALL checkpoints
    (so the learning curve is on identical evaluation points).
    """
    rng = np.random.default_rng(seed)
    panel = []
    for c in range(n_contexts):
        k = int(rng.integers(world.K))
        o = int(rng.integers(world.O))
        ctx = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                     world.spec, n_rows)
        for _ in range(n_query_per_context):
            target = int(rng.integers(world.d))
            qrow = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                          world.spec, 1)[0]
            x_q = np.delete(qrow, target)
            ob = int(bin_y(qrow[target]))
            panel.append((ctx, x_q, target, ob))
    return panel


def make_eval_panel_contexts(world, n_contexts: int, n_rows: int, seed: int,
                             n_query_per_context: int = 1):
    """The DISTINCT contexts of the eval panel, in panel order.

    ``make_eval_panel`` appends the SAME context array object once per query, so
    a panel with n_query_per_context=2 has 120 entries over 60 distinct
    contexts. Anything that averages a per-CONTEXT quantity over
    ``[q[0] for q in panel]`` therefore counts every context twice: the mean is
    unaffected (the multiplicity is uniform) but the standard error comes out a
    factor sqrt(n_query_per_context) too small, which silently inflates every
    z in Amendment F.2b's near-tie clause.

    This is built by deduplicating an actual panel rather than by replaying the
    generation loop, so it cannot drift from what the PFN is scored on: the
    contexts depend on every prior RNG draw including the per-query ones.
    """
    panel = make_eval_panel(world, n_contexts, n_rows, seed,
                            n_query_per_context=n_query_per_context)
    seen: dict[int, np.ndarray] = {}
    for ctx, *_ in panel:
        seen.setdefault(id(ctx), ctx)
    out = list(seen.values())
    if len(out) != n_contexts:
        raise AssertionError(
            f"expected {n_contexts} distinct contexts, found {len(out)}")
    return out


def make_classifier_panel(world, n_contexts: int, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    panel = []
    for c in range(n_contexts):
        k = int(rng.integers(world.K))
        o = int(rng.integers(world.O))
        ctx = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                     world.spec, n_rows)
        panel.append(ctx)
    return panel
