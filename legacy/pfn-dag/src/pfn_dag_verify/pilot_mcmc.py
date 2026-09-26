"""Independent oracle-precision estimator: adaptive MCMC + thermodynamic integration.

Targets the same order-specific posterior pi_o(theta) propto p(theta) p(D|theta,o)
as the SMC, but with a chain-based sampler (no resampling, no importance
weights), making it genuinely independent and immune to importance-weight
degeneracy. The predictive comes directly from the beta=1 posterior samples;
the order marginal likelihood comes from thermodynamic integration over a
beta-ladder of the posterior mean log-likelihood.

Independence contract: this module imports NONE of the SMC or AIS estimator
code. It uses only the frozen scientific substrate (pilot_shared) plus its own
proposal, acceptance, evidence, and predictive code.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .pilot_shared import (
    D_DIM,
    N_ORDERINGS,
    N_BINS,
    N_CONT,
    N_SIGN,
    fleet,
    log_prior_density_z_torch,
    z_to_native_torch,
    make_sigma_torch,
    validity_torch,
    production_quadrature,
    P_VALID_MC,
    NEGLOG_P_VALID,
)


# ---------------------------------------------------------------------------
# Own likelihood assembly (independent implementation)
# ---------------------------------------------------------------------------

def _params_for(S: torch.Tensor, pi: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Spi = S[:, pi][:, :, pi]
    L = torch.linalg.cholesky(Spi)
    d = torch.diagonal(L, dim1=-2, dim2=-1)
    Lunit = L / d[..., None, :]
    U = torch.linalg.inv(Lunit)
    b = torch.sqrt(torch.clamp(d ** 2, min=1e-12) / 2.0)
    return Lunit, U, b


def _al_scale(b: torch.Tensor, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def likelihood_batch(z: torch.Tensor, sign: torch.Tensor, context: torch.Tensor, ordering: int, prior: str) -> torch.Tensor:
    """(N,) log p(context | z, sign, ordering); -inf on invalid particles."""
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    sd, rho = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho, sign)
    ev = torch.linalg.eigvalsh(S)
    ok = (ev[:, 0] > 1e-6) & validity_torch(S)
    ll = torch.full((z.shape[0],), -torch.inf, dtype=torch.float64, device=z.device)
    idx = torch.nonzero(ok, as_tuple=False).flatten()
    if idx.numel() == 0:
        return ll
    perm = fleet().ORDERINGS[ordering]
    x = context[:, perm].to(torch.float64)
    chunk = 256
    for cst in range(0, idx.numel(), chunk):
        cidx = idx[cst:cst + chunk]
        try:
            _, U, b = _params_for(S[cidx], perm)
            resid = torch.einsum("ndj,kj->ndk", U, x)
            bm = b[:, :, None]
            if gaussian:
                sc = bm * math.sqrt(2.0)
                lpdf = -0.5 * (resid / sc) ** 2 - torch.log(sc) - 0.5 * math.log(2 * math.pi)
            else:
                a, c = _al_scale(bm, r)
                sh = resid + (a - c)
                lpdf = torch.where(sh >= 0, -sh / a, sh / c) - torch.log(a + c)
            ll[cidx] = lpdf.sum(dim=(1, 2))
        except torch.linalg.LinAlgError:
            continue
    return ll


# ---------------------------------------------------------------------------
# Own mode finder (independent implementation, gradient-free)
# ---------------------------------------------------------------------------

def _target_np(z: np.ndarray, sg: np.ndarray, context: np.ndarray, ordering: int, prior: str) -> float:
    """Single-point log target p*L in native coords (for the mode search)."""
    from .pilot_shared import log_prior_density_z, z_to_native as z2n, make_sigma as mk_sigma
    sd, rho = z2n(z)
    S = mk_sigma(sd, rho, sg)
    ev = np.linalg.eigvalsh(S)
    if ev[0] <= 1e-6 or not bool(np.all(fleet().validity_keep(S[None]))):
        return -np.inf
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
    lp = float(np.sum(np.log(np.clip(s, 1e-300, 1)) + np.log(np.clip(1 - s, 1e-300, 1))) + 6.0 * math.log(0.5) - NEGLOG_P_VALID)
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    _, U, b = fleet().params_for(S[None], fleet().ORDERINGS[ordering])
    x = context[:, fleet().ORDERINGS[ordering]]
    resid = np.einsum("aj,kj->ak", U[0], x)
    bm = b[0]
    if gaussian:
        sc = bm * math.sqrt(2.0)
        lpdf = -0.5 * (resid / sc[:, None]) ** 2 - np.log(sc[:, None]) - 0.5 * math.log(2 * math.pi)
    else:
        c = np.sqrt(2.0 * bm * bm / (1.0 + r * r)); a = r * c
        sh = resid + (a - c)[:, None]
        lpdf = np.where(sh >= 0, -sh / a[:, None], sh / c[:, None]) - np.log((a + c)[:, None])
    return lp + float(lpdf.sum())


def find_mode(context: np.ndarray, ordering: int, prior: str, *, n_starts: int = 24, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Locate the posterior mode by coordinate ascent from prior-sampled starts."""
    from .pilot_shared import sample_prior_z
    rng = np.random.default_rng(seed)
    z, sg = sample_prior_z((), n_starts * 40, rng)
    best = None
    best_val = -np.inf
    for i in range(n_starts):
        zc = z[i].copy(); sgc = sg[i].copy()
        val = _target_np(zc, sgc, context, ordering, prior)
        for _ in range(300):
            # coordinate ascent with random steps, occasional sign flips
            improved = False
            for dim in range(N_CONT):
                step = 0.3
                for sign_try in (1.0, -1.0):
                    zc2 = zc.copy(); zc2[dim] += sign_try * step
                    v2 = _target_np(zc2, sgc, context, ordering, prior)
                    if v2 > val:
                        zc = zc2; val = v2; improved = True
                        break
            if np.random.default_rng(seed + i).random() < 0.05:
                k = np.random.default_rng(seed + i + 1).integers(N_SIGN)
                sgc2 = sgc.copy(); sgc2[k] *= -1
                v2 = _target_np(zc, sgc2, context, ordering, prior)
                if v2 > val:
                    sgc = sgc2; val = v2
            if not improved:
                break
        if val > best_val:
            best_val = val; best = (zc, sgc)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Adaptive Metropolis-Hastings (own implementation)
# ---------------------------------------------------------------------------

def _mcmc_chain(
    context: np.ndarray,
    ordering: int,
    prior: str,
    z0: torch.Tensor,
    sg0: torch.Tensor,
    beta: float,
    n_iter: int,
    n_chains: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Run n_chains parallel adaptive-MH chains at temperature beta.

    Returns (z, sign, ll, accept_history). The proposal covariance is adapted
    from the running chain (Robbins-Monro scaling toward ~0.23 acceptance).
    """
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device)
    g.manual_seed(int(rng.integers(0, 2**63)))
    ctx = torch.as_tensor(context, dtype=torch.float64, device=device)
    # Accept a single start (expand) or a batch of starts (use directly or cycle).
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
        sg0 = sg0.unsqueeze(0)
    n0 = z0.shape[0]
    if n0 == 1:
        z = z0.expand(n_chains, -1).clone()
        sg = sg0.expand(n_chains, -1).clone()
    elif n0 >= n_chains:
        z = z0[:n_chains].clone()
        sg = sg0[:n_chains].clone()
    else:
        idx = torch.arange(n_chains, device=device) % n0
        z = z0[idx].clone()
        sg = sg0[idx].clone()
    ll = likelihood_batch(z, sg, ctx, ordering, prior)
    lp = log_prior_density_z_torch(z, sg)
    target = lp + beta * ll
    # adaptive covariance
    step = 0.5
    cov = 0.2 * torch.eye(N_CONT, dtype=torch.float64, device=device)
    try:
        Lc = torch.linalg.cholesky(cov)
    except RuntimeError:
        Lc = torch.eye(N_CONT, dtype=torch.float64, device=device)
    accept_hist: list[float] = []
    n_acc = 0
    n_tot = 0
    n_iter_chunks = max(1, n_iter // 10)
    z_hist = torch.empty((n_iter, n_chains, N_CONT), dtype=torch.float64, device=device)
    sg_hist = torch.empty((n_iter, n_chains, N_SIGN), dtype=torch.float64, device=device)
    ll_hist = torch.empty((n_iter, n_chains), dtype=torch.float64, device=device)
    for it in range(n_iter):
        eps = torch.randn(n_chains, N_CONT, dtype=torch.float64, device=device, generator=g)
        z_prop = z + step * (eps @ Lc.T)
        flip = torch.rand(n_chains, N_SIGN, device=device, generator=g) < 0.12
        sg_prop = torch.where(flip, -sg, sg)
        lp_p = log_prior_density_z_torch(z_prop, sg_prop)
        ll_p = likelihood_batch(z_prop, sg_prop, ctx, ordering, prior)
        target_p = lp_p + beta * ll_p
        a = torch.exp(torch.clamp(target_p - target, max=0.0))
        u = torch.rand(n_chains, device=device, generator=g)
        acc = u < a
        z = torch.where(acc[:, None], z_prop, z)
        sg = torch.where(acc[:, None], sg_prop, sg)
        ll = torch.where(acc, ll_p, ll)
        target = torch.where(acc, target_p, target)
        z_hist[it] = z
        sg_hist[it] = sg
        ll_hist[it] = ll
        n_acc += int(acc.sum().item())
        n_tot += n_chains
        # update the adaptive covariance every 50 iterations from valid samples
        if (it + 1) % 50 == 0:
            accept_hist.append(n_acc / n_tot if n_tot else 0.0)
            if accept_hist[-1] > 0.0:
                step *= math.exp(0.6 * (accept_hist[-1] - 0.23))
                step = float(np.clip(step, 0.02, 4.0))
            fin = torch.isfinite(ll)
            if fin.sum() >= 10:
                zf = z[fin]
                zm = zf.mean(dim=0)
                zc = zf - zm
                cov = (zc.T @ zc) / max(zf.shape[0], 1) + 0.05 * torch.eye(N_CONT, dtype=torch.float64, device=device)
                try:
                    Lc = torch.linalg.cholesky(cov)
                except RuntimeError:
                    pass
            n_acc = 0
            n_tot = 0
    # return the full history, discarding burn-in (first 10%)
    burn = n_iter // 10
    z_hist = z_hist[burn:].reshape(-1, N_CONT)
    sg_hist = sg_hist[burn:].reshape(-1, N_SIGN)
    ll_hist = ll_hist[burn:].reshape(-1)
    return z_hist, sg_hist, ll_hist, accept_hist


def run_mcmc_predictive(
    context: np.ndarray,
    ordering: int,
    prior: str,
    *,
    n_chains: int = 200,
    n_iter: int = 2500,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Posterior samples and predictive from an adaptive-MH chain at beta=1."""
    z_mode, sg_mode = find_mode(context, ordering, prior, n_starts=24, seed=seed + 7)
    z0 = torch.as_tensor(z_mode, dtype=torch.float64, device=device)
    sg0 = torch.as_tensor(sg_mode, dtype=torch.float64, device=device)
    z, sg, ll, acc = _mcmc_chain(context, ordering, prior, z0, sg0, 1.0, n_iter, n_chains, seed, device)
    fin = torch.isfinite(ll)
    return {
        "z": z[fin].detach().cpu().numpy(),
        "sign": sg[fin].detach().cpu().numpy(),
        "ll": ll[fin].detach().cpu().numpy(),
        "accept": float(np.mean(acc)) if acc else 0.0,
        "n_effective": int(fin.sum().item()),
    }


def mcmc_evidence_ti(
    context: np.ndarray,
    ordering: int,
    prior: str,
    *,
    betas: np.ndarray,
    n_chains: int = 200,
    n_iter_per_beta: int = 1500,
    seed: int,
    device: torch.device,
) -> float:
    """Thermodynamic integration estimate of log Z_o(D).

    log Z = int_0^1 E_{pi_beta}[log L] d beta, trapezoidal over the beta ladder.
    The beta=0 chain MUST be initialized from the exact prior (the integrand is
    steepest there and the posterior mode is not in the prior bulk).
    """
    from .pilot_shared import sample_prior_z
    betas = np.asarray(betas, dtype=np.float64)
    if len(betas) < 2 or betas[0] > 1e-3 or abs(betas[-1] - 1.0) > 1e-12:
        raise RuntimeError("TI beta ladder must span [0,1] with a fine low-beta end")
    if len(np.unique(np.round(betas * 1e6))) != len(betas):
        raise RuntimeError("TI beta ladder has duplicate/too-close betas")
    ctx = torch.as_tensor(context, dtype=torch.float64, device=device)
    # beta=0 chain from the exact prior
    zp, sgp = sample_prior_z((), n_chains, np.random.default_rng(seed + 1000))
    z_cur = torch.as_tensor(zp, dtype=torch.float64, device=device)
    sg_cur = torch.as_tensor(sgp, dtype=torch.float64, device=device)
    means: list[float] = []
    for bi, b in enumerate(betas):
        z, sg, ll, _ = _mcmc_chain(
            context, ordering, prior, z_cur, sg_cur, float(b),
            n_iter_per_beta, n_chains, seed + bi, device)
        fin = torch.isfinite(ll)
        if fin.sum() < 10:
            raise RuntimeError("MCMC-TI chain has too few valid samples")
        means.append(float(torch.mean(ll[fin]).item()))
        # warm-start the next beta from the last valid states
        finidx = torch.nonzero(fin).flatten()
        z_cur = z[finidx[-1]]
        sg_cur = sg[finidx[-1]]
    m = np.asarray(means)
    # trapezoid: logZ = sum_i (b_{i+1}-b_i) * (m_i + m_{i+1})/2
    return float(np.sum(np.diff(betas) * (m[:-1] + m[1:]) / 2.0))


# ---------------------------------------------------------------------------
# Predictive aggregation (independent implementation)
# ---------------------------------------------------------------------------

def order_predictive(
    context: np.ndarray,
    query: np.ndarray,
    ordering: int,
    prior: str,
    mcmc: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """100-bin predictive from MCMC posterior samples (equal weights)."""
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    values, bins, lw = production_quadrature()
    z = torch.as_tensor(mcmc["z"], dtype=torch.float64, device=device)
    sg = torch.as_tensor(mcmc["sign"], dtype=torch.float64, device=device)
    sd, rho = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho, sg)
    perm = fleet().ORDERINGS[ordering]
    pts = np.empty((len(values), 4))
    pts[:, :3] = query
    pts[:, 3] = values
    xp = torch.as_tensor(pts[:, perm], dtype=torch.float64, device=device)
    log_num = torch.full((len(values),), -torch.inf, dtype=torch.float64, device=device)
    chunk = 1024
    for st in range(0, len(z), chunk):
        en = min(len(z), st + chunk)
        _, U, b = _params_for(S[st:en], perm)
        resid = torch.einsum("bdj,kj->bdk", U, xp)
        bm = b[:, :, None]
        if gaussian:
            sc = bm * math.sqrt(2.0)
            lpdf = -0.5 * (resid / sc) ** 2 - torch.log(sc) - 0.5 * math.log(2 * math.pi)
        else:
            a, c = _al_scale(bm, r)
            sh = resid + (a - c)
            lpdf = torch.where(sh >= 0, -sh / a, sh / c) - torch.log(a + c)
        logj = lpdf.sum(dim=1)
        m = torch.max(logj, dim=0).values
        log_num = torch.logaddexp(log_num, m + torch.log(torch.sum(torch.exp(logj - m), dim=0)))
    weighted = log_num + torch.as_tensor(lw, dtype=torch.float64, device=device)
    shifted = weighted - torch.max(weighted)
    prob = torch.zeros(N_BINS, dtype=torch.float64, device=device)
    prob = prob.index_add(0, torch.as_tensor(bins, dtype=torch.int64, device=device), torch.exp(shifted))
    prob = prob / prob.sum()
    return prob.detach().cpu().numpy()
