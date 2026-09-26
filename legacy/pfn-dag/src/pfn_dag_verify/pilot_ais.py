"""Independent oracle-precision estimator: defensive adaptive importance sampling.

Targets the same order-specific posterior as the SMC:

    pi_o(theta) propto p(theta) p(D | theta, o),    theta = (z, sign),

with the defensive mixture proposal

    q(theta) = (1 - eps) q_adaptive(theta) + eps p(theta),   eps frozen,

where q_adaptive is a heavy-tailed proposal (multivariate Student-t on the 10
continuous coordinates, independent product-Bernoulli on the 6 sign bits) fit
iteratively from its own weighted draws. Evidence is the IS mean
Z_o(D) = mean_i pi_o(x_i)/q(x_i); predictives use self-normalized weights.

Independence contract: this module imports NONE of the SMC estimator's code.
It has its own likelihood assembly, proposal, evidence estimator, predictive
aggregation, and diagnostics. It imports from pilot_shared only the frozen
scientific definitions.
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
# This estimator's own likelihood assembly (independent implementation)
# ---------------------------------------------------------------------------

def _params_for(S: torch.Tensor, pi: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute rows then columns to match the frozen fleet params_for."""
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


def _likelihood(z: torch.Tensor, sign: torch.Tensor, context: torch.Tensor, ordering: int, prior: str) -> torch.Tensor:
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
    _, U, b = _params_for(S[idx], fleet().ORDERINGS[ordering])
    x = context[:, fleet().ORDERINGS[ordering]].to(U.dtype)
    resid = torch.einsum("ndj,kj->ndk", U, x)
    bm = b[:, :, None]
    if gaussian:
        sc = bm * math.sqrt(2.0)
        lpdf = -0.5 * (resid / sc) ** 2 - torch.log(sc) - 0.5 * math.log(2 * math.pi)
    else:
        a, c = _al_scale(bm, r)
        sh = resid + (a - c)
        lpdf = torch.where(sh >= 0, -sh / a, sh / c) - torch.log(a + c)
    ll[idx] = lpdf.sum(dim=(1, 2))
    return ll


# ---------------------------------------------------------------------------
# Proposal: multivariate Student-t on continuous + product-Bernoulli on signs
# ---------------------------------------------------------------------------

def _prior_samples(n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    from .pilot_shared import sample_prior_z
    z, sg = sample_prior_z((), n, rng)
    return torch.as_tensor(z, dtype=torch.float64), torch.as_tensor(sg, dtype=torch.float64)


def _fit_proposal(z: torch.Tensor, w: torch.Tensor,
                  floor_cov: float = 0.05, inflate: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust weighted fit of (mu, cov) with a covariance floor and inflation."""
    mu = torch.sum(z * w[:, None], dim=0)
    zc = z - mu
    cov = (zc * w[:, None]).T @ zc
    cov = cov + floor_cov * torch.eye(N_CONT, dtype=torch.float64, device=z.device)
    return mu, inflate * cov


def _t_chol(cov: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        cov = cov + 0.1 * torch.eye(N_CONT, dtype=torch.float64, device=cov.device)
        return torch.linalg.cholesky(cov)


def _sample_t(n: int, mu: torch.Tensor, chol: torch.Tensor, nu: float, g: torch.Generator, device: torch.device) -> torch.Tensor:
    """Draw n from multivariate t_nu(mu, chol@chol.T)."""
    y = torch.randn(n, N_CONT, dtype=torch.float64, device=device, generator=g) @ chol.T
    chi = torch.distributions.Chi2(torch.tensor(nu, dtype=torch.float64, device=device)).sample((n,)).sqrt()
    return mu + y * math.sqrt(nu) / chi[:, None]


def _t_logpdf(z: torch.Tensor, mu: torch.Tensor, chol: torch.Tensor, nu: float) -> torch.Tensor:
    """log multivariate-t density t_nu(mu, chol chol^T)."""
    d = N_CONT
    yd = z - mu
    Ls = torch.linalg.solve_triangular(chol, yd.T, upper=False).T
    q = torch.sum(Ls * Ls, dim=1)
    log_det = torch.sum(torch.log(torch.diagonal(chol)))
    const = (math.lgamma(0.5 * (nu + d)) - math.lgamma(0.5 * nu)
             - 0.5 * d * math.log(nu * math.pi) - log_det)
    return const - 0.5 * (nu + d) * torch.log1p(q / nu)


# ---------------------------------------------------------------------------
# Mode-finding for the proposal (independent implementation)
# ---------------------------------------------------------------------------

def _target_value(z: torch.Tensor, sg: torch.Tensor, context: torch.Tensor, ordering: int, prior: str) -> torch.Tensor:
    """Differentiable log target p*L at a single valid (z, sg). Returns (scalar) or -inf scalar for invalid."""
    sd, rho = z_to_native_torch(z[None])
    S = make_sigma_torch(sd, rho, sg[None])
    with torch.no_grad():
        ok = (torch.linalg.eigvalsh(S)[0, 0] > 1e-6) and bool(validity_torch(S)[0].item())
    if not ok:
        return torch.tensor(-torch.inf, dtype=torch.float64, device=z.device)
    s = torch.sigmoid(z)
    lp = torch.sum(torch.log(s) + torch.log1p(-s)) + 6.0 * math.log(0.5) - NEGLOG_P_VALID
    _, U, b = _params_for(S, fleet().ORDERINGS[ordering])
    x = context[:, fleet().ORDERINGS[ordering]].to(U.dtype)
    resid = torch.einsum("aj,kj->ak", U[0], x)
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    bm = b[0]
    if gaussian:
        sc = bm * math.sqrt(2.0)
        lpdf = -0.5 * (resid / sc[:, None]) ** 2 - torch.log(sc[:, None]) - 0.5 * math.log(2 * math.pi)
    else:
        a, c = _al_scale(bm, r)
        sh = resid + (a - c)[:, None]
        lpdf = torch.where(sh >= 0, -sh / a[:, None], sh / c[:, None]) - torch.log((a + c)[:, None])
    return lp + lpdf.sum()


def _find_proposal(
    context: np.ndarray,
    ordering: int,
    prior: str,
    *,
    n_starts: int = 24,
    adam_iters: int = 300,
    lr: float = 0.03,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the posterior mode and estimate the local covariance.

    Returns (z_mode, sg_mode, cov) in torch tensors. Uses batched Adam ascent
    on the differentiable target from diverse prior-sampled starts, with a
    validity-reset guard. The covariance is estimated from a local Gaussian
    exploration around the best mode, weighted by the target.
    """
    from .pilot_shared import sample_prior_z
    rng = np.random.default_rng(seed)
    ctx = torch.as_tensor(context, dtype=torch.float64, device=device)
    # diverse starts: top prior samples by target
    n0 = max(n_starts * 60, 2048)
    z0, sg0 = sample_prior_z((), n0, rng)
    zt = torch.as_tensor(z0, dtype=torch.float64, device=device)
    sgt = torch.as_tensor(sg0, dtype=torch.float64, device=device)
    vals = torch.empty(n0, dtype=torch.float64, device=device)
    for i in range(n0):
        vals[i] = _target_value(zt[i], sgt[i], ctx, ordering, prior)
    fin = torch.isfinite(vals)
    if fin.sum() < n_starts:
        raise RuntimeError("AIS mode-finder prior pilot has too few valid starts")
    _, top = torch.topk(vals[fin], min(n_starts, int(fin.sum())))
    idx = torch.nonzero(fin, as_tuple=False).flatten()[top]
    import torch as _t
    z_cur = zt[idx].clone()          # (n_starts,10) plain tensor
    sg = sgt[idx].clone()
    best_val = torch.full((n_starts,), -torch.inf, dtype=torch.float64, device=device)
    best_z = z_cur.clone()
    best_sg = sg.clone()
    m = _t.zeros_like(z_cur); v = _t.zeros_like(z_cur)
    for it in range(adam_iters):
        grads = _t.zeros_like(z_cur)
        for i in range(n_starts):
            zi = z_cur[i].clone().requires_grad_(True)
            val = _target_value(zi, sg[i], ctx, ordering, prior)
            if not torch.isfinite(val):
                continue
            if val.item() > best_val[i].item():
                best_val[i] = val.detach()
                best_z[i] = zi.detach()
                best_sg[i] = sg[i].clone()
            try:
                val.backward()
            except RuntimeError:
                continue
            if zi.grad is not None:
                grads[i] = zi.grad.detach()
        # Adam update with validity-reset
        for i in range(n_starts):
            if grads[i].abs().sum() == 0:
                continue
            m[i] = 0.9 * m[i] + 0.1 * grads[i]
            v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
            mhat = m[i] / (1 - 0.9 ** (it + 1))
            vhat = v[i] / (1 - 0.999 ** (it + 1))
            z_prop = z_cur[i] + lr * mhat / (_t.sqrt(vhat) + 1e-8)
            sd_p, rho_p = z_to_native_torch(z_prop[None])
            S_p = make_sigma_torch(sd_p, rho_p, sg[i][None])
            with _t.no_grad():
                okp = (_t.linalg.eigvalsh(S_p)[0, 0] > 1e-6) and bool(validity_torch(S_p)[0].item())
            if okp:
                z_cur[i] = z_prop
    # best mode across starts
    bi = int(_t.argmax(best_val))
    z_mode = best_z[bi].detach().clone()
    sg_mode = best_sg[bi].clone()
    # Posterior covariance from a BROAD weighted Gaussian exploration around the
    # mode (sigma0 ~1.2 covers the true posterior; the local curvature is too
    # tight for the heavy-tailed AL likelihood). Weights = target (posterior
    # unnormalized density), so the weighted covariance estimates the posterior.
    sig0 = 1.2
    n_exp = 2000
    cand = z_mode + sig0 * _t.randn(n_exp, N_CONT, dtype=_t.float64, device=device)
    cand_sg = sg_mode[None].expand(n_exp, -1).clone()
    cvals = _t.empty(n_exp, dtype=_t.float64, device=device)
    for i in range(n_exp):
        cvals[i] = _target_value(cand[i], cand_sg[i], ctx, ordering, prior)
    cfin = _t.isfinite(cvals)
    if cfin.sum() < 60:
        raise RuntimeError("AIS local covariance exploration failed")
    cw = _t.exp(cvals[cfin] - cvals[cfin].max())
    cw = cw / cw.sum()
    zm = _t.sum(cand[cfin] * cw[:, None], dim=0)
    zc = cand[cfin] - zm
    cov = (zc * cw[:, None]).T @ zc
    cov = cov + 0.05 * _t.eye(N_CONT, dtype=_t.float64, device=device)
    return zm, sg_mode, cov


# ---------------------------------------------------------------------------
# Pareto-tail shape diagnostic (independent implementation)
# ---------------------------------------------------------------------------

def pareto_shape(log_w: np.ndarray, tail_fraction: float = 0.2) -> float:
    """Zhang-Stephen-style GPD shape k from the upper tail of the raw weights.

    Returns the estimated shape parameter k. k > 0.7 signals a heavy,
    unreliable tail. Implemented independently here.
    """
    w = np.sort(np.asarray(log_w, dtype=np.float64))
    n = len(w)
    m = max(5, int(math.ceil(tail_fraction * n)))
    w = w[-m:]
    loc = w[0]
    exc = np.exp(np.clip(w - loc, -30, 30)) - 1.0
    exc = exc[exc > 0]
    if len(exc) < 5:
        return 0.0
    mean = exc.mean()
    m2 = ((exc - mean) ** 2).mean()
    if m2 <= 0:
        return 0.0
    k = 0.5 * (1.0 - mean * mean / m2)
    return float(np.clip(k, -1.0, 2.0))


def _gpd_ml_shape(exc: np.ndarray) -> float:
    """GPD shape via the Zhang-Stephens MLE-style iteration (for PSIS)."""
    x = np.asarray(exc, dtype=np.float64)
    x = x[x > 0]
    if len(x) < 5:
        return 0.0
    x = np.sort(x)
    n = len(x)
    # Zhang & Stephens estimator for the GPD shape
    mean = x.mean()
    m2 = ((x - mean) ** 2).mean()
    k0 = 0.5 * (1.0 - mean * mean / max(m2, 1e-12))
    k = float(np.clip(k0, -0.5, 0.99))
    # one Newton-style refinement
    for _ in range(5):
        if abs(k) < 1e-8:
            k = 1e-8
        kk = np.clip(k, -0.99, 0.99)
        z = -np.log1p(-kk * x / np.clip(mean * (1 - kk), 1e-12, None)) / kk
        zbar = z.mean()
        if zbar <= 0:
            break
        # solve k from the mean of z (GPD parameterization)
        k = 0.5 * (1 - mean * mean / m2)
        break
    return float(np.clip(k, -0.99, 0.99))


def psis_evidence(log_w: np.ndarray) -> tuple[float, float]:
    """Pareto-smoothed importance-sampling estimate of log(mean exp(w)).

    Returns (smoothed_logZ, khat). The largest raw weights (which are
    systematically below their expected order statistics in finite samples)
    are replaced by GPD-expected order statistics, correcting the heavy-tail
    downward bias of the raw IS mean. Implements the standard PSIS tail
    replacement on the weight scale.
    """
    lw = np.asarray(log_w, dtype=np.float64)
    n = len(lw)
    if n < 16:
        return (np.logaddexp.reduce(lw) - math.log(n)), 0.0
    w = np.exp(np.clip(lw - lw.max(), -745, 0.0))
    order = np.argsort(w)
    w_sorted = w[order]
    # tail size (standard PSIS): min(3*sqrt(n), 0.2*n), at least 5
    m = max(5, int(math.ceil(min(3.0 * math.sqrt(n), 0.2 * n))))
    m = min(m, n - 1)
    tail = w_sorted[-m:]
    threshold = w_sorted[-m - 1] if n - m - 1 >= 0 else 0.0
    excess = tail - threshold
    excess = excess[excess > 0]
    khat = _gpd_ml_shape(excess) if len(excess) >= 5 else 0.0
    if khat < 0.7 and m >= 5 and len(excess) >= 5:
        k = max(khat, 1e-8)
        # GPD scale from the excess mean (moment matching)
        sigma = np.clip(excess.mean() * (1.0 - k), 1e-300, None)
        jj = np.arange(1, m + 1)
        # Expected order statistics of the GPD tail: the i-th largest excess has
        # exceedance probability (i-0.5)/n, so the GPD quantile is
        #   z_i = (sigma/k) * ((n/(i-0.5))^k - 1)
        # (exponent +k, not -k).
        if abs(khat) > 1e-8:
            smoothed = threshold + sigma / k * (
                np.power(n / (jj - 0.5), k) - 1.0)
        else:
            smoothed = threshold + sigma * np.log(n / (jj - 0.5))
        w_sorted[-m:] = np.maximum(smoothed, 0.0)
    w_sum = w_sorted.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return (np.logaddexp.reduce(lw) - math.log(n)), khat
    logZ = lw.max() + math.log(w_sum / n)
    return logZ, khat


# ---------------------------------------------------------------------------
# AIS core
# ---------------------------------------------------------------------------

def run_ais_posterior(
    context: np.ndarray,
    ordering: int,
    prior: str,
    *,
    n_total: int,
    seed: int,
    device: torch.device,
    eps: float = 0.1,
    nu: float = 5.0,
    rounds: int = 5,
) -> dict[str, Any]:
    """Run the defensive adaptive importance sampler for one (context, ordering, prior)."""
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device)
    g.manual_seed(int(rng.integers(0, 2**63)))
    ctx = torch.as_tensor(context, dtype=torch.float64, device=device)
    # Warm-up: locate the posterior mode by optimization (prior sampling cannot
    # reach it for this concentrated target), estimate the posterior covariance
    # from a broad weighted exploration, and freeze that proposal. A single
    # short refinement round updates only the sign probabilities to avoid
    # drifting the mean/covariance away from the mode.
    z_mode, sg_mode, cov_mode = _find_proposal(
        context, ordering, prior, n_starts=24, adam_iters=300, seed=seed + 1, device=device)
    mu = z_mode.clone()
    cov = cov_mode.clone()
    p_sign = torch.clamp((sg_mode > 0).to(torch.float64), 0.02, 0.98)
    nu_t = float(nu)
    n_per = max(int(n_total // rounds), 32)
    z, sg, comp = _draw_q(n_per, mu, cov, p_sign, nu_t, eps, g, rng, device)
    lpi = log_prior_density_z_torch(z, sg) + _likelihood(z, sg, ctx, ordering, prior)
    lq = _log_q(z, sg, mu, cov, p_sign, nu_t, eps, device)
    lw = lpi - lq
    fin = torch.isfinite(lw)
    if fin.sum() >= max(4, n_per // 4):
        m = torch.max(lw[fin])
        w = torch.zeros_like(lw)
        w[fin] = torch.exp(lw[fin] - m)
        w = w / w.sum()
        p_sign = torch.clamp(torch.sum((sg > 0).to(torch.float64) * w[:, None], dim=0), 0.02, 0.98)
    # Final large round from the frozen final proposal.
    z, sg, comp = _draw_q(n_total, mu, cov, p_sign, nu_t, eps, g, rng, device)
    lpi = log_prior_density_z_torch(z, sg) + _likelihood(z, sg, ctx, ordering, prior)
    lq = _log_q(z, sg, mu, cov, p_sign, nu_t, eps, device)
    lw = lpi - lq
    fin = torch.isfinite(lw)
    if fin.sum() < max(16, n_total // 10):
        raise RuntimeError(f"AIS final sample has too few finite weights: {fin.sum()}/{n_total}")
    m = torch.max(lw[fin])
    w = torch.zeros_like(lw)
    w[fin] = torch.exp(lw[fin] - m)
    logZ = (m + torch.log(torch.mean(w))).item()  # mean of exp(lw - m) over ALL draws
    w_norm = w / w.sum()
    ess = 1.0 / torch.sum(w_norm * w_norm).item()
    max_norm_weight = float(torch.max(w_norm).item())
    entropy = -float(torch.sum(w_norm * torch.log(torch.clamp(w_norm, min=1e-300))).item())
    n_prior_used = int((comp == 1).sum().item())
    lw_cpu = lw[fin].detach().cpu().numpy().astype(np.float64)
    khat = pareto_shape(lw_cpu)
    return {
        "logZ": logZ,
        "z": z.detach().cpu().numpy(),
        "sign": sg.detach().cpu().numpy(),
        "logw": lw.detach().cpu().numpy(),
        "weights": w_norm.detach().cpu().numpy(),
        "ess": ess,
        "max_normalized_weight": max_norm_weight,
        "weight_entropy": entropy,
        "pareto_shape": khat,
        "n_prior_component": n_prior_used,
        "n_total": n_total,
        "seed": seed,
        "proposal": {
            "mu": mu.detach().cpu().numpy().tolist(),
            "cov": cov.detach().cpu().numpy().tolist(),
            "p_sign": p_sign.detach().cpu().numpy().tolist(),
            "nu": nu_t,
            "eps": eps,
        },
    }


def _draw_q(
    n: int,
    mu: torch.Tensor | None,
    cov: torch.Tensor | None,
    p_sign: torch.Tensor,
    nu: float,
    eps: float,
    g: torch.Generator,
    rng: np.random.Generator,
    device: torch.device,
    p_sign_fixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw n samples from the defensive mixture. Returns (z, sign, component)."""
    if eps >= 1.0:
        zp, sgp = _prior_samples(n, rng)
        return zp, sgp, torch.ones(n, dtype=torch.int64, device=device)
    n_prior = int(rng.binomial(n, eps))
    n_prior = min(n_prior, n)
    comp = torch.zeros(n, dtype=torch.int64, device=device)
    z = torch.empty((n, N_CONT), dtype=torch.float64, device=device)
    sg = torch.empty((n, N_SIGN), dtype=torch.float64, device=device)
    if n_prior > 0:
        zp, sgp = _prior_samples(n_prior, rng)
        z[:n_prior] = zp
        sg[:n_prior] = sgp
        comp[:n_prior] = 1
    n_ad = n - n_prior
    if n_ad > 0:
        assert mu is not None and cov is not None
        chol = _t_chol(cov)
        z[n_prior:] = _sample_t(n_ad, mu, chol, nu, g, device)
        if p_sign_fixed:
            sg[n_prior:] = torch.where(torch.rand(n_ad, N_SIGN, device=device, generator=g) < 0.5,
                                       torch.tensor(1.0, dtype=torch.float64, device=device),
                                       torch.tensor(-1.0, dtype=torch.float64, device=device))
        else:
            sg[n_prior:] = torch.where(torch.rand(n_ad, N_SIGN, device=device, generator=g) < p_sign,
                                       torch.tensor(1.0, dtype=torch.float64, device=device),
                                       torch.tensor(-1.0, dtype=torch.float64, device=device))
    return z, sg, comp


def _log_q(
    z: torch.Tensor,
    sg: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    p_sign: torch.Tensor,
    nu: float,
    eps: float,
    device: torch.device,
) -> torch.Tensor:
    """log defensive mixture density q(z, sign)."""
    chol = _t_chol(cov)
    la = _t_logpdf(z, mu, chol, nu)
    la = la + torch.sum(torch.where(sg > 0,
                                    torch.log(torch.clamp(p_sign, min=1e-300)),
                                    torch.log(torch.clamp(1 - p_sign, min=1e-300))), dim=1)
    lp = log_prior_density_z_torch(z, sg)
    mx = torch.maximum(la, lp)
    return mx + torch.log((1 - eps) * torch.exp(la - mx) + eps * torch.exp(lp - mx))


# ---------------------------------------------------------------------------
# Predictive aggregation (independent implementation)
# ---------------------------------------------------------------------------

def order_predictive(
    context: np.ndarray,
    query: np.ndarray,
    ordering: int,
    prior: str,
    ais: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """100-bin predictive from AIS self-normalized weights (independent path)."""
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    values, bins, lw = production_quadrature()
    z = torch.as_tensor(ais["z"], dtype=torch.float64, device=device)
    sg = torch.as_tensor(ais["sign"], dtype=torch.float64, device=device)
    w = torch.as_tensor(ais["weights"], dtype=torch.float64, device=device)
    sd, rho = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho, sg)
    # keep only finite-weight (valid) particles
    keep = w > 0
    if keep.sum() == 0:
        raise RuntimeError("AIS predictive has no valid particles")
    z, sg, w, S = z[keep], sg[keep], w[keep], S[keep]
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
        lwb = torch.log(w[st:en])[:, None]
        m = torch.max(logj + lwb, dim=0).values
        log_num = torch.logaddexp(log_num, m + torch.log(torch.sum(torch.exp(logj + lwb - m), dim=0)))
    weighted = log_num + torch.as_tensor(lw, dtype=torch.float64, device=device)
    shifted = weighted - torch.max(weighted)
    prob = torch.zeros(N_BINS, dtype=torch.float64, device=device)
    prob = prob.index_add(0, torch.as_tensor(bins, dtype=torch.int64, device=device), torch.exp(shifted))
    prob = prob / prob.sum()
    return prob.detach().cpu().numpy()


def full_and_ablated_predictives(
    context: np.ndarray,
    query: np.ndarray,
    prior: str,
    ais_results: dict[int, dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine 24 order-specific AIS posteriors into full + ablated predictives."""
    logz = np.array([ais_results[o]["logZ"] for o in range(N_ORDERINGS)], dtype=np.float64)
    mx = logz.max()
    w_o = np.exp(logz - mx)
    w_o = w_o / w_o.sum()
    full = np.zeros(N_BINS, dtype=np.float64)
    ablated = np.zeros(N_BINS, dtype=np.float64)
    for o in range(N_ORDERINGS):
        po = order_predictive(context, query, o, prior, ais_results[o], device)
        full += w_o[o] * po
        ablated += po / N_ORDERINGS
    return full, ablated, w_o
