"""Primary oracle-precision estimator: annealed sequential Monte Carlo.

Targets, for each context D, prior P, and ordering o:

    pi_o(theta) propto p(theta) p(D | theta, o),    theta = (z, sign),

with tempered targets pi_t propto p(theta) p(D|theta,o)^{beta_t},
0 = beta_0 < ... < beta_T = 1. Produces the SMC normalizing-constant estimate
of the order marginal likelihood Z_o(D), the order-conditioned 100-bin
posterior predictive, and the full / uniformly-ablated mixtures.

Independence contract: this module implements its own likelihood assembly,
proposal, evidence estimator, predictive aggregation, and decision logic. It
imports from pilot_shared only frozen scientific definitions (prior density,
validity, transforms, fleet, quadrature) and never imports the AIS module.
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
)


# ---------------------------------------------------------------------------
# Torch likelihood assembly (this module's own implementation)
# ---------------------------------------------------------------------------

def _residual_logpdf(residual: torch.Tensor, b: torch.Tensor, r: float, gaussian: bool) -> torch.Tensor:
    """residual (..., dims); b (..., dims). Returns log density per entry."""
    if gaussian:
        scale = b * math.sqrt(2.0)
        return -0.5 * (residual / scale) ** 2 - torch.log(scale) - 0.5 * math.log(2 * math.pi)
    a, c = _al_scales(b, r)
    shifted = residual + (a - c)
    return torch.where(shifted >= 0, -shifted / a, shifted / c) - torch.log(a + c)


def _al_scales(b: torch.Tensor, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def _params_for_torch(S: torch.Tensor, pi: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch port of the frozen fleet params_for (S (...,4,4)) -> (Lunit, U, b).

    The permutation permutes rows then columns (S[n, pi[i], pi[j]]), matching
    the frozen numpy fleet. Ellipsis indexing would wrongly touch only the last
    axis, so the explicit row/column indexing is used.
    """
    Spi = S[:, pi][:, :, pi]
    L = torch.linalg.cholesky(Spi)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    Lunit = L / diag[..., None, :]
    Dv = diag ** 2
    U = torch.linalg.inv(Lunit)
    b = torch.sqrt(torch.clamp(Dv, min=1e-12) / 2.0)
    return Lunit, U, b


def log_likelihood_batch(
    z: torch.Tensor,
    sign: torch.Tensor,
    context: torch.Tensor,
    ordering: int,
    prior: str,
) -> torch.Tensor:
    """Vectorized p(context | z, sign, ordering) -> (N,) float64.

    Invalid (non-PD / validity-violating) particles get -inf. This is this
    estimator's own likelihood assembly.
    """
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    sd, rho_mag = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho_mag, sign)
    ev = torch.linalg.eigvalsh(S)
    pd = ev[:, 0] > 1e-6
    valid = validity_torch(S)
    ok = pd & valid
    ll = torch.full((z.shape[0],), -torch.inf, dtype=torch.float64, device=z.device)
    vidx = torch.nonzero(ok, as_tuple=False).flatten()
    if vidx.numel() == 0:
        return ll
    perm = fleet().ORDERINGS[ordering]
    x = context[:, perm].to(torch.float64)  # (30,4)
    # Process in chunks so one numerically-marginal matrix cannot invalidate
    # the whole batch (batch Cholesky fails entirely if any member fails).
    chunk = 256
    for cst in range(0, vidx.numel(), chunk):
        cidx = vidx[cst:cst + chunk]
        try:
            _, U, b = _params_for_torch(S[cidx], perm)
            residual = torch.einsum("ndj,kj->ndk", U, x)
            lpdf = _residual_logpdf(residual, b[:, :, None], r, gaussian)
            ll[cidx] = lpdf.sum(dim=(1, 2))
        except torch.linalg.LinAlgError:
            # numerically marginal positive-definiteness: leave those as -inf
            continue
    return ll


def _systematic_resample(logw: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Systematic resampling. Returns indices (N,)."""
    n = logw.numel()
    w = torch.exp(logw - logw.max())
    w = w / w.sum()
    u = (torch.arange(n, device=logw.device, dtype=torch.float64) + torch.rand(1, device=logw.device, generator=rng, dtype=torch.float64)) / n
    c = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(c, u, right=True)
    return torch.clamp(idx, max=n - 1)


# ---------------------------------------------------------------------------
# SMC core
# ---------------------------------------------------------------------------

def run_smc_posterior(
    context: np.ndarray,
    ordering: int,
    prior: str,
    *,
    n_particles: int,
    seed: int,
    device: torch.device,
    cess_target: float = 0.5,
    resample_frac: float = 0.5,
    mh_steps: int = 6,
    sign_flip_prob: float = 0.12,
    max_temps: int = 60,
    target_accept: float = 0.23,
) -> dict[str, Any]:
    """Run the annealed SMC for one (context, ordering, prior).

    context: (30,4) float64 numpy. Returns the result dict with posterior
    particles, log evidence, and full diagnostics.
    """
    n = int(n_particles)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device)
    g.manual_seed(int(rng.integers(0, 2**63)))
    ctx = torch.as_tensor(context, dtype=torch.float64, device=device)
    # Initial particles: exact prior (shared with other orderings only through
    # the prior; the continuation uses this ordering's own RNG streams).
    z_np, sign_np = _sample_prior_np(n, rng)
    z = torch.as_tensor(z_np, dtype=torch.float64, device=device)
    sign = torch.as_tensor(sign_np, dtype=torch.float64, device=device)
    logw = torch.zeros(n, dtype=torch.float64, device=device)
    ll = log_likelihood_batch(z, sign, ctx, ordering, prior)
    logZ = 0.0
    beta = 0.0
    temps = [0.0]
    inc_log_normalizers: list[float] = []
    resample_events: list[float] = []
    ess_history: list[float] = []
    accept_history: list[float] = []
    posterior_mean_ll_history: list[float] = []
    n_finite = torch.isfinite(ll).sum().item()
    if n_finite < n * 0.5:
        raise RuntimeError(f"SMC prior initialization failed: only {n_finite}/{n} valid particles")
    prior_mean_ll = float(torch.mean(ll[torch.isfinite(ll)]).item())
    while beta < 1.0 - 1e-12:
        # --- adaptive temperature by weighted conditional ESS ---
        # The current (possibly non-uniform) particle weights enter the
        # conditional-ESS criterion; an unweighted criterion over-estimates it.
        w_cur = torch.exp(logw - logw.max())
        w_cur = w_cur / w_cur.sum()
        lo, hi = beta, 1.0
        beta_new = None
        for _ in range(24):
            mid = lo + (hi - lo) / 2.0
            d = mid - beta
            if d < 1e-10:
                beta_new = mid
                break
            m = torch.max(d * ll)
            inc = w_cur * torch.exp(d * ll - m)
            num = inc.sum()
            den = (w_cur * torch.exp(2.0 * d * ll - 2.0 * m)).sum()
            cess = n * num * num / den if den > 0 else 0.0
            if cess.item() >= cess_target * n:
                lo = mid
            else:
                hi = mid
        if beta_new is None:
            beta_new = lo if (lo - beta) > 1e-7 else min(1.0, beta + 0.01)
        if beta_new - beta < 1e-6:
            beta_new = min(1.0, beta + 0.01)
        dbeta = beta_new - beta
        # Weighted incremental log-normalizer: sum_i w_i * exp(dbeta*ll_i),
        # where w_i are the current (possibly non-uniform) particle weights.
        log_inc = logw + dbeta * ll
        m2 = torch.max(log_inc)
        logw_old_sum = torch.logsumexp(logw, dim=0)
        inc_val = (m2 + torch.log(torch.sum(torch.exp(log_inc - m2))) - logw_old_sum).item()
        logZ += inc_val
        inc_log_normalizers.append(inc_val)
        logw = logw + dbeta * ll
        w = torch.exp(logw - logw.max())
        w = w / w.sum()
        ess = 1.0 / torch.sum(w * w)
        ess_history.append(ess.item())
        posterior_mean_ll_history.append(float(torch.sum(w * ll).item()))
        # --- resampling ---
        if ess.item() < resample_frac * n:
            idx = _systematic_resample(logw, g)
            z, sign, logw, ll = z[idx], sign[idx], torch.full_like(logw, -math.log(n)), ll[idx]
            resample_events.append(beta_new)
        beta = beta_new
        temps.append(beta)
        # --- rejuvenation: adaptive covariance MH on z + sign flips ---
        w_cur = torch.exp(logw - logw.max())
        w_cur = w_cur / w_cur.sum()
        zm = torch.sum(z * w_cur[:, None], dim=0)
        zc = z - zm
        cov = (zc * w_cur[:, None]).T @ zc + 1e-4 * torch.eye(N_CONT, dtype=torch.float64, device=device)
        try:
            Lc = torch.linalg.cholesky(cov)
        except RuntimeError:
            Lc = torch.eye(N_CONT, dtype=torch.float64, device=device)
        target_logp = log_prior_density_z_torch(z, sign) + beta * ll
        n_acc = 0
        n_tot = 0
        step = 1.0
        accept_hist: list[float] = []
        for _ in range(mh_steps):
            eps = torch.randn(n, N_CONT, dtype=torch.float64, device=device, generator=g)
            z_prop = z + step * (eps @ Lc.T)
            flip = torch.rand(n, N_SIGN, device=device, generator=g, dtype=torch.float64) < sign_flip_prob
            sign_prop = torch.where(flip, -sign, sign)
            lp_p = log_prior_density_z_torch(z_prop, sign_prop)
            ll_p = log_likelihood_batch(z_prop, sign_prop, ctx, ordering, prior)
            target_p = lp_p + beta * ll_p
            a = torch.exp(torch.clamp(target_p - target_logp, max=0.0))
            u = torch.rand(n, device=device, generator=g, dtype=torch.float64)
            acc = u < a
            z = torch.where(acc[:, None], z_prop, z)
            sign = torch.where(acc[:, None], sign_prop, sign)
            ll = torch.where(acc, ll_p, ll)
            target_logp = torch.where(acc, target_p, target_logp)
            n_acc += acc.sum().item()
            n_tot += n
            if (_ + 1) % 5 == 0:
                accept_hist.append(n_acc / n_tot if n_tot else 0.0)
                # Robbins-Monro step scaling toward the ~0.23 target acceptance
                if step > 0.02 and step < 6.0:
                    step *= math.exp(0.6 * (accept_hist[-1] - target_accept))
        accept_history.append(n_acc / n_tot if n_tot else 0.0)
        if len(temps) > max_temps:
            raise RuntimeError("SMC exceeded the temperature cap")
    w = torch.exp(logw - logw.max())
    w = w / w.sum()
    uniq = torch.unique(z, dim=0).shape[0]
    return {
        "logZ": logZ,
        "z": z.detach().cpu().numpy(),
        "sign": sign.detach().cpu().numpy(),
        "logw": logw.detach().cpu().numpy(),
        "weights": w.detach().cpu().numpy(),
        "ll": ll.detach().cpu().numpy(),
        "ess": 1.0 / torch.sum(w * w).item(),
        "n_particles": n,
        "seed": seed,
        "temperatures": temps,
        "incremental_log_normalizers": inc_log_normalizers,
        "resample_events": resample_events,
        "ess_history": ess_history,
        "accept_history": accept_history,
        "posterior_mean_ll_history": posterior_mean_ll_history,
        "prior_mean_ll": prior_mean_ll,
        "unique_particles": int(uniq),
    }


def thermodynamic_integration(smc: dict[str, Any]) -> float:
    """Independent logZ estimate from the same SMC via path sampling.

    log Z = int_0^1 E_{pi_beta}[log L] d beta, trapezoidal rule over the
    recorded temperature schedule and per-temperature posterior means of log L.
    This is an independent estimator of the evidence that does not share the
    incremental-normalizer arithmetic.
    """
    betas = np.asarray(smc["temperatures"], dtype=np.float64)
    means = np.asarray(smc["posterior_mean_ll_history"], dtype=np.float64)
    if len(betas) < 2 or len(means) != len(betas) - 1:
        raise RuntimeError("SMC history is incomplete for thermodynamic integration")
    # means[t] is the posterior mean of ll at beta = betas[t+1]
    # (recorded after reaching beta_{t+1}). The beta=0 endpoint is E_prior[ll];
    # use it for the first interval instead of substituting means[0].
    e0 = float(smc.get("prior_mean_ll", np.nan))
    db = np.diff(betas)
    lo = means
    hi = np.concatenate([[e0], means])
    return float(np.sum(db * (lo + hi[:len(db)]) / 2.0))


def _sample_prior_np(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n exact-prior points in (z, sign) using the shared prior sampler."""
    from .pilot_shared import sample_prior_z
    return sample_prior_z((), n, rng)


# ---------------------------------------------------------------------------
# Predictive aggregation (this module's own implementation)
# ---------------------------------------------------------------------------

def order_predictive(
    context: np.ndarray,
    query: np.ndarray,
    ordering: int,
    prior: str,
    smc: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Order-conditioned 100-bin predictive from the SMC posterior.

    p(y | x_q, D, o) = sum_i w_i p(y | x_q, z_i, sign_i, o), evaluated on the
    frozen production quadrature.
    """
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    values, bins, lw = production_quadrature()
    n = len(smc["z"])
    z = torch.as_tensor(smc["z"], dtype=torch.float64, device=device)
    sign = torch.as_tensor(smc["sign"], dtype=torch.float64, device=device)
    w = torch.as_tensor(smc["weights"], dtype=torch.float64, device=device)
    # keep only finite-weight (valid) particles
    keep = (w > 0) & torch.isfinite(w)
    if not keep.any():
        raise RuntimeError("SMC predictive has no valid particles")
    z, sign, w = z[keep], sign[keep], w[keep]
    sd, rho_mag = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho_mag, sign)
    perm = fleet().ORDERINGS[ordering]
    pts = np.empty((len(values), 4))
    pts[:, :3] = query
    pts[:, 3] = values
    xp = torch.as_tensor(pts[:, perm], dtype=torch.float64, device=device)
    log_num = torch.full((len(values),), -torch.inf, dtype=torch.float64, device=device)
    chunk = 1024
    for st in range(0, n, chunk):
        en = min(n, st + chunk)
        _, U, b = _params_for_torch(S[st:en], perm)
        residual = torch.einsum("bdj,kj->bdk", U, xp)  # (c,4,V)
        lpdf = _residual_logpdf(residual, b[:, :, None], r, gaussian)
        logj = lpdf.sum(dim=1)  # (c,V)
        lwb = torch.log(w[st:en])[:, None]
        m = torch.max(logj + lwb, dim=0).values
        log_num = torch.logaddexp(log_num, m + torch.log(torch.sum(torch.exp(logj + lwb - m), dim=0)))
    weighted = log_num + torch.as_tensor(lw, dtype=torch.float64, device=device)
    shifted = weighted - torch.max(weighted)
    dens = torch.exp(shifted)
    prob = torch.zeros(N_BINS, dtype=torch.float64, device=device)
    prob = prob.index_add(0, torch.as_tensor(bins, dtype=torch.int64, device=device), dens)
    prob = prob / prob.sum()
    return prob.detach().cpu().numpy()


def full_and_ablated_predictives(
    context: np.ndarray,
    query: np.ndarray,
    prior: str,
    smc_results: dict[int, dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine 24 order-specific SMC posteriors into full + ablated predictives.

    Returns (full, ablated, ordering_posterior). p(o|D) propto Z_o(D).
    """
    logz = np.array([smc_results[o]["logZ"] for o in range(N_ORDERINGS)], dtype=np.float64)
    mx = logz.max()
    w_o = np.exp(logz - mx)
    w_o = w_o / w_o.sum()
    full = np.zeros(N_BINS, dtype=np.float64)
    ablated = np.zeros(N_BINS, dtype=np.float64)
    for o in range(N_ORDERINGS):
        po = order_predictive(context, query, o, prior, smc_results[o], device)
        full += w_o[o] * po
        ablated += po / N_ORDERINGS
    return full, ablated, w_o
