"""Branch B: exact finite-prior enumeration oracle.

Given a finite library of K valid covariance atoms (the exact training prior
of the PFN), this module computes the EXACT Bayesian posterior over those K
atoms for each of the 24 orderings. All quantities are exact up to float64:
no importance sampling, no annealing, no Monte Carlo error.

This is the ground truth against which the PFN's outputs are compared.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .pilot_shared import (
    fleet,
    D_DIM,
    N_ORDERINGS,
    N_BINS,
    production_quadrature,
)


def _params_for(S: np.ndarray, pi: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy params_for matching the frozen fleet."""
    Spi = S[:, pi][:, :, pi]
    L = np.linalg.cholesky(Spi)
    diag = np.diagonal(L, axis1=1, axis2=2)
    Lunit = L / diag[:, None, :]
    U = np.linalg.inv(Lunit)
    b = np.sqrt(np.maximum(diag ** 2, 1e-12) / 2.0)
    return Lunit, U, b


def _residual_logpdf(residual: np.ndarray, b: np.ndarray, r: float, gaussian: bool) -> np.ndarray:
    """(K, d) residual array, (K, d) b array -> (K, d) logpdf."""
    if gaussian:
        sc = b * math.sqrt(2.0)
        return -0.5 * (residual / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)
    c = np.sqrt(2.0 * b * b / (1.0 + r * r))
    a = r * c
    shifted = residual + (a - c)
    return np.where(shifted >= 0, -shifted / a, shifted / c) - np.log(a + c)


def exact_likelihood(
    sigmas: np.ndarray,
    context: np.ndarray,
    ordering: int,
    prior: str,
) -> np.ndarray:
    """p(D | sigma_k, o) for all K atoms, one ordering. Returns (K,) float64."""
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    perm = fleet().ORDERINGS[ordering]
    x = context[:, perm]
    K = len(sigmas)
    ll = np.empty(K, dtype=np.float64)
    chunk = 512
    for st in range(0, K, chunk):
        en = min(K, st + chunk)
        Sk = sigmas[st:en]
        _, U, b = _params_for(Sk, perm)
        resid = np.einsum("kdj,mj->kdm", U, x)  # (c,4,30)
        lpdf = _residual_logpdf(resid, b[:, :, None], r, gaussian)
        ll[st:en] = lpdf.sum(axis=(1, 2))
    return ll


def exact_evidence_and_posterior(
    sigmas: np.ndarray,
    context: np.ndarray,
    prior: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact order evidence Z_o(D) and order posterior p(o|D).

    Returns (logZ_24, w_o_24, ll_Kx24). logZ[o] = log mean_k p(D|sigma_k, o).
    The prior over atoms is uniform (1/K).
    """
    K = len(sigmas)
    logZ = np.empty(N_ORDERINGS, dtype=np.float64)
    ll_all = np.empty((K, N_ORDERINGS), dtype=np.float64)
    for o in range(N_ORDERINGS):
        ll_all[:, o] = exact_likelihood(sigmas, context, o, prior)
        mx = ll_all[:, o].max()
        logZ[o] = mx + math.log(np.mean(np.exp(ll_all[:, o] - mx)))
    mx = logZ.max()
    w_o = np.exp(logZ - mx)
    w_o /= w_o.sum()
    return logZ, w_o, ll_all


def order_predictive(
    sigmas: np.ndarray,
    context: np.ndarray,
    query: np.ndarray,
    ordering: int,
    prior: str,
) -> np.ndarray:
    """Exact order-conditioned 100-bin predictive p(y | x_q, D, o).

    For the uniform-atom posterior: p = mean_k p(y|x_q, sigma_k, o).
    Evaluated on the frozen production quadrature.
    """
    K = len(sigmas)
    r = 2.0 if prior == "N" else float(fleet().R_OF["C"])
    gaussian = prior == "N"
    values, bins, lw = production_quadrature()
    perm = fleet().ORDERINGS[ordering]
    pts = np.empty((len(values), 4))
    pts[:, :3] = query
    pts[:, 3] = values
    xp = pts[:, perm]
    log_num = np.full(len(values), -np.inf, dtype=np.float64)
    chunk = 512
    for st in range(0, K, chunk):
        en = min(K, st + chunk)
        Sk = sigmas[st:en]
        _, U, b = _params_for(Sk, perm)
        resid = np.einsum("kdj,mj->kdm", U, xp)  # (c, 4, V)
        lpdf = _residual_logpdf(resid, b[:, :, None], r, gaussian)
        logj = lpdf.sum(axis=1)  # (c, V)
        m = logj.max(axis=0)
        log_num = np.logaddexp(log_num, m + np.log(np.sum(np.exp(logj - m), axis=0)) - math.log(K))
    weighted = log_num + lw
    shifted = weighted - weighted.max()
    prob = np.bincount(bins.astype(np.int64), weights=np.exp(shifted),
                       minlength=N_BINS).astype(np.float64)
    prob /= prob.sum()
    return prob


def full_and_ablated(
    sigmas: np.ndarray,
    context: np.ndarray,
    query: np.ndarray,
    prior: str,
    w_o: np.ndarray | None = None,
    ll_all: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact full and ordering-ablated 100-bin predictives.

    p_full = sum_o p(o|D) p(y|x_q, D, o)
    p_ablated = (1/24) sum_o p(y|x_q, D, o)
    V = E_post[NLL_ablated - NLL_full] (ordering value).
    """
    if w_o is None or ll_all is None:
        _, w_o, ll_all = exact_evidence_and_posterior(sigmas, context, prior)
    full = np.zeros(N_BINS, dtype=np.float64)
    ablated = np.zeros(N_BINS, dtype=np.float64)
    for o in range(N_ORDERINGS):
        po = order_predictive(sigmas, context, query, o, prior)
        full += w_o[o] * po
        ablated += po / N_ORDERINGS
    return full, ablated, float(np.mean(-np.log(np.maximum(ablated, 1e-300)) + np.log(np.maximum(full, 1e-300))))


def exact_context_evaluation(
    sigmas: np.ndarray,
    context: np.ndarray,
    query: np.ndarray,
    outcome_bin: int,
    prior: str,
) -> dict[str, Any]:
    """Complete exact evaluation for one context under the finite prior.

    Returns everything: evidence, order posterior, predictives, NLLs, ordering
    value, and diagnostic traces.
    """
    K = len(sigmas)
    logZ, w_o, ll_all = exact_evidence_and_posterior(sigmas, context, prior)
    full, ablated, V = full_and_ablated(sigmas, context, query, prior, w_o, ll_all)
    nll_full = -np.log(max(full[outcome_bin], 1e-300))
    nll_ablated = -np.log(max(ablated[outcome_bin], 1e-300))
    # sequential evidence increments: add context rows one at a time
    seq_logZ = np.zeros((len(context), N_ORDERINGS), dtype=np.float64)
    for t in range(1, len(context) + 1):
        sub = context[:t]
        logZ_t, _, _ = exact_evidence_and_posterior(sigmas, sub, prior)
        seq_logZ[t - 1] = logZ_t
    # posterior entropy
    H_o = -np.sum(w_o * np.log(np.maximum(w_o, 1e-300)))
    # per-atom posterior mass distribution
    w_atom_full = np.zeros(K, dtype=np.float64)
    for o in range(N_ORDERINGS):
        ll = ll_all[:, o]
        w_full = np.exp(ll - ll.max())
        w_full /= w_full.sum()
        w_atom_full += w_o[o] * w_full
    return {
        "logZ": logZ,
        "ordering_posterior": w_o,
        "full_probability": full,
        "ablated_probability": ablated,
        "nll_full": float(nll_full),
        "nll_ablated": float(nll_ablated),
        "ordering_value": float(V),
        "posterior_entropy": float(H_o),
        "sequential_logZ": seq_logZ,
        "atom_posterior_eff_n": int(np.sum(w_atom_full > 0.01 / K)),
    }
