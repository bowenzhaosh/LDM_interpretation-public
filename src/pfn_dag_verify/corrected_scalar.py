"""Corrected scalar brute-force reference oracle (Track A3).

A deliberately simple, slow, obviously-correct implementation of the same
quantities the vectorized oracle computes, used to generate golden fixtures
and cross-check the vectorized implementation on tiny worlds.

Independence strategy: the scalar oracle shares ONLY the SEM parameter
resolution (`corrected_sem.params_for`, the single source of truth mandated
by A0). It re-implements, with different machinery, everything that could
harbor an indexing/orientation bug:
- posterior enumeration: explicit triple loop over k, o, rows;
- predictive integrals: a dense uniform y-grid with the trapezoid rule
  (NOT the production quadrature, NOT bin-counting);
- observational conditioning: the joint density is evaluated on the dense
  grid and normalized by a grid sum (joint numerator/denominator);
- interventional predictives: a dense-grid marginalization over the free
  residuals (independent of `interventional_density_grid`).

The vectorized oracle must agree with this within strict tolerances
(enforced in tests/test_corrected_invariants.py).
"""

from __future__ import annotations

import math

import numpy as np

from .corrected_sem import (
    ResidualSpec,
    params_for,
    residual_logpdf,
)

# numpy 1.26 name is np.trapz; numpy 2.x renamed to np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# Dense-grid defaults. Wide enough that the tails of the residual laws are
# covered; the AL component is exponentially bounded, 12*scale is safe.
Y_LO, Y_HI = -12.0, 12.0
N_GRID = 60001          # dense uniform grid over y
N_GRID_RESID = 24001    # dense grid per free residual (interventional)


def _dense_y_grid() -> np.ndarray:
    return np.linspace(Y_LO, Y_HI, N_GRID)


def scalar_posterior(
    sigmas: np.ndarray,
    orderings: tuple[tuple[int, ...], ...],
    spec: ResidualSpec,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact p(k, o | D) and p(o | D), explicit loops, float64.

    Returns (w_lo (K*O,), w_o (O,)).
    """
    K = sigmas.shape[0]
    O = len(orderings)
    context = np.asarray(context, dtype=np.float64)
    n = context.shape[0]
    d = context.shape[1]
    logw = np.empty(K * O, dtype=np.float64)
    for k in range(K):
        for o in range(O):
            sp = params_for(sigmas[k], orderings[o], r=spec.r)
            lpd = 0.0
            for row in range(n):
                # e = x_pi @ U.T, one row at a time.
                x = context[row]
                x_pi = np.array([x[j] for j in sp.perm], dtype=np.float64)
                e = np.zeros(d, dtype=np.float64)
                for a in range(d):
                    s = 0.0
                    for b in range(d):
                        s += x_pi[b] * sp.U[a, b]
                    e[a] = s
                for a in range(d):
                    lpd += residual_logpdf(e[a], sp.b[a], spec)
            logw[k * O + o] = lpd
    mx = logw.max()
    w = np.exp(logw - mx)
    w = w / w.sum()
    w_o = w.reshape(K, O).sum(axis=0)
    return w, w_o


def _joint_density_on_grid(
    sigmas: np.ndarray,
    orderings: tuple[tuple[int, ...], ...],
    spec: ResidualSpec,
    x_q: np.ndarray,
    target: int,
    y_grid: np.ndarray,
) -> np.ndarray:
    """p(x_q, y) for every (k,o) and every y on the grid.

    Returns (K*O, len(y_grid)) unnormalized joint densities.
    """
    K = sigmas.shape[0]
    O = len(orderings)
    d = len(x_q) + 1
    cols = [i for i in range(d) if i != target]
    out = np.empty((K * O, y_grid.shape[0]), dtype=np.float64)
    for lo in range(K * O):
        k, o = divmod(lo, O)
        sp = params_for(sigmas[k], orderings[o], r=spec.r)
        for gi, y in enumerate(y_grid):
            x = np.empty(d, dtype=np.float64)
            x[cols] = x_q
            x[target] = y
            x_pi = np.array([x[j] for j in sp.perm], dtype=np.float64)
            e = np.array([sum(x_pi[b] * sp.U[a, b] for b in range(d))
                          for a in range(d)], dtype=np.float64)
            out[lo, gi] = sum(residual_logpdf(e[a], sp.b[a], spec) for a in range(d))
    return out


def scalar_obs_predictive(
    sigmas: np.ndarray,
    orderings: tuple[tuple[int, ...], ...],
    spec: ResidualSpec,
    x_q: np.ndarray,
    target: int,
    w_lo: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """p(y in bin | x_q, D) under posterior weights w_lo, dense-grid exact.

    bins is the (B+1,) bin-edge array. The joint numerator/denominator is
    integrated by the trapezoid rule on the dense y grid.
    """
    K = sigmas.shape[0]
    O = len(orderings)
    y_grid = _dense_y_grid()
    joint = _joint_density_on_grid(sigmas, orderings, spec, x_q, target, y_grid)
    # numerator: sum over (k,o) w * p(x_q, y). The per-state log-densities are
    # ABSOLUTE (comparable across states), so a single global scale is used to
    # avoid overflow; the scale cancels in the ratio num/den.
    gmax = joint.max()
    num = np.zeros(y_grid.shape[0], dtype=np.float64)
    for lo in range(K * O):
        num += w_lo[lo] * np.exp(joint[lo] - gmax)
    # normalize: divide by grid integral of the posterior-weighted joint.
    den = _trapz(num, y_grid)
    if den <= 0:
        raise FloatingPointError("scalar denominator nonpositive")
    # bin probabilities by trapezoid within each bin.
    prob = np.zeros(len(bins) - 1, dtype=np.float64)
    for b in range(len(bins) - 1):
        lo_e, hi_e = bins[b], bins[b + 1]
        mask = (y_grid >= lo_e) & (y_grid <= hi_e)
        if mask.any():
            prob[b] = _trapz(num[mask], y_grid[mask]) / den
    s = prob.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("scalar predictive did not normalize")
    return prob / s


def scalar_int_predictive(
    sigmas: np.ndarray,
    orderings: tuple[tuple[int, ...], ...],
    spec: ResidualSpec,
    intervened: int,
    value: float,
    target: int,
    w_lo: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """p(y in bin | do(x_intervened = value), D), dense-grid marginalization.

    For each latent state, express the target as an affine function of the
    free residuals and marginalize on a dense grid per residual. Deliberately
    independent of corrected_sem.interventional_density_grid.
    """
    K = sigmas.shape[0]
    O = len(orderings)
    y_grid = _dense_y_grid()
    prob = np.zeros(len(bins) - 1, dtype=np.float64)
    for lo in range(K * O):
        k, o = divmod(lo, O)
        sp = params_for(sigmas[k], orderings[o], r=spec.r)
        d = sp.d
        s_target = sp.perm.index(target)
        s_int = sp.perm.index(intervened)
        if s_target == s_int:
            # point mass at value
            for b in range(len(bins) - 1):
                if bins[b] <= value <= bins[b + 1]:
                    prob[b] += w_lo[lo]
            continue
        # free residuals up to and including the target, excluding the
        # intervened node (the target's own residual is always free)
        before = [j for j in range(d) if j != s_int and j <= s_target]
        m = len(before)
        # symbolic forward substitution (independent re-derivation)
        # x_pi[j] = sum_{i<j} beta[j,i] x_pi[i] + e[j] with beta = -U (the
        # SEM structural coefficients); x_pi[s_int] = value
        L = -sp.U
        C = np.zeros((d, m + 1), dtype=np.float64)
        for j in range(d):
            if j == s_int:
                C[j, m] = value
                continue
            acc = np.zeros(m + 1, dtype=np.float64)
            for i in range(j):
                if i == s_int:
                    acc += L[j, i] * C[s_int, :]
                else:
                    acc += L[j, i] * C[i, :]
            if j in before:
                acc[before.index(j)] += 1.0
            C[j, :] = acc
        a = C[s_target, :m]
        const = C[s_target, m]
        # marginalize over the free residuals on a dense grid
        if m == 0:
            y_val = const
            for b in range(len(bins) - 1):
                if bins[b] <= y_val <= bins[b + 1]:
                    prob[b] += w_lo[lo]
            continue
        # build dense grids for residuals 2..m (or integrate analytically for m==1)
        b_free = sp.b[before]
        if m == 1:
            a1 = a[0]
            dens = np.empty(y_grid.shape[0], dtype=np.float64)
            for gi, y in enumerate(y_grid):
                e1 = (y - const) / a1
                dens[gi] = math.exp(residual_logpdf(e1, b_free[0], spec) - math.log(abs(a1)))
            for b in range(len(bins) - 1):
                lo_e, hi_e = bins[b], bins[b + 1]
                mask = (y_grid >= lo_e) & (y_grid <= hi_e)
                if mask.any():
                    prob[b] += w_lo[lo] * _trapz(dens[mask], y_grid[mask])
            continue
        # m >= 2: vectorized trapezoid marginalization over the free residuals
        # (m-1)-D (for d=3, m <= 2 so this is 1-D). y = a1 e1 + a_rest@e_rest
        # + const, so
        #   p_Y(y) = (1/|a1|) * int prod_{j>=2} p_{e_j}(e_j)
        #                    p_{e_1}((y - const - a_rest@e_rest)/a1) de_2..de_m
        # evaluated on a dense grid per residual and integrated by the
        # trapezoid rule (independent of the production quadrature).
        a1 = a[0]
        a_rest = a[1:]
        b0 = b_free[0]
        # build a dense grid for residuals 2..m (1-D for d=3)
        rest_grid = np.linspace(Y_LO, Y_HI, 6001)
        p_rest = np.exp(residual_logpdf(rest_grid, b_free[1], spec))
        # (len(rest_grid),) ; for m>2 this becomes a tensor product
        # integrand over e_rest, broadcasting over y:
        #   p_e1((y - const - a_rest@e_rest)/a1)   (N_y, N_rest)
        e1_mat = (y_grid[:, None] - const - np.outer(a_rest, rest_grid)) / a1
        integ = np.exp(residual_logpdf(e1_mat, b0, spec)) * p_rest[None, :]
        dens = _trapz(integ, rest_grid, axis=1) / abs(a1)
        for b in range(len(bins) - 1):
            lo_e, hi_e = bins[b], bins[b + 1]
            mask = (y_grid >= lo_e) & (y_grid <= hi_e)
            if mask.any():
                prob[b] += w_lo[lo] * _trapz(dens[mask], y_grid[mask])
    s = prob.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("scalar interventional predictive did not normalize")
    return prob / s
