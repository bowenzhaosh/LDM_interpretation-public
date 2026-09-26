"""Corrected canonical SEM substrate (Track A0).

ONE source of truth for the SEM transformation, shared by the corrected
oracle, the scalar reference oracle, the identifiability machinery, and the
Track B model data generators. Nothing else re-implements this algebra.

Conventions (documented here, enforced by tests):
- ROW-VECTOR: a data matrix is (n_rows, d); each row is one observation.
- A causal order `perm` is a tuple of variable indices giving the topological
  order. The variable with index perm[0] is the root.
- For covariance atom Sigma and order perm, the permuted covariance is
  Spi = Sigma[perm][:, perm]. Its Cholesky factor L satisfies Spi = L @ L.T.
- L_unit = L / diag(L) (unit lower triangular), U = inv(L_unit) (unit lower
  triangular), b = sqrt(diag(L)^2 / 2) are the residual scale parameters.
- OBSERVATIONAL generation (the forward map, matching the frozen generator):
      x_pi = e @ L_unit.T        e is (n, d), independent residuals
      x[:, perm] = x_pi          (de-permute back to original coordinates)
  Residual recovery is exact:
      e == x_pi @ U.T            x_pi @ U.T = e @ L_unit.T @ U.T
                                 = e @ (U @ L_unit).T = e @ I = e
- This module is the ONLY place that computes Cholesky/L_unit/U/b and the
  residual log-densities. The invariant tests in
  tests/test_corrected_invariants.py pin these down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import numpy as np


def all_orderings(d: int) -> tuple[tuple[int, ...], ...]:
    """All topological orders for a d-dimensional world, root-first."""
    return tuple(permutations(range(d)))


# Frozen prior box (d=4 values from pilot_shared, reused for d=3).
LOG_SD_LO, LOG_SD_HI = math.log(0.6), math.log(1.5)
RHO_LO, RHO_HI = 0.3, 0.8

# Frozen validity bounds.
BETA_MAX = 1.5
B_LO, B_HI = 0.3, 1.3

# Asymmetric-Laplace skew parameter (matches fleet().R_OF["C"] = 4.0).
R_C = 4.0
R_N = 2.0


@dataclass(frozen=True)
class SEMParams:
    """Fully-resolved SEM parameters for one (Sigma, order) latent state.

    All arrays are (d, d) or (d,), row-vector convention.
    """
    d: int
    perm: tuple[int, ...]                  # topological order (root first)
    sigma: np.ndarray                       # (d, d) covariance atom
    L: np.ndarray                           # Cholesky of Sigma[perm][:, perm]
    L_unit: np.ndarray                      # unit lower triangular
    U: np.ndarray                           # inv(L_unit), unit lower triangular
    b: np.ndarray                           # (d,) residual scale params
    r: float                                # AL skew-shape parameter


def params_for(
    sigma: np.ndarray,
    perm: Sequence[int],
    *,
    r: float = R_C,
) -> SEMParams:
    """Canonical SEM resolution for one covariance atom and one order.

    sigma is (d, d) float64 PD; perm is a permutation of range(d).
    This is the ONLY implementation of the Cholesky -> L_unit -> U -> b map
    used by the corrected campaign. (The frozen d=4 generator has its own
    copy; we do not modify it and we do not drift from its algebra.)
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square matrix")
    d = sigma.shape[0]
    perm = tuple(int(p) for p in perm)
    if tuple(sorted(perm)) != tuple(range(d)):
        raise ValueError(f"perm must be a permutation of range({d})")
    Spi = sigma[np.ix_(perm, perm)]
    L = np.linalg.cholesky(Spi)
    diag = np.diagonal(L)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("Cholesky diagonal must be positive and finite")
    # Column-wise normalization: L_unit[j, i] = L[j, i] / diag[i], so that
    # L_unit @ diag(diag(L)) == L (matching the frozen generator's batched
    # `L / diag[:, None, :]`, which aligns with the trailing/column axis).
    L_unit = L / diag[None, :]
    U = np.linalg.inv(L_unit)
    b = np.sqrt(np.maximum(diag ** 2, 1e-12) / 2.0)
    return SEMParams(d=d, perm=perm, sigma=sigma, L=L, L_unit=L_unit, U=U,
                     b=b, r=float(r))


def residuals_from_data(x_pi: np.ndarray, sp: SEMParams) -> np.ndarray:
    """Recover residuals e = x_pi @ U.T from permuted-coordinate data."""
    return np.asarray(x_pi, dtype=np.float64) @ sp.U.T


def forward_map(e: np.ndarray, sp: SEMParams) -> np.ndarray:
    """Forward map: x_pi = e @ L_unit.T. e is (n, d) row-vector residuals."""
    return np.asarray(e, dtype=np.float64) @ sp.L_unit.T


def to_permuted(x: np.ndarray, sp: SEMParams) -> np.ndarray:
    """Reorder original-coordinate data into permuted (causal) coordinates."""
    return np.asarray(x, dtype=np.float64)[:, list(sp.perm)]


def from_permuted(x_pi: np.ndarray, sp: SEMParams) -> np.ndarray:
    """Reverse of to_permuted: de-permute causal-coordinate data back."""
    x = np.empty_like(x_pi)
    x[:, list(sp.perm)] = x_pi
    return x


# ---------------------------------------------------------------------------
# Residual distributions (Gaussian / AsymmetricLaplace / epsilon mixture).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualSpec:
    """The residual law used by a world: p_e = (1-eps) N + eps AL.

    eps = 0.0 -> pure Gaussian (observationally order-unidentifiable);
    eps = 1.0 -> pure asymmetric Laplace (non-Gaussian structure).
    r is the AL shape parameter (r=4.0 matches the frozen C prior).
    """
    eps: float = 0.0
    r: float = R_C

    def __post_init__(self) -> None:
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError("eps must be in [0, 1]")
        if self.r <= 0 or not math.isfinite(self.r):
            raise ValueError("r must be positive and finite")


def _al_ac(b: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def residual_logpdf(
    e: np.ndarray,
    b: np.ndarray,
    spec: ResidualSpec,
) -> np.ndarray:
    """Log density of the residual mixture, broadcast over trailing dims.

    e and b broadcast: for a (n, d) e and (d,) b, returns (n, d).
    Gaussian component: N(0, (b*sqrt(2))^2), matching the frozen generator.
    AL component: the frozen AL with density
      p(z) = exp(-z/a)/(a+c) for z>=0, exp(z/c)/(a+c) for z<0,
      where z = e + (a - c), a = r*c, c = sqrt(2 b^2 / (1 + r^2)).
    """
    e = np.asarray(e, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sc = b * math.sqrt(2.0)
    log_n = -0.5 * (e / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)
    eps = spec.eps
    if eps <= 0.0:
        return log_n
    c = np.sqrt(2.0 * b * b / (1.0 + spec.r * spec.r))
    a = spec.r * c
    shifted = e + (a - c)
    log_al = np.where(shifted >= 0, -shifted / a, shifted / c) - np.log(a + c)
    if eps >= 1.0:
        return log_al
    l1 = np.log1p(-eps) + log_n
    l2 = np.log(eps) + log_al
    m = np.maximum(l1, l2)
    return m + np.log(np.exp(l1 - m) + np.exp(l2 - m))


def residual_logdensity_row(
    x: np.ndarray,
    sp: SEMParams,
    spec: ResidualSpec,
) -> np.ndarray:
    """Log joint density p(x) for one (or several) observational rows.

    x is (n, d) in ORIGINAL coordinates. e = x_pi @ U.T, and since
    det(U) = det(L_unit) = 1 (unit triangular), p(x) = prod_j p_e(e_j).
    Returns (n,) log densities.
    """
    e = residuals_from_data(to_permuted(x, sp), sp)
    return residual_logpdf(e, sp.b, spec).sum(axis=-1)


# ---------------------------------------------------------------------------
# Observational conditional density (exact, 1-D in target).
# ---------------------------------------------------------------------------

def conditional_joint_logdensity(
    y: np.ndarray,
    x_q: np.ndarray,
    target: int,
    sp: SEMParams,
    spec: ResidualSpec,
) -> np.ndarray:
    """Unnormalized log density log p(x_q, y) at query x_q (all-but-target).

    x_q is (d-1,): the observed covariates (all variables except `target`).
    y is a scalar or 1-D array of target values. The SEM is linear and
    invertible, so e = (x_q, y) @ U.T is affine in y; p(x_q, y) = prod_j
    p_e(e_j). Returns log p(x_q, y) for each y (unnormalized over y; the
    caller normalizes by integrating over y).
    """
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    d = sp.d
    n = y.shape[0]
    x = np.empty((n, d), dtype=np.float64)
    cols = [i for i in range(d) if i != target]
    x[:, cols] = x_q[None, :]
    x[:, target] = y
    return residual_logdensity_row(x, sp, spec)


# ---------------------------------------------------------------------------
# Interventional conditional density (exact, low-D integral over free residuals).
# ---------------------------------------------------------------------------

def interventional_linear_forms(
    sp: SEMParams,
    intervened: int,
    target: int,
    value: float,
) -> tuple[int, np.ndarray, float]:
    """Linear form of the target under do(x_intervened = value).

    Returns (m, a, const) such that, under the intervention,
        x_pi[s_target] = sum_j a[j] e[j] + const,
    where e[j] are the residuals of the m causal positions j that are strictly
    before the target and not the intervened node (their residuals remain free
    and influence the target), listed in ascending order. For d=3, m <= 2.
    """
    d = sp.d
    # Structural coefficients of the SEM are beta = -U (the frozen generator's
    # own convention, `beta = -U` in validity_keep): x_pi[j] = sum_{i<j}
    # (-U[j,i]) x_pi[i] + e[j]. The diagonal -1 never enters the i<j sum.
    L = -sp.U
    s_target = sp.perm.index(target)
    s_int = sp.perm.index(intervened)
    if s_target == s_int:
        return 0, np.zeros(0, dtype=np.float64), float(value)

    # Free residuals influencing y: every position j <= s_target except the
    # intervened node. This includes the target's OWN residual (e[s_target]),
    # whose basis coefficient is 1.
    before_target = [j for j in range(d) if j != s_int and j <= s_target]
    m = len(before_target)
    # Forward-substitute on a symbolic basis to express every x_pi[j] as
    # aff(e[before_target], const), where the intervened node is pinned.
    # C[j, :] : coefficients on the m free residuals; C[j, m] : const.
    C = np.zeros((d, m + 1), dtype=np.float64)
    for j in range(d):
        if j == s_int:
            C[j, :] = 0.0
            C[j, m] = float(value)
            continue
        acc = np.zeros(m + 1, dtype=np.float64)
        for i in range(j):
            if i == s_int:
                acc += L[j, i] * C[s_int, :]
            else:
                acc += L[j, i] * C[i, :]
        if j in before_target:
            bi = before_target.index(j)
            acc[bi] += 1.0
        C[j, :] = acc
    a = C[s_target, :m].copy()
    const = float(C[s_target, m])
    return m, a, const


def _residual_quad_nodes(b: float, spec: ResidualSpec, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights for integrating over one residual law.

    The residual law has variance ~ 2 b^2 (Gaussian part) with exponential
    tails (AL part). We integrate on [-H, H] with H = 12 * b * sqrt(2), which
    covers > 12 sigma of the Gaussian and the exponential tails to ~1e-5.
    """
    H = 12.0 * b * math.sqrt(2.0)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Gauss-Legendre on [-1,1]: int g(x) dx ~ sum w g(x), sum(w)=2. Map to
    # [-H,H] via u=H*x, du=H dx, so the weights scale by H (NOT H/2).
    return nodes * H, weights * H


def interventional_density_grid(
    y: np.ndarray,
    target: int,
    intervened: int,
    value: float,
    sp: SEMParams,
    spec: ResidualSpec,
    n_quad: int = 200,
) -> np.ndarray:
    """Exact density p(y | do(x_intervened = value)) on the y grid.

    For target == intervened, a point mass at `value`. For a descendant or
    ancestor target, y = a^T e_free + const with the free residuals
    independent, so
        p_Y(y) = (1/|a_1|) * int prod_{j>=2} p_{e_j}(e_j) *
                 p_{e_1}((y - const - sum_{j>=2} a_j e_j) / a_1) de_2..de_m
    integrated by Gauss-Legendre quadrature (m-1 dimensions; for d=3 this is
    at most 1-D, for d=4 at most 2-D). Exact for Gaussian residuals.
    """
    m, a, const = interventional_linear_forms(sp, intervened, target, value)
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if m == 0:
        # Only reached when target == intervened: point mass at `const`. As a
        # density on the y grid this is zero except at the point; the caller
        # (int_query_operator) handles the degenerate case separately.
        return np.zeros_like(y, dtype=np.float64)
    # Free residual positions (causal order) with nonzero basis coefficients.
    before_target = [j for j in range(sp.d) if j != sp.perm.index(intervened)
                     and j <= sp.perm.index(target)]
    b_free = sp.b[before_target]
    if m == 1:
        a1 = a[0]
        e1 = (y - const) / a1
        lp = residual_logpdf(e1, b_free[0], spec) - np.log(abs(a1))
        mx = lp.max()
        return np.exp(lp - mx)  # normalization handled by caller quadrature

    # m >= 2: integrate over the residuals except the pivot. Choose the pivot
    # as the free residual with the largest |a_j| (dividing by a near-zero
    # coefficient would be numerically unstable).
    pivot = int(np.argmax(np.abs(a)))
    rest = [j for j in range(m) if j != pivot]
    a1 = a[pivot]
    if abs(a1) < 1e-12:
        raise FloatingPointError("degenerate interventional coefficient")
    a_rest = a[rest]
    b0 = b_free[pivot]
    b_rest = b_free[rest]
    grids = [_residual_quad_nodes(b_rest[j], spec, n_quad) for j in range(m - 1)]
    # tensor grid over the non-pivot residuals
    mesh = np.meshgrid(*[g[0] for g in grids], indexing="ij")
    weights = np.meshgrid(*[g[1] for g in grids], indexing="ij")
    e_rest = np.stack([mm.ravel() for mm in mesh], axis=1)   # (N, m-1)
    w_rest = np.stack([ww.ravel() for ww in weights], axis=1)
    prodw = np.prod(w_rest, axis=1)                           # (N,)
    lp_rest = residual_logpdf(e_rest, b_rest, spec).sum(axis=1)  # (N,)
    out = np.empty(y.shape, dtype=np.float64)
    for yi, yy in enumerate(y):
        num = yy - const - e_rest @ a_rest
        e1 = num / a1
        lp1 = residual_logpdf(e1, b0, spec)
        lpy = lp1 + lp_rest + np.log(prodw) - np.log(abs(a1))
        mx = lpy.max()
        # Return a DENSITY (consistent with the m==1 branch), not a log value.
        out[yi] = np.exp(mx) * np.sum(np.exp(lpy - mx))
    return out


def interventional_density_coarse(
    target: int,
    intervened: int,
    value: float,
    sp: SEMParams,
    spec: ResidualSpec,
    n_quad: int = 200,
    n_y: int = 601,
    y_lo: float = -12.0,
    y_hi: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast m>=2 interventional density on a coarse y-grid, for the panel
    operator used in the identifiability analysis.

    The exact density p(y | do(...)) is smooth; for the identifiability LP
    (tolerance ~1e-3) a coarse-grid evaluation interpolated onto the production
    quadrature nodes is sufficient and ~50x faster than per-node integration.
    Returns (y_grid, density).
    """
    y_grid = np.linspace(y_lo, y_hi, n_y)
    return y_grid, interventional_density_grid(y_grid, target, intervened, value,
                                               sp, spec, n_quad=n_quad)


# ---------------------------------------------------------------------------
# Sampling helpers (used only by data generation / Track B, never by the
# exact oracle).
# ---------------------------------------------------------------------------

def sample_residuals(
    rng: np.random.Generator,
    b: np.ndarray,
    spec: ResidualSpec,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Sample residual matrix of shape shape + (d,) from the mixture law."""
    b = np.asarray(b, dtype=np.float64)
    d = b.shape[0]
    full_shape = tuple(shape) + (d,)
    eps = spec.eps
    if eps <= 0.0:
        return rng.normal(0.0, math.sqrt(2.0) * b[None, :], full_shape)
    a, c = _al_ac(b, spec.r)
    al = (rng.exponential(a[None, :], full_shape)
          - rng.exponential(c[None, :], full_shape) - (a - c)[None, :])
    if eps >= 1.0:
        return al
    gauss = rng.normal(0.0, math.sqrt(2.0) * b[None, :], full_shape)
    # Amendment E.1: the declared law is ELEMENTWISE -- one independent component
    # draw per COORDINATE, matching residual_logpdf and this module's own
    # ResidualSpec docstring. A per-row mask (rng.random(shape + (1,))) shares the
    # component across the d coordinates of a row and samples a different law.
    mask = rng.random(full_shape) < eps
    return np.where(mask, al, gauss)


def generate_observational(
    rng: np.random.Generator,
    sigma: np.ndarray,
    perm: Sequence[int],
    spec: ResidualSpec,
    n: int,
) -> np.ndarray:
    """Generate n observational rows from (sigma, perm) via the forward map.

    Returns (n, d) in ORIGINAL coordinates. Equivalent to the frozen
    generator's gen_data with the forward map x_pi = e @ L_unit.T.
    """
    sp = params_for(sigma, perm, r=spec.r)
    e = sample_residuals(rng, sp.b, spec, (n,))
    x_pi = forward_map(e, sp)
    return from_permuted(x_pi, sp)


def generate_interventional(
    rng: np.random.Generator,
    sigma: np.ndarray,
    perm: Sequence[int],
    spec: ResidualSpec,
    n: int,
    intervened: int,
    value: float,
) -> np.ndarray:
    """Forward-simulate n rows under do(x_intervened = value).

    Structural equations are enforced directly: every variable except the
    intervened node is computed as its parent-sum plus its independent
    residual. The intervened node is pinned to `value` (its own residual is
    never sampled). Returns (n, d) in original coordinates.
    """
    sp = params_for(sigma, perm, r=spec.r)
    d = sp.d
    s_int = sp.perm.index(intervened)
    # Structural coefficients beta = -U (see interventional_linear_forms).
    L = -sp.U
    # Sample residuals for all positions except the intervened node.
    e_all = sample_residuals(rng, sp.b, spec, (n,))
    x_pi = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        if j == s_int:
            x_pi[:, j] = float(value)
            continue
        acc = np.zeros(n, dtype=np.float64)
        for i in range(j):
            if i == s_int:
                acc += L[j, i] * float(value)
            else:
                acc += L[j, i] * x_pi[:, i]
        x_pi[:, j] = acc + e_all[:, j]
    return from_permuted(x_pi, sp)
