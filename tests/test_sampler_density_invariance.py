"""Amendment E.9 — standing sampler <-> density invariance test.

Every generation path must sample from the residual law the exact oracle scores
against. The declared law (Amendment E.1, and already the docstring contract of
``ResidualSpec``) is ELEMENTWISE:

    p_eps(z) = (1 - eps) N(z; 0, (b sqrt2)^2) + eps AL(z; b, r)

applied independently to each coordinate of each row.

E.9.1 ERRATUM (recorded 2026-08-21, under E.9's own "must be strengthened"
clause). Amendment E.9 as written proposed comparing each path's mean
log-likelihood to the quadrature entropy of the declared law. That statistic has
essentially ZERO power against the defect it was written to catch: per-coordinate,
per-row and per-dataset mixing all induce the SAME per-coordinate marginal, and
``residual_logpdf(...).sum(-1)`` depends only on the marginals. Measured at
eps=0.25 with N=4e5, per-coordinate and per-row samples give mean summed
log-density -5.191943 and -5.190646 against a Monte-Carlo SE of 0.0022, i.e.
indistinguishable. The laws differ only in the DEPENDENCE across coordinates.

The test is therefore strengthened to a joint model-selection statistic. For a
block of rows we score three candidate joint densities that share those marginals
and differ only in mixing granularity, and require the declared (per-coordinate)
law to win:

    log p_coord(E)   = sum_ij  log[(1-e) N(E_ij) + e AL(E_ij)]
    log p_row(E)     = sum_i   log[(1-e) prod_j N(E_ij) + e prod_j AL(E_ij)]
    log p_dataset(E) = log[(1-e) prod_ij N(E_ij) + e prod_ij AL(E_ij)]

All three are normalized densities on the block, so the comparison is a proper
scoring rule. The quadrature-entropy check of the original E.9 is RETAINED as a
marginal-calibration and normalization guard, which catches a different bug class.

ACCEPTANCE CRITERION, stated in advance by E.9: this test MUST FAIL on the
pre-fix code at eps in {0.1, 0.25, 0.5} for ``sample_residuals`` and
``_vectorized_observational``, and MUST PASS for every path at every eps after
the E.1 unification.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# numpy renamed trapz -> trapezoid in 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

from pfn_dag_verify import corrected_world as cworld
from pfn_dag_verify.corrected_models import _vectorized_observational
from pfn_dag_verify.corrected_sem import (
    ResidualSpec,
    _al_ac,
    generate_observational,
    params_for,
    residual_logpdf,
    residuals_from_data,
    sample_residuals,
    to_permuted,
)

MIXED_EPS = (0.1, 0.25, 0.5)
DEGENERATE_EPS = (0.0, 1.0)
R = 4.0


# --------------------------------------------------------------------------
# Component log-densities. These MUST reproduce residual_logpdf's mixture
# exactly; test_helpers_match_production is the guard against drift.
# --------------------------------------------------------------------------

def _log_n(e: np.ndarray, b: np.ndarray) -> np.ndarray:
    sc = b * math.sqrt(2.0)
    return -0.5 * (e / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)


def _log_al(e: np.ndarray, b: np.ndarray, r: float = R) -> np.ndarray:
    a, c = _al_ac(b, r)
    shifted = e + (a - c)
    return np.where(shifted >= 0, -shifted / a, shifted / c) - np.log(a + c)


def _lse2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = np.maximum(x, y)
    return m + np.log(np.exp(x - m) + np.exp(y - m))


def _block_scores(E: np.ndarray, b: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """E is (n_blocks, n_rows, d). Returns per-block log-density under each law."""
    lo, hi = math.log1p(-eps), math.log(eps)
    lN, lA = _log_n(E, b), _log_al(E, b)
    return {
        "coord": _lse2(lo + lN, hi + lA).sum(axis=(1, 2)),
        "row": _lse2(lo + lN.sum(axis=2), hi + lA.sum(axis=2)).sum(axis=1),
        "dataset": _lse2(lo + lN.sum(axis=(1, 2)), hi + lA.sum(axis=(1, 2))),
    }


def _assert_declared_law_wins(E, b, eps, path_name, min_z=6.0):
    """The per-coordinate law must beat both rivals by a decisive margin."""
    s = _block_scores(E, b, eps)
    n = len(s["coord"])
    for rival in ("row", "dataset"):
        diff = s["coord"] - s[rival]
        se = diff.std(ddof=1) / math.sqrt(n)
        z = diff.mean() / se if se > 0 else math.inf
        assert z > min_z, (
            f"{path_name} at eps={eps}: samples favour the '{rival}' mixing law "
            f"over the declared per-coordinate law "
            f"(mean log-density difference {diff.mean():+.5f} +/- {se:.5f}, z={z:+.1f}). "
            f"The generation path does not sample from the law residual_logpdf scores."
        )


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("eps", MIXED_EPS)
def test_helpers_match_production(eps):
    """Local component densities must reproduce residual_logpdf exactly."""
    b = np.array([0.8, 1.0, 1.2])
    spec = ResidualSpec(eps=eps, r=R)
    e = np.random.default_rng(7).normal(0.0, 1.5, (64, 3))
    mixed = _lse2(math.log1p(-eps) + _log_n(e, b), math.log(eps) + _log_al(e, b))
    np.testing.assert_allclose(mixed, residual_logpdf(e, b, spec), atol=1e-12)


@pytest.mark.parametrize("eps", MIXED_EPS + DEGENERATE_EPS)
def test_declared_law_normalizes_and_matches_quadrature_entropy(eps):
    """Retained E.9 marginal check: the declared law integrates to 1, and a
    per-coordinate sample's mean log-density matches the quadrature entropy."""
    b = np.array([0.8, 1.0, 1.2])
    spec = ResidualSpec(eps=eps, r=R)
    z = np.linspace(-80.0, 80.0, 400_001)
    neg_h = 0.0
    for j, bj in enumerate(b):
        bj_arr = np.array([bj])
        logp = residual_logpdf(z[:, None], bj_arr, spec)[:, 0]
        p = np.exp(logp)
        mass = _trapz(p, z)
        assert abs(mass - 1.0) < 2e-6, f"coord {j} density integrates to {mass:.8f}"
        neg_h += _trapz(p * logp, z)

    rng = np.random.default_rng(11)
    n = 200_000
    a, c = _al_ac(b, R)
    al = (rng.exponential(a, (n, 3)) - rng.exponential(c, (n, 3)) - (a - c))
    gauss = rng.normal(0.0, math.sqrt(2.0) * b, (n, 3))
    if eps <= 0.0:
        e = gauss
    elif eps >= 1.0:
        e = al
    else:
        e = np.where(rng.random((n, 3)) < eps, al, gauss)
    ll = residual_logpdf(e, b, spec).sum(axis=1)
    se = ll.std(ddof=1) / math.sqrt(n)
    assert abs(ll.mean() - neg_h) < 6.0 * se + 1e-4, (
        f"eps={eps}: mean log-density {ll.mean():.6f} vs quadrature -H {neg_h:.6f}"
    )


@pytest.mark.parametrize("eps", DEGENERATE_EPS)
def test_endpoints_are_granularity_free(eps):
    """At eps=0 and eps=1 there is no mixture, so all three laws coincide."""
    b = np.array([0.8, 1.0, 1.2])
    spec = ResidualSpec(eps=eps, r=R)
    E = sample_residuals(np.random.default_rng(3), b, spec, (2_000, 4))
    lo = math.log1p(-eps) if eps < 1.0 else -math.inf
    hi = math.log(eps) if eps > 0.0 else -math.inf
    lN, lA = _log_n(E, b), _log_al(E, b)
    coord = _lse2(lo + lN, hi + lA).sum(axis=(1, 2))
    row = _lse2(lo + lN.sum(axis=2), hi + lA.sum(axis=2)).sum(axis=1)
    np.testing.assert_allclose(coord, row, atol=1e-9)


# --------------------------------------------------------------------------
# The three production generation paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("eps", MIXED_EPS)
def test_sample_residuals_uses_declared_law(eps):
    b = np.array([0.8, 1.0, 1.2])
    spec = ResidualSpec(eps=eps, r=R)
    E = sample_residuals(np.random.default_rng(101), b, spec, (30_000, 4))
    _assert_declared_law_wins(E, b, eps, "sample_residuals")


@pytest.mark.parametrize("eps", MIXED_EPS)
def test_generate_observational_uses_declared_law(eps):
    world = cworld.make_world(k=1, d=3, eps=eps)
    spec = world.spec
    perm = world.orderings[0]
    sigma = world.sigmas[0]
    sp = params_for(sigma, perm, r=spec.r)
    rng = np.random.default_rng(202)
    n_blocks, n_rows = 30_000, 4
    X = generate_observational(rng, sigma, perm, spec, n_blocks * n_rows)
    E = residuals_from_data(to_permuted(X, sp), sp).reshape(n_blocks, n_rows, 3)
    _assert_declared_law_wins(E, sp.b, eps, "generate_observational")


@pytest.mark.parametrize("eps", MIXED_EPS)
def test_vectorized_observational_uses_declared_law(eps):
    """The training-data path. This is the one commit 3360b99 changed."""
    world = cworld.make_world(k=1, d=3, eps=eps)
    spec = world.spec
    sp = params_for(world.sigmas[0], world.orderings[0], r=spec.r)
    n_blocks, n_rows = 30_000, 4
    rng = np.random.default_rng(303)
    k_idx = np.zeros(n_blocks, dtype=int)
    o_idx = np.zeros(n_blocks, dtype=int)
    X = _vectorized_observational(world, rng, k_idx, o_idx, n_rows)
    E = residuals_from_data(to_permuted(np.asarray(X, dtype=np.float64), sp), sp)
    _assert_declared_law_wins(E, sp.b, eps, "_vectorized_observational")
