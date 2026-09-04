"""Track A4 — mandatory invariant test suite for the corrected campaign.

These tests are the GATE: no corrected-campaign experiment (Track A2, B1, B2,
B3, tomography) may be launched until this suite passes in full. They pin
down the corrected SEM algebra, the exact finite-prior posterior, the ablations
and ordering value, proper-scoring properties, numerical agreement with the
independent scalar oracle, float64 golden fixtures, and provenance
completeness. Tolerances are fixed a priori (no threshold tuning after
results).

Run with:  PYTHONPATH=src python3 -m pytest tests/test_corrected_invariants.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pfn_dag_verify.corrected_scalar as scalar
import pfn_dag_verify.corrected_sem as csem
import pfn_dag_verify.corrected_world as cworld
from pfn_dag_verify.corrected_oracle import (
    World,
    ablated_weights,
    exact_joint_posterior,
    exact_order_value,
    int_query_operator,
    kl_divergence,
    obs_query_operator,
)
from pfn_dag_verify.corrected_sem import (
    ResidualSpec,
    forward_map,
    generate_interventional,
    generate_observational,
    params_for,
    residuals_from_data,
)
from pfn_dag_verify.pilot_shared import N_BINS, production_quadrature

BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_world(k: int = 4, d: int = 3, eps: float = 0.0, seed: int | None = None) -> World:
    if seed is None:
        seed = cworld.WORLD_SEED_ROOT + 500_000 + 1000 * d + k + int(eps * 100)
    return cworld.make_world(k=k, d=d, eps=eps, seed=seed)


def _context(n_rows: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n_rows, 3))


def _gaussian_world() -> World:
    return _tiny_world(k=4, d=3, eps=0.0)


# ---------------------------------------------------------------------------
# 1. SEM algebra
# ---------------------------------------------------------------------------

def test_u_l_unit_identity():
    world = _gaussian_world()
    for k in range(world.K):
        for o in range(world.O):
            sp = world.sem_params(k, o)
            assert np.allclose(sp.U @ sp.L_unit, np.eye(3), atol=1e-12), (k, o)


def test_residual_recovery():
    world = _gaussian_world()
    rng = np.random.default_rng(123)
    for k in range(world.K):
        for o in range(world.O):
            sp = world.sem_params(k, o)
            e = rng.normal(0.0, 1.0, (20, 3))
            x_pi = forward_map(e, sp)
            e2 = residuals_from_data(x_pi, sp)
            assert np.allclose(e2, e, atol=1e-12), (k, o)


def test_generated_covariance_matches():
    world = _gaussian_world()
    rng = np.random.default_rng(456)
    for k in range(world.K):
        for o in range(world.O):
            sp = world.sem_params(k, o)
            x = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                       ResidualSpec(eps=0.0), 200_000)
            cov = np.cov(x, rowvar=False)
            target = world.sigmas[k]
            assert np.allclose(cov, target, atol=0.05), (k, o)


def test_intervention_obeys_structural_equations():
    """do(x_j = c) forward sim must pin x_j and produce structural residuals
    (via forward substitution, the correct definition under a pinned value)
    that are independent draws from the residual law.

    The structural residual of variable j (j != s_int) is
        e_j = x_pi[:, j] - sum_{i<j} L[j, i] * x_pi[:, i]
    (using the actual, possibly-pinned, values of earlier variables). Under the
    generative model these are independent with the residual law: for eps=0,
    N(0, 2 b[j]^2). Check moments on a large sample."""
    world = _gaussian_world()
    rng = np.random.default_rng(789)
    n = 150_000
    for k in range(world.K):
        for o in range(world.O):
            sp = world.sem_params(k, o)
            for iv in range(3):
                value = 1.7
                x = generate_interventional(rng, world.sigmas[k], world.orderings[o],
                                            ResidualSpec(eps=0.0), n, iv, value)
                x_pi = x[:, list(sp.perm)]
                s_int = sp.perm.index(iv)
                assert np.allclose(x_pi[:, s_int], value, atol=1e-12), (k, o, iv)
                # forward-substitution residuals (structural coefficients
                # beta = -U, matching the frozen generator's validity rule)
                beta = -sp.U
                e = np.zeros((n, 3), dtype=np.float64)
                for j in range(3):
                    parent = np.zeros(n)
                    for i in range(j):
                        parent += beta[j, i] * x_pi[:, i]
                    e[:, j] = x_pi[:, j] - parent
                for j in range(3):
                    if j == s_int:
                        continue
                    var_target = 2.0 * sp.b[j] ** 2
                    var_est = np.var(e[:, j])
                    assert abs(var_est - var_target) / var_target < 0.03, (k, o, iv, j)
                # residuals of non-intervened variables are mutually independent
                others = [j for j in range(3) if j != s_int]
                if len(others) == 2:
                    corr = np.corrcoef(e[:, others[0]], e[:, others[1]])[0, 1]
                    assert abs(corr) < 0.05, (k, o, iv)


def test_interventional_gaussian_analytic():
    """Gaussian world, SINGLE order (0,1,2): interventional predictive of a
    descendant y under do(x_root = c) is exactly N(mu, sigma^2). The SEM
    structural coefficients are beta = -U (the frozen generator's `beta = -U`
    convention), so
        mu    = (beta[2,0] + beta[2,1]*beta[1,0]) * c = L_unit[2,0] * c
        sigma^2 = beta[2,1]^2 * 2 b[1]^2 + 2 b[2]^2 = U[2,1]^2 * 2 b[1]^2 + 2 b[2]^2
    (The interventional distribution is ORDER-DEPENDENT even under Gaussianity,
    so the test must fix one order rather than average over the full prior.)"""
    base = _tiny_world(k=1, d=3, eps=0.0)
    world = World(sigmas=base.sigmas, spec=base.spec, orderings=((0, 1, 2),))
    sp = world.sem_params(0, 0)
    iv, target, value = 0, 2, 1.7
    s_t = sp.perm.index(target)
    L = sp.L_unit
    U = sp.U
    mu = L[s_t, 0] * value                       # = (beta20 + beta21*beta10)*c
    var = U[s_t, 1] ** 2 * (2.0 * sp.b[1] ** 2) + 2.0 * sp.b[2] ** 2
    s = math.sqrt(var)
    edges = BIN_EDGES
    analytic = np.array([0.5 * (math.erf((edges[b + 1] - mu) / (s * math.sqrt(2)))
                                - math.erf((edges[b] - mu) / (s * math.sqrt(2))))
                         for b in range(N_BINS)])
    post = exact_joint_posterior(world, _context(n_rows=0))
    iq = int_query_operator(world, iv, value, target)
    got = iq.predictive(post["w_lo"])
    assert np.allclose(got, analytic, atol=5e-3), "interventional Gaussian mismatch"


def test_interventional_order_dependence_gaussian():
    """Even for eps=0, the interventional predictive differs across causal
    orders (do-calculus: the parent sets differ), while the observational
    predictive is order-invariant. This is the mathematical basis for Q3
    identifiability under Gaussianity."""
    world = _tiny_world(k=1, d=3, eps=0.0)
    ctx = np.zeros((0, 3))
    post = exact_joint_posterior(world, ctx)
    iv, target, value = 0, 2, 1.7
    iq = int_query_operator(world, iv, value, target)
    preds = [iq.op[o] for o in range(world.O)]
    # some pair differs
    maxdiff = max(np.max(np.abs(preds[a] - preds[b]))
                  for a in range(world.O) for b in range(a + 1, world.O))
    assert maxdiff > 1e-3, "interventional predictives unexpectedly identical across orders"
    # observational predictives ARE identical across orders for eps=0
    xq = np.array([0.2, -0.3])
    qop = obs_query_operator(world, xq, target=2)
    obs_preds = [obs_query_operator(world, xq, target=2).predictive(
        np.eye(world.K * world.O)[o]) for o in range(world.O)]
    # single-atom: per-order observational predictive with one-hot weight
    obs_preds = []
    for o in range(world.O):
        w = np.zeros(world.K * world.O)
        w[o] = 1.0
        obs_preds.append(qop.predictive(w))
    maxdiff_obs = max(np.max(np.abs(obs_preds[a] - obs_preds[b]))
                      for a in range(world.O) for b in range(a + 1, world.O))
    assert maxdiff_obs < 1e-9, "observational predictives differ across orders for eps=0"


# ---------------------------------------------------------------------------
# 2. Exact Bayes identities
# ---------------------------------------------------------------------------

def test_k1_posterior():
    world = _tiny_world(k=1, d=3, eps=0.0)
    post = exact_joint_posterior(world, _context())
    assert np.allclose(post["w_o"], np.full(world.O, 1.0 / world.O), atol=1e-12)


def test_single_order():
    world = World(sigmas=_tiny_world(k=2).sigmas,
                  spec=ResidualSpec(eps=0.0),
                  orderings=((0, 1, 2),))
    post = exact_joint_posterior(world, _context())
    assert post["w_o"].shape == (1,)
    assert np.allclose(post["w_o"], [1.0], atol=1e-12)


def test_empty_context_equals_prior():
    world = _gaussian_world()
    ctx = np.zeros((0, 3))
    post = exact_joint_posterior(world, ctx)
    assert np.allclose(post["w_o"], np.full(world.O, 1.0 / world.O), atol=1e-12)
    xq = np.array([0.3, -0.5])
    q_full = obs_query_operator(world, xq, target=2)
    p_empty = q_full.predictive(post["w_lo"])
    p_prior = q_full.predictive(ablated_weights(world, post, "prior"))
    assert np.allclose(p_empty, p_prior, atol=1e-12)


def test_duplicate_atoms_no_change():
    """Replacing atom B by two identical copies with B's prior mass split
    equally must not change the exact posterior predictive. (A uniform prior
    over a duplicate list would DOUBLE B's prior mass — that is a different
    prior, so the comparison must preserve prior mass.)"""
    base = _tiny_world(k=2, d=3, eps=0.0)
    A, B = base.sigmas[0], base.sigmas[1]
    sig1 = np.stack([A, B], axis=0)
    prior1 = np.array([0.5, 0.5])
    sig2 = np.stack([A, B, B], axis=0)
    prior2 = np.array([0.5, 0.25, 0.25])
    w1 = World(sigmas=sig1, spec=ResidualSpec(eps=0.0), orderings=base.orderings,
               prior_atom=prior1)
    w2 = World(sigmas=sig2, spec=ResidualSpec(eps=0.0), orderings=base.orderings,
               prior_atom=prior2)
    ctx = _context(n_rows=4, seed=11)
    xq = np.array([0.1, 0.2])
    p1 = obs_query_operator(w1, xq, target=2).predictive(
        exact_joint_posterior(w1, ctx)["w_lo"])
    p2 = obs_query_operator(w2, xq, target=2).predictive(
        exact_joint_posterior(w2, ctx)["w_lo"])
    assert np.allclose(p1, p2, atol=1e-9)


def test_atom_relabeling_invariance():
    world = _gaussian_world()
    perm_atoms = [2, 0, 3, 1]
    sig2 = world.sigmas[perm_atoms]
    w2 = World(sigmas=sig2, spec=world.spec, orderings=world.orderings)
    ctx = _context(n_rows=4, seed=13)
    xq = np.array([-0.4, 0.9])
    p1 = obs_query_operator(world, xq, target=2).predictive(
        exact_joint_posterior(world, ctx)["w_lo"])
    p2 = obs_query_operator(w2, xq, target=2).predictive(
        exact_joint_posterior(w2, ctx)["w_lo"])
    assert np.allclose(p1, p2, atol=1e-9)


def test_order_relabeling_equivariance():
    world = _gaussian_world()
    rev = list(reversed(world.orderings))
    w2 = World(sigmas=world.sigmas, spec=world.spec, orderings=tuple(rev))
    ctx = _context(n_rows=4, seed=17)
    xq = np.array([0.5, 0.0])
    post1 = exact_joint_posterior(world, ctx)
    post2 = exact_joint_posterior(w2, ctx)
    # predictive invariance
    p1 = obs_query_operator(world, xq, target=2).predictive(post1["w_lo"])
    p2 = obs_query_operator(w2, xq, target=2).predictive(post2["w_lo"])
    assert np.allclose(p1, p2, atol=1e-9)
    # order posterior relabels: p_w2(o') = p_w1(rev(o'))
    assert np.allclose(post2["w_o"][::-1], post1["w_o"], atol=1e-12)


def test_sequential_matches_batch():
    world = _gaussian_world()
    ctx = _context(n_rows=6, seed=19)
    ll = np.zeros((6, world.K * world.O), dtype=np.float64)
    for t in range(1, 7):
        post_t = exact_joint_posterior(world, ctx[:t])
        ll[t - 1] = post_t["logZ_lo"] - post_t["logZ"]
    # sequential log-posterior increments: batch logZ_lo(t) - logZ_lo(t-1)
    # equals the incremental row loglikelihood (up to prior normalization).
    seq_inc = np.diff(ll, axis=0)  # (5, K*O)
    for t in range(5):
        # posterior at t+1 from posterior at t and row t (0-indexed row t+1)
        ctxp = ctx[t + 1:t + 2]
        llrow = np.array([
            csem.residual_logdensity_row(ctxp, world.sem_params(k, o), world.spec).sum()
            for k in range(world.K) for o in range(world.O)])
        # batch ratio: w_{t+1}/w_t ~ exp(llrow) up to the new normalizer
        w_t = np.exp(ll[t] - ll[t].max())
        w_tp1 = np.exp(ll[t + 1] - ll[t + 1].max())
        ratio = w_tp1 / (w_t + 1e-300)
        # compare log ratio to llrow - constant
        assert np.allclose(ratio / np.max(ratio), np.exp(llrow - llrow.max()), atol=1e-6)


def test_gaussian_order_uniform_posterior():
    """eps=0: all orders give identical observational likelihood; order
    posterior is uniform for fixed covariance."""
    world = _gaussian_world()
    ctx = _context(n_rows=6, seed=23)
    post = exact_joint_posterior(world, ctx)
    assert np.allclose(post["w_o"], np.full(world.O, 1.0 / world.O), atol=1e-12)
    # identical per-atom likelihoods across orders
    for k in range(world.K):
        ll_o = post["logZ_lo"].reshape(world.K, world.O)[k]
        assert np.allclose(ll_o, ll_o[0], atol=1e-9)


def test_gaussian_full_equals_order_ablated():
    world = _gaussian_world()
    ctx = _context(n_rows=5, seed=29)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([0.2, -0.8])
    q = obs_query_operator(world, xq, target=2)
    w_full = ablated_weights(world, post, "full")
    w_abl = ablated_weights(world, post, "order_ablated")
    p_full = q.predictive(w_full)
    p_abl = q.predictive(w_abl)
    assert np.allclose(p_full, p_abl, atol=1e-10)


def test_gaussian_order_value_zero():
    world = _gaussian_world()
    ctx = _context(n_rows=5, seed=31)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([0.2, -0.8])
    q = obs_query_operator(world, xq, target=2)
    _, _, V = exact_order_value(world, post, q)
    assert abs(V) < 1e-10


# ---------------------------------------------------------------------------
# 3. Proper scoring
# ---------------------------------------------------------------------------

def test_bayes_nll_unbeatable():
    """Expected NLL of a deliberately perturbed predictor >= Bayes NLL."""
    world = _tiny_world(k=4, d=3, eps=1.0)  # non-Gaussian: order matters
    rng = np.random.default_rng(41)
    n = 4000
    ctx = _context(n_rows=5, seed=43)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([0.3, -0.4])
    q = obs_query_operator(world, xq, target=2)
    p_bayes = q.predictive(post["w_lo"])
    # sample y from the exact posterior predictive
    y_bins = rng.choice(N_BINS, size=n, p=p_bayes)
    # perturbed predictor: mixture with the prior predictive
    p_prior = q.predictive(ablated_weights(world, post, "prior"))
    p_pert = 0.8 * p_bayes + 0.2 * p_prior
    nll_bayes = -np.log(np.maximum(p_bayes[y_bins], 1e-300))
    nll_pert = -np.log(np.maximum(p_pert[y_bins], 1e-300))
    assert np.mean(nll_bayes) <= np.mean(nll_pert) + 1e-3


def test_ordering_value_nonnegative():
    world = _tiny_world(k=4, d=3, eps=1.0)
    for seed in (51, 53):
        ctx = _context(n_rows=4, seed=seed)
        post = exact_joint_posterior(world, ctx)
        xq = np.array([0.1, 0.7])
        q = obs_query_operator(world, xq, target=2)
        _, _, V = exact_order_value(world, post, q)
        assert V >= -1e-9, V


def test_mc_score_diff_converges_to_kl():
    world = _tiny_world(k=4, d=3, eps=1.0)
    rng = np.random.default_rng(61)
    ctx = _context(n_rows=4, seed=63)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([-0.3, 0.2])
    q = obs_query_operator(world, xq, target=2)
    p_full = q.predictive(ablated_weights(world, post, "full"))
    p_abl = q.predictive(ablated_weights(world, post, "order_ablated"))
    kl = kl_divergence(p_full, p_abl)
    n = 60_000
    y_bins = rng.choice(N_BINS, size=n, p=p_full)
    mc = np.mean(np.log(np.maximum(p_full[y_bins], 1e-300))
                 - np.log(np.maximum(p_abl[y_bins], 1e-300)))
    assert abs(mc - kl) < 0.02


# ---------------------------------------------------------------------------
# 4. Numerical
# ---------------------------------------------------------------------------

def test_predictives_normalize():
    world = _tiny_world(k=4, d=3, eps=1.0)
    ctx = _context(n_rows=4, seed=67)
    post = exact_joint_posterior(world, ctx)
    for target in range(3):
        xq = np.array([0.0, 0.5]) if target != 2 else np.array([0.0, 0.5])
        q = obs_query_operator(world, xq, target=target)
        for mode in ("full", "order_ablated", "atom_ablated", "prior"):
            p = q.predictive(ablated_weights(world, post, mode))
            assert abs(p.sum() - 1.0) < 1e-9, (target, mode)
    # interventional
    for iv in range(3):
        iq = int_query_operator(world, iv, 1.0, target=(iv + 1) % 3)
        p = iq.predictive(post["w_lo"])
        assert abs(p.sum() - 1.0) < 1e-9


def test_tail_bins_handled():
    """Extreme tail bins carry finite, non-NaN mass and all bins are covered."""
    world = _tiny_world(k=4, d=3, eps=1.0)
    ctx = _context(n_rows=4, seed=71)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([2.0, -2.0])  # pushes the target into the tail
    q = obs_query_operator(world, xq, target=2)
    p = q.predictive(post["w_lo"])
    assert np.all(np.isfinite(p))
    assert p[0] + p[-1] > 0.0  # tail bins are actually used
    assert np.all(p >= 0.0)


def test_analytic_gaussian_cdf():
    """Single-atom Gaussian world: conditional predictive must match the
    analytic multivariate-Gaussian conditional N(mu, sigma^2)."""
    world = _tiny_world(k=1, d=3, eps=0.0)
    sigma = world.sigmas[0]
    target = 2
    xq = np.array([0.4, -0.7])
    q = np.array([0, 1])
    S_qq = sigma[np.ix_(q, q)]
    S_tq = sigma[target, q]
    S_tt = sigma[target, target]
    mu = S_tq @ np.linalg.solve(S_qq, xq)
    var = S_tt - S_tq @ np.linalg.solve(S_qq, S_tq)
    s = math.sqrt(max(var, 1e-12))
    edges = BIN_EDGES
    analytic = np.array([0.5 * (math.erf((edges[b + 1] - mu) / (s * math.sqrt(2)))
                                - math.erf((edges[b] - mu) / (s * math.sqrt(2))))
                         for b in range(N_BINS)])
    post = exact_joint_posterior(world, _context(n_rows=0))
    qop = obs_query_operator(world, xq, target=target)
    got = qop.predictive(post["w_lo"])
    assert np.allclose(got, analytic, atol=5e-3), "Gaussian conditional mismatch"


def test_scalar_vectorized_agreement(monkeypatch):
    """Independent scalar oracle vs vectorized oracle on a tiny non-Gaussian
    world. The scalar oracle is structurally independent (dense grid, explicit
    loops); agreement pins the integration, not just the SEM algebra."""
    monkeypatch.setattr(scalar, "N_GRID", 8001)
    world = _tiny_world(k=2, d=3, eps=1.0)
    ctx = _context(n_rows=3, seed=73)
    # posterior agreement
    w_scal, w_o_scal = scalar.scalar_posterior(world.sigmas, world.orderings,
                                               world.spec, ctx)
    post = exact_joint_posterior(world, ctx)
    assert np.allclose(w_scal, post["w_lo"], atol=1e-9)
    assert np.allclose(w_o_scal, post["w_o"], atol=1e-9)
    # observational predictive agreement
    xq = np.array([0.2, -0.3])
    p_scal = scalar.scalar_obs_predictive(world.sigmas, world.orderings, world.spec,
                                          xq, 2, post["w_lo"], BIN_EDGES)
    p_vec = obs_query_operator(world, xq, target=2).predictive(post["w_lo"])
    assert np.allclose(p_scal, p_vec, atol=2e-3), "obs predictive mismatch"
    # interventional predictive agreement
    p_scal_int = scalar.scalar_int_predictive(world.sigmas, world.orderings, world.spec,
                                              0, 1.0, 2, post["w_lo"], BIN_EDGES)
    p_vec_int = int_query_operator(world, 0, 1.0, 2).predictive(post["w_lo"])
    assert np.allclose(p_scal_int, p_vec_int, atol=5e-3), "int predictive mismatch"


def test_float64_golden_fixtures():
    """Frozen golden values for the deterministic d=3 K=8 eps=1.0 world.

    Values were computed once from the (then-current) implementation after the
    SEM algebra, Gaussian-CDF, and scalar-agreement tests passed; they pin
    regression. Recomputing them requires an independent re-derivation, not a
    silent update.
    """
    world = cworld.make_world(k=8, d=3, eps=1.0)
    ctx = cworld.default_context(world, n_rows=5, seed=990_000_101)
    post = exact_joint_posterior(world, ctx)
    xq = np.array([0.3, -0.6])
    q = obs_query_operator(world, xq, target=2)
    p = q.predictive(post["w_lo"])
    H = -np.sum(post["w_o"] * np.log(np.maximum(post["w_o"], 1e-300)))
    # Golden constants (frozen from the verified implementation).
    assert abs(H - 0.7455801261537632) < 1e-6, f"order-posterior entropy drifted: {H}"
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.allclose(p, _GOLDEN_PREDICTIVE, atol=1e-9)
    # order posterior is strongly peaked (the non-Gaussian context is
    # informative about order): p(o=4) ~ 0.62, p(o=5) ~ 0.36.
    assert post["w_o"][4] > 0.5 and post["w_o"][5] > 0.3


_GOLDEN_PREDICTIVE = np.array([
    9.586150815480832e-24, 1.5576840504297703e-23, 4.126819414529791e-23, 1.0969349002035675e-22,
    2.9227180522946684e-22, 7.800950551753728e-22, 2.084749520613982e-21, 5.576410594274351e-21,
    1.4925923573774735e-20, 3.997011558654116e-20, 1.0707352142587134e-19, 2.869079904464366e-19,
    7.68937035216599e-19, 2.061149745710462e-18, 5.5257270094952026e-18, 1.481587422724763e-17,
    3.9730692119999014e-17, 1.0656039569733656e-16, 2.8586104560722133e-16, 7.67069598732583e-16,
    2.059140240010971e-15, 5.5308420773793735e-15, 1.4869222345564295e-14, 4.003223905514526e-14,
    1.080334890766336e-13, 2.9271034830710235e-13, 7.985251651914247e-13, 2.204324928662922e-12,
    6.20957582211529e-12, 1.8095661927898228e-11, 5.562834664871003e-11, 1.845658737485539e-10,
    6.730853204614547e-10, 2.7087505004792953e-09, 1.1876532379371486e-08, 5.549813283838805e-08,
    2.7050433038031523e-07, 1.3531185927257583e-06, 6.873659115348599e-06, 3.523575127945638e-05,
    0.00018160925497136683, 0.0009391829675953328, 0.004817200979348446, 0.019179465664346373,
    0.047757465481013, 0.06564417789294555, 0.06718544103475062, 0.06907526300463684,
    0.07086038886253, 0.07273738229881721, 0.07390578710633502, 0.07522486527154895,
    0.07682196254252455, 0.07868335687136405, 0.07629773098950773, 0.06443059269022894,
    0.04572033049893924, 0.030359003526119584, 0.020162823323020506, 0.013393993307477073,
    0.008899673942447808, 0.005915004752321971, 0.003932477698266823, 0.002615305743656967,
    0.0017399647382785278, 0.0011580814058573653, 0.0007711514351048679, 0.0005137669524080627,
    0.0003424881565476355, 0.0002284589659894668, 0.00015250645416118054, 0.0001018881987068218,
    6.81330862976476e-05, 4.5607752989747254e-05, 3.0564630160137146e-05, 2.050967768236013e-05,
    1.3782356420105506e-05, 9.276529514151055e-06, 6.25495906803734e-06, 4.22597529097496e-06,
    2.8614542258620054e-06, 1.942248573543347e-06, 1.3218684984126426e-06, 9.022982081180655e-07,
    6.178852679841769e-07, 4.24601947841255e-07, 2.928829170842168e-07, 2.0284495717859382e-07,
    1.4109404091118314e-07, 9.859084337047328e-08, 6.922245794531822e-08, 4.884564708186231e-08,
    3.464519273401736e-08, 2.4703000017487633e-08, 1.7708304262486573e-08, 1.2762455596649933e-08,
    9.247175375409864e-09, 6.735468420540287e-09, 4.931246124936161e-09, 1.4392595264040965e-08,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 5. Provenance
# ---------------------------------------------------------------------------

def test_provenance_metadata():
    world = cworld.make_world(k=8, d=3, eps=1.0)
    meta = cworld.world_metadata(world, seed=cworld.WORLD_SEED_ROOT)
    assert meta["accepted_atom_count"] == 8
    assert len(meta["library_sha256"]) == 64 and meta["library_sha256"] != "0" * 64
    assert len(meta["config_sha256"]) == 64 and meta["config_sha256"] != "0" * 64
    assert meta["source"] == "corrected_world.sample_valid_atoms"
    assert meta["schema_version"] == 1


def test_interventional_coarse_matches_exact():
    """Guards the coarse tail-clamping path (Flaw 2): for an AL world, the
    coarse interventional operator must agree with the exact one to 5e-3,
    including the far tail bins (where np.interp would clamp)."""
    world = _tiny_world(k=2, d=3, eps=1.0)
    for iv, val, t in [(0, 1.8, 2), (1, -1.2, 0), (2, 0.6, 1)]:
        op_exact = int_query_operator(world, iv, val, t, coarse=False)
        op_coarse = int_query_operator(world, iv, val, t, coarse=True)
        assert np.allclose(op_exact.op, op_coarse.op, atol=5e-3), (iv, val, t)


def test_lp_l1_residual_correct():
    """Guards the L1 LP linearization (Flaw 3): the projection residual must
    equal the DIRECT L1 predictive residual recomputed from the returned w."""
    from pfn_dag_verify.corrected_tomography import (
        _lp_project_or_ranges, build_panel_operator, build_query_panels)
    world = _tiny_world(k=2, d=3, eps=1.0)
    ctx = _context(n_rows=3, seed=77)
    post = exact_joint_posterior(world, ctx)
    panel = build_query_panels(world)["Q1"]
    op = build_panel_operator(world, panel, coarse=True)
    targ = op.exact_predictive(post["w_lo"])
    # Perturb the target so the projection residual is substantial (the exact
    # target is realizable to ~1e-15, which would make the factor-2 guard
    # degenerate). A one-sided L1 linearization would report residual = L1/2.
    n_bins = op.num[0].shape[1]
    targ["obs"][0]["pred"] = 0.7 * targ["obs"][0]["pred"] + 0.3 / n_bins
    proj = _lp_project_or_ranges(world, op, targ, "project", 1e-3)
    w = proj["w_lo"]
    # direct L1 over the panel residuals
    total = 0.0
    for i in range(len(op.num)):
        num = w @ op.num[i]; den = float(w @ op.den[i])
        t = targ["obs"][i]
        total += np.sum(np.abs(num - t["pred"] * den)) / max(t["den"], 1e-12)
    for j in range(len(op.op)):
        if targ["int"][j] is None:
            continue
        total += np.sum(np.abs((w @ op.op[j]) - targ["int"][j]["pred"]))
    # Two-sided L1 linearization: the reported projection residual equals the
    # direct L1. A one-sided linearization would report exactly half (the
    # negative half of each residual runs free). Allow 5% LP-solver slack.
    assert abs(proj["residual"] - total) < 0.05 * max(total, 1e-9), (proj["residual"], total)


def test_int_operator_rows_valid():
    """Guards the m>=2 pivot (Flaw 6): every Q3 interventional row is finite
    and sums to 1, for every (iv, value, target) in the panel."""
    from pfn_dag_verify.corrected_identifiability import build_query_panels
    world = _tiny_world(k=8, d=3, eps=1.0)
    panels = build_query_panels(world)
    for spec in panels["Q3"].int:
        iq = int_query_operator(world, spec.intervened, spec.value, spec.target)
        assert np.all(np.isfinite(iq.op)), (spec)
        assert np.all(np.abs(iq.op.sum(axis=1) - 1.0) < 1e-9), (spec, iq.op.sum(axis=1))


def test_panel_nested():
    """Guards the nesting claim (Flaw 5): Q1's query spec is included in Q2."""
    from pfn_dag_verify.corrected_identifiability import build_query_panels
    world = _tiny_world(k=2, d=3, eps=1.0)
    panels = build_query_panels(world)
    def key(spec):
        return (spec.target, tuple(np.round(spec.x_q, 12)))
    q1_keys = {key(s) for s in panels["Q1"].obs}
    q2_keys = {key(s) for s in panels["Q2"].obs}
    q3_keys = {key(s) for s in panels["Q3"].obs}
    assert q1_keys <= q2_keys <= q3_keys


def test_no_broad_exception_swallowing():
    """Corrected modules raise on degenerate input rather than returning
    plausible-but-wrong values."""
    from pfn_dag_verify.corrected_sem import params_for, ResidualSpec
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD
    with pytest.raises(Exception):
        params_for(bad, (0, 1))
    with pytest.raises(ValueError):
        ResidualSpec(eps=1.5)
    with pytest.raises(ValueError):
        ResidualSpec(eps=-0.1)
    world = _gaussian_world()
    # degenerate context (all NaN) must raise, not silently return junk
    ctx = np.full((3, 3), np.nan)
    with pytest.raises((FloatingPointError, ValueError)):
        exact_joint_posterior(world, ctx)
