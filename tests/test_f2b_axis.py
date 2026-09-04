"""Amendment F.2b — the entropy-deficit primary axis.

Three things are pinned here.

1. The deficit is a correct normalized entropy deficit, with the endpoint and
   zero conventions the amendment states.
2. The claim F.2b rests on for its own legitimacy: Spearman on this many cells
   is invariant to strictly monotone re-axing, so adopting the axis pre-data
   cannot be gate-shopping. If that ever stops being true, the amendment's
   argument stops being true with it and this test should fail loudly.
3. The near-tie clause is binding in the direction stated: an unresolved
   adjacent pair is reported and does NOT by itself fail gate (c).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pfn_dag_verify.corrected_oracle import (
    order_entropy_deficit,
    panel_order_entropy_deficit,
)
from pfn_dag_verify import corrected_verdict as V


class _FakeWorld:
    """Minimal stand-in: the deficit reads ``O``, ``prior_order`` and ``w_o``."""

    def __init__(self, O: int = 6, prior_order=None):
        self.O = O
        self.prior_order = (np.full(O, 1.0 / O) if prior_order is None
                            else np.asarray(prior_order, dtype=float))


def _post(w_o):
    return {"w_o": np.asarray(w_o, dtype=np.float64)}


# --- the estimand ----------------------------------------------------------

def test_uniform_order_marginal_has_zero_deficit():
    w = _FakeWorld(6)
    d = order_entropy_deficit(w, post=_post(np.full(6, 1 / 6)))
    assert d == pytest.approx(0.0, abs=1e-12)


def test_point_mass_has_unit_deficit():
    w = _FakeWorld(6)
    d = order_entropy_deficit(w, post=_post([1.0, 0, 0, 0, 0, 0]))
    assert d == pytest.approx(1.0, abs=1e-12)


def test_zeros_use_the_0log0_convention():
    """Half the mass on two orderings: H = log 2, deficit = 1 - log2/log6."""
    w = _FakeWorld(6)
    d = order_entropy_deficit(w, post=_post([0.5, 0.5, 0, 0, 0, 0]))
    assert d == pytest.approx(1.0 - math.log(2) / math.log(6), abs=1e-12)
    assert np.isfinite(d)


def test_deficit_is_bounded_on_random_posteriors():
    w = _FakeWorld(6)
    rng = np.random.default_rng(20260822)
    for _ in range(200):
        p = rng.dirichlet(np.full(6, rng.uniform(0.05, 5.0)))
        d = order_entropy_deficit(w, post=_post(p))
        assert -1e-12 <= d <= 1.0 + 1e-12


def test_deficit_is_renormalization_invariant():
    w = _FakeWorld(6)
    p = np.array([0.4, 0.3, 0.2, 0.05, 0.03, 0.02])
    a = order_entropy_deficit(w, post=_post(p))
    b = order_entropy_deficit(w, post=_post(p * 7.3))
    assert a == pytest.approx(b, abs=1e-12)


def test_degenerate_marginal_raises_rather_than_returning_a_number():
    w = _FakeWorld(6)
    with pytest.raises(FloatingPointError):
        order_entropy_deficit(w, post=_post(np.zeros(6)))


def test_single_ordering_world_is_rejected():
    """O = 1 makes log O = 0. Refuse rather than divide by zero."""
    with pytest.raises(ValueError):
        order_entropy_deficit(_FakeWorld(1), post=_post([1.0]))


def test_panel_average_is_mean_of_per_context_deficits():
    """F.2b averages the deficit over contexts, NOT the deficit of a pooled
    posterior. Those differ, and the amendment picks the former."""
    class W:
        O = 6

        def __init__(self):
            self.calls = []

    posts = [np.array([1.0, 0, 0, 0, 0, 0]), np.full(6, 1 / 6)]
    per = [order_entropy_deficit(_FakeWorld(6), post=_post(p)) for p in posts]

    import pfn_dag_verify.corrected_oracle as O
    orig = O.exact_joint_posterior
    O.exact_joint_posterior = lambda world, ctx: _post(posts[int(ctx)])
    try:
        out = panel_order_entropy_deficit(_FakeWorld(6), [0, 1])
    finally:
        O.exact_joint_posterior = orig

    assert out["deficit_mean"] == pytest.approx(float(np.mean(per)))
    assert out["n_contexts"] == 2
    # the pooled posterior of these two is NOT uniform-averaged to the same thing
    pooled = order_entropy_deficit(_FakeWorld(6), post=_post(sum(posts) / 2))
    assert out["deficit_mean"] != pytest.approx(pooled)


# --- the legitimacy claim --------------------------------------------------

Y_SE = 0.002        # F.2d.1 weights are 1/SE_y^2, so a cell needs a positive one
N_CTX = 500         # F.2's per-half floor


def _cells(xs, ys, ses=None, axis="deficit"):
    """Synthetic cells. ys[0] must be the eps=0 cell's fidelity, and gate (b)
    requires its CI to cover FIDELITY_NULL, so ys[0] is 0 in these fixtures."""
    ses = ses if ses is not None else [1e-6] * len(xs)
    out = []
    for i, (x, y, se) in enumerate(zip(xs, ys, ses)):
        # F.2d.6: y and SE_y come from half-B contexts and only from there.
        # ``from_summary`` builds contexts that reproduce (y, SE_y) exactly
        # under the linearization, so these fixtures keep testing what they were
        # written to test without opening a second y-path.
        kw = dict(y=y, y_se=Y_SE, n_contexts=N_CTX,
                  fidelity_seeds=[y - 0.001, y, y + 0.001],
                  regret_seeds=[0.0, 0.0, 0.0], regret_panel_se=0.001)
        # eps starts at 0.1, not 0: under F.2c an eps=0 cell must have ZERO
        # readout gain (ceiling == floor), and giving one unit gain would --
        # correctly -- trip gate (b) as an instrument violation. The eps=0
        # behaviour is covered in test_verdict_power, not here.
        eps = float(i + 1) / 10
        if axis == "deficit":
            c = V.Cell.from_summary(eps, width_exact=0.5,
                                    deficit_exact=x, deficit_se=se, **kw)
        else:
            c = V.Cell.from_summary(eps, width_exact=1.0 - x, **kw)
        out.append(c)
    return out


def test_spearman_is_invariant_to_strictly_monotone_reaxing():
    """THE claim Amendment F.2b's own legitimacy argument rests on.

    If re-axing could move rho other than by reordering cells, then swapping the
    x-axis pre-data would be a way of buying a result, and F.2b would be
    gate-shopping rather than a validity fix. It cannot, because rho reads ranks.
    """
    rng = np.random.default_rng(7)
    for _ in range(50):
        x = np.sort(rng.uniform(0.01, 0.99, 5))
        y = rng.uniform(-1, 1, 5)
        base = V._spearman(x, y)
        for f in (np.log, np.sqrt, lambda v: v ** 3, lambda v: np.expm1(4 * v)):
            assert V._spearman(f(x), y) == pytest.approx(base, abs=1e-12)


def test_reaxing_changes_the_gate_only_by_reordering():
    """The contrapositive: a re-axing that DOES reorder may change rho."""
    y = np.array([0.1, 0.5, 0.2, 0.9, 0.7])
    x1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x2 = np.array([0.1, 0.3, 0.2, 0.4, 0.5])          # swaps cells 1 and 2
    assert V._spearman(x1, y) != pytest.approx(V._spearman(x2, y))


def test_spearman_uses_midranks_on_ties():
    """Preregistered tie convention. Hand-checkable: x has one tied pair."""
    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    from scipy import stats
    assert V.SPEARMAN_TIE_METHOD == "average"
    rx = stats.rankdata(x, method="average")
    assert list(rx) == [1.0, 2.5, 2.5, 4.0]
    assert V._spearman(x, y) == pytest.approx(
        float(np.corrcoef(rx, stats.rankdata(y, method="average"))[0, 1]))


def test_constant_axis_is_not_evaluable_rather_than_zero():
    assert math.isnan(V._spearman(np.array([1.0, 1, 1, 1]), np.array([1.0, 2, 3, 4])))


# --- the near-tie clause ---------------------------------------------------

def test_overlapping_adjacent_cis_are_reported_unresolved():
    cells = _cells([0.10, 0.101, 0.40, 0.80], [0.0, 0.2, 0.3, 0.4],
                   ses=[0.01, 0.01, 0.001, 0.001])
    res = V.adjacent_resolution(cells)
    assert len(res) == 3
    assert res[0]["resolved"] is False          # 0.10 vs 0.101, ses 0.01
    assert res[1]["resolved"] is True
    assert res[2]["resolved"] is True


def test_an_unresolved_pair_does_not_by_itself_fail_the_gate():
    """The binding direction of the near-tie clause.

    Under F.2d.1 the gate is a weighted slope rather than a rank statistic, and
    the clause holds a fortiori: the two unresolved rungs sit at nearly the same
    x, so they contribute almost nothing to Sxx and the slope is carried by the
    rungs the dial does separate. A near-tie degrades the fit smoothly instead
    of hinging on which side of it two cells fall.
    """
    cells = _cells([0.10, 0.101, 0.40, 0.80], [0.10, 0.12, 0.25, 0.40],
                   ses=[0.01, 0.01, 0.001, 0.001])
    out = V.classify(cells)
    assert any(not r["resolved"] for r in out["adjacent_resolution"])
    assert out["verdict"] == V.BRANCH_1
    assert any("near-tie clause" in e for e in out["evidence"])


# --- axis plumbing ---------------------------------------------------------

def test_both_axes_are_oriented_larger_means_more_identified():
    c = V.Cell(eps=0.5, width_exact=0.26, deficit_exact=0.73, deficit_se=0.01)
    assert c.axis_value(V.PRIMARY_AXIS) == pytest.approx(0.73)
    assert c.axis_value(V.SECONDARY_AXIS) == pytest.approx(0.74)


def test_unknown_axis_raises():
    c = V.Cell(eps=0.0, width_exact=0.5)
    with pytest.raises(ValueError):
        c.axis_value("nope")


def test_cells_without_a_deficit_cannot_enter_the_primary_association():
    """Pre-F artifacts carry no deficit, so the honest answer is NOT_EVALUABLE
    on the primary axis even when the width axis would have been evaluable."""
    cells = _cells([0.2, 0.4, 0.6, 0.8], [0.1, 0.2, 0.3, 0.4], axis="width")
    primary = V.association(cells, axis=V.PRIMARY_AXIS)
    secondary = V.association(cells, axis=V.SECONDARY_AXIS)
    assert primary["evaluable"] is False
    assert "deficit" in primary["reason"]
    assert secondary["evaluable"] is True


def test_secondary_axis_is_reported_but_never_gates():
    """Width axis anti-correlated, deficit axis correlated: the verdict follows
    the PRIMARY axis and the disagreement is surfaced in the evidence."""
    cells = []
    # deficit rises with the fidelity; the width axis is built to run BACKWARDS
    for i, (d, wax, y) in enumerate([(0.10, 0.90, 0.0), (0.30, 0.60, 0.2),
                                     (0.60, 0.30, 0.3), (0.90, 0.10, 0.4)]):
        cells.append(V.Cell.from_summary(
            (i + 1) / 10, y=y, y_se=Y_SE, n_contexts=N_CTX,
            width_exact=1.0 - wax, deficit_exact=d, deficit_se=1e-4,
            fidelity_seeds=[y - 1e-3, y, y + 1e-3],
            regret_seeds=[0.0] * 3, regret_panel_se=1e-3))
    out = V.classify(cells)
    assert out["association"]["axis"] == V.PRIMARY_AXIS
    assert out["association_secondary"]["axis"] == V.SECONDARY_AXIS
    assert out["verdict"] == V.BRANCH_1
    assert any("DISAGREE" in e for e in out["evidence"])


# --- guards added after the F.2b design audit ------------------------------

def test_wrong_world_posterior_is_rejected():
    """The post= path is exactly where a posterior from the wrong world gets
    threaded through. Without the shape guard a d=2 posterior normalised by
    log 6 returns a plausible 0.98."""
    with pytest.raises(ValueError, match="w_o has shape"):
        order_entropy_deficit(_FakeWorld(6), post=_post([0.5, 0.5]))


def test_non_uniform_order_prior_is_rejected():
    """log O is the prior's order entropy only when the order prior is uniform.
    A tilted prior would put a spurious identifiability floor at eps=0, which is
    where gate (b)'s null lives."""
    tilted = _FakeWorld(6, prior_order=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="not uniform"):
        order_entropy_deficit(tilted, post=_post(np.full(6, 1 / 6)))


def test_single_context_panel_is_refused_not_given_zero_se():
    """se = 0.0 would make adjacent_resolution declare every rung resolved,
    failing OPEN in the clause whose job is to report what cannot be separated."""
    import pfn_dag_verify.corrected_oracle as O
    orig = O.exact_joint_posterior
    O.exact_joint_posterior = lambda world, ctx: _post(np.full(6, 1 / 6))
    try:
        with pytest.raises(ValueError, match="at least 2 contexts"):
            panel_order_entropy_deficit(_FakeWorld(6), [0])
    finally:
        O.exact_joint_posterior = orig


def test_zero_separation_reports_unresolved():
    """Fail closed: no measured SE is not the same as a resolved pair."""
    cells = [V.Cell(eps=0.0, width_exact=1.0, deficit_exact=0.1, deficit_se=0.0),
             V.Cell(eps=0.5, width_exact=0.3, deficit_exact=0.9, deficit_se=0.0)]
    res = V.adjacent_resolution(cells)
    assert res[0]["resolved"] is False


def test_eval_panel_contexts_dedups_and_matches_the_scoring_panel():
    """The SE bug this prevents is a factor sqrt(n_query_per_context), which
    inflates every near-tie z by the same factor."""
    from pfn_dag_verify.corrected_world import make_world
    from pfn_dag_verify.corrected_models import make_eval_panel, make_eval_panel_contexts
    w = make_world(k=8, d=3, eps=1.0)          # eps=1 short-circuits: fast
    panel = make_eval_panel(w, 6, 20, seed=770_000_000, n_query_per_context=2)
    ctxs = make_eval_panel_contexts(w, 6, 20, seed=770_000_000, n_query_per_context=2)
    assert len(panel) == 12 and len(ctxs) == 6
    # value equality, not identity: the helper builds its own panel from the
    # same seed, so the arrays are equal but distinct objects.
    for i, c in enumerate(ctxs):
        assert np.array_equal(c, panel[2 * i][0])
    # and the panel really did repeat each context, which is the bug this guards
    assert panel[0][0] is panel[1][0]
