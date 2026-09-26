"""Amendment E.4 — the verdict rule must be falsifiable by the model it judges.

The superseded rule fired BRANCH-1-equivalent on

    (exact Q1 width >= 0.20) AND (in-task regret < 0.05)

Both terms are satisfied by the EXACT BAYES ORACLE: width is a substrate
property the model cannot change, and the oracle's regret against itself is 0.
So the rule granted its affirmative branch to a "model" that is definitionally
incapable of failing it. E.4 requires a test that pins this shut.

The battery below fixes the measured width ladder and varies only the model-side
quantities.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfn_dag_verify.corrected_verdict import (
    BRANCH_1,
    BRANCH_2,
    BRANCH_3,
    BRANCH_4,
    Cell,
    apply_written_default,
    classify,
)

# The measured exact ladder, POST-Amendment-E. Both axes, so a re-axing of the
# gate cannot silently invalidate these fixtures.
#
#   deficit  = Amendment F.2b PRIMARY axis: panel mean of 1 - H(p(o|D))/log O on
#              the 60-context eval panel (seed 770000000, n_rows 20), with its
#              across-context SE.
#   width    = SECONDARY axis: ident_eps*.json Q1 avg_width_mean, post-D1.
#
# The eps=0 deficit is 0 to one machine epsilon (2.2e-16) and the order marginal
# is uniform to 4.7e-15, which is structural rather than lucky: under Gaussian
# noise every ordering reproduces the same covariance, so the likelihood is
# ordering-invariant and the posterior never leaves its prior.
#
# The PRE-fix widths (0.4510 / 0.3232 / 0.2073) were used here until 2026-08-22.
# They are retired; see AMENDMENT_F.md F.4 and the retired-constant tripwire.
LADDER = [
    # eps,  deficit,   deficit_se,  width
    (0.0,   0.000000,  0.000000,    1.000000),
    (0.1,   0.035356,  0.007700,    0.437820),
    (0.25,  0.176728,  0.016100,    0.316973),
    (0.5,   0.458825,  0.028700,    0.260840),
    (1.0,   0.762581,  0.024600,    0.125500),
]


# F.2c instrument gain per rung. Measured mean pairwise anchor KL across the
# order-conditional predictives: 0 / 0.0031 / 0.0158 / 0.0588 / 0.3493, EXACTLY
# zero at eps=0 by the Gaussian order-invariance identity. The map below is the
# monotone stand-in the fixtures use; only its shape matters here.
GAIN = {0.0: 0.0, 0.1: 0.20, 0.25: 0.35, 0.5: 0.60, 1.0: 1.00}
FLOOR_FRAC = 0.02        # untrained model: a little instrument leakage
Y_SE = 0.01              # F.2d.1 weight is 1/SE_y^2, so every cell needs one
N_CTX = 500              # F.2's per-half floor
Y_JITTER = 0.01          # matched to Y_SE: s^2 lands near 1, the calibrated case


def _ctx(raw_mean, floor, gain, y_se, n=N_CTX):
    """Half-B per-context arrays whose ratio-of-means and linearized SE are
    exactly (raw_mean - floor)/gain and y_se.

    With a constant floor and ceiling the influence value collapses to
    ``(model_i - mean(model)) / gain``, so a symmetric two-point model curve of
    half-width ``y_se * gain * sqrt(n - 1)`` has exactly the requested SE.
    """
    if gain <= 0.0:                       # eps=0: no gain, hence no fidelity
        return [raw_mean] * n, [floor] * n, [floor] * n
    d = y_se * gain * np.sqrt(n - 1)
    model = [raw_mean + d] * (n // 2) + [raw_mean - d] * (n // 2)
    return model, [floor] * n, [floor + gain] * n


def _cells(regret_fn, fidelity_fn=None, seeds=3, rng_seed=0, y_se=Y_SE,
           gain=None, jitter=Y_JITTER):
    """Synthetic cells on the measured ladder.

    ``fidelity_fn`` is called with (eps, x), x the PRIMARY axis coordinate, and
    returns the CALIBRATED y -- the fraction of achievable that F.2d.1 reads.
    The raw readout is then derived as ``floor + gain * y``, which is the
    direction the real instrument runs and keeps the fixtures saying what they
    mean. A fixture that says "fidelity tracks identifiability" now tracks it on
    the axis and in the units the gate actually uses.

    ``jitter`` puts the cells slightly off any exact line. Without it a perfect
    fit gives a zero weighted residual and an infinite estimated-scale t, which
    is a numerical cliff rather than a positive control.
    """
    rng = np.random.default_rng(rng_seed)
    gain = GAIN if gain is None else gain
    out = []
    for eps, dfc, dse, w in LADDER:
        r = [regret_fn(eps) + rng.normal(0, 1e-4) for _ in range(seeds)]
        g = gain[eps]
        floor = FLOOR_FRAC * g
        m_ctx = f_ctx = c_ctx = None
        f = []
        if fidelity_fn is not None:
            y = fidelity_fn(eps, dfc) + (rng.normal(0, jitter) if jitter else 0.0)
            raw = floor + g * y
            m_ctx, f_ctx, c_ctx = _ctx(raw, floor, g, y_se)
            f = [raw + rng.normal(0, 1e-3) for _ in range(seeds)]
        out.append(Cell(eps=eps, width_exact=w, regret_seeds=r, fidelity_seeds=f,
                        deficit_exact=dfc, deficit_se=dse,
                        model_ctx=m_ctx, floor_ctx=f_ctx, ceiling_ctx=c_ctx))
    return out


def test_exact_bayes_oracle_does_not_reach_branch_1():
    """THE E.4 REQUIREMENT. The oracle attains regret 0 at every cell and leaves
    the width ladder untouched. With no measured model-side fidelity those are
    the only facts available, and they must NOT buy the affirmative branch."""
    res = classify(_cells(lambda eps: 0.0))
    assert res["verdict"] != BRANCH_1, (
        "the exact Bayes oracle reached the affirmative branch on predictive "
        "optimality alone; the rule has no power over the model")
    assert res["verdict"] == BRANCH_4
    assert not res["association"]["evaluable"]


def test_predictively_perfect_but_flat_fidelity_is_branch_2():
    """Regret 0 everywhere, structural fidelity that does not track
    identifiability. This is capture-without-tracking and must not be BRANCH 1."""
    res = classify(_cells(lambda eps: 0.0, lambda eps, x: 0.0))
    assert res["verdict"] == BRANCH_2
    assert res["association"]["evaluable"]
    assert not res["association"]["slope_passes"]


def test_fidelity_tracking_identifiability_is_branch_1():
    """Positive control: the rule CAN fire when the model-side series tracks
    (1 - width) and the eps=0 cell sits at the null."""
    res = classify(_cells(lambda eps: 0.0, lambda eps, x: 0.9 * x))
    assert res["verdict"] == BRANCH_1
    assert res["association"]["slope_passes"]
    assert res["association"]["slope"] > 0
    assert res["association"]["p_one_sided"] <= res["association"]["alpha"]


def test_anti_tracking_fidelity_is_not_branch_1():
    """Fidelity that runs the WRONG way must not be granted the branch."""
    res = classify(_cells(lambda eps: 0.0, lambda eps, x: 0.9 * (1.0 - x)))
    assert res["verdict"] == BRANCH_2


def test_regret_gate_failure_is_branch_3():
    res = classify(_cells(lambda eps: 0.9 if eps == 1.0 else 0.0,
                          lambda eps, x: 0.9 * x))
    assert res["verdict"] == BRANCH_3


def test_violated_eps0_null_blocks_branch_1():
    """Tracking present, but the eps=0 readout turns out to HAVE gain, where the
    Gaussian order-invariance identity says the ceiling equals the floor. That
    is an instrument that is not measuring what F.2c assumes, and it blocks the
    affirmative branch however well the rest of the ladder behaves.

    The violation has to be built into the GAIN, not into the fidelity function:
    under F.2c gate (b) stopped being a fidelity test -- it could never fail as
    one, since the readout statistic is identically zero at eps=0 for any model.
    An earlier version of this test varied the fidelity instead and passed by
    accident, through anti-tracking rather than through gate (b).
    """
    broken = {**GAIN, 0.0: 0.3}       # eps=0 readout with real gain
    res = classify(_cells(lambda eps: 0.0, lambda eps, x: 0.9 * x, gain=broken))
    assert res["verdict"] != BRANCH_1
    assert any("gate (b) VIOLATED" in e for e in res["evidence"])
    assert res["association"]["slope_passes"], (
        "the fixture is meant to isolate gate (b): the slope should still pass")


def test_too_few_cells_is_not_evaluable():
    """MIN_CELLS is 3 under F.2c (eps=0 leaves by construction, and D6 can still
    take eps=0.1). Two surviving cells is below the floor and must not evaluate."""
    cells = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    # eps=0 already carries no fidelity; strip two more to leave 2
    for c in cells:
        if c.eps in (0.1, 0.25):
            c.model_ctx = c.floor_ctx = c.ceiling_ctx = None
    res = classify(cells)
    assert res["verdict"] == BRANCH_4
    assert res["association"]["n_cells"] == 2
    assert not res["association"]["evaluable"]


def test_written_default_selects_d2_headline_after_the_date():
    res = classify(_cells(lambda eps: 0.0))
    assert res["verdict"] == BRANCH_4
    before = apply_written_default(res, "2026-09-03")
    after = apply_written_default(res, "2026-09-04")
    assert not before["written_default_applied"]
    assert after["written_default_applied"]
    assert after["headline"] == "D2_CORE"


@pytest.mark.parametrize("gate", [0.05])
def test_negative_point_regret_is_legitimate(gate):
    """E.3: the Bayes predictive hedges, so a negative per-panel regret is
    legitimate and must not be treated as a gate failure."""
    res = classify(_cells(lambda eps: -0.006, lambda eps, x: 0.9 * x),
                   regret_gate=gate)
    assert res["verdict"] == BRANCH_1


# --- the F.2b analogue of the E.4 power requirement ------------------------

def test_the_primary_axis_is_actually_consulted():
    """E.4's power test asks whether the rule is sensitive to the MODEL. This
    asks whether it is sensitive to the AXIS the amendment just made primary.

    Nothing else in this file would catch a ``classify`` that read
    ``deficit_exact`` nowhere: with fidelity rising monotonically and the widths
    left in place, a verdict computed off the demoted axis would look correct.
    So permute the deficits across cells while holding every fidelity, regret
    and width fixed, and require the verdict to move. If it does not, the
    primary axis is decorative.
    """
    tracking = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    base = classify(tracking)
    assert base["verdict"] == BRANCH_1

    # reverse the deficit ladder; fidelity, regret and width are untouched
    scrambled = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    deficits = [c.deficit_exact for c in scrambled][::-1]
    ses = [c.deficit_se for c in scrambled][::-1]
    for c, d, se in zip(scrambled, deficits, ses):
        c.deficit_exact, c.deficit_se = d, se
    flipped = classify(scrambled)

    assert flipped["verdict"] != BRANCH_1, (
        "reversing the primary axis left the affirmative branch standing; "
        "classify is not reading deficit_exact")
    assert flipped["association"]["spearman"] < 0 < base["association"]["spearman"]


def test_the_two_axes_are_not_reported_as_independent_corroboration():
    """Two slopes on the same four y values against two monotonically related
    x's are one measurement expressed in two units. Under the old rank gate they
    were bit-identical; under F.2d.1 they differ numerically, which is worse --
    small disagreement invites reading them as a replication. The output must
    say what they are."""
    res = classify(_cells(lambda eps: 0.0, lambda eps, x: 0.9 * x))
    cmp = res["axis_comparison"]
    assert cmp["comparable"] and cmp["rank_orders_identical"]
    # the RANK statistics still coincide exactly; the slopes need not
    assert res["association"]["spearman"] == res["association_secondary"]["spearman"]
    assert any("re-expression" in e.lower() for e in res["evidence"])


# --- Amendment F.2c --------------------------------------------------------

def test_eps0_leaves_the_fidelity_series_by_construction():
    """At eps=0 every ordering reproduces the same covariance, so the readout's
    ceiling equals its floor and no fidelity is definable. This is a fact about
    eps, not a decision about the model, and it must not require anyone to
    remember to exclude the cell."""
    cells = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    zero = next(c for c in cells if c.eps == 0.0)
    assert zero.calib_gain == 0.0
    assert zero.has_fidelity is False
    assert zero.fidelity_norm is None
    assert zero.has_deficit and zero.deficit_exact == 0.0   # x-only anchor
    res = classify(cells)
    assert 0.0 not in res["association"]["eps_cells"]
    assert res["association"]["n_cells"] == 4


def test_three_surviving_cells_still_evaluate_but_carry_the_caveat():
    """Removing eps=0 leaves four cells and D6 can still take eps=0.1, so the
    old '>= 4 cells' wording had zero slack. Three must evaluate, loudly."""
    cells = [c for c in _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
             if c.eps not in (0.0, 0.1)]
    assert len(cells) == 3
    res = classify(cells)
    assert res["association"]["evaluable"]
    assert res["association"]["n_cells"] == 3
    assert any("fallback IN FORCE" in e for e in res["evidence"])
    assert res["association"]["mode"] == "descriptive"


def test_a_rising_instrument_floor_cannot_manufacture_the_association():
    """The tautology channel, as a test. An untrained model has zero structural
    tracking by construction, but its RAW readout rises with eps because the
    anchors separate. Calibrated, its fraction-of-achievable is flat, so it must
    not reach the affirmative branch."""
    cells = []
    for eps, dfc, dse, w in LADDER:
        gain = GAIN[eps]
        floor = 0.4 * gain                      # heavy instrument leakage
        # the model IS the untrained floor: raw y rises with eps purely because
        # the measuring stick lengthens, so calibrated y is flat at zero
        m, f, c = _ctx(floor, floor, gain, Y_SE)
        cells.append(Cell(eps=eps, width_exact=w, deficit_exact=dfc, deficit_se=dse,
                          regret_seeds=[0.0] * 3, regret_panel_se=1e-4,
                          model_ctx=m, floor_ctx=f, ceiling_ctx=c,
                          fidelity_seeds=[floor - 1e-4, floor, floor + 1e-4]))
    res = classify(cells)
    assert res["verdict"] != BRANCH_1, (
        "a model with zero structural tracking reached the affirmative branch "
        "on instrument gain alone; F.2c's calibration is not being applied")
    leak = res["calibration_leakage"]
    assert leak["measured"] and leak["floor_spearman_vs_axis"] > 0.9


# --- defects the pre-lock red team confirmed, as regressions ---------------

def test_a_cell_that_enters_the_association_cannot_skip_gate_a():
    """E.4 (a) is 'below the gate at EVERY cell', and a cell supplying x and y
    to gate (c) is one of them.

    The regret and the fidelity are separate artifacts written by separate jobs,
    so one landing without the other is the ORDINARY case. Testing only the
    cells that happened to carry a regret let a missing file turn BRANCH_3 into
    BRANCH_1: the cell contributed a degree of freedom to the affirmative gate
    while being exempt from the predictive one, behind an evidence line reading
    'gate (a) holds at all N scored cells'.
    """
    cells = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    assert classify(cells)["verdict"] == BRANCH_1
    for c in cells:
        if c.eps == 1.0:
            c.regret_seeds = []                  # its trackb artifact never landed
    res = classify(cells)
    assert res["verdict"] == BRANCH_4
    assert res["unscored_cells"] == [1.0]
    assert any("NOT EVALUABLE at eps=1" in e for e in res["evidence"])


def test_a_single_seed_without_a_panel_se_has_no_regret_ci():
    """A zero-width CI has an upper bound equal to the point estimate, so the
    one-sided gate passed on a bare number with no uncertainty in it."""
    c = Cell.from_summary(0.5, y=0.5, y_se=0.01, n_contexts=500,
                          deficit_exact=0.4, deficit_se=0.01,
                          regret_seeds=[0.001])
    assert c.regret_ci() is None
    assert c.panel_se_used is False
    c.regret_panel_se = 0.002
    assert c.panel_se_used is True
    lo, hi = c.regret_ci()
    assert hi > c.regret > lo


def test_a_cell_with_a_fidelity_but_no_x_is_named_not_dropped():
    """It is excluded from the association either way. What it must not do is
    be reported as surviving a series it was excluded from."""
    cells = _cells(lambda eps: 0.0, lambda eps, x: 0.9 * x)
    for c in cells:
        if c.eps == 0.5:
            c.deficit_exact = c.deficit_se = None
    res = classify(cells)
    assoc = res["association"]
    assert assoc["cells_without_axis"] == [0.5]
    row = next(r for r in assoc["survivorship"] if r["eps"] == 0.5)
    assert row["survives"] is True and row["in_association"] is False
    assert "no deficit coordinate" in row["excluded_reason"]
