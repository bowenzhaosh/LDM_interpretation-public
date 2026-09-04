"""Amendment E.4 — the C5 verdict.

REBUILT FROM SPEC. The superseded module had two classifier functions with
incompatible vocabularies, and the one that produced ``raw/verdict.json`` had no
power over the question it decided: it fired on

    (exact Q1 identified-set width >= 0.20) AND (in-task Bayes regret < 0.05)

where the first term is a property of the SUBSTRATE's query operator and the
second is near zero for anything competent. Substituting the exact Bayes oracle
for the PFN passed that rule. It also emitted the string ``THESIS_SUPPORTED``
for what was semantically the capture-without-structure branch. Both functions
and all four old label strings are retired; nothing in the pre-Amendment-E
``raw/verdict.json`` is citable.

The rebuilt rule takes a per-eps tuple

    (width_exact, pfn_fidelity +/- CI, expected_regret +/- CI)

where ``pfn_fidelity`` is a MEASURED model-side quantity (Amendment E.5: the
PFN's own d=3 s/w tracking statistic, the same estimand family as C1 at d=2),
and ``expected_regret`` is the E.3 prior-averaged exact expected regret with a
panel-level CI. Requirement (c) below is what makes the rule model-sensitive;
``tests/test_verdict_power.py`` enforces that the exact Bayes oracle fails it.

Branches (preregistered, Amendment E.4):

  BRANCH 1  STRUCTURE_TRACKS_IDENTIFIABILITY
      (a) expected_regret below the gate at EVERY cell, AND
      (b) the eps=0 fidelity CI covers the no-signal value, AND
      (c) SUPERSEDED BY F.2d.1. As signed under E.4c this read "positive with a
          bootstrap CI excluding zero, by both an isotonic fit and a Spearman
          rank correlation, agreeing in sign". That bootstrap resampled seeds
          within cells while holding the cell assignment fixed and fired at
          40-48% under the null; it is retired with cause and DELETED. The gate
          is now a one-sided WEIGHTED SLOPE of the calibrated fidelity on the
          entropy deficit, weights 1/SE_y^2, t on df = k-2, alpha 0.05, over the
          surviving eps>0 cells. Below MIN_CELLS_PREFERRED the slope is
          descriptive and F.2d.4's extreme-cell contrast decides instead.


  BRANCH 2  PREDICTIVE_CAPTURE_WITHOUT_STRUCTURAL_TRACKING
      (a) holds and (c) fails.

  BRANCH 3  PREDICTIVE_GATE_FAILS
      (a) fails at any cell. Not a paper outcome; an instrument or budget finding.

  BRANCH 4  NOT_EVALUABLE
      fewer than MIN_CELLS cells carry a calibrated fidelity with a finite SE,
      or a cell that enters the association carries no expected regret, or the
      eps=0 null is violated. F.2c: eps=0 leaves the fidelity series by
      construction, so four surviving cells is the CEILING, not the target.

AMENDMENT F.2b — PRIMARY X-AXIS. Gate (c)'s x-axis is the panel mean of the
normalized order-marginal entropy deficit, 1 - H(p(o|D))/log O, averaged over
the SAME eval panel the PFN is scored on (``corrected_oracle.
panel_order_entropy_deficit``). The LP identified-set width is DEMOTED to a
secondary axis, reported in the appendix and never gating.

Why, stated so it cannot be read as gate-shopping after the fact. Spearman on
five cells is invariant to any strictly monotone re-axing, so F.2b can change
the gate ONLY by reordering cells. Its value is validity and near-tie
diagnosis, not passing anything, which is exactly why it is adopted now,
before the model-side series exists. Two structural reasons:

  * the width ladder is computed on a separately configured context panel whose
    n_rows the ident artifacts never recorded (F.5), so x and y sat at
    different information levels; the deficit is measured on the scoring panel,
    so alignment is automatic rather than declared;
  * the deficit is a functional of the exact posterior and touches no linear
    program, so it cannot inherit an LP solver path.

Both axes are always computed and reported. ``association_secondary`` in the
result carries the width-axis answer so a reader can see whether the two agree.

NEAR-TIE CLAUSE (F.2b, binding). Adjacent rungs whose deficit CIs overlap are
reported as UNRESOLVED. An unresolved adjacent pair does NOT fail gate (c) on
its own: with the rungs ordered but two of them statistically indistinguishable,
the rank correlation is still the preregistered gate, and the pair is a
precision statement about the dial, not evidence against tracking.

WRITTEN DEFAULT (E.4, binding): if the verdict is not evaluable by 2026-09-04,
the d=2 headline is automatically selected and the eps-dial ships as a pending
section. ``DEFAULT_DATE`` records that date; ``apply_written_default`` applies it.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from .artifact_status import (EST_FIDELITY_NORM, EST_IDENT_WIDTH,
                              EST_ORDER_EVIDENCE, EST_REGRET_INTASK,
                              require_live)
from .corrected_identifiability_run import HEADLINE_N_ROWS
from .split_panel import HALF_A, HALF_B

# --- preregistered constants (Amendment E.4) -------------------------------
REGRET_GATE = 0.05          # nats: "essentially Bayes-optimal" on the trained task
MIN_CELLS = 3               # F.2c: eps=0 exits the fidelity series by
                            # construction, so 4 surviving cells is the CEILING
                            # and D6 can still take eps=0.1. A 4-cell wording
                            # would leave zero slack; 3 is the floor, with the
                            # written default attached (MIN_CELLS_PREFERRED).
MIN_CELLS_PREFERRED = 4     # below this the verdict ships with the F.2c caveat
FIDELITY_NULL = 0.0         # no-signal value of the RAW tracking statistic.
                            # Under F.2c gate (b) is no longer a fidelity test
                            # (see the eps=0 clause), so nothing gates on this;
                            # it is retained for the reproduce-uniform series.
DEFAULT_DATE = "2026-09-04"

# --- Amendment F.2d: the gate is a one-sided weighted slope -----------------
# E.4c's "Spearman with a bootstrap CI excluding zero" is RETIRED WITH CAUSE and
# its implementation is deleted rather than bypassed (F.2d.3). It resampled the
# wrong unit -- seeds within a cell share one panel, and the panel SE runs 3.1x
# to 12.5x the across-seed SE -- and it had no null, because the cell assignment
# never moved. Measured false-positive rate under a null where fidelity is
# independent of the axis: 0.40-0.48 against a nominal 0.025.
#
# A retired statistic that stays callable is the same disease as a retired
# artifact that stays loadable (F.7/F.8), so N_BOOT and BOOT_SEED are gone with
# it. tests/test_retired_constants.py is the tripwire that keeps them gone.
ALPHA = 0.05                # one-sided, H0: slope <= 0
CONTRAST_Z = 1.96           # F.2d.4 extreme-cell contrast, two-sided 95% CI

# --- Amendment F.2b --------------------------------------------------------
PRIMARY_AXIS = "deficit"    # panel mean order-marginal entropy deficit
SECONDARY_AXIS = "width"    # 1 - LP identified-set width (appendix only)
# Preregistered tie handling: midranks. scipy.stats.spearmanr already averages
# tied ranks, but the convention is pinned here and applied explicitly so the
# rule does not depend on a library default that could change under us.
SPEARMAN_TIE_METHOD = "average"
NEAR_TIE_Z = 1.96           # adjacent-rung CI overlap test

# --- Amendment F.2c: calibrated fidelity -----------------------------------
# The y-instrument's GAIN is itself a function of eps. The order-conditional
# predictive anchors coincide exactly at eps=0 (measured mean pairwise KL 0.0)
# and separate as eps rises, in lockstep with the primary axis (measured
# Spearman +1.000). So part of any RAW fidelity increase is the measuring stick
# lengthening, with no model in it.
#
# As a confound that must be removed. As physics it is the identifiability null
# appearing inside the instrument: at eps=0 no readout can have gain because
# there is no order information to read. Both readings are correct, and the
# standard calibration design serves both.
#
# Three curves through the IDENTICAL readout pipeline:
#   ceiling  the exact Bayes posterior pushed through the readout
#   floor    a dose-0 / untrained checkpoint pushed through the readout
#   model    the trained PFN
# and y := (model - floor) / (ceiling - floor), the fraction of achievable.
#
# The FLOOR curve is what quantifies the leakage: if an untrained model's raw
# readout rises with eps, raw-y associations are confounded by exactly that
# much. The Spearman establishes that the channel exists, not its magnitude.
CALIB_MIN_GAIN = 1e-6       # ceiling - floor below this: fidelity undefined

BRANCH_1 = "STRUCTURE_TRACKS_IDENTIFIABILITY"
BRANCH_2 = "PREDICTIVE_CAPTURE_WITHOUT_STRUCTURAL_TRACKING"
BRANCH_3 = "PREDICTIVE_GATE_FAILS"
BRANCH_4 = "NOT_EVALUABLE"


@dataclass
class Cell:
    """One eps cell of the dial.

    deficit_exact  F.2b PRIMARY x-axis: panel mean of 1 - H(p(o|D))/log O.
                   Measured on HALF A of the cell's panel (F.2d.5). None = not
                   measured.
    deficit_se     across-context SE of that mean (panel contexts are iid prior
                   draws, so this is the SE of the prior-averaged estimand)
    width_exact    exact Q1 identified-set average width; F.2b SECONDARY, appendix
    regret_seeds   per-seed E.3 prior-averaged exact expected regret, HALF B
    model_ctx      F.2d.6: per-context readout of the trained model on HALF B,
                   averaged over seeds. With floor_ctx and ceiling_ctx these are
                   THE source of y and SE_y; nothing else may supply them.
    fidelity_seeds per-seed panel means of the raw readout. DIAGNOSTIC ONLY --
                   the across-seed spread is reported beside SE_y and does not
                   enter the gate (F.2d.1 reads the context SE).
    """

    eps: float
    # Optional under F.2b: the LP width is the DEMOTED axis, so a cell must be
    # able to exist without one. The eps=0.5 simplex cell that could not clear a
    # 24h wall is precisely the case that must not drop a rung from the ladder.
    width_exact: float | None = None
    regret_seeds: list[float] = field(default_factory=list)
    fidelity_seeds: list[float] = field(default_factory=list)
    regret_panel_se: float | None = None
    deficit_exact: float | None = None
    deficit_se: float | None = None
    # F.2c/F.2d.6 calibration, per HALF-B context, in the RAW readout's units.
    model_ctx: list[float] | None = None
    floor_ctx: list[float] | None = None
    ceiling_ctx: list[float] | None = None

    # -- F.2d.6: y and SE_y come from the per-context arrays, and only from them
    def _ctx(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The half-B triple, or None if this cell carries no calibrated readout.

        There is exactly ONE way to supply y. A cell that could take either a
        per-context array or a precomputed scalar would let the two disagree,
        and the disagreeing one would look entirely plausible -- the failure
        mode F.7 exists to stop. Simulations and tests build their contexts
        through ``Cell.from_summary``, which is an exact representation rather
        than a second path.
        """
        if self.model_ctx is None or self.floor_ctx is None or self.ceiling_ctx is None:
            return None
        m = np.asarray(self.model_ctx, dtype=float)
        f = np.asarray(self.floor_ctx, dtype=float)
        c = np.asarray(self.ceiling_ctx, dtype=float)
        if not (m.shape == f.shape == c.shape):
            raise ValueError(
                f"eps={self.eps:g}: model/floor/ceiling context arrays have "
                f"shapes {m.shape}/{f.shape}/{c.shape}. They are three readouts "
                "of the SAME half-B contexts and must be aligned elementwise.")
        if m.ndim != 1 or m.size < 2:
            raise ValueError(f"eps={self.eps:g}: need >= 2 half-B contexts, got {m.size}")
        return m, f, c

    @classmethod
    def from_summary(cls, eps: float, *, y: float, y_se: float,
                     n_contexts: int = 500, **kw) -> "Cell":
        """A cell whose half-B readout is summarised by (y, SE_y).

        NOT a second y-path: this builds per-context arrays that reproduce the
        given y and SE_y EXACTLY under the F.2d.6 linearization, so the same
        code computes the same quantities from the same place. With a constant
        floor of 0 and ceiling of 1 the influence value collapses to
        ``u_i = model_i - y``, so a symmetric two-point model curve with
        half-width ``y_se * sqrt(n - 1)`` has exactly the requested SE.

        For simulations and tests, where the estimand is (y, SE_y) by
        construction and there are no real contexts to carry.
        """
        if n_contexts < 2 or n_contexts % 2:
            raise ValueError("n_contexts must be even and >= 2")
        if not (y_se > 0) or not np.isfinite(y_se):
            raise ValueError("y_se must be finite and positive")
        half = n_contexts // 2
        delta = y_se * math.sqrt(n_contexts - 1)
        model = [y + delta] * half + [y - delta] * half
        return cls(eps=eps, model_ctx=model, floor_ctx=[0.0] * n_contexts,
                   ceiling_ctx=[1.0] * n_contexts, **kw)

    @property
    def n_contexts(self) -> int | None:
        ctx = self._ctx()
        return None if ctx is None else int(ctx[0].size)

    @property
    def fidelity_floor(self) -> float | None:
        """Panel mean of the dose-0 curve. Derived, never stored: the floor is a
        summary of half B, and storing it beside the contexts it summarises is
        how the two drift apart."""
        ctx = self._ctx()
        return None if ctx is None else float(ctx[1].mean())

    @property
    def fidelity_ceiling(self) -> float | None:
        ctx = self._ctx()
        return None if ctx is None else float(ctx[2].mean())

    @property
    def calib_gain(self) -> float | None:
        """How much readout there is to have at this eps. Zero means the
        instrument has no gain here, so no fidelity is definable -- a statement
        about eps, not about the model."""
        ctx = self._ctx()
        return None if ctx is None else float(ctx[2].mean() - ctx[1].mean())

    def _influence(self) -> np.ndarray | None:
        """F.2d.6 linearization. a_i = model_i - floor_i, b_i = ceiling_i -
        floor_i, u_i = (a_i - y * b_i) / mean(b). Carries the floor/ceiling
        correlation that treating the denominator as fixed would drop."""
        ctx = self._ctx()
        if ctx is None:
            return None
        m, f, c = ctx
        a, b = m - f, c - f
        bbar = float(b.mean())
        if abs(bbar) < CALIB_MIN_GAIN:
            return None
        return (a - (float(a.mean()) / bbar) * b) / bbar

    @property
    def fidelity_norm(self) -> float | None:
        """Fraction of achievable: RATIO OF PANEL MEANS (F.2d.6), never the mean
        of per-context ratios. Per-context ratios have unbounded variance
        wherever a context's gain is near zero, and their mean estimates a
        different functional."""
        ctx = self._ctx()
        if ctx is None:
            return None
        m, f, c = ctx
        g = float(c.mean() - f.mean())
        if g < CALIB_MIN_GAIN:
            return None
        return float(m.mean() - f.mean()) / g

    @property
    def fidelity_norm_se(self) -> float | None:
        """SE of y from the iid half-B contexts (F.2d.1 reads this one)."""
        u = self._influence()
        if u is None or u.size < 2:
            return None
        se = float(np.std(u, ddof=1)) / math.sqrt(u.size)
        return se if np.isfinite(se) and se > 0.0 else None

    @property
    def fidelity_norm_se_seed(self) -> float | None:
        """Across-SEED SE of the normalized fidelity. Reported as a diagnostic
        beside SE_y; it does NOT weight the gate. F.2 measured the context
        component at 3.1x-12.5x this one, which is why F.2d.1 reads contexts."""
        g = self.calib_gain
        if g is None or g < CALIB_MIN_GAIN or len(self.fidelity_seeds) < 2:
            return None
        fl = self.fidelity_floor
        ns = [(v - fl) / g for v in self.fidelity_seeds]
        return float(np.std(ns, ddof=1)) / math.sqrt(len(ns))

    @property
    def has_deficit(self) -> bool:
        return self.deficit_exact is not None

    def axis_value(self, axis: str) -> float | None:
        """The x-axis coordinate under either convention.

        Both are oriented so that LARGER means MORE identified, which is what
        makes gate (c) a positive-association test on either axis.
        """
        if axis == PRIMARY_AXIS:
            return self.deficit_exact
        if axis == SECONDARY_AXIS:
            return None if self.width_exact is None else 1.0 - self.width_exact
        raise ValueError(f"unknown axis {axis!r}")

    def deficit_ci(self, z: float = NEAR_TIE_Z) -> tuple[float, float] | None:
        if self.deficit_exact is None or self.deficit_se is None:
            return None
        return (self.deficit_exact - z * self.deficit_se,
                self.deficit_exact + z * self.deficit_se)

    @property
    def has_fidelity(self) -> bool:
        """F.2d.7: a cell carries a fidelity only where the instrument has
        MEASURED gain and the panel yields a finite positive SE.

        At eps=0 the ceiling equals the floor by the Gaussian order-invariance
        identity, so this is False BY CONSTRUCTION and eps=0 leaves the fidelity
        series without any decision being taken about it. It carries the
        null-type claim instead (reproduce-uniform), which is a stronger
        statement than a fidelity value there would have been. F.2d.7 extends
        the same treatment to any OTHER cell whose measured gain collapses --
        the case not known in advance -- and ``survivorship`` surfaces every
        exit with its measured gain so none of them is silent.
        """
        return (self.fidelity_norm is not None
                and self.fidelity_norm_se is not None)

    def survivorship(self) -> dict:
        """Why this cell is in or out of the fidelity series (F.2d.7)."""
        out = {"eps": self.eps, "n_contexts": self.n_contexts,
               "gain": self.calib_gain, "survives": self.has_fidelity}
        if self.has_fidelity:
            out["reason"] = "measured gain above CALIB_MIN_GAIN with a finite SE"
            return out
        if self._ctx() is None:
            out["reason"] = "no half-B per-context readout on disk"
        elif self.calib_gain is not None and self.calib_gain < CALIB_MIN_GAIN:
            out["reason"] = (f"measured ceiling - floor = {self.calib_gain:.3e} < "
                             f"{CALIB_MIN_GAIN}: the instrument has no gain here, "
                             "so no fidelity is definable")
            if abs(self.eps) < 1e-12:
                out["reason"] += " (expected at eps=0 by the order-invariance identity)"
        else:
            out["reason"] = "the half-B panel yields no finite positive SE for y"
        return out

    @property
    def regret(self) -> float | None:
        return float(np.mean(self.regret_seeds)) if self.regret_seeds else None

    @property
    def fidelity(self) -> float | None:
        """RAW readout panel mean, in the readout's own units. Under F.2c this
        is confounded by instrument gain and is NOT what gate (c) reads; use
        ``fidelity_norm``. Kept because the raw series is what
        ``calibration_leakage`` reports against."""
        ctx = self._ctx()
        return None if ctx is None else float(ctx[0].mean())

    def regret_ci(self, z: float = 1.96) -> tuple[float, float] | None:
        """Panel-level CI, or None when the cell cannot support one.

        E.3: a per-panel negative is legitimate, so this CI is used ONE-SIDED
        against the gate -- only its upper bound matters. That makes a too-small
        SE anti-conservative in exactly the direction that matters, which is why
        two fallbacks are gone.

        The panel SE is the estimand's SE and is what F.1 asks for. The
        across-seed SE is a LAST RESORT and is the quantity E.3 retired: F.2
        measures the panel component at 3.1x to 12.5x it, so a CI built from the
        across-seed spread alone is several times too narrow. It is used only
        when no panel SE is on the artifact, and `panel_se_used` records which.

        Returning `se = 0.0` for a single seed with no panel SE, as this used to,
        gave a ZERO-WIDTH CI whose upper bound is the point estimate -- so gate
        (a) passed on a bare number with no uncertainty at all. That is now None,
        and a cell with no CI is not evaluable rather than passing.
        """
        if not self.regret_seeds:
            return None
        se = self.regret_panel_se
        if se is None:
            if len(self.regret_seeds) < 2:
                return None
            se = float(np.std(self.regret_seeds, ddof=1)) / np.sqrt(len(self.regret_seeds))
        if not np.isfinite(se) or se <= 0.0:
            return None
        m = self.regret
        return (m - z * se, m + z * se)

    @property
    def panel_se_used(self) -> bool:
        """Whether regret_ci rests on the panel SE rather than the across-seed
        fallback. Reported, so a CI several times too narrow cannot pass for the
        estimand's own."""
        return self.regret_panel_se is not None

    def fidelity_ci(self, z: float = 1.96) -> tuple[float, float] | None:
        """CI on the NORMALIZED fidelity from the half-B contexts, which is what
        gate (c) reads.

        Deliberately not the raw-y CI. Under F.2c the raw readout is confounded
        by instrument gain, so handing a caller a raw CI from a method with this
        name is the same failure mode as a retired artifact staying loadable:
        it is available, it looks right, and it answers a question nobody meant
        to ask. ``fidelity`` remains as the raw mean and is named that way.
        """
        y, se = self.fidelity_norm, self.fidelity_norm_se
        if y is None or se is None:
            return None
        return (y - z * se, y + z * se)


def _rho_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    """Rho given midranks already formed. Split out so the bootstrap does not
    re-rank a FIXED x ten thousand times per association; the arithmetic is
    identical to ``_spearman``."""
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with the preregistered tie convention (F.2b).

    Ties take MIDRANKS (``SPEARMAN_TIE_METHOD``). This is what
    ``scipy.stats.spearmanr`` does, but the ranks are formed explicitly here so
    the preregistered rule does not silently depend on a library default. With
    midranks, rho is the Pearson correlation of the ranks.
    """
    if len(x) < 3:
        return float("nan")
    return _rho_from_ranks(stats.rankdata(x, method=SPEARMAN_TIE_METHOD),
                           stats.rankdata(y, method=SPEARMAN_TIE_METHOD))


def adjacent_resolution(cells: list[Cell], z: float = NEAR_TIE_Z) -> list[dict]:
    """Which adjacent rungs the dial actually resolves (F.2b near-tie clause).

    Cells are ordered by the primary axis and each neighbouring pair is tested
    for CI overlap. An overlapping pair is UNRESOLVED. Per the binding clause
    this is a precision statement and never fails gate (c) by itself; it is
    reported so a reader can see which part of the dial the instrument can and
    cannot separate.
    """
    have = [c for c in cells if c.has_deficit and c.deficit_se is not None]
    have.sort(key=lambda c: c.deficit_exact)
    out = []
    for lo, hi in zip(have, have[1:]):
        gap = hi.deficit_exact - lo.deficit_exact
        sep = float(np.hypot(lo.deficit_se, hi.deficit_se))
        out.append({
            "eps_lo": lo.eps, "eps_hi": hi.eps,
            "gap": float(gap),
            "z_gap": float(gap / sep) if sep > 0 else float("inf"),
            # A zero separation means no SE was measured, which is not the
            # same as a resolved pair. Fail closed.
            "resolved": bool(sep > 0.0 and gap >= z * sep),
        })
    return out


def _isotonic_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Signed strength of the best monotone increasing fit: the fitted range,
    signed by whether an increasing fit beats a decreasing one."""
    from sklearn.isotonic import IsotonicRegression

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    up = IsotonicRegression(increasing=True).fit_transform(xs, ys)
    dn = IsotonicRegression(increasing=False).fit_transform(xs, ys)
    sse_up = float(np.sum((ys - up) ** 2))
    sse_dn = float(np.sum((ys - dn) ** 2))
    rng = float(up.max() - up.min())
    return rng if sse_up <= sse_dn else -float(dn.max() - dn.min())


def calibration_leakage(cells: list[Cell]) -> dict:
    """How much of a RAW fidelity rise is the instrument, not the model (F.2c).

    The floor curve is an untrained model pushed through the same readout. Its
    true structural tracking is zero at every eps by construction, so any rise
    in it across eps is pure instrument gain. Reported as the Spearman of the
    floor against the primary axis, plus the fraction of the model's raw rise
    that the floor alone accounts for.

    A floor that rises monotonically with the axis is the finding, not a nuisance:
    it says a raw-y association would have been confounded by exactly that much,
    and it is the quantitative form of the Spearman +1.000 anchor-separation
    result that motivated F.2c.
    """
    have = [c for c in cells if c.fidelity_floor is not None
            and c.axis_value(PRIMARY_AXIS) is not None]  # floor is derived from half B
    if len(have) < 3:
        return {"measured": False, "n_cells": len(have),
                "reason": "fewer than 3 cells carry a calibration floor"}
    have.sort(key=lambda c: c.axis_value(PRIMARY_AXIS))
    x = np.array([c.axis_value(PRIMARY_AXIS) for c in have], dtype=float)
    fl = np.array([c.fidelity_floor for c in have], dtype=float)
    out = {"measured": True, "n_cells": len(have),
           "eps_cells": [c.eps for c in have],
           "floor": [float(v) for v in fl],
           "floor_spearman_vs_axis": _spearman(x, fl),
           "floor_range": float(fl.max() - fl.min())}
    raw = [c for c in have if c.fidelity is not None]
    if len(raw) >= 3:
        rw = np.array([c.fidelity for c in raw])
        rr = float(rw.max() - rw.min())
        out["raw_range"] = rr
        out["floor_share_of_raw_rise"] = (float(out["floor_range"] / rr)
                                          if rr > 0 else None)
    return out


def axis_comparison(cells: list[Cell]) -> dict:
    """How the two x-axes relate — WITHOUT pretending they corroborate.

    Both ``_spearman`` and ``_isotonic_slope`` depend on x only through its rank
    order, and the bootstrap resamples y alone from a fixed seed. So whenever
    the two axes rank the cells the same way, the primary and secondary
    association blocks are bit-identical. Printing both as though two
    measurements agreed would misrepresent one statistic computed twice.

    What is actually informative is whether the axes ORDER the cells the same
    way, because that is the only channel through which a re-axing can move the
    gate at all. That is what this reports.
    """
    usable = [c for c in cells
              if c.axis_value(PRIMARY_AXIS) is not None
              and c.axis_value(SECONDARY_AXIS) is not None]
    if len(usable) < 2:
        return {"comparable": False, "n_cells": len(usable)}
    xp = np.array([c.axis_value(PRIMARY_AXIS) for c in usable], dtype=float)
    xs = np.array([c.axis_value(SECONDARY_AXIS) for c in usable], dtype=float)
    rp = stats.rankdata(xp, method=SPEARMAN_TIE_METHOD)
    rs = stats.rankdata(xs, method=SPEARMAN_TIE_METHOD)
    return {
        "comparable": True,
        "n_cells": len(usable),
        "eps_cells": [c.eps for c in usable],
        "x_primary": [float(v) for v in xp],
        "x_secondary": [float(v) for v in xs],
        "rank_primary": [float(v) for v in rp],
        "rank_secondary": [float(v) for v in rs],
        "rank_orders_identical": bool(np.array_equal(rp, rs)),
    }


def permutation_p(x, y) -> float:
    """One-sided EXACT permutation p for the rank association.

    The preregistered E.4c bootstrap resamples SEEDS WITHIN cells while holding
    the cell assignment fixed, so it measures whether rho is stably positive
    GIVEN this assignment and never whether the assignment could have arisen by
    chance. Measured against a null where fidelity is independent of the axis,
    that gate fires 40-48% of the time against a nominal 2.5%; this statistic
    fires 2.0% (5 cells) and 5.0% (4 cells).

    Reported alongside the bootstrap. It does NOT gate: E.4c is a signed clause
    and changing what decides BRANCH 1 is an amendment, not a bug fix.

    With n cells the exact one-sided floor is 1/n!: 0.0083 at 5 cells, but
    0.0417 at 4 -- so at 4 cells only a PERFECT ordering can clear alpha = 0.05.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = _spearman(x, y)
    if not np.isfinite(obs):
        return float("nan")
    rx = stats.rankdata(x, method=SPEARMAN_TIE_METHOD)
    ry = stats.rankdata(y, method=SPEARMAN_TIE_METHOD)
    hits = tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        # permuting y permutes its ranks; no re-ranking needed
        if _rho_from_ranks(rx, ry[list(perm)]) >= obs - 1e-12:
            hits += 1
    return hits / tot


def weighted_slope(x, y, se_y) -> dict:
    """F.2d.1: one-sided weighted least-squares slope test.

    WLS of y on x with weights 1 / SE_y^2. ``H0: slope <= 0``, one-sided p from
    the t distribution with ``df = k - 2``.

    The df in the signed clause fixes the variance convention: ``t`` on ``k - 2``
    degrees of freedom is the ESTIMATED-SCALE model, ``Var(b) = s^2 / Sxx`` with
    ``s^2`` the weighted residual mean square. A known-variance model would take
    ``Var(b) = 1 / Sxx`` and refer to a normal. Estimated scale is the more
    demanding of the two whenever the cells scatter about the line by more than
    their SEs predict, which is the case that should be demanding.

    Errors in x attenuate the slope toward zero (F.2d.1). For a gate that fires
    only on a POSITIVE slope that is conservative, so it is accepted rather than
    corrected; correcting it would loosen the gate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se_y, dtype=float)
    k = x.size
    if k < 3:
        return {"evaluable": False, "reason": f"{k} cells; a slope needs >= 3"}
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        return {"evaluable": False,
                "reason": "a cell reports a non-positive or non-finite SE_y, so "
                          "its inverse-variance weight is undefined"}
    w = 1.0 / np.square(se)
    W = float(w.sum())
    xb = float((w * x).sum() / W)
    yb = float((w * y).sum() / W)
    dx = x - xb
    Sxx = float((w * dx * dx).sum())
    if Sxx <= 0.0:
        return {"evaluable": False,
                "reason": "the x series is constant across cells; the slope is "
                          "undefined and this is an instrument failure, not a "
                          "capture-without-tracking result"}
    b = float((w * dx * (y - yb)).sum() / Sxx)
    a = yb - b * xb
    resid = y - (a + b * x)
    df = k - 2
    s2 = float((w * resid * resid).sum() / df)
    se_b = math.sqrt(s2 / Sxx) if s2 > 0.0 else 0.0
    perfect = se_b == 0.0
    if perfect:
        # Zero weighted residual: the cells lie exactly on a line. Under the
        # estimated-scale model this is infinite evidence, which is a numerical
        # cliff rather than a scientific result. It is FLAGGED loudly instead of
        # being smoothed over, because on real half-B panels it cannot happen
        # and its appearance would mean the y series is synthetic or degenerate.
        tstat = math.inf if b > 0 else (-math.inf if b < 0 else 0.0)
        pval = 0.0 if b > 0 else (1.0 if b < 0 else 0.5)
    else:
        tstat = b / se_b
        pval = float(stats.t.sf(tstat, df))
    return {"evaluable": True, "slope": b, "intercept": float(a),
            "slope_se": float(se_b), "t": float(tstat), "df": int(df),
            "p_one_sided": float(pval), "chi2_resid": float(s2 * df),
            "scale_s2": s2, "perfect_fit": bool(perfect),
            "weighted_x_mean": xb, "weighted_y_mean": yb,
            "passes": bool(np.isfinite(b) and b > 0.0 and pval <= ALPHA)}


def extreme_cell_contrast(cells: list[Cell], axis: str = PRIMARY_AXIS) -> dict:
    """F.2d.4: BRANCH 1's minimum evidence when fewer than 4 cells survive.

    The weakest against the strongest surviving cell by x, one-sided z on dy
    with a 95% CI that must exclude zero. The two cells sit at different eps and
    therefore on DIFFERENT panels (the world carries eps), so their SEs add in
    quadrature without a covariance term.
    """
    usable = [c for c in cells if c.has_fidelity and c.axis_value(axis) is not None]
    if len(usable) < 2:
        return {"evaluable": False, "n_cells": len(usable),
                "reason": "fewer than 2 surviving cells carry both an x and a y"}
    usable.sort(key=lambda c: c.axis_value(axis))
    lo, hi = usable[0], usable[-1]
    dy = hi.fidelity_norm - lo.fidelity_norm
    se = math.hypot(hi.fidelity_norm_se, lo.fidelity_norm_se)
    if not (se > 0.0) or not np.isfinite(se):
        return {"evaluable": False, "reason": "the contrast SE is not positive"}
    z = dy / se
    return {"evaluable": True, "axis": axis,
            "eps_lo": lo.eps, "eps_hi": hi.eps,
            "x_lo": lo.axis_value(axis), "x_hi": hi.axis_value(axis),
            "y_lo": lo.fidelity_norm, "y_hi": hi.fidelity_norm,
            "delta_y": float(dy), "delta_se": float(se), "z": float(z),
            "ci": [float(dy - CONTRAST_Z * se), float(dy + CONTRAST_Z * se)],
            "p_one_sided": float(stats.norm.sf(z)),
            "passes": bool(dy - CONTRAST_Z * se > 0.0)}


def survivorship_report(cells: list[Cell]) -> list[dict]:
    """F.2d.7: every cell's in/out status with its measured gain. A cell never
    leaves the fidelity series silently."""
    return [c.survivorship() for c in sorted(cells, key=lambda c: c.eps)]


def association(cells: list[Cell], axis: str = PRIMARY_AXIS) -> dict:
    """Association between exact identifiability and calibrated model fidelity.

    The gate is F.2d.1's one-sided weighted slope of y on x, weights 1 / SE_y^2,
    with the per-cell SEs coming from the iid half-B contexts. The E.4c
    bootstrap that used to live here is deleted, not bypassed (F.2d.3).

    ``axis`` selects the x convention: ``PRIMARY_AXIS`` is F.2b's panel mean
    order-marginal entropy deficit, ``SECONDARY_AXIS`` is 1 - LP width and is
    reported for the appendix only. Both are oriented larger = more identified.

    Below ``MIN_CELLS_PREFERRED`` cells the slope is still computed and reported
    but the mode is DESCRIPTIVE (F.2d.4): the gate then reads the extreme-cell
    contrast instead, and ``passes`` reflects whichever is in force.
    """
    usable = [c for c in cells
              if c.has_fidelity and c.axis_value(axis) is not None]
    exits = survivorship_report(cells)
    # A cell with a fidelity but NO x is excluded from the association while
    # survivorship affirmatively reports it as surviving, which lets the report
    # claim a cell survived a series it was excluded from -- and it silently
    # demotes the gate a cell at a time. Record the exclusion on the cell.
    no_x = [c.eps for c in cells if c.has_fidelity and c.axis_value(axis) is None]
    for row in exits:
        row["in_association"] = bool(row["survives"] and row["eps"] not in no_x)
        if row["survives"] and not row["in_association"]:
            row["excluded_reason"] = (
                f"carries a calibrated fidelity but no {axis} coordinate, so it "
                "cannot enter the association")
    if len(usable) < MIN_CELLS:
        missing_axis = len(no_x)
        reason = (f"{len(usable)} cells with a defined fidelity and a {axis} "
                  f"coordinate < MIN_CELLS={MIN_CELLS}. Note F.2c: eps=0 leaves "
                  "the fidelity series by construction (ceiling == floor), so "
                  "4 surviving cells is the ceiling, not the target")
        if missing_axis:
            reason += f" ({missing_axis} had fidelity but no {axis} coordinate)"
        return {"evaluable": False, "n_cells": len(usable), "axis": axis,
                "reason": reason, "survivorship": exits,
                "cells_without_axis": no_x}

    usable.sort(key=lambda c: c.axis_value(axis))
    x = np.array([c.axis_value(axis) for c in usable], dtype=float)
    y = np.array([c.fidelity_norm for c in usable], dtype=float)
    se_y = np.array([c.fidelity_norm_se for c in usable], dtype=float)

    fit = weighted_slope(x, y, se_y)
    if not fit["evaluable"]:
        return {"evaluable": False, "n_cells": len(usable), "axis": axis,
                "reason": fit["reason"], "survivorship": exits}

    # Leave-one-cell-out on the SLOPE, as a robustness read on whether one rung
    # carries the fit. Reported, never gating.
    jack = []
    for i in range(len(usable)):
        keep = [q for q in range(len(usable)) if q != i]
        if len(keep) >= 3:
            f2 = weighted_slope(x[keep], y[keep], se_y[keep])
            if f2["evaluable"]:
                jack.append((usable[i].eps, f2["slope"], f2["p_one_sided"]))

    contrast = extreme_cell_contrast(usable, axis=axis)
    descriptive = len(usable) < MIN_CELLS_PREFERRED
    # F.2d.4: "gate_passes in descriptive mode reads the conjunction, and both
    # must pass." Conjunct 1 is the extreme-cell contrast AS IMPLEMENTED -- a 95%
    # CI excluding zero, CONTRAST_Z = 1.96, which is stricter than the one-sided
    # alpha the option simulation used. Conjunct 2 is F.2d.1's descriptive
    # weighted slope on the same cells and the same inverse-variance weights,
    # positive with one-sided p <= ALPHA from t on k - 2 df.
    #
    # The slope conjunct is what carries the calibration. Measured, the contrast
    # ALONE runs a false-positive rate of 0.33525 against a nominal 0.05 at
    # cell-level dispersion sd 0.10, because it reads only the two endpoints and
    # dispersion the SEs do not know about passes straight through it; the slope
    # stayed between 0.04675 and 0.05400 on the same rows. A conjunction cannot
    # fire more often than its calibrated conjunct, so the composite is bounded
    # at or below the slope's rate under every null tested.
    gate_passes = ((bool(fit["passes"]) and bool(contrast.get("passes")))
                   if descriptive else bool(fit["passes"]))

    out = {
        "evaluable": True,
        "axis": axis,
        "mode": "descriptive" if descriptive else "gating",
        "n_cells": len(usable),
        "eps_cells": [c.eps for c in usable],
        "x": [float(v) for v in x],
        "x_se": [None if c.deficit_se is None else float(c.deficit_se)
                 for c in usable],
        "y": [float(v) for v in y],
        "y_se": [float(v) for v in se_y],
        "y_se_seed": [c.fidelity_norm_se_seed for c in usable],
        "n_contexts": [c.n_contexts for c in usable],
        # --- F.2d.1, the gate ---
        "slope": fit["slope"], "slope_se": fit["slope_se"],
        "t": fit["t"], "df": fit["df"], "p_one_sided": fit["p_one_sided"],
        "alpha": ALPHA, "perfect_fit": fit["perfect_fit"],
        "scale_s2": fit["scale_s2"],
        "slope_passes": bool(fit["passes"]),
        # --- F.2d.2, the companion; reported, NEVER gating ---
        "permutation_p": permutation_p(x, y),
        "permutation_p_floor": 1.0 / math.factorial(len(usable)),
        # --- descriptive rank statistics, for the figure captions ---
        "spearman": _spearman(x, y),
        "isotonic_signed_range": _isotonic_slope(x, y),
        "jackknife_slope": [[float(e), float(s), float(q)] for e, s, q in jack],
        "jackknife_slope_min": float(min(s for _, s, _ in jack)) if jack else None,
        "jackknife_slope_max": float(max(s for _, s, _ in jack)) if jack else None,
        # --- F.2d.4 ---
        "extreme_contrast": contrast,
        "gate_passes": gate_passes,
        "survivorship": exits,
        "cells_without_axis": no_x,
    }
    if no_x:
        out["reason_partial"] = (
            f"{len(no_x)} cell(s) carry a fidelity but no {axis} coordinate and "
            f"are EXCLUDED from this association: eps {no_x}")
    # F.2's post-W2 quality flag. Reported, not a gate: it is a fact about
    # precision a reader is entitled to before weighing the slope, and turning
    # it into a pass/fail after the sizing rule has fired would be the cap
    # argument again in a second costume.
    y_range = float(y.max() - y.min())
    widest = float(np.max(1.96 * se_y))
    out["precision_flag"] = bool(y_range > 0 and widest > 0.5 * y_range)
    out["y_range"] = y_range
    out["widest_ci_halfwidth"] = widest
    return out


def classify(cells: list[Cell], regret_gate: float = REGRET_GATE) -> dict:
    """The Amendment E.4 verdict. Defaults to a non-affirmative branch."""
    ev: list[str] = []

    # --- (a) predictive gate at every cell ---------------------------------
    scored = [c for c in cells if c.regret is not None]
    if not scored:
        return {"verdict": BRANCH_4, "amendment": "E.4",
                "evidence": ["no cell carries an expected regret"],
                "association": {"evaluable": False}}

    # E.4 BRANCH 1 (a) is "expected regret below the gate at EVERY cell", and a
    # cell that supplies x and y to gate (c) is emphatically one of them. Testing
    # only the cells that HAPPEN to carry a regret let a missing file turn
    # BRANCH_3 into BRANCH_1: the regret and the fidelity are separate artifacts
    # written by separate jobs, so one landing without the other is the ordinary
    # case. The cell would contribute a degree of freedom to the affirmative gate
    # while being exempt from the predictive one, behind an evidence line reading
    # "gate (a) holds at all N scored cells".
    in_assoc = [c for c in cells if c.has_fidelity]
    unscored = [c for c in in_assoc if c.regret is None]
    if unscored:
        for c in unscored:
            ev.append(f"gate (a) NOT EVALUABLE at eps={c.eps:g}: the cell carries "
                      "a calibrated fidelity and would enter gate (c), but no "
                      "expected regret. A cell cannot be exempt from the "
                      "predictive gate and count toward the association.")
        return {"verdict": BRANCH_4, "amendment": "E.4/F.2b/F.2d", "evidence": ev,
                "association": {"evaluable": False,
                                "reason": "cells without a regret would enter it"},
                "association_secondary": {"evaluable": False},
                "unscored_cells": [c.eps for c in unscored],
                "survivorship": survivorship_report(cells)}

    failed = []
    no_ci = []
    for c in scored:
        ci = c.regret_ci()
        if ci is None:
            # regret seeds present but no usable CI: a single seed with no panel
            # SE. This used to be skipped by an `if ci is not None` guard, which
            # meant a cell with no measurable uncertainty could not FAIL gate (a).
            no_ci.append(c.eps)
            continue
        # One-sided: the gate asks whether regret is BELOW the bound. E.3 makes
        # a negative point estimate legitimate, so only the upper bound matters.
        if ci[1] >= regret_gate:
            failed.append((c.eps, c.regret, ci))
    if no_ci:
        for e in no_ci:
            ev.append(f"gate (a) NOT EVALUABLE at eps={e:g}: the cell has a "
                      "regret point estimate but no panel SE and fewer than two "
                      "seeds, so its CI has no width. A gate applied to a bare "
                      "number is not a gate.")
        return {"verdict": BRANCH_4, "amendment": "E.4/F.2b/F.2d", "evidence": ev,
                "association": {"evaluable": False,
                                "reason": "a scored cell has no usable regret CI"},
                "association_secondary": {"evaluable": False},
                "cells_without_regret_ci": no_ci,
                "survivorship": survivorship_report(cells)}
    for c in scored:
        ci = c.regret_ci()
        ev.append(f"eps={c.eps:g}: expected regret {c.regret:+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] vs gate {regret_gate}"
                  + ("" if c.panel_se_used else
                     "  [ACROSS-SEED SE, no panel SE on the artifact: F.2 "
                     "measures the panel component at 3.1x-12.5x this, so this "
                     "CI is several times too narrow and the one-sided gate "
                     "errs toward passing]"))

    assoc = association(cells, axis=PRIMARY_AXIS)
    assoc2 = association(cells, axis=SECONDARY_AXIS)
    surviv = survivorship_report(cells)
    resolution = adjacent_resolution(cells)
    axcmp = axis_comparison(cells)
    leak = calibration_leakage(cells)

    if failed:
        for eps, m, ci in failed:
            ev.append(f"GATE (a) FAILS at eps={eps:g}: regret CI upper bound "
                      f"{ci[1]:+.4f} >= {regret_gate}")
        return {"verdict": BRANCH_3, "amendment": "E.4/F.2b/F.2d", "evidence": ev,
                "association": assoc, "association_secondary": assoc2,
                "adjacent_resolution": resolution,
                "axis_comparison": axcmp,
                "calibration_leakage": leak,
                "survivorship": surviv}

    ev.append(f"gate (a) holds at all {len(scored)} scored cells "
              f"(ladder carries {len(cells)}; {len(in_assoc)} enter the "
              "association, all of them scored)")

    # --- (b) the eps=0 null -------------------------------------------------
    # F.2c. eps=0 no longer carries a fidelity: the six order-conditional
    # predictives coincide there by the Gaussian order-invariance identity, so
    # the readout's ceiling equals its floor and no fidelity is definable. Gate
    # (b) therefore stops being a fidelity test -- which it could never fail,
    # since the readout statistic is identically zero there for ANY model -- and
    # becomes a null-TYPE claim: the model must reproduce the uniform order
    # posterior. That is a stronger statement than a fidelity value would have
    # been, and it is carried in the reproduce-uniform series, not here.
    zero = next((c for c in cells if abs(c.eps) < 1e-12), None)
    null_ok = True
    if zero is not None:
        g = zero.calib_gain
        if g is not None and g >= CALIB_MIN_GAIN:
            # The identity did not hold, which means the readout or the world is
            # not what F.2c assumes. That is an instrument finding.
            null_ok = False
            ev.append(f"gate (b) VIOLATED: the eps=0 readout has gain "
                      f"{g:.3e} >= {CALIB_MIN_GAIN}, but the Gaussian "
                      "order-invariance identity says ceiling == floor there. "
                      "The instrument is not measuring what F.2c assumes.")
        else:
            ev.append("gate (b): eps=0 is an x-only anchor (ceiling == floor by "
                      "the Gaussian order-invariance identity); the null-type "
                      "claim is carried by the reproduce-uniform series, not by "
                      "a fidelity value (F.2c)")
    else:
        ev.append("gate (b): no eps=0 cell present")

    # --- (c) association ----------------------------------------------------
    if not assoc.get("evaluable"):
        ev.append(f"gate (c) NOT EVALUABLE on the {PRIMARY_AXIS} axis: "
                  f"{assoc.get('reason')}")
        for s in surviv:
            if not s["survives"]:
                ev.append(f"F.2d.7: eps={s['eps']:g} is OUT of the fidelity "
                          f"series -- {s['reason']}")
        return {"verdict": BRANCH_4, "amendment": "E.4/F.2b/F.2d", "evidence": ev,
                "association": assoc, "association_secondary": assoc2,
                "adjacent_resolution": resolution,
                "axis_comparison": axcmp,
                "calibration_leakage": leak,
                "survivorship": surviv}

    for s in surviv:
        if not s["survives"]:
            ev.append(f"F.2d.7: eps={s['eps']:g} is OUT of the fidelity series "
                      f"-- {s['reason']}")
    ev.append(f"gate (c), F.2d.1 weighted slope ({PRIMARY_AXIS} axis) over "
              f"{assoc['n_cells']} cells: slope {assoc['slope']:+.4f} "
              f"+/- {assoc['slope_se']:.4f}, t {assoc['t']:+.3f} on df "
              f"{assoc['df']}, one-sided p {assoc['p_one_sided']:.4f} vs alpha "
              f"{ALPHA}")
    if assoc["perfect_fit"]:
        ev.append("F.2d.1 NUMERICAL CLIFF: the weighted residual is exactly zero, "
                  "so the estimated-scale t is infinite. On a real half-B panel "
                  "this cannot happen; treat the y series as synthetic or "
                  "degenerate rather than as infinite evidence.")
    ev.append(f"F.2d.2 companion (reported, NOT gating): exact permutation p "
              f"{assoc['permutation_p']:.4f}, floor "
              f"{assoc['permutation_p_floor']:.4f}; descriptive Spearman "
              f"{assoc['spearman']:+.3f}, isotonic signed range "
              f"{assoc['isotonic_signed_range']:+.4f}")
    if assoc["precision_flag"]:
        ev.append(f"F.2 precision flag: the widest cell CI half-width "
                  f"{assoc['widest_ci_halfwidth']:.4f} exceeds half the "
                  f"cross-cell y range {assoc['y_range']:.4f}. The ladder is "
                  "being read at a resolution the panel does not support; this "
                  "is reported, not gated.")
    if axcmp.get("comparable"):
        if axcmp["rank_orders_identical"]:
            ev.append(
                f"axis comparison: the {PRIMARY_AXIS} and {SECONDARY_AXIS} axes "
                f"rank all {axcmp['n_cells']} cells identically. The secondary "
                "block is therefore a RE-EXPRESSION of the primary one -- the "
                "same y values against a monotone relabelling of x -- and NOT a "
                "second measurement. Under the retired rank gate the two were "
                "bit-identical; under the F.2d.1 slope they differ numerically, "
                "which is worse rather than better, because small disagreement "
                "invites reading them as a replication. They are reported this "
                "way on purpose.")
        else:
            ev.append(
                f"axis comparison: the two axes DISAGREE on cell order "
                f"(primary ranks {axcmp['rank_primary']}, secondary "
                f"{axcmp['rank_secondary']}). This is the only channel through "
                "which the F.2b re-axing can move the gate. Gate on primary; "
                "report both and say which cells reorder.")
    if assoc2.get("evaluable"):
        ev.append(f"appendix ({SECONDARY_AXIS} axis, a RE-EXPRESSION and not a "
                  f"second measurement): slope {assoc2['slope']:+.4f}, "
                  f"one-sided p {assoc2['p_one_sided']:.4f}")
    unresolved = [r for r in resolution if not r["resolved"]]
    if unresolved:
        pairs = ", ".join(f"eps {r['eps_lo']:g}/{r['eps_hi']:g} (z={r['z_gap']:.2f})"
                          for r in unresolved)
        ev.append(f"F.2b near-tie clause: {len(unresolved)} adjacent rung pair(s) "
                  f"UNRESOLVED at z={NEAR_TIE_Z} [{pairs}]. Per the binding clause "
                  "this does not fail gate (c) on its own; the F.2d.1 weighted "
                  "slope remains the gate and the pair is a precision statement "
                  "about the dial, not evidence against tracking. It holds a "
                  "fortiori under the slope, which uses the rung SPACING rather "
                  "than only the order: a near-tie contributes almost nothing to "
                  "Sxx instead of hinging on which side of it two cells fall.")

    if assoc["mode"] == "descriptive":
        con = assoc["extreme_contrast"]
        ev.append(f"F.2d.4 fallback IN FORCE: {assoc['n_cells']} surviving eps>0 "
                  f"cells is below the preferred {MIN_CELLS_PREFERRED}, so the "
                  f"slope above is DESCRIPTIVE (df {assoc['df']}) and BRANCH 1's "
                  "minimum evidence is the extreme-cell contrast.")
        if con.get("evaluable"):
            ev.append(f"F.2d.4 extreme-cell contrast eps={con['eps_lo']:g} -> "
                      f"{con['eps_hi']:g}: dy {con['delta_y']:+.4f} "
                      f"[{con['ci'][0]:+.4f}, {con['ci'][1]:+.4f}], z "
                      f"{con['z']:+.3f}, one-sided p {con['p_one_sided']:.4f}")
        else:
            ev.append(f"F.2d.4 extreme-cell contrast NOT EVALUABLE: "
                      f"{con.get('reason')}")
    if leak.get("measured"):
        share = leak.get("floor_share_of_raw_rise")
        ev.append(
            f"F.2c calibration leakage: the untrained floor moves "
            f"{leak['floor_range']:+.4f} across the dial with Spearman "
            f"{leak['floor_spearman_vs_axis']:+.3f} against the primary axis"
            + (f", accounting for {share:.0%} of the raw fidelity rise"
               if share is not None else "")
            + ". The gate reads fraction-of-achievable, so this is removed from "
              "y; it is reported because it is the size of the confound a raw-y "
              "association would have carried.")
    else:
        ev.append(f"F.2c calibration NOT measured ({leak.get('reason')}); the "
                  "leakage the calibration removes is therefore unquantified")

    passes_c = bool(assoc["gate_passes"])

    payload = {"evidence": ev, "association": assoc,
               "association_secondary": assoc2,
               "adjacent_resolution": resolution,
               "axis_comparison": axcmp,
               "calibration_leakage": leak,
               "survivorship": surviv,
               "amendment": "E.4/F.2b/F.2d", "primary_axis": PRIMARY_AXIS}

    if passes_c and null_ok:
        ev.append("gates (a), (b), (c) all hold")
        return {"verdict": BRANCH_1, **payload}

    if not passes_c:
        which = ("the extreme-cell contrast" if assoc["mode"] == "descriptive"
                 else "the weighted slope")
        ev.append(f"gate (c) fails on {which}: calibrated structural fidelity "
                  "does not track exact average-case identifiability")
        return {"verdict": BRANCH_2, **payload}

    ev.append("gate (c) holds but gate (b) does not: the eps=0 null is violated, "
              "which indicates a fidelity instrument with a nonzero floor")
    return {"verdict": BRANCH_4, **payload}


def apply_written_default(result: dict, today: str) -> dict:
    """E.4's binding default: not evaluable by DEFAULT_DATE -> d=2 headline."""
    out = dict(result)
    if result["verdict"] in (BRANCH_4,) and today >= DEFAULT_DATE:
        out["headline"] = "D2_CORE"
        out["written_default_applied"] = True
        out["evidence"] = list(result["evidence"]) + [
            f"E.4 written default applied on {today} (>= {DEFAULT_DATE}): "
            "d=2 headline auto-selected, eps-dial ships as a pending section"]
    else:
        out["headline"] = {BRANCH_1: "EPS_DIAL", BRANCH_2: "CAPTURE_WITHOUT_TRACKING",
                           BRANCH_3: "NOT_A_PAPER_OUTCOME"}.get(result["verdict"], "PENDING")
        out["written_default_applied"] = False
    return out


_W2_RE = re.compile(r"^w2_eps(?P<tag>\d+p\d+)_s(?P<seed>\d+)$")
_TAG_RE = re.compile(r"^(?P<kind>ident|deficit|fidelity)_eps"
                     r"(?P<tag>\d+p\d+)"
                     r"(?:_(?P<solver>highs|highs_ipm))?$")


def _eps_of(tag: str) -> float:
    return float(tag.replace("p", "."))


def eps_tag(eps: float) -> str:
    """The canonical filename tag for an eps, and the INVERSE of ``_eps_of``.

    Exported so producers stop rolling their own. A hand-rolled
    ``f"{eps:g}".replace(".", "p")`` yields "0" and "1" at the endpoints, which
    ``_TAG_RE`` does not match -- and the loader used to skip a non-matching
    file in silence. That cost the eps=0 gate (b) anchor and the eps=1 top rung
    out of a seven-cell ladder, with no error and a verdict computed on five.
    """
    tag = f"{eps:g}".replace(".", "p")
    if "p" not in tag:
        tag += "p0"
    if _eps_of(tag) != float(eps):
        raise ValueError(f"tag {tag!r} does not round-trip to eps {eps!r}")
    return tag


def cells_from_raw(raw_dir: Path, solver: str | None = None) -> list[Cell]:
    """Assemble cells from a campaign raw/ directory.

    The cell set is the UNION of the tags that carry a primary-axis coordinate,
    a fidelity, or a regret — deliberately NOT the ident glob. Keying existence
    on the ident file would let a missing or unfinished LP drop a rung, and
    under F.2b the LP is the demoted axis: the eps=0.5 simplex cell that could
    not clear a 24h wall must still be able to enter the primary association.

    Fidelity is left EMPTY unless a post-Amendment-E fidelity file is present:
    the pre-Amendment-E artifacts carry no model-side structural quantity, so
    the honest result on them is BRANCH 4 (NOT_EVALUABLE), not a verdict.

    ``solver`` disambiguates directories that hold one ident file per solver
    (``raw/postE/`` has both ``highs`` and ``highs_ipm``). Passing None there
    raises rather than silently emitting two cells per eps.
    """
    widths: dict[str, float] = {}
    seen_solvers: dict[str, set[str]] = {}
    for f in sorted(raw_dir.glob("ident_eps*.json")):
        if f.name.endswith(".env.json"):
            # `cluster/ident.sbatch` writes `ident_eps<tag>.env.json` beside every
            # result: host, solver, numpy/scipy versions, git head. It is
            # PROVENANCE, not a measurement, and it matches this glob only by
            # accident of naming. Skipping it is narrow on purpose -- a sidecar
            # counts as a sidecar only when the measurement it describes is
            # actually here. A lone sidecar means a cell was launched and its
            # result never landed, which is precisely the missing rung this
            # loader exists to catch, so that case falls through and raises.
            if f.with_name(f.name[: -len(".env.json")] + ".json").exists():
                continue
        m = _TAG_RE.match(f.stem)
        if m is None:
            # NEVER skip. A file that matches the glob but not the tag pattern
            # is a live artifact the loader cannot see, which is the F.7 failure
            # with the opposite sign -- and it does not announce itself, because
            # a ladder short one rung looks exactly like a ladder.
            raise ValueError(
                f"{f.name} matches the ident glob but not the tag pattern "
                f"{_TAG_RE.pattern!r}. Rename it with corrected_verdict.eps_tag "
                "rather than leaving a live artifact no loader can read.")
        tag, slv = m["tag"], m["solver"]
        seen_solvers.setdefault(tag, set()).add(slv or "")
        if solver is not None and (slv or "") != solver:
            continue
        doc = json.loads(f.read_text())
        # F.8: fail CLOSED on anything that is not a live width measurement.
        # raw/ still holds the widths E.2 retired under exactly the filenames
        # this glob matches, and raw/ has one solver per tag so nothing else
        # would raise. The retired VALUES are listed in
        # tests/test_retired_constants.py, the only place in the tree allowed to
        # name them; this is the tag that stops the retired FILE. Under F.2b the
        # secondary axis feeds axis_comparison, so a stale secondary silently
        # corrupts the one diagnostic that reports whether the re-axing moved
        # anything -- and a crossover-legacy re-run is tagged DIAGNOSTIC, which
        # this refuses for the same reason.
        require_live(doc, EST_IDENT_WIDTH, f"{f.name} (secondary width axis)")
        if "n_rows" not in doc.get("config", {}):
            raise ValueError(
                f"{f.name} has no n_rows in its config block, so it predates "
                "Amendment F.5 clause 1. Regenerate it, or point --raw at a "
                "post-F directory.")
        if doc["config"]["n_rows"] != HEADLINE_N_ROWS:
            raise ValueError(
                f"{f.name} has n_rows={doc['config']['n_rows']}, headline is "
                f"{HEADLINE_N_ROWS} (F.5 clause 2). The nrows10 set is a valid "
                "sensitivity measurement and belongs in the errata appendix, "
                "not in the ladder.")
        w = doc.get("aggregates", {}).get("Q1", {}).get("avg_width_mean")
        if w is not None:
            widths[tag] = float(w)
    ambiguous = {t: sorted(v) for t, v in seen_solvers.items() if len(v) > 1}
    if ambiguous and solver is None:
        raise ValueError(
            f"{raw_dir} holds multiple solvers per eps ({ambiguous}); pass "
            "solver= explicitly rather than letting the glob pick")

    tags = set(widths)
    for pat, kind in (("deficit_eps*.json", "deficit"), ("fidelity_eps*.json", "fidelity")):
        for f in sorted(raw_dir.glob(pat)):
            m = _TAG_RE.match(f.stem)
            if m is None:
                raise ValueError(
                    f"{f.name} matches the {kind} glob but not the tag pattern "
                    f"{_TAG_RE.pattern!r}. Rename it with "
                    "corrected_verdict.eps_tag rather than leaving a live "
                    "artifact no loader can read.")
            tags.add(m["tag"])
    for f in sorted(raw_dir.glob("w2_eps*_s*.json")):
        m = _W2_RE.match(f.stem)
        if m is None:
            raise ValueError(
                f"{f.name} matches the W2 glob but not the tag pattern "
                f"{_W2_RE.pattern!r}. Rename it with corrected_verdict.eps_tag "
                "rather than leaving a live artifact no loader can read.")
        tags.add(m["tag"])

    cells: list[Cell] = []
    per_cell_hashes: dict[str, dict[str, str]] = {}
    for tag in sorted(tags, key=_eps_of):
        eps = _eps_of(tag)
        width = widths.get(tag)
        # The regret is gate (a)'s only input and gate (a) is what separates
        # BRANCH_3 from BRANCH_1/2, so it gets exactly the checks the other two
        # gate inputs already get. Until F.8 it got none: EST_REGRET_INTASK and
        # EST_REGRET_MIXED were named in artifact_status as "the pair already
        # confused once" and then referenced nowhere, which made the guard dead
        # code while the pre-F artifacts -- the mixed-target series E.3 retired,
        # untagged and therefore UNKNOWN -- fed the gate without complaint.
        regret_seeds: list[float] = []
        regret_panel_se = None
        ses = []
        for it in sorted(raw_dir.glob(f"w2_eps{tag}_s*.json")):
            doc = json.loads(it.read_text())
            require_live(doc, EST_REGRET_INTASK, f"{it.name} (gate (a) regret)")
            art = doc.get("artifact", {})
            if art.get("half") != HALF_B:
                raise ValueError(
                    f"{it.name} is not tagged half {HALF_B}. F.2d.5 scores the "
                    "regret on half B only; a regret computed on the deficit's "
                    "own contexts shares draws with the x-axis.")
            if art.get("n_rows") != HEADLINE_N_ROWS:
                raise ValueError(
                    f"{it.name} has n_rows={art.get('n_rows')}, headline is "
                    f"{HEADLINE_N_ROWS}")
            fin = doc["final"]
            regret_seeds.append(float(fin["in_task_regret"]))
            se = fin.get("in_task_regret_se")
            if se is None:
                raise ValueError(
                    f"{it.name} carries in_task_regret without "
                    "in_task_regret_se. Gate (a) would fall back to the "
                    "across-seed SE, which E.3 retired and F.2 measures at "
                    "3.1x-12.5x too narrow; regenerate the artifact.")
            ses.append(float(se))
            per_cell_hashes.setdefault(tag, {})[f"regret_s{doc['seed']}"] = (
                doc.get("panel", {}).get("panel_sha256"))
        if ses:
            # The PANEL SE is the estimand's SE and is what F.1 asks for. It was
            # sitting in the artifact and never read, so regret_ci fell back to
            # the across-seed SE -- the exact component E.3 retired. On a
            # ONE-SIDED upper-bound gate a too-narrow CI errs toward passing.
            # All seeds share ONE panel, so the panel term is a fixed effect
            # common to them: average it rather than shrinking it by sqrt(n).
            regret_panel_se = float(np.mean(ses))
        # F.2d.5/6. The fidelity artifact must carry the HALF-B per-context
        # readouts for all three curves; a summary would be a second y-path and
        # the two paths would eventually disagree. fidelity_seeds is loaded as
        # the across-seed DIAGNOSTIC only.
        fid_seeds: list[float] = []
        m_ctx = f_ctx = c_ctx = None
        fid_f = raw_dir / f"fidelity_eps{tag}.json"
        if fid_f.exists():
            fj = json.loads(fid_f.read_text())
            require_live(fj, EST_FIDELITY_NORM, f"{fid_f.name} (gate (c) y-axis)")
            if fj.get("half") != HALF_B:
                raise ValueError(
                    f"{fid_f.name} is not tagged half B. F.2d.5 computes y on "
                    "half B only; a readout scored on the deficit's own contexts "
                    "would let one unusually informative context push x and y "
                    "the same way and manufacture the association.")
            if fj.get("n_rows") != HEADLINE_N_ROWS:
                raise ValueError(
                    f"{fid_f.name} has n_rows={fj.get('n_rows')}, headline is "
                    f"{HEADLINE_N_ROWS}")
            m_ctx = fj["model_ctx"]
            f_ctx = fj["floor_ctx"]
            c_ctx = fj["ceiling_ctx"]
            fid_seeds = list(fj.get("raw_panel_mean_seeds", []))
            per_cell_hashes.setdefault(tag, {})["fidelity"] = fj.get("panel_sha256")
        # F.2b primary axis. Absent until the deficit is measured on the scoring
        # panel; a cell without it simply cannot enter the primary association,
        # which is why the honest result on pre-F artifacts is NOT_EVALUABLE.
        dfc = dse = None
        dfc_f = raw_dir / f"deficit_eps{tag}.json"
        if dfc_f.exists():
            dj = json.loads(dfc_f.read_text())
            require_live(dj, EST_ORDER_EVIDENCE, f"{dfc_f.name} (primary x-axis)")
            if dj.get("half") != HALF_A:
                raise ValueError(
                    f"{dfc_f.name} is not tagged half A. F.2d.5 computes the "
                    "deficit on half A only, disjoint from the readouts.")
            dfc = dj.get("deficit_mean")
            dse = dj.get("deficit_se")
            # F.5 clause 2, extended to the primary axis. n_rows is the dominant
            # confound for this estimand: a wrong one moves a cell a full rung
            # while producing an entirely plausible number.
            if dj.get("n_rows") != HEADLINE_N_ROWS:
                raise ValueError(
                    f"{dfc_f.name} has n_rows={dj.get('n_rows')}, headline is "
                    f"{HEADLINE_N_ROWS}")
            per_cell_hashes.setdefault(tag, {})["deficit"] = dj.get("panel_sha256")
        cells.append(Cell(eps=eps,
                          width_exact=None if width is None else float(width),
                          regret_seeds=regret_seeds, fidelity_seeds=fid_seeds,
                          regret_panel_se=regret_panel_se,
                          model_ctx=m_ctx, floor_ctx=f_ctx, ceiling_ctx=c_ctx,
                          deficit_exact=dfc, deficit_se=dse))
    # Each eps has its OWN panel (the world carries eps), so the hashes differ by
    # construction and are recorded rather than compared for equality; what would
    # be wrong is two cells at the SAME eps disagreeing, which the tag keying
    # already prevents.
    # F.2d.5, checked by COMPARISON rather than by counting. The previous
    # version asserted only that the NUMBER of distinct hashes equalled the
    # number of cells, which two cross-wired panels satisfy exactly as well as
    # two correct ones -- and it skipped entirely when an artifact omitted the
    # field, so a hash-less pair passed by being unmeasurable.
    for tag, seen in sorted(per_cell_hashes.items()):
        for kind, h in seen.items():
            if not h:
                raise ValueError(
                    f"the {kind} artifact for eps tag {tag} carries no "
                    "panel_sha256, so nothing can confirm its half came from the "
                    "same panel as the other half's")
        if len(set(seen.values())) > 1:
            raise ValueError(
                f"eps tag {tag}: the halves report different panels {seen}. "
                "F.2d.5 splits ONE panel into two, so a mismatch means x and y "
                "were computed against different draws and the association is "
                "between quantities from different worlds.")
    across = {tag: next(iter(v.values())) for tag, v in per_cell_hashes.items() if v}
    if len(set(across.values())) != len(across):
        raise ValueError(
            f"two eps report the SAME panel hash ({across}); the world carries "
            "eps, so distinct cells cannot share a panel and the ladder would be "
            "one cell repeated")
    return cells


if __name__ == "__main__":
    import argparse
    from datetime import date

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--today", type=str, default=date.today().isoformat())
    a = p.parse_args()
    res = apply_written_default(classify(cells_from_raw(a.raw)), a.today)
    print(json.dumps(res, indent=2))
