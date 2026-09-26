"""Amendment F.2d, T3 — validate the new gate with the instrument that convicted
its predecessor, BEFORE the lock.

The E.4c bootstrap was retired because a null simulation measured it firing
40-48% of the time against a nominal 2.5%. The replacement does not get to skip
that test. Everything here runs before Amendment F is locked, so a failure is a
reason to change the gate rather than a footnote about it.

Four blocks:

  A  NULL simulations. y independent of x, at the measured per-cell SE PATTERN
     and the real rung spacing. Reports the exact false-positive rate.
  B  POWER. A true positive slope injected across a plausible range, on the
     primary ladder, on the secondary ladder, and on a NEAR-TIE SWEEP that holds
     the x range fixed while moving one interior rung toward its neighbour. The
     sweep is the honest test of what a near-tie costs: the primary/secondary
     gap conflates two things, because the two ladders differ in RANGE as well
     as in spacing and a compressed range lowers Sxx on its own.
  C  PERMUTATION FLOORS at k = 4, 5, 6, checked against 1/k! by construction.
  D  NEGATIVE CONTROLS in both directions: the gate must fail on a null-slope
     synthetic and pass on a planted monotone one.
  E  THE F.2d.4 FALLBACK, which is the statistic that decides BRANCH 1 when
     fewer than four cells survive. It is a separate rule with separate
     properties and it is measured separately, reading ``gate_passes`` -- the
     quantity classify() actually consults -- rather than ``slope_passes``.

A note on what "measured per-cell SEs" can mean today. SE_y does not exist yet:
the calibrated fidelity series is what W2 produces. What IS measured is the
in-task panel SE PATTERN across rungs (F.2, base scale, 60 contexts: 0.0068 /
0.0170 / 0.0193 / 0.0238 at eps 0 / 0.1 / 0.25 / 1.0), and under the
estimated-scale convention the WLS t depends on the weights only through their
RATIOS -- multiplying every SE by a constant leaves t exactly unchanged. So the
measured pattern is the part that matters and the scale is swept. That property
is asserted in block A rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pfn_dag_verify.artifact_status import DIAGNOSTIC, stamp          # noqa: E402
from pfn_dag_verify import corrected_verdict as V                      # noqa: E402

# --- the real ladders -------------------------------------------------------
# PRIMARY (F.2b): panel mean normalized order-marginal entropy deficit, measured
# on HALF A of the W2 panel at n = 500 (raw/postF/deficit_w2/).
# SECONDARY (appendix): 1 - Q1 avg_width_mean, post-D1, IPM.
# eps=0 is excluded from both: F.2c takes it out of the fidelity series by
# construction, so it is never a cell the association sees.
EPS = [0.1, 0.25, 0.5, 1.0]
X_PRIMARY = [0.035537, 0.155389, 0.415179, 0.766417]
X_SECONDARY = [1 - 0.437814, 1 - 0.316971, 1 - 0.260840, 1 - 0.125500]
# The post-D6 ladder: if D6 cuts eps=0.1, three cells survive and F.2d.4's
# contrast becomes the gate.
EPS_3 = [0.25, 0.5, 1.0]
X_PRIMARY_3 = [0.155389, 0.415179, 0.766417]
SE_PATTERN_3 = [0.0193, 0.0215, 0.0238]

# The measured in-task panel SE pattern (F.2, 60 contexts). Used as the RELATIVE
# weight pattern; the scale is swept because SE_y itself is not yet measured.
#
# DISCLOSURE, because this is the highest-leverage input here: the eps=0.5 entry
# is INTERPOLATED. That in-task cell has never been run, so its spread is
# unobserved, and the per-cell misreport rows below exist precisely because an
# interpolated weight is the one most likely to be wrong.
SE_PATTERN = [0.0170, 0.0193, 0.0215, 0.0238]      # eps 0.1 / 0.25 / 0.5 / 1.0
SE_INTERPOLATED = {0.5: "no in-task cell has been run at eps=0.5"}
ALPHA = V.ALPHA


def _cells(x, y, se_y, eps=None):
    eps = eps if eps is not None else EPS
    return [V.Cell.from_summary(e, y=float(yy), y_se=float(s), n_contexts=500,
                                deficit_exact=float(xx), deficit_se=1e-4)
            for e, xx, yy, s in zip(eps, x, y, se_y)]


def gate(x, y, se_y, eps=None) -> dict:
    """The F.2d.1 gate exactly as classify() applies it, with no fixture in the
    path that classify would not also use."""
    a = V.association(_cells(x, y, se_y, eps), axis=V.PRIMARY_AXIS)
    return a


# ---------------------------------------------------------------- block A
def null_fpr(n_trials: int, rng, x, se_scale: float, mode: str,
             extra_sd: float = 0.0, se_misreport=1.0) -> dict:
    """FPR under a null where y carries no dependence on x.

    mode:
      'calibrated'  the reported SE is the truth. The textbook case; the t test
                    should land at alpha and anything else is an implementation
                    bug rather than a modelling question.
      'random_cell' each cell additionally carries a cell-level offset that the
                    context SE knows nothing about. This is the realistic threat
                    and it is precisely what the retired bootstrap could not see.
      'misreported' the truth is se_misreport x the reported SE. Pass a VECTOR:
                    a scalar is absorbed exactly by the estimated-scale
                    convention -- it rescales every weight by the same factor,
                    leaves the ratios alone, and s^2 soaks up the rest, so a
                    uniform misreport measures nothing the calibrated row has
                    not already measured. The threat that survives that argument
                    is HETEROGENEOUS misspecification, and worst at a
                    high-leverage cell, where a single overconfident weight can
                    tilt the line without inflating the residual that would have
                    penalised it.
    """
    se_rep = np.array(SE_PATTERN) * se_scale
    se_true = se_rep * np.asarray(se_misreport, dtype=float)
    x = np.asarray(x, dtype=float)
    fired = 0
    perm_fired = 0
    ps = []
    for _ in range(n_trials):
        y = 0.5 + rng.normal(0.0, se_true)
        if mode == "random_cell" and extra_sd > 0:
            y = y + rng.normal(0.0, extra_sd, size=y.size)
        a = gate(x, y, se_rep)
        if not a["evaluable"]:
            continue
        ps.append(a["p_one_sided"])
        fired += bool(a["slope_passes"])
        perm_fired += bool(a["permutation_p"] <= ALPHA)
    n = len(ps)
    return {"mode": mode, "n_trials": n, "se_scale": se_scale,
            "extra_sd": extra_sd,
            "se_misreport": np.asarray(se_misreport, dtype=float).tolist(),
            "fpr_slope": fired / n if n else None,
            "fpr_permutation": perm_fired / n if n else None,
            "nominal": ALPHA,
            "p_median": float(np.median(ps)) if n else None}


# ---------------------------------------------------------------- block B
def power_curve(n_trials: int, rng, x, se_scale: float, betas) -> list[dict]:
    se = np.array(SE_PATTERN) * se_scale
    x = np.asarray(x, dtype=float)
    out = []
    for b in betas:
        hits = perm_hits = 0
        n = 0
        for _ in range(n_trials):
            y = 0.2 + b * x + rng.normal(0.0, se)
            a = gate(x, y, se)
            if not a["evaluable"]:
                continue
            n += 1
            hits += bool(a["slope_passes"])
            perm_hits += bool(a["permutation_p"] <= ALPHA)
        out.append({"beta": float(b), "n_trials": n,
                    "power_slope": hits / n if n else None,
                    "power_permutation": perm_hits / n if n else None})
    return out


# ---------------------------------------------------------------- block C
def near_tie_sweep(n_trials: int, rng, betas) -> list[dict]:
    """What a near-tie costs, with the x RANGE held fixed.

    The primary-vs-secondary power gap conflates spacing with range: the two
    ladders differ in both, and a compressed range lowers Sxx on its own. Here
    the endpoints are pinned and one interior rung slides toward its neighbour,
    so the only thing changing is how much of the range the interior points
    resolve.
    """
    se = np.array(SE_PATTERN)
    lo, hi = X_PRIMARY[0], X_PRIMARY[-1]
    out = []
    for frac in (0.0, 0.25, 0.5, 0.9, 0.99):
        x2 = X_PRIMARY[1]
        x3 = X_PRIMARY[2] - frac * (X_PRIMARY[2] - x2)     # rung 3 slides to rung 2
        x = np.array([lo, x2, x3, hi])
        row = {"tie_fraction": frac, "x": [float(v) for v in x],
               "gap_23": float(x3 - x2)}
        for b in betas:
            hits = n = 0
            for _ in range(n_trials):
                y = 0.2 + b * x + rng.normal(0.0, se)
                a = gate(x, y, se)
                if not a["evaluable"]:
                    continue
                n += 1
                hits += bool(a["slope_passes"])
            row[f"power_beta_{b}"] = hits / n if n else None
        out.append(row)
    return out


def fallback_block(n_trials: int, rng) -> dict:
    """F.2d.4's extreme-cell contrast, measured as the GATE it becomes.

    Below four surviving cells the slope is descriptive and the contrast decides
    BRANCH 1, so the contrast has its own operating characteristics and they are
    not the slope's. Reading ``slope_passes`` on a 4-cell ladder never touches
    this rule, which is how it went unmeasured.
    """
    se = np.array(SE_PATTERN_3)
    x = np.array(X_PRIMARY_3)

    def cells3(y):
        return [V.Cell.from_summary(e, y=float(yy), y_se=float(s), n_contexts=500,
                                    deficit_exact=float(xx), deficit_se=1e-4)
                for e, xx, yy, s in zip(EPS_3, x, y, se)]

    null = []
    for extra in (0.0, 0.02, 0.05, 0.10):
        slope = contrast = gate_h = n = 0
        for _ in range(n_trials):
            y = 0.5 + rng.normal(0.0, se)
            if extra:
                y = y + rng.normal(0.0, extra, size=y.size)
            a = V.association(cells3(y), axis=V.PRIMARY_AXIS)
            if not a["evaluable"] or a["mode"] != "descriptive":
                continue
            n += 1
            slope += bool(a["slope_passes"])
            contrast += bool(a["extreme_contrast"]["passes"])
            gate_h += bool(a["gate_passes"])
        null.append({"cell_dispersion_sd": extra, "n_trials": n,
                     "fpr_descriptive_slope": slope / n if n else None,
                     "fpr_contrast": contrast / n if n else None,
                     "fpr_gate": gate_h / n if n else None})
    power = []
    for b in (0.1, 0.2, 0.3, 0.5):
        hits = n = 0
        for _ in range(n_trials // 4):
            y = 0.2 + b * x + rng.normal(0.0, se)
            a = V.association(cells3(y), axis=V.PRIMARY_AXIS)
            if not a["evaluable"]:
                continue
            n += 1
            hits += bool(a["gate_passes"])
        power.append({"beta": b, "n_trials": n,
                      "power_gate": hits / n if n else None})
    worst = max((r["fpr_gate"] or 0.0) for r in null)
    return {"note": ("F.2d.4's contrast reads only the two endpoint SEs and has "
                     "no residual-based scale estimate, so cell-level dispersion "
                     "the SEs do not know about passes straight through. The "
                     "slope test in the same rows is protected from exactly that "
                     "by its estimated-scale s^2."),
            "null": null, "power": power,
            "worst_fpr_gate": worst,
            "exceeds_twice_nominal": bool(worst > 2 * ALPHA)}


def permutation_floors() -> list[dict]:
    """A perfect monotone ordering must return exactly 1/k!, and the floor must
    fall as cells are added. This is what makes F.6r's two extra rungs worth a
    fleet: 1/24 leaves no room under alpha=0.05 for even one adjacent swap."""
    out = []
    for k in (4, 5, 6):
        x = np.arange(float(k))
        y = np.arange(float(k))                       # perfect ordering
        p_perfect = V.permutation_p(x, y)
        y_swap = y.copy()
        y_swap[[k - 2, k - 1]] = y_swap[[k - 1, k - 2]]   # one adjacent swap
        out.append({"k": k, "floor": 1.0 / math.factorial(k),
                    "p_perfect_ordering": p_perfect,
                    "p_one_adjacent_swap": V.permutation_p(x, y_swap),
                    "floor_matches": abs(p_perfect - 1.0 / math.factorial(k)) < 1e-12,
                    "perfect_clears_alpha": p_perfect <= ALPHA,
                    "one_swap_clears_alpha": V.permutation_p(x, y_swap) <= ALPHA})
    return out


# ---------------------------------------------------------------- block D
def negative_controls(rng) -> dict:
    """Both directions. A gate that only ever refuses is as useless as one that
    only ever fires."""
    se = np.array(SE_PATTERN)
    x = np.array(X_PRIMARY)
    flat = gate(x, 0.5 + rng.normal(0, se), se)
    planted = gate(x, 0.2 + 0.6 * x + rng.normal(0, se), se)
    return {
        "null_slope_synthetic": {"slope": flat["slope"], "p": flat["p_one_sided"],
                                 "passes": flat["slope_passes"]},
        "planted_monotone": {"slope": planted["slope"], "p": planted["p_one_sided"],
                             "passes": planted["slope_passes"]},
        "both_controls_correct": (not flat["slope_passes"]) and planted["slope_passes"],
    }


def scale_invariance_check(rng) -> dict:
    """Under the estimated-scale convention the t depends on the weights only
    through their ratios. Asserted, not assumed: it is what licenses sweeping
    the SE scale in block A instead of pretending SE_y is already measured."""
    se = np.array(SE_PATTERN)
    y = 0.2 + 0.4 * np.array(X_PRIMARY) + rng.normal(0, se)
    a = gate(X_PRIMARY, y, se)
    b = gate(X_PRIMARY, y, se * 37.0)
    return {"t_at_scale_1": a["t"], "t_at_scale_37": b["t"],
            "p_at_scale_1": a["p_one_sided"], "p_at_scale_37": b["p_one_sided"],
            "invariant": abs(a["t"] - b["t"]) < 1e-9}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    res = {"amendment": "F.2d", "alpha": ALPHA, "trials": a.trials,
           "seed": a.seed, "eps_cells": EPS,
           "x_primary": X_PRIMARY, "x_secondary": X_SECONDARY,
           "se_pattern": SE_PATTERN}

    res["scale_invariance"] = scale_invariance_check(rng)

    res["null"] = []
    # A UNIFORM misreport is absorbed exactly by the estimated-scale
    # convention, so it is kept only as the demonstration of that fact and the
    # informative rows are the per-cell ones. The leverage cells on both ladders
    # are the endpoints -- the deficit ladder is bottom-heavy and the width
    # ladder top-heavy -- so both ends are understated in turn.
    for mode, kw in (("calibrated", {}),
                     ("misreported_uniform", {"se_misreport": 1.5}),
                     ("misreported_uniform", {"se_misreport": 0.667}),
                     ("misreported_percell", {"se_misreport": [1, 1, 1, 2]}),
                     ("misreported_percell", {"se_misreport": [1, 1, 1, 4]}),
                     ("misreported_percell", {"se_misreport": [2, 1, 1, 1]}),
                     ("misreported_percell", {"se_misreport": [4, 1, 1, 1]}),
                     ("misreported_percell", {"se_misreport": [1, 3, 0.5, 1]}),
                     ("random_cell", {"extra_sd": 0.02}),
                     ("random_cell", {"extra_sd": 0.05})):
        for xs, name in ((X_PRIMARY, "primary"), (X_SECONDARY, "secondary")):
            r = null_fpr(a.trials, rng, xs, 1.0,
                         "misreported" if mode.startswith("misreported") else mode,
                         **kw)
            r["null_shape"] = mode
            r["ladder"] = name
            res["null"].append(r)

    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    res["power_primary"] = power_curve(a.trials // 4, rng, X_PRIMARY, 1.0, betas)
    res["power_secondary_near_tie"] = power_curve(
        a.trials // 4, rng, X_SECONDARY, 1.0, betas)

    # A near-tie with the x RANGE held fixed, which is what the claim needs.
    res["near_tie_sweep"] = near_tie_sweep(a.trials // 4, rng, betas=[0.2, 0.5])

    # Block E: the F.2d.4 fallback, the statistic that decides BRANCH 1 below
    # four cells. Measured on gate_passes -- what classify() consults.
    res["fallback_k3"] = fallback_block(a.trials, rng)

    res["permutation_floors"] = permutation_floors()
    res["negative_controls"] = negative_controls(rng)

    stamp(res, "f2d_gate_validation", DIAGNOSTIC)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"out": str(a.out),
                      "scale_invariant": res["scale_invariance"]["invariant"],
                      "controls_ok": res["negative_controls"]["both_controls_correct"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
