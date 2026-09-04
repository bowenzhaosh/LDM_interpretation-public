"""Independent simulation of the three F.2d.4 repair options, plus the one nobody listed.

Written to check a Fable adjudication panel rather than to take its answer on faith,
and kept because the numbers below are what the ruling rests on.

Setup matches the post-D6 three-cell ladder where F.2d.4 is actually in force: the
measured half-A entropy deficits at eps 0.25 / 0.5 / 1.0 as x, and the worst-cell
per-panel SE pattern as the per-cell SE. Cell-level dispersion is added on top of
the per-cell SE, which is the overdispersion the endpoint SEs cannot see and the
thing that convicted the clause.

  current : one-sided z on (y_hi - y_lo) / sqrt(SE_lo^2 + SE_hi^2)
  OPTION 1: the same difference with its SE inflated by sqrt(s2) from the
            descriptive fit, referred to t on df = k - 2 = 1
  OPTION 2: the current contrast AND a positive descriptive slope at alpha
  OPTION 4: delete the contrast; the fallback IS the descriptive slope
            (not on the recorded option list)
"""
from __future__ import annotations

import math
import numpy as np
from scipy import stats

X = np.array([0.155389, 0.415179, 0.766417])   # measured deficit_w2 half-A means
SE = np.array([0.0193, 0.0215, 0.0238])        # measured worst-cell SE pattern
ALPHA = 0.05
DF = len(X) - 2                                # 1


def one_trial(rng, beta, sd_cell):
    y = 0.5 + beta * X + rng.normal(0, SE) + rng.normal(0, sd_cell, len(X))
    w = 1.0 / SE ** 2
    W = w.sum()
    xbar = (w * X).sum() / W
    ybar = (w * y).sum() / W
    dx = X - xbar
    Sxx = (w * dx * dx).sum()
    b = (w * dx * (y - ybar)).sum() / Sxx
    resid = y - (ybar + b * dx)
    s2 = (w * resid * resid).sum() / DF
    se_b = math.sqrt(s2 / Sxx) if s2 > 0 else 0.0
    slope = bool(b > 0 and se_b > 0 and (1 - stats.t.cdf(b / se_b, DF)) <= ALPHA)

    d = y[-1] - y[0]
    se_d = math.sqrt(SE[0] ** 2 + SE[-1] ** 2)
    current = bool(d > 0 and (1 - stats.norm.cdf(d / se_d)) <= ALPHA)

    se_d1 = se_d * math.sqrt(s2) if s2 > 0 else float("inf")
    opt1 = bool(d > 0 and np.isfinite(se_d1) and se_d1 > 0
                and (1 - stats.t.cdf(d / se_d1, DF)) <= ALPHA)
    return slope, current, opt1, bool(slope and current)


def sweep(n=30000, seed=20260824):
    rng = np.random.default_rng(seed)
    print(f"{'':22s} {'OPT4=slope':>11s} {'current':>9s} {'OPT1':>9s} {'OPT2':>9s} "
          f"{'|OPT1-OPT4|':>12s}")
    print("--- NULL (beta = 0): false-positive rate, nominal 0.05")
    for sd in (0.0, 0.02, 0.05, 0.10):
        r = np.array([one_trial(rng, 0.0, sd) for _ in range(n)])
        print(f"  sd={sd:<18} {r[:,0].mean():11.4f} {r[:,1].mean():9.4f} "
              f"{r[:,2].mean():9.4f} {r[:,3].mean():9.4f} "
              f"{(r[:,2]!=r[:,0]).mean():12.4f}")
    print("--- ALTERNATIVE: power")
    for beta in (0.1, 0.2, 0.3, 0.5):
        for sd in (0.0, 0.05):
            r = np.array([one_trial(rng, beta, sd) for _ in range(n)])
            print(f"  beta={beta} sd={sd:<11} {r[:,0].mean():11.4f} {r[:,1].mean():9.4f} "
                  f"{r[:,2].mean():9.4f} {r[:,3].mean():9.4f} "
                  f"{(r[:,2]!=r[:,0]).mean():12.4f}")
    print()
    print("The last column is the finding: OPTION 1 and the descriptive slope reach")
    print("different verdicts in well under 1% of trials. At k=3 the endpoint")
    print("difference and the weighted slope are nearly the same linear functional of")
    print("y, because the middle cell carries almost no leverage; once the contrast is")
    print("given the residual scale it was missing, the two become one test.")


def measure_and_persist(out, n=40000, seed=20260824):
    """The measurement the F.2d.4 ruling makes a precondition of its own adoption.

    The validation script's docstring binds any replacement: "the replacement does
    not get to skip that test". So the adopted conjunction's null FPR and power at
    k=3 are measured here and PERSISTED, in the same shape as fallback_block, rather
    than printed and lost. An unpersisted number cannot ground an adoption under this
    project's artifact-only rule, which is also why Option 1's characteristics being
    computed-but-not-written was a fair objection to them.
    """
    import json
    from datetime import date
    rng = np.random.default_rng(seed)
    doc = {"note": ("Operating characteristics of the F.2d.4 repair options on the "
                    "post-D6 three-cell ladder. OPTION_2 is the conjunction the Fable "
                    "panel ruled for; OPTION_1 and OPTION_4 are recorded so the "
                    "comparison is on disk rather than in a transcript."),
           "x": X.tolist(), "se": SE.tolist(), "alpha": ALPHA, "df": DF,
           "n_trials": n, "seed": seed, "date": date.today().isoformat(),
           "null": [], "power": []}
    for sd in (0.0, 0.02, 0.05, 0.10):
        r = np.array([one_trial(rng, 0.0, sd) for _ in range(n)])
        doc["null"].append(dict(cell_dispersion_sd=sd, n_trials=n,
                                fpr_descriptive_slope=float(r[:, 0].mean()),
                                fpr_current_contrast=float(r[:, 1].mean()),
                                fpr_option1=float(r[:, 2].mean()),
                                fpr_option2_conjunction=float(r[:, 3].mean()),
                                option1_vs_slope_disagreement=float((r[:, 2] != r[:, 0]).mean())))
    for beta in (0.1, 0.2, 0.3, 0.5):
        for sd in (0.0, 0.05):
            r = np.array([one_trial(rng, beta, sd) for _ in range(n)])
            doc["power"].append(dict(beta=beta, cell_dispersion_sd=sd, n_trials=n,
                                     power_descriptive_slope=float(r[:, 0].mean()),
                                     power_current_contrast=float(r[:, 1].mean()),
                                     power_option1=float(r[:, 2].mean()),
                                     power_option2_conjunction=float(r[:, 3].mean())))
    worst = max(x["fpr_option2_conjunction"] for x in doc["null"])
    doc["option2_worst_fpr"] = worst
    doc["option2_exceeds_twice_nominal"] = bool(worst > 2 * ALPHA)
    # The ruling's self-voiding floor: power < 0.15 at beta = 0.2 voids the fallback.
    at_b02 = [x for x in doc["power"] if x["beta"] == 0.2]
    doc["option2_power_at_beta_0p2"] = {str(x["cell_dispersion_sd"]):
                                        x["power_option2_conjunction"] for x in at_b02}
    doc["option2_clears_0p15_power_floor"] = bool(
        min(x["power_option2_conjunction"] for x in at_b02) >= 0.15)
    doc["artifact"] = {"estimand": "f2d4_repair_option_characteristics",
                       "status": "live", "amendment": "F", "schema": 1}
    out.write_text(json.dumps(doc, indent=2))
    return doc


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if "--persist" in sys.argv:
        o = Path("campaigns/corrected_20260812/raw/postF/f2d4_option_characteristics.json")
        o.parent.mkdir(parents=True, exist_ok=True)
        d = measure_and_persist(o)
        print(f"wrote {o}")
        print(f"  OPTION 2 worst null FPR      {d['option2_worst_fpr']:.5f}  "
              f"(nominal {ALPHA}, exceeds 2x: {d['option2_exceeds_twice_nominal']})")
        print(f"  OPTION 2 power at beta=0.2   {d['option2_power_at_beta_0p2']}")
        print(f"  clears the 0.15 power floor  {d['option2_clears_0p15_power_floor']}")
    else:
        sweep()
