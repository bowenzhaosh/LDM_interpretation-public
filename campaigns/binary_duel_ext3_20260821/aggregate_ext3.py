#!/usr/bin/env python3
"""Aggregate PREREG-EXT3 results: seeds x scales replication (exp1/exp2),
d=3 readout port (exp4), patching Phase B (exp3b). Numbers + gates only.
Emits a single JSON summary to stdout-adjacent file aggregate_ext3.json.
"""
import json, glob, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
SCALES = ["base", "mid", "large", "xl"]
PRIORS = ["A", "C", "N"]


def mc(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), float(xs.std(ddof=1) if len(xs) > 1 else np.nan)


def load(pat):
    return sorted(glob.glob(os.path.join(R, pat)))


def keyby(d, k):
    return d.get(k)


def report_exp1():
    out = {}
    for scale in SCALES:
        fs = [f for f in load("exp1/exp1_*.json") if json.load(open(f)).get("scale") == scale]
        if not fs:
            out[scale] = {"n": 0}
            continue
        rs = [json.load(open(f)) for f in fs]
        n = len(rs)
        bw = [r["e1a_natural"]["reg_w"]["theil_slope"] for r in rs]
        rho = [r["e1a_natural"]["reg_w"]["spearman_rho"] for r in rs]
        ci_excl = [not r["e1a_natural"]["reg_w"]["slope_ci_incl0"] for r in rs]
        neg = [b < 0 for b in bw]
        gain = [r["e1b_crossscore"]["gain_spec_mean"] for r in rs]
        gspec = [r["e1b_crossscore"]["gate_spec"] for r in rs]
        gctrl = [r["e1b_crossscore"]["gate_ctrl6"] for r in rs]
        gnull = [r["e1c_gaussian_null"].get("frac_finite_w", r["e1c_gaussian_null"].get("mean_abs_s", np.nan)) for r in rs]
        bm, bsd = mc(bw)
        gm, gsd = mc(gain)
        out[scale] = dict(
            n=n, beta_w_mean=bm, beta_w_sd=bsd, spearman_mean=float(np.mean(rho)),
            frac_ci_excl0=float(np.mean(ci_excl)), frac_negative=float(np.mean(neg)),
            gain_spec_mean=gm, gain_spec_sd=gsd, frac_gate_spec=float(np.mean(gspec)),
            frac_gate_ctrl6=float(np.mean(gctrl)), gaussian_null_frac_finite_w=float(np.mean(gnull)))
    return out


def report_exp2():
    out = {}
    for scale in SCALES:
        fs = [f for f in load("exp2/exp2_*.json") if json.load(open(f)).get("scale") == scale]
        if not fs:
            out[scale] = {"n": 0}
            continue
        rs = [json.load(open(f)) for f in fs]
        n = len(rs)
        w_mis = [r["e2a"]["mean_logit_w_mis"] for r in rs]
        w_ctrl = [r["e2a"]["mean_logit_w_ctrl"] for r in rs]
        out[scale] = dict(
            n=n, mean_logit_w_mis=float(np.mean(w_mis)), mean_logit_w_ctrl=float(np.mean(w_ctrl)),
            frac_gate_w=float(np.mean([r["e2a"]["gate_w"] for r in rs])),
            frac_gate_harm=float(np.mean([r["e2a"]["gate_harm"] for r in rs])),
            frac_gate_benefit=float(np.mean([r["e2a"]["gate_benefit"] for r in rs])),
            frac_gate_w_monotone=float(np.mean([r["e2b"]["gate_w_monotone"] for r in rs])),
            frac_gate_nll_monotone=float(np.mean([r["e2b"]["gate_nll_monotone"] for r in rs])),
            frac_gate_control_flat=float(np.mean([r["e2b"]["gate_control_flat"] for r in rs])))
    return out


def report_exp4():
    out = {}
    for prior in PRIORS:
        fs = [f for f in load("exp4/exp4_*.json") if json.load(open(f)).get("prior") == prior]
        if not fs:
            out[prior] = {"n": 0}
            continue
        rs = [json.load(open(f)) for f in fs]
        n = len(rs)
        bw = [r["e1a_natural"]["reg_w"]["theil_slope"] for r in rs]
        rho = [r["e1a_natural"]["reg_w"]["spearman_rho"] for r in rs]
        mono = [r["e1a_natural"]["reg_w"]["monotone_frac"] for r in rs]
        ci_excl = [not r["e1a_natural"]["reg_w"]["slope_ci_incl0"] for r in rs]
        gain = [r["e1b_crossscore"]["gain_spec_mean"] for r in rs]
        gspec = [r["e1b_crossscore"]["gate_spec"] for r in rs]
        beta_pc = [r["positive_control"]["beta_w"] for r in rs]
        gnull = [r["gaussian_null"]["frac_finite_w"] for r in rs]
        bm, bsd = mc(bw)
        out[prior] = dict(
            n=n, beta_w_mean=bm, beta_w_sd=bsd, spearman_mean=float(np.mean(rho)),
            monotone_frac_mean=float(np.mean(mono)), frac_ci_excl0=float(np.mean(ci_excl)),
            gain_spec_mean=float(np.mean(gain)), frac_gate_spec=float(np.mean(gspec)),
            positive_control_beta_w=float(np.mean(beta_pc)),
            gaussian_null_frac_finite_w=float(np.mean(gnull)))
    return out


def report_exp3b():
    fs = load("exp3b/exp3b_*.json")
    rs = [json.load(open(f)) for f in fs]
    ranks = [1, 2, 4, 8, 16]
    out = {"n": len(rs), "sites": {}}
    for site in ["resid0", "query1"]:
        srs = [r for r in rs if r.get("site") == site]
        o = {"n": len(srs), "ranks": {}}
        for r in ranks:
            key = f"rank{r}"
            ts = [r_[key]["transfer_mean"] for r_ in srs]
            ps = [r_[key]["transfer_p"] for r_ in srs]
            ad = [r_[key]["anti_donor_rate"] for r_ in srs]
            pb = [r_[key]["placebo_transfer_mean"] for r_ in srs]
            gates = [r_[key]["gate"] for r_ in srs]
            tm, tsd = mc(ts)
            o["ranks"][key] = dict(
                transfer_mean=tm, transfer_sd=tsd,
                frac_p_lt_005=float(np.mean([p is not None and p < 0.05 for p in ps])),
                anti_donor_mean=float(np.mean(ad)),
                placebo_transfer_mean=float(np.mean(pb)),
                frac_gate=float(np.mean(gates)))
        o["gate_phaseB_all"] = bool(all(r_["gate_phaseB"] for r_ in srs))
        out["sites"][site] = o
    return out


def main():
    summary = {
        "exp1_seeds_scales": report_exp1(),
        "exp2_seeds_scales": report_exp2(),
        "exp4_d3_readout": report_exp4(),
        "exp3b_phaseB": report_exp3b(),
    }
    with open(os.path.join(HERE, "aggregate_ext3.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
