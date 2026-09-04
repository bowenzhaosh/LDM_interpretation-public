#!/usr/bin/env python3
"""Aggregate binary-duel results across all seeds into one summary dict/printout."""
import json, glob, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def mean_ci(xs):
    xs = np.asarray(xs, float)
    m = xs.mean()
    sd = xs.std(ddof=1) if len(xs) > 1 else float("nan")
    return m, sd


def main():
    print("# BINARY-DUEL AGGREGATE\n")

    # ---- Exp1 ----
    print("## Exp1 — behavioral use")
    f1 = sorted(glob.glob(os.path.join(R, "exp1", "exp1_*.json")))
    print(f"({len(f1)} seeds)")
    e1a_w = []; e1a_s = []; e1b = []
    for f in f1:
        r = json.load(open(f))
        e1a_w.append(r["e1a_natural"]["reg_w"])
        e1a_s.append(r["e1a_natural"]["reg_s"])
        e1b.append(r["e1b_crossscore"])
    print("\nE1a (natural):")
    for nm, arr in [("logit w ~ ell", [e["theil_slope"] for e in e1a_w]),
                    ("s ~ ell", [e["theil_slope"] for e in e1a_s])]:
        m, sd = mean_ci(arr)
        print(f"  {nm:14s} theil mean {m:+.3f} ± {sd:.3f}")
    m, sd = mean_ci([e["spearman_rho"] for e in e1a_w])
    print(f"  spearman rho  mean {m:+.3f} ± {sd:.3f}")
    all_ci_excl = all(not e["slope_ci_incl0"] for e in e1a_w)
    print(f"  slope CI excl 0 in ALL seeds: {all_ci_excl}")
    print("\nE1b (cross-score, fresh outcomes):")
    m, sd = mean_ci([e["gain_spec_mean"] for e in e1b])
    print(f"  gain_spec mean {m:+.4f} ± {sd:.4f}  (all seeds gate_spec: {all(e['gate_spec'] for e in e1b)})")
    m, sd = mean_ci([e["gain_neut_mean"] for e in e1b])
    print(f"  gain_neut mean {m:+.4f} ± {sd:.4f}  (control-6): {all(e['gate_ctrl6'] for e in e1b)}")

    # ---- Exp2 ----
    print("\n## Exp2 — deliberately mislead")
    f2 = sorted(glob.glob(os.path.join(R, "exp2", "exp2_*.json")))
    print(f"({len(f2)} seeds)")
    if f2:
        for nm in ["e2a", "e2b"]:
            gs = [json.load(open(f))[nm] for f in f2]
            if nm == "e2a":
                m1, s1 = mean_ci([g["mean_logit_w_mis"] for g in gs])
                m2, s2 = mean_ci([g["mean_logit_w_ctrl"] for g in gs])
                mh, _ = mean_ci([g["harm_A_mean"] for g in gs])
                mb, _ = mean_ci([g["benefit_B_mean"] for g in gs])
                print(f"\nE2a: logit w misleading {m1:+.2f} ± {s1:.2f} vs ctrl {m2:+.2f} ± {s2:.2f}")
                print(f"  NLL harm on A {mh:+.3f}, benefit on B {mb:+.3f}")
                print(f"  gates: w {all(g['gate_w'] for g in gs)}, harm {all(g['gate_harm'] for g in gs)}, benefit {all(g['gate_benefit'] for g in gs)}")
            else:
                print("\nE2b dose curve (mean over seeds):")
                ds = ["delta_0.00", "delta_0.10", "delta_0.25", "delta_0.50"]
                print(f"  {'delta':12s} {'w_wrong':>10s} {'w_ctrl':>10s} {'nll_A_wrong':>12s}")
                for dk in ds:
                    ww = np.mean([g[dk]["mean_logit_w_wrong"] for g in gs])
                    wc = np.mean([g[dk]["mean_logit_w_control"] for g in gs])
                    na = np.mean([g[dk]["mean_nll_A_wrong"] for g in gs])
                    print(f"  {dk:12s} {ww:10.3f} {wc:10.3f} {na:12.4f}")
                print(f"  monotone gates: w {all(g['gate_w_monotone'] for g in gs)}, nll {all(g['gate_nll_monotone'] for g in gs)}, ctrl-flat {all(g['gate_control_flat'] for g in gs)}")

    # ---- Exp3 ----
    print("\n## Exp3 — internal causal-order interchange (Phase A)")
    f3 = sorted(glob.glob(os.path.join(R, "exp3", "exp3_*.json")))
    print(f"({len(f3)} seeds)")
    if f3:
        phases = [json.load(open(f))["phaseA"] for f in f3]
        print(f"  gate_phaseA all seeds: {all(p['gate_phaseA'] for p in phases)}")
        sites = phases[0]["sites"]
        print(f"  {'site':14s} {'transfer_mean':>13s} {'anti_donor':>10s} {'wilcox_p<0.05':>13s}")
        for s in sites:
            ts = [p[s]["transfer_mean"] for p in phases]
            ad = [p[s]["anti_donor_rate"] for p in phases]
            ps = [p[s]["transfer_p"] for p in phases]
            psig = sum(1 for p in ps if p is not None and p < 0.05)
            m, sd = mean_ci(ts)
            print(f"  {s:14s} {m:13.4f} {np.mean(ad):10.2f} {psig}/{len(phases)}")
        print(f"\n  placebo (matched-magnitude random):")
        pl = [p["placebo_rs"] for p in phases]
        m, sd = mean_ci([p["transfer_mean"] for p in pl])
        psig = sum(1 for p in pl if p["transfer_p"] is not None and p["transfer_p"] < 0.05)
        print(f"    transfer mean {m:+.4f} ± {sd:.4f}, significant in {psig}/{len(phases)} seeds")


if __name__ == "__main__":
    main()
