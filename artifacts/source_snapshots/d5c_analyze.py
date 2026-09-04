#!/usr/bin/env python3
"""d5c ANALYSIS — pre-registered statistics (d5c_PREREG.md v2+v3, locked before fleet results).

Inputs: nets/ (fleet checkpoints M_<prior>_s<seed>_st<steps>.pt), d5c_evalset.npz (N=1500,
sources A/C/B with oracles pA,pB,pC), d5b_evalset.npz (secondary L/T pair).
PRIMARY: pair (A,C), 10k arm, same-probe-set between-net contrast on |pA-pC|>=0.15 subset
(A+C-source contexts), seed-level jackknife, source strata, per-net own-alignment,
permuted-oracle null. Verdict per prereg. Everything else descriptive.

Operationalizations pinned here (before results seen):
- Probe ladder: logistic C=0.1 / logistic C=10 / MLP(64) (sklearn, max_iter=800, early_stopping).
  Probe-train N=1500, held-out N=500 (per prior, fixed seeds). RUNG SELECTION = weakest rung
  whose MEDIAN held-out own-prior AUC across seeds is >= 0.62 for BOTH arms of the pair;
  fallback = rung with max min-arm median AUC, flagged POWER-LIMITED.
- Seed-level aggregation: 16x16 rho_cross matrix -> A-side row means and C-side col means;
  jackknife CI = mean +/- t_{K-1,0.975} * sqrt((K-1)/K * sum((m_i - mbar)^2)) computed
  separately over the A-seed factor and C-seed factor; REPORT THE WIDER.
- Permuted-oracle null: 500 context-permutations of Delta-oracle; two-sided p of mean rho_cross.
Usage: python3 d5c_analyze.py [nets_dir] [steps=10000]
"""
import sys, os, json, time, math, glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, t as tdist

HERE = "/Users/bowenzhao/.claude/skills/night-mode/runs/pfn-dag-tokenfree-prequential"
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/bowenzhao/.claude/skills/validate/runs/pfn-dag-readout-pilot/poc")

NETS = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/nets"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
sys.argv = sys.argv[:1]   # d5c_gate0 parses argv at import time
import d5c_gate0 as G
import experiment_v3bump as E
N_PROBE, N_HELD = 1500, 500
THR = 0.15
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

# ---------------- net + batched reps ----------------
def load_net(path):
    m = E.PFNModel(d_model=256, d_ff=512, n_heads=4, n_layers=2)
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m

@torch.no_grad()
def reps_batched(model, Ds, bs=256):
    out = []
    tok = torch.tensor([2])
    for i in range(0, len(Ds), bs):
        ct = torch.tensor(np.asarray(Ds[i:i+bs]), dtype=torch.float32)
        ce = model.point_embed(ct)
        te = model.token_embed(tok).unsqueeze(1).expand(ct.shape[0], 1, -1)
        o = model.transformer(torch.cat([te, ce], 1))
        out.append(o[:, 1:, :].mean(1).numpy())
    return np.concatenate(out)

# ---------------- probe machinery ----------------
def gen_probe_set(prior, n, seed):
    """prior in A/B/C (AL r) or L/T; reuses validated generators."""
    rng = np.random.default_rng(seed)
    Ds, ys = [], []
    for cls in (1, 2):
        for _ in range(n // 2):
            Sigma, *_ = E.sample_valid_Sigma(rng)
            p = E.sigma_to_params_G1(Sigma) if cls == 1 else E.sigma_to_params_G2(Sigma)
            if prior in ("A", "B", "C"):
                r = {"A": G.R_A, "B": G.R_B, "C": G.R_C}[prior]
                D = G.sample_data_al(cls, p, E.N_CONTEXT, rng, r)
            elif prior == "L":
                D = (E.sample_data_G1 if cls == 1 else E.sample_data_G2)(*p, E.N_CONTEXT, rng)
            else:  # T: t(5) variance-matched
                a_, b1, b2 = p
                s1 = b1 * math.sqrt(2 * 3 / 5.0); s2 = b2 * math.sqrt(2 * 3 / 5.0)
                e1 = rng.standard_t(5.0, E.N_CONTEXT) * s1
                e2 = rng.standard_t(5.0, E.N_CONTEXT) * s2
                D = (np.stack([e1, a_ * e1 + e2], 1) if cls == 1
                     else np.stack([a_ * e1 + e2, e1], 1))
            Ds.append(D); ys.append(1 if cls == 1 else 0)
    Ds = np.array(Ds); ys = np.array(ys)
    perm = np.random.default_rng(seed + 999).permutation(len(ys))   # interleave classes
    return Ds[perm], ys[perm]

PROBE_SEED = {"A": 11, "B": 12, "C": 13, "L": 14, "T": 15}

def make_probe(kind):
    if kind == "logC0.1": return LogisticRegression(max_iter=3000, C=0.1)
    if kind == "logC10":  return LogisticRegression(max_iter=3000, C=10.0)
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, early_stopping=True,
                         random_state=0)

def fit_apply(kind, rep_tr, y_tr, rep_held, rep_ev):
    mu, sd = rep_tr.mean(0), rep_tr.std(0) + 1e-8
    clf = make_probe(kind).fit((rep_tr - mu) / sd, y_tr)
    ph_held = clf.predict_proba((rep_held - mu) / sd)[:, 1]
    ph_ev = clf.predict_proba((rep_ev - mu) / sd)[:, 1]
    return ph_held, ph_ev

def jack_ci(vals):
    """t-interval over seed-level summaries."""
    v = np.asarray(vals, float); K = len(v)
    m = v.mean(); se = v.std(ddof=1) / math.sqrt(K)
    h = tdist.ppf(0.975, K - 1) * se
    return float(m), float(m - h), float(m + h)

def ece_cont(p, target, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum() == 0: continue
        e += m.sum() / len(p) * abs(target[m].mean() - p[m].mean())
    return float(e)

# ---------------- pair analysis (the pre-registered engine) ----------------
def analyze_pair(tagX, tagY, pX, pY, Ds_ev, ys_ev, src_ev, strata, nets_dir, steps,
                 seedsX, seedsY, label):
    """tagX/tagY: prior letters; pX/pY: own oracles on the eval set; strata: list of source
    labels defining the two strata. Returns full result dict."""
    log(f"=== pair ({tagX},{tagY}) @ {steps} steps [{label}] ===")
    use = np.isin(src_ev, strata)
    pXu, pYu, ysu, srcu = pX[use], pY[use], ys_ev[use], src_ev[use]
    Dsu = Ds_ev[use]
    dp = pXu - pYu; dis = np.abs(dp) >= THR
    log(f"  eval n={use.sum()}, disagreement subset n={int(dis.sum())}")

    # probe sets (train + held-out)
    DX, yX = gen_probe_set(tagX, N_PROBE + N_HELD, PROBE_SEED[tagX])
    DY, yY = gen_probe_set(tagY, N_PROBE + N_HELD, PROBE_SEED[tagY])
    sets = {tagX: (DX, yX), tagY: (DY, yY)}

    # load nets + compute reps
    nets = {}
    for tag, seeds in ((tagX, seedsX), (tagY, seedsY)):
        for s in seeds:
            p = f"{nets_dir}/M_{tag}_s{s}_st{steps}.pt"
            if os.path.exists(p): nets[(tag, s)] = load_net(p)
    log(f"  loaded {len(nets)} nets")
    rep_ev, rep_tr, rep_held = {}, {}, {}
    for key, net in nets.items():
        rep_ev[key] = reps_batched(net, Dsu)
        for ps, (Dp, yp) in sets.items():
            rep_tr[(key, ps)] = reps_batched(net, Dp[:N_PROBE])
            rep_held[(key, ps)] = reps_batched(net, Dp[N_PROBE:])
    log(f"  reps done")

    # probe ladder + rung selection (median held-out own-prior AUC >= 0.62 both arms)
    ladder = ["logC0.1", "logC10", "MLP64"]
    held_auc = {}; phat = {}
    for kind in ladder:
        for key in nets:
            for ps, (Dp, yp) in sets.items():
                ph_h, ph_e = fit_apply(kind, rep_tr[(key, ps)], yp[:N_PROBE],
                                       rep_held[(key, ps)], rep_ev[key])
                held_auc[(kind, key, ps)] = float(roc_auc_score(yp[N_PROBE:], ph_h))
                phat[(kind, key, ps)] = ph_e
    med_auc = {}
    for kind in ladder:
        for tag in (tagX, tagY):
            own = [held_auc[(kind, (tag, s), tag)] for t2, s in nets if t2 == tag
                   for _ in [0]]
            own = [held_auc[(kind, k, tag)] for k in nets if k[0] == tag]
            med_auc[(kind, tag)] = float(np.median(own))
    rung = None
    for kind in ladder:
        if med_auc[(kind, tagX)] >= 0.62 and med_auc[(kind, tagY)] >= 0.62:
            rung = kind; break
    power_limited = rung is None
    if power_limited:
        rung = max(ladder, key=lambda k: min(med_auc[(k, tagX)], med_auc[(k, tagY)]))
    log(f"  rung selected: {rung} (power_limited={power_limited}); median heldout own-AUCs: "
        + ", ".join(f"{k}:{tagX}={med_auc[(k,tagX)]:.3f}/{tagY}={med_auc[(k,tagY)]:.3f}" for k in ladder))

    sX = sorted(s for t, s in nets if t == tagX); sY = sorted(s for t, s in nets if t == tagY)
    res = dict(pair=f"{tagX}-{tagY}", steps=steps, label=label, n_eval=int(use.sum()),
               n_dis=int(dis.sum()), rung=rung, power_limited=bool(power_limited),
               median_heldout_auc={f"{k}_{t}": med_auc[(k, t)] for k in ladder for t in (tagX, tagY)},
               seedsX=sX, seedsY=sY)

    def contrast_stats(probe_set, mask):
        M = np.zeros((len(sX), len(sY)))
        for i, si in enumerate(sX):
            for j, sj in enumerate(sY):
                d_net = phat[(rung, (tagX, si), probe_set)] - phat[(rung, (tagY, sj), probe_set)]
                M[i, j] = spearmanr(d_net[mask], dp[mask]).correlation
        rows = M.mean(1); cols = M.mean(0)
        mA = jack_ci(rows); mC = jack_ci(cols)
        wider = mA if (mA[2] - mA[1]) >= (mC[2] - mC[1]) else mC
        return dict(mean=float(M.mean()), ci_wider=[wider[1], wider[2]],
                    ci_A_factor=[mA[1], mA[2]], ci_C_factor=[mC[1], mC[2]])

    # (i) overall contrast, both probe sets
    res["contrast"] = {ps: contrast_stats(ps, dis) for ps in (tagX, tagY)}
    # (ii) strata
    res["contrast_strata"] = {}
    for st in strata:
        mask = dis & (srcu == st)
        res["contrast_strata"][st] = ({ps: contrast_stats(ps, mask) for ps in (tagX, tagY)}
                                      if mask.sum() >= 20 else f"n={int(mask.sum())} too small")
    # (iii) per-net own-oracle alignment on the subset (matched probe set)
    own_align = {}
    for tag, oracle in ((tagX, pXu), (tagY, pYu)):
        vals = [spearmanr(phat[(rung, (tag, s), tag)][dis], oracle[dis]).correlation
                for s in (sX if tag == tagX else sY)]
        m, lo, hi = jack_ci(vals)
        own_align[tag] = dict(mean=m, ci=[lo, hi], per_seed=[float(v) for v in vals])
    res["own_alignment"] = own_align
    # (v) permuted-oracle null (on probe set tagX version)
    rngp = np.random.default_rng(0)
    obs = res["contrast"][tagX]["mean"]
    Mnull = []
    d_nets = [phat[(rung, (tagX, si), tagX)] - phat[(rung, (tagY, sj), tagY)]
              for si in sX[:4] for sj in sY[:4]]   # 16 pairs suffice for the null
    dsub = [d[dis] for d in d_nets]; dpd = dp[dis]
    for _ in range(500):
        perm = rngp.permutation(len(dpd))
        Mnull.append(np.mean([spearmanr(d, dpd[perm]).correlation for d in dsub]))
    Mnull = np.array(Mnull)
    res["perm_null"] = dict(p_two_sided=float((np.abs(Mnull) >= abs(obs)).mean()),
                            null_2p5=float(np.percentile(Mnull, 2.5)),
                            null_97p5=float(np.percentile(Mnull, 97.5)))
    # within-prior null (descriptive)
    for tag, ss in ((tagX, sX), (tagY, sY)):
        vals = [spearmanr((phat[(rung, (tag, si), tag)] - phat[(rung, (tag, sj), tag)])[dis],
                          dp[dis]).correlation
                for si in ss for sj in ss if si < sj]
        res[f"within_{tag}"] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
    # descriptives: const-0.5 null, extremities, ECE vs own oracle (+ isotonic), v1 fracs
    res["descriptive"] = dict(
        const05_closer_to_Y=float((np.abs(0.5 - pYu[dis]) < np.abs(0.5 - pXu[dis])).mean()),
        extremity=dict(X=float(np.abs(pXu[dis] - 0.5).mean()), Y=float(np.abs(pYu[dis] - 0.5).mean())))
    ece = {}
    for tag, oracle in ((tagX, pXu), (tagY, pYu)):
        raw, recal = [], []
        ss = sX if tag == tagX else sY
        for s in ss:
            ph = phat[(rung, (tag, s), tag)]
            raw.append(ece_cont(ph, oracle))
            # isotonic recalibration fit on held-out probe data (labels), applied to eval
            ph_h, _ = fit_apply(rung, rep_tr[((tag, s), tag)], sets[tag][1][:N_PROBE],
                                rep_held[((tag, s), tag)], rep_ev[(tag, s)])
            iso = IsotonicRegression(out_of_bounds="clip").fit(ph_h, sets[tag][1][N_PROBE:])
            recal.append(ece_cont(iso.predict(ph), oracle))
        ece[tag] = dict(raw_median=float(np.median(raw)), recal_median=float(np.median(recal)))
    res["ece_vs_own_oracle"] = ece
    return res

def verdict_primary(r):
    """Pre-registered rule (v2 as amended by v3)."""
    c_ok = all(r["contrast"][ps]["mean"] > 0 and r["contrast"][ps]["ci_wider"][0] > 0
               for ps in r["contrast"])
    st = r["contrast_strata"]
    st_ok = all(isinstance(v, dict) and
                all(v[ps]["mean"] > 0 and v[ps]["ci_wider"][0] > 0 for ps in v)
                for v in st.values())
    own_ok = all(v["ci"][0] > 0 for v in r["own_alignment"].values())
    null_ok = r["perm_null"]["null_2p5"] <= 0 <= r["perm_null"]["null_97p5"]
    in_band = all(-0.08 <= r["contrast"][ps]["ci_wider"][0] and
                  r["contrast"][ps]["ci_wider"][1] <= 0.08 for ps in r["contrast"])
    if c_ok and st_ok and own_ok and null_ok:
        v = "CONFIRMED (prior-specific encoding)"
    elif own_ok and in_band:
        v = "TRUE-NEGATIVE-SUPPORTED (decodable but not prior-specific)"
    elif not own_ok:
        v = "INCONCLUSIVE-POWER (own-oracle alignment not established)"
    else:
        v = "INCONCLUSIVE"
    wording = ("posterior" if all(e["recal_median"] <= 0.06 for e in r["ece_vs_own_oracle"].values())
               else "ranking")
    return v, wording

def main():
    ev = np.load(f"{HERE}/d5c_evalset.npz")
    Ds, ys, src = ev["D"], ev["y"], ev["src"]
    pA, pB, pC = ev["pA"], ev["pB"], ev["pC"]
    out = dict(prereg="d5c_PREREG.md v3", nets_dir=NETS)

    # PRIMARY: (A,C) @10k
    r_pri = analyze_pair("A", "C", pA, pC, Ds, ys, src, ["A", "C"], NETS, 10000,
                         list(range(16)), list(range(16)), "PRIMARY")
    v, wording = verdict_primary(r_pri)
    r_pri["verdict"] = v; r_pri["claim_wording"] = wording
    out["primary_AC_10k"] = r_pri
    log(f"PRIMARY VERDICT: {v} (claim wording: {wording})")

    # encoding-depth arm: (A,C) @40k (8 seeds)
    try:
        out["depth_AC_40k"] = analyze_pair("A", "C", pA, pC, Ds, ys, src, ["A", "C"], NETS,
                                           40000, list(range(8)), list(range(8)), "DEPTH")
    except Exception as ex:
        out["depth_AC_40k"] = f"failed: {ex}"
    # mirror descriptive: (A,B) @10k
    try:
        out["mirror_AB_10k"] = analyze_pair("A", "B", pA, pB, Ds, ys, src, ["A", "B"], NETS,
                                            10000, list(range(16)), list(range(8)), "MIRROR-DESC")
    except Exception as ex:
        out["mirror_AB_10k"] = f"failed: {ex}"
    # secondary: (L,T) on the d5b eval set
    try:
        ev2 = np.load(f"{HERE}/d5b_evalset.npz")
        out["secondary_LT_10k"] = analyze_pair("L", "T", ev2["ppi"], ev2["ppip"], ev2["D"],
                                               ev2["y"], ev2["src"], ["pi", "pip"], NETS, 10000,
                                               list(range(8)), list(range(8)), "SECONDARY")
    except Exception as ex:
        out["secondary_LT_10k"] = f"failed: {ex}"

    out["wallclock_s"] = time.time() - t0
    json.dump(out, open(f"{HERE}/d5c_results.json", "w"), indent=2, default=str)
    log(f"saved -> d5c_results.json")
    log(f"FINAL PRIMARY VERDICT: {v}")

if __name__ == "__main__":
    main()
