#!/usr/bin/env python3
"""d5c GATE 0 — mirror-skew asymmetric-Laplace prior pair (pi_A r=2 right-skew, pi_B r=1/2
left-skew), oracle-only. Must pass BEFORE any net training (d5c_PREREG.md section 3a).

AL construction: e = (L1 - a) - (L2 - c), L1~Exp(mean a), L2~Exp(mean c).
  mean 0; Var = a^2 + c^2 = 2 b^2 (matched to the Laplace-scale b from sigma_to_params);
  logpdf(x) = -log(a+c) + ( -(x+a-c)/a  if x+a-c>=0  else  (x+a-c)/c ).
  r = a/c: pi_A r=2, pi_B r=1/2. Mirror symmetry: pi_B(x) = pi_A(-x).
Gate criteria (pre-registered):
  mean|p_A - p_B| >= 0.08; frac(|dp|>=0.15) >= 0.15;
  extremity symmetry on the disagreement subset: |mean|p_A-0.5| - mean|p_B-0.5|| <= 0.03;
  both oracle AUCs >= 0.75; 2nd-moment logistic classifier AUC in [0.45,0.55].
Saves the canonical d5c eval set (contexts + both AL oracles) -> d5c_evalset.npz.
Usage: python3 d5c_gate0.py [N=600] [QUAD=15]
"""
import sys, json, time, math
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
POC = "/Users/bowenzhao/.claude/skills/validate/runs/pfn-dag-readout-pilot/poc"
sys.path.insert(0, POC)
import experiment_v3bump as E
OUT = "/Users/bowenzhao/.claude/skills/night-mode/runs/pfn-dag-tokenfree-prequential"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 900
QUAD = int(sys.argv[2]) if len(sys.argv) > 2 else 15
R_A, R_B, R_C = 2.0, 0.5, 4.0   # A/B = mirror pair (primary); C = same-direction control (v2)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def al_ac(b, r):
    """Laplace-scale b (Var 2b^2) + skew ratio r=a/c -> (a, c)."""
    c = math.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c

def al_sample(b, r, n, rng):
    a, c = al_ac(b, r)
    return rng.exponential(a, n) - rng.exponential(c, n) - (a - c)

def al_logpdf_sum(x, b, r):
    a, c = al_ac(b, r)
    z = x + (a - c)
    pos = z >= 0
    out = np.where(pos, -z / a, z / c) - math.log(a + c)
    return float(out.sum())

def loglik_G1_al(data, p, r):
    a_, b1, b2 = p
    x, y = data[:, 0], data[:, 1]
    return al_logpdf_sum(x, b1, r) + al_logpdf_sum(y - a_ * x, b2, r)

def loglik_G2_al(data, p, r):
    a_, b1, b2 = p
    x, y = data[:, 0], data[:, 1]
    return al_logpdf_sum(y, b1, r) + al_logpdf_sum(x - a_ * y, b2, r)

def oracle_posterior_al(data, grid, r):
    s1v, s2v, rho_v, lw1, lw2, lwr = grid
    lG1, lG2 = [], []
    for i, s1 in enumerate(s1v):
        for j, s2 in enumerate(s2v):
            for k, rho in enumerate(rho_v):
                Sigma = E.build_sigma_from_grid_point(s1, s2, rho)
                if not E.accept_Sigma(Sigma): continue
                lw = lw1[i] + lw2[j] + lwr[k]
                lG1.append(lw + loglik_G1_al(data, E.sigma_to_params_G1(Sigma), r))
                lG2.append(lw + loglik_G2_al(data, E.sigma_to_params_G2(Sigma), r))
    lm1, lm2 = logsumexp(lG1), logsumexp(lG2)
    return math.exp(lm1 - logsumexp([lm1, lm2]))

def sample_data_al(cls, p, n, rng, r):
    a_, b1, b2 = p
    e1 = al_sample(b1, r, n, rng)
    e2 = al_sample(b2, r, n, rng)
    if cls == 1:
        return np.stack([e1, a_ * e1 + e2], axis=1)
    return np.stack([a_ * e1 + e2, e1], axis=1)

def pair_stats(p1, p2, ys, name, thr=0.15):
    dp = p1 - p2; dis = np.abs(dp) >= thr
    e1 = float(np.abs(p1[dis] - 0.5).mean()) if dis.sum() else float("nan")
    e2 = float(np.abs(p2[dis] - 0.5).mean()) if dis.sum() else float("nan")
    return dict(pair=name, mean_abs_dp=float(np.abs(dp).mean()),
                frac_dp_ge_015=float(dis.mean()), n_dis=int(dis.sum()),
                max_abs_dp=float(np.abs(dp).max()),
                extremity_1=e1, extremity_2=e2, extremity_gap=abs(e1 - e2))

def main():
    log(f"d5c GATE 0 (v2) | AL priors A r={R_A} / B r={R_B} / C r={R_C} | N={N} QUAD={QUAD}")
    grid = E.make_sigma_grid(QUAD)
    rng = np.random.default_rng(8181)
    # moment sanity of the sampler (R1 construction check)
    for r in (R_A, R_B, R_C):
        chk = al_sample(1.0, r, 200000, np.random.default_rng(1))
        sk = ((chk - chk.mean())**3).mean() / chk.std()**3
        log(f"  sampler check b=1,r={r}: mean={chk.mean():+.4f} var={chk.var():.4f} (want 2.0) skew={sk:+.3f}")
    Ds, ys, src, pA, pB, pC = [], [], [], [], [], []
    for prior, r in (("A", R_A), ("B", R_B), ("C", R_C)):
        for cls in (1, 2):
            for _ in range(N // 6):
                Sigma, *_ = E.sample_valid_Sigma(rng)
                p = E.sigma_to_params_G1(Sigma) if cls == 1 else E.sigma_to_params_G2(Sigma)
                D = sample_data_al(cls, p, E.N_CONTEXT, rng, r)
                Ds.append(D); ys.append(1 if cls == 1 else 0); src.append(prior)
                pA.append(oracle_posterior_al(D, grid, R_A))
                pB.append(oracle_posterior_al(D, grid, R_B))
                pC.append(oracle_posterior_al(D, grid, R_C))
        log(f"  source {prior} done ({len(Ds)} contexts)")
    Ds = np.array(Ds); ys = np.array(ys); src = np.array(src)
    pA = np.array(pA); pB = np.array(pB); pC = np.array(pC)
    ab_idx = (src == "A") | (src == "B")     # primary-pair eval contexts
    ac_idx = (src == "A") | (src == "C")     # discriminator-pair eval contexts
    AB = pair_stats(pA[ab_idx], pB[ab_idx], ys[ab_idx], "A-B mirror")
    AC = pair_stats(pA[ac_idx], pC[ac_idx], ys[ac_idx], "A-C same-direction")
    auc = {k: float(roc_auc_score(ys, v)) for k, v in (("A", pA), ("B", pB), ("C", pC))}
    feats = np.array([[d[:,0].mean(), d[:,1].mean(), d[:,0].var(),
                       np.cov(d[:,0], d[:,1])[0,1], d[:,1].var()] for d in Ds])
    perm = np.random.default_rng(3).permutation(len(ys)); half = len(ys) // 2
    clf = LogisticRegression(max_iter=2000).fit(feats[perm[:half]], ys[perm[:half]])
    mom_auc = float(roc_auc_score(ys[perm[half:]], clf.predict_proba(feats[perm[half:]])[:, 1]))
    checks_AB = dict(disagreement=AB["mean_abs_dp"] >= 0.08, frac=AB["frac_dp_ge_015"] >= 0.15,
                     extremity_sym=AB["extremity_gap"] <= 0.03,
                     oracle_aucs=(auc["A"] >= 0.75 and auc["B"] >= 0.75),
                     moment_null=0.45 <= mom_auc <= 0.55)
    checks_AC = dict(disagreement=AC["mean_abs_dp"] >= 0.06, frac=AC["frac_dp_ge_015"] >= 0.08,
                     extremity_sym=AC["extremity_gap"] <= 0.05, oracle_auc=auc["C"] >= 0.75)
    gate_AB = all(checks_AB.values()); gate_AC = all(checks_AC.values())
    np.savez(f"{OUT}/d5c_evalset.npz", D=Ds, y=ys, src=src, pA=pA, pB=pB, pC=pC)
    res = dict(N=len(ys), r=dict(A=R_A, B=R_B, C=R_C), AB=AB, AC=AC, oracle_aucs=auc,
               moment_clf_auc=mom_auc,
               checks_AB={k: bool(v) for k, v in checks_AB.items()},
               checks_AC={k: bool(v) for k, v in checks_AC.items()},
               gate0_AB_pass=bool(gate_AB), gate0_AC_pass=bool(gate_AC),
               wallclock_s=time.time() - t0)
    json.dump(res, open(f"{OUT}/d5c_gate0.json", "w"), indent=2)
    log(f"  A-B: mean|dp|={AB['mean_abs_dp']:.3f} frac={AB['frac_dp_ge_015']:.3f} (n={AB['n_dis']}) "
        f"ext_gap={AB['extremity_gap']:.3f}")
    log(f"  A-C: mean|dp|={AC['mean_abs_dp']:.3f} frac={AC['frac_dp_ge_015']:.3f} (n={AC['n_dis']}) "
        f"ext_gap={AC['extremity_gap']:.3f}")
    log(f"  oracle AUCs: {auc}   moment-null={mom_auc:.3f} ([0.45,0.55])")
    log(f"  GATE 0 A-B (PRIMARY): {'PASS' if gate_AB else 'FAIL'} {checks_AB}")
    log(f"  GATE 0 A-C (discriminator): {'PASS' if gate_AC else 'FAIL'} {checks_AC}")

if __name__ == "__main__":
    main()
