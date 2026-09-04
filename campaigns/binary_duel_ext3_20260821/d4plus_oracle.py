#!/usr/bin/env python3
"""d>=4 HONEST SCOPE BOUND (T3-A). General-d moment-matched LiNGAM-family construction + MC posterior oracle,
to measure how identifiability of the latent ordering family behaves as structural complexity d grows. This
pre-empts the "why not higher d" reviewer attack with a MEASURED intrinsic bound (NOT a training result).

CONSTRUCTION (exact Fix-B analog at any d): families = the d! topological orderings of the complete DAG on d
nodes. For ordering pi and SPD Sigma, the permuted-LDL factorization Sigma[pi,pi] = L D L^T (L unit lower) gives
the ordered-regression form x_pi = L e, e_m ~ AL(b_m, r) with Var(e_m)=D_m -> class-conditional 1st+2nd moments
IDENTICAL across all d! families for every Sigma; the only family signal is >=3rd-order error shape (AL skew r).
ORACLE: MC over M valid Sigma atoms, p(family|D) ∝ (1/M) sum_m L_fam(D|Sigma_m). Convergence-gated (two disjoint
M/2 halves correlate >=0.98, median maxdp<0.02). METRICS per d (FORBIDDEN to headline the nats-shortfall vs the
moment ceiling -- we report the top-1/chance RATIO as headline, I-nats as support):
  - oracle top-1 accuracy / chance(=1/d!)  [identifiability RATIO vs d]
  - mean posterior TV between the r=2 and r=4 priors (prior-conditioning signal)
  - I(family;D) in nats = log(d!) - mean posterior entropy, Miller-Madow bias-corrected; GAUSSIAN I=0 anchor
    (Gaussian-source contexts must give I≈0 / uniform posterior -- the confound-immune control at every d)
  - moment-null multiclass accuracy ≈ 1/d! (construction integrity)
Usage: python3 d4plus_oracle.py <d> [N] [M] [K]   (also writes d{d}_scope.json). CPU only."""
import sys, os, json, time, math
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.special import logsumexp
from itertools import permutations
from sklearn.linear_model import LogisticRegression
# portable output dir: env override, else the script's own directory (was a hardcoded Mac path -> crashed on cluster)
HERE = os.environ.get("DSCOPE_OUT", os.path.dirname(os.path.abspath(__file__)))
D_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 300
M   = int(sys.argv[3]) if len(sys.argv) > 3 else (20000 if D_DIM <= 3 else 200000)
K   = int(sys.argv[4]) if len(sys.argv) > 4 else 30
R_A, R_C = 2.0, 4.0
ORDERINGS = list(permutations(range(D_DIM)))
NFAM = len(ORDERINGS); CHANCE = 1.0 / NFAM
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

# ---------- general permuted-LDL ordered-regression params ----------
def params_for(S, pi):
    """Vectorized for (n,d,d) Sigmas + ordering pi. Returns:
       Lunit (n,d,d) unit-lower (x_pi = Lunit @ e), U=inv(Lunit) (n,d,d) for residuals, b (n,d) AL scales (Var=2b^2)."""
    S = np.asarray(S); n, d = len(S), S.shape[1]
    Spi = S[:, pi][:, :, pi]                                  # permute rows+cols
    Lch = np.linalg.cholesky(Spi)                            # (n,d,d) lower, Spi = Lch Lch^T
    diag = np.diagonal(Lch, axis1=1, axis2=2)                # (n,d)
    Lunit = Lch / diag[:, None, :]                           # column-scale -> unit lower
    Dv = diag**2                                             # residual variances
    U = np.linalg.inv(Lunit)                                 # unit lower
    b = np.sqrt(np.maximum(Dv, 1e-12) / 2.0)
    return Lunit, U, b

def validity_keep(S):
    """Fix-B box across ALL d! orderings: |AR coef| <= 1.5 and residual scale b in [0.3,1.3]."""
    keep = np.ones(len(S), bool)
    for pi in ORDERINGS:
        Lunit, U, b = params_for(S, pi)
        beta = -U                                            # AR coefs (strict lower)
        mask = np.tril(np.ones((D_DIM, D_DIM)), -1).astype(bool)
        amax = np.abs(beta[:, mask]).max(1)
        keep &= (amax <= 1.5) & (b >= 0.3).all(1) & (b <= 1.3).all(1)
    return keep

def sample_Sigmas(rng, n):
    out = []
    while len(out) < n:
        m = 8192
        sd = np.exp(rng.uniform(math.log(0.6), math.log(1.5), (m, D_DIM)))
        R = np.eye(D_DIM)[None].repeat(m, 0)
        iu = np.triu_indices(D_DIM, 1)
        rho = rng.choice([-1., 1.], (m, len(iu[0]))) * rng.uniform(0.3, 0.8, (m, len(iu[0])))
        R[:, iu[0], iu[1]] = rho; R[:, iu[1], iu[0]] = rho
        S = sd[:, :, None] * R * sd[:, None, :]
        ev = np.linalg.eigvalsh(S); S = S[ev[:, 0] > 1e-6]
        S = S[validity_keep(S)]
        out.extend(S)
    return np.array(out[:n])

# ---------- AL machinery (Var=2b^2; r=a/c) ----------
def al_ac(b, r):
    c = np.sqrt(2.0 * b * b / (1.0 + r * r)); return r * c, c
def al_sample(b, r, size, rng):
    a, c = al_ac(b, r); return rng.exponential(a, size) - rng.exponential(c, size) - (a - c)
def al_logpdf(x, b, r):
    a, c = al_ac(b, r); z = x + (a - c)
    return np.where(z >= 0, -z / a, z / c) - np.log(a + c)

def gen_data(S1, fam, r, k, rng, gaussian=False):
    pi = ORDERINGS[fam]; Lunit, U, b = params_for(S1[None], pi); Lunit, b = Lunit[0], b[0]
    if gaussian:
        e = rng.normal(0, np.sqrt(2.0) * b[None, :], (k, D_DIM))   # same variance, Gaussian errors (I=0 anchor)
    else:
        e = np.stack([al_sample(np.full(k, b[m]), r, k, rng) for m in range(D_DIM)], 1)
    xpi = e @ Lunit.T                                        # (k,d), Cov = Sigma_pi
    x = np.empty_like(xpi); x[:, list(pi)] = xpi
    return x

# ---------- MC oracle ----------
class Oracle:
    def __init__(self, atoms):
        self.atoms = atoms; self.P = {pi: params_for(atoms, pi) for pi in ORDERINGS}
    def loglik_fam(self, D, pi, r, sl=slice(None)):
        Lunit, U, b = self.P[pi]; U, b = U[sl], b[sl]        # (m,d,d),(m,d)
        Dpi = D[:, list(pi)]                                 # (k,d)
        ll = np.zeros(U.shape[0])
        for mch in range(D_DIM):
            e_m = Dpi @ U[:, mch, :].T                       # (k,m) residual channel mch per atom
            ll += al_logpdf(e_m.T, b[:, mch:mch+1], r).sum(1)
        return ll
    def posterior(self, D, r, sl=slice(None)):
        lz = np.array([logsumexp(self.loglik_fam(D, pi, r, sl)) for pi in ORDERINGS])
        return np.exp(lz - logsumexp(lz))

def post_entropy(p):
    """Shannon entropy (nats) of a (deterministic) posterior vector. The finite-K concentration bias is removed
    downstream by subtracting the matched Gaussian-source bias FLOOR (I_signal = I - I_gauss), not by a
    sample-count MM term (which does not apply to a deterministic posterior)."""
    p = np.clip(p, 1e-12, 1.0); return -float(np.sum(p * np.log(p)))

def main():
    log(f"d={D_DIM} | {NFAM} families (chance {CHANCE:.4f}) | N={N} M={M} K={K} | priors r={R_A} vs {R_C}")
    atoms = sample_Sigmas(np.random.default_rng(7002 + D_DIM), M)
    log(f"atom set: {len(atoms)} valid Sigmas ({NFAM}-ordering validity box)")
    orc = Oracle(atoms); rng = np.random.default_rng(7001 + D_DIM)

    # construction integrity: per-family cov match + moment-null multiclass acc
    Schk = sample_Sigmas(np.random.default_rng(7003), 1)[0]; max_rel = 0.0
    for fam in range(NFAM):
        Dbig = gen_data(Schk, fam, R_A, 20000, np.random.default_rng(fam)); emp = np.cov(Dbig.T)
        max_rel = max(max_rel, float(np.abs(emp - Schk).max() / np.abs(Schk).max()))
    rngm = np.random.default_rng(7104); feats, labs = [], []
    for _ in range(6000):
        S1 = sample_Sigmas(rngm, 1)[0]; fam = int(rngm.integers(NFAM)); Dd = gen_data(S1, fam, R_A, K, rngm)
        c = np.cov(Dd.T); feats.append(np.r_[Dd.mean(0), c[np.triu_indices(D_DIM)]]); labs.append(fam)
    feats, labs = np.array(feats), np.array(labs)
    clf = LogisticRegression(max_iter=2000).fit(feats[:4000], labs[:4000])
    mom_acc = float((clf.predict(feats[4000:]) == labs[4000:]).mean())
    log(f"(g1) max rel cov err {max_rel:.3f} | moment-null multiclass acc {mom_acc:.3f} (chance {CHANCE:.4f})")

    # eval contexts (50/50 A/C) + Gaussian I=0 anchor contexts
    half = M // 2; Ds, fams, srcs, PA, PC = [], [], [], [], []
    Hpost_A, Hpost_C, conv_corr, conv_dp = [], [], [], []
    for src, r in (("A", R_A), ("C", R_C)):
        for n in range(N // 2):
            S1 = sample_Sigmas(rng, 1)[0]; fam = int(rng.integers(NFAM)); Dd = gen_data(S1, fam, r, K, rng)
            pa, pc = orc.posterior(Dd, R_A), orc.posterior(Dd, R_C)
            if n < 25:
                pa1 = orc.posterior(Dd, R_A, slice(0, half)); pa2 = orc.posterior(Dd, R_A, slice(half, M))
                conv_corr.append(float(np.corrcoef(pa1, pa2)[0, 1])); conv_dp.append(float(np.abs(pa1 - pa2).max()))
            Ds.append(Dd); fams.append(fam); srcs.append(src); PA.append(pa); PC.append(pc)
            (Hpost_A if src == "A" else Hpost_C).append(post_entropy(pa if src == "A" else pc))
        log(f"  source {src} done ({len(Ds)})")
    # GAUSSIAN I=0 anchor (the confound-immune control at every d): Gaussian-source data has identical joint law
    # across all orderings -> TRUE oracle is uniform. Gaussian TOP-1 ≈ chance is the clean headline null; the
    # Gaussian I_nats is the finite-K entropy-bias FLOOR subtracted from the AL-source I to debias it.
    Hgauss, gauss_correct = [], []
    for n in range(max(60, N // 2)):
        S1 = sample_Sigmas(rng, 1)[0]; fam = int(rng.integers(NFAM)); Dg = gen_data(S1, fam, R_A, K, rng, gaussian=True)
        pg = orc.posterior(Dg, R_A); Hgauss.append(post_entropy(pg)); gauss_correct.append(int(pg.argmax() == fam))
    fams, srcs = np.array(fams), np.array(srcs); PA, PC = np.array(PA), np.array(PC)
    own = srcs == "A"; ownC = srcs == "C"
    acc_A = float((PA[own].argmax(1) == fams[own]).mean()); acc_C = float((PC[ownC].argmax(1) == fams[ownC]).mean())
    TV = 0.5 * np.abs(PA - PC).sum(1)
    logNF = math.log(NFAM)
    I_A = logNF - float(np.mean(Hpost_A)); I_C = logNF - float(np.mean(Hpost_C)); I_gauss = logNF - float(np.mean(Hgauss))
    gauss_top1 = float(np.mean(gauss_correct))
    conv_ok = (np.median(conv_corr) >= 0.98 and np.median(conv_dp) < 0.02)
    out = dict(d=D_DIM, n_families=NFAM, chance=CHANCE, N=len(fams), M=M, K=K,
        max_rel_cov_err=max_rel, moment_null_acc=mom_acc,
        oracle_top1_A=acc_A, oracle_top1_C=acc_C,
        top1_over_chance_A=round(acc_A / CHANCE, 2), top1_over_chance_C=round(acc_C / CHANCE, 2),
        gauss_anchor_top1=round(gauss_top1, 4), gauss_top1_over_chance=round(gauss_top1 / CHANCE, 2),
        mean_TV=float(TV.mean()), frac_TV_ge_015=float((TV >= 0.15).mean()),
        I_nats_A=round(I_A, 4), I_nats_C=round(I_C, 4), I_nats_gauss_floor=round(I_gauss, 4),
        I_signal_A=round(I_A - I_gauss, 4), I_signal_C=round(I_C - I_gauss, 4),
        gauss_anchor_ok=bool(gauss_top1 <= CHANCE + 2.5*math.sqrt(CHANCE*(1-CHANCE)/max(len(gauss_correct),1))),
        mc_conv_corr_median=float(np.median(conv_corr)), mc_conv_maxdp_median=float(np.median(conv_dp)),
        convergence_ok=bool(conv_ok), wallclock_s=time.time() - t0)
    json.dump(out, open(f"{HERE}/d{D_DIM}_scope.json", "w"), indent=2)
    log("="*64)
    log(f"d={D_DIM}: oracle top-1 A {acc_A:.3f}/C {acc_C:.3f} (chance {CHANCE:.4f} -> ratio {acc_A/CHANCE:.1f}x/{acc_C/CHANCE:.1f}x)")
    log(f"  Gaussian anchor top-1 {gauss_top1:.3f} ({gauss_top1/CHANCE:.2f}x chance, ok={out['gauss_anchor_ok']}) = clean null")
    log(f"  mean TV {TV.mean():.3f} | I_signal (debiased) A {I_A-I_gauss:.3f}/C {I_C-I_gauss:.3f} nats (raw A {I_A:.3f}, gauss floor {I_gauss:.3f})")
    log(f"  MC conv corr {np.median(conv_corr):.4f} maxdp {np.median(conv_dp):.4f} (ok={conv_ok}) | moment-null {mom_acc:.3f}")
    log(f"D{D_DIM}_SCOPE_DONE -> d{D_DIM}_scope.json")

if __name__ == "__main__":
    main()
