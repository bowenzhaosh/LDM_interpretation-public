#!/usr/bin/env python3
"""stage1_functional_law — Stage-1 "Functional Update Law" pipeline (PFN evidence integration).

Framing (d=2 binary latent family). For a dataset D of K (x,y) pairs drawn from one of two
causal families Z in {G1, G2} (linear SEMs with asymmetric-Laplace noise, Fix-B Sigma prior),
exact Bayes gives
      logit w*(D) = logit pi + ell(D),   ell(D) = log p(D|G1) - log p(D|G2),
where pi is the prior over families and ell(D) is the prior-free log evidence ratio.
The PFN induces a posterior weight w_theta(D) (operationalized below); we test whether
      g_theta(D) = logit w_theta(D)  ~  a_theta + h_theta(ell(D))
with the specific laws that Bayes requires:

  E1  Prior-shift invariance:  g_theta(D; pi) = c + gamma*logit(pi) + h(ell(D)).
      Bayes predicts c=0, gamma=1, h shared across pi.  A fixed-prior agent gives gamma=0.
  E2  Independent-evidence additivity:  A_theta = g(D1 u D2) - g(D1) - g(D2) + a_theta ~ 0,
      and E|A_theta| must be small relative to the total posterior-odds change and DECLINE
      with training (dose checkpoints).
  E3  Evidence sufficiency:  g depends on D only through ell(D).  Residual
      r(D) = g(D) - a_theta - h(ell(D)) must be unexplained by nuisance properties
      (context size, coefficient magnitude, input scale, output variance, noise scale,
      family-conditional predictive separation).
  E4  Sequential/martingale coherence:  w_theta(D_n) ~ E_{p*(.|D_n)}[w_theta(D_{n+1})],
      i.e. the induced posterior is (close to) a martingale under the oracle posterior
      predictive.  Necessary for Bayesian coherence (not a mechanism claim).

Operationalization of w_theta(D).  The model is a 100-bin autoregressive conditional
predictor p_theta(y|x*,D).  The oracle family-conditional posterior predictives
q_G(y|x*,D) = int p(y|x*,Sigma,G) p(Sigma|D,Z=G) dSigma are computed by quadrature over the
Sigma grid (vectorized).  w_theta(D) is the mixture weight that best explains the model's
predictive as w*q1 + (1-w)*q2:
      w_theta(D) = argmax_w sum_{q,b} p_theta(b|x*_q,D) log(w q1(b|x*_q,D) + (1-w) q2(b|x*_q,D)).
The vectorized AL oracle is copied VERBATIM from E1_complexity_capacity.py / b2_calib_floor.py
and self-checked against the validated scalar oracle G.oracle_posterior_al before any number
is read (faithfulness gate).

Model checkpoints: dose_nets/M_{scale}_{prior}_s{seed}_dose{step}.pt  (base: 16 seeds x
[0,100,300,1000,3000,6000,12000]; xl sparser).  CPU only.

Usage:
  python3 stage1_functional_law.py --smoke
  python3 stage1_functional_law.py --prior AL40 --scale base --seeds 3 --dose-steps 0,100,300,1000,3000,6000,12000
"""
import sys, os, json, glob, time, math, argparse, re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.special import logsumexp
import scipy.stats as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- locate the e18b-committed modules (local AND cluster layouts) -------------------------------
def _find_src():
    cands = [os.environ.get("STAGE1_SRC", "")]
    # local:  pfn-dag/evidence-integration  -> ../G-experiments/e18b-committed
    # cluster:pfn-dag-e18b/evidence-integration -> ..   (e18b mirror at the cluster root)
    cands += [os.path.normpath(os.path.join(HERE, "..", "G-experiments", "e18b-committed")),
              os.path.normpath(os.path.join(HERE, "..")),
              os.path.normpath(os.path.join(HERE, "..", "..", "G-experiments", "e18b-committed"))]
    for p in cands:
        if p and os.path.exists(os.path.join(p, "e21_fleet.py")):
            return p
    raise SystemExit(f"[stage1] cannot locate e21_fleet.py (tried {cands}); set STAGE1_SRC")
SRC = _find_src()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
DOSE = os.environ.get("DOSE_OUT", os.path.join(SRC, "dose_nets"))

# import the guarded modules with argv stripped (they parse sys.argv at import time)
_argv = sys.argv
sys.argv = sys.argv[:1]
import e21_fleet as EF            # PFNModel, SCALES, N_CONTEXT, N_BINS, BIN_EDGES, NULL_TOK
import d5c_analyze as A           # E = experiment_v3bump (construction/oracle), G = d5c_gate0
sys.argv = _argv
E = A.E
G = A.G

torch.set_num_threads(int(os.environ.get("NTHREAD", "8")))

N_BINS = E.N_BINS
BIN_CENTERS = 0.5 * (E.BIN_EDGES[:-1] + E.BIN_EDGES[1:])
NULL_TOK = 2
AL_PRIORS = ("AL15", "AL20", "AL25", "AL30", "AL35", "AL40", "AL45", "AL50", "L")
R_OF = {p: (1.0 if p == "L" else int(p[2:]) / 10.0) for p in AL_PRIORS}
R_OF.update({"N": None, "T3": None, "T5": None})

t0 = time.time()
def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# ===================================================================================================
# spline basis (clamped B-splines, Cox-de Boor; dependency-free fallback)
# ===================================================================================================
def _bsp_basis(x, j, p, t):
    """Single clamped B-spline basis function B_{j,p}(x) via Cox-de Boor recursion."""
    if p == 0:
        return ((t[j] <= x) & (x < t[j + 1])).astype(float)
    v = np.zeros_like(x, float)
    d1 = t[j + p] - t[j]
    d2 = t[j + p + 1] - t[j + 1]
    if d1 > 0:
        v += (x - t[j]) / d1 * _bsp_basis(x, j, p - 1, t)
    if d2 > 0:
        v += (t[j + p + 1] - x) / d2 * _bsp_basis(x, j + 1, p - 1, t)
    return v


def bspline_design(x, n_knots=8, degree=3):
    """Clamped B-spline design matrix with quantile interior knots.  Returns (n, n_basis), t."""
    x = np.asarray(x, float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(hi - lo) or hi - lo < 1e-9:
        hi = lo + 1.0
    qs = np.linspace(0, 1, n_knots + 2)[1:-1]
    interior = np.quantile(x, qs)
    t = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    return bspline_design_fixed(x, t, degree), t


def bspline_design_fixed(x, t, degree=3):
    """Evaluate the clamped B-spline design on a FIXED knot vector t (n, n_basis)."""
    x = np.asarray(x, float)
    n_basis = len(t) - degree - 1
    X = np.zeros((len(x), n_basis), float)
    for j in range(n_basis):
        X[:, j] = _bsp_basis(x, j, degree, t)
    # right-endpoint: include x == hi in the final active interval
    hi = t[-1]
    end = np.isclose(x, hi)
    if end.any() and n_basis:
        X[end, -1] = 1.0
        X[end, :-1] = 0.0
    return X


# ===================================================================================================
# AL / Gaussian / Student-t residual log-densities (vectorized; matches E1_complexity_capacity)
# ===================================================================================================
def resid_mat(R, B, prior):
    """Log-density of residuals R (G, n) under per-row scales B (G,), prior noise family.
    AL construction (from d5c_gate0): e = (L1 - a) - (L2 - c), L1~Exp(mean a), L2~Exp(mean c),
    r = a/c.  Gaussian (N): sd = sqrt(2) B  (Var=2B^2).  Student-t: t(nu) scaled to Var=2B^2."""
    if prior == "L" or prior.startswith("AL"):
        r = 1.0 if prior == "L" else int(prior[2:]) / 10.0
        c = np.sqrt(2.0 * B * B / (1.0 + r * r))
        a = r * c
        z = R + (a - c)[:, None]
        return np.where(z >= 0, -z / a[:, None], z / c[:, None]) - np.log(a + c)[:, None]
    if prior == "N":
        s2b = math.sqrt(2.0) * B
        return st.norm.logpdf(R, 0.0, s2b[:, None])
    if prior.startswith("T"):
        nu = float(prior[1:])
        s = B * math.sqrt(2 * (nu - 2) / nu)
        return st.t.logpdf(R / s[:, None], nu) - np.log(s)[:, None]
    raise ValueError(f"unsupported prior {prior}")


# ===================================================================================================
# vectorized AL oracle over the Fix-B Sigma grid
# ===================================================================================================
class GridOracle:
    """Precomputes the sigma-grid parameter arrays and, for a FIXED query grid, the per-grid-point
    family-conditional bin predictives.  Per-context work then reduces to two weighted sums."""

    def __init__(self, prior, quad=15, queries=None):
        self.prior = prior
        if prior.startswith("AL"):
            self.r = int(prior[2:]) / 10.0
        elif prior == "L":
            self.r = 1.0
        else:
            self.r = None
        self.queries = (np.linspace(-3, 3, 7) if queries is None
                        else np.asarray(queries, float))
        self.nq = len(self.queries)
        s1v, s2v, rho_v, lw1, lw2, lwr = E.make_sigma_grid(quad)
        GLW, A1, B11, B12, A2, B21, B22 = ([] for _ in range(7))
        for i, s1 in enumerate(s1v):
            for j, s2 in enumerate(s2v):
                for k, rho in enumerate(rho_v):
                    Sig = E.build_sigma_from_grid_point(s1, s2, rho)
                    if not E.accept_Sigma(Sig):
                        continue
                    a1, b11, b12 = E.sigma_to_params_G1(Sig)
                    a2, b21, b22 = E.sigma_to_params_G2(Sig)
                    GLW.append(lw1[i] + lw2[j] + lwr[k])
                    A1.append(a1); B11.append(b11); B12.append(b12)
                    A2.append(a2); B21.append(b21); B22.append(b22)
        self.GLW = np.asarray(GLW, float)
        self.A1 = np.asarray(A1, float); self.B11 = np.asarray(B11, float); self.B12 = np.asarray(B12, float)
        self.A2 = np.asarray(A2, float); self.B21 = np.asarray(B21, float); self.B22 = np.asarray(B22, float)
        self.Gp = len(self.GLW)
        # per-(grid point, query) family-conditional log-probabilities over the 100 bins
        self.cp1 = np.zeros((self.Gp, self.nq, N_BINS), np.float64)
        self.cp2 = np.zeros((self.Gp, self.nq, N_BINS), np.float64)
        for qi, xq in enumerate(self.queries):
            lp1 = resid_mat(BIN_CENTERS[None, :] - self.A1[:, None] * xq, self.B12, prior)
            self.cp1[:, qi, :] = lp1 - logsumexp(lp1, axis=1)[:, None]
            lp2 = (resid_mat(BIN_CENTERS[None, :], self.B21, prior)
                   + resid_mat(xq - self.A2[:, None] * BIN_CENTERS[None, :], self.B22, prior))
            self.cp2[:, qi, :] = lp2 - logsumexp(lp2, axis=1)[:, None]

    # ---- context log-likelihoods under each family ----
    def ctx_loglik(self, D):
        X = np.asarray(D[:, 0], float); Y = np.asarray(D[:, 1], float)
        K = len(X)
        Xr = np.broadcast_to(X[None, :], (self.Gp, K))
        Yr = np.broadcast_to(Y[None, :], (self.Gp, K))
        lG1 = (self.GLW + resid_mat(Xr, self.B11, self.prior).sum(1)
               + resid_mat(Yr - self.A1[:, None] * X, self.B12, self.prior).sum(1))
        lG2 = (self.GLW + resid_mat(Yr, self.B21, self.prior).sum(1)
               + resid_mat(Xr - self.A2[:, None] * Y, self.B22, self.prior).sum(1))
        return lG1, lG2

    def log_evidence(self, D):
        lG1, lG2 = self.ctx_loglik(D)
        return float(logsumexp(lG1) - logsumexp(lG2))

    def oracle_posterior(self, D):
        return _sigmoid(self.log_evidence(D))

    # ---- full per-context oracle bundle (cacheable) ----
    def eval_context(self, D):
        lG1, lG2 = self.ctx_loglik(D)
        lm1, lm2 = float(logsumexp(lG1)), float(logsumexp(lG2))
        ell = lm1 - lm2
        w1 = np.exp(lG1 - lm1)              # within-family posterior over grid points
        w2 = np.exp(lG2 - lm2)
        q1 = np.exp(logsumexp(self.cp1.reshape(self.Gp, -1), axis=0, b=w1[:, None])).reshape(self.nq, N_BINS)
        q2 = np.exp(logsumexp(self.cp2.reshape(self.Gp, -1), axis=0, b=w2[:, None])).reshape(self.nq, N_BINS)
        q1 /= q1.sum(1, keepdims=True); q2 /= q2.sum(1, keepdims=True)
        tv = float(0.5 * np.abs(q1 - q2).sum(1).mean())
        return dict(ell=ell, q1=q1, q2=q2, tv=tv, w_star=_sigmoid(ell))

    # ---- faithful vs the validated scalar oracle ----
    def faithfulness(self, rng, n=4):
        md = 0.0
        for _ in range(n):
            Sig, *_ = E.sample_valid_Sigma(rng)
            cls = 1 if rng.random() < 0.5 else 2
            p = E.sigma_to_params_G1(Sig) if cls == 1 else E.sigma_to_params_G2(Sig)
            D = G.sample_data_al(cls, p, E.N_CONTEXT, rng, self.r)
            pv = _sigmoid(self.log_evidence(D))
            po = G.oracle_posterior_al(D, E.make_sigma_grid(15), self.r)
            md = max(md, abs(pv - po))
        log(f"  faithfulness: max|vectorized - scalar oracle| = {md:.2e} (gate < 1e-6)")
        assert md < 1e-6, f"vectorized oracle diverges from validated scalar oracle ({md})"


class OracleCache:
    """Memoizes eval_context for repeated D arrays (shared across dose steps / nets)."""
    def __init__(self, orc):
        self.orc = orc
        self.cache = {}

    def get(self, D):
        key = np.ascontiguousarray(D, np.float64).tobytes()
        v = self.cache.get(key)
        if v is None:
            v = self.orc.eval_context(D)
            self.cache[key] = v
        return v


# ===================================================================================================
# small numeric helpers
# ===================================================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def al_sample(b, r, n, rng):
    """Asymmetric-Laplace sample with scale b and skew r (from d5c_gate0)."""
    c = math.sqrt(2.0 * b * b / (1.0 + r * r))
    a = r * c
    return rng.exponential(a, n) - rng.exponential(c, n) - (a - c)


def gen_context(rng, r, K, base_rate=0.5):
    """Draw one context of K rows from the e18b generative process: Sigma~Fix-B, Z~Bernoulli(base_rate),
    then K iid (x,y) from the family-conditional AL linear SEM.  Returns (D, cls, Sigma, params)."""
    Sig, *_ = E.sample_valid_Sigma(rng)
    cls = 1 if rng.random() < base_rate else 2
    p = E.sigma_to_params_G1(Sig) if cls == 1 else E.sigma_to_params_G2(Sig)
    D = G.sample_data_al(cls, p, K, rng, r)
    return D, cls, Sig, p


def load_net(scale, prior, seed, step):
    path = os.path.join(DOSE, f"M_{scale}_{prior}_s{seed}_dose{step}.pt")
    if not os.path.exists(path):
        return None
    net = EF.PFNModel(**EF.SCALES[scale])
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


# ===================================================================================================
# model predictive + mixture-weight recovery
# ===================================================================================================
@torch.no_grad()
def model_bin_probs(net, D, queries):
    """Model's predicted y-distribution over the 100 bins for each query x*.  Returns (Q, N_BINS)."""
    ct = torch.as_tensor(np.asarray(D, np.float32)[None], dtype=torch.float32)
    qx = torch.as_tensor(queries[None, :, None], dtype=torch.float32)
    tok = torch.tensor([NULL_TOK])
    logits = net(ct, qx, tok)[0]
    return torch.softmax(logits, dim=-1).numpy()


def _mixture_argmax(P, q1, q2, w_grid=401):
    """Grid + golden-section argmax of sum P log(w q1 + (1-w) q2) over w in [0,1]."""
    def negL(w):
        mix = w * q1 + (1.0 - w) * q2
        return -float((P * np.log(mix)).sum())

    ws = np.linspace(0.0, 1.0, w_grid)
    Ls = np.array([-negL(w) for w in ws])
    i0 = int(np.argmax(Ls))
    lo = ws[max(0, i0 - 1)]; hi = ws[min(len(ws) - 1, i0 + 1)]
    if hi - lo < 1e-4:
        w = ws[i0]
    else:
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a); d = a + gr * (b - a)
        fc, fd = negL(c), negL(d)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a); fc = negL(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a); fd = negL(d)
            if abs(b - a) < 1e-6:
                break
        w = 0.5 * (a + b)
    return min(1.0 - 1e-6, max(1e-6, float(w))), ws


def recover_w(net, D, oq, queries, w_grid=401):
    """Recover the induced posterior weight w_theta(D) by fitting the model predictive as a
    mixture w*q1 + (1-w)*q2.  Returns a dict with w, g=logit(w), per-query w, fit quality."""
    P = model_bin_probs(net, D, queries)                    # (Q, N_BINS)
    P = np.clip(P, 1e-12, 1.0)
    P /= P.sum(1, keepdims=True)
    q1 = np.clip(oq["q1"], 1e-12, 1.0)
    q2 = np.clip(oq["q2"], 1e-12, 1.0)

    w, ws = _mixture_argmax(P, q1, q2, w_grid)
    g = math.log(w / (1.0 - w))
    # per-query weights (diagnostic: how well 'one w' fits across queries)
    wq = []
    for qi in range(P.shape[0]):
        p1 = q1[qi]; p2 = q2[qi]; pp = P[qi]
        Lq = [float((pp * np.log(np.clip(w * p1 + (1 - w) * p2, 1e-12, 1))).sum()) for w in ws]
        wq.append(float(ws[int(np.argmax(Lq))]))
    mix = np.clip(w * q1 + (1.0 - w) * q2, 1e-12, 1.0)
    ce_mix = -float((P * np.log(mix)).sum())
    ce1 = -float((P * np.log(q1)).sum())
    ce2 = -float((P * np.log(q2)).sum())
    return dict(w=w, g=g, w_per_query=wq, wq_std=float(np.std(wq)), tv=oq["tv"],
                ce_mix=ce_mix, ce1=ce1, ce2=ce2,
                fit_gain=max(0.0, min(ce1, ce2) - ce_mix))


# ===================================================================================================
# functional-law fit (clamped spline) + OLS/CV helpers
# ===================================================================================================
def _std_cols(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def ols_ridge(X, y, ridge=1e-3):
    Xa = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    n, k = Xa.shape
    A = Xa.T @ Xa + ridge * np.eye(k)
    A[0, 0] -= ridge            # do not penalize the intercept
    return np.linalg.solve(A, Xa.T @ y)


def r2_score(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cv_r2(X, y, folds=5, ridge=1e-3, seed=0):
    """5-fold CV R^2 for ridge regression of y on X (X already includes the spline basis)."""
    n = len(y)
    if n < 2 * folds:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold = np.array_split(idx, folds)
    yh = np.empty(n)
    for f in fold:
        tr = np.setdiff1d(idx, f)
        Xs, mu, sd = _std_cols(X[tr])
        beta = ols_ridge(Xs, y[tr], ridge)
        Xte = (X[f] - mu) / sd
        yh[f] = np.concatenate([np.ones((len(f), 1)), Xte], axis=1) @ beta
    return float(r2_score(y, yh))


class LawFit:
    """Functional law g ~ a + h(ell) with h a clamped cubic spline (quantile knots)."""

    def __init__(self, ells, gs, n_knots=8, degree=3, ridge=1e-3):
        self.ells = np.asarray(ells, float)
        self.gs = np.asarray(gs, float)
        B, t = bspline_design(self.ells, n_knots=n_knots, degree=degree)
        self.B, self.knots, self.degree = B, t, degree
        Xs, mu, sd = _std_cols(B)
        self.mu, self.sd = mu, sd
        beta_full = ols_ridge(Xs, self.gs, ridge)   # [intercept, beta_1..k]
        self.intercept = float(beta_full[0])
        self.beta = beta_full[1:]
        self.r2 = r2_score(self.gs, self.predict(self.ells))
        self.n = len(self.ells)

    def predict(self, ells):
        ells = np.asarray(ells, float)
        # clamped spline is only identified inside [knots[0], knots[-1]]; clip for out-of-range eval
        lo, hi = self.knots[0], self.knots[-1]
        ells = np.clip(ells, lo, hi)
        B = bspline_design_fixed(ells, self.knots, self.degree)
        Bs = (B - self.mu) / self.sd
        return float(self.intercept) + Bs @ self.beta

    def a_theta(self):
        return float(self.predict(np.array([0.0])))

    def residual(self, gs, ells):
        return np.asarray(gs, float) - self.predict(ells)


# ===================================================================================================
# pool generation (net-independent; computed once in main)
# ===================================================================================================
def gen_e1_pool(orc, r, priors, n_per_prior, seed):
    rng = np.random.default_rng(seed)
    pools = {}
    for pi in priors:
        ctxs = [gen_context(rng, r, E.N_CONTEXT, base_rate=pi) for _ in range(n_per_prior)]
        pools[str(pi)] = ctxs
    return pools


def gen_e2_pairs(orc, r, n_pairs, sizes, seed):
    """Independent row blocks D1 (n1) and D2 (n2) from the same latent task."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        n1, n2 = sizes[rng.integers(len(sizes))], sizes[rng.integers(len(sizes))]
        Sig, *_ = E.sample_valid_Sigma(rng)
        cls = 1 if rng.random() < 0.5 else 2
        p = E.sigma_to_params_G1(Sig) if cls == 1 else E.sigma_to_params_G2(Sig)
        D1 = G.sample_data_al(cls, p, int(n1), rng, r)
        D2 = G.sample_data_al(cls, p, int(n2), rng, r)
        pairs.append(dict(D1=D1, D2=D2, n1=int(n1), n2=int(n2), cls=cls, params=p))
    return pairs


def gen_e3_pool(orc, r, n_ctx, ks, seed):
    """Varied-K contexts with recorded generative properties (nuisance features)."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_ctx):
        K = int(ks[rng.integers(len(ks))])
        D, cls, Sig, p = gen_context(rng, r, K, base_rate=0.5)
        a, b1, b2 = p
        pool.append(dict(D=D, K=K, cls=cls, a=a, b1=b1, b2=b2,
                         log_std_x=float(np.log(np.std(D[:, 0]) + 1e-9)),
                         log_std_y=float(np.log(np.std(D[:, 1]) + 1e-9)),
                         mean_abs_x=float(np.mean(np.abs(D[:, 0]))),
                         mean_abs_y=float(np.mean(np.abs(D[:, 1])))))
    return pool


def gen_e4_pool(orc, r, n_ctx, n_samples, seed):
    """D_n contexts + posterior-predictive continuations D_{n+1} (net-independent)."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_ctx):
        D, cls, Sig, p = gen_context(rng, r, E.N_CONTEXT, base_rate=0.5)
        lG1, lG2 = orc.ctx_loglik(D)
        w_star = _sigmoid(float(logsumexp(lG1) - logsumexp(lG2)))
        w1 = np.exp(lG1 - logsumexp(lG1)); w2 = np.exp(lG2 - logsumexp(lG2))
        cont = []
        for _ in range(n_samples):
            fam = 1 if rng.random() < w_star else 2
            gp = int(rng.choice(orc.Gp, p=(w1 if fam == 1 else w2)))
            if fam == 1:
                a_, b1, b2 = orc.A1[gp], orc.B11[gp], orc.B12[gp]
                e1 = al_sample(b1, r, 1, rng)[0]; e2 = al_sample(b2, r, 1, rng)[0]
                x, y = e1, a_ * e1 + e2
            else:
                a_, b1, b2 = orc.A2[gp], orc.B21[gp], orc.B22[gp]
                e1 = al_sample(b1, r, 1, rng)[0]; e2 = al_sample(b2, r, 1, rng)[0]
                y, x = e1, a_ * e1 + e2
            cont.append(np.concatenate([D, np.array([[x, y]])], axis=0))
        pool.append(dict(D=D, cont=cont, w_star=w_star, cls=cls))
    return pool


# ===================================================================================================
# E1 — prior-shift invariance
# ===================================================================================================
def e1_joint_fit(rows):
    """rows: list of dict(pi, logit_pi, ell, g).  Fits
       g = c + gamma*logit_pi + h(ell)   (spline h)
    and the nested models: no-prior (gamma=0) and per-prior h (interaction)."""
    n = len(rows)
    if n < 40:
        return dict(n=n, ok=False, reason="too few contexts")
    logit_pi = np.array([r["logit_pi"] for r in rows], float)
    ell = np.array([r["ell"] for r in rows], float)
    g = np.array([r["g"] for r in rows], float)
    B, _ = bspline_design(ell, n_knots=8, degree=3)
    Bs, mu, sd = _std_cols(B)

    def fit_model(cols, ridge=1e-6):
        X = np.concatenate(cols, axis=1)
        Xs, m, s = _std_cols(X)
        beta_full = ols_ridge(Xs, g, ridge)         # [intercept, beta_1..k]
        inter = float(beta_full[0]); beta = beta_full[1:]
        yh = inter + (X - m) / s @ beta
        return inter, beta, r2_score(g, yh), yh

    # model 0: shared h only (no prior term)
    i0, b0, r0, yh0 = fit_model([Bs])
    # model 1: shared h + linear prior term
    i1, b1, r1, yh1 = fit_model([Bs, logit_pi[:, None]])
    gamma = float(b1[-1])
    # model 2: per-prior h (full interaction)
    priors = sorted(set(r["pi"] for r in rows))
    cols = []
    for pp in priors:
        msk = np.array([r["pi"] == pp for r in rows])
        cols.append(msk[:, None])
        cols.append(msk[:, None] * Bs)
    i2, b2, r2, yh2 = fit_model(cols)
    # F-tests (in-sample RSS, OLS-free approx with ridge; interpret as descriptive)
    def fstat(r_full, r_red, k_full, k_red):
        rss_f = (1 - r_full) * g.var() * n
        rss_r = (1 - r_red) * g.var() * n
        if rss_f <= 0:
            return float("nan"), float("nan")
        df1 = k_full - k_red
        df2 = n - k_full - 1
        F = ((rss_r - rss_f) / df1) / (rss_f / max(df2, 1))
        return float(F), float(st.f.sf(F, df1, df2))
    k0 = 1 + Bs.shape[1]
    k1 = k0 + 1
    k2 = 1 + len(priors) * (1 + Bs.shape[1])
    F_gamma, p_gamma = fstat(r1, r0, k1, k0)
    F_int, p_int = fstat(r2, r1, k2, k1)
    r0_cv = cv_r2(Bs, g); r1_cv = cv_r2(np.concatenate([Bs, logit_pi[:, None]], 1), g)
    return dict(n=n, priors=priors,
                gamma=gamma,
                r2_no_prior=r0, r2_joint=r1, r2_interaction=r2,
                r2cv_no_prior=r0_cv, r2cv_joint=r1_cv,
                delta_r2_prior=r1 - r0, delta_r2_interaction=r2 - r1,
                F_gamma=F_gamma, p_gamma=p_gamma, F_interaction=F_int, p_interaction=p_int,
                ok=True)


def run_e1(net, orc, ocache, pools, queries):
    out = {"per_prior": {}}
    rows = []
    for pstr, ctxs in pools.items():
        pi = float(pstr)
        lpi = math.log(pi / (1.0 - pi))
        recs = []
        for D, cls, Sig, p in ctxs:
            oq = ocache.get(D)
            rw = recover_w(net, D, oq, queries)
            recs.append(dict(ell=oq["ell"], g=rw["g"], w=rw["w"], w_star=oq["w_star"],
                             wq_std=rw["wq_std"], fit_gain=rw["fit_gain"]))
            rows.append(dict(pi=pi, logit_pi=lpi, ell=oq["ell"], g=rw["g"]))
        ells = np.array([r["ell"] for r in recs]); gs = np.array([r["g"] for r in recs])
        lw = np.array([math.log(pi / (1 - pi)) + r["ell"] for r in recs])
        out["per_prior"][pstr] = dict(
            n=len(recs),
            mean_ell=float(ells.mean()), mean_g=float(gs.mean()),
            corr_g_ell=float(np.corrcoef(ells, gs)[0, 1]) if len(recs) > 2 else float("nan"),
            corr_g_logitWpi=float(np.corrcoef(gs, lw)[0, 1]) if len(recs) > 2 else float("nan"),
            g_at_ell0=float(np.interp(0.0, np.sort(ells), np.sort(gs))) if len(recs) > 1 else float("nan"),
            mean_wq_std=float(np.mean([r["wq_std"] for r in recs])),
            mean_fit_gain=float(np.mean([r["fit_gain"] for r in recs])))
    out["joint"] = e1_joint_fit(rows)
    return out


# ===================================================================================================
# E2 — independent-evidence additivity
# ===================================================================================================
def run_e2(net, orc, ocache, pairs, a_theta, queries):
    As = []; denoms = []; recs = []
    for pr in pairs:
        o1 = ocache.get(pr["D1"]); o2 = ocache.get(pr["D2"])
        D12 = np.concatenate([pr["D1"], pr["D2"]], axis=0)
        o12 = ocache.get(D12)
        r1 = recover_w(net, pr["D1"], o1, queries)
        r2 = recover_w(net, pr["D2"], o2, queries)
        r12 = recover_w(net, D12, o12, queries)
        A = r12["g"] - r1["g"] - r2["g"] + a_theta
        denom = abs(r12["g"] - a_theta)
        ell1, ell2, ell12 = o1["ell"], o2["ell"], o12["ell"]
        lsum = abs(ell1) + abs(ell2)
        conc = max(abs(ell1), abs(ell2)) / (lsum + 1e-9)
        agree = "agreement" if (ell1 * ell2 > 0) else ("conflict" if (ell1 * ell2 < 0) else "ambiguous")
        tbin = "high" if abs(ell12) >= 8.0 else ("mid" if abs(ell12) >= 3.0 else "low")
        As.append(A); denoms.append(denom)
        recs.append(dict(n1=pr["n1"], n2=pr["n2"], ell1=ell1, ell2=ell2, ell12=ell12,
                         g1=r1["g"], g2=r2["g"], g12=r12["g"], A=A, denom=denom,
                         agree=agree, conc=conc, tbin=tbin,
                         wq1=r1["wq_std"], wq2=r2["wq_std"], wq12=r12["wq_std"],
                         fg1=r1["fit_gain"], fg2=r2["fit_gain"], fg12=r12["fit_gain"]))
    As = np.array(As); denoms = np.array(denoms)
    mean_denom = float(denoms.mean())
    mean_absA = float(np.abs(As).mean())
    med_absA = float(np.median(np.abs(As)))
    ratio = mean_absA / mean_denom if mean_denom > 1e-9 else float("nan")
    wq_all = np.array([r["wq1"] for r in recs] + [r["wq2"] for r in recs] + [r["wq12"] for r in recs])
    fg_all = np.array([r["fg1"] for r in recs] + [r["fg2"] for r in recs] + [r["fg12"] for r in recs])

    def strat(key, fn):
        groups = {}
        for r_ in recs:
            k = fn(r_)
            groups.setdefault(k, []).append(r_)
        return {k: dict(n=len(v), mean_absA=float(np.mean([abs(x["A"]) for x in v])),
                        mean_denom=float(np.mean([x["denom"] for x in v])),
                        ratio=(float(np.mean([abs(x["A"]) for x in v]) /
                               max(np.mean([x["denom"] for x in v]), 1e-9)))) for k, v in groups.items()}

    out = dict(n=len(recs), mean_absA=mean_absA, med_absA=med_absA,
               mean_abs_denom=mean_denom, ratio=ratio,
               mean_absA_over_mean_abs_denom=ratio,
               mean_wq_std=float(wq_all.mean()), mean_fit_gain=float(fg_all.mean()),
               strata=dict(
                   agreement=strat("agree", lambda r_: r_["agree"]),
                   block_size=strat("bs", lambda r_: f"{min(r_['n1'],r_['n2'])}-{max(r_['n1'],r_['n2'])}"),
                   total_evidence=strat("tbin", lambda r_: r_["tbin"]),
                   concentration=strat("conc", lambda r_: ("conc" if r_["conc"] >= 0.8 else "balanced"))))
    # compact per-pair detail for downstream re-analysis
    out["_recs"] = [{k: r_[k] for k in ("n1", "n2", "ell1", "ell2", "ell12", "g1", "g2", "g12",
                                         "A", "denom", "agree")} for r_ in recs]
    return out


# ===================================================================================================
# E3 — evidence-sufficiency counterfactuals
# ===================================================================================================
def e3_nuisance_regression(resids, features, folds=5):
    """features: dict name -> array.  Returns CV R^2 of residual on all nuisances and per-nuisance."""
    names = list(features)
    Xall = np.column_stack([features[nm] for nm in names])
    # standardize
    Xs = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-9)
    r2_all = cv_r2(Xs, resids, folds)
    per = {}
    for i, nm in enumerate(names):
        per[nm] = dict(r2=cv_r2(Xs[:, i:i + 1], resids, folds),
                       corr=float(np.corrcoef(Xs[:, i], resids)[0, 1]))
    return dict(r2_all=r2_all, per_nuisance=per)


def run_e3(net, orc, ocache, pool, law, queries):
    resids = []; feats = {k: [] for k in ("K", "log_std_x", "log_std_y", "coef_mag",
                                           "noise_scale", "mean_abs_x", "separation")}
    recs = []
    for it in pool:
        D = it["D"]
        oq = ocache.get(D)
        rw = recover_w(net, D, oq, queries)
        res = float(rw["g"] - law.predict(np.array([oq["ell"]])))
        resids.append(res)
        feats["K"].append(float(it["K"]))
        feats["log_std_x"].append(it["log_std_x"])
        feats["log_std_y"].append(it["log_std_y"])
        feats["coef_mag"].append(abs(it["a"]))
        feats["noise_scale"].append(0.5 * (it["b1"] + it["b2"]))
        feats["mean_abs_x"].append(it["mean_abs_x"])
        feats["separation"].append(oq["tv"])
        recs.append(dict(ell=oq["ell"], g=rw["g"], residual=res, K=it["K"], tv=oq["tv"],
                         wq_std=rw["wq_std"], fit_gain=rw["fit_gain"]))
    resids = np.array(resids)
    reg = e3_nuisance_regression(resids, {k: np.array(v) for k, v in feats.items()})
    # within-ell-bin correlation of g with each nuisance (matched on ell)
    ells = np.array([r["ell"] for r in recs])
    gs = np.array([r["g"] for r in recs])
    bins = np.quantile(ells, np.linspace(0, 1, 6))
    bin_corr = {k: [] for k in ("K", "coef_mag", "log_std_x", "separation")}
    for b in range(5):
        lo, hi = bins[b], bins[b + 1]
        msk = (ells >= lo - 1e-9) & (ells < hi + 1e-9)
        if msk.sum() >= 5:
            for k in bin_corr:
                v = np.array([feats[k][i] for i in range(len(recs)) if msk[i]])
                bin_corr[k].append(float(np.corrcoef(v, gs[msk])[0, 1]) if v.std() > 0 else float("nan"))
    out = dict(n=len(recs),
               nuisance_R2=reg["r2_all"],
               per_nuisance=reg["per_nuisance"],
               within_ell_bin_corr={k: float(np.nanmean(v)) if v else float("nan")
                                    for k, v in bin_corr.items()},
               mean_abs_residual=float(np.abs(resids).mean()),
               residual_sd=float(resids.std()),
               mean_wq_std=float(np.mean([r["wq_std"] for r in recs])),
               mean_fit_gain=float(np.mean([r["fit_gain"] for r in recs])))
    # matched pairs: within ell bins, greedy pair by max K-gap; report dg
    dg = []; dnu = []
    for b in range(5):
        lo, hi = bins[b], bins[b + 1]
        msk = [i for i in range(len(recs)) if (ells[i] >= lo - 1e-9) and (ells[i] < hi + 1e-9)]
        used = set()
        for i in range(len(msk)):
            if msk[i] in used:
                continue
            best = None; bg = -1.0
            for j in range(i + 1, len(msk)):
                if msk[j] in used:
                    continue
                gap = abs(feats["K"][msk[i]] - feats["K"][msk[j]])
                if gap > bg:
                    bg = gap; best = msk[j]
            if best is not None:
                used.add(msk[i]); used.add(best)
                dg.append(abs(gs[msk[i]] - gs[best]))
                dnu.append(bg)
    if dg:
        out["matched_pairs_K"] = dict(n=len(dg), mean_abs_dg=float(np.mean(dg)),
                                      mean_dK=float(np.mean(dnu)))
    return out


# ===================================================================================================
# E4 — sequential/martingale coherence
# ===================================================================================================
def run_e4(net, orc, ocache, e4_pool, queries):
    disc = []; absdisc = []; oracle_disc = []; paired_diff = []; paired_abs = []
    per_ctx = []
    for it in e4_pool:
        D = it["D"]
        o0 = ocache.get(D)
        rw0 = recover_w(net, D, o0, queries)
        w0 = rw0["w"]
        wc = []; wc_star = []
        for Dc in it["cont"]:
            oc = ocache.get(Dc)
            rwc = recover_w(net, Dc, oc, queries)
            wc.append(rwc["w"]); wc_star.append(oc["w_star"])
        wc = np.array(wc); wc_star = np.array(wc_star)
        d = float(w0 - wc.mean())
        disc.append(d); absdisc.append(abs(d))
        # oracle positive control: posterior martingale should hold (same samples, paired)
        do = float(it["w_star"] - wc_star.mean())
        oracle_disc.append(do)
        pd = d - do                        # model violation minus oracle MC noise (paired)
        paired_diff.append(pd); paired_abs.append(abs(pd))
        per_ctx.append(dict(w0=w0, w_next_mean=float(wc.mean()), w_star0=it["w_star"],
                            oracle_w_next_mean=float(wc_star.mean()), disc=d, oracle_disc=do,
                            paired_diff=pd, n=len(wc), wq_std0=rw0["wq_std"]))
    disc = np.array(disc); absdisc = np.array(absdisc); od = np.array(oracle_disc)
    pd = np.array(paired_diff); pa = np.array(paired_abs)
    return dict(n=len(e4_pool),
                mean_discrepancy=float(disc.mean()),
                mean_abs_discrepancy=float(absdisc.mean()),
                sd_discrepancy=float(disc.std()),
                oracle_control_mean_abs=float(np.abs(od).mean()),
                oracle_control_mean=float(od.mean()),
                paired_diff_mean=float(pd.mean()),
                paired_diff_mean_abs=float(pa.mean()),
                paired_diff_sd=float(pd.std()),
                mean_wq_std=float(np.mean([it["wq_std0"] for it in per_ctx])),
                per_context=per_ctx)


def selftest(orc, queries, seed=0):
    """Internal consistency checks: mixture-weight recovery recovers a planted w*, and the
    E4 posterior-predictive sampler respects the oracle martingale property."""
    rng = np.random.default_rng(seed)
    # (1) planted-mixture recovery
    errs = []
    for _ in range(8):
        Sig, *_ = E.sample_valid_Sigma(rng)
        cls = 1 if rng.random() < 0.5 else 2
        p = E.sigma_to_params_G1(Sig) if cls == 1 else E.sigma_to_params_G2(Sig)
        D = G.sample_data_al(cls, p, E.N_CONTEXT, rng, orc.r)
        oq = orc.eval_context(D)
        w_true = float(rng.random())
        q1 = np.clip(oq["q1"], 1e-12, 1.0); q2 = np.clip(oq["q2"], 1e-12, 1.0)
        P = w_true * q1 + (1.0 - w_true) * q2
        P /= P.sum(1, keepdims=True)
        w_hat, _ = _mixture_argmax(P, q1, q2)
        errs.append(abs(w_hat - w_true))
    max_err = float(max(errs))
    # (2) oracle martingale control over a handful of contexts
    pool = gen_e4_pool(orc, orc.r, 6, 8, seed + 5)
    ctrl = []
    for it in pool:
        wc_star = np.array([orc.eval_context(Dc)["w_star"] for Dc in it["cont"]])
        ctrl.append(abs(it["w_star"] - wc_star.mean()))
    max_ctrl = float(max(ctrl))
    log(f"  SELFTEST: planted-w recovery maxerr={max_err:.2e} (gate<1e-3) | "
        f"oracle martingale control max={max_ctrl:.2e} (gate<1e-3)")
    assert max_err < 1e-3, f"mixture recovery fails to recover planted w ({max_err})"
    assert max_ctrl < 1e-3, f"E4 sampler violates the oracle martingale property ({max_ctrl})"
    return dict(w_recovery_maxerr=max_err, oracle_martingale_max=max_ctrl)


# ===================================================================================================
# aggregation over seeds
# ===================================================================================================
def agg_over_seeds(per_seed_dicts):
    """Median + per-seed list over a list of per-net result dicts (same structure)."""
    if not per_seed_dicts:
        return None

    def med(vals):
        vals = [v for v in vals
                if v is not None and not (isinstance(v, (float, np.floating)) and math.isnan(float(v)))]
        if not vals:
            return float("nan")
        return float(np.median([float(v) for v in vals]))

    def rec_merge(ds):
        d0 = ds[0]
        res = {"_n_seed": len(ds)}
        for k in d0.keys():
            vals = [d.get(k) for d in ds]
            v = vals[0]
            if v is None:
                res[k] = None
            elif isinstance(v, dict):
                res[k] = rec_merge(vals)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                allkeys = sorted({kk for d in vals if d for it in d for kk in it})
                sub = {}
                n_items = sum(len(d) for d in vals if d)
                for kk in allkeys:
                    items = [it[kk] for d in vals if d for it in d if kk in it]
                    nums = [x for x in items
                            if isinstance(x, (int, float, np.floating))
                            and not (isinstance(x, (float, np.floating)) and math.isnan(float(x)))]
                    strs = [x for x in items if isinstance(x, str)]
                    if nums:
                        sub[kk] = med(nums)
                    elif strs:
                        sub[kk] = strs[0]
                res[k] = {"_n_items": n_items, **sub}
            elif isinstance(v, (int, float, np.floating)):
                res[k] = med(vals)
            elif isinstance(v, str):
                res[k] = v
            else:
                res[k] = vals
        return res

    return rec_merge(per_seed_dicts)


# ===================================================================================================
# main
# ===================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=os.environ.get("STAGE1_PRIOR", "AL40"))
    ap.add_argument("--scale", default=os.environ.get("STAGE1_SCALE", "base"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dose-steps", default="0,100,300,1000,3000,6000,12000")
    ap.add_argument("--final-step", type=int, default=12000)
    ap.add_argument("--quad", type=int, default=15)
    ap.add_argument("--queries", default="-3,3,7")
    ap.add_argument("--n-cal", type=int, default=120)
    ap.add_argument("--n-per-prior", type=int, default=60)
    ap.add_argument("--n-pairs", type=int, default=120)
    ap.add_argument("--n-e3", type=int, default=150)
    ap.add_argument("--n-e4", type=int, default=24)
    ap.add_argument("--e4-samples", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260731)
    a = ap.parse_args()

    if a.smoke:
        a.seeds = min(a.seeds, 2); a.n_cal = 12; a.n_per_prior = 8
        a.n_pairs = 10; a.n_e3 = 12; a.n_e4 = 3; a.e4_samples = 3
        a.dose_steps = "0,12000"; a.final_step = 12000

    steps = [int(s) for s in a.dose_steps.split(",")]
    qs = [float(x) for x in a.queries.split(",")]
    queries = np.linspace(qs[0], qs[1], int(qs[2]))

    out = dict(meta=dict(prior=a.prior, scale=a.scale, seeds=a.seeds, dose_steps=steps,
                         final_step=a.final_step, quad=a.quad, queries=queries.tolist(),
                         smoke=a.smoke, seed=a.seed, src=SRC, dose_dir=DOSE))

    # ---- oracle + faithfulness ----
    log(f"stage1 start: prior={a.prior} scale={a.scale} seeds={a.seeds} steps={steps}")
    orc = GridOracle(a.prior, quad=a.quad, queries=queries)
    r = orc.r                       # single source of truth for the AL skew (None for N/T)
    if r is None:
        raise SystemExit(f"[stage1] prior {a.prior!r} is not an AL/Laplace noise family; "
                         f"the AL data sampler + oracle require r (supported: {AL_PRIORS}).")
    log(f"  oracle built: {orc.Gp} grid points, {orc.nq} queries, AL r={r}")
    orc.faithfulness(np.random.default_rng(7))
    ocache = OracleCache(orc)
    if a.selftest:
        sres = dict(meta=dict(prior=a.prior, scale=a.scale, seed=a.seed, quad=a.quad),
                    selftest=selftest(orc, queries, a.seed))
        sfn = a.out or os.path.join(HERE, "stage1_results", f"selftest_{a.prior}.json")
        os.makedirs(os.path.dirname(sfn), exist_ok=True)
        json.dump(sres, open(sfn, "w"), indent=2)
        log(f"SELFTEST OK -> {sfn}"); return

    # ---- pools (net-independent) ----
    priors = [0.1, 0.2, 0.5, 0.8, 0.9]
    e1_pools = gen_e1_pool(orc, r, priors, a.n_per_prior, a.seed + 1)
    e2_pairs = gen_e2_pairs(orc, r, a.n_pairs, [5, 10, 15], a.seed + 2)
    e3_pool = gen_e3_pool(orc, r, a.n_e3, [10, 20, 30, 40], a.seed + 3)
    e4_pool = gen_e4_pool(orc, r, a.n_e4, a.e4_samples, a.seed + 4)
    log(f"  pools: E1={sum(len(v) for v in e1_pools.values())} E2={len(e2_pairs)} E3={len(e3_pool)} E4={len(e4_pool)}")

    # ---- per-seed / per-step loops ----
    e2_by_step = {str(s): [] for s in steps}
    e1_seeds = []; e3_seeds = []; e4_seeds = []
    law_summary = []
    manifest_missing = []
    for seed in range(a.seeds):
        for step in steps:
            net = load_net(a.scale, a.prior, seed, step)
            if net is None:
                manifest_missing.append(f"M_{a.scale}_{a.prior}_s{seed}_dose{step}.pt")
                continue
            tag = f"s{seed}_dose{step}"
            # calibration law at THIS checkpoint
            cal_Ds = [gctx[0] for gctx in e1_pools["0.5"]]
            cal_ells = [ocache.get(D)["ell"] for D in cal_Ds]
            cal_gs = [recover_w(net, D, ocache.get(D), queries)["g"] for D in cal_Ds]
            law = LawFit(cal_ells, cal_gs)
            a_theta = law.a_theta()
            # E2 (all steps)
            e2r = run_e2(net, orc, ocache, e2_pairs, a_theta, queries)
            e2r["step"] = step; e2r["seed"] = seed; e2r["a_theta"] = a_theta
            e2r["cal_r2"] = law.r2
            e2_by_step[str(step)].append(e2r)
            if step == a.final_step:
                law_summary.append(dict(seed=seed, a_theta=a_theta, r2=law.r2))
                e1_seeds.append(run_e1(net, orc, ocache, e1_pools, queries))
                e3_seeds.append(run_e3(net, orc, ocache, e3_pool, law, queries))
                e4_seeds.append(run_e4(net, orc, ocache, e4_pool, queries))
            log(f"  {tag}: a_theta={a_theta:+.3f} calR2={law.r2:.3f} | "
                f"E2 mean|A|={e2r['mean_absA']:.3f} ratio={e2r['ratio']:.3f}")

    if manifest_missing:
        log(f"  WARNING: {len(manifest_missing)} missing checkpoints, e.g. {manifest_missing[:4]}")
    if not any(e2_by_step.values()):
        raise SystemExit(f"no checkpoints found for {a.scale}/{a.prior} in {DOSE}")

    # ---- aggregate ----
    out["calibration"] = dict(per_seed=law_summary,
                              a_theta_median=float(np.median([x["a_theta"] for x in law_summary])),
                              cal_r2_median=float(np.median([x["r2"] for x in law_summary])))
    out["E2_additivity"] = {s: agg_over_seeds(v) for s, v in e2_by_step.items()}
    # dose gate: paired decline of ratio from first to final step
    s0, sT = str(steps[0]), str(a.final_step)
    if s0 in out["E2_additivity"] and sT in out["E2_additivity"]:
        r0 = out["E2_additivity"][s0]; rT = out["E2_additivity"][sT]
        if r0 and rT:
            _decl = bool(rT.get("ratio", 1e9) < r0.get("ratio", 1e18))
            _small = bool(rT.get("ratio", 1e9) < 0.5)
            out["E2_additivity"]["_gate"] = dict(
                ratio_init=r0.get("ratio"), ratio_final=rT.get("ratio"),
                mean_absA_init=r0.get("mean_absA"), mean_absA_final=rT.get("mean_absA"),
                declined=_decl, final_ratio_small=_small, gate_pass=bool(_decl and _small))
    if e1_seeds:
        out["E1_prior_shift"] = dict(per_seed=e1_seeds, agg=agg_over_seeds(e1_seeds))
        gammas = [x["joint"]["gamma"] for x in e1_seeds if x["joint"].get("ok")]
        out["E1_prior_shift"]["gamma_median"] = float(np.median(gammas)) if gammas else float("nan")
    if e3_seeds:
        out["E3_sufficiency"] = dict(per_seed=e3_seeds, agg=agg_over_seeds(e3_seeds))
    if e4_seeds:
        out["E4_martingale"] = dict(per_seed=e4_seeds, agg=agg_over_seeds(e4_seeds))
    out["wallclock_s"] = round(time.time() - t0, 1)

    fn = a.out or os.path.join(HERE, "stage1_results", f"stage1_{a.prior}_{a.scale}{'_smoke' if a.smoke else ''}.json")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(out, open(fn, "w"), indent=2, default=str)
    log(f"wrote {fn}")
    log(f"DONE in {out['wallclock_s']}s")

    # ---- console summary ----
    print("\n" + "=" * 88)
    print(f"STAGE-1 FUNCTIONAL LAW  prior={a.prior} scale={a.scale}  smoke={a.smoke}")
    if "E1_prior_shift" in out:
        j = out["E1_prior_shift"]["agg"]["joint"]
        g = j.get("gamma") if isinstance(j.get("gamma"), (int, float)) else float("nan")
        print(f"E1  gamma={g:+.3f} (Bayes=1)  dR2(prior)={j.get('delta_r2_prior', float('nan')):+.4f}  "
              f"dR2(interaction)={j.get('delta_r2_interaction', float('nan')):+.4f}  "
              f"p_int={j.get('p_interaction', float('nan')):.3f}")
    for s in steps:
        e = out["E2_additivity"].get(str(s))
        if e is None:
            continue
        print(f"E2  step={s:>6}: mean|A|={e['mean_absA']:.4f}  mean|odds-change|={e['mean_abs_denom']:.3f}  "
              f"ratio={e['ratio']:.3f}  (n={e['n']})")
    if "E2_additivity" in out and "_gate" in out["E2_additivity"]:
        gt = out["E2_additivity"]["_gate"]
        print(f"E2  GATE: ratio {gt['ratio_init']:.3f} -> {gt['ratio_final']:.3f}  "
              f"declined={gt['declined']}  final_small={gt['final_ratio_small']}")
    if "E3_sufficiency" in out:
        a3 = out["E3_sufficiency"]["agg"]
        print(f"E3  nuisance R2={a3.get('nuisance_R2'):.4f}  mean|resid|={a3.get('mean_abs_residual'):.4f}")
    if "E4_martingale" in out:
        a4 = out["E4_martingale"]["agg"]
        print(f"E4  mean|w(Dn)-E[w(Dn+1)]|={a4.get('mean_abs_discrepancy'):.4f}  "
              f"oracle_control={a4.get('oracle_control_mean_abs'):.4f}  "
              f"paired_diff={a4.get('paired_diff_mean_abs'):.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
