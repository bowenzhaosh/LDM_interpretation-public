"""Corrected exact finite-prior oracle (Track A1 + A2).

Everything here is exact up to float64 for a finite prior over latent states
z = (k, o) (covariance atom k, causal order o). There is NO Monte Carlo error:
the posterior p(k,o | D) is computed by enumeration with logsumexp, and every
predictive is computed through the exact joint numerator/denominator.

Crucially, this corrects the two Branch B oracle defects:
- Covariance-atom weights ARE conditioned on the context likelihood
  p(D | k, o) (no uniform-atom two-stage calculation).
- The ordering value is the expected log-score difference / KL, not an
  unweighted arithmetic average over log bin ratios.

Latent-state layout: latent index lo = k * O + o, where O = len(orderings).
Arrays over latent states are (K*O,); arrays over orders are (O,).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .corrected_sem import (
    SEMParams,
    ResidualSpec,
    all_orderings,
    conditional_joint_logdensity,
    interventional_density_grid,
    params_for,
    residual_logdensity_row,
)
from .pilot_shared import N_BINS, production_quadrature


@dataclass(frozen=True)
class World:
    """A finite exact world: K covariance atoms + a residual law + orderings.

    sigmas: (K, d, d) float64 PD covariance atoms (the finite prior support).
    spec:   ResidualSpec (epsilon / AL-shape).
    orderings: tuple of d-tuples, the causal orders (default: all).
    prior_atom: (K,) prior probabilities over atoms (default uniform).
    prior_order: (O,) prior probabilities over orders (default uniform).
    """
    sigmas: np.ndarray
    spec: ResidualSpec
    orderings: tuple[tuple[int, ...], ...] | None = None
    prior_atom: np.ndarray | None = None
    prior_order: np.ndarray | None = None

    def __post_init__(self) -> None:
        self_sigmas = np.asarray(self.sigmas, dtype=np.float64)
        object.__setattr__(self, "sigmas", self_sigmas)
        if self_sigmas.ndim != 3 or self_sigmas.shape[1] != self_sigmas.shape[2]:
            raise ValueError("sigmas must be (K, d, d)")
        object.__setattr__(self, "d", self_sigmas.shape[1])
        object.__setattr__(self, "K", self_sigmas.shape[0])
        if self.orderings is None:
            object.__setattr__(self, "orderings", all_orderings(self.d))
        object.__setattr__(self, "O", len(self.orderings))
        object.__setattr__(self, "n_latent", self.K * self.O)
        if self.prior_atom is None:
            object.__setattr__(self, "prior_atom",
                               np.full(self.K, 1.0 / self.K, dtype=np.float64))
        if self.prior_order is None:
            object.__setattr__(self, "prior_order",
                               np.full(self.O, 1.0 / self.O, dtype=np.float64))
        pa = np.asarray(self.prior_atom, dtype=np.float64)
        po = np.asarray(self.prior_order, dtype=np.float64)
        if pa.shape != (self.K,) or po.shape != (self.O,):
            raise ValueError("prior_atom/prior_order shape mismatch")
        if not np.all(pa > 0) or not np.all(po > 0):
            raise ValueError("priors must be strictly positive")
        pa = pa / pa.sum()
        po = po / po.sum()
        object.__setattr__(self, "prior_atom", pa)
        object.__setattr__(self, "prior_order", po)

    def prior_lo(self) -> np.ndarray:
        """(K*O,) latent prior p(k, o) = p_atom[k] p_order[o]."""
        return np.outer(self.prior_atom, self.prior_order).ravel()

    # -- helpers ------------------------------------------------------------
    def sem_params(self, k: int, o: int) -> SEMParams:
        return params_for(self.sigmas[k], self.orderings[o], r=self.spec.r)

    def lo_to_ko(self, lo: int) -> tuple[int, int]:
        return divmod(lo, self.O)

    def params_cache(self) -> dict[str, np.ndarray]:
        """Precomputed per-(k,o) SEM arrays for all latent states.

        Cached lazily: L_unit (K,O,d,d), U (K,O,d,d), b (K,O,d), and the
        per-order permutation index arrays. The Track B generators use this so
        a training batch avoids K*O per-item Cholesky factorizations.
        """
        cache = getattr(self, "_params_cache", None)
        if cache is not None:
            return cache
        K, O, d = self.K, self.O, self.d
        L_unit = np.zeros((K, O, d, d), dtype=np.float64)
        U = np.zeros((K, O, d, d), dtype=np.float64)
        b = np.zeros((K, O, d), dtype=np.float64)
        for k in range(K):
            for o in range(O):
                sp = params_for(self.sigmas[k], self.orderings[o], r=self.spec.r)
                L_unit[k, o] = sp.L_unit
                U[k, o] = sp.U
                b[k, o] = sp.b
        perms = np.array(self.orderings, dtype=np.int64)  # (O, d)
        cache = {"L_unit": L_unit, "U": U, "b": b, "perms": perms}
        object.__setattr__(self, "_params_cache", cache)
        return cache


# ---------------------------------------------------------------------------
# Context likelihood and exact joint posterior.
# ---------------------------------------------------------------------------

def context_loglikelihoods(
    world: World,
    context: np.ndarray,
) -> np.ndarray:
    """log p(D | k, o) for all latent states, (K*O,), exact float64.

    context is (n, d). Rows are IID: p(D | k, o) = prod_rows p(x_row | k, o).
    """
    K, O, d = world.K, world.O, world.d
    context = np.asarray(context, dtype=np.float64)
    out = np.empty(K * O, dtype=np.float64)
    for k in range(K):
        for o in range(O):
            sp = world.sem_params(k, o)
            lpd = residual_logdensity_row(context, sp, world.spec)  # (n,)
            out[k * O + o] = lpd.sum()
    return out


def exact_joint_posterior(
    world: World,
    context: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact joint latent posterior p(k, o | D).

    Returns dict with:
      logZ_lo   (K*O,)  log unnormalized weights log p(k,o|D) (incl. uniform prior)
      w_lo      (K*O,)  normalized p(k, o | D)
      w_o       (O,)    p(o | D)
      w_k_given_o (K, O) p(k | o, D)
      logZ      float   log p(D) under the uniform latent prior
    """
    K, O = world.K, world.O
    ll = context_loglikelihoods(world, context)           # (K*O,)
    log_prior = np.log(world.prior_lo())
    logZ_lo = ll + log_prior
    mx = logZ_lo.max()
    w_lo = np.exp(logZ_lo - mx)
    total = w_lo.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("posterior weights are degenerate")
    w_lo = w_lo / total
    logZ = mx + math.log(total)
    w_o = w_lo.reshape(K, O).sum(axis=0)
    w_k_given_o = w_lo.reshape(K, O) / np.maximum(w_o[None, :], 1e-300)
    return {
        "logZ_lo": logZ_lo,
        "w_lo": w_lo,
        "w_o": w_o,
        "w_k_given_o": w_k_given_o,
        "logZ": float(logZ),
    }


# ---------------------------------------------------------------------------
# Exact average-case identifiability (Amendment F.2b).
# ---------------------------------------------------------------------------

def order_entropy_deficit(
    world: World,
    context: np.ndarray | None = None,
    post: dict[str, np.ndarray] | None = None,
) -> float:
    """Normalized order-marginal entropy deficit at ONE context.

        1 - H(p(o | D)) / log O

    0 when the data leave the ordering exactly at its prior and 1 when a single
    ordering carries all the posterior mass. ``World.__post_init__`` builds
    ``prior_order`` uniform on O = d! orderings, so log O IS the prior's order
    entropy and the deficit is exactly the fraction of it the data removed.

    Amendment F.2b makes the panel average of this quantity the PRIMARY x-axis
    of the association gate, demoting the LP identified-set width to the
    appendix. Two properties motivate the swap and both are structural rather
    than empirical:

      * It is a functional of the exact posterior alone. Nothing here calls a
        linear program, so the axis cannot inherit an LP solver path (the
        eps=0.5 simplex cell that could not clear a 24h wall while the interior
        point method cleared it in 16 minutes is invisible to this function).
      * It is evaluated on whatever contexts it is handed, so when it is handed
        the PFN's own eval panel the x and y of the gate sit on the same object
        at the same information level. The width ladder is computed on a
        separately configured panel whose n_rows the ident artifacts never
        recorded, which is Amendment F.5's disclosure item.

    Passing ``post`` reuses an already-computed posterior; otherwise one is
    computed from ``context``.
    """
    if world.O < 2:
        raise ValueError("order entropy deficit is undefined for O < 2")
    if post is None:
        if context is None:
            raise ValueError("pass either context or post")
        post = exact_joint_posterior(world, context)
    p = np.asarray(post["w_o"], dtype=np.float64)
    if p.shape != (world.O,):
        # The post= path exists so a caller can reuse an already-computed
        # posterior, which is exactly the path where one from the WRONG world
        # gets threaded through. Without this, a d=2 posterior normalised by
        # log 6 returns a plausible number.
        raise ValueError(f"w_o has shape {p.shape}, world.O = {world.O}")
    # log O is the prior's order entropy only if the order prior is uniform,
    # which make_world always builds but World does not require. A tilted prior
    # would put a spurious identifiability floor at eps=0, which is where gate
    # (b)'s null lives.
    if not np.allclose(world.prior_order, 1.0 / world.O, rtol=0, atol=1e-12):
        raise ValueError(
            "order prior is not uniform; log O is not its entropy and the "
            "deficit would carry a spurious floor. Normalise by H(prior_order) "
            "and amend F.2b before using this world.")
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("order marginal is degenerate")
    p = p / total
    nz = p[p > 0.0]                       # 0 log 0 == 0, by omission
    H = float(-(nz * np.log(nz)).sum())
    d = 1.0 - H / math.log(world.O)
    # Float excursion only: at eps=0 the posterior is uniform to ~5e-15 and H
    # can exceed log O by an ulp, printing a negative identifiability at the
    # very cell gate (b) leans on.
    if not (-1e-9 <= d <= 1.0 + 1e-9):
        raise FloatingPointError(f"deficit {d!r} outside [0, 1] beyond tolerance")
    return float(min(1.0, max(0.0, d)))


def panel_order_entropy_deficit(
    world: World,
    contexts,
) -> dict[str, float]:
    """Panel average of :func:`order_entropy_deficit`, with its across-context SE.

    The average is taken OVER contexts of the per-context deficit, E[1 - H/log O],
    and deliberately not the deficit of a pooled posterior. Amendment F.2b's
    estimand is average-case identifiability: how identified a typical dataset
    drawn from the prior leaves the ordering. Pooling first would answer a
    different question (how identified the ordering is given all panels at once)
    and would not be the quantity the PFN is scored against.

    ``se`` is the across-context standard error. Panel contexts are iid draws
    from the world prior (``make_eval_panel`` samples k ~ U(K), o ~ U(O) per
    context, matching ``World.prior_lo``), which is what licenses that SE as the
    SE of the prior-averaged estimand rather than of a fixed finite population.
    """
    ds = np.array([order_entropy_deficit(world, context=c) for c in contexts],
                  dtype=np.float64)
    n = int(ds.size)
    if n < 2:
        # Returning se = 0.0 here would make adjacent_resolution declare every
        # rung resolved, failing open in the one clause whose job is to report
        # what the instrument cannot separate.
        raise ValueError(f"need at least 2 contexts for a panel SE, got {n}")
    se = float(np.std(ds, ddof=1) / np.sqrt(n))
    return {"deficit_mean": float(ds.mean()), "deficit_se": se, "n_contexts": n,
            "deficit_min": float(ds.min()), "deficit_max": float(ds.max())}


# ---------------------------------------------------------------------------
# Weight ablations (A2).
# ---------------------------------------------------------------------------

def ablated_weights(
    world: World,
    post: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Latent-posterior weights (K*O,) for an explicit ablation.

    mode in {"full", "order_ablated", "atom_ablated", "prior"}.

    - full:          w(k,o) = p(k,o | D)
    - order_ablated: w(k,o) = (1/|O|) p(k | o, D)     (order marginal flattened)
    - atom_ablated:  w(k,o) = (1/K) p(o | D)          (atom marginal flattened)
    - prior:         w(k,o) = 1/(K*|O|)
    """
    K, O = world.K, world.O
    w_lo = post["w_lo"]
    if mode == "full":
        return w_lo.copy()
    if mode == "order_ablated":
        w_k_given_o = post["w_k_given_o"]               # (K, O)
        # Preserve atom posterior conditional on each order; flatten the order
        # marginal to the ORDER PRIOR (not uniform when a non-uniform prior is
        # set): w(k,o) = p_order(o) p(k | o, D).
        out = w_k_given_o * world.prior_order[None, :]
        return out.ravel() / out.sum()
    if mode == "atom_ablated":
        w_o = post["w_o"]                                # (O,)
        # Preserve order posterior; flatten atoms to the ATOM PRIOR:
        # w(k,o) = p_atom(k) p(o | D).
        out = world.prior_atom[:, None] * w_o[None, :]
        return np.asarray(out, dtype=np.float64).ravel()
    if mode == "prior":
        return world.prior_lo().copy()
    raise ValueError(f"unknown ablation mode: {mode}")


# ---------------------------------------------------------------------------
# Query operators: joint numerator/denominator (observational) and
# intervention operator (interventional).
# ---------------------------------------------------------------------------

@dataclass
class ObsQueryOperator:
    """Joint numerator/denominator operators for one observational query.

    For the query (x_q, target), define per latent state ko:
      num[ko, b] = p(x_q, y in bin b | ko)     (numerator rows, (K*O, B))
      den[ko]    = p(x_q | ko)                  (denominator, (K*O,))
    Then for any latent posterior weights w (normalized),
      p(y in b | x_q, D) = (w @ num)[b] / (w @ den).
    """
    target: int
    x_q: np.ndarray
    num: np.ndarray        # (K*O, B)
    den: np.ndarray        # (K*O,)

    def predictive(self, w_lo: np.ndarray) -> np.ndarray:
        num = w_lo @ self.num          # (B,)
        den = float(w_lo @ self.den)
        if den <= 0 or not np.isfinite(den):
            raise FloatingPointError("observational denominator is nonpositive")
        return num / den


@dataclass
class IntQueryOperator:
    """Interventional operator for one do() query.

    op[ko, b] = p(y in bin b | do(x_intervened = value), ko). For any
    normalized latent posterior w,
      p(y in b | do(...), D) = (w @ op)[b].
    """
    intervened: int
    value: float
    target: int
    op: np.ndarray          # (K*O, B)

    def predictive(self, w_lo: np.ndarray) -> np.ndarray:
        return w_lo @ self.op


# -- quadrature reuse --------------------------------------------------------
_Q = None


def _quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _Q
    if _Q is None:
        _Q = production_quadrature()
    return _Q


def obs_query_operator(
    world: World,
    x_q: np.ndarray,
    target: int,
) -> ObsQueryOperator:
    """Build the joint num/den operator for an observational query.

    x_q is (d-1,) the observed covariates (all variables except `target`).
    Exact per-latent-state bin integrals via the frozen production quadrature.
    """
    K, O, B = world.K, world.O, N_BINS
    values, bins, logw = _quadrature()
    d = world.d
    x_q = np.asarray(x_q, dtype=np.float64).reshape(-1)
    if x_q.shape[0] != d - 1:
        raise ValueError(f"x_q must be (d-1,)=({d - 1},)")
    # Per-latent-state unnormalized joint density log p(x_q, y_node | ko).
    # (K*O, n_nodes)
    joint = np.empty((K * O, values.shape[0]), dtype=np.float64)
    for lo in range(K * O):
        k, o = divmod(lo, O)
        sp = world.sem_params(k, o)
        joint[lo] = conditional_joint_logdensity(values, x_q, target, sp, world.spec)
    joint_w = joint + logw[None, :]              # weighted log mass
    mx = joint_w.max(axis=1, keepdims=True)
    mass = np.exp(joint_w - mx)                  # (K*O, n_nodes)
    num = np.zeros((K * O, B), dtype=np.float64)
    np.add.at(num, (np.arange(K * O)[:, None], bins[None, :]), mass)
    den = mass.sum(axis=1)                       # (K*O,)
    num *= np.exp(mx)
    den *= np.exp(mx.ravel())
    return ObsQueryOperator(target=target, x_q=x_q, num=num, den=den)


def int_query_operator(
    world: World,
    intervened: int,
    value: float,
    target: int,
    n_quad: int = 200,
    coarse: bool = False,
) -> IntQueryOperator:
    """Build the interventional operator for do(x_intervened = value), target y.

    Exact per-latent-state bin probabilities via the interventional density
    integrated on the production quadrature. When `coarse` is True (used by the
    identifiability panel, tolerance ~1e-3), m>=2 densities are evaluated on a
    coarse y-grid and interpolated, which is much faster.
    """
    K, O, B = world.K, world.O, N_BINS
    values, bins, logw = _quadrature()
    op = np.zeros((K * O, B), dtype=np.float64)
    edges = np.linspace(-8.0, 8.0, B + 1)
    from .corrected_sem import interventional_density_coarse, interventional_linear_forms
    for lo in range(K * O):
        k, o = divmod(lo, O)
        sp = world.sem_params(k, o)
        if target == intervened:
            # Point mass at `value`: all mass in the containing bin.
            b_idx = int(np.clip(np.searchsorted(edges[1:-1], value), 0, B - 1))
            op[lo, b_idx] = 1.0
            continue
        if coarse:
            m, _, _ = interventional_linear_forms(sp, intervened, target, value)
            if m >= 2:
                # Coarse-grid interpolation for the bulk; the quadrature TAIL
                # nodes reach |y| ~ 1e4, and np.interp would clamp to the edge
                # density there (injecting spurious tail mass for the heavy-tailed
                # AL worlds). Evaluate the far-tail nodes exactly instead.
                yg, dg = interventional_density_coarse(target, intervened, value,
                                                       sp, world.spec, n_quad=n_quad)
                interior = np.abs(values) <= yg[-1]
                dens = np.interp(values, yg, dg)
                if not np.all(interior):
                    dens_far = interventional_density_grid(
                        values[~interior], target, intervened, value, sp, world.spec)
                    dens[~interior] = dens_far
            else:
                dens = interventional_density_grid(values, target, intervened, value,
                                                   sp, world.spec)
        else:
            dens = interventional_density_grid(values, target, intervened, value, sp, world.spec)
        # bin probability = int_bin dens(y) dy ~ sum over quadrature nodes in
        # the bin of dens * node weight. The production quadrature weights
        # encode dy (interior and tail nodes).
        mass = dens * np.exp(logw)
        op[lo] = np.bincount(bins, weights=mass, minlength=B)
        s = op[lo].sum()
        if s > 0:
            op[lo] /= s
    return IntQueryOperator(intervened=intervened, value=value, target=target, op=op)


# ---------------------------------------------------------------------------
# Predictive value / ordering value (A2, proper-scoring formulation).
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over a probability vector, nonnegative up to float64."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.maximum(p, 1e-300)
    q = np.maximum(q, 1e-300)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def expected_log_score_diff(p_w1: np.ndarray, p_w2: np.ndarray) -> float:
    """Expected log-score difference / KL between two exact predictives.

    value(w1 vs w2) = E_{y ~ p_w1}[-log p_w2(y) + log p_w1(y)] = KL(p_w1||p_w2).
    Nonnegative up to numerical precision.
    """
    return kl_divergence(p_w1, p_w2)


def exact_order_value(
    world: World,
    post: dict[str, np.ndarray],
    operator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact full and order-ablated predictives + KL ordering value.

    operator is an ObsQueryOperator or IntQueryOperator.
    Returns (p_full, p_ablated, V) with V = KL(p_full || p_order_ablated).
    """
    w_full = ablated_weights(world, post, "full")
    w_order_abl = ablated_weights(world, post, "order_ablated")
    p_full = operator.predictive(w_full)
    p_abl = operator.predictive(w_order_abl)
    return p_full, p_abl, kl_divergence(p_full, p_abl)
