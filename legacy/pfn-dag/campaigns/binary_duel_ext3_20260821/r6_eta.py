"""R6 -- Amendment E.7 (D7): the eta ceiling, at d=3 with the d=2 retrofit.

E.7's estimand, verbatim:

    eta = (model matched-pair gain) / (exact-achievable matched-pair gain
           on the same pairs)

with the mirror preregistered as the MAX-CONTRAST ordering: for a context with
true ordering o, the mirrored partner is the ordering in the 6-element set that
maximizes the exact achievable gain against o, computed by enumeration. At d=2
the set has two members, so the argmax is over a singleton and the retrofit is
definitional -- which is exactly why E.7 calls it a retrofit.

WHAT E.7 DOES NOT SAY, AND WHAT THIS FILE ASSUMES
-------------------------------------------------
"exact-achievable" occurs three times in the whole repository, all three inside
E.7, and is never operationalized. This file adopts the reading that follows
from E.3, and states it here so a reader can reject it without reading code:

    Both gains are the SAME functional, differing only in whose predictive
    plays the model's part. Writing p* for the exact order-conditional
    predictive under the TRUE ordering, p' for the exact one under the mirror,
    and q*, q' for the model's predictives given D_true and D_mirror,

        gain_model = E_{y ~ p*} [ log q*(y) - log q'(y) ]
        gain_exact = E_{y ~ p*} [ log p*(y) - log p'(y) ]  =  KL(p* || p')

    Both are bin-summed exact expectations over the 100 native bins, as E.3
    and F.1 require, and neither draws an outcome.

Two consequences worth stating in advance, because they are what make the
ratio readable:

  * Substituting the exact predictive for the model (q := p) makes the two
    gains identical, so eta == 1 EXACTLY. That is a positive control this
    module asserts rather than hopes for; see `positive_control`.
  * gain_exact is `core.anchor_kl` / `exp4.anchor_kl` under a different name.
    The denominator was therefore already computed at d=3 and reported as
    `mean_anchor_kl`; it has never been computed at d=2, which is the whole
    of what the "retrofit" owes.

The reference law is the PLUG-IN order-conditional density at the pair's own
Sigma, not the posterior predictive that marginalizes Sigma. Both readings are
defensible; this one matches every anchor already on disk (`anchor_logprobs`
is a plug-in single-Sigma construction) and it is the one that makes the
denominator a statement about the substrate rather than about an inference
procedure. E.7's guard sentence -- "a low eta with a small exact-achievable
denominator is a statement about the substrate's headroom, not about the
model" -- is only enforceable if the denominator is published beside the
ratio, so every artifact this module writes carries the numerator, the
denominator, both SEs, n, and the DISTRIBUTION of the mirrors selected
(`mirror_histogram`). The per-context mirror is not persisted; if a later
analysis needs it, that is a schema change and not a re-run.

THE ONE FORK THAT IS NOT MINE TO SETTLE
---------------------------------------
E.7's numerator is "the model matched-pair gain", and "matched pair" is a defined
term from C2: a pair of CONTEXTS (D_true, D_mirror) built from one Sigma and one
set of residual draws under two orderings. The model is therefore scored on two
different contexts, and that is what this file computes. The denominator here is
the plug-in order-conditional anchor at the pair's own Sigma, which does not
depend on the realized context at all.

The two halves are consequently not the same functional of the same object, and
one property is lost that a reader may expect: eta > 1 is POSSIBLE. A model that
notices the two realized contexts differ in EMPIRICAL covariance can out-discriminate
an oracle working from the population Sigma. That is a real effect, not an error,
and the artifact counts how often each pair exceeds 1 so it can be seen rather
than inferred.

Three denominators are defensible and E.7 chooses none of them:

  D1  the plug-in anchors at the true Sigma (what this file uses). This is the
      CEILING in the strongest sense -- it is what a knower of Sigma and of both
      orderings could extract -- which is why the deferred item is named "the eta
      ceiling", and it is already the convention of the only d=3 number on disk
      (exp4's `mean_anchor_kl`). Context-free, so eta is unbounded above.
  D2  the exact Bayes posterior predictive formed on each of the two contexts,
      fully parallel to the model. Apples-to-apples, still unbounded, and it needs
      `EC.oracle_posterior` per context, which costs a quadrature per pair.
  D3  one context and two ORDERING anchors, so numerator and denominator share a
      reference law and eta = 1 - KL(p*||q)/den <= 1 becomes a hard invariant.
      This is the cleaner object, and it is NOT E.7's numerator: it abandons the
      two-context matched pair that C2 defines.

D1 is computed and reported. The eta>1 count is the diagnostic that says whether
the choice mattered on this substrate. If it is ~0, all three agree in practice.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not touch the eps dial. This is the ext3 substrate (continuous Sigma
prior, AL residuals with fixed skew r), which is where the only d=3 matched-pair
machinery and the only trained d=3 models exist. Amendment E's eps-dial world
(campaigns/corrected_20260812) has neither, so an eta there would need new
matched-pair code AND the retrain that Amendment F's lock is holding. That
fork is unsigned and this file does not pre-empt it: what it answers is the
hypothesis-space-scaling question (d=2 vs d=3), not the dial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from itertools import permutations

import numpy as np

ORDERINGS_D3 = list(permutations(range(3)))

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SCHEMA = 1
AMENDMENT = "E.7/D7"
ESTIMAND = "c2_matched_pair_efficiency_eta"

# A denominator this small means the two orderings are predictively
# indistinguishable on this context, so the ratio is 0/0. F.2d.7's survivorship
# discipline says such a unit exits the series and is surfaced, never dropped
# silently. The constant matches corrected_verdict.CALIB_MIN_GAIN.
MIN_DENOM = 1e-6


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def ratio_of_means(num: np.ndarray, den: np.ndarray) -> dict:
    """eta as a RATIO OF MEANS, with the linearized (delta-method) SE.

    F.2d.6 settled the same question for the calibrated y and settled it this
    way: the mean of per-context ratios is a different estimand and it is the
    one that blows up when a denominator approaches zero. The influence
    function is u_i = (num_i - eta*den_i)/mean(den), whose sd/sqrt(n) is the SE
    of the ratio; it reduces to the usual SE when den is constant.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    n = len(num)
    dbar = float(den.mean())
    nbar = float(num.mean())
    eta = nbar / dbar
    u = (num - eta * den) / dbar
    se = float(np.std(u, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return dict(
        eta=float(eta),
        eta_se=se,
        eta_ci_lo=float(eta - 1.96 * se),
        eta_ci_hi=float(eta + 1.96 * se),
        num_mean=nbar,
        num_se=float(np.std(num, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        den_mean=dbar,
        den_se=float(np.std(den, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        n=int(n),
        eta_per_pair_median=float(np.nanmedian(num / np.where(den > MIN_DENOM, den, np.nan))),
    )


def exact_gain(p_true_log: np.ndarray, a_log: np.ndarray, b_log: np.ndarray) -> float:
    """E_{y ~ p_true}[ log a(y) - log b(y) ], bin-summed, averaged over queries.

    Every gain in this module is this one functional. `p_true_log` is always the
    exact predictive under the true ordering; (a, b) is (model|D_true,
    model|D_mirror) for the numerator and (exact_true, exact_mirror) for the
    denominator. Rows are (n_query, N_BINS) log-normalized.
    """
    p = np.exp(p_true_log)
    return float(np.mean(np.sum(p * (a_log - b_log), axis=1)))


# ---------------------------------------------------------------------------
# d = 3: the general order-conditional predictive
# ---------------------------------------------------------------------------
def ordering_logpredictive(pi, U, b, r, qx_grid, bin_centers, log_bw, al_logpdf):
    """log p(x2 | x0, x1) under one ordering, on the bin grid, (nq, N_BINS).

    U whitens the PERMUTED coordinate vector: e = U * x[list(pi)], unit lower
    triangular so |det| = 1, and the joint log-density is sum_j log f(e_j; b_j)
    with b likewise indexed in the permuted frame. Holding (x0, x1) fixed and
    sweeping x2 across the bin grid gives the conditional up to an additive
    constant, and the row normalization removes it -- which is also why the
    terms that do not involve x2, and the bin-width factors, may be carried
    along rather than special-cased away.

    This replaces exp4's two hand-written cases (x2-as-sink, x2-in-the-middle)
    with the one form that covers all six orderings, and `assert_matches_exp4`
    checks it reproduces both of them to float tolerance before any d=3 number
    is computed.
    """
    nq = len(qx_grid)
    nb = len(bin_centers)
    d = U.shape[0]
    if d != 3:
        # The body writes columns 0, 1 and 2. At d >= 4 the rest would be read
        # uninitialized and the result would still look like a normalized
        # log-predictive, so this refuses rather than returns.
        raise NotImplementedError(
            f"ordering_logpredictive is written for d=3 (two conditioning "
            f"coordinates and one swept); got d={d}")
    cols = list(pi)
    q = np.asarray(qx_grid, float)
    bc = np.asarray(bin_centers, float)
    # Vectorized over queries. The loop this replaces cost ~26 s per pair once the
    # mirror selection moved to 500 independent queries, which is 5.8 h for one
    # 800-pair cell; the array form is the same arithmetic in one pass.
    X = np.empty((nq, nb, d))
    X[:, :, 0] = q[:, 0:1]
    X[:, :, 1] = q[:, 1:2]
    X[:, :, 2] = bc[None, :]
    E = X[:, :, cols] @ U.T                       # (nq, nb, d) residuals, permuted frame
    lp = np.zeros((nq, nb))
    for j in range(d):
        lp = lp + al_logpdf(E[:, :, j], b[j], r)
    lp = lp + d * log_bw
    m = lp.max(axis=1, keepdims=True)
    return lp - (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True)))


def _logsumexp(a):
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def predictive_classes(logp_by_ordering, tol=1e-10):
    """Group orderings whose CONDITIONAL law is identical, to machine precision.

    At d=3 this is not hypothetical. Orderings (0,1,2) and (1,0,2) both put x2 in
    the sink, so the only term of the joint that varies with x2 is the same one in
    both, and the two conditionals p(x2|x0,x1) agree to ~7e-15. They are one
    hypothesis as far as this readout can see, and an argmax that treats them as
    two lets an unstable sort decide which context gets built as the mirror --
    where the denominator is identical but the NUMERATOR is not, because the model
    is handed a materially different array.

    Returns a list of index lists, each sorted, each an equivalence class. The
    class representative is its smallest index, which makes the choice a property
    of the ordering set rather than of numpy's sort implementation.
    """
    n = len(logp_by_ordering)
    unassigned = list(range(n))
    classes = []
    while unassigned:
        head = unassigned.pop(0)
        cls = [head]
        rest = []
        for j in unassigned:
            if np.max(np.abs(logp_by_ordering[head] - logp_by_ordering[j])) <= tol:
                cls.append(j)
            else:
                rest.append(j)
        unassigned = rest
        classes.append(sorted(cls))
    return classes


def max_contrast_mirror(logp_by_ordering, true_idx, logp_select=None, tol=1e-10):
    """E.7's mirror: argmax over the ordering set of the exact achievable gain.

    Three things this does that a bare argmax does not:

    1. It selects on `logp_select` when given -- predictives built from an
       INDEPENDENT, larger query draw. Selecting the maximum on the same queries
       that then report it is a winner's curse: the per-query gain has a
       coefficient of variation near 1, so a 20-query mean carries ~22% noise, and
       the max of five such means is biased upward by 4-5% (t = 3 to 4 against a
       high-M reference). The selection step touches no model, so making it
       accurate is nearly free.
    2. It excludes the true ordering's WHOLE equivalence class, not just the true
       index. An ordering predictively identical to the true one has zero
       achievable gain by construction and is not a candidate for a maximum; at
       d=3 exactly such an ordering exists in half the contexts.
    3. It breaks ties by smallest index within the winning class, so the choice is
       deterministic. E.7 registers no tie rule, so this file registers one and
       reports it rather than leaving it to `np.argsort`'s quicksort.

    Returns (mirror_idx, gain, runner_up_gain, n_candidates, tied_at_top).
    """
    sel = logp_select if logp_select is not None else logp_by_ordering
    classes = predictive_classes(sel, tol=tol)
    true_class = next(c for c in classes if true_idx in c)
    gains, reps = [], []
    for c in classes:
        if c is true_class:
            continue
        reps.append(c[0])
        gains.append(exact_gain(sel[true_idx], sel[true_idx], sel[c[0]]))
    if not reps:
        return None, 0.0, float("nan"), 0, 0, 0.0
    gains = np.asarray(gains)
    top = float(gains.max())
    tied = [reps[i] for i in range(len(reps)) if top - gains[i] <= tol]
    best = min(tied)
    others = np.sort(gains)[::-1]
    runner = float(others[1]) if len(others) > 1 else float("nan")
    # The gain is REPORTED on the measurement predictives, never on the selection
    # ones, so the denominator is an unbiased estimate at the mirror so chosen.
    gain = exact_gain(logp_by_ordering[true_idx], logp_by_ordering[true_idx],
                      logp_by_ordering[best])
    # `top` and `runner` both live on the SELECTION grid; the caller needs both to
    # standardize the margin on one draw. Mixing the measurement-grid winner with a
    # selection-grid runner-up compares three different objects and reports the
    # winner's-curse repair as if it had failed.
    return best, float(gain), runner, len(reps), len(tied), top


def _forward_substitution_logpredictive(pi, Lunit, b, r, qx_grid, bin_centers,
                                        log_bw, al_logpdf):
    """The same conditional by an independent route: solve L e = x[pi] for e.

    `ordering_logpredictive` uses U (the whitening matrix) directly. This one
    inverts Lunit instead, which is a different code path reaching the same
    density, so agreement between them pins the coordinate frame. It exists
    because the exp4 comparison structurally cannot: PI_A, PI_B, (1,0,2) and
    (2,1,0) are all self-inverse permutations, so confusing `pi` with its inverse
    is INVISIBLE on exactly the two orderings exp4 hard-codes, while corrupting
    (1,2,0) and (2,0,1) by tens of nats -- and those two carry most of the
    max-contrast argmax's mass.
    """
    nb = len(bin_centers)
    out = np.zeros((len(qx_grid), nb))
    for qi, (x0, x1) in enumerate(qx_grid):
        X = np.empty((nb, 3))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2] = bin_centers
        E = np.linalg.solve(Lunit, X[:, list(pi)].T).T
        lp = np.zeros(nb)
        for j in range(3):
            lp = lp + al_logpdf(E[:, j], b[j], r)
        lp = lp + 3 * log_bw
        out[qi] = lp - _logsumexp(lp)
    return out


def assert_predictive_is_sound(orc, exp4, prior, qx_grid, n_sigma=20, seed=20260824,
                               rtol=1e-11, atol=1e-11):
    """Two references, all six orderings, many Sigmas -- run before any measurement.

    (1) Against exp4's two hand-written anchors, which pins the AL parameterization
        and the bin conventions to the instrument already on disk.
    (2) Against an independent forward-substitution build, for all six orderings,
        which pins the coordinate frame that (1) is blind to.

    A failure here means every d=3 number below it is wrong in a way nothing
    downstream would catch, which is why it runs first and raises rather than warns.

    On tolerance: an earlier draft argued from conditioning that 1e-7 was needed.
    The measurement says otherwise. Across a full 800-pair cell the worst observed
    disagreement between the two routes is 5.7e-14, so the tolerance is set at 1e-11,
    which keeps three orders of headroom over what is actually observed and is still
    twelve orders inside the failure it exists to catch: permuting the frame instead
    of inverse-permuting it corrupts orderings (1,2,0) and (2,0,1) by 15 to 64 nats.
    The worst observed deviation is returned rather than only compared, so the
    tolerance can keep being set from data instead of from argument.
    """
    r = exp4.R_OF.get(prior, 2.0)
    rng = np.random.default_rng(seed)
    # ONE batched draw. orc.sample_Sigmas rejection-samples 8192 candidates at a
    # time and validity-checks each against all six orderings (two cholesky-class
    # factorizations per ordering), so asking for one Sigma costs the same as
    # asking for a thousand. Calling it in a loop paid that price n_sigma times.
    sigmas = orc.sample_Sigmas(rng, n_sigma)
    checked = 0
    worst = 0.0
    for S in sigmas:
        ref_A, ref_B = exp4.anchor_logprobs(S, prior, qx_grid, orc)
        for pi in ORDERINGS_D3:
            Lunit, U, b = orc.params_for(S[None], pi)
            got = ordering_logpredictive(pi, U[0], b[0], r, qx_grid,
                                         exp4.BIN_CENTERS, exp4.LOG_BW, orc.al_logpdf)
            ref = _forward_substitution_logpredictive(
                pi, Lunit[0], b[0], r, qx_grid, exp4.BIN_CENTERS,
                exp4.LOG_BW, orc.al_logpdf)
            worst = max(worst, float(np.max(np.abs(got - ref))))
            if not np.allclose(got, ref, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"ordering {pi}: U-frame and forward-substitution disagree, "
                    f"max |delta| = {float(np.max(np.abs(got - ref))):.3e}")
            if pi == exp4.PI_A or pi == exp4.PI_B:
                exp_ref = ref_A if pi == exp4.PI_A else ref_B
                worst = max(worst, float(np.max(np.abs(got - exp_ref))))
                if not np.allclose(got, exp_ref, rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"ordering {pi}: disagrees with exp4's hand-written anchor, "
                        f"max |delta| = {float(np.max(np.abs(got - exp_ref))):.3e}")
            checked += 1
    return checked, worst


def edge_bin_mass(logp):
    """Fraction of each row's mass sitting in the two edge bins. Measured, not fatal.

    The first version of this raised above 1e-2 and it fired on the second pair of
    the first cell, at 2.4e-2, on the MIRROR conditional. That was the right thing
    to measure and the wrong thing to do about it.

    The model's predictive is a softmax over exactly these 100 bins, so it cannot
    place mass outside them whatever the truth is. Renormalizing the exact anchor
    onto the same support is therefore not a distortion of the ceiling; it is what
    makes the ceiling one the model could in principle reach. An anchor scored on a
    wider grid than the model can express would be a bound the model is structurally
    forbidden from attaining, which is not what "exactly achievable" should mean.

    The mirror is the law most likely to escape the grid, since it is selected to
    maximize divergence from p*, so its edge mass is recorded separately. The effect
    on the denominator is small and one-directional: truncation shrinks the KL, so
    it inflates eta, measured at 0.2 to 0.3% typically and 3% at worst.

    `hard_stop_edge_mass` is where renormalization would genuinely stop being
    benign; it is set far above anything this substrate produces so that a later
    substrate with a looser validity box cannot pass silently.
    """
    return np.exp(logp)[:, [0, -1]].sum(axis=1)


HARD_STOP_EDGE_MASS = 0.25


def frac_below(v, thresh):
    """Fraction below a threshold, EXCLUDING undefined entries rather than counting
    them as passing. `nan < x` is silently False, so the naive form reports "no pair
    had an unstable margin" when the truth is "no pair had a measurable one".
    Returns (fraction, n_defined)."""
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any():
        return float("nan"), 0
    return float((v[m] < thresh).mean()), int(m.sum())


def check_edge_mass(logp, label):
    e = edge_bin_mass(logp)
    worst = float(e.max())
    if worst > HARD_STOP_EDGE_MASS:
        raise ValueError(
            f"{label}: {worst:.3e} of the conditional's mass is in the two edge "
            f"bins, above {HARD_STOP_EDGE_MASS}. Renormalizing onto the model's "
            "support is defensible at this substrate's scale; at this magnitude it "
            "is not, and the grid needs revisiting before the number is believed")
    return worst, float(e.mean())


def cross_path_deviation(driver_value, independent_value):
    """|a/b - 1| between two quantities computed by DIFFERENT implementations.

    Twice now a control here has been written so that both sides ran the same
    expression on the same inputs, and returned 0.0 for every input including
    deliberately corrupted ones. Bit-identical arguments through one function is
    not a cross-check however the algebra reads, so the callers are now required
    to build one side through `_forward_substitution_logpredictive`, which reaches
    the same density by inverting Lunit instead of applying U -- a separate code
    path that a frame or indexing error moves off 1.
    """
    if not np.isfinite(independent_value) or abs(independent_value) <= MIN_DENOM:
        return float("nan")
    return float(abs(driver_value / independent_value - 1.0))


def independent_gain(orc, exp4, S, pi_true, pi_other, r, qx_grid):
    """KL(p_true || p_other) built entirely through the forward-substitution path.

    Nothing here calls `ordering_logpredictive`, so this is the independent side of
    `cross_path_deviation`.
    """
    outs = []
    for pi in (pi_true, pi_other):
        Lunit, _U, b = orc.params_for(S[None], pi)
        outs.append(_forward_substitution_logpredictive(
            pi, Lunit[0], b[0], r, qx_grid, exp4.BIN_CENTERS,
            exp4.LOG_BW, orc.al_logpdf))
    return exact_gain(outs[0], outs[0], outs[1])


def mirror_arm_control(logp_true, lq_true, lq_mirror):
    """Split eta>1 into its two possible causes, which mean opposite things.

    num = KL(p*||q') - KL(p*||q*), so eta exceeds 1 either because the model
    tracks the true ordering well (small KL(p*||q*)) or merely because it predicts
    badly under the mirror context (large KL(p*||q')). Only the first is a ceiling
    exceedance. Publishing both KLs is what lets a reader tell them apart instead
    of inferring.
    """
    return (exact_gain(logp_true, logp_true, lq_true),
            exact_gain(logp_true, logp_true, lq_mirror))


def control_block(num, den, neut, subs_dev, kl_true, kl_mirror) -> dict:
    """Everything a reader needs to decide whether the ratio means anything.

    Reported beside eta rather than gated on: a control that silently passes
    teaches nothing, so the numbers are published and a disagreement with the
    headline is visible without re-running anything.

    Two changes forced by the pre-run audit. The eta>1 fraction is split by cause
    (`mean_kl_true` vs `mean_kl_mirror`), because eta>1 means opposite things
    depending on which arm produced it. And the fractions mask non-finite entries
    before comparing, since `nan > 1.0` is silently False and would count a dropped
    pair as a clean one.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    neut = np.asarray(neut, float)
    n = len(num)
    if n == 0:
        raise ValueError("no pairs survived; refusing to aggregate an empty series")
    per_pair = num / np.where(den > MIN_DENOM, den, np.nan)
    m = np.isfinite(per_pair)
    nb = ratio_of_means(neut, den)
    return dict(
        eta_neutral=nb["eta"],
        eta_neutral_se=nb["eta_se"],
        control6_gain_mean=float(neut.mean()),
        substitution_control_max_dev=float(np.max(subs_dev)) if len(subs_dev) else float("nan"),
        mean_kl_p_true_vs_model_true=float(np.mean(kl_true)),
        mean_kl_p_true_vs_model_mirror=float(np.mean(kl_mirror)),
        frac_pairs_eta_gt_1=float(np.mean(per_pair[m] > 1.0)) if m.any() else float("nan"),
        frac_pairs_eta_negative=float(np.mean(per_pair[m] < 0.0)) if m.any() else float("nan"),
        n_pairs_finite_ratio=int(m.sum()),
        den_min_observed=float(den.min()),
        den_q01_observed=float(np.quantile(den, 0.01)),
    )


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def stamp(doc: dict, status: str = "live") -> dict:
    doc["artifact"] = dict(estimand=ESTIMAND, status=status,
                           amendment=AMENDMENT, schema=SCHEMA)
    return doc


def env_block(extra=None, source_files=()) -> dict:
    """Enough to tie a number to the bytes and the hardware that produced it.

    Twenty-one artifacts were written today by a build whose published control was
    vacuous, and nothing inside them said which build. The code hashes are here so
    that cannot recur silently: a reader can tell at a glance whether an artifact
    predates a fix without diffing anything.
    """
    d = dict(host=platform.node(), python=sys.version.split()[0],
             numpy=np.__version__,
             slurm_job_id=os.environ.get("SLURM_JOB_ID", "none"),
             partition=os.environ.get("SLURM_JOB_PARTITION", "none"))
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        d["torch"] = "absent"
    for f in source_files:
        try:
            d[f"sha256_{os.path.basename(f)}"] = hashlib.sha256(
                open(f, "rb").read()).hexdigest()[:16]
        except OSError:
            d[f"sha256_{os.path.basename(f)}"] = "unreadable"
    if extra:
        d.update(extra)
    return d


def file_sha256(path) -> str:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return "unreadable"


def sha256_of(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, np.float64)).tobytes()).hexdigest()
