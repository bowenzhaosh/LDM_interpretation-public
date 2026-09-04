"""R6 driver -- eta at d=3 (E.7's max-contrast mirror) and d=2 (the retrofit).

One (d, prior, scale, seed, dose) cell per invocation. CPU-only.

The estimand, the reference law, the mirror rule and the controls live in
r6_eta.py; this file is substrate plumbing and the artifact writer. Read
r6_eta's header first -- it states the one assumption E.7 leaves open.

Both arms write the SAME schema, because the point of the retrofit is that the
two numbers are comparable. Where they are NOT comparable, the artifact says so
in its own fields rather than leaving it to a reader: at d=3 the mirror is a
maximum over five candidate laws while at d=2 it is the only alternative, and a
maximum over five is systematically the larger number. `eta_duel` is therefore
computed alongside, on the adjacent transposition that IS the structural analog
of the d=2 mirror, so a d2-vs-d3 comparison can be made on a matched convention.
Without it, roughly an eighth of any measured drop would be the convention.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_infra(start):
    """Walk up looking for infra/config.py rather than assuming one fixed hop.

    From the cluster working dir (.../experiments/binary-duel) the sibling hop is
    right. From the vendored copy (.../campaigns/binary_duel_ext3_20260821) it
    resolves to .../campaigns/infra, which does not exist, so the d=2 arm could
    only ever run on the cluster and its first execution would have been the cited
    one. Searching costs nothing and makes the file runnable from both.
    """
    d = start
    for _ in range(5):
        cand = os.path.join(os.path.dirname(d), "infra")
        if os.path.exists(os.path.join(cand, "config.py")):
            return cand
        d = os.path.dirname(d)
    return os.path.join(os.path.dirname(start), "infra")   # cluster default


for _p in (_HERE, _find_infra(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r6_eta as R

M_SELECT_DEFAULT = 500     # mirror selection; exact-substrate only, no model call
M_MEASURE_DEFAULT = 20     # the queries eta is reported on


def _agg(num, den, neut, subs, klt, klm, cls, extra):
    doc = R.ratio_of_means(np.asarray(num), np.asarray(den))
    doc.update(R.control_block(num, den, neut, subs, klt, klm))
    # F.2d.6's ratio-of-means weights each pair by its denominator, so a pooled
    # eta over two structurally different halves is a denominator-weighted blend.
    # The halves are published so the blend is visible.
    cls = np.asarray(cls)
    for c in (1, 2):
        k = cls == c
        if k.sum() >= 2:
            h = R.ratio_of_means(np.asarray(num)[k], np.asarray(den)[k])
            doc[f"eta_cls{c}"] = h["eta"]
            doc[f"eta_cls{c}_se"] = h["eta_se"]
            doc[f"den_cls{c}_mean"] = h["den_mean"]
            doc[f"n_cls{c}"] = h["n"]
    doc.update(extra)
    return doc


def _d2_independent_gain(prior, pT, pM, cls, xA, bin_centers, bin_width):
    """KL(p_true || p_mirror) at d=2 through a SEPARATE density implementation.

    The version this replaces rebuilt the two conditionals with the same
    `core._bin_logp` calls that `core.anchor_logprobs` already makes, so it agreed
    to the last bit by construction and could only ever catch a labelling slip
    confined to one line. The AL log-density is written out here from its own
    parameters (a = r c, c = sqrt(2 b^2 / (1 + r^2)), centered by (a - c)) so the
    two sides share no arithmetic, and the check has a real float-level residual
    instead of an exact zero that means nothing.
    """
    if prior == "N":
        raise ValueError("prior N has no AL skew; the d=2 arm refuses it upstream")
    r = 1.0 if prior == "L" else int(prior[2:]) / 10.0

    def al_bin_logp(y, loc, scale):
        c = math.sqrt(2.0 / (1.0 + r * r))
        a = r * c
        z = (np.asarray(y, float) - np.asarray(loc, float)) / (scale * math.sqrt(2) + 1e-9)
        zc = z * math.sqrt(2.0) + (a - c)
        lp = np.where(zc >= 0.0, -zc / a, zc / c) - math.log(a + c)
        return lp - math.log(scale) + math.log(bin_width)

    a_t, b1_t, b2_t = pT
    a_m, b1_m, b2_m = pM
    bc = np.asarray(bin_centers, float)
    nb = len(bc)
    lt = np.zeros((len(xA), nb))
    lm = np.zeros((len(xA), nb))
    for qi, x in enumerate(xA):
        if cls == 1:
            rt = al_bin_logp(bc, a_t * x, b2_t)
            rm = al_bin_logp(bc, 0.0, b1_m) + al_bin_logp(np.full(nb, float(x)), a_m * bc, b2_m)
        else:
            rt = al_bin_logp(bc, 0.0, b1_t) + al_bin_logp(np.full(nb, float(x)), a_t * bc, b2_t)
            rm = al_bin_logp(bc, a_m * x, b2_m)
        lt[qi] = rt - R._logsumexp(rt)
        lm[qi] = rm - R._logsumexp(rm)
    return R.exact_gain(lt, lt, lm)


def run_d2(prior, scale, seed, dose, n_pairs, M, rng):
    import core as C

    if prior == "N":
        # At d=2 under a Gaussian residual law the two order-conditional anchors
        # coincide by construction (measured denominators at median -3e-18), so
        # every pair fails MIN_DENOM and the run dies several minutes later with
        # "no pairs survived", which reads as a data problem rather than as the
        # structural fact it is. d=3 already refuses N up front; so does this now.
        raise SystemExit(
            "prior 'N' is the Gaussian-null arm: at d=2 its two order-conditional "
            "anchors are identical, so the exact achievable gain is zero for every "
            "pair and eta is undefined rather than small. This is the same "
            "order-invariance that puts the eps=0 identified-set width at exactly 1.")
    if n_pairs < 1:
        raise SystemExit(f"--n-pairs must be >= 1, got {n_pairs}")

    model = C.load_pfn(scale, prior, seed, dose)
    num, den, neut, subs, klt, klm, cls_list, tails2 = [], [], [], [], [], [], [], []
    dropped = 0
    t0 = time.time()
    for _ in range(n_pairs):
        pr = C.matched_pair(prior, rng)
        cls = pr["cls_true"]
        pT, pM = pr["params_true"], pr["params_mirror"]
        # Queries are drawn from the true process exactly as E1b draws them; the
        # sampled outcome is DISCARDED, which is the whole of the E.3 repair.
        xA, _y_discarded = C.fresh_outcomes(prior, cls, pT, M, rng)
        pG1, pG2 = (pT, pM) if cls == 1 else (pM, pT)
        logpG1, logpG2 = C.anchor_logprobs(pG1, pG2, prior, xA)
        logp_true, logp_mirror = (logpG1, logpG2) if cls == 1 else (logpG2, logpG1)

        d_exact = R.exact_gain(logp_true, logp_true, logp_mirror)
        if not np.isfinite(d_exact) or d_exact <= R.MIN_DENOM:
            dropped += 1
            continue
        lq_true = C.model_predictive(model, pr["D_true"], xA)
        lq_mirror = C.model_predictive(model, pr["D_mirror"], xA)
        z1 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
        z2 = C._noise_std_resid(prior, C.N_CONTEXT, rng)
        lq_same = C.model_predictive(
            model, C.context_from_std(prior, cls, pT, z1, z2), xA)

        num.append(R.exact_gain(logp_true, lq_true, lq_mirror))
        den.append(d_exact)
        neut.append(R.exact_gain(logp_true, lq_true, lq_same))
        a, b = R.mirror_arm_control(logp_true, lq_true, lq_mirror)
        klt.append(a); klm.append(b)
        # Independent route to the same denominator: the substrate's own
        # anchor_kl, indexed by CLASS rather than by true/mirror.
        # core.anchor_kl on the SAME arrays is the same formula on the same inputs
        # and cannot see an indexing error. The independent route rebuilds each
        # order-conditional predictive from its parameter triple through the AL
        # density directly, then recomputes the gain.
        subs.append(R.cross_path_deviation(
            d_exact, _d2_independent_gain(prior, pT, pM, cls, xA,
                                          C.BIN_CENTERS, C.BIN_EDGES[1] - C.BIN_EDGES[0])))
        tails2.append((R.check_edge_mass(logp_true, "p_true")[0],
                       R.check_edge_mass(logp_mirror, "p_mirror")[0]))
        cls_list.append(cls)

    return _agg(num, den, neut, subs, klt, klm, cls_list, dict(
        d=2, O=2, prior=prior, scale=scale, model_seed=seed, dose=dose,
        n_pairs_requested=n_pairs, n_pairs_kept=len(num),
        n_pairs_dropped_below_min_denom=dropped,
        M_measure=M, M_select=None,
        max_edge_bin_mass_true=float(np.max([a for a, _ in tails2])) if tails2 else float("nan"),
        max_edge_bin_mass_mirror=float(np.max([b for _, b in tails2])) if tails2 else float("nan"),
        max_edge_bin_mass_select=float("nan"),   # d=2 selects nothing; kept for schema parity
        mirror_convention="definitional: O=2, the unique opposite ordering",
        wallclock_s=time.time() - t0))


def run_d3(prior, seed, dose, n_pairs, M, M_select, rng, nets_dir):
    import exp4_d3_readout as X

    if prior not in X.R_OF:
        # `R_OF.get(prior, 2.0)` would silently hand prior="N" an AL(r=2) skew and
        # score AL data against AL anchors while the checkpoint is the Gaussian-null
        # arm -- a finite, plausible, meaningless number. exp4's Gaussian arm has its
        # own generator and anchors and this file does not reimplement them.
        raise SystemExit(
            f"prior {prior!r} has no registered skew (R_OF = {sorted(X.R_OF)}). "
            "The Gaussian-null arm needs exp4.anchor_logprobs_gauss and a "
            "gaussian=True generator; it is not supported here.")
    if n_pairs < 1:
        raise SystemExit(f"--n-pairs must be >= 1, got {n_pairs}")
    orc = X.load_orc()
    ORDERINGS = list(permutations(range(X.D_DIM)))
    r = X.R_OF[prior]
    model = X.load_model(prior, seed, dose, nets_dir)

    checked, assert_worst = R.assert_predictive_is_sound(orc, X, prior, X.Q0, n_sigma=20)

    # Same batching argument as inside the assert: one draw for the whole cell.
    # This is the single largest cost in the d=3 arm and it is pure substrate
    # sampling -- no model, no estimand -- so batching changes which Sigmas are
    # drawn and nothing about their law.
    sigmas = orc.sample_Sigmas(rng, n_pairs)

    num, den, neut, subs, klt, klm, cls_list = [], [], [], [], [], [], []
    num_duel, den_duel = [], []
    mirrors, margins_z, ncand, ntied, tails, degen = [], [], [], [], [], []
    tails_mir, tails_sel = [], []
    dropped = 0
    duel_dropped = 0
    t0 = time.time()
    for pair_i in range(n_pairs):
        S = sigmas[pair_i]
        cls = int(rng.integers(0, 2)) + 1
        piT = X.PI_A if cls == 1 else X.PI_B
        piD = X.PI_B if cls == 1 else X.PI_A          # the d=2-comparable duel partner
        z = orc.al_sample(1.0, r, (X.N_CONTEXT, X.D_DIM), rng)

        x0s, x1s, _y = X.fresh_outcomes(prior, cls, S, M, rng, orc)
        qx = np.stack([x0s, x1s], axis=1)
        s0s, s1s, _ys = X.fresh_outcomes(prior, cls, S, M_select, rng, orc)
        qsel = np.stack([s0s, s1s], axis=1)

        def enum(grid):
            out = []
            for pi in ORDERINGS:
                _, U, b = orc.params_for(S[None], pi)
                out.append(R.ordering_logpredictive(
                    pi, U[0], b[0], r, grid, X.BIN_CENTERS, X.LOG_BW, orc.al_logpdf))
            return out

        logp = enum(qx)
        logp_sel = enum(qsel)
        ti = ORDERINGS.index(piT)
        tails.append(R.check_edge_mass(logp[ti], 'p_true')[0])
        mi, d_exact, runner, ncd, ntd, top_sel = R.max_contrast_mirror(
            logp, ti, logp_select=logp_sel)
        if mi is None or not np.isfinite(d_exact) or d_exact <= R.MIN_DENOM:
            dropped += 1
            continue

        piM = ORDERINGS[mi]
        D_true = X.context_from_std3(S, piT, z, orc)
        lq_true = X.model_predictive_batch(model, D_true[None], qx[None])[0]
        lq_mirror = X.model_predictive_batch(
            model, X.context_from_std3(S, piM, z, orc)[None], qx[None])[0]
        lq_duel = X.model_predictive_batch(
            model, X.context_from_std3(S, piD, z, orc)[None], qx[None])[0]
        z_same = orc.al_sample(1.0, r, (X.N_CONTEXT, X.D_DIM), rng)
        lq_same = X.model_predictive_batch(
            model, X.context_from_std3(S, piT, z_same, orc)[None], qx[None])[0]

        num.append(R.exact_gain(logp[ti], lq_true, lq_mirror))
        den.append(d_exact)
        di = ORDERINGS.index(piD)
        dd = R.exact_gain(logp[ti], logp[ti], logp[di])
        # The duel denominator is the one that actually approaches zero on this
        # substrate (measured min 1.0e-5 at prior C against 0.143 for the
        # max-contrast arm), so the survivorship guard belongs here, not only on
        # the arm that never needs it.
        if np.isfinite(dd) and dd > R.MIN_DENOM:
            num_duel.append(R.exact_gain(logp[ti], lq_true, lq_duel))
            den_duel.append(dd)
        else:
            duel_dropped += 1
        neut.append(R.exact_gain(logp[ti], lq_true, lq_same))
        a, b = R.mirror_arm_control(logp[ti], lq_true, lq_mirror)
        klt.append(a); klm.append(b)
        # The driver's own numerator expression with the model's two arguments
        # replaced by the exact predictives, checked against the denominator
        # max_contrast_mirror returned from its own path.
        # Independent route: rebuild both predictives through forward substitution
        # (inverting Lunit) rather than through U, then recompute the gain.
        subs.append(R.cross_path_deviation(
            d_exact, R.independent_gain(orc, X, S, piT, piM, r, qx)))
        cls_list.append(cls)
        tails_mir.append(R.check_edge_mass(logp[mi], "p_mirror")[0])
        # The SELECTION grid is the one that actually picks the mirror and it was
        # never checked. It escapes the grid far more than the measurement grid
        # does: 1.8% of pairs above 0.01 at prior C, worst 0.20 against a hard stop
        # of 0.25.
        tails_sel.append(max(R.check_edge_mass(logp_sel[ti], "p_true_select")[0],
                             R.check_edge_mass(logp_sel[mi], "p_mirror_select")[0]))
        mirrors.append(",".join(map(str, piM)))
        degen.append(ntd > 1)
        # The margin is standardized by ITS OWN query sampling error, not by the
        # denominator: den is several times the relevant noise scale, so a margin
        # measured against it reports stability the selection does not have.
        # Both terms on the SELECTION grid, standardized by that grid's own error.
        gs = np.sum(np.exp(logp_sel[ti]) * (logp_sel[ti] - logp_sel[mi]), axis=1)
        margins_z.append(float((top_sel - runner) /
                               (gs.std(ddof=1) / math.sqrt(len(gs)) + 1e-12))
                         if np.isfinite(runner) else float("nan"))
        ncand.append(ncd); ntied.append(ntd)

    duel = R.ratio_of_means(np.asarray(num_duel), np.asarray(den_duel))
    hist = {}
    for m_ in mirrors:
        hist[m_] = hist.get(m_, 0) + 1
    mz = np.asarray(margins_z)
    return _agg(num, den, neut, subs, klt, klm, cls_list, dict(
        d=3, O=6, prior=prior, scale="d3", model_seed=seed, dose=dose,
        n_pairs_requested=n_pairs, n_pairs_kept=len(num),
        n_pairs_dropped_below_min_denom=dropped,
        M_measure=M, M_select=M_select,
        mirror_convention="max-contrast argmax over predictive equivalence classes (E.7)",
        mirror_histogram=hist,
        mean_candidate_classes=float(np.mean(ncand)),
        frac_pairs_tied_at_top=float(np.mean(np.asarray(ntied) > 1)),
        argmax_margin_z_median=float(np.nanmedian(mz)) if len(mz) else float("nan"),
        frac_argmax_margin_z_below_1p96=R.frac_below(mz, 1.96)[0],
        n_argmax_margins_defined=R.frac_below(mz, 1.96)[1],
        max_edge_bin_mass_true=float(np.max(tails)) if tails else float("nan"),
        mean_edge_bin_mass_true=float(np.mean(tails)) if tails else float("nan"),
        max_edge_bin_mass_mirror=float(np.max(tails_mir)) if tails_mir else float("nan"),
        mean_edge_bin_mass_mirror=float(np.mean(tails_mir)) if tails_mir else float("nan"),
        max_edge_bin_mass_select=float(np.max(tails_sel)) if tails_sel else float("nan"),
        eta_duel=duel["eta"], eta_duel_se=duel["eta_se"],
        num_duel_mean=duel["num_mean"], den_duel_mean=duel["den_mean"],
        n_duel=duel["n"], n_duel_dropped_below_min_denom=duel_dropped,
        den_duel_min_observed=float(np.min(den_duel)) if den_duel else float("nan"),
        den_duel_q01_observed=(float(np.quantile(den_duel, 0.01))
                               if den_duel else float("nan")),
        frac_pairs_mirror_class_degenerate=float(np.mean(degen)),
        train_ordering_support="uniform over all 6 (d3_train_fleet.py:86 "
                               "fams = rng.integers(0, len(ORDERINGS), BATCH))",
        predictive_soundness_checks=checked,
        predictive_soundness_max_delta=assert_worst,
        wallclock_s=time.time() - t0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, required=True, choices=(2, 3))
    p.add_argument("--prior", required=True)
    p.add_argument("--scale", default="base")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dose", type=int, required=True)
    p.add_argument("--n-pairs", type=int, default=800)
    p.add_argument("--M", type=int, default=M_MEASURE_DEFAULT)
    p.add_argument("--M-select", type=int, default=M_SELECT_DEFAULT)
    p.add_argument("--eval-seed", type=int, default=901)
    p.add_argument("--nets-dir", default=os.environ.get("NETS3"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true",
                   help="stamp DIAGNOSTIC so a require_live reader refuses this file")
    a = p.parse_args(argv)
    if a.d == 3 and not a.nets_dir:
        raise SystemExit("--d 3 needs --nets-dir (or NETS3); load_model would "
                         "otherwise die inside os.path.join with a TypeError")

    rng = np.random.default_rng(a.eval_seed)
    if a.d == 2:
        doc = run_d2(a.prior, a.scale, a.seed, a.dose, a.n_pairs, a.M, rng)
    else:
        doc = run_d3(a.prior, a.seed, a.dose, a.n_pairs, a.M, a.M_select, rng, a.nets_dir)

    doc["eval_seed"] = a.eval_seed
    doc["env"] = R.env_block(source_files=(
        os.path.join(_HERE, "r6_eta.py"), os.path.join(_HERE, "r6_run.py")))
    R.stamp(doc, "diagnostic" if a.diagnostic else "live")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2))
    keys = ("d", "prior", "scale", "model_seed", "dose", "eta", "eta_se",
            "eta_duel", "eta_duel_se", "num_mean", "den_mean", "eta_neutral",
            "substitution_control_max_dev", "n", "frac_pairs_eta_gt_1")
    print(json.dumps({k: doc[k] for k in keys if k in doc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
