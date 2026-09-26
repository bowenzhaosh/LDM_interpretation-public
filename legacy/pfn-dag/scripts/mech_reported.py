#!/usr/bin/env python
"""Reported-not-gated re-fits on the Amendment G confirmatory artifacts (FIGURE_PLAN.md L1-L4, L6).

Amendment G is LOCKED (AMENDMENT_G.sha256). Nothing this script produces is a gate; every
output file is stamped "REPORTED_NOT_GATED". It never writes into confirm/, predgain_confirm/
or the verdict, and it does not modify scripts/mech_gates.py: it IMPORTS the registered recipe
from there (loaders, panel assertions, OLS, bootstrap RNG scheme, N_BOOT, SE_MAX), so a re-fit
of a gate cell reproduces the verdict's b, a and SE(b) to the last bit (asserted per cell in
refits.json[*].matches_verdict) and every reported cell is fitted by the same code path.

L1  refits.json          (a, b, SE(b), r) for EVERY scored gate-panel cell on disk, seed-averaged
                         and per seed: the 10k/25k/2M dose rungs, every LR-grid arm, and the
                         wave-2 cells (small, large@eps0.5, large@2M) once they are scored.
L2  l2_dose_pairs.json   paired b differences between consecutive doses of the same recipe
                         (k, o asserted equal), incl. 500k -> 2M for base@1e-3 at each eps.
L3  l3_decomposition.json
      bins       G-decile / quintile conditional means of regret with SE, per cell
      matched_G  G1 (100k, 500k), the adjacent-eps contrasts, and G3, re-fitted after
                 reweighting both cells to the pooled G distribution (definition below),
                 bootstrap SE and z; and a common-support-trimmed OLS as a secondary estimator
      census     G_i <= 0; regret_i < 0 (model outscores q_full); S_prior > S_abl
      bands      atom band mean(S_abl - S_prior), order band mean(S_full - S_abl), ratio
L4  l4_invariance.json   G_i invariance across cells at fixed eps (max |dG_i| vs base@1e-3@100k)
                         and the per-context Q1 width spread from ident_grid (F1c error bars)
L6  l6_sampler_disagreement.json
                         per-row SMC-vs-MCMC |dNLL| and order-posterior JS from the oracle-
                         precision pilot's joined_raw.npz, checked against verification.json
L9  l9_identification.json
                         is b identified as order-specific? per cell: r^2 of the OLS, Cook's distance,
                         b after dropping the 5 most influential contexts (by |G| and by Cook's D), the
                         (G_i <= 0) x (regret_i < 0) cross-tab, and difficulty proxies (H(q_full),
                         S(q_full) level) regressed against regret with and without G.
L8  l8_walltimes.json    measured Slurm wall-clock (sacct ElapsedRaw) of every Amendment G training job,
                         one job = three seeds on one GPU, and the compute normalisation G6/G3 need.
L7  l7_g4_readout.json   the G4 statistic mean_i[JS(w_proj, unif) - JS(w_proj, w_exact)] at every eps
                         with a readout on disk: eps 0.5 / 0.75 reproduce the verdict bit-for-bit
                         (same mech_gates.g4_cell, same RNG label); eps 1.0 (FIGURE_PLAN C1, job
                         238153) is REPORTED. Per-context arrays for F4(c) and the reference scale
                         JS(w_exact, unif).

Matched-G definitions (L3), three estimators of b_hi - b_lo on a common G distribution, all
from pooled-quantile bins of G_i over the two cells, all bootstrapped on the same context
resamples (N_BOOT; independent per cell when the cells are different worlds, joint when they
share contexts; bins, weights and strata recomputed inside every resample; RNG = BOOT_SEED +
label, as mech_gates):
  matched    reweight each cell to the pooled distribution: a context of cell C in pooled decile
             j gets weight p_pool(j) / p_C(j); bins empty in either cell are dropped from both
             (retained fractions reported); weighted OLS in each cell. Exact match of the
             weighted G distributions, but a thin tail in one cell receives extreme weights.
  matched_to_lo / matched_to_hi
             the same reweighting with ONE cell as the target: bins are the target cell's own
             deciles, the target is unweighted, the other cell gets p_target(j) / p_other(j).
             The three targets (lo, pooled, hi) bracket the recipe-dependence of the contrast;
             none is registered.
  trimmed    plain OLS in each cell restricted to the common support [max of minima, min of
             maxima]. Same range, not the same distribution within it.
  stratified within each pooled bin (deciles and quintiles) an OLS slope per cell where both
             have >= 3 contexts; the per-bin slope differences averaged with pooled bin mass.
             The local-slope comparison at matched G; low power because within-bin G ranges
             are narrow.
The registered G1/G3 statistics are the RAW (unmatched) contrasts; every matched number is
descriptive and is printed beside the raw one, never in its place.

Usage (from the repo root):
  .venv/bin/python scripts/mech_reported.py [--only l1,l3] [--out campaigns/mech_20260827/reported]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mech_gates as MG  # noqa: E402  -- G.0-digested; imported, never edited
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

STATUS = "REPORTED_NOT_GATED"
SEEDS = [3, 4, 5]
UNITS = ["mean"] + SEEDS
CAMP = ROOT / "campaigns/mech_20260827"
DEFAULT_ROOTS = [CAMP / "predgain_confirm", CAMP / "predgain_confirm_n40"]
VERDICT = CAMP / "confirm/AMENDMENT_G_VERDICT.json"
IDENT_GRID = ROOT / "campaigns/corrected_20260812/raw/postF/ident_grid"
PILOT = ROOT / "campaigns/phase1_ordering_20260803/oracle_precision_pilot_run1"
LOCK = ROOT / "AMENDMENT_G.lock.json"
REF_CELL = ("base", MG.REG_LR, 100_000)           # L4 invariance reference (G2's lower rung)
N_BINS_MATCH = 10
DOSE_LADDER = (10_000, 25_000, 100_000, 500_000, 2_000_000)

CELL_RE = re.compile(r"^(?P<arch>[a-z]+(?:_n40)?)_lr(?P<lr>[0-9.e-]+)_d(?P<dose>\d+)$")
FILE_RE = re.compile(r"^predgain_eps(?P<tag>[0-9p]+)_ck(?P<step>\d+)(?:_(?P<kind>[a-z0-9]+))?\.npz$")

_INPUTS: dict = {}


def _rel(p: Path) -> str:
    return os.path.relpath(Path(p).resolve(), ROOT)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _note_input(p: Path) -> None:
    k = _rel(p)
    if k not in _INPUTS:
        _INPUTS[k] = _sha(p)


def _fnum(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


# ------------------------------------------------------------------ discovery

def discover(roots) -> list:
    cells = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            m = CELL_RE.match(d.name)
            if not (d.is_dir() and m):
                continue
            arch, lr, dose = m["arch"], float(m["lr"]), int(m["dose"])
            for f in sorted(d.glob("predgain_eps*_ck*.npz")):
                fm = FILE_RE.match(f.name)
                if not fm:
                    continue
                eps = float(fm["tag"].replace("p", "."))
                step, kind = int(fm["step"]), fm["kind"] or "orig"
                if step != dose:            # hot checkpoint of a longer run: never fitted here
                    print(f"  skip (step {step} != dose {dose}, not a decayed endpoint): {f}")
                    continue
                if eps_tag(eps) != fm["tag"]:
                    raise SystemExit(f"{f}: eps tag {fm['tag']} does not round-trip through eps_tag({eps})")
                z = np.load(f)
                if int(z["step"]) != step or str(z["kind"]) != kind or d.name not in str(z["nets"]):
                    raise SystemExit(f"{f}: artifact says step={int(z['step'])} kind={str(z['kind'])} nets={str(z['nets'])}; "
                                     f"filename says step={step} kind={kind} cell={d.name}")
                cells.append(dict(dir=d, rel=_rel(d), arch=arch, lr=lr, dose=dose, eps=eps,
                                  step=step, kind=kind, file=f, scale=MG.scale_tag(arch, lr)))
                _note_input(f)
    keys = [(c["rel"], c["eps"], c["step"], c["kind"]) for c in cells]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"duplicate (dir, eps, step, kind) among scored files: {[k for k in keys if keys.count(k) > 1]}")
    return cells


def meta(c: dict) -> dict:
    return {"dir": c["rel"], "arch": c["arch"], "lr": c["lr"], "dose": c["dose"], "eps": c["eps"],
            "step": c["step"], "kind": c["kind"], "scale": c["scale"], "file": _rel(c["file"])}


def verdict_index(v: dict) -> dict:
    idx = {}

    def add(c):
        idx[(os.path.normpath(c["dir"]), float(c["eps"]), int(c["step"]), str(c["seed"]))] = c

    for c in v["cells"]:
        add(c)
    for c in v["n40_cells"]:
        add(c)
    for g in ("G5a_capacity_eps075", "G5b_capacity_eps1", "G6_capacity_vs_dose"):
        for part in v["gates"][g]["parts"]:
            for arm in (part.get("cells") or {}).values():
                for c in arm:
                    add(c)
    return idx


def raw_arrays(c: dict):
    """S query-mean per predictor, k, o for one cell (the mech_gates loader with its assertions)."""
    z = MG._load_raw(c["dir"], c["eps"], c["step"], c["kind"], c["scale"])
    names = list(z["names"])
    S = z["S"].mean(axis=1)
    cols = {n: S[:, i] for i, n in enumerate(names)}
    model_cols = [cols[n] for n in names if n.startswith("model_s")]
    cols["model_mean"] = np.mean(model_cols, axis=0)
    return cols, z["k"], z["o"]


# ------------------------------------------------------------------ L1

def l1_refits(cells, vidx) -> tuple[list, dict]:
    out, fits = [], {}
    for c in cells:
        for u in UNITS:
            label = f"cell:{c['arch']}:{c['lr']}:{c['eps']}:{c['step']}:{u}"
            r = MG.cell(c["dir"], c["eps"], c["step"], u, None, kind=c["kind"],
                        scale_expected=c["scale"], label=label)
            fits[(c["rel"], c["eps"], c["step"], c["kind"], str(u))] = r
            vc = vidx.get((os.path.normpath(c["rel"]), c["eps"], c["step"], str(u)))
            rec = {**meta(c), "unit": str(u), "b": r["b"], "a": r["a"], "se_b": r["se_b"],
                   "r_regret_G": r["r_regret_G"], "G_mean": r["G_order_mean"],
                   "regret_mean": r["regret_mean"], "n_ctx": r["n_ctx"],
                   "se_within_SE_MAX": r["evaluable"], "boot_label": label,
                   "in_verdict": vc is not None}
            if vc is not None:
                rec["matches_verdict"] = bool(abs(vc["b"] - r["b"]) < 1e-12 and abs(vc["a"] - r["a"]) < 1e-12
                                              and abs(vc["se_b"] - r["se_b"]) < 1e-12)
                rec["verdict"] = {"b": vc["b"], "a": vc["a"], "se_b": vc["se_b"]}
            out.append(rec)
    return out, fits


# ------------------------------------------------------------------ L2

def l2_dose_pairs(cells, fits, verdict) -> list:
    groups: dict = {}
    for c in cells:
        if c["kind"] != "orig":
            continue
        groups.setdefault((c["arch"], c["lr"], c["eps"]), []).append(c)
    g2 = {}
    for part in verdict["gates"]["G2_dose_direction"]["parts"]:
        g2[float(part["primary"]["eps"])] = part
    out = []
    for (arch, lr, eps), cs in sorted(groups.items()):
        cs = sorted(cs, key=lambda c: c["dose"])
        for lo, hi in zip(cs, cs[1:]):
            is_g2 = (arch, lr, lo["dose"], hi["dose"]) == ("base", MG.REG_LR, 100_000, 500_000)
            rec = {"arch": arch, "lr": lr, "eps": eps, "dose_lo": lo["dose"], "dose_hi": hi["dose"],
                   "dir_lo": lo["rel"], "dir_hi": hi["rel"], "in_verdict_as": "G2" if is_g2 else None,
                   "units": []}
            for u in UNITS:
                label = (f"G2:{eps}:{lo['dose']}:{hi['dose']}:{u}" if is_g2
                         else f"L2:{arch}:{lr}:{eps}:{lo['dose']}:{hi['dose']}:{u}")
                f_lo, f_hi = fits[(lo["rel"], eps, lo["step"], "orig", str(u))], fits[(hi["rel"], eps, hi["step"], "orig", str(u))]
                t = MG.contrast_paired((lo["dir"], eps, lo["step"], "orig", lo["scale"]),
                                       (hi["dir"], eps, hi["step"], "orig", hi["scale"]), u, None, MG.ALPHA,
                                       f_lo, f_hi, label=label)
                # mech_gates refuses the bootstrap when a mean cell has SE(b) > SE_MAX; the point
                # difference is still a number and is carried with se/z = None in that case.
                ur = {"unit": str(u), "boot_label": label, "b_lo": f_lo["b"], "b_hi": f_hi["b"],
                      "evaluable": bool(t.get("evaluable")), "diff": t.get("diff", f_hi["b"] - f_lo["b"]),
                      "se": t.get("se"), "z": t.get("z"), "z_below_minus_z01": t.get("passes"),
                      "se_within_SE_MAX": {"lo": f_lo["evaluable"], "hi": f_hi["evaluable"]}}
                if is_g2 and eps in g2:
                    src = g2[eps]["primary"] if u == "mean" else next(
                        s for s in g2[eps]["per_seed"] if str(s["seed"]) == str(u))
                    ur["matches_verdict"] = bool(abs(src["diff"] - t["diff"]) < 1e-12
                                                 and abs(src["se"] - t["se"]) < 1e-12)
                rec["units"].append(ur)
            out.append(rec)
    for eps in (0.5, 0.75, 1.0):
        if not any(r["in_verdict_as"] == "G2" and r["eps"] == eps for r in out):
            raise SystemExit(f"L2: the registered G2 pair (base@1e-3, 100k->500k, eps={eps}) was not found on disk")
    return out


# ------------------------------------------------------------------ L3

def _wols(G, r, w):
    w = w / w.sum()
    Gm, rm = (w * G).sum(), (w * r).sum()
    b = (w * (G - Gm) * (r - rm)).sum() / (w * (G - Gm) ** 2).sum()
    return float(b), float(rm - b * Gm)


def _matched_weights(GA, GB, n_bins):
    pool = np.concatenate([GA, GB])
    edges = np.quantile(pool, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    jA = np.clip(np.searchsorted(edges, GA, side="right") - 1, 0, n_bins - 1)
    jB = np.clip(np.searchsorted(edges, GB, side="right") - 1, 0, n_bins - 1)
    nA, nB = np.bincount(jA, minlength=n_bins), np.bincount(jB, minlength=n_bins)
    keep = (nA > 0) & (nB > 0)
    if keep.sum() == 0:
        raise SystemExit("matched-G: no pooled bin holds contexts from both cells")
    p_pool = np.where(keep, nA + nB, 0).astype(float)
    p_pool /= p_pool.sum()
    pA = np.where(keep, nA, 0) / nA[keep].sum()
    pB = np.where(keep, nB, 0) / nB[keep].sum()
    wA = np.where(keep[jA], p_pool[jA] / np.where(pA[jA] > 0, pA[jA], 1), 0.0)
    wB = np.where(keep[jB], p_pool[jB] / np.where(pB[jB] > 0, pB[jB], 1), 0.0)
    return wA, wB, float(nA[keep].sum() / len(GA)), float(nB[keep].sum() / len(GB)), int(keep.sum()), float(max(wA.max(), wB.max()))


def _target_weights(G_t, G_o, n_bins):
    """Weights making the OTHER cell's G histogram match the TARGET cell's decile histogram.
    Bins are the target's own deciles between its real min and max; other-cell contexts outside
    the target's support get weight 0 and are counted in frac_outside_other.
    Returns (w_target, w_other, retained_frac_target, retained_frac_other, n_bins_kept, frac_outside_other)."""
    edges = np.quantile(G_t, np.linspace(0, 1, n_bins + 1))
    jt = np.clip(np.searchsorted(edges, G_t, side="right") - 1, 0, n_bins - 1)
    inside = (G_o >= edges[0]) & (G_o <= edges[-1])
    jo = np.clip(np.searchsorted(edges, G_o, side="right") - 1, 0, n_bins - 1)
    nt = np.bincount(jt, minlength=n_bins)
    no = np.bincount(jo[inside], minlength=n_bins)
    keep = (nt > 0) & (no > 0)
    if keep.sum() == 0:
        raise SystemExit("matched-G (target): no target bin holds contexts from both cells")
    pt = np.where(keep, nt, 0) / nt[keep].sum()
    po = np.where(keep, no, 0) / no[keep].sum()
    w_o = np.where(inside & keep[jo], pt[jo] / np.where(po[jo] > 0, po[jo], 1), 0.0)
    w_t = keep[jt].astype(float)
    return (w_t, w_o, float(w_t.mean()), float((w_o > 0).mean()), int(keep.sum()), float(1 - inside.mean()))


def _stratified(GA, rA, GB, rB, n_bins):
    """Pooled-bin-mass average of the per-bin OLS slope differences (b_B - b_A)."""
    pool = np.concatenate([GA, GB])
    edges = np.quantile(pool, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    jA = np.clip(np.searchsorted(edges, GA, side="right") - 1, 0, n_bins - 1)
    jB = np.clip(np.searchsorted(edges, GB, side="right") - 1, 0, n_bins - 1)
    diffs, wts, slopes = [], [], []
    for j in range(n_bins):
        sA, sB = jA == j, jB == j
        if sA.sum() < 3 or sB.sum() < 3 or GA[sA].std() == 0 or GB[sB].std() == 0:
            continue
        bA, bB = MG._ols(rA[sA], GA[sA])[0], MG._ols(rB[sB], GB[sB])[0]
        diffs.append(bB - bA); wts.append(sA.sum() + sB.sum()); slopes.append((j, bA, bB, int(sA.sum()), int(sB.sum())))
    if not diffs:
        raise SystemExit("stratified matched-G: no bin holds >= 3 contexts from both cells")
    wts = np.array(wts, float)
    return float((np.array(diffs) * wts).sum() / wts.sum()), slopes


def _matched_contrast(rA, GA, rB, GB, paired: bool, label: str) -> dict:
    """b_B - b_A on a common G distribution: pooled-reweighted, common-support-trimmed and
    stratified (see the module docstring). One bootstrap over contexts (joint if paired)
    drives all three."""
    rng = MG._rng(label)
    wA, wB, fA, fB, nb, wmax = _matched_weights(GA, GB, N_BINS_MATCH)
    bA, aA = _wols(GA, rA, wA)
    bB, aB = _wols(GB, rB, wB)
    lo, hi = max(GA.min(), GB.min()), min(GA.max(), GB.max())
    mA, mB = (GA >= lo) & (GA <= hi), (GB >= lo) & (GB <= hi)
    tA, tB = MG._ols(rA[mA], GA[mA])[0], MG._ols(rB[mB], GB[mB])[0]
    s10, slopes10 = _stratified(GA, rA, GB, rB, 10)
    s5, slopes5 = _stratified(GA, rA, GB, rB, 5)
    # one cell as the target
    tA_w, oB_w, fA_lo, fB_lo, nb_lo, out_lo = _target_weights(GA, GB, N_BINS_MATCH)      # target = lo (A)
    tB_w, oA_w, fB_hi, fA_hi, nb_hi, out_hi = _target_weights(GB, GA, N_BINS_MATCH)      # target = hi (B)
    lo_bA, lo_bB = _wols(GA, rA, tA_w)[0], _wols(GB, rB, oB_w)[0]
    hi_bA, hi_bB = _wols(GA, rA, oA_w)[0], _wols(GB, rB, tB_w)[0]
    dm, dt, d10, d5, dlo, dhi = [], [], [], [], [], []
    n = len(GA)
    for _ in range(MG.N_BOOT):
        iA = MG._boot_idx(n, rng)
        iB = iA if paired else MG._boot_idx(len(GB), rng)
        gA, gB, xA, xB = GA[iA], GB[iB], rA[iA], rB[iB]
        vA, vB, *_ = _matched_weights(gA, gB, N_BINS_MATCH)
        dm.append(_wols(gB, xB, vB)[0] - _wols(gA, xA, vA)[0])
        ta, ob, *_ = _target_weights(gA, gB, N_BINS_MATCH)
        dlo.append(_wols(gB, xB, ob)[0] - _wols(gA, xA, ta)[0])
        tb, oa, *_ = _target_weights(gB, gA, N_BINS_MATCH)
        dhi.append(_wols(gB, xB, tb)[0] - _wols(gA, xA, oa)[0])
        l2, h2 = max(gA.min(), gB.min()), min(gA.max(), gB.max())
        sA, sB = (gA >= l2) & (gA <= h2), (gB >= l2) & (gB <= h2)
        dt.append(MG._ols(xB[sB], gB[sB])[0] - MG._ols(xA[sA], gA[sA])[0])
        d10.append(_stratified(gA, xA, gB, xB, 10)[0])
        d5.append(_stratified(gA, xA, gB, xB, 5)[0])
    dm, dt, d10, d5, dlo, dhi = (np.array(x) for x in (dm, dt, d10, d5, dlo, dhi))
    raw = MG._ols(rB, GB)[0] - MG._ols(rA, GA)[0]
    return {
        "boot_label": label, "paired_resample": paired, "n_bins": N_BINS_MATCH, "n_bins_kept": nb,
        "matched_to_lo": {"target": "lo", "b_lo": lo_bA, "b_hi": lo_bB, "diff": lo_bB - lo_bA, "se": float(dlo.std(ddof=1)),
                          "z": float((lo_bB - lo_bA) / dlo.std(ddof=1)), "retained_frac_lo": fA_lo, "retained_frac_hi": fB_lo,
                          "n_bins_kept": nb_lo, "frac_hi_outside_lo_support": out_lo},
        "matched_to_hi": {"target": "hi", "b_lo": hi_bA, "b_hi": hi_bB, "diff": hi_bB - hi_bA, "se": float(dhi.std(ddof=1)),
                          "z": float((hi_bB - hi_bA) / dhi.std(ddof=1)), "retained_frac_lo": fA_hi, "retained_frac_hi": fB_hi,
                          "n_bins_kept": nb_hi, "frac_lo_outside_hi_support": out_hi},
        "stratified_deciles": {"diff": s10, "se": float(d10.std(ddof=1)), "z": float(s10 / d10.std(ddof=1)),
                               "per_bin": [{"bin": j, "b_lo": a_, "b_hi": b_, "n_lo": na, "n_hi": nb_} for j, a_, b_, na, nb_ in slopes10]},
        "stratified_quintiles": {"diff": s5, "se": float(d5.std(ddof=1)), "z": float(s5 / d5.std(ddof=1)),
                                 "per_bin": [{"bin": j, "b_lo": a_, "b_hi": b_, "n_lo": na, "n_hi": nb_} for j, a_, b_, na, nb_ in slopes5]},
        "raw": {"b_lo": MG._ols(rA, GA)[0], "b_hi": MG._ols(rB, GB)[0], "diff": raw},
        "matched": {"b_lo": bA, "b_hi": bB, "a_lo": aA, "a_hi": aB, "diff": bB - bA,
                    "se": float(dm.std(ddof=1)), "z": float((bB - bA) / dm.std(ddof=1)),
                    "ci99_upper_onesided": float((bB - bA) + MG.Z[0.01] * dm.std(ddof=1)),
                    "retained_frac_lo": fA, "retained_frac_hi": fB, "max_weight": wmax,
                    "G_mean_lo_raw": float(GA.mean()), "G_mean_hi_raw": float(GB.mean()),
                    "G_mean_lo_weighted": float((wA / wA.sum() * GA).sum()),
                    "G_mean_hi_weighted": float((wB / wB.sum() * GB).sum())},
        "trimmed": {"support": [float(lo), float(hi)], "n_lo": int(mA.sum()), "n_hi": int(mB.sum()),
                    "b_lo": tA, "b_hi": tB, "diff": tB - tA, "se": float(dt.std(ddof=1)),
                    "z": float((tB - tA) / dt.std(ddof=1))},
    }


def l3_decomposition(cells, verdict) -> dict:
    by = {(c["arch"], c["lr"], c["dose"], c["eps"], c["kind"]): c for c in cells}
    arrays = {}

    def get(key):
        if key not in arrays:
            arrays[key] = raw_arrays(by[key])
        return arrays[key]

    def rg(key, unit):
        cols, k, o = get(key)
        m = cols["model_mean"] if unit == "mean" else cols[f"model_s{unit}"]
        return cols["full"] - m, cols["full"] - cols["abl"], k, o

    # ---- bins
    bins = []
    for key, c in sorted(by.items(), key=lambda kv: kv[1]["rel"] + str(kv[1]["eps"])):
        reg, G, _, _ = rg(key, "mean")
        rec = {**meta(c), "unit": "mean"}
        for name, q in (("decile", 10), ("quintile", 5)):
            edges = np.quantile(G, np.linspace(0, 1, q + 1))
            j = np.clip(np.searchsorted(edges, G, side="right") - 1, 0, q - 1)
            rows = []
            for bnum in range(q):
                s = j == bnum
                rows.append({"bin": bnum, "n": int(s.sum()), "G_lo": float(edges[bnum]), "G_hi": float(edges[bnum + 1]),
                             "G_mean": float(G[s].mean()), "regret_mean": float(reg[s].mean()),
                             "regret_se": float(reg[s].std(ddof=1) / np.sqrt(s.sum())),
                             "frac_regret_lt0": float((reg[s] < 0).mean())})
            rec[name] = rows
        bins.append(rec)

    # ---- matched-G contrasts
    base = ("base", MG.REG_LR)
    specs = []
    for dose in (100_000, 500_000):
        specs.append(("G1", dose, (*base, dose, 0.5, "orig"), (*base, dose, 1.0, "orig"), False))
        specs.append(("eps0.5->0.75", dose, (*base, dose, 0.5, "orig"), (*base, dose, 0.75, "orig"), False))
        specs.append(("eps0.75->1.0", dose, (*base, dose, 0.75, "orig"), (*base, dose, 1.0, "orig"), False))
    specs.append(("G3", 500_000, (*base, 500_000, 0.75, "orig"), ("base_n40", MG.REG_LR, 500_000, 0.75, "n40ext"), True))
    matched = []
    for name, dose, klo, khi, paired in specs:
        if klo not in by or khi not in by:
            matched.append({"contrast": name, "dose": dose, "status": "ABSENT", "lo": list(klo), "hi": list(khi)})
            continue
        rec = {"contrast": name, "dose": dose, "lo": meta(by[klo]), "hi": meta(by[khi]), "units": []}
        for u in UNITS:
            rA, GA, kA, oA = rg(klo, u)
            rB, GB, kB, oB = rg(khi, u)
            if paired and not (np.array_equal(kA, kB) and np.array_equal(oA, oB)):
                raise SystemExit(f"{name}: paired cells do not share latents")
            rec["units"].append({"unit": str(u), **_matched_contrast(
                rA, GA, rB, GB, paired, label=f"L3match:{name}:{dose}:{u}")})
        # the registered (unmatched) verdict number for the same contrast, for the caption
        if name == "G1":
            part = next(p for p in verdict["gates"]["G1_eps_direction"]["parts"] if int(p["primary"]["step"]) == dose)
            rec["verdict_raw"] = {"diff": part["primary"]["diff"], "se": part["primary"]["se"], "z": part["primary"]["z"]}
        if name == "G3":
            part = verdict["gates"]["G3_signal_share"]["parts"][0]
            rec["verdict_raw"] = {"diff": part["primary"]["diff"], "se": part["primary"]["se"], "z": part["primary"]["z"]}
        matched.append(rec)

    # ---- census + bands (every cell, unit mean; per-seed model-beats-full counts too)
    census, bands = [], []
    for key, c in sorted(by.items(), key=lambda kv: kv[1]["rel"] + str(kv[1]["eps"])):
        cols, _, _ = get(key)
        reg, G = cols["full"] - cols["model_mean"], cols["full"] - cols["abl"]
        n = len(G)
        census.append({**meta(c), "n_ctx": n,
                       "G_le0": {"n": int((G <= 0).sum()), "frac": float((G <= 0).mean()), "G_min": float(G.min()),
                                 "G_max": float(G.max())},
                       "model_outscores_full": {"unit_mean": int((reg < 0).sum()),
                                                **{f"seed_{s}": int(((cols["full"] - cols[f"model_s{s}"]) < 0).sum()) for s in SEEDS}},
                       "prior_outscores_abl": int((cols["prior"] > cols["abl"]).sum()),
                       "abl_outscores_full": int((cols["abl"] > cols["full"]).sum())})
        atom, order, total = cols["abl"] - cols["prior"], cols["full"] - cols["abl"], cols["full"] - cols["prior"]
        mg = cols["model_mean"] - cols["prior"]
        bands.append({**meta(c), "unit": "mean",
                      "atom_band": float(atom.mean()), "atom_band_se": float(atom.std(ddof=1) / np.sqrt(n)),
                      "order_band": float(order.mean()), "order_band_se": float(order.std(ddof=1) / np.sqrt(n)),
                      "atom_over_order": float(atom.mean() / order.mean()),
                      "total_available": float(total.mean()), "model_over_prior": float(mg.mean()),
                      "capture_total": float(mg.mean() / total.mean()),
                      "regret_mean": float(reg.mean()), "regret_over_order_band": float(reg.mean() / order.mean())})
    return {"bins": bins, "matched_G": matched, "census": census, "bands": bands}


# ------------------------------------------------------------------ L4

def l4_invariance(cells, ident_dir: Path) -> dict:
    by_eps: dict = {}
    for c in cells:
        by_eps.setdefault(c["eps"], []).append(c)
    inv = []
    for eps, cs in sorted(by_eps.items()):
        ref = next((c for c in cs if (c["arch"], c["lr"], c["dose"], c["kind"]) == (*REF_CELL, "orig")), None)
        if ref is None:
            inv.append({"eps": eps, "status": "NO_REFERENCE_CELL", "reference": list(REF_CELL)})
            continue
        colsR, kR, oR = raw_arrays(ref)
        GR = colsR["full"] - colsR["abl"]
        rows = []
        for c in sorted(cs, key=lambda c: c["rel"]):
            cols, k, o = raw_arrays(c)
            G = cols["full"] - cols["abl"]
            same = bool(len(k) == len(kR) and np.array_equal(k, kR) and np.array_equal(o, oR))
            d = np.abs(G - GR) if same else None
            rows.append({**meta(c), "same_latents": same,
                         "max_abs_dG": float(d.max()) if same else None, "mean_abs_dG": float(d.mean()) if same else None,
                         "max_abs_dS_full": float(np.abs(cols["full"] - colsR["full"]).max()) if same else None,
                         "max_abs_dS_abl": float(np.abs(cols["abl"] - colsR["abl"]).max()) if same else None,
                         "G_mean": float(G.mean()), "note": None if same else "latents differ from the reference: dG not defined"})
        inv.append({"eps": eps, "reference": meta(ref), "cells": rows,
                    "max_abs_dG_over_orig_cells": float(max(r["max_abs_dG"] for r in rows if r["kind"] == "orig" and r["same_latents"]))})

    width = []
    for f in sorted(ident_dir.glob("ident_eps*_highs_ipm.json")):
        _note_input(f)
        g = json.load(open(f))
        vals, keypath = [], None
        for ctx in g["contexts"]:
            q1 = ctx["Q1"]
            v = q1.get("avg_width")
            if v is None:
                for kk, vv in q1.items():          # one level down, first key named avg_width
                    if isinstance(vv, dict) and "avg_width" in vv:
                        v, keypath = vv["avg_width"], f"Q1.{kk}.avg_width"
                        break
            else:
                keypath = "Q1.avg_width"
            vals.append(None if v is None else float(v))
        ok = [v for v in vals if v is not None]
        if not ok:
            raise SystemExit(f"{f}: no Q1.avg_width in any context")
        arr = np.array(ok)
        width.append({"file": _rel(f), "eps": float(g["world"]["eps"]), "key": keypath, "n_ctx": len(ok),
                      "n_missing": len(vals) - len(ok), "values": ok,
                      "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(ok) > 1 else None,
                      "se": float(arr.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else None,
                      "min": float(arr.min()), "max": float(arr.max()),
                      "aggregates_Q1": g.get("aggregates", {}).get("Q1"),
                      "Q1_keys_ctx0": sorted(g["contexts"][0]["Q1"].keys())})
    return {"G_invariance": inv, "Q1_width_spread": width}


# ------------------------------------------------------------------ L6

def _js_rows(p, q):
    return MG._js(p, q)


def l6_sampler(pilot: Path) -> dict:
    """Prior codes follow the verifier's fixed convention (src/pfn_dag_verify/pilot_verify.py:69:
    0 -> C, 1 -> N). The five NLL/JS statistics must reproduce verification.json under that
    mapping or the run refuses. The registered row_catastrophe count (15 for C) is the FULL-NLL-only
    count (CLAIM_LEDGER C8's wording); the verifier source's formula (full OR ablated) gives 16.
    Both are written; neither is changed."""
    raw, ver = pilot / "joined_raw.npz", pilot / "verification.json"
    _note_input(raw); _note_input(ver)
    z = np.load(raw)
    v = json.load(open(ver))
    ps, pm = z["smc_order_posterior"], z["mcmc_order_posterior"]
    ps = ps / ps.sum(1, keepdims=True); pm = pm / pm.sum(1, keepdims=True)
    js = _js_rows(ps, pm)
    dfull = np.abs(z["smc_full_nll"] - z["mcmc_full_nll"])
    dabl = np.abs(z["smc_ablated_nll"] - z["mcmc_ablated_nll"])
    convention = {0: "C", 1: "N"}
    per_code, match = {}, {}
    for code, lab in convention.items():
        s = z["prior_code"] == code
        if s.sum() == 0:
            raise SystemExit(f"pilot: no rows with prior_code {code}")
        pc = {"label": lab, "n": int(s.sum()),
              "nll_full_median_abs": float(np.median(dfull[s])), "nll_full_max_abs": float(dfull[s].max()),
              "nll_ablated_median_abs": float(np.median(dabl[s])), "nll_ablated_max_abs": float(dabl[s].max()),
              "order_js_median": float(np.median(js[s])), "order_js_p95": float(np.percentile(js[s], 95)),
              "order_js_max": float(js[s].max()),
              "row_catastrophe_full_only": int((dfull[s] > 0.5).sum()),
              "row_catastrophe_full_or_ablated": int(((dfull[s] > 0.5) | (dabl[s] > 0.5)).sum()),
              "rows_js_gt_0p5": int((js[s] > 0.5).sum())}
        per_code[str(code)] = pc
        d = {"nll_full_median_abs": abs(pc["nll_full_median_abs"] - v[f"smc_mcmc_nll_full_median_abs_{lab}"]),
             "nll_ablated_median_abs": abs(pc["nll_ablated_median_abs"] - v[f"smc_mcmc_nll_ablated_median_abs_{lab}"]),
             "nll_full_max_abs": abs(pc["nll_full_max_abs"] - v[f"smc_mcmc_nll_full_max_abs_{lab}"]),
             "order_js_median": abs(pc["order_js_median"] - v[f"order_js_median_{lab}"]),
             "order_js_p95": abs(pc["order_js_p95"] - v[f"order_js_p95_{lab}"])}
        if max(d.values()) >= 1e-9:
            raise SystemExit(f"pilot: prior_code {code} does not reproduce verification.json's _{lab} statistics: {d}")
        match[f"code{code}={lab}"] = {"abs_diffs_nll_js": d, "all_within_1e-9": True,
                                     "registered_row_catastrophe": int(v[f"row_catastrophe_{lab}"]),
                                     "equals_full_only": bool(v[f"row_catastrophe_{lab}"] == pc["row_catastrophe_full_only"]),
                                     "equals_full_or_ablated": bool(v[f"row_catastrophe_{lab}"] == pc["row_catastrophe_full_or_ablated"])}
    return {"source": _rel(raw), "verification": _rel(ver), "n_rows": int(len(js)),
            "js_log_base": "e (nats)", "code_label": {str(k): lab for k, lab in convention.items()},
            "code_label_source": "src/pfn_dag_verify/pilot_verify.py:69 ((0, 'C'), (1, 'N'))",
            "per_prior_code": per_code, "match_verification": match,
            "catastrophe_note": "verification.json / TERMINAL_STATUS.json / CLAIM_LEDGER C8 record the FULL-NLL-only count; "
                                "pilot_verify.py:82 in the working tree computes full OR ablated. Flagged in WORKLOG 2026-09-01; not resolved here.",
            "rows": {"row_id": z["row_id"].tolist(), "prior_code": z["prior_code"].tolist(),
                     "dnll_full": dfull.tolist(), "dnll_ablated": dabl.tolist(), "order_js": js.tolist()}}


# ------------------------------------------------------------------ L7

def l7_g4_readout(readout: Path, fleet_cell: Path, verdict: dict, step: int = 500_000,
                  eps_list=(0.5, 0.75, 1.0)) -> dict:
    vparts = {float(p["primary"]["eps"]): p for p in verdict["gates"]["G4_instrument"]["parts"]}
    out = []
    for eps in eps_list:
        ex_path = readout / f"eps{eps_tag(eps)}.npz"
        projs = [readout / f"proj_eps{eps_tag(eps)}_s{s}_ck{step}.npz" for s in SEEDS]
        if not ex_path.is_file() or not all(pp.is_file() for pp in projs):
            out.append({"eps": eps, "step": step, "status": "ABSENT", "oracle": _rel(ex_path)})
            continue
        _note_input(ex_path)
        for pp in projs:
            _note_input(pp)
        prim = MG.g4_cell(readout, eps, step, list(SEEDS), None, MG.ALPHA, fleet_cell=fleet_cell)
        per = [MG.g4_cell(readout, eps, step, s, None, MG.ALPHA, fleet_cell=fleet_cell) for s in SEEDS]
        ex = np.load(ex_path)
        ref = MG._js(ex["w_exact"], ex["w_uniform"])
        ju, je, ok = [], [], []
        for pp in projs:
            pr = np.load(pp)
            w = pr["w_proj"]
            ok.append(np.isfinite(w).all(1))
            ju.append(MG._js(w, ex["w_uniform"]))
            je.append(MG._js(w, ex["w_exact"]))
        ok = np.all(ok, axis=0)
        ju, je = np.mean(ju, 0), np.mean(je, 0)
        # shape of the projected posterior (the attack lens's point: the gate statistic says w_proj is
        # nearer uniform than it is to the truth, NOT that w_proj is flatter than w_exact)
        def _ent(w):
            w = np.maximum(w, 1e-300)
            return -(w * np.log(w)).sum(1)
        cz = np.load(fleet_cell / f"predgain_eps{eps_tag(eps)}_ck{step}.npz")
        same_panel = all(int(cz[k]) == int(ex[k]) for k in ("panel_seed", "split_seed", "n_per_half", "n_rows"))
        o_true = cz["o"] if same_panel else None
        if o_true is not None and not (ex["w_exact"].shape[1] == 6 and int(o_true.max()) < 6 and len(o_true) == len(ex["w_exact"])):
            raise SystemExit(f"G4 eps={eps}: ordering code / column mismatch between the readout and the fleet cell")
        ents, maxw, agree, true_o = [], [], [], []
        for pp in projs:
            w = np.load(pp)["w_proj"][ok]
            ents.append(float(_ent(w).mean())); maxw.append(float(np.median(w.max(1))))
            agree.append(float((w.argmax(1) == ex["w_exact"][ok].argmax(1)).mean()))
            if o_true is not None:
                true_o.append(float((w.argmax(1) == o_true[ok]).mean()))
        shape = {"O": int(ex["w_exact"].shape[1]), "log_O": float(np.log(ex["w_exact"].shape[1])),
                 "proj_entropy_mean_seedavg": float(np.mean(ents)), "exact_entropy_mean": float(_ent(ex["w_exact"][ok]).mean()),
                 "proj_median_max_weight_seedavg": float(np.mean(maxw)),
                 "js_proj_uniform_mean": float(ju[ok].mean()), "js_proj_exact_mean": float(je[ok].mean()), "js_exact_uniform_mean": float(ref[ok].mean()),
                 "argmax_proj_eq_argmax_exact_seedavg": float(np.mean(agree)),
                 "argmax_proj_eq_true_o_seedavg": float(np.mean(true_o)) if true_o else None,
                 "argmax_exact_eq_true_o": float((ex["w_exact"][ok].argmax(1) == o_true[ok]).mean()) if o_true is not None else None,
                 "true_o_source": _rel(fleet_cell / f"predgain_eps{eps_tag(eps)}_ck{step}.npz") if same_panel else "PANEL MISMATCH: not computed",
                 "assumption": "w_exact columns are indexed by the same ordering code as the fleet cell's o array "
                               "(both enumerate o in range(O) through corrected_oracle.exact_joint_posterior)"}
        rec = {"eps": eps, "step": step, "status": "OK", "oracle": _rel(ex_path), "projections": [_rel(pp) for pp in projs],
               "proj_shape": shape,
               "in_verdict": eps in vparts, "primary": prim, "per_seed": per,
               "sign_agreement": bool(all(t["diff"] < 0 for t in per)),
               "ref_js_exact_uniform": {"mean": float(ref.mean()), "median": float(np.median(ref)),
                                        "p05": float(np.percentile(ref, 5)), "p95": float(np.percentile(ref, 95))},
               "n_ctx_all_seeds_solved": int(ok.sum()),
               "per_context": {"solved": ok.tolist(),
                               "js_proj_uniform_seedmean": [float(x) if o_ else None for x, o_ in zip(ju, ok)],
                               "js_proj_exact_seedmean": [float(x) if o_ else None for x, o_ in zip(je, ok)],
                               "js_exact_uniform": ref.tolist()}}
        if eps in vparts:
            vp = vparts[eps]["primary"]
            rec["matches_verdict"] = bool(abs(vp["diff"] - prim["diff"]) < 1e-12 and abs(vp["se"] - prim["se"]) < 1e-12
                                          and all(abs(a["diff"] - b["diff"]) < 1e-12 for a, b in zip(vparts[eps]["per_seed"], per)))
            rec["verdict"] = {"diff": vp["diff"], "se": vp["se"], "z": vp["z"]}
        out.append(rec)
    return {"readout_dir": _rel(readout), "fleet_cell": _rel(fleet_cell), "cells": out,
            "note": "G4 is registered at eps 0.5 and 0.75 only; the eps 1.0 cell is REPORTED (C1, Slurm 238153)."}


# ------------------------------------------------------------------ L8

PARAMS = {"base": 279_140, "large": 2_136_676, "small": 40_612, "base_n40": 279_140}
JOB_RE = re.compile(r"^ldm-G-(?P<arch>[a-z]+(?:_n40)?)-lr(?P<lr>[0-9.]+)-d(?P<dose>\d+)-e(?P<eps>[0-9.]+)$")


def l8_walltimes(sacct_path: Path, verdict: dict) -> dict:
    """sacct -X rows (JobID JobName State Elapsed ElapsedRaw NNodes AllocTRES Start End) for the
    Amendment G training jobs. One job trains the three seeds of one (arch, lr, dose, eps) cell on
    one GPU, so ElapsedRaw is the wall-clock of a cell, not of a seed. Parameter counts come from
    AMENDMENT_G.md G.5 and are asserted against any *.provenance.json found locally."""
    _note_input(sacct_path)
    jobs, running = [], []
    row_re = re.compile(r"^(?P<job>\S+)\s+(?P<name>ldm-G-\S+)\s+(?P<state>.+?)\s+(?P<elapsed>(?:\d+-)?\d+:\d\d:\d\d)\s+(?P<raw>\d+)\s+(?P<nnodes>\d+)\s+(?P<alloc>\S+)")
    for line in open(sacct_path):
        m = row_re.match(line.strip())
        if not m:
            continue
        jm = JOB_RE.match(m["name"])
        if not jm:
            continue
        arch, lr, dose, eps = jm["arch"], float(jm["lr"]), int(jm["dose"]), float(jm["eps"])
        rec = {"job": m["job"], "name": m["name"], "state": m["state"], "elapsed": m["elapsed"], "elapsed_s": int(m["raw"]),
               "arch": arch, "lr": lr, "dose": dose, "eps": eps, "n_rows": 40 if arch.endswith("_n40") else 20,
               "params": PARAMS.get(arch), "param_steps": PARAMS.get(arch, 0) * dose,
               "rows_seen": (40 if arch.endswith("_n40") else 20) * dose, "alloc": m["alloc"]}
        (jobs if m["state"] == "COMPLETED" else running).append(rec)
    names = [j["name"] for j in jobs]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise SystemExit(f"L8: more than one COMPLETED job per cell name: {dups}")
    prov = sorted(ROOT.glob("campaigns/**/nets/**/*.provenance.json"))
    prov_checked = {}
    for pf in prov[:50]:
        try:
            pj = json.load(open(pf)); n = pj.get("n_params"); sc = str(pj.get("scale", pf.name.split("_s")[0]))
        except Exception:
            continue
        arch = sc.split("_lr")[0]
        if n is not None and arch in PARAMS:
            prov_checked[arch] = {"n_params": int(n), "file": _rel(pf), "matches_G5": bool(int(n) == PARAMS[arch])}
            if int(n) != PARAMS[arch]:
                raise SystemExit(f"L8: n_params for {arch} in {pf} = {n} != AMENDMENT_G G.5 value {PARAMS[arch]}")

    def find(arch, lr, dose, eps):
        return next((j for j in jobs if (j["arch"], j["lr"], j["dose"], j["eps"]) == (arch, lr, dose, eps)), None)

    def ratio(num, den, key):
        return (num[key] / den[key]) if (num and den and den[key]) else None

    sel = {p["label"] + ":" + g: p["selected_lr"] for g in ("G5a_capacity_eps075", "G5b_capacity_eps1", "G6_capacity_vs_dose")
           for p in verdict["gates"][g]["parts"] if "selected_lr" in p}
    lr_g5a = verdict["gates"]["G5a_capacity_eps075"]["parts"][0]["selected_lr"]
    lr_g6 = verdict["gates"]["G6_capacity_vs_dose"]["parts"][0]["selected_lr"]
    g6_l, g6_b = find("large", lr_g6["large"], 500_000, 0.75), find("base", lr_g6["base"], 2_000_000, 0.75)
    g6_b_recipe = find("base", MG.REG_LR, 2_000_000, 0.75)
    g3_40, g3_20 = find("base_n40", MG.REG_LR, 500_000, 0.75), find("base", MG.REG_LR, 500_000, 0.75)
    g5a_l, g5a_b = find("large", lr_g5a["large"], 500_000, 0.75), find("base", lr_g5a["base"], 500_000, 0.75)
    norm = {
        "G6 large@LR*@500k vs base@LR*(2M)@2M (the registered G6 arms)": {
            "selected_lr": lr_g6, "large": g6_l, "base": g6_b,
            "wall_ratio_large_over_base": ratio(g6_l, g6_b, "elapsed_s"),
            "param_steps_ratio_large_over_base": ratio(g6_l, g6_b, "param_steps")},
        "G6 large@LR*@500k vs base@1e-3@2M (the recipe-rate 2M cell, for reference)": {
            "large": g6_l, "base": g6_b_recipe, "wall_ratio_large_over_base": ratio(g6_l, g6_b_recipe, "elapsed_s")},
        "G5a large@LR*@500k vs base@LR*@500k": {
            "selected_lr": lr_g5a, "large": g5a_l, "base": g5a_b, "wall_ratio_large_over_base": ratio(g5a_l, g5a_b, "elapsed_s"),
            "param_steps_ratio_large_over_base": ratio(g5a_l, g5a_b, "param_steps")},
        "G3 n40 vs n20 (base@1e-3@500k)": {
            "n40": g3_40, "n20": g3_20, "wall_ratio_n40_over_n20": ratio(g3_40, g3_20, "elapsed_s"),
            "rows_seen_ratio_n40_over_n20": ratio(g3_40, g3_20, "rows_seen"),
            "rows_seen_n40_500k_over_n20_2M": (g3_40["rows_seen"] / (20 * 2_000_000)) if g3_40 else None},
    }
    return {"source": _rel(sacct_path), "unit": "one Slurm job = the three seeds of one cell on one GPU (condo-cse5100)",
            "params_source": "AMENDMENT_G.md G.5 (base 279,140; large 2,136,676; small 40,612)", "params": PARAMS,
            "params_provenance_check": prov_checked, "selected_lr_source": sel,
            "n_completed": len(jobs), "n_running_excluded": len(running), "running": running,
            "jobs": jobs, "normalisation": norm}


# ------------------------------------------------------------------ L9

def _cooks(G, r):
    X = np.column_stack([np.ones_like(G), G])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    res = r - X @ beta
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p_ = 2
    s2 = (res ** 2).sum() / (len(G) - p_)
    return (res ** 2 / (p_ * s2)) * h / (1 - h) ** 2


def l9_identification(cells) -> dict:
    """Facts a reviewer asks for when told 'b is a regression slope': fit quality, leverage, sign
    cross-tab, and whether context difficulty explains regret once G is in the model. All
    descriptive; nothing here is a gate."""
    targets = [("base", MG.REG_LR, 500_000, e, "orig") for e in (0.5, 0.75, 1.0)] + \
              [("large", 0.0003, 500_000, e, "orig") for e in (0.75, 1.0)] + \
              [("base", MG.REG_LR, 2_000_000, e, "orig") for e in (0.5, 0.75, 1.0)]
    by = {(c["arch"], c["lr"], c["dose"], c["eps"], c["kind"]): c for c in cells}
    out = []
    for key in targets:
        if key not in by:
            out.append({"cell": list(key), "status": "ABSENT"}); continue
        c = by[key]
        z = MG._load_raw(c["dir"], c["eps"], c["step"], c["kind"], c["scale"])
        names = list(z["names"]); S = z["S"].mean(axis=1)
        col = {n: S[:, i] for i, n in enumerate(names)}
        model = np.mean([col[n] for n in names if n.startswith("model_s")], axis=0)
        reg, G = col["full"] - model, col["full"] - col["abl"]
        b, a = MG._ols(reg, G)
        r2 = float(np.corrcoef(reg, G)[0, 1] ** 2)
        D = _cooks(G, reg)
        topG = np.argsort(-np.abs(G))[:5]; topD = np.argsort(-D)[:5]
        keepG = np.ones(len(G), bool); keepG[topG] = False
        keepD = np.ones(len(G), bool); keepD[topD] = False
        # difficulty proxies: entropy of the full predictive (mean over queries) and its level S(q_full)
        P = z["P"].astype(np.float64)                        # (ctx, q, pred, bins)
        pf = P[:, :, names.index("full"), :]
        Hf = -(np.maximum(pf, 1e-300) * np.log(np.maximum(pf, 1e-300))).sum(-1).mean(1)
        Sf = col["full"]
        def r2_of(X, y):
            X = np.column_stack([np.ones(len(y))] + list(X))
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()), beta
        r2_H, _ = r2_of([Hf], reg); r2_S, _ = r2_of([Sf], reg)
        r2_GH, beta_GH = r2_of([G, Hf], reg); r2_GS, beta_GS = r2_of([G, Sf], reg)
        neg = G <= 0
        out.append({**meta(c), "unit": "mean", "b": b, "a": a, "r2": r2,
                    "cooks_D_max": float(D.max()), "n_cooks_D_gt_4_over_n": int((D > 4 / len(G)).sum()),
                    "b_drop5_largest_absG": MG._ols(reg[keepG], G[keepG])[0],
                    "b_drop5_largest_cooksD": MG._ols(reg[keepD], G[keepD])[0],
                    "crosstab": {"n_G_le0": int(neg.sum()), "n_G_le0_and_model_beats_full": int((neg & (reg < 0)).sum()),
                                 "frac_model_beats_full_given_G_le0": float((reg[neg] < 0).mean()) if neg.any() else None,
                                 "frac_model_beats_full_given_G_gt0": float((reg[~neg] < 0).mean())},
                    "difficulty_proxies": {"r2_regret_on_H_full": r2_H, "r2_regret_on_S_full": r2_S,
                                           "r2_regret_on_G": r2, "r2_regret_on_G_and_H": r2_GH, "r2_regret_on_G_and_S": r2_GS,
                                           "b_with_H_in_model": float(beta_GH[1]), "b_with_S_in_model": float(beta_GS[1]),
                                           "H_full_mean": float(Hf.mean()), "corr_G_H": float(np.corrcoef(G, Hf)[0, 1]),
                                           "corr_G_S": float(np.corrcoef(G, Sf)[0, 1])}})
    return {"cells": out, "note": "descriptive; 'drop 5' refits are leverage checks, not estimates"}


# ------------------------------------------------------------------ main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="+", default=DEFAULT_ROOTS)
    p.add_argument("--verdict", type=Path, default=VERDICT)
    p.add_argument("--ident-grid", type=Path, default=IDENT_GRID)
    p.add_argument("--pilot", type=Path, default=PILOT)
    p.add_argument("--out", type=Path, default=CAMP / "reported")
    p.add_argument("--readout", type=Path, default=CAMP / "confirm/phase1")
    p.add_argument("--sacct", type=Path, default=CAMP / "reported/sacct_ldmG_wall.txt")
    p.add_argument("--only", type=str, default="l1,l2,l3,l4,l6,l7,l8,l9")
    p.add_argument("--force", action="store_true", help="allow a partial --only run into an output dir written by another script version")
    a = p.parse_args()
    only = set(a.only.lower().split(",")) | {"l1"}        # L1 always runs: it carries the verdict cross-check
    t0 = time.time()
    MG._SEEDS_EXPECTED[:] = SEEDS
    out_rel = _rel(a.out)
    for forbidden in ("campaigns/mech_20260827/confirm", "campaigns/mech_20260827/predgain_confirm", "campaigns/mech_20260827/predgain_confirm_n40",
                      "campaigns/mech_20260827/predgain_select"):
        if out_rel == forbidden or out_rel.startswith(forbidden + "/"):
            raise SystemExit(f"refusing --out {out_rel}: registered artifact directory")
    a.out.mkdir(parents=True, exist_ok=True)
    this_sha = _sha(Path(__file__))
    old_manifest = a.out / "MANIFEST.json"
    if old_manifest.is_file() and only != {"l1", "l2", "l3", "l4", "l6", "l7", "l8", "l9"} and not a.force:
        prev = json.load(open(old_manifest)).get("script_sha256")
        if prev != this_sha:
            raise SystemExit(f"partial --only run into {out_rel} whose MANIFEST was written by script {str(prev)[:12]} != this {this_sha[:12]}; "
                             f"run the full set or pass --force")
    _note_input(a.verdict)
    if LOCK.is_file():
        _note_input(LOCK)
    verdict = json.load(open(a.verdict))
    if verdict["headline"] != {k: "PASS" for k in verdict["headline"]}:
        print("NOTE: verdict headline is not all-PASS:", verdict["headline"])
    vidx = verdict_index(verdict)
    cells = discover(a.roots)
    print(f"[reported] {len(cells)} scored (cell, eps) files under {[_rel(r) for r in a.roots]}")

    stamp = {"status": STATUS, "amendment": "G (LOCKED; nothing here is a gate)", "script": "scripts/mech_reported.py",
             "verdict": _rel(a.verdict), "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED, "se_max": MG.SE_MAX,
             "seeds": SEEDS, "primary_unit": "per-context mean over seeds (decision-B convention)"}

    written: dict = {}

    def dump(name, payload):
        (a.out / name).write_text(json.dumps({**stamp, **payload}, indent=1, default=_fnum, allow_nan=False))
        written[name] = _sha(a.out / name)
        print(f"  wrote {_rel(a.out / name)}")

    # L1 first, always: every other section's stamp carries the verdict cross-check result
    refits, fits = l1_refits(cells, vidx)
    nv = [r for r in refits if r["in_verdict"]]
    bad = [r for r in nv if not r["matches_verdict"]]
    print(f"[L1] {len(refits)} fits; {len(nv)} are verdict cells, {len(nv) - len(bad)} reproduce the verdict bit-for-bit"
          + (f"; MISMATCH: {[(r['dir'], r['eps'], r['unit']) for r in bad]}" if bad else ""))
    if bad or len(nv) != len(vidx):
        raise SystemExit(f"verdict cross-check FAILED: {len(nv)} of {len(vidx)} verdict cells found on disk, {len(bad)} mismatched; "
                         f"nothing is written")
    stamp["verdict_reproduced"] = {"n": len(nv), "of": len(vidx), "mismatch": len(bad), "bit_for_bit": True}
    dump("refits.json", {"cells": refits, "n_verdict_cells": len(nv), "n_verdict_mismatch": len(bad)})
    if "l2" in only:
        pairs = l2_dose_pairs(cells, fits, verdict)
        print(f"[L2] {len(pairs)} paired dose steps")
        for r in pairs:
            m = r["units"][0]
            se = f"{m['se']:.4f}" if m["se"] is not None else "n/a(SE>SE_MAX)"
            z = f"{m['z']:+.2f}" if m["z"] is not None else "n/a"
            print(f"   {r['arch']:9s} lr={r['lr']:<7g} eps={r['eps']:<4} {r['dose_lo']:>7}->{r['dose_hi']:<7} "
                  f"diff={m['diff']:+.4f} se={se} z={z}" + ("  [G2]" if r["in_verdict_as"] else ""))
        dump("l2_dose_pairs.json", {"pairs": pairs, "note": "REPORTED. G2's registered range is 100k->500k; "
                                    "no plateau/saturation language attaches to any other step whatever it says."})
    if "l3" in only:
        d3 = l3_decomposition(cells, verdict)
        for m in d3["matched_G"]:
            if m.get("status") == "ABSENT":
                print(f"[L3] {m['contrast']} @ {m['dose']}: ABSENT"); continue
            u = m["units"][0]
            print(f"[L3] {m['contrast']:12s} dose={m['dose']:<7} raw diff={u['raw']['diff']:+.4f}  matched "
                  f"{u['matched']['diff']:+.4f} +/- {u['matched']['se']:.4f} (z {u['matched']['z']:+.2f}, "
                  f"kept {u['matched']['retained_frac_lo']:.2f}/{u['matched']['retained_frac_hi']:.2f})  trimmed "
                  f"{u['trimmed']['diff']:+.4f} +/- {u['trimmed']['se']:.4f}  to-lo {u['matched_to_lo']['diff']:+.4f} +/- {u['matched_to_lo']['se']:.4f}  "
                  f"to-hi {u['matched_to_hi']['diff']:+.4f} +/- {u['matched_to_hi']['se']:.4f}  strat10 "
                  f"{u['stratified_deciles']['diff']:+.4f} +/- {u['stratified_deciles']['se']:.4f}  strat5 "
                  f"{u['stratified_quintiles']['diff']:+.4f} +/- {u['stratified_quintiles']['se']:.4f}")
        for c in d3["census"]:
            if (c["arch"], c["lr"], c["dose"]) == ("base", MG.REG_LR, 500_000):
                print(f"[L3] census base@1e-3 500k eps={c['eps']}: G<=0 {c['G_le0']['n']} ({c['G_le0']['frac']:.3f}, "
                      f"min {c['G_le0']['G_min']:+.4f}); model>full {c['model_outscores_full']['unit_mean']}; "
                      f"prior>abl {c['prior_outscores_abl']}")
        for b in d3["bands"]:
            if (b["arch"], b["lr"], b["dose"]) == ("base", MG.REG_LR, 500_000):
                print(f"[L3] bands base@1e-3 500k eps={b['eps']}: atom {b['atom_band']:.4f} order {b['order_band']:.4f} "
                      f"ratio {b['atom_over_order']:.2f}")
        dump("l3_decomposition.json", {"matched_G_definition": __doc__.split("Matched-G definitions (L3)")[1].split("Usage (from")[0].strip(), **d3})
    if "l4" in only:
        d4 = l4_invariance(cells, a.ident_grid)
        for e in d4["G_invariance"]:
            if "cells" in e:
                print(f"[L4] eps={e['eps']}: max|dG_i| over orig cells = {e['max_abs_dG_over_orig_cells']:.3e}; "
                      + "; ".join(f"{r['arch']}: max {r['max_abs_dG']:.3e}" for r in e["cells"] if r["kind"] != "orig"))
        for w in d4["Q1_width_spread"]:
            print(f"[L4] width eps={w['eps']}: key={w['key']} n={w['n_ctx']} mean={w['mean']:.5f} sd={w['sd']}")
        dump("l4_invariance.json", d4)
    if "l6" in only:
        d6 = l6_sampler(a.pilot)
        print(f"[L6] codes->labels {d6['code_label']}; "
              + "; ".join(f"code{c}={v['label']}: js med {v['order_js_median']:.4f} p95 {v['order_js_p95']:.4f} catastrophe full-only {v['row_catastrophe_full_only']} / full|abl {v['row_catastrophe_full_or_ablated']}"
                          for c, v in d6["per_prior_code"].items()))
        dump("l6_sampler_disagreement.json", d6)
    if "l7" in only:
        d7 = l7_g4_readout(a.readout, MG.cell_dir(a.roots[0], "base", MG.REG_LR, 500_000), verdict)
        for c in d7["cells"]:
            if c["status"] != "OK":
                print(f"[L7] eps={c['eps']}: {c['status']}"); continue
            pr = c["primary"]
            print(f"[L7] G4 eps={c['eps']}: diff={pr['diff']:+.4f} se={pr['se']:.4f} z={pr['z']:+.2f} n_ctx={pr['n_ctx']} "
                  f"lp_failed={pr['n_lp_failed']} per-seed=" + ",".join(f"{t['diff']:+.4f}" for t in c["per_seed"])
                  + f"  ref JS(exact,unif) mean {c['ref_js_exact_uniform']['mean']:.4f}  JS(proj,unif) {c['proj_shape']['js_proj_uniform_mean']:.4f}"
                  + f"  H(proj) {c['proj_shape']['proj_entropy_mean_seedavg']:.3f} H(exact) {c['proj_shape']['exact_entropy_mean']:.3f} of log6 {c['proj_shape']['log_O']:.3f}"
                  + f"  argmax=exact {c['proj_shape']['argmax_proj_eq_argmax_exact_seedavg']:.3f} argmax=true {c['proj_shape']['argmax_proj_eq_true_o_seedavg']} (exact {c['proj_shape']['argmax_exact_eq_true_o']})"
                  + (f"  [verdict match: {c['matches_verdict']}]" if c["in_verdict"] else "  [REPORTED]"))
        dump("l7_g4_readout.json", d7)
    if "l8" in only and a.sacct.is_file():
        d8 = l8_walltimes(a.sacct, verdict)
        for k, v in d8["normalisation"].items():
            print(f"[L8] {k}: " + ", ".join(f"{kk}={vv:.3f}" for kk, vv in v.items() if isinstance(vv, float)))
        dump("l8_walltimes.json", d8)
    if "l9" in only:
        d9 = l9_identification(cells)
        for c in d9["cells"]:
            if c.get("status") == "ABSENT":
                continue
            dp = c["difficulty_proxies"]
            print(f"[L9] {c['arch']} lr={c['lr']:g} d={c['dose']} eps={c['eps']}: r2={c['r2']:.3f} CooksD max {c['cooks_D_max']:.2f} "
                  f"b={c['b']:.3f} drop5|G| {c['b_drop5_largest_absG']:.3f} drop5D {c['b_drop5_largest_cooksD']:.3f} | "
                  f"r2 on H {dp['r2_regret_on_H_full']:.3f} on S {dp['r2_regret_on_S_full']:.3f} b|H {dp['b_with_H_in_model']:.3f} | "
                  f"P(model>full|G<=0) {c['crosstab']['frac_model_beats_full_given_G_le0']:.2f} vs G>0 {c['crosstab']['frac_model_beats_full_given_G_gt0']:.2f}")
        dump("l9_identification.json", d9)

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unknown"
    lock = json.load(open(LOCK)) if LOCK.is_file() else {}
    mg_sha = _sha(ROOT / "scripts/mech_gates.py")
    manifest = {**stamp, "git_head": head, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "script_sha256": this_sha, "mech_gates_sha256": mg_sha,
                "mech_gates_matches_lock": bool(lock.get("digests", {}).get("scripts/mech_gates.py") == mg_sha),
                "python": platform.python_version(), "numpy": np.__version__, "only": sorted(only),
                "n_cells": len(cells), "cells": [meta(c) for c in cells], "inputs_sha256": _INPUTS,
                "elapsed_s": round(time.time() - t0, 1)}
    manifest["outputs_sha256"] = written
    manifest["skipped_sections"] = sorted({"l2", "l3", "l4", "l6", "l7", "l8", "l9"} - only) + (["l8 (no sacct file)"] if "l8" in only and not a.sacct.is_file() else [])
    manifest["verdict_reproduced"] = stamp["verdict_reproduced"]
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, allow_nan=False))
    print(f"[reported] MANIFEST written; mech_gates.py matches lock: {manifest['mech_gates_matches_lock']}; "
          f"{manifest['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
