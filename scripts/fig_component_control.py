#!/usr/bin/env python
"""F6 — the size-matched non-order component control (REPORTED, never gated).

Reading A: the PFN's shortfall against the exact predictive is specific to the ORDERING
component.  Reading B: it is specific to the SMALLEST component of the achievable gain,
whatever its kind.  The control mirrors every atom-coarsened oracle with an order-coarsened
one and compares slopes at MATCHED component size.

Sources, in order of authority:
  campaigns/mech_ext_20260902/reported/component_control.json   (the reported statistics;
      PRESPEC campaigns/mech_ext_20260902/PRESPEC_component_control.md, sha256 8c6697cf...)
  campaigns/mech_ext_20260902/coarse/coarse_eps{tag}.npz        (S_mean, G_bar, family)
  campaigns/mech_20260827/predgain_confirm/<cell>/predgain_eps{tag}_ck{step}.npz  (regret)
Panel (a) re-derives the two family curves from the coarse + registered .npz with the PRESPEC
estimator and REFUSES unless the re-derivation reproduces the reported medians and the
registered b to 1e-12.  Panels (b) and (c) read the reported JSON only.

Style follows scripts/figures_iclr.py (same rcParams, palette, panel labels, draft stamp).
Run from the repo root:
  .venv/bin/python scripts/fig_component_control.py [--draft] [--out DIR]

The DRAFT watermark is OPT-IN (`--draft`); submission renders carry none.
`--no-draft` is still accepted and is a no-op, so old invocations keep working.
figure_numbers.json is merged via scripts/figure_numbers_io.py (shared with figures_iclr.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_numbers_io import merge_figure_numbers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "campaigns/mech_ext_20260902/reported/component_control.json"
COARSE = ROOT / "campaigns/mech_ext_20260902/coarse"
OUT = ROOT / "paper/figs"

# ---- palette: identical to scripts/figures_iclr.py (validated 2026-09-01)
EPS_C = {0.5: "#86b6ef", 0.75: "#2a78d6", 1.0: "#104281"}
ATOM_C, ORDER_C = "#2a78d6", "#eb6834"          # blue / orange: colourblind-safe pair
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42,
})
W = 5.5
FIG = "F6"
NUMBERS: dict = {}
SRC_JSON = "mech_ext_20260902/reported/component_control.json (REPORTED, never gated)"

DECISION_CELL = "base_lr0.001_d500000"
LARGE_CELL = "large_lr0.0003_d500000"
PANEL_A_EPS = 0.75
CELL_LABEL = {"base_lr0.001_d10000": "base@1e-3, 10k", "base_lr0.001_d25000": "base@1e-3, 25k",
              "base_lr0.001_d100000": "base@1e-3, 100k", "base_lr0.001_d500000": "base@1e-3, 500k",
              "base_lr0.001_d2000000": "base@1e-3, 2M", "large_lr0.0003_d500000": "large@3e-4, 500k"}


def note(panel, key, value, source):
    NUMBERS.setdefault(FIG, {}).setdefault(panel, {})[key] = {"value": value, "source": source}


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right", color=INK)


def eps_tag(e):
    return {0.5: "0p5", 0.75: "0p75", 1.0: "1p0"}[e]


def rec(doc, cell, eps, unit="mean"):
    for r in doc["results"]:
        if (r["cell"], r["eps"], r["unit"]) == (cell, eps, unit):
            return r
    raise SystemExit(f"no record for {cell} eps {eps} unit {unit}")


def slopes(reg, G):
    """OLS slope of reg on each column of G (n, m) -> (m,); PRESPEC estimand, = mech_gates._ols."""
    Gc = G - G.mean(axis=0, keepdims=True)
    rc = reg - reg.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (Gc * rc[:, None]).sum(axis=0) / (Gc * Gc).sum(axis=0)


def family_curves(r):
    """Re-derive (Gbar, b, family) for every coarsened oracle; refuse unless it reproduces
    the reported medians and the registered b to 1e-12."""
    zc = np.load(COARSE / f"coarse_eps{eps_tag(r['eps'])}.npz")
    zr = np.load(ROOT / r["scored_file"])
    names = list(zr["names"])
    S = zr["S"].mean(axis=1)
    full = S[:, names.index("full")]
    model = S[:, [i for i, n in enumerate(names) if n.startswith("model_s")]].mean(axis=1)
    reg = full - model
    Sc = zc["S_mean"]
    if abs(float(np.max(np.abs(Sc[:, 0] - full)))) > 1e-9:
        raise SystemExit("coarse pass and scored cell disagree on S_i(full)")
    Gall = Sc[:, [0]] - Sc
    Gbar = Gall.mean(axis=0)
    b = slopes(reg, Gall)
    fam = np.array([str(f) for f in zc["family"]])
    gmin = r["G_min"]
    atom = (fam == "atom") & (Gbar >= gmin)
    order = (fam == "order") & (Gbar >= gmin)
    checks = {
        "n_atom_used": (int(atom.sum()), r["n_atom_used"], 0),
        "n_order_used": (int(order.sum()), r["n_order_used"], 0),
        "atom_b_median": (float(np.median(b[atom])), r["atom_family"]["b_median"], 1e-12),
        "order_b_median": (float(np.median(b[order])), r["order_family"]["b_median"], 1e-12),
        "Gbar_order": (float(Gbar[1]), r["Gbar_order"], 1e-12),
        "b_registered": (float(b[1]), r["b_order_registered_estimator"], 1e-12),
    }
    for k, (got, want, tol) in checks.items():
        if abs(got - want) > tol:
            raise SystemExit(f"REFUSING: panel (a) re-derivation disagrees on {k}: {got!r} vs {want!r}")
    print(f"  panel (a) re-derivation reproduces the reported {len(checks)} anchors exactly")
    return Gbar, b, fam, atom, order


def binned(x, y, edges):
    xs, ys, lo, hi = [], [], [], []
    for a, z in zip(edges, edges[1:]):
        m = (x >= a) & (x < z)
        if m.sum() >= 3:
            xs.append(float(np.sqrt(a * z))); ys.append(float(np.median(y[m])))
            lo.append(float(np.quantile(y[m], .25))); hi.append(float(np.quantile(y[m], .75)))
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


# ------------------------------------------------------------------ (a)
def panel_a(ax, doc):
    r = rec(doc, DECISION_CELL, PANEL_A_EPS)
    Gbar, b, fam, atom, order = family_curves(r)
    gmin, gstar = r["G_min"], r["Gbar_order"]
    d = r["delta"]["Gbar_order"]

    ax.axvspan(1.5e-3, gmin, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(np.sqrt(1.6e-3 * gmin), 0.33, "excluded: Ḡ$^X$ < 0.005", rotation=90, ha="center",
            va="center", fontsize=5.0, color=MUTED, zorder=1)
    ax.axhline(0, color=BASE, lw=0.8)

    # counts computed, not hardcoded, and stated as "plotted of enumerated": the caption's
    # 4140 / 203 are the ENUMERATED family sizes; 1 atom and 4 order oracles fall inside the
    # excluded Ḡ^X < 0.005 band and are not drawn.
    n_all = {"atom": int((np.asarray(fam) == "atom").sum()), "order": int((np.asarray(fam) == "order").sum())}
    for m, c, key, sym in ((atom, ATOM_C, "atom", "atom-coarsened Π"), (order, ORDER_C, "order", "order-coarsened H")):
        lab = f"{sym}  ({int(m.sum())} of {n_all[key]} plotted)"
        ax.scatter(Gbar[m], b[m], s=2.2, color=c, alpha=0.30, lw=0, zorder=2)
        xs, ys, lo, hi = binned(Gbar[m], b[m], np.geomspace(gmin, 0.42, 13))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.20, lw=0, zorder=3)
        ax.plot(xs, ys, color=c, lw=1.3, zorder=4, label=lab)

    note("a", "family_sizes", {"enumerated": n_all, "plotted_Gbar_ge_G_min": {"atom": int(atom.sum()), "order": int(order.sum())}, "G_min": gmin}, SRC_JSON)

    # the registered order ablation: the one point the spine's b is fitted on
    ax.plot(gstar, r["b_order_registered_estimator"], marker="*", ms=8, color=ORDER_C,
            mec=SURF, mew=0.5, lw=0, zorder=6)
    ax.annotate("registered $q_{\\rm abl}$: $b=%.3f$" % r["b_order_registered_estimator"],
                (gstar, r["b_order_registered_estimator"]), xytext=(5, 5),
                textcoords="offset points", fontsize=5.2, color=ORDER_C, va="bottom")
    ax.axvline(gstar, color=INK, lw=0.7, ls=(0, (2, 2)), zorder=5)
    ax.text(gstar * 1.15, 0.685, "Ḡ* = Ḡ$_{\\rm order}$ = %.4f" % gstar,
            fontsize=5.2, color=INK2, ha="left", va="top")

    # Delta at matched size: the 25-nearest medians on either side
    ao, bo = d["atom25_b_median"], d["order25_b_median"]
    ax.plot([gstar] * 2, [ao, bo], color=INK, lw=1.1, zorder=7, solid_capstyle="butt")
    for yv in (ao, bo):
        ax.plot([gstar * 0.93, gstar * 1.07], [yv, yv], color=INK, lw=1.1, zorder=7)
    ax.annotate("Δ(Ḡ*) = %+.3f\n[%.3f, %.3f]" % (d["delta"], *d["ci95"]), (gstar, 0.5 * (ao + bo)),
                xytext=(7, 0), textcoords="offset points", fontsize=5.6, color=INK,
                ha="left", va="center", fontweight="bold")

    ax.set_xscale("log"); ax.set_xlim(1.5e-3, 0.55); ax.set_ylim(-0.08, 0.70)
    ax.set_xlabel("component size Ḡ$^X$ (nats, oracle)")
    ax.set_ylabel("$b^X$: slope of regret$_i$ on $G^X_i$")
    ax.set_title("ε = %g, base@1e-3, 500k gate panel" % PANEL_A_EPS, loc="left", fontsize=6, color=INK2, pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=1.3, fontsize=5.5)

    note("a", "Gbar_order", gstar, SRC_JSON)
    note("a", "b_registered_abl", r["b_order_registered_estimator"], SRC_JSON + " (reproduces the spine's b)")
    note("a", "atom_family_median_b", r["atom_family"]["b_median"], SRC_JSON)
    note("a", "order_family_median_b", r["order_family"]["b_median"], SRC_JSON)
    note("a", "atom25_b_median", ao, SRC_JSON)
    note("a", "order25_b_median", bo, SRC_JSON)
    note("a", "delta_Gbar_order", {"delta": d["delta"], "ci95": d["ci95"], "se": d["se"], "z": d["z"]}, SRC_JSON)
    note("a", "n_atom_used", r["n_atom_used"], SRC_JSON)
    note("a", "n_order_used", r["n_order_used"], SRC_JSON)
    note("a", "atom_family_Gbar_range", r["atom_family"]["Gbar_range"], SRC_JSON)
    note("a", "order_family_Gbar_range", r["order_family"]["Gbar_range"], SRC_JSON)


# ------------------------------------------------------------------ (b)
def panel_b(ax, doc):
    rows = [(DECISION_CELL, e) for e in (0.5, 0.75, 1.0)] + [(LARGE_CELL, e) for e in (0.75, 1.0)]
    labs = []
    for i, (cell, eps) in enumerate(rows):
        r = rec(doc, cell, eps)
        labs.append(f"{CELL_LABEL[cell].split(',')[0]}, ε={eps:g}")
        for key, dy, mfc, ms in (("Gbar_order", +0.16, None, 4.2), ("0.02", -0.16, SURF, 3.6)):
            d = r["delta"][key]
            c = EPS_C[eps]
            ax.errorbar(d["delta"], i + dy, xerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]],
                        fmt="o", ms=ms, color=c, mfc=c if mfc is None else mfc, mew=1.0,
                        capsize=1.5, elinewidth=0.8, lw=0, zorder=3)
            note("b", f"{cell} eps{eps:g} Gstar={key}",
                 {"delta": d["delta"], "ci95": d["ci95"], "se": d["se"], "z": d["z"],
                  "Gstar": d["Gstar"], "readout": r["decision_rule_readout"]}, SRC_JSON)
        ax.text(-0.082, i - 0.40, labs[-1], fontsize=5.6, color=INK2, va="center", ha="left")
        if r["decision_rule_readout"] != "A":
            ax.text(0.255, i + 0.22, "CI covers 0:\nunresolved", fontsize=5.2, color=INK2, va="center", ha="left")
    ax.axvline(0, color=INK, lw=0.8)
    ax.axhline(2.5, color=BASE, lw=0.8)
    ax.text(0.585, 1.0, "decision cell\n(pre-specified)", rotation=90, fontsize=5.2, color=MUTED, va="center", ha="center")
    ax.text(0.585, 3.5, "reported", rotation=90, fontsize=5.2, color=MUTED, va="center", ha="center")
    ax.set_yticks([]); ax.set_ylim(len(rows) - 0.45, -0.65)
    ax.set_xlim(-0.09, 0.60); ax.set_xlabel("Δ(Ḡ*)  (95 % bootstrap CI)")
    ax.set_title("reading A ⇔ Δ > 0  ·  500k", loc="left", fontsize=6, color=INK2, pad=2)
    ax.legend(handles=[Line2D([], [], marker="o", ms=4.2, lw=0, color=INK2, label="Ḡ* = Ḡ$_{\\rm order}$(ε)"),
                       Line2D([], [], marker="o", ms=3.6, lw=0, color=INK2, mfc=SURF, label="Ḡ* = 0.02 (fixed)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=1.0, fontsize=5.5)


# ------------------------------------------------------------------ (c)
def panel_c(ax, doc):
    seeds = ["3", "4", "5"]
    n_pos = n_tot = 0
    cells_shown = set()
    for r in doc["results"]:
        if r["unit"] not in seeds or "delta" not in r["delta"]["Gbar_order"]:
            continue
        m = rec(doc, r["cell"], r["eps"])
        x, y = m["delta"]["Gbar_order"]["delta"], r["delta"]["Gbar_order"]["delta"]
        ax.plot(x, y, marker="D" if r["cell"] == LARGE_CELL else "o", ms=3.0,
                color=EPS_C[r["eps"]], mfc=SURF if r["cell"] == LARGE_CELL else EPS_C[r["eps"]],
                mew=0.8, lw=0, zorder=3)
        n_tot += 1; n_pos += int(y > 0); cells_shown.add((r["cell"], r["eps"]))
        note("c", f"{r['cell']} eps{r['eps']:g} s{r['unit']}",
             {"delta_seed": y, "delta_mean": x, "readout": r["decision_rule_readout"]}, SRC_JSON)
    lim = (-0.05, 1.20)
    ax.plot(lim, lim, color=BASE, lw=0.8, zorder=1)
    ax.axhline(0, color=INK, lw=0.8, zorder=2); ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Δ(Ḡ$_{\\rm order}$), seed-mean model"); ax.set_ylabel("Δ(Ḡ$_{\\rm order}$), single seed")
    # counted, never hardcoded: the shipped 2026-09-03T02:03 render said "17 cell×ε" and
    # 51/51 because it predated the 02:05 rewrite of component_control.json, which added the
    # per-seed units of large_lr0.0003_d500000 at ε=0.5 (review-claims-audit.md D4).
    ax.set_title(f"{len(cells_shown)} cell×ε × {n_tot // max(len(cells_shown), 1)} seeds",
                 loc="left", fontsize=6, color=INK2, pad=2)
    ax.text(0.04, 0.95, f"{n_pos}/{n_tot} per-seed Δ > 0", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8, color=INK, fontweight="bold")
    # the one cluster whose CI covers zero
    lz = [rec(doc, LARGE_CELL, 1.0, u)["delta"]["Gbar_order"]["delta"] for u in seeds]
    ax.annotate("large@3e-4, ε=1\n(CI covers 0)", (rec(doc, LARGE_CELL, 1.0)["delta"]["Gbar_order"]["delta"], float(np.mean(lz))),
                xytext=(0.97, 0.09), textcoords="axes fraction", fontsize=5.2, color=INK2, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=3))
    ax.legend(handles=[Line2D([], [], marker="o", ms=3.0, lw=0, color=EPS_C[e], label=f"ε={e:g}") for e in (0.5, 0.75, 1.0)]
                     + [Line2D([], [], marker="D", ms=3.0, lw=0, color=INK2, mfc=SURF, label="large@3e-4")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.0, fontsize=5.5, columnspacing=0.8)
    note("c", "per_seed_delta_positive", f"{n_pos}/{n_tot}", SRC_JSON)
    note("c", "cells_shown", {"n_cell_x_eps": len(cells_shown),
                              "mean_only_not_plotted": "base_n40_lr0.001_d500000 @ eps 0.75"}, SRC_JSON)


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--draft", action="store_true",
                   help="stamp a DRAFT <date> watermark (OFF by default: submission renders carry none)")
    p.add_argument("--no-draft", action="store_true", help=argparse.SUPPRESS)  # accepted, no-op; back-compat
    p.add_argument("--out", default=str(OUT), help="output directory (default paper/figs)")
    a = p.parse_args()
    draft = bool(a.draft) and not a.no_draft
    OUT = Path(a.out)
    doc = json.load(open(CC))
    if not doc["status"].startswith("REPORTED"):
        raise SystemExit(f"REFUSING: unexpected status {doc['status']!r}")
    print(f"[F6] {doc['status']}  prespec {doc['prespec_sha256'][:8]}  decision {doc['decision']}")

    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw={"wspace": 0.42, "width_ratios": [1.30, 1.05, 0.88]})
    panel_label(axs[0], "a"); panel_a(axs[0], doc)
    panel_label(axs[1], "b"); panel_b(axs[1], doc)
    panel_label(axs[2], "c"); panel_c(axs[2], doc)

    fig.text(0.5, -0.255,
             "REPORTED, NEVER GATED — the Amendment G verdict is untouched.  The decision rule was fixed before the run\n"
             f"(PRESPEC {doc['prespec']}, sha256 {doc['prespec_sha256'][:8]}); decision cell {CELL_LABEL[DECISION_CELL]}, readout A at all three ε.\n"
             f"Bootstrap: n_boot={doc['n_boot']}, seed {doc['boot_seed']}, {doc['n_nearest']} nearest partitions per family, medians recomputed inside every resample,\n"
             "contexts resampled; seed enters only as the n=3 sign check in (c).",
             ha="center", va="top", fontsize=5.0, color=INK2, linespacing=1.5)

    note("meta", "status", doc["status"], SRC_JSON)
    note("meta", "prespec_sha256", doc["prespec_sha256"], SRC_JSON)
    note("meta", "decision_per_eps", doc["decision_per_eps"], SRC_JSON)
    note("meta", "bootstrap", {"n_boot": doc["n_boot"], "boot_seed": doc["boot_seed"],
                               "n_nearest": doc["n_nearest"], "G_min": doc["G_min"]}, SRC_JSON)
    note("meta", "blind_rederivation_eps0p5", {
        "expected": 0.46969190283755774, "independent": 0.46969190283755735, "verdict": "CONFIRMED",
        "tol": "rel:0.02"}, ".claude/redteam/compctrl-delta-eps05.json (CLAIM_LEDGER RT-2 / CC-1)")
    note("meta", "joint_b_prime_caveat", {
        str(e): {"b_prime": rec(doc, DECISION_CELL, e)["joint"]["b_prime_atom"],
                 "se": rec(doc, DECISION_CELL, e)["joint"]["se_b_prime"]} for e in (0.5, 0.75, 1.0)},
        SRC_JSON + " — CLAIM_LEDGER CC-5: the primary rule governs; the 'equivalently b′ within ±0.05' clause fails at ε=0.5")

    if draft:
        fig.text(0.005, 0.005, f"DRAFT {time.strftime('%Y-%m-%d')} · {SRC_JSON}", fontsize=4.5,
                 color=MUTED, ha="left", va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F6_component_control.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F6_component_control.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  wrote paper/figs/F6_component_control.{pdf,png}")

    # merge into the shared ledger of figure numbers (never clobber another figure's block).
    # Was inline here; now scripts/figure_numbers_io.py, shared with figures_iclr.py, which
    # used to overwrite this file wholesale.  The old `draft` field was sticky-True
    # (`doc.get("draft", True) or draft`) and could never return to False; it is now
    # per-figure with a top-level any().
    doc_n = merge_figure_numbers(OUT / "figure_numbers.json", {FIG: NUMBERS[FIG]}, draft)
    print(f"  merged F6 into {OUT / 'figure_numbers.json'} (figures now: {sorted(doc_n['figures'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
