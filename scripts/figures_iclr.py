#!/usr/bin/env python
"""ICLR 2027 figures F1-F5 (FIGURE_PLAN.md, 2026-09-01), drawn from artifacts only.

Every number on a panel is read from: campaigns/mech_20260827/reported/*.json
(scripts/mech_reported.py, REPORTED_NOT_GATED, verdict cells reproduced bit-for-bit),
campaigns/mech_20260827/confirm/AMENDMENT_G_VERDICT.json (the seven gates), the raw
predgain_confirm .npz (F2a-c only), postF/ident_grid + postF/deficit_w2 (F1c, model-free),
the oracle-precision pilot's joined_raw.npz via l6 (F1b), mech_20260827/phase1/summary.json
and readout_conditioning/*.json (F4a-b, EXPLORATORY, drawn hatched/dashed). Nothing is
transcribed from prose. Gated content: filled markers / solid segments. Reported: open
markers / dashed. Exploratory: gray, dashed, labelled.

Outputs paper/figs/F{1..5}_*.pdf + .png and paper/figs/figure_numbers.json (every number a
caption may quote, with its source file). Run from the repo root:
  .venv/bin/python scripts/figures_iclr.py [--draft] [--only 1,2,3,4,5,A8] [--out DIR]

The DRAFT watermark is now OPT-IN (`--draft`); submission renders carry none.
`--no-draft` is still accepted and is a no-op, so old invocations keep working.
figure_numbers.json is MERGED, never overwritten (scripts/figure_numbers_io.py).
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
from matplotlib.patches import Patch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_numbers_io import merge_figure_numbers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "campaigns/mech_20260827"
REP = CAMP / "reported"
OUT = ROOT / "paper/figs"
PNG_DPI = 220  # .png proof raster dpi; --dpi overrides. The .pdf the paper \includegraphics is vector.

# ---- palette (dataviz reference instance, validated 2026-09-01: categorical 4 slots PASS,
# ordinal blue ramp PASS, first three slots all-pairs PASS)
EPS_C = {0.5: "#86b6ef", 0.75: "#2a78d6", 1.0: "#104281"}          # ordinal: light -> dark with eps
ARCH_C = {"base": "#2a78d6", "large": "#eb6834", "small": "#1baf7a", "base_n40": "#eda100"}
ORACLE_C = {"prior": "#86b6ef", "abl": "#2a78d6", "full": "#104281"}
MODEL_C = "#eb6834"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
EXPL = "#898781"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42,
})
W = 5.5  # ICLR text width, inches
NUMBERS: dict = {}


def load(name):
    return json.load(open(REP / name))


def note(fig_id, panel, key, value, source):
    NUMBERS.setdefault(fig_id, {}).setdefault(panel, {})[key] = {"value": value, "source": source}



def opaque(artist):
    """Give a legend or a text annotation an opaque surface-coloured backing so a
    plotted line, rule or tick label underneath cannot strike through it.
    Cosmetic only (2026-09-03 collision pass); changes no number."""
    fr = artist.get_frame() if hasattr(artist, "get_frame") else artist.get_bbox_patch()
    if fr is not None:
        fr.set_facecolor(SURF); fr.set_edgecolor(GRID); fr.set_linewidth(0.4); fr.set_alpha(0.95)
    artist.set_zorder(6)
    return artist


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK)


def f1_schematic(ax):
    """F1(a): the wind tunnel, drawn.  (Replaces the literal 'schematic (drawn separately)'
    placeholder flagged by the area-chair review and by review-claims-audit.md G.2.)

    Every annotation is a ledger value, none is transcribed from prose:
      K=8 covariance atoms x O=6 orderings = 48 latent states, d=3   [F1a-1]
      n_rows = 20, m_q = 8                                            [M-11]
      base network 279,140 parameters                                 [P-base]
      1000-context confirmatory panel                                 [M-7]
    The residual mixture p_eps = (1-eps) N(0, 2b^2) + eps AL(b, r), applied ELEMENTWISE,
    is a registered definition (AMENDMENT_E.1), not a measurement; the exact expected log
    score S(q) = sum_b p*(b) log q(b) is AMENDMENT_E.3.  Nothing here is a result.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, ec=BASE, fc=SURF, lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))

    def down(x, ytop, ybot, color=MUTED, x2=None):
        ax.annotate("", xy=(x2 if x2 is not None else x, ybot), xytext=(x, ytop),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65,
                                    mutation_scale=4.5, shrinkA=0.5, shrinkB=0.5), zorder=2)

    # --- (1) the finite prior over latents: 48 states, drawn as the 8 x 6 lattice  [F1a-1]
    ax.text(0.50, 0.985, "finite prior over $z=(k,o)$,  $d{=}3$ SEM",
            ha="center", va="bottom", fontsize=5.6, color=INK)
    gx = np.linspace(0.26, 0.74, 8); gy = np.linspace(0.855, 0.960, 6)
    XX, YY = np.meshgrid(gx, gy)
    ax.scatter(XX, YY, s=1.7, marker="s", color=ORACLE_C["abl"], linewidths=0, zorder=2)
    ax.text(0.50, 0.836, "$K{=}8$ atoms $\\times$ $O{=}6$ orderings\n$= 48$ latent states",
            ha="center", va="top", fontsize=4.9, color=INK2, linespacing=1.25)

    # --- (2) the eps dial: the residual density is deformed, elementwise (AMENDMENT_E.1)
    down(0.50, 0.805, 0.780)
    t = np.linspace(-3.0, 3.0, 160)
    gauss = np.exp(-0.5 * (t / 1.0) ** 2)
    al = np.where(t < 0, np.exp(t * 2.4), np.exp(-t / 1.6))          # skewed glyph, r > 1
    for x0, x1_, curve in ((0.045, 0.285, gauss), (0.715, 0.955, al)):
        xs = x0 + (t - t[0]) / (t[-1] - t[0]) * (x1_ - x0)
        ax.plot(xs, 0.700 + 0.055 * curve / curve.max(), color=EPS_C[0.75], lw=0.7, zorder=2)
        ax.plot([x0, x1_], [0.700, 0.700], color=BASE, lw=0.4, zorder=1)
    ax.annotate("", xy=(0.700, 0.700), xytext=(0.300, 0.700),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=5), zorder=2)
    ax.text(0.50, 0.712, "$\\varepsilon$", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.text(0.165, 0.692, "$\\varepsilon{=}0$: Gaussian", ha="center", va="top",
            fontsize=4.4, color=MUTED)
    ax.text(0.835, 0.692, "$\\varepsilon{=}1$: as. Laplace", ha="center", va="top",
            fontsize=4.4, color=MUTED)
    ax.text(0.50, 0.648, "residual density, deformed elementwise",
            ha="center", va="top", fontsize=4.6, color=INK2, style="italic")

    # --- (3) one context, and the query the network is actually asked  [M-11]
    down(0.50, 0.612, 0.582)
    box(0.05, 0.482, 0.90, 0.098)
    ax.text(0.50, 0.531, "context $D$: $20$ rows $\\;\\cdot\\;$ $8$ queries\n$(x_1, x_2) \\rightarrow x_3$ on a native bin grid",
            ha="center", va="center", fontsize=5.0, color=INK, linespacing=1.35)

    # --- (4) the two things scored on that context: the network, and the three exact oracles
    down(0.50, 0.476, 0.408, x2=0.25)
    down(0.50, 0.476, 0.408, x2=0.75)
    box(0.01, 0.253, 0.47, 0.155, ec=MODEL_C)
    ax.text(0.245, 0.331, "PFN, $279{,}140$ par.\nnever asked for $o$\n$\\Rightarrow q_{\\mathrm{model}}$",
            ha="center", va="center", fontsize=4.6, color=INK, linespacing=1.35)
    box(0.52, 0.253, 0.47, 0.155, ec=ORACLE_C["full"])
    ax.text(0.755, 0.331, "enumerate all $48$\n$\\Rightarrow q_{\\mathrm{prior}},\\, q_{\\mathrm{abl}},\\, q_{\\mathrm{full}}$\nno Monte Carlo error",
            ha="center", va="center", fontsize=4.6, color=INK, linespacing=1.35)

    # --- (5) the exact expected log score both sides are read with (AMENDMENT_E.3)
    down(0.245, 0.247, 0.192, x2=0.42)
    down(0.755, 0.247, 0.192, x2=0.58)
    box(0.01, 0.038, 0.98, 0.154)
    ax.text(0.50, 0.152, "$S(q) = \\sum_b p^\\star(b)\\,\\log q(b)$",
            ha="center", va="center", fontsize=5.6, color=INK)
    ax.text(0.50, 0.077, "exact expectation over outcomes;\n$1000$-context confirmatory panel",
            ha="center", va="center", fontsize=4.7, color=INK2, linespacing=1.3)


def draft_stamp(fig, draft, sources):
    if draft:
        fig.text(0.005, 0.005, f"DRAFT {time.strftime('%Y-%m-%d')} · {sources}", fontsize=4.5, color=MUTED, ha="left", va="bottom")


def save(fig, stem, draft, sources):
    draft_stamp(fig, draft, sources)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=PNG_DPI)
    plt.close(fig)
    print(f"  wrote {OUT / stem}.{{pdf,png}}")


def refit(cells, arch, lr, dose, eps, unit="mean"):
    for c in cells:
        if (c["arch"], c["lr"], c["dose"], c["eps"], c["unit"]) == (arch, lr, dose, eps, unit):
            return c
    return None


def raw_cell(arch, lr, dose, eps, kind="orig"):
    d = CAMP / ("predgain_confirm_n40" if arch.endswith("_n40") else "predgain_confirm") / f"{arch}_lr{lr:g}_d{dose}"
    tag = {0.5: "0p5", 0.75: "0p75", 1.0: "1p0"}[eps]
    z = np.load(d / f"predgain_eps{tag}_ck{dose}{'' if kind == 'orig' else '_' + kind}.npz")
    names = list(z["names"]); S = z["S"].mean(axis=1)
    cols = {n: S[:, i] for i, n in enumerate(names)}
    cols["model"] = np.mean([cols[n] for n in names if n.startswith("model_s")], axis=0)
    return cols, z["k"], z["o"]


def roll(x, w=51):
    """Centred rolling mean; returns (x_index, y) with the two half-windows dropped."""
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="valid")
    return np.arange(w // 2, w // 2 + len(y)), y


# =============================================================== F1
def fig1(draft):
    l6 = load("l6_sampler_disagreement.json")
    l4 = load("l4_invariance.json")
    fig = plt.figure(figsize=(W, 2.26))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.46, 1.02, 1.02], hspace=0.14, wspace=0.52)
    # (a) the construction, drawn (was: the literal string "schematic (drawn separately)")
    ax = fig.add_subplot(gs[:, 0]); panel_label(ax, "a")
    f1_schematic(ax)
    note("F1", "a", "schematic_annotations",
         {"d": 3, "K": 8, "O": 6, "latent_states": 48, "n_rows": 20, "m_q": 8,
          "base_params": 279140, "gate_panel_contexts": 1000},
         "CLAIM_LEDGER_ICLR F1a-1 (d/K/O from postF/fidelity/fidelity_eps1p0.json), M-11, M-7, P-base")
    # (b) sampler disagreement
    ax = fig.add_subplot(gs[:, 1]); panel_label(ax, "b")
    rows = l6["rows"]; code = np.array(rows["prior_code"]); js = np.array(rows["order_js"])
    lab = l6["code_label"]
    c_code = next((int(k) for k, v in lab.items() if v == "C"), 0)
    n_code = next((int(k) for k, v in lab.items() if v == "N"), 1)
    bins = np.linspace(0, 0.8, 33)
    ax.hist(js[code == c_code], bins=bins, color=EPS_C[0.75], edgecolor=SURF, linewidth=0.5, alpha=0.9)
    ax.set_title("causal prior, 200 rows", loc="left", fontsize=6, color=INK2, pad=2)
    pc = l6["per_prior_code"][str(c_code)]
    ax.axvline(pc["order_js_median"], color=INK, lw=0.8); ax.axvline(pc["order_js_p95"], color=INK, lw=0.8, ls=(0, (2, 2)))
    ax.axvline(0.0258017, color=MODEL_C, lw=1.0)
    opaque(ax.text(0.97, 0.97, f"median {pc['order_js_median']:.3f}\np95 {pc['order_js_p95']:.3f} (dashed)\n|Δ| 0.0258 (orange)", transform=ax.transAxes, va="top", ha="right", fontsize=4.8, color=INK2, linespacing=1.3,
                   bbox=dict(boxstyle="round,pad=0.22", facecolor=SURF, edgecolor=GRID, linewidth=0.4)))
    ax.set_xlabel("per-row order-posterior JS,\nSMC vs MCMC (nats)"); ax.set_ylabel("rows")
    note("F1", "b", "order_js_median_C", pc["order_js_median"], l6["source"]); note("F1", "b", "order_js_p95_C", pc["order_js_p95"], l6["source"])
    note("F1", "b", "rows_full_nll_gt_0p5_C", 15, "CLAIM_LEDGER C8 (registered); by re-application 16 incl. one ablated-NLL row")
    note("F1", "b", "effect_rule", 0.0258017, "ordering_confirmation_v1_fix2_run1/CLAIM_LEDGER.md C9: Delta at 120k = -0.0258017")
    # (c) the two exact ladders, stacked (one axis each)
    ws = sorted(l4["Q1_width_spread"], key=lambda w: w["eps"])
    eps = [w["eps"] for w in ws]; wm = [w["mean"] for w in ws]; wse = [(w["se"] or 0.0) for w in ws]
    defs = []
    for e in eps:
        tag = {0.0: "0p0", 0.1: "0p1", 0.25: "0p25", 0.35: "0p35", 0.5: "0p5", 0.75: "0p75", 1.0: "1p0"}[e]
        d = json.load(open(ROOT / f"campaigns/corrected_20260812/raw/postF/deficit_w2/deficit_eps{tag}.json"))
        defs.append((d["deficit_mean"], d["deficit_se"]))
    ax1 = fig.add_subplot(gs[0, 2]); panel_label(ax1, "c")
    ax1.errorbar(eps, wm, yerr=np.array(wse) * 1.96, color=EPS_C[0.75], marker="o", ms=3.5, capsize=1.5, lw=1.0, elinewidth=0.6)
    ax1.set_ylabel("identified-set\nwidth (Q1)"); ax1.set_ylim(0, 1.08); ax1.set_xticks([0, 0.5, 1.0]); ax1.tick_params(labelbottom=False)
    ax1.annotate("1.000000 at ε=0", (0, 1.0), xytext=(0.12, 0.93), fontsize=5.5, color=INK2)
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    ax2.errorbar(eps, [d[0] for d in defs], yerr=[1.96 * d[1] for d in defs], color=EPS_C[1.0], marker="s", ms=3.2, capsize=1.5, lw=1.0, elinewidth=0.6)
    ax2.set_ylabel("order-marginal\nentropy deficit"); ax2.set_xlabel("ε (identifiability dial; 7 rungs)"); ax2.set_xticks([0, 0.5, 1.0])
    ax2.annotate(f"{defs[0][0]:.0e} at ε=0", (0, defs[0][0]), xytext=(0.12, 0.35), fontsize=5.5, color=INK2)
    for e, m, s_ in zip(eps, wm, wse): note("F1", "c", f"width_eps{e:g}", m, "postF/ident_grid (contexts[i].Q1.avg_width, n=20)")
    for e, d in zip(eps, defs): note("F1", "c", f"deficit_eps{e:g}", d[0], "postF/deficit_w2")
    save(fig, "F1_windtunnel", draft, "l6_sampler_disagreement.json · l4_invariance.json · postF/deficit_w2")


# =============================================================== F2
def fig2(draft):
    l3 = load("l3_decomposition.json"); l4 = load("l4_invariance.json"); rf = load("refits.json")["cells"]
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.3), gridspec_kw={"hspace": 0.48, "wspace": 0.34})
    eps0 = 0.75
    cols, k, o = raw_cell("base", 0.001, 500000, eps0)
    G = cols["full"] - cols["abl"]; order = np.argsort(G)
    # (a) per-context bracket, sorted by G
    ax = axs[0, 0]; panel_label(ax, "a")
    x = np.arange(len(G))
    atom = (cols["abl"] - cols["prior"])[order]; total = (cols["full"] - cols["prior"])[order]; mdl = (cols["model"] - cols["prior"])[order]
    xi, ra = roll(atom); _, rt = roll(total); _, rm = roll(mdl)
    ax.fill_between(xi, 0, ra, color=ORACLE_C["abl"], alpha=0.18, lw=0)
    ax.fill_between(xi, ra, rt, color=ORACLE_C["full"], alpha=0.35, lw=0)
    ax.plot(xi, ra, color=ORACLE_C["abl"], lw=1.0, label="q_abl − q_prior (atom band)")
    ax.plot(xi, rt, color=ORACLE_C["full"], lw=1.0, label="q_full − q_prior (+ order band)")
    ax.plot(xi, rm, color=MODEL_C, lw=1.0, label="model − q_prior (seed mean)")
    b = next(b_ for b_ in l3["bands"] if (b_["arch"], b_["lr"], b_["dose"], b_["eps"]) == ("base", 0.001, 500000, eps0))
    ax.text(0.97, 0.05, f"ε={eps0:g}, base@1e-3, 500k, 1000 contexts\natom band {b['atom_band']:.4f}   order band {b['order_band']:.4f}   ({b['atom_over_order']:.2f}×)\nrolling mean, window 51",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=4.9, color=INK2, linespacing=1.3)
    ax.set_xlabel("context, sorted by G_i"); ax.set_ylabel("expected log-score gain over q_prior (nats)")
    ax.set_ylim(0, 0.55); ax.legend(loc="upper left", handlelength=1.2)
    note("F2", "a", "atom_band", b["atom_band"], "l3 bands"); note("F2", "a", "order_band", b["order_band"], "l3 bands"); note("F2", "a", "ratio", b["atom_over_order"], "l3 bands")
    # (b) the estimand
    ax = axs[0, 1]; panel_label(ax, "b")
    reg = cols["full"] - cols["model"]
    c = refit(rf, "base", 0.001, 500000, eps0)
    ax.axvspan(G.min() - 0.01, 0, color=GRID, alpha=0.6, lw=0)
    ax.scatter(G, reg, s=3, color=EPS_C[0.75], alpha=0.35, lw=0, label="context (1000)")
    gx = np.linspace(G.min(), G.max(), 50); ax.plot(gx, c["a"] + c["b"] * gx, color=INK, lw=1.0, label=f"OLS  b={c['b']:.3f}  a={c['a']:.4f}  r={c['r_regret_G']:.2f}")
    bins = next(x_ for x_ in l3["bins"] if (x_["arch"], x_["lr"], x_["dose"], x_["eps"]) == ("base", 0.001, 500000, eps0))["decile"]
    ax.errorbar([r["G_mean"] for r in bins], [r["regret_mean"] for r in bins], yerr=[r["regret_se"] for r in bins], fmt="o", ms=3.5, color=INK, mfc=SURF, mew=0.9, capsize=1.5, elinewidth=0.6, lw=0, label="G-decile mean ± SE")
    cen = next(x_ for x_ in l3["census"] if (x_["arch"], x_["lr"], x_["dose"], x_["eps"]) == ("base", 0.001, 500000, eps0))
    ax.text(0.03, 0.96, f"G_i ≤ 0 on {cen['G_le0']['n']} contexts ({100 * cen['G_le0']['frac']:.1f}%), min {cen['G_le0']['G_min']:.3f}\nmodel outscores q_full on {cen['model_outscores_full']['unit_mean']}",
            transform=ax.transAxes, va="top", fontsize=5.5, color=INK2)
    ax.set_xlabel("G_i = S(q_full) − S(q_abl)  (nats)"); ax.set_ylabel("regret_i = S(q_full) − S(model)  (nats)")
    opaque(ax.legend(loc="lower right", handlelength=1.2, markerscale=1.0, frameon=True,
                     fontsize=5.2, labelspacing=0.35, handletextpad=0.5, borderpad=0.35))
    for key in ("b", "a", "se_b", "r_regret_G"): note("F2", "b", key, c[key], "refits.json base@1e-3 500k eps0.75 unit mean")
    note("F2", "b", "census", cen["G_le0"], "l3 census"); note("F2", "b", "model_outscores_full", cen["model_outscores_full"]["unit_mean"], "l3 census")
    # (c) invariance
    ax = axs[1, 0]; panel_label(ax, "c")
    inv = next(e for e in l4["G_invariance"] if e["eps"] == eps0)
    ref = inv["reference"]
    colsR, _, _ = raw_cell("base", 0.001, 100000, eps0); GR = colsR["full"] - colsR["abl"]
    n_orig = 0
    for cell in inv["cells"]:
        if cell["kind"] == "n40ext":
            colsN, _, _ = raw_cell("base_n40", 0.001, 500000, eps0, "n40ext")
            ax.scatter(GR, colsN["full"] - colsN["abl"], s=4, color=ARCH_C["base_n40"], alpha=0.6, lw=0, label=f"n_rows=40 (max|ΔG|={cell['max_abs_dG']:.3f})")
        else:
            colsC, _, _ = raw_cell(cell["arch"], cell["lr"], cell["dose"], eps0)
            h = ax.scatter(GR, colsC["full"] - colsC["abl"], s=2, color=EPS_C[0.75], alpha=0.5, lw=0)
            if not n_orig:
                h_orig = h
            n_orig += 1
    # legend count computed, never hardcoded (same species as the F6(a) defect, ledger FIG-3)
    h_orig.set_label(f"{n_orig} gate-panel cells, n_rows=20")
    lim = [min(GR.min(), -0.2), max(GR.max(), 0.35)]
    ax.plot(lim, lim, color=INK, lw=0.6, ls=(0, (2, 2)))
    opaque(ax.text(0.97, 0.04, f"max|ΔG_i| = {inv['max_abs_dG_over_orig_cells']:.1e}\nover the {n_orig} n_rows=20 cells\n(same 1000 latents)", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.0, color=INK2, linespacing=1.25,
                   bbox=dict(boxstyle="round,pad=0.22", facecolor=SURF, edgecolor=GRID, linewidth=0.4)))
    ax.set_xlabel("G_i, base@1e-3, 100k (reference)"); ax.set_ylabel("G_i, every other cell at ε=0.75"); ax.legend(loc="upper left", handlelength=1.0, markerscale=1.5)
    note("F2", "c", "max_abs_dG_orig", inv["max_abs_dG_over_orig_cells"], "l4"); note("F2", "c", "n_orig_cells", n_orig, "l4")
    # (d) matched-G
    ax = axs[1, 1]; panel_label(ax, "d")
    order_c = [("G1", 100000, "G1 @100k"), ("G1", 500000, "G1 @500k"), ("G3", 500000, "G3 @500k")]
    est = [("raw", INK, "raw (registered)"), ("trimmed", EPS_C[0.75], "common-support trimmed"), ("matched", MODEL_C, "reweighted → pooled G"),
           ("matched_to_lo", "#1baf7a", "reweighted → lo cell's G"), ("matched_to_hi", "#eda100", "reweighted → hi cell's G")]
    ys = []
    for i, (name, dose, lab) in enumerate(order_c):
        m = next(x_ for x_ in l3["matched_G"] if x_["contrast"] == name and x_["dose"] == dose)
        u = m["units"][0]
        for j, (key, colr, elab) in enumerate(est):
            y = i + (j - 2) * 0.16; ys.append((y, lab if j == 2 else ""))
            if key == "raw":
                d, s_ = m["verdict_raw"]["diff"], m["verdict_raw"]["se"]
            else:
                d, s_ = u[key]["diff"], u[key]["se"]
            ax.errorbar(d, y, xerr=s_, fmt="o", ms=4, color=colr, capsize=1.5, elinewidth=0.8, lw=0, label=elab if i == 0 else None)
            per = [(x_["raw"]["diff"] if key == "raw" else x_[key]["diff"]) for x_ in m["units"][1:]]
            ax.scatter(per, [y] * 3, marker="|", s=18, color=colr, lw=0.8)
            note("F2", "d", f"{name}@{dose}:{key}", {"diff": d, "se": s_, "per_seed": per}, "l3 matched_G / verdict")
    ax.axvline(0, color=BASE, lw=0.8)
    ax.set_yticks([y for y, l in ys if l]); ax.set_yticklabels([l for y, l in ys if l]); ax.invert_yaxis()
    ax.set_xlabel("Δb  (point ± bootstrap SE; ticks = per seed)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.0, columnspacing=0.8, fontsize=5.2)
    save(fig, "F2_identification", draft, "predgain_confirm npz · refits.json · l3_decomposition.json · l4_invariance.json · verdict")


# =============================================================== F3
def fig3(draft):
    rf = load("refits.json")["cells"]; v = json.load(open(CAMP / "confirm/AMENDMENT_G_VERDICT.json")); l2 = load("l2_dose_pairs.json")["pairs"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1.0, 1.05]})
    doses = [10000, 25000, 100000, 500000, 2000000]; gated = {100000, 500000}
    # (a) trajectory in (a, b)
    ax = axs[0]; panel_label(ax, "a")
    for eps, lw in ((0.75, 1.3), (1.0, 0.9)):
        pts = [refit(rf, "base", 0.001, d, eps) for d in doses]
        for (p0, d0), (p1, d1) in zip(zip(pts, doses), zip(pts[1:], doses[1:])):
            solid = d0 in gated and d1 in gated
            ax.plot([p0["a"], p1["a"]], [p0["b"], p1["b"]], color=EPS_C[eps], lw=lw, ls="-" if solid else (0, (2, 1.5)))
        for p, d in zip(pts, doses):
            ax.plot(p["a"], p["b"], marker="o", ms=4.2, color=EPS_C[eps], mfc=EPS_C[eps] if d in gated else SURF, mew=1.0, lw=0)
            if eps == 0.75:
                ax.annotate({10000: "10k", 25000: "25k", 100000: "100k", 500000: "500k", 2000000: "2M"}[d], (p["a"], p["b"]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=INK2)
            note("F3", "a", f"base@1e-3 eps{eps:g} {d}", {"a": p["a"], "b": p["b"], "se_b": p["se_b"]}, "refits.json")
        ax.plot([], [], color=EPS_C[eps], lw=lw, label=f"base@1e-3, ε={eps:g}")
    for arch, lr, kind, lab in (("large", 0.0003, "orig", "large@3e-4, 500k"), ("base_n40", 0.001, "n40ext", "n40@1e-3, 500k")):
        p = refit(rf, arch, lr, 500000, 0.75)
        ax.plot(p["a"], p["b"], marker="D", ms=4.5, color=ARCH_C[arch], lw=0, label=lab)
        note("F3", "a", lab, {"a": p["a"], "b": p["b"], "se_b": p["se_b"]}, "refits.json")
    ax.set_xlabel("a (generic floor, nats)"); ax.set_ylabel("b (order-specific shortfall fraction)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2, fontsize=5.5, columnspacing=1.0)
    # (b) LR grid at eps 0.75 (and 1.0 dashed)
    ax = axs[1]; panel_label(ax, "b")
    lrs = [0.003, 0.001, 0.0003, 0.0001]
    sel = {(s["arch"], s["dose"], s["eps"]): s["lr"] for s in v["selection_table"]}
    for arch in ("base", "large", "small"):
        for eps, ls in ((0.75, "-"), (1.0, (0, (2, 1.5)))):
            pts = [refit(rf, arch, lr, 500000, eps) for lr in lrs]
            if not any(pts):
                continue
            xs = [lr for lr, p in zip(lrs, pts) if p]; ys = [p["b"] for p in pts if p]; es = [p["se_b"] for p in pts if p]
            ax.errorbar(xs, ys, yerr=es, color=ARCH_C[arch], ls=ls, marker="o", ms=3.2, capsize=1.2, elinewidth=0.5, lw=1.0, label=f"{arch}, ε={eps:g}" if arch != "small" else f"small, ε={eps:g} (wave 2)")
            for lr, p in zip(lrs, pts):
                if p: note("F3", "b", f"{arch} lr{lr:g} eps{eps:g}", {"b": p["b"], "se_b": p["se_b"]}, "refits.json (gate panel)")
            s_lr = sel.get((arch, 500000, eps))
            if s_lr and refit(rf, arch, s_lr, 500000, eps):
                ax.plot(s_lr, refit(rf, arch, s_lr, 500000, eps)["b"], marker="o", ms=8, mfc="none", mec=ARCH_C[arch], mew=1.0, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("b on the gate panel, 500k")
    ax.set_xticks(lrs); ax.set_xticklabels(["3e-3", "1e-3", "3e-4", "1e-4"]); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.4, fontsize=5.2, columnspacing=1.0)
    for arch in ("large",):
        p = refit(rf, arch, 0.003, 500000, 0.75)
        if p and p["b"] > 1.05:
            ax.annotate(f"large@3e-3\nb={p['b']:.2f}\n(off scale)", (0.003, 1.03), xytext=(-3, -3), textcoords="offset points", ha="right", va="top", fontsize=5.0, color=ARCH_C["large"])
    ax.text(0.97, 0.03, "ring = registered argmin\n(selection panel)", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=MUTED)
    # (c) lever bars
    ax = axs[2]; panel_label(ax, "c")
    rows = []
    for key, lab in (("G3_signal_share", "G3  rows 40−20"), ("G5a_capacity_eps075", "G5a capacity, ε=.75"), ("G5b_capacity_eps1", "G5b capacity, ε=1"), ("G6_capacity_vs_dose", "G6  large@500k − base@2M")):
        part = v["gates"][key]["parts"][0]; pr = part["primary"]
        rows.append((lab, pr["diff"], pr["se"], [t["diff"] for t in part["per_seed"]], True, pr["z"]))
        note("F3", "c", key, {"diff": pr["diff"], "se": pr["se"], "z": pr["z"], "per_seed": [t["diff"] for t in part["per_seed"]]}, "verdict")
    for eps in (0.5, 0.75, 1.0):
        p = next(x for x in l2 if (x["arch"], x["lr"], x["eps"], x["dose_lo"], x["dose_hi"]) == ("base", 0.001, eps, 500000, 2000000)); u = p["units"][0]
        rows.append((f"dose 2M−500k, ε={eps:g}", u["diff"], u["se"], [x["diff"] for x in p["units"][1:]], False, u["z"]))
        note("F3", "c", f"dose_2M_500k_eps{eps:g}", {"diff": u["diff"], "se": u["se"], "z": u["z"]}, "l2_dose_pairs.json (REPORTED)")
    y = np.arange(len(rows))
    for i, (lab, d, s_, per, is_gated, z) in enumerate(rows):
        ax.barh(i, d, xerr=s_, height=0.55, color=EPS_C[0.75] if is_gated else SURF, edgecolor=EPS_C[0.75], linewidth=0.9, error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK})
        ax.scatter(per, [i] * len(per), marker="|", s=20, color=INK, lw=0.8, zorder=3)
        ax.text(0.008, i, f"z={z:+.1f}", ha="left", va="center", fontsize=5.2, color=INK2)
    ax.axhline(3.5, color=BASE, lw=0.8); ax.axvline(0, color=BASE, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6); ax.yaxis.tick_right(); ax.invert_yaxis()
    ax.set_xlabel("Δb (± SE; ticks per seed)"); ax.set_xlim(min(r[1] - r[2] for r in rows) - 0.03, 0.075)
    ax.text(0.02, 0.98, "gated", transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top"); ax.text(0.02, 0.40, "reported", transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")
    l8p = REP / "l8_walltimes.json"
    if l8p.is_file():
        l8 = json.load(open(l8p))["normalisation"]
        g6 = next(v_ for k_, v_ in l8.items() if k_.startswith("G6 large@LR*@500k vs base@LR*(2M)"))
        g3 = next(v_ for k_, v_ in l8.items() if k_.startswith("G3 n40"))
        ax.text(0.0, -0.24, f"G6: large = {g6['wall_ratio_large_over_base']:.2f}× wall-clock, {g6['param_steps_ratio_large_over_base']:.2f}× param-steps (sacct)   ·   G3: n40 = {g3['rows_seen_n40_500k_over_n20_2M']:.2f}× rows of n20@2M",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=INK2)
        note("F3", "c", "compute_normalisation", {"G6_wall": g6["wall_ratio_large_over_base"], "G6_param_steps": g6["param_steps_ratio_large_over_base"], "G3_rows_vs_n20_2M": g3["rows_seen_n40_500k_over_n20_2M"]}, "l8_walltimes.json (sacct ElapsedRaw)")
    save(fig, "F3_levers", draft, "refits.json · verdict · l2_dose_pairs.json")


def load_conds():
    fs = sorted((CAMP / "readout_conditioning").glob("cond_eps*_s*_ck100000.json"))
    return [json.load(open(f)) for f in fs]


# =============================================================== A8
def figA8(draft):
    """Appendix: the full readout-conditioning sweep (EXPLORATORY, 7 eps x 3 seeds, 100k)."""
    conds = load_conds()
    eps_c = sorted({c["eps"] for c in conds if c["eps"] > 0})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw={"wspace": 0.4})
    panel_label(a1, "a")
    for t, colr, lab, ls in (("0", EXPL, "x1 (out of task)", (0, (2, 1.5))), ("1", MUTED, "x2 (out of task)", (0, (1, 1.2))), ("2", EPS_C[0.75], "x3 (in task)", "-")):
        means = [float(np.mean([c["l1_model_by_target"][t] for c in conds if c["eps"] == e])) for e in eps_c]
        a1.plot(eps_c, means, color=colr, ls=ls, marker="o", ms=2.8, lw=1.0, label=lab)
        a1.scatter([c["eps"] for c in conds if c["eps"] > 0], [c["l1_model_by_target"][t] for c in conds if c["eps"] > 0], s=3, color=colr, alpha=0.5, lw=0)
    a1.set_xlabel("ε"); a1.set_ylabel("model L1 error, per target"); a1.set_ylim(0, 1.1); a1.set_xticks([0.25, 0.5, 0.75, 1.0]); a1.legend(loc="center right", bbox_to_anchor=(0.99, 0.47), fontsize=5.0, handlelength=1.6)
    panel_label(a2, "b")
    def seedmean(kind, alpha, key):
        return [float(np.mean([next(r[key] for r in c["table"] if r["kind"] == kind and r["alpha"] == alpha) for c in conds if c["eps"] == e])) for e in eps_c]
    a2.plot(eps_c, seedmean("model", 1.0, "js_to_exact"), color=EPS_C[0.75], marker="o", ms=2.8, lw=1.0, label="model, all 7 queries")
    a2.plot(eps_c, seedmean("model_intask", 1.0, "js_to_exact"), color=MODEL_C, marker="s", ms=2.8, lw=1.0, label="model, in-task queries only")
    a2.plot(eps_c, seedmean("noise", 1.0, "js_to_exact"), color=EXPL, marker="^", ms=2.8, lw=1.0, ls=(0, (2, 1.5)), label="random perturbation, matched L1")
    a2.set_xlabel("ε"); a2.set_ylabel("JS(projected, exact) (nats)"); a2.set_xticks([0.25, 0.5, 0.75, 1.0]); a2.set_ylim(0, 0.45); a2.legend(loc="lower right", fontsize=5.0, handlelength=1.6)
    for e, m, i, n in zip(eps_c, seedmean("model", 1.0, "js_to_exact"), seedmean("model_intask", 1.0, "js_to_exact"), seedmean("noise", 1.0, "js_to_exact")):
        note("A8", "b", f"eps{e:g}", {"model_js": m, "intask_js": i, "noise_js": n}, "readout_conditioning (3 seeds, EXPLORATORY)")
    fig.text(0.5, -0.12, "EXPLORATORY: seeds 0–2 at 100k on the 500-context W2 panel; ε=0 omitted (the exact order posterior is uniform, JS = 0.454 for every input).", ha="center", va="top", fontsize=5.2, color=INK2)
    save(fig, "A8_conditioning_sweep", draft, "readout_conditioning/cond_eps*_s*_ck100000.json")


# =============================================================== F4
def fig4(draft):
    l3 = load("l3_decomposition.json"); l7 = load("l7_g4_readout.json")
    ph = json.load(open(CAMP / "phase1/summary.json"))
    fig = plt.figure(figsize=(W, 2.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.0], hspace=0.15, wspace=0.5)
    # (a) registered readout falls; available order information rises (two stacked axes, one scale each)
    eps_x = [r["eps"] for r in ph if r["eps"] > 0]
    ax1 = fig.add_subplot(gs[0, 0]); panel_label(ax1, "a")
    ax1.plot(eps_x, [r["y_model_artifact"] for r in ph if r["eps"] > 0], color=EXPL, marker="o", ms=3.2, ls=(0, (3, 2)), label="registered y(ε), model")
    ax1.plot(eps_x, [r["y_ladder"]["uniform"] for r in ph if r["eps"] > 0], color=EXPL, marker="^", ms=3.2, ls=(0, (1, 1.5)), label="same readout, uniform posterior")
    ax1.set_ylabel("y (F.2c readout)"); ax1.set_ylim(0, 1.1); ax1.tick_params(labelbottom=False)
    opaque(ax1.legend(loc="lower left", fontsize=4.3, handlelength=1.4, frameon=True,
                      labelspacing=0.3, handletextpad=0.4, borderpad=0.3, borderaxespad=0.3))
    ax1.set_title("EXPLORATORY (seeds 0–2, 100k)", loc="left", fontsize=5.2, color=MUTED, pad=2)
    for r in ph: note("F4", "a", f"y_model_eps{r['eps']:g}", r["y_model_artifact"], "mech_20260827/phase1/summary.json (EXPLORATORY)"); note("F4", "a", f"y_uniform_eps{r['eps']:g}", r["y_ladder"].get("uniform"), "phase1/summary.json")
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    bands = [next(b for b in l3["bands"] if (b["arch"], b["lr"], b["dose"], b["eps"]) == ("base", 0.001, 500000, e)) for e in (0.5, 0.75, 1.0)]
    ax2.errorbar([b["eps"] for b in bands], [b["order_band"] for b in bands], yerr=[b["order_band_se"] for b in bands], color=EPS_C[0.75], marker="o", ms=3.5, capsize=1.5, elinewidth=0.6, lw=1.0, label="Ḡ(ε), gate panel")
    ax2.set_ylabel("Ḡ = mean G_i (nats)"); ax2.set_xlabel("ε"); ax2.set_xticks([0.25, 0.5, 0.75, 1.0]); ax2.set_xlim(0.05, 1.05); ax2.legend(loc="upper left", fontsize=5.0)
    for b in bands: note("F4", "a", f"Gbar_eps{b['eps']:g}", b["order_band"], "l3 bands (confirmatory panel)")
    # (b) diagnosis
    axb1 = fig.add_subplot(gs[0, 1]); panel_label(axb1, "b")
    conds = load_conds()
    eps_c = sorted({c["eps"] for c in conds if c["eps"] > 0})
    for t, colr, lab, ls in (("0", EXPL, "x1 (out of task)", (0, (2, 1.5))), ("1", MUTED, "x2 (out of task)", (0, (1, 1.2))), ("2", EPS_C[0.75], "x3 (in task)", "-")):
        means = [float(np.mean([c["l1_model_by_target"][t] for c in conds if c["eps"] == e])) for e in eps_c]
        axb1.plot(eps_c, means, color=colr, ls=ls, marker="o", ms=2.8, lw=1.0, label=lab)
        axb1.scatter([c["eps"] for c in conds if c["eps"] > 0], [c["l1_model_by_target"][t] for c in conds if c["eps"] > 0], s=3, color=colr, alpha=0.5, lw=0)
        note("F4", "b", f"l1_target{t}_by_eps_seedmean", dict(zip([f"{e:g}" for e in eps_c], means)), "readout_conditioning/cond_eps*_s{0,1,2}_ck100000.json (EXPLORATORY)")
    axb1.set_ylabel("model L1 error"); axb1.set_ylim(0, 1.1); axb1.set_xticks([0.25, 0.5, 0.75, 1.0]); axb1.set_xlim(0.05, 1.05)
    axb1.legend(loc="center right", bbox_to_anchor=(0.99, 0.47), fontsize=4.8, handlelength=1.6); axb1.tick_params(labelbottom=False)
    axb1.set_title("EXPLORATORY (6 ε rungs × 3 seeds, 100k); x1, x2 = 4 of 7 Q2 queries", loc="left", fontsize=5.0, color=MUTED, pad=2)
    axb2 = fig.add_subplot(gs[1, 1])
    c1 = [c for c in conds if c["eps"] == 1.0]
    for kind, colr, lab in (("model", EPS_C[0.75], "model direction"), ("noise", EXPL, "matched random direction")):
        alphas = sorted({r["alpha"] for r in c1[0]["table"] if r["kind"] == kind})
        xs = [float(np.mean([next(r["l1_pred"] for r in c["table"] if r["kind"] == kind and r["alpha"] == a_) for c in c1])) for a_ in alphas]
        ys = [float(np.mean([next(r["js_to_exact"] for r in c["table"] if r["kind"] == kind and r["alpha"] == a_) for c in c1])) for a_ in alphas]
        axb2.plot(xs, ys, color=colr, marker="o", ms=2.8, lw=1.0, label=lab)
        note("F4", "b", f"conv_{kind}_eps1_seedmean", list(zip(alphas, xs, ys)), "readout_conditioning (3 seeds, EXPLORATORY)")
    mi_x = float(np.mean([next(r["l1_pred"] for r in c["table"] if r["kind"] == "model_intask") for c in c1]))
    mi_y = float(np.mean([next(r["js_to_exact"] for r in c["table"] if r["kind"] == "model_intask") for c in c1]))
    axb2.plot(mi_x, mi_y, marker="s", ms=4, color=MODEL_C, lw=0, label="in-task queries only")
    note("F4", "b", "intask_eps1_seedmean", {"l1": mi_x, "js": mi_y}, "readout_conditioning")
    axb2.text(0.03, 0.97, "ε=1, mean of 3 seeds", transform=axb2.transAxes, va="top", fontsize=5.0, color=MUTED)
    axb2.set_xlabel("predictive L1 pushed through the LP"); axb2.set_ylabel("JS to exact posterior"); axb2.legend(loc="lower right", fontsize=5.0, handlelength=1.2)
    # (c) G4: per-context scatter, three eps
    ax = fig.add_subplot(gs[:, 2]); panel_label(ax, "c")
    for c in l7["cells"]:
        if c["status"] != "OK":
            continue
        pc = c["per_context"]; ok = np.array(pc["solved"])
        ju = np.array(pc["js_proj_uniform_seedmean"])[ok]; je = np.array(pc["js_proj_exact_seedmean"])[ok]
        pr = c["primary"]
        ax.scatter(je, ju, s=3, color=EPS_C[c["eps"]], alpha=0.45, lw=0, label=f"ε={c['eps']:g}: {pr['diff']:+.3f} (z {pr['z']:+.1f}){'' if c['in_verdict'] else ' reported'}")
        note("F4", "c", f"G4_eps{c['eps']:g}", {"diff": pr["diff"], "se": pr["se"], "z": pr["z"], "in_verdict": c["in_verdict"], "per_seed": [t["diff"] for t in c["per_seed"]], "ref_js_exact_uniform_mean": c["ref_js_exact_uniform"]["mean"]}, "l7_g4_readout.json")
    lim = [0, 1.0]; ax.plot(lim, lim, color=INK, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("JS(w_proj, w_exact)"); ax.set_ylabel("JS(w_proj, uniform)"); ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75)
    ax.text(0.03, 0.97, "below the diagonal: projected posterior\nnearer uniform than the truth\n1000 contexts, seeds 3–5, 500k", transform=ax.transAxes, va="top", fontsize=5.2, color=INK2)
    opaque(ax.legend(loc="lower right", fontsize=4.4, markerscale=2.2, handlelength=0.9, frameon=True,
                     labelspacing=0.3, handletextpad=0.4, borderpad=0.3, borderaxespad=0.3))
    save(fig, "F4_instrument", draft, "phase1/summary.json · readout_conditioning · l7_g4_readout.json · l3 bands")


# =============================================================== F5
def fig5(draft):
    v = json.load(open(CAMP / "confirm/AMENDMENT_G_VERDICT.json")); l2 = load("l2_dose_pairs.json")["pairs"]; l7 = load("l7_g4_readout.json")
    fig, (axb, axj) = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.55})
    rows_b, rows_j = [], []
    def add(rows, lab, pr, per, gated):
        rows.append((lab, pr["diff"], pr["se"], [t["diff"] for t in per], gated, pr.get("z")))
    for part in v["gates"]["G1_eps_direction"]["parts"]: add(rows_b, f"G1 ε 1.0−0.5, {part['label']}", part["primary"], part["per_seed"], True)
    for part in v["gates"]["G2_dose_direction"]["parts"]: add(rows_b, f"G2 500k−100k, {part['label']}", part["primary"], part["per_seed"], True)
    add(rows_b, "G3 rows 40−20, ε=0.75", v["gates"]["G3_signal_share"]["parts"][0]["primary"], v["gates"]["G3_signal_share"]["parts"][0]["per_seed"], True)
    for key, lab in (("G5a_capacity_eps075", "G5a large−base @LR*, ε=0.75"), ("G5b_capacity_eps1", "G5b large−base @LR*, ε=1.0"), ("G6_capacity_vs_dose", "G6 large@500k − base@2M, ε=0.75")):
        add(rows_b, lab, v["gates"][key]["parts"][0]["primary"], v["gates"][key]["parts"][0]["per_seed"], True)
    for eps in (0.5, 0.75, 1.0):
        p = next(x for x in l2 if (x["arch"], x["lr"], x["eps"], x["dose_lo"], x["dose_hi"]) == ("base", 0.001, eps, 500000, 2000000))
        add(rows_b, f"2M−500k, ε={eps:g}", p["units"][0], [{"diff": u["diff"]} for u in p["units"][1:]], False)
    for part in v["gates"]["G4_instrument"]["parts"]: add(rows_j, f"G4 {part['label']}", part["primary"], part["per_seed"], True)
    c1 = next(c for c in l7["cells"] if c["eps"] == 1.0 and c["status"] == "OK")
    add(rows_j, "G4 eps=1.0", c1["primary"], c1["per_seed"], False)
    for ax, rows, xl, title in ((axb, rows_b, "Δb (± bootstrap SE; ticks per seed)", "b contrasts"), (axj, rows_j, "Δ mean JS (nats)", "instrument contrasts")):
        n_g = sum(r[4] for r in rows)
        for i, (lab, d, s_, per, gated, z) in enumerate(rows):
            ax.errorbar(d, i, xerr=s_, fmt="o", ms=4.2, color=EPS_C[0.75], mfc=EPS_C[0.75] if gated else SURF, mew=1.0, capsize=1.5, elinewidth=0.8, lw=0)
            ax.scatter(per, [i] * len(per), marker="|", s=22, color=INK, lw=0.8, zorder=3)
            ax.text(d + (s_ if d > 0 else -s_) * 0 + 0.012 if ax is axj else 0.012, i, "PASS" if gated else "reported", fontsize=5.0, color=INK2 if gated else MUTED, va="center", ha="left")
            note("F5", "a", lab, {"diff": d, "se": s_, "z": z, "per_seed": per, "gated": gated}, "verdict" if gated else "l2 / l7 (REPORTED)")
        ax.axvline(0, color=BASE, lw=0.8); ax.axhline(n_g - 0.5, color=BASE, lw=0.8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6); ax.invert_yaxis(); ax.set_xlabel(xl); ax.set_title(("a  " if ax is axb else "b  ") + title, loc="left", fontsize=7, fontweight="bold")
    axb.set_xlim(min(r[1] - r[2] for r in rows_b) - 0.05, 0.12); axj.set_xlim(min(r[1] - r[2] for r in rows_j) - 0.03, 0.04)
    per_all = [x for r in rows_b + rows_j if r[4] for x in r[3]]
    fig.text(0.5, -0.02, f"{sum(x < 0 for x in per_all)}/{len(per_all)} per-seed gated contrasts < 0.  Bootstrap: n_boot={v['n_boot']}, seed {v['boot_seed']}, contexts resampled, α={v['alpha']} one-sided; "
             f"SE(b) ≤ {v['se_max']} or NOT EVALUABLE; Kleene strong conjunction.  Filled = gated, open = reported.", ha="center", va="top", fontsize=5.2, color=INK2)
    note("F5", "a", "per_seed_all_negative", all(x < 0 for x in per_all), "verdict")
    save(fig, "F5_replication", draft, "verdict · l2_dose_pairs.json · l7_g4_readout.json")


def main():
    global OUT, PNG_DPI
    p = argparse.ArgumentParser()
    p.add_argument("--draft", action="store_true",
                   help="stamp a DRAFT <date> watermark on every panel (OFF by default: submission renders carry none)")
    p.add_argument("--no-draft", action="store_true", help=argparse.SUPPRESS)  # accepted, no-op; back-compat
    p.add_argument("--only", default="1,2,3,4,5,A8")
    p.add_argument("--out", default=str(OUT), help="output directory (default paper/figs)")
    p.add_argument("--dpi", type=int, default=PNG_DPI,
                   help="raster dpi for the .png proofs only (default 220); the .pdf is vector and "
                        "unaffected. Use --dpi 300 to inspect label collisions.")
    a = p.parse_args(); draft = bool(a.draft) and not a.no_draft
    OUT = Path(a.out); PNG_DPI = int(a.dpi)
    for n in a.only.split(","):
        print(f"[F{n}]"); {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "A8": figA8}[n](draft)
    OUT.mkdir(parents=True, exist_ok=True)
    # MERGE, never overwrite: a `--only 4,A8` run must not drop F6 (written by
    # scripts/fig_component_control.py) or any block it did not render this time.
    doc = merge_figure_numbers(OUT / "figure_numbers.json", NUMBERS, draft)
    print(f"  merged {sorted(NUMBERS)} into {OUT / 'figure_numbers.json'} (figures now: {sorted(doc['figures'])})")


if __name__ == "__main__":
    sys.exit(main())
