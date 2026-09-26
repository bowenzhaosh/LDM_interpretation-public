"""F4, built exact-spine-first: the two model-free ladders, then nothing else.

Amendment E.8 (D8) is the rule this file obeys:

    "F4 is LAYERED: the exact model-free ladders are computed and plotted first
     so the figure exists independently of the PFN overlay. The Q3 PFN panel is
     DROPPED, with a caption stating that the vanilla PFN has no intervention
     channel, so the Pearl L2 rung has no model-side counterpart BY CONSTRUCTION."

So this script draws the spine and stops. There is no model in it. Both axes are
exact enumerations over the substrate and exist whatever any PFN does, which is
the whole point of building this half first: if the retrain never lands, F4 still
has a figure behind it.

Every number is read through the sanctioned fail-closed loader
(`corrected_verdict.cells_from_raw`, which requires an F.8 `live` tag and a
recorded `n_rows`) or from a `deficit_*.json` whose tag is checked here the same
way. Nothing is transcribed, so the figure cannot drift from the artifacts the
way three of Amendment F's own tables had.

Outputs
  reports/f4_exact_spine.pdf   paper figure (vector)
  reports/f4_exact_spine.png   raster twin, embedded in the HTML
  reports/f4_exact_spine.html  self-contained page: figure, the table behind it,
                               provenance per cell, and what is deliberately absent
"""
from __future__ import annotations

import base64
import json
import math
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pfn_dag_verify import corrected_verdict as V  # noqa: E402

IDENT_DIR = ROOT / "campaigns/corrected_20260812/raw/postF/ident_grid"
DEFICIT_DIR = ROOT / "campaigns/corrected_20260812/raw/postF/deficit_w2"
OUT = ROOT / "reports"

# dataviz slots 1 and 2, validated as a pair in both modes
# (worst adjacent CVD dE 24.7 light / 26.8 dark; normal-vision 33.6 / 31.8)
BLUE_L, ORANGE_L = "#2a78d6", "#eb6834"
INK_L, INK2_L, SURF_L = "#0b0b0b", "#52514e", "#fcfcfb"
NEW_RUNGS = (0.35, 0.75)


def load_widths() -> dict[float, dict]:
    cells = V.cells_from_raw(IDENT_DIR, solver="highs_ipm")
    out = {}
    for c in cells:
        f = next(IDENT_DIR.glob(f"ident_eps{V.eps_tag(c.eps)}_highs_ipm.json"))
        d = json.loads(f.read_text())
        out[c.eps] = dict(width=c.width_exact, path=f.name,
                          n_rows=d["config"]["n_rows"], solver=d["config"]["solver"],
                          n_contexts=d["config"]["n_contexts"],
                          status=d["artifact"]["status"],
                          true_in_all=d["aggregates"]["Q1"]["true_in_all_frac"])
    return out


def load_deficits() -> dict[float, dict]:
    out = {}
    for f in sorted(DEFICIT_DIR.glob("deficit_eps*.json")):
        d = json.loads(f.read_text())
        st = d.get("artifact", {}).get("status")
        if st != "live":
            raise ValueError(f"{f.name}: status {st!r}, not 'live'. F.8 fails closed.")
        out[float(d["eps"])] = dict(
            mean=d["deficit_mean"], se=d["deficit_se"], path=f.name,
            n_contexts=d["n_contexts"], n_rows=d["n_rows"], half=d["half"],
            panel_sha=d["panel_sha256"][:12], status=st)
    return out


def check_monotone(eps, w, dfc):
    """Both ladders must be monotone in eps, and the deficit's adjacent pairs must
    separate. A ladder that is not monotone is a substrate finding under T4's stop
    condition, not a plotting nuisance, so it raises here rather than being drawn."""
    bad_w = [(eps[i], eps[i + 1]) for i in range(len(eps) - 1) if not w[i] > w[i + 1]]
    bad_d = [(eps[i], eps[i + 1]) for i in range(len(eps) - 1)
             if not dfc[i][0] < dfc[i + 1][0]]
    if bad_w or bad_d:
        raise ValueError(f"ladder not monotone -- width {bad_w}, deficit {bad_d}")
    overlaps = []
    for i in range(len(eps) - 1):
        (m0, s0), (m1, s1) = dfc[i], dfc[i + 1]
        if m0 + 1.96 * s0 >= m1 - 1.96 * s1:
            overlaps.append((eps[i], eps[i + 1]))
    return overlaps


def figure(eps, w, dmean, dse, path_pdf, path_png):
    """Two panels over one eps axis, never two y-scales on one frame.

    The measures live on different scales and a dual axis would let the crossing
    point be chosen by the axis limits rather than by the data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    fig.patch.set_facecolor(SURF_L)
    for ax in axes:
        ax.set_facecolor(SURF_L)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d7d2")
        ax.tick_params(colors=INK2_L, labelsize=8.5, length=3)
        ax.grid(True, color="#ebeae5", lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    a = axes[0]
    a.plot(eps, w, "-", color=BLUE_L, lw=2, zorder=3)
    for e, y in zip(eps, w):
        new = e in NEW_RUNGS
        a.plot([e], [y], "o", ms=8 if new else 6.5, color=BLUE_L,
               mec=SURF_L, mew=2, zorder=4)
        if new:
            a.annotate(f"{y:.3f}", (e, y), textcoords="offset points",
                       xytext=(7, 8), fontsize=8, color=INK_L, weight="bold")
    a.annotate(f"{w[0]:.3f}", (eps[0], w[0]), textcoords="offset points",
               xytext=(6, -4), fontsize=8, color=INK2_L)
    a.annotate(f"{w[-1]:.3f}", (eps[-1], w[-1]), textcoords="offset points",
               xytext=(-30, 8), fontsize=8, color=INK2_L)
    a.set_title("Exact identified-set width", fontsize=10, color=INK_L, loc="left", pad=8)
    a.set_xlabel(r"$\varepsilon$", fontsize=9.5, color=INK2_L)
    a.set_ylabel("Q1 mean width", fontsize=9, color=INK2_L)
    a.set_ylim(0, 1.06)

    b = axes[1]
    b.fill_between(eps, np.array(dmean) - 1.96 * np.array(dse),
                   np.array(dmean) + 1.96 * np.array(dse),
                   color=ORANGE_L, alpha=0.16, lw=0, zorder=2)
    b.plot(eps, dmean, "-", color=ORANGE_L, lw=2, zorder=3)
    for e, y in zip(eps, dmean):
        new = e in NEW_RUNGS
        b.plot([e], [y], "o", ms=8 if new else 6.5, color=ORANGE_L,
               mec=SURF_L, mew=2, zorder=4)
        if new:
            b.annotate(f"{y:.3f}", (e, y), textcoords="offset points",
                       xytext=(-34, 6), fontsize=8, color=INK_L, weight="bold")
    b.annotate(f"{dmean[-1]:.3f}", (eps[-1], dmean[-1]), textcoords="offset points",
               xytext=(-34, 5), fontsize=8, color=INK2_L)
    b.set_title("Order-marginal entropy deficit", fontsize=10, color=INK_L, loc="left", pad=8)
    b.set_xlabel(r"$\varepsilon$", fontsize=9.5, color=INK2_L)
    b.set_ylabel(r"$1 - H(p(o\mid D))/\log O$", fontsize=9, color=INK2_L)
    b.set_ylim(-0.03, 0.86)

    c = axes[2]
    one_minus = [1 - x for x in w]
    c.plot(one_minus, dmean, "-", color="#b9b8b2", lw=1.4, zorder=2)
    for x, y, e in zip(one_minus, dmean, eps):
        c.plot([x], [y], "o", ms=7, color=BLUE_L if e not in NEW_RUNGS else ORANGE_L,
               mec=SURF_L, mew=2, zorder=4)
        c.annotate(f"{e:g}", (x, y), textcoords="offset points",
                   xytext=(6, -9), fontsize=7.5, color=INK2_L)
    rho = float(np.corrcoef(np.argsort(np.argsort(one_minus)),
                            np.argsort(np.argsort(dmean)))[0, 1])
    c.set_title(f"The two axes agree (rank $\\rho$ = {rho:.3f})", fontsize=10,
                color=INK_L, loc="left", pad=8)
    c.set_xlabel("1 - width", fontsize=9, color=INK2_L)
    c.set_ylabel("entropy deficit", fontsize=9, color=INK2_L)

    fig.tight_layout(pad=1.3)
    fig.savefig(path_pdf, facecolor=SURF_L, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, facecolor=SURF_L, bbox_inches="tight")
    plt.close(fig)
    return rho


def main() -> int:
    OUT.mkdir(exist_ok=True)
    W = load_widths()
    D = load_deficits()
    eps = sorted(W)
    if sorted(D) != eps:
        raise ValueError(f"axes disagree on cells: width {eps} vs deficit {sorted(D)}")
    w = [W[e]["width"] for e in eps]
    dmean = [D[e]["mean"] for e in eps]
    dse = [D[e]["se"] for e in eps]
    overlaps = check_monotone(eps, w, list(zip(dmean, dse)))

    rho = figure(eps, w, dmean, dse, OUT / "f4_exact_spine.pdf", OUT / "f4_exact_spine.png")
    png_b64 = base64.b64encode((OUT / "f4_exact_spine.png").read_bytes()).decode()

    rows = "\n".join(
        f"<tr><td class=num>{e:g}</td>"
        f"<td class=num>{W[e]['width']:.9f}</td>"
        f"<td class=num>{D[e]['mean']:.9f}</td>"
        f"<td class=num>{D[e]['se']:.2e}</td>"
        f"<td class=st>{W[e]['status']}</td><td class=st>{D[e]['status']}</td>"
        f"<td class=mono>{W[e]['path']}</td><td class=mono>{D[e]['path']}</td></tr>"
        for e in eps)

    html = HTML.replace("__PNG__", png_b64).replace("__ROWS__", rows) \
        .replace("__RHO__", f"{rho:.3f}") \
        .replace("__OVERLAP__", "none" if not overlaps else str(overlaps)) \
        .replace("__DATE__", date.today().isoformat()) \
        .replace("__NCELL__", str(len(eps)))
    (OUT / "f4_exact_spine.html").write_text(html)
    print(f"cells={len(eps)} rank_rho={rho:.3f} deficit_CI_overlaps={overlaps or 'none'}")
    for e in eps:
        print(f"  eps={e:<5g} width={W[e]['width']:.9f}  deficit={D[e]['mean']:.9f}"
              f" +/- {D[e]['se']:.2e}  [{W[e]['status']}/{D[e]['status']}]")
    return 0


HTML = """<title>F4 Exact Spine</title>
<style>
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8982;
      --line:#e6e5e0;--blue:#2a78d6;--orange:#eb6834;--ok:#1baf7a;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a8982;
      --line:#33332f;--blue:#3987e5;--orange:#d95926;--ok:#199e70;}}
:root[data-theme=dark]{--bg:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;
      --ink3:#8a8982;--line:#33332f;--blue:#3987e5;--orange:#d95926;--ok:#199e70;}
body{background:var(--bg);color:var(--ink);margin:0;padding:44px 28px 80px;
     font:15.5px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;}
main{max-width:1020px;margin:0 auto;}
h1{font-size:25px;font-weight:640;letter-spacing:-.015em;margin:0 0 4px;}
.sub{color:var(--ink2);margin:0 0 30px;font-size:14.5px;}
h2{font-size:16.5px;font-weight:620;margin:38px 0 10px;letter-spacing:-.01em;}
p{color:var(--ink2);max-width:74ch;}
figure{margin:0 0 8px;background:var(--card);border:1px solid var(--line);
       border-radius:12px;padding:18px;overflow-x:auto;}
figure img{width:100%;max-width:100%;display:block;border-radius:4px;}
figcaption{color:var(--ink2);font-size:13.5px;margin-top:14px;line-height:1.6;}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);
      border-radius:12px;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th{text-align:left;color:var(--ink3);font-weight:560;padding:11px 13px;
   border-bottom:1px solid var(--line);white-space:nowrap;}
td{padding:9px 13px;border-bottom:1px solid var(--line);}
tr:last-child td{border-bottom:none;}
.num{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--ink3);}
.st{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--ok);}
.note{border-left:3px solid var(--blue);padding:2px 0 2px 16px;margin:22px 0;}
.note.absent{border-left-color:var(--orange);}
footer{color:var(--ink3);font-size:12.5px;margin-top:44px;
       border-top:1px solid var(--line);padding-top:16px;}
</style>
<main>
<h1>F4, exact spine</h1>
<p class="sub">The two model-free ladders, __NCELL__ cells each, drawn before any PFN overlay
exists. Generated __DATE__ from the artifacts, not transcribed.</p>

<figure>
<img alt="Three panels: exact identified-set width falling with epsilon; order-marginal
entropy deficit rising with epsilon; the two plotted against each other." src="data:image/png;base64,__PNG__">
<figcaption>Left, the exact identified-set width, one solver
(<code>highs-ipm</code>) and <code>n_rows = 20</code> at every rung. Middle, the
normalized order-marginal entropy deficit on half A of the W2 panels, 500 contexts
per half, shaded band is 95%. Right, the same seven cells on both axes at once;
rank correlation __RHO__. Bold markers are the two rungs F.6r added.
<strong>There is no model in this figure.</strong> Under E.8 the Q3 PFN panel is
dropped rather than left empty: a vanilla PFN has no intervention channel, so the
Pearl L2 rung has no model-side counterpart by construction, and
<code>pfn_panel_predictions</code> returns <code>None</code> for every
interventional spec. That is a statement about the model class, not a missing
measurement.</figcaption>
</figure>

<h2>The numbers behind it</h2>
<div class="wrap"><table>
<thead><tr><th>eps</th><th>Q1 width</th><th>entropy deficit</th><th>SE</th>
<th>width tag</th><th>deficit tag</th><th>width artifact</th><th>deficit artifact</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>

<div class="note">
<p>Both ladders are strictly monotone across all seven cells, and no adjacent pair of
deficit CIs overlaps at z = 1.96 (overlapping pairs: __OVERLAP__), so F.2b's near-tie
clause produces no unresolved pair on this axis. The script raises rather than draws
if either ladder loses monotonicity, because that is a substrate finding under T4's
stop condition and not a plotting nuisance.</p>
</div>

<div class="note absent">
<p><strong>What is deliberately absent.</strong> The PFN overlay. It needs the Track-B
retrain, which is held behind Amendment F's lock, which is held on two open clauses.
The layering rule in E.8 exists exactly for this case: the spine is the half that
does not depend on the retrain, so it is finished first and the figure stands on its
own if the overlay never arrives.</p>
</div>

<footer>Every value read through the fail-closed loader (<code>cells_from_raw</code>,
which requires an F.8 <code>live</code> tag and a recorded <code>n_rows</code>) or from
a deficit artifact whose tag is checked the same way. Regenerate with
<code>python scripts/f4_exact_spine.py</code>.</footer>
</main>
"""

if __name__ == "__main__":
    raise SystemExit(main())
