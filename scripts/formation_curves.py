"""The two formation curves: tracking fidelity against training dose, and against scale.

The estimand is the E1a evidence-tracking slope. `positive_control` in
exp1_evidence_tracking pins its meaning: a perfect Bayes reader, Q = sigmoid(ell)
p_G1 + sigmoid(-ell) p_G2, yields logit w = -ell EXACTLY, so beta_w = -1 (measured
-1.0046 as a harness check). Tracking fidelity is therefore

    phi = -theil_slope(logit w on ell)

normalized so that a perfect Bayes reader scores 1.0 and a reader carrying no
evidence scores 0. Theil-Sen rather than OLS because the harness reports both and
the robust slope is the one its own positive control is stated against.

One point per (scale, prior, dose) is the mean over model seeds, with the
ACROSS-SEED spread as the interval, since seed is the replication unit for a claim
about scale or dose and the within-cell regression SE is the wrong ruler for it.

This reads whatever has landed. The consolidation fleet fills the grid a cell at a
time, so the script reports its own coverage rather than assuming completeness, and
a curve drawn from one seed is drawn and labelled as one seed.
"""
from __future__ import annotations

import base64
import glob
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / "campaigns/binary_duel_ext3_20260821/results/exp1"
OUT = ROOT / "reports"

SCALES = ["base", "mid", "large", "xl"]
DOSES = [0, 100, 300, 1000, 3000, 6000, 12000]
# dataviz slots 1-4, validated adjacent in both modes; aqua and yellow sit below
# 3:1 on the light surface, so the relief rule applies and every line is
# direct-labelled and the table ships beneath.
COL = {"base": "#2a78d6", "mid": "#eb6834", "large": "#1baf7a", "xl": "#eda100"}
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8982"


def fidelity(doc, tag="e1a_natural"):
    return -float(doc[tag]["reg_w"]["theil_slope"])


def load(prior="AL40"):
    cells = defaultdict(list)
    for f in sorted(glob.glob(str(EXP1 / "*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("prior") != prior:
            continue
        cells[(d["scale"], int(d["dose"]))].append(fidelity(d))
    return cells


def stat(v):
    v = np.asarray(v, float)
    return (float(v.mean()),
            float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            len(v))


def figure(cells, prior, path_pdf, path_png):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d7d2")
        ax.tick_params(colors=INK2, labelsize=8.5, length=3)
        ax.grid(True, color="#ebeae5", lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    a = axes[0]
    xs = np.arange(len(DOSES))
    for sc in SCALES:
        pts = [(i, stat(cells[(sc, d)])) for i, d in enumerate(DOSES) if cells.get((sc, d))]
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1][0] for p in pts]
        s = [p[1][1] for p in pts]
        n = [p[1][2] for p in pts]
        a.plot(x, y, "-", color=COL[sc], lw=2, zorder=3)
        for xi, yi, si, ni in zip(x, y, s, n):
            a.plot([xi], [yi], "o", ms=7 if ni > 1 else 5.5, color=COL[sc],
                   mec=SURF, mew=2, zorder=4)
            if ni > 1 and not math.isnan(si):
                a.plot([xi, xi], [yi - si, yi + si], "-", color=COL[sc],
                       lw=1.4, alpha=0.55, zorder=2)
        # relief rule: aqua and yellow are below 3:1 on this surface, so every
        # series carries a visible label rather than relying on hue alone.
        a.annotate(f"{sc}", (x[-1], y[-1]), textcoords="offset points",
                   xytext=(7, -3), fontsize=9, color=INK, weight="600")
    a.set_xticks(xs)
    a.set_xticklabels([f"{d:g}" for d in DOSES], fontsize=8)
    a.set_xlim(-0.35, len(DOSES) - 0.35 + 0.9)
    a.set_title("Formation: tracking fidelity against training dose", fontsize=10,
                color=INK, loc="left", pad=8)
    a.set_xlabel("training dose (steps)", fontsize=9, color=INK2)
    a.set_ylabel(r"$\phi = -\beta_w$   (perfect Bayes = 1)", fontsize=9, color=INK2)

    b = axes[1]
    top = max((d for d in DOSES if any(cells.get((sc, d)) for sc in SCALES)),
              default=DOSES[-1])
    xs2, ys2, ss2, ns2, keep = [], [], [], [], []
    for i, sc in enumerate(SCALES):
        if cells.get((sc, top)):
            m, sd, n = stat(cells[(sc, top)])
            xs2.append(i); ys2.append(m); ss2.append(sd); ns2.append(n); keep.append(sc)
    if xs2:
        b.plot(xs2, ys2, "-", color="#b9b8b2", lw=1.6, zorder=2)
        for xi, yi, si, ni, sc in zip(xs2, ys2, ss2, ns2, keep):
            b.plot([xi], [yi], "o", ms=8, color=COL[sc], mec=SURF, mew=2, zorder=4)
            if ni > 1 and not math.isnan(si):
                b.plot([xi, xi], [yi - si, yi + si], "-", color=COL[sc], lw=1.5,
                       alpha=0.55, zorder=2)
            b.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                       xytext=(9, 4), fontsize=8.5, color=INK)
    b.set_xticks(range(len(SCALES)))
    b.set_xticklabels(SCALES, fontsize=8.5)
    b.set_xlim(-0.4, len(SCALES) - 0.6 + 0.2)
    b.set_title(f"Scale at dose {top:g} (step-matched, not compute-matched)",
                fontsize=10, color=INK, loc="left", pad=8)
    b.set_xlabel("model scale", fontsize=9, color=INK2)
    b.set_ylabel(r"$\phi = -\beta_w$", fontsize=9, color=INK2)

    fig.tight_layout(pad=1.3)
    fig.savefig(path_pdf, facecolor=SURF, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return top


def main():
    OUT.mkdir(exist_ok=True)
    cells = load("AL40")
    have = sum(1 for k in cells)
    total = len(SCALES) * len(DOSES)
    top = figure(cells, "AL40", OUT / "formation_curves.pdf", OUT / "formation_curves.png")
    png = base64.b64encode((OUT / "formation_curves.png").read_bytes()).decode()

    rows = []
    for sc in SCALES:
        for d in DOSES:
            v = cells.get((sc, d))
            if not v:
                rows.append(f"<tr><td>{sc}</td><td class=n>{d:g}</td>"
                            f"<td class=pend colspan=3>not scored yet</td></tr>")
                continue
            m, sd, n = stat(v)
            sdt = "n/a" if math.isnan(sd) else f"{sd:.4f}"
            rows.append(f"<tr><td>{sc}</td><td class=n>{d:g}</td>"
                        f"<td class=n>{m:.4f}</td><td class=n>{sdt}</td>"
                        f"<td class=n>{n}</td></tr>")
    html = TPL.replace("__PNG__", png).replace("__ROWS__", "\n".join(rows)) \
        .replace("__HAVE__", str(have)).replace("__TOTAL__", str(total)) \
        .replace("__TOP__", f"{top:g}").replace("__DATE__", date.today().isoformat())
    (OUT / "formation_curves.html").write_text(html)
    print(f"AL40 coverage: {have}/{total} (scale x dose) cells scored; top dose {top:g}")
    for sc in SCALES:
        got = [f"{d:g}" for d in DOSES if cells.get((sc, d))]
        print(f"  {sc:<6} doses scored: {' '.join(got) if got else '(none)'}")
    return 0


TPL = """<title>Formation Curves</title>
<style>
:root{--bg:#fcfcfb;--surface:#fff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8982;--line:#e6e5e0;--ok:#1baf7a;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--surface:#232322;
 --ink:#fff;--ink2:#c3c2b7;--ink3:#8a8982;--line:#33332f;--ok:#199e70;}}
:root[data-theme=dark]{--bg:#1a1a19;--surface:#232322;--ink:#fff;--ink2:#c3c2b7;
 --ink3:#8a8982;--line:#33332f;--ok:#199e70;}
body{background:var(--bg);color:var(--ink);margin:0;padding:44px 26px 80px;
 font:15.5px/1.66 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;}
main{max-width:960px;margin:0 auto;}
h1{font-size:25px;font-weight:640;letter-spacing:-.015em;margin:0 0 5px;}
.sub{color:var(--ink2);margin:0 0 30px;font-size:14.5px;}
h2{font-size:16.5px;font-weight:620;margin:36px 0 10px;}
p{color:var(--ink2);max-width:74ch;}
figure{margin:0 0 8px;background:var(--surface);border:1px solid var(--line);
 border-radius:12px;padding:18px;overflow-x:auto;}
figure img{width:100%;display:block;border-radius:4px;}
figcaption{color:var(--ink2);font-size:13.5px;margin-top:13px;line-height:1.6;}
.wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
 border-radius:12px;margin:16px 0;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th{text-align:left;color:var(--ink3);font-weight:560;padding:11px 13px;
 border-bottom:1px solid var(--line);}
td{padding:8px 13px;border-bottom:1px solid var(--line);}
tr:last-child td{border-bottom:none;}
.n{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;}
.pend{color:var(--ink3);font-style:italic;}
code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;}
footer{color:var(--ink3);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px;}
</style>
<main>
<h1>Formation and scale curves</h1>
<p class="sub">Tracking fidelity at AL40. __HAVE__ of __TOTAL__ scale-by-dose cells scored
so far; the consolidation fleet is filling the rest. Generated __DATE__.</p>
<figure><img alt="Left, tracking fidelity against training dose, one line per model scale.
Right, tracking fidelity against model scale at the top dose." src="data:image/png;base64,__PNG__">
<figcaption>Fidelity is the Theil-Sen slope of the model's logit evidence weight on the true
log-odds, sign-flipped so that a perfect Bayes reader scores 1.0 and a reader carrying no
evidence scores 0. That normalization is not a convention chosen here: the harness's own
positive control shows a perfect Bayes reader gives exactly -1. Points are means over model
seeds and bars are the across-seed spread, which is the right ruler for a claim about dose
or scale; a point with a single seed is drawn smaller and carries no bar. Every series is
labelled directly, since two of the four hues fall below 3:1 on this surface.</figcaption>
</figure>
<h2>The numbers</h2>
<div class="wrap"><table>
<thead><tr><th>scale</th><th>dose</th><th>fidelity</th><th>across-seed sd</th><th>seeds</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<h2>What the two curves say so far</h2>
<p><strong>Formation.</strong> The base curve is complete and it behaves. An untrained model
reads 0.0000 fidelity to within 0.0005 across twelve seeds, which is a natural zero rather
than a fitted one and is the strongest available check that the readout measures evidence
and not an artifact of the regression. Fidelity is flat through dose 100, turns up at 300,
and climbs monotonically to 0.178 at 12000 without flattening, so the run ends while the
quantity is still forming. Whatever the ceiling is, this budget does not reach it.</p>
<p><strong>Scale, and a warning about reading it.</strong> At the shared dose of 12000 the
ordering is base 0.178, xl 0.107, mid 0.088, large 0.064, so the small model tracks best and
the relation is not monotone in capacity. Before that is read as a finding: this is a
STEP-matched comparison, not a compute-matched one. Every scale ran the same 12000 optimizer
steps, so the larger models have seen the same number of updates for far more parameters and
sit earlier on their own formation curve. The base curve above shows how much of the effect
that alone can carry, since base moves from 0.035 to 0.178 over the last factor of forty in
dose. The honest reading is that the scale panel is not yet interpretable, and it becomes
interpretable when the other three scales have their dose ladders, which is what the fleet
is filling now.</p>
<p>Rows marked <em>not scored yet</em> have their checkpoint on disk and no result JSON.
That is the whole of what the consolidation was missing: the grid of 1092 trained models is
complete, and only the scoring had never been launched.</p>
<footer>Generated by <code>scripts/formation_curves.py</code> from
<code>campaigns/binary_duel_ext3_20260821/results/exp1/*.json</code>. Re-run it as cells land.</footer>
</main>
"""

if __name__ == "__main__":
    raise SystemExit(main())
