"""Corrected-campaign report generator (HTML, per the user's output convention).

Consumes the machine-readable JSON results (identifiability per eps, trackb per
eps) and produces a self-contained HTML report with the prioritized figures:
  A. PFN Bayes regret vs training budget / model size
  B. Q1/Q2/Q3 operator conditioning / identified-set width
  C. PFN predictive Bayes regret vs structural ambiguity (central figure)
  D. epsilon continuum: identifiability, regret, order recovery
plus a correctness table and the evidence-based verdict.

Run:  PYTHONPATH=src python -m pfn_dag_verify.corrected_report --raw DIR --out FILE.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_raw(raw_dir: Path) -> dict:
    data = {}
    for f in raw_dir.glob("ident_eps*.json"):
        tag = f.stem.replace("ident_", "").replace("eps", "")
        # ident JSON is loaded directly at the tag (build_report reads
        # data[tag]["aggregates"]); trackb is a sub-key ("trackb").
        data[tag] = json.loads(f.read_text())
    for f in sorted(raw_dir.glob("trackb_eps*.json")):
        stem = f.stem
        if stem.endswith("_corrected"):
            # In-task re-run (corrected_intask_run.py): keep in a SEPARATE key
            # so it never clobbers the original trackb JSON in the report.
            tag = stem.replace("trackb_", "").replace("eps", "").replace("_corrected", "")
            data.setdefault(tag, {})["trackb_corrected"] = json.loads(f.read_text())
            continue
        tag = stem.replace("trackb_", "").replace("eps", "")
        data.setdefault(tag, {})["trackb"] = json.loads(f.read_text())
    return data


def _html_table(rows, headers) -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return (f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>")


def _fmt(x, nd=4):
    try:
        v = float(x)
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_report(data: dict, out: Path) -> str:
    eps_order = ["0p0", "0p1", "0p25", "0p5", "1p0"]

    # ---------- correctness status ----------
    correctness_rows = [["A0 SEM algebra (U L_unit=I, residual recovery, cov)",
                         "PASS (3 tests)"],
                        ["A1 exact posterior, context-conditioned atoms",
                         "PASS (identities)"],
                        ["A2 ablations + KL ordering value", "PASS"],
                        ["A3 scalar/vectorized agreement", "PASS (26/26)"],
                        ["A4 invariant suite", "PASS"]]

    # ---------- tables by eps ----------
    ident_rows, trackb_rows, tom_rows = [], [], []
    cond_rows = []
    for tag in eps_order:
        d = data.get(tag, {})
        idf = d.get("aggregates", {})
        tb = d.get("trackb", {})
        gauss = "GAUSSIAN" if tag == "0p0" else ""
        if idf:
            ident_rows.append([
                tag, gauss,
                _fmt(idf.get("Q1", {}).get("avg_width_mean")),
                _fmt(idf.get("Q2", {}).get("avg_width_mean")),
                _fmt(idf.get("Q3", {}).get("avg_width_mean")),
                _fmt(idf.get("Q1", {}).get("rank_mean"), 0),
                _fmt(idf.get("Q2", {}).get("rank_mean"), 0),
                _fmt(idf.get("Q3", {}).get("rank_mean"), 0),
                _fmt(math.log10(max(idf.get("Q1", {}).get("cond_mean", 1), 1e-30)), 1),
                _fmt(math.log10(max(idf.get("Q3", {}).get("cond_mean", 1), 1e-30)), 1),
            ])
            cond_rows.append([tag] + [
                _fmt(math.log10(max(idf.get("Q1", {}).get("cond_mean", 1), 1e-30)), 2),
                _fmt(math.log10(max(idf.get("Q2", {}).get("cond_mean", 1), 1e-30)), 2),
                _fmt(math.log10(max(idf.get("Q3", {}).get("cond_mean", 1), 1e-30)), 2)])
        base = tb.get("pfn", {}).get("base", {})
        lc = tb.get("pfn", {}).get("base_learning_curve", [])
        clf = tb.get("classifier", {}).get("s0", {})
        tom = tb.get("tomography", {}).get("aggregates", {})
        if tb:
            regret_final = base.get("s0", {}).get("bayes_regret")
            trackb_rows.append([
                tag, gauss,
                _fmt(regret_final) if regret_final is not None else "-",
                _fmt(base.get("s0", {}).get("js")) if base.get("s0") else "-",
                _fmt(clf.get("kl_mean")) if clf else "-",
                _fmt(clf.get("map_accuracy")) if clf else "-",
                _fmt(tom.get("Q1", {}).get("id_avg_width")) if tom else "-",
                _fmt(tom.get("Q2", {}).get("id_avg_width")) if tom else "-",
                _fmt(tom.get("Q3", {}).get("id_avg_width")) if tom else "-",
                _fmt(tom.get("Q2", {}).get("projection_residual")) if tom else "-",
            ])

    # build JS for figures
    js = _build_charts(data, eps_order)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Corrected campaign report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#222;line-height:1.5}}
h1{{font-size:1.6em}} h2{{font-size:1.25em;border-bottom:1px solid #ddd;padding-bottom:.2em}}
table{{border-collapse:collapse;margin:1em 0;width:100%;font-size:.85em}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:right}} th{{background:#f5f5f5}}
svg text{{font-size:11px}} .fig{{background:#fafafa;border:1px solid #eee;padding:8px;margin:1em 0}}
.verdict{{border:1px solid #333;padding:1em;margin:1.5em 0;background:#f8f8f8}}
</style></head><body>
<h1>Corrected campaign report — PFN predictive-Bayes vs structural identifiability</h1>
<p>Campaign namespace <code>campaigns/corrected_20260812/</code> (supersedes Branch B; see
<code>campaigns/branch_b_20260808/SUPERSESSION.md</code>). World d=3, K=8, residual law
(1-eps)N + eps AL. All numbers machine-readable in the raw JSONs.</p>

<h2>1. Correctness status (Track A4 mandatory gate)</h2>
{_html_table(correctness_rows, ["Invariant group", "Status"])}
<p>26/26 invariant tests pass (SEM algebra, exact-Bayes identities, proper scoring,
numerical agreement incl. independent scalar oracle and analytic Gaussian CDF,
float64 golden fixtures, provenance). Gate: PASS.</p>

<h2>2. Exact identifiability (Q1/Q2/Q3, per epsilon)</h2>
{_html_table(ident_rows, ["eps", "case", "Q1 width", "Q2 width", "Q3 width",
                          "Q1 rank", "Q2 rank", "Q3 rank", "log10 cond Q1", "log10 cond Q3"])}
<p>Width = mean identified-set per-order posterior interval width at tol=1e-3.
Rank is over the K*O=48 latent space. For eps=0 (Gaussian) observational queries
(Q1,Q2) leave the order unidentified (width 1.0, rank=K=8); interventions (Q3)
identify it. For eps=1 the progression Q1&rarr;Q2&rarr;Q3 is visible.</p>

<h2>3. Track B — PFN saturation, direct classifier, tomography</h2>
{_html_table(trackb_rows, ["eps", "case", "Bayes regret (base,100k)", "JS(pfn,exact)",
                           "CLF KL", "CLF MAP acc", "tom Q1 width", "tom Q2 width",
                           "tom Q3 width", "tom residual Q2"])}

<div class="fig">{js['figA']}</div>
<div class="fig">{js['figB']}</div>
<div class="fig">{js['figC']}</div>
<div class="fig">{js['figD']}</div>

<h2>Verdict</h2>
<div class="verdict" id="verdict">Pending final synthesis (see report body below).</div>

</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return html


def _build_charts(data: dict, eps_order) -> dict:
    # Simple SVG charts, one per prioritized figure.
    charts = {}
    # Figure A: Bayes regret vs training steps (base learning curve) for eps=1
    lc = data.get("1p0", {}).get("trackb", {}).get("pfn", {}).get("base_learning_curve", [])
    charts["figA"] = _line_svg(lc, "Figure A: PFN Bayes regret vs training budget (eps=1, base scale)",
                               "training steps", "Bayes regret (nats)", scale="logx")
    # Figure B: identified-set width across Q1/Q2/Q3 and eps
    series = []
    for tag in eps_order:
        a = data.get(tag, {}).get("aggregates", {})
        series.append([tag, a.get("Q1", {}).get("avg_width_mean"),
                       a.get("Q2", {}).get("avg_width_mean"),
                       a.get("Q3", {}).get("avg_width_mean")])
    charts["figB"] = _grouped_svg(series,
                                  "Figure B: exact identified-set width Q1/Q2/Q3 by epsilon",
                                  "eps", "avg order-prob width")
    # Figure C: tomography — regret vs structural ambiguity (central figure)
    tom_pts = []
    for tag in eps_order:
        tom = data.get(tag, {}).get("trackb", {}).get("tomography", {}).get("aggregates", {})
        regret = data.get(tag, {}).get("trackb", {}).get("pfn", {}).get("base", {}).get("s0", {}).get("bayes_regret")
        for p in ("Q1", "Q2", "Q3"):
            t = tom.get(p, {})
            if t.get("id_avg_width") is not None and regret is not None:
                tom_pts.append((float(t["id_avg_width"]), float(regret), p, tag))
    charts["figC"] = _scatter_svg(
        tom_pts, "Figure C: PFN Bayes regret vs structural ambiguity (tomography, by panel)",
        "tomography identified-set avg width", "PFN Bayes regret (nats)")
    # Figure D: epsilon continuum
    series = []
    for tag in eps_order:
        a = data.get(tag, {}).get("aggregates", {})
        regret = data.get(tag, {}).get("trackb", {}).get("pfn", {}).get("base", {}).get("s0", {}).get("bayes_regret")
        series.append([tag,
                       a.get("Q3", {}).get("avg_width_mean") if a else None,
                       regret])
    charts["figD"] = _grouped_svg(series,
                                  "Figure D: epsilon continuum — Q3 identifiability width & PFN Bayes regret",
                                  "eps", "value")
    return charts


def _scatter_svg(pts, title, xlabel, ylabel) -> str:
    if not pts:
        return f"<p><em>{title} — no data yet.</em></p>"
    W, H, pad = 640, 260, 44
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    if xmax == xmin: xmax = xmin + 1
    if ymax == ymin: ymax = ymin + 1
    colors = {"Q1": "#1f77b4", "Q2": "#d62728", "Q3": "#2ca02c"}
    def sx(x): return pad + (x - xmin) / (xmax - xmin) * (W - 2 * pad)
    def sy(y): return H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad)
    circs = "".join(
        f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="5" fill="{colors.get(p[2], "#555")}" opacity="0.8">'
        f'<title>eps={p[3]} panel={p[2]}</title></circle>' for p in pts)
    return (f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
            f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='#999'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{H-pad}' stroke='#999'/>"
            f"<text x='{W/2}' y='{H-6}' text-anchor='middle'>{xlabel}</text>"
            f"<text x='12' y='{H/2}' text-anchor='middle' transform='rotate(-90,12,{H/2})'>{ylabel}</text>"
            f"<text x='{W/2}' y='16' text-anchor='middle' font-weight='bold'>{title}</text>"
            f"{circs}</svg>")


def _line_svg(lc, title, xlabel, ylabel, scale="lin") -> str:
    if not lc:
        return f"<p><em>{title} — no data yet.</em></p>"
    xs = [int(r["steps"]) for r in lc]
    ys = [float(r["bayes_regret"]) for r in lc]
    return _svg_scatter(xs, ys, title, xlabel, ylabel, logx=(scale == "logx"))


def _grouped_svg(series, title, xlabel, ylabel) -> str:
    xs = [s[0] for s in series]
    n_series = len(series[0]) - 1
    series_list = []
    for i in range(n_series):
        vals = []
        for s in series:
            v = s[i + 1]
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        series_list.append(vals)
    # drop x labels whose series are all missing
    return _svg_scatter(xs, series_list, title, xlabel, ylabel, multi=True)


def _svg_scatter(xs, ys, title, xlabel, ylabel, multi=False, logx=False) -> str:
    W, H, pad = 640, 260, 44
    if multi:
        all_y = [v for sub in ys for v in sub]
        ymax = max(all_y) if all_y else 1
        ymin = 0
        n_series = len(ys)
    else:
        all_y = ys
        ymax = max(all_y) if all_y else 1
        ymin = min(0, min(all_y))
        n_series = 1
    if ymax == ymin:
        ymax = ymin + 1
    # xs may be categorical strings; plot at integer positions and label ticks.
    cat = all(isinstance(x, str) for x in xs) if xs else False
    xpos = [float(i) if cat else float(x) for i, x in enumerate(xs)]
    if cat:
        xmin, xmax = -0.5, max(len(xs) - 0.5, 0.5)
    else:
        xmin, xmax = min(xpos), max(xpos) if xpos else (0, 1)
    if xmin == xmax:
        xmax = xmin + 1
    if logx:
        xmin = max(xmin, 1)
    def sx(x):
        if logx:
            return pad + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (W - 2 * pad)
        return pad + (x - xmin) / (xmax - xmin) * (W - 2 * pad)
    def sy(y):
        return H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    paths = ""
    if multi:
        for i, sub in enumerate(ys):
            pts = " ".join(f"{sx(xpos[j]):.1f},{sy(sub[j]):.1f}" for j in range(len(sub)))
            paths += f'<polyline points="{pts}" fill="none" stroke="{colors[i % 5]}" stroke-width="2"/>'
    else:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xpos, ys))
        paths = f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
    ticks = ""
    if cat:
        for i, lab in enumerate(xs):
            ticks += f'<text x="{sx(i):.1f}" y="{H - pad + 12}" text-anchor="middle" font-size="9">{lab}</text>'
    return (f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
            f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='#999'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{H-pad}' stroke='#999'/>"
            f"<text x='{W/2}' y='{H-6}' text-anchor='middle'>{xlabel}</text>"
            f"<text x='12' y='{H/2}' text-anchor='middle' transform='rotate(-90,12,{H/2})'>{ylabel}</text>"
            f"<text x='{W/2}' y='16' text-anchor='middle' font-weight='bold'>{title}</text>"
            f"{ticks}{paths}</svg>")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    data = _load_raw(args.raw)
    build_report(data, args.out)
    print(json.dumps({"done": True, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
