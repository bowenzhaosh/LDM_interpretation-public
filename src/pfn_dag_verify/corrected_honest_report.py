"""Honest-disposition report for the corrected campaign (HTML).

Supersedes the first-pass corrected_report.html. The results-audit rejected the
drawn verdict because the stored 'bayes_regret' is a mean over a mixed-target
panel while the PFN is trained to predict only column d-1 (no target-index
input): ~2/3 of the panel is out-of-task. This report:

  1. Reports the IN-TASK (target=d-1) regret from the corrected re-runs
     (corrected_intask_run.py) as the primary quantity, with the mixed-panel
     aggregate kept for continuity.
  2. Recomputes the verdict on in-task numbers (classify_intask).
  3. Discloses the items the audit flagged: the undisclosed nrows10 ident set,
     the degenerate eps0p0 tomography placeholder, the half-budget classifier,
     solver provenance, single-seed original reporting, no code/version
     pinning, and eps0.5 being skipped in the intask fleet.

Run:  PYTHONPATH=src python -m pfn_dag_verify.corrected_honest_report \
          --raw DIR --out FILE.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .corrected_report import _fmt, _html_table, _line_svg
from .corrected_verdict import apply_written_default, cells_from_raw, classify

EPS_ORDER = ["0p0", "0p1", "0p25", "0p5", "1p0"]
IN_TASK_EPS = ["0p0", "0p1", "0p25", "1p0"]   # eps0.5 skipped by design

STYLE = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222;line-height:1.5}
h1{font-size:1.6em} h2{font-size:1.25em;border-bottom:1px solid #ddd;padding-bottom:.2em} h3{font-size:1.05em}
table{border-collapse:collapse;margin:1em 0;width:100%;font-size:.85em}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:right} th{background:#f5f5f5}
svg text{font-size:11px} .fig{background:#fafafa;border:1px solid #eee;padding:8px;margin:1em 0}
.verdict{border:2px solid #333;padding:1em;margin:1.5em 0;background:#f8f8f8}
.warn{border:1px solid #b8860b;background:#fdf6e3;padding:.5em 1em;margin:1em 0}
table.left th,table.left td{text-align:left}
"""


def _seed_stats(pfn: dict) -> dict:
    """Mean/SE/min/max over seeds of the given key."""
    seeds = [pfn[k] for k in ("s0", "s1", "s2") if k in pfn]
    vals = [s["in_task_regret"] for s in seeds if s.get("in_task_regret") is not None]
    mx = [s["bayes_regret"] for s in seeds if s.get("bayes_regret") is not None]
    out = {"n_seeds": len(seeds)}
    if vals:
        out["in_task_mean"] = float(np.mean(vals))
        out["in_task_se"] = float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else None
        out["in_task_minmax"] = [float(min(vals)), float(max(vals))]
    if mx:
        out["mixed_mean"] = float(np.mean(mx))
        out["mixed_se"] = float(np.std(mx) / math.sqrt(len(mx))) if len(mx) > 1 else None
    return out


def build_honest_report(data: dict, out: Path) -> str:
    # ---------- exact identifiability (n_rows=20, the recorded set) ----------
    ident_rows = []
    for tag in EPS_ORDER:
        a = data.get(tag, {}).get("aggregates", {})
        if not a:
            continue
        ident_rows.append([
            tag,
            "GAUSSIAN" if tag == "0p0" else "",
            _fmt(a.get("Q1", {}).get("avg_width_mean")),
            _fmt(a.get("Q2", {}).get("avg_width_mean")),
            _fmt(a.get("Q3", {}).get("avg_width_mean")),
            _fmt(a.get("Q1", {}).get("rank_mean"), 0),
            _fmt(a.get("Q3", {}).get("rank_mean"), 0),
        ])

    # ---------- nrows10 disclosure ----------
    n10_rows = []
    for tag in ["0p1", "0p25", "0p5"]:
        a20 = data.get(tag, {}).get("aggregates", {})
        a10 = data.get(tag + "_nrows10", {}).get("aggregates", {})
        w20 = a20.get("Q1", {}).get("avg_width_mean")
        w10 = a10.get("Q1", {}).get("avg_width_mean")
        if w20 is not None and w10 is not None:
            n10_rows.append([tag, _fmt(w20), _fmt(w10),
                             _fmt(w10 / w20, 2) if w20 else "-"])

    # ---------- in-task trackb ----------
    it_rows, per_seed_rows = [], []
    for tag in IN_TASK_EPS:
        d = data.get(tag, {})
        idf = d.get("aggregates", {})
        tb = d.get("trackb", {})
        tbc = d.get("trackb_corrected", {})
        pfn_c = tbc.get("pfn", {})
        st = _seed_stats(pfn_c)
        clf = tb.get("classifier", {}).get("s0", {})
        tom = tb.get("tomography", {}).get("aggregates", {})
        lc = tbc.get("base_in_task_learning_curve", [])
        lc_final = lc[-1]["in_task_regret"] if lc else None
        rising = (len(lc) >= 4 and lc_final is not None
                  and lc_final > lc[-2]["in_task_regret"])
        if not st or "in_task_mean" not in st:
            continue
        it_rows.append([
            tag, st["n_seeds"],
            _fmt(st["mixed_mean"]), _fmt(st["mixed_se"]),
            _fmt(st["in_task_mean"]), _fmt(st["in_task_se"]),
            (f"{st['in_task_minmax'][0]:.3f}-{st['in_task_minmax'][1]:.3f}"
             if "in_task_minmax" in st else "-"),
            _fmt(lc_final) if lc_final is not None else "-",
            "RISING" if rising else "flat",
            _fmt(idf.get("Q1", {}).get("avg_width_mean")),
            _fmt(clf.get("kl_mean")) if clf else "-",
        ])
        for seed_name in ("s0", "s1", "s2"):
            s = pfn_c.get(seed_name)
            if s:
                per_seed_rows.append([
                    tag, seed_name, _fmt(s["in_task_regret"]),
                    _fmt(s["bayes_regret"]), _fmt(s["final_loss"]),
                    _fmt(s.get("in_task_frac")),
                ])

    # ---------- tomography (with eps0p0 placeholder flag) ----------
    tom_rows = []
    for tag in EPS_ORDER:
        tom = data.get(tag, {}).get("trackb", {}).get("tomography", {})
        a = tom.get("aggregates", {})
        if not a:
            continue
        placeholder = (tag == "0p0" and tom.get("tol", 1) > 0.05)
        tom_rows.append([
            tag, _fmt(tom.get("tol"), 4),
            ("<b>PLACEHOLDER</b>" if placeholder
             else _fmt(a.get("Q1", {}).get("id_avg_width"))),
            _fmt(a.get("Q1", {}).get("id_avg_width")),
            _fmt(a.get("Q2", {}).get("id_avg_width")),
            _fmt(a.get("Q3", {}).get("id_avg_width")),
            _fmt(a.get("Q1", {}).get("structural_tv")),
            _fmt(a.get("Q3", {}).get("structural_tv")),
        ])

    # ---------- figures ----------
    fig_lc = _intask_lc_svg(data)
    fig_mixed = _mixed_vs_intask_svg(it_rows, IN_TASK_EPS)

    # ---------- verdict ----------
    # Amendment E.4 verdict, computed from the per-eps cell tuple on disk.
    from datetime import date
    from pathlib import Path as _P
    _raw = _P(__file__).resolve().parents[2] / "campaigns" / "corrected_20260812" / "raw"
    v = apply_written_default(classify(cells_from_raw(_raw)), date.today().isoformat())
    verdict_html = f"""<div class="verdict" id="verdict">
<h3>Verdict (Amendment E.4): {v['verdict']} &mdash; headline {v['headline']}</h3>
<ul style="text-align:left">{''.join(f'<li>{e}</li>' for e in v['evidence'])}</ul>
</div>"""

    n10_html = ""
    if n10_rows:
        n10_html = f"""
<h2>4. Identifiability at n_rows=10 (undisclosed set — audit disclosure)</h2>
{_html_table(n10_rows, ["eps", "Q1 width n=20", "Q1 width n=10", "ratio 10/20"])}
<p class="warn">The <code>nrows10_ident_eps*.json</code> files share the recorded
config (seed 990200000, n_contexts=20, tol=1e-3) but differ in n_rows, which is
NOT recorded in the file config (filename-only). Q1 identified-set width grows
when fewer context rows are available (+25%/+36%/+79% at eps 0.1/0.25/0.5) — the
order posterior is less identified from 10-row contexts than from 20-row ones.
The main tables use the n_rows=20 set. Note the eps0.5 nrows10 file does not
carry the <code>highs-ipm</code> solver tag the n_rows=20 eps0.5 file records
(it predates the LP-degeneracy fix).</p>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Corrected campaign report (honest disposition)</title>
<style>{STYLE}</style></head><body>
<h1>Corrected campaign — honest disposition (post-audit)</h1>
<p>World d=3, K=8, residual law (1-eps)N + eps AL. Campaign namespace
<code>campaigns/corrected_20260812/</code>. This report replaces the first-pass
<code>corrected_report.html</code> after the 6-lens results-audit rejected the
drawn verdict as invalid-as-instrumented. <b>All regret numbers are on the
trained task (target=d-1); the stored mixed-panel aggregate is kept only for
continuity.</b></p>

{verdict_html}

<h2>1. In-task PFN Bayes regret (corrected re-runs, 3 seeds each)</h2>
{_html_table(it_rows, ["eps", "seeds", "mixed regret (mean)", "mixed SE",
                       "<b>in-task regret (mean)</b>", "in-task SE",
                       "in-task min-max", "in-task LC final", "LC@100k trend",
                       "ident Q1 width", "CLF KL (s0, half-budget)"])}
<p>In-task = queries with target=d-1 (the column the PFN is trained to predict;
n = in_task_frac of the 120-query fixed panel). CLF KL is the ORIGINAL
half-budget (50k) classifier, not re-run in the corrected fleet. eps0.5 was
skipped in the intask fleet by design (least-pinned ident point).</p>
<div class="fig">{fig_lc}</div>

<h2>2. Per-seed detail</h2>
{_html_table(per_seed_rows, ["eps", "seed", "in-task regret", "mixed regret",
                             "final train loss", "in-task frac"])}

<h2>3. Mixed-panel vs in-task regret</h2>
<div class="fig">{fig_mixed}</div>
<p>The mixed-panel regret averages targets {0,1,2} ~uniformly while the PFN has
no target-index input, so ~2/3 of the mixed panel is out-of-task. If the two
columns diverge, the original 'bayes_regret' was not measuring the trained task.</p>

{n10_html}

<h2>5. Tomography (incl. eps0p0 placeholder flag)</h2>
{_html_table(tom_rows, ["eps", "tol", "status", "Q1 width", "Q2 width",
                        "Q3 width", "struct TV Q1", "struct TV Q3"])}
<p class="warn">eps0p0 tomography ran at tol=0.093 (93x the exact 1e-3), returns
id_avg_width 1.0 for all three panels and IDENTICAL structural JS/TV across
Q1/Q2/Q3 (0.4539 / 0.8333), Q1 cond ~3.5e11: it is a degenerate placeholder, not
a measurement, and is excluded from any structural read.</p>

<h2>6. Provenance &amp; audit disclosures</h2>
<ul style="text-align:left">
<li><b>Original trackb reporting was single-seed</b> (s0 in the first-pass
report); s0 was the HIGHEST of the 3 seeds at eps1.0 (0.661 vs mean 0.638). The
corrected fleet reports all 3 seeds.</li>
<li><b>No code/version pinning</b>: training determinism across the original runs
is unrecorded (no torch/CUDA pin). Corrected re-runs are "same recipe and seed",
not bit-identical, so per-target decomposition on a re-trained model is the
quantity of interest; the mixed aggregate should match the original within
training noise.</li>
<li><b>Classifier at half budget</b> (50k vs PFN 100k).</li>
<li><b>Solver provenance</b>: eps0.5 ident ran under HiGHS ipm + crossover:off
after the LP degenerate-cycling failure; solver-invariance validated per
instance. The nrows10 eps0.5 file predates this and lacks the tag.</li>
<li><b>eps0.5 skipped</b> in the intask fleet (least-pinned ident point).</li>
<li><b>No in-task run for the classifier or the small/large scales</b>: the
fleet re-trained base-scale only, the disposition witness.</li>
</ul>

</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return html


def _intask_lc_svg(data: dict) -> str:
    """Multi-series in-task learning curve across the eps that have corrected data."""
    series = []
    for tag in IN_TASK_EPS:
        lc = data.get(tag, {}).get("trackb_corrected", {}).get("base_in_task_learning_curve", [])
        if lc:
            series.append((tag, lc))
    if not series:
        return "<p><em>In-task learning curve — no corrected data yet.</em></p>"
    xs_all = sorted({int(r["steps"]) for _, lc in series for r in lc})
    W, H, pad = 640, 260, 44
    all_v = [float(r["in_task_regret"]) for _, lc in series for r in lc]
    ymax = max(all_v) * 1.1 if all_v else 1
    ymin = 0
    n = max(len(xs_all), 1)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    def sx(i): return pad + i / (n - 1) * (W - 2 * pad) if n > 1 else W / 2
    def sy(v): return H - pad - (v - ymin) / (ymax - ymin) * (H - 2 * pad)
    polys = ""
    for idx, (tag, lc) in enumerate(series):
        by = {int(r["steps"]): float(r["in_task_regret"]) for r in lc}
        pts = " ".join(f"{sx(i):.1f},{sy(by[x]):.1f}"
                       for i, x in enumerate(xs_all) if by.get(x) is not None)
        polys += (f'<polyline points="{pts}" fill="none" stroke="{colors[idx % 4]}" '
                  f'stroke-width="2"/><text x="{sx(n-1)+2:.1f}" y="{sy(by[xs_all[-1]]):.1f}" '
                  f'font-size="9" fill="{colors[idx % 4]}">{tag}</text>')
    xticks = "".join(f'<text x="{sx(i):.1f}" y="{H-pad+12}" text-anchor="middle" font-size="9">{x}</text>'
                     for i, x in enumerate(xs_all))
    return (f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
            f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='#999'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{H-pad}' stroke='#999'/>"
            f"<text x='{W/2}' y='{H-6}' text-anchor='middle'>training steps</text>"
            f"<text x='12' y='{H/2}' text-anchor='middle' transform='rotate(-90,12,{H/2})'>in-task regret (nats)</text>"
            f"<text x='{W/2}' y='16' text-anchor='middle' font-weight='bold'>In-task Bayes regret vs training budget (seed-0 checkpoints)</text>"
            f"{xticks}{polys}</svg>")


def _mixed_vs_intask_svg(it_rows, eps_list) -> str:
    if not it_rows:
        return "<p><em>Mixed vs in-task — no corrected data yet.</em></p>"
    W, H, pad = 640, 260, 44
    rows = [r for r in it_rows]
    mixed = [float(r[2]) for r in rows]
    intask = [float(r[4]) for r in rows]
    ymax = max(mixed + intask) * 1.15 or 1
    ymin = 0
    n = max(len(rows), 1)
    def sx(i): return pad + i / (n - 1) * (W - 2 * pad) if n > 1 else W / 2
    def sy(v): return H - pad - (v - ymin) / (ymax - ymin) * (H - 2 * pad)
    mp = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(mixed))
    ip = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(intask))
    ticks = "".join(f'<text x="{sx(i):.1f}" y="{H-pad+12}" text-anchor="middle" font-size="9">{r[0]}</text>'
                    for i, r in enumerate(rows))
    return (f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
            f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='#999'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{H-pad}' stroke='#999'/>"
            f"<text x='{W/2}' y='{H-6}' text-anchor='middle'>eps</text>"
            f"<text x='12' y='{H/2}' text-anchor='middle' transform='rotate(-90,12,{H/2})'>regret (nats)</text>"
            f"<text x='{W/2}' y='16' text-anchor='middle' font-weight='bold'>Mixed-panel vs in-task regret (3-seed means)</text>"
            f"{ticks}"
            f'<polyline points="{mp}" fill="none" stroke="#999" stroke-width="2" stroke-dasharray="4,3"/>'
            f'<text x="{W-pad}" y="{sy(mixed[-1]):.1f}" font-size="9" fill="#666">mixed</text>'
            f'<polyline points="{ip}" fill="none" stroke="#d62728" stroke-width="2"/>'
            f'<text x="{W-pad}" y="{sy(intask[-1])+12:.1f}" font-size="9" fill="#d62728">in-task</text>'
            f"</svg>")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from .corrected_report import _load_raw
    data = _load_raw(args.raw)
    # nrows10 set: not matched by the report globs (filename-prefixed); load it
    # under <tag>_nrows10 so the disclosure table can read it.
    for f in sorted(args.raw.glob("nrows10_ident_eps*.json")):
        tag = f.stem.replace("nrows10_ident_", "").replace("eps", "")
        data[tag + "_nrows10"] = json.loads(f.read_text())
    build_honest_report(data, args.out)
    print(json.dumps({"done": True, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
