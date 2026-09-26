"""Self-contained HTML report for the oracle-precision pilot result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def render(verification: dict, joined_path: Path, out_path: Path) -> None:
    gates = verification.get("gates", {})
    g = gates.get("gates", {})
    rows = []
    rows.append(f"<h1>Oracle-precision pilot — result</h1>")
    rows.append(f"<p>Rows: {verification.get('n_rows')}. "
                f"Verdict: <b>{'ALL GATES PASS' if gates.get('all_pass') else 'GATES FAIL'}</b></p>")
    rows.append("<table><tr><th>Gate</th><th>Pass</th></tr>")
    for k, v in g.items():
        rows.append(f"<tr><td>{k}</td><td>{'PASS' if v else 'FAIL'}</td></tr>")
    rows.append("</table>")
    for pname in ("C", "N"):
        rows.append(f"<h3>Prior {pname}</h3><table>")
        for metric in ("smc_mcmc_nll_full_median_abs", "smc_mcmc_nll_ablated_median_abs",
                       "smc_mcmc_nll_full_max_abs", "smc_mcmc_nll_ablated_max_abs",
                       "order_js_median", "order_js_p95", "row_catastrophe_count"):
            v = gates.get(metric, {}).get(pname)
            rows.append(f"<tr><td>{metric}</td><td>{v}</td></tr>")
        rows.append("</table>")
    with np.load(joined_path, allow_pickle=False) as z:
        if "smc_logZ" in z.files:
            smc_logz = z["smc_logZ"]
            rows.append(f"<h3>SMC order logZ summary</h3>")
            rows.append(f"<p>mean {smc_logz.mean():.2f}, std {smc_logz.std():.2f}</p>")
    (out_path).write_text(
        "<html><body>" + "\n".join(rows) + "</body></html>")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    verification = json.loads(a.verification.read_text())
    render(verification, a.joined, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
