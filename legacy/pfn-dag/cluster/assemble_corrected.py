#!/usr/bin/env python3
"""Assemble the corrected-campaign report + verdict from harvested results.

Usage: python3 cluster/assemble_corrected.py [--open]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "campaigns/corrected_20260812/raw"
OUT = REPO / "campaigns/corrected_20260812/report/corrected_report.html"
VERDICT = REPO / "campaigns/corrected_20260812/raw/verdict.json"


HONEST_OUT = REPO / "campaigns/corrected_20260812/report/corrected_honest_report.html"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--open", action="store_true")
    p.add_argument("--honest", action="store_true",
                   help="honest-disposition report (in-task regret + disclosures)")
    args = p.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from pfn_dag_verify.corrected_report import _load_raw
    if args.honest:
        from pfn_dag_verify.corrected_honest_report import build_honest_report
        from pfn_dag_verify.corrected_verdict import (
            apply_written_default, cells_from_raw, classify)
        data = _load_raw(RAW)
        for f in sorted(RAW.glob("nrows10_ident_eps*.json")):
            tag = f.stem.replace("nrows10_ident_", "").replace("eps", "")
            data[tag + "_nrows10"] = json.loads(f.read_text())
        html = build_honest_report(data, HONEST_OUT)
        # Amendment E.4: the verdict is computed from the per-eps cell tuple, not
        # from the report's display dict.
        res = apply_written_default(classify(cells_from_raw(RAW)),
                                    date.today().isoformat())
        out = HONEST_OUT
    else:
        from pfn_dag_verify.corrected_report import build_report
        from pfn_dag_verify.corrected_verdict import (
            apply_written_default, cells_from_raw, classify)
        data = _load_raw(RAW)
        html = build_report(data, OUT)
        res = apply_written_default(classify(cells_from_raw(RAW)),
                                    date.today().isoformat())
        out = OUT
    VERDICT.write_text(json.dumps(res, indent=2))
    print(f"report -> {out}")
    print(f"verdict: {res['verdict']}")
    for e in res["evidence"][:8]:
        print(f"  - {e}")
    if args.open:
        subprocess.run(["open", str(out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
