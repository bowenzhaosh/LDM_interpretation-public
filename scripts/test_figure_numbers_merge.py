#!/usr/bin/env python
"""Regression test: `figures_iclr.py` must MERGE figure_numbers.json, not overwrite it.

The bug this pins down (area-chair review; review-claims-audit.md G): figures_iclr.py
ended with

    (OUT / "figure_numbers.json").write_text(json.dumps({... "figures": NUMBERS ...}))

so any partial run dropped every block it had not just rendered. It had already
happened: the file committed at 2026-09-03T02:03 contained only A8, F4 and F6 —
F1, F2, F3 and F5 were gone.

Two levels:
  * unit      — merge_figure_numbers() preserves a foreign block byte-for-byte.
  * end-to-end — run the REAL scripts into a temp dir: fig_component_control.py
                 writes F6, then `figures_iclr.py --only 1` runs, and F6 must
                 still be there afterwards with its content unchanged.

Run:  .venv/bin/python scripts/test_figure_numbers_merge.py     (exit 0 = pass)
  or: .venv/bin/python -m pytest scripts/test_figure_numbers_merge.py -q
(The repo's pytest testpaths is ["tests"], and tests/ is sealed in
.claude/protected.sha256, so this test lives in scripts/ and is self-running.)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_numbers_io import merge_figure_numbers  # noqa: E402

PY = ROOT / ".venv/bin/python"

F6_FIXTURE = {
    "a": {"atom_family_b_median": {"value": 0.1234, "source": "component_control.json"}},
    "c": {"per_seed_delta_positive": {"value": "51/51", "source": "component_control.json"}},
    "meta": {"status": {"value": "REPORTED_NEVER_GATED", "source": "component_control.json"}},
}


def test_merge_preserves_foreign_block(tmp_path: Path | None = None) -> None:
    d = Path(tmp_path or tempfile.mkdtemp()); d.mkdir(parents=True, exist_ok=True)
    fp = d / "figure_numbers.json"
    fp.write_text(json.dumps({"generated": "2026-09-03T02:03:49-0500", "draft": True,
                              "figures": {"F6": F6_FIXTURE}}, indent=1))

    doc = merge_figure_numbers(fp, {"F1": {"a": {"k": {"value": 8, "source": "F1a-1"}}}}, draft=False)

    assert sorted(doc["figures"]) == ["F1", "F6"], doc["figures"].keys()
    assert doc["figures"]["F6"] == F6_FIXTURE, "F6 block was mutated by a merge that did not touch it"
    # per-figure draft flags: F6 keeps the watermark it was rendered with, F1 does not get one
    assert doc["draft_by_figure"] == {"F1": False, "F6": True}, doc["draft_by_figure"]
    assert doc["draft"] is True, "top-level draft must be any(per-figure)"

    # ...and re-rendering F6 without a watermark must be able to clear the flag,
    # which the old sticky `doc.get("draft", True) or draft` could never do.
    doc = merge_figure_numbers(fp, {"F6": F6_FIXTURE}, draft=False)
    assert doc["draft"] is False, doc["draft_by_figure"]
    assert sorted(doc["figures"]) == ["F1", "F6"]
    print("PASS  unit: merge preserves the foreign F6 block and un-sticks the draft flag")


def test_figures_iclr_partial_run_keeps_f6(tmp_path: Path | None = None) -> None:
    """End-to-end, against the real scripts and the real artifacts."""
    d = Path(tmp_path or tempfile.mkdtemp()); d.mkdir(parents=True, exist_ok=True)
    fp = d / "figure_numbers.json"

    r = subprocess.run([str(PY), str(ROOT / "scripts/fig_component_control.py"), "--out", str(d)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    f6_before = json.loads(fp.read_text())["figures"]["F6"]
    assert f6_before, "fig_component_control.py wrote no F6 block"

    r = subprocess.run([str(PY), str(ROOT / "scripts/figures_iclr.py"), "--only", "1", "--out", str(d)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    doc = json.loads(fp.read_text())
    assert "F6" in doc["figures"], "REGRESSION: figures_iclr.py dropped the F6 block"
    assert doc["figures"]["F6"] == f6_before, "REGRESSION: figures_iclr.py altered the F6 block"
    assert "F1" in doc["figures"], "figures_iclr.py wrote no F1 block"
    assert doc["figures"]["F1"]["a"]["schematic_annotations"]["value"]["latent_states"] == 48
    assert doc["draft"] is False, doc["draft_by_figure"]
    print(f"PASS  end-to-end: after `figures_iclr.py --only 1`, figures = {sorted(doc['figures'])}, "
          f"F6 block intact ({len(json.dumps(f6_before))} bytes)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        test_merge_preserves_foreign_block(Path(t) / "u")
    with tempfile.TemporaryDirectory() as t:
        test_figures_iclr_partial_run_keeps_f6(Path(t) / "e")
    print("2/2 PASS")
