#!/usr/bin/env python
"""Shared read-modify-write for paper/figs/figure_numbers.json.

Why this exists: `scripts/figures_iclr.py` used to *overwrite* the file wholesale
(`fp.write_text(json.dumps({... "figures": NUMBERS ...}))`), so any figure block
written by another script was silently dropped on its next run. That already
happened once — the file checked in at 2026-09-03T02:03 held only A8, F4 and F6,
because a `--only 4,A8` run of `figures_iclr.py` clobbered the F1/F2/F3/F5 blocks
and `fig_component_control.py` then merged F6 back in.

`merge_figure_numbers()` is now the only writer. It preserves every block it did
not just render, and it tracks the draft watermark **per figure** instead of the
old sticky top-level `"draft": doc.get("draft", True) or draft`, which could never
go back to False once it had been True.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["merge_figure_numbers", "load_figure_numbers"]


def load_figure_numbers(path) -> dict:
    """Return the document at `path`, or an empty skeleton if it is absent/corrupt."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def merge_figure_numbers(path, blocks: dict, draft: bool, generated: str | None = None) -> dict:
    """Merge `blocks` ({fig_id: {panel: {...}}}) into the JSON at `path` and write it back.

    Only the figure ids present in `blocks` are replaced; every other block is
    carried through untouched. Returns the written document.
    """
    p = Path(path)
    doc = load_figure_numbers(p)

    figures = doc.get("figures")
    if not isinstance(figures, dict):
        figures = {}

    per_fig = doc.get("draft_by_figure")
    if not isinstance(per_fig, dict):
        # Migration from the old scalar flag: attribute it to everything already on disk.
        per_fig = {k: bool(doc.get("draft", False)) for k in figures}

    for fig_id, block in blocks.items():
        figures[fig_id] = block
        per_fig[fig_id] = bool(draft)
    per_fig = {k: bool(v) for k, v in sorted(per_fig.items()) if k in figures}

    out = {
        "generated": generated or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "draft": any(per_fig.values()),
        "draft_by_figure": per_fig,
        "figures": figures,
    }
    for k, v in doc.items():          # keep any top-level key a future writer adds
        out.setdefault(k, v)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    return out
