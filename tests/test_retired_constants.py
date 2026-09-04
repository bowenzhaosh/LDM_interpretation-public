"""Amendment F.7 — the retired-constant tripwire.

Two retired-number regressions reached the live preregistration precisely
because nothing mechanical objected: F.2 quoted panel SEs from the retired
mixed-target panel, and F.5 used the retired ladder as an unlabelled baseline.
The fix is not another careful read. It is a test that fails when a retired
number appears outside a file that legitimately records history, so the next
one has to be added deliberately.

Adding a file to ALLOW is a deliberate act. Adding a number to RETIRED is how a
result gets retired.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# value -> what it was. Written as the literal digit strings a reader would type.
RETIRED = {
    "0.4510": "pre-D1 Q1 width at eps=0.1 (now 0.437820)",
    "0.3232": "pre-D1 Q1 width at eps=0.25 (now 0.316973)",
    "0.2073": "pre-D1 Q1 width at eps=0.5 (now 0.260840)",
    "-0.0058": "retired in-task regret at eps=0",
    "-0.0093": "retired in-task regret at eps=0.1",
    "0.0114": "retired in-task regret at eps=0.25",
    "0.0124": "retired in-task regret at eps=1.0",
    "0.2722": "retired classifier MAP at eps=0",
    "0.2778": "retired classifier MAP at eps=0.25",
    "0.8722": "retired classifier MAP at eps=1.0",
    # NOT ".0009" on its own: that collides with an unrelated live threshold
    # (phase1_qualification_verify's js_strict_p95_max). The retired thing is
    # the honest report's claim that this WAS the regret SE, so pin the phrase.
    "se .0009": "the across-seed SE the honest report quoted as if it were the panel SE",
}

# Files that legitimately record what was believed when. Each entry is a repo
# relative path or a directory prefix, with the reason it is exempt.
ALLOW = {
    "AMENDMENT_E.md": "records the pre-fix ladder ON PURPOSE so the regeneration is a comparison",
    "AMENDMENT_F.md": "records what E retired, labelled as retired",
    "campaigns/corrected_20260812/raw/": "the artifacts themselves",
    "campaigns/binary_duel_20260813/": "vendored historical results",
    "campaigns/binary_duel_ext3_20260821/": "vendored historical results",
    "tests/test_retired_constants.py": "this file, which must name them to forbid them",
    "scripts/compare_ident_envs.py": "labelled PRE_FIX_Q1",
    "scripts/f4_attribution_2x2.py": "the F.4 diagnostic whose SUBJECT is that number",
    "tests/test_verdict_power.py": "names them in a comment recording the repoint",
}

SEARCH_SUFFIXES = {".py", ".md", ".html", ".sbatch", ".sh", ".txt"}


def _allowed(rel: str) -> bool:
    return any(rel == a or rel.startswith(a) for a in ALLOW)


def _candidates():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in SEARCH_SUFFIXES:
            continue
        rel = str(p.relative_to(ROOT))
        if rel.startswith(".git/") or "__pycache__" in rel:
            continue
        yield p, rel


@pytest.mark.parametrize("value", sorted(RETIRED))
def test_retired_constant_appears_only_where_history_is_recorded(value):
    # word-boundary-ish: the digits must not be part of a longer number, which
    # is what makes gain_spec_mean 0.20738333 a false positive rather than a hit
    pat = re.compile(re.escape(value) + r"(?!\d)")
    offenders = []
    for p, rel in _candidates():
        if _allowed(rel):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, (
        f"retired constant {value} ({RETIRED[value]}) appears outside the "
        f"history allow-list:\n  " + "\n  ".join(offenders[:10]))


def test_the_allow_list_entries_still_exist():
    """An allow-list that has rotted is an allow-list that stops protecting."""
    missing = [a for a in ALLOW if not (ROOT / a).exists()]
    assert not missing, f"allow-list references paths that no longer exist: {missing}"
