"""Amendment F.8 — estimand and status tags on every result artifact.

Two of the errors this campaign has paid for share one root: a RETIRED ARTIFACT
was still consumable.

  * A power analysis was computed on ``trackb_eps*.json``, the mixed-target
    panel E.3 retired, and reported as though it described the in-task gate. The
    ratios came out about threefold inflated and the context-count projection
    about fourfold wrong.
  * ``cells_from_raw`` globbed ``ident_eps*.json`` in whatever directory it was
    handed, and ``raw/`` still holds the widths E.2 retired under exactly those
    filenames, with one solver per tag so nothing raised.
  * ``nrows10_ident_eps*.json`` was the same disease with the opposite sign: a
    live artifact that no loader could see, because the glob happened not to
    match it.

The retired-CONSTANT tripwire stops a dead number reaching a document. This
stops a dead FILE reaching a computation. A loader that filters on status cannot
silently consume a retired artifact any more than an amendment can silently
quote a retired constant.

Discipline: writers stamp, readers require. An artifact with no tag is UNKNOWN,
not live, so everything written before this amendment fails closed by default.
"""
from __future__ import annotations

from typing import Any

LIVE = "live"              # current instrument, citable
RETIRED = "retired"        # superseded; kept for the record, never computed on
DIAGNOSTIC = "diagnostic"  # an attribution or instrument probe, never a headline
UNKNOWN = "unknown"        # no tag: pre-F.8, treated as not-live

STATUSES = (LIVE, RETIRED, DIAGNOSTIC, UNKNOWN)

# Estimands this campaign distinguishes. The mixed-target/in-task split is the
# one that cost us a power analysis, so it is named explicitly rather than left
# to a filename convention.
EST_IDENT_WIDTH = "ident_q1_avg_width"
EST_ORDER_EVIDENCE = "normalized_order_evidence"     # F.2b primary axis
EST_REGRET_INTASK = "in_task_expected_regret"
EST_REGRET_MIXED = "mixed_target_bayes_regret"       # retired by E.3
EST_FIDELITY_RAW = "readout_tracking_slope_raw"
EST_FIDELITY_NORM = "readout_tracking_fraction_of_achievable"   # F.2c
EST_C2_ETA = "c2_matched_pair_efficiency_eta"                   # E.7 / D7 (R6)


def stamp(doc: dict[str, Any], estimand: str, status: str = LIVE,
          amendment: str = "F") -> dict[str, Any]:
    """Attach the tag block. Writers call this; it mutates and returns doc."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    doc["artifact"] = {"estimand": estimand, "status": status,
                       "amendment": amendment, "schema": 1}
    return doc


def status_of(doc: dict[str, Any]) -> str:
    return str(doc.get("artifact", {}).get("status", UNKNOWN))


def estimand_of(doc: dict[str, Any]) -> str | None:
    return doc.get("artifact", {}).get("estimand")


def require_live(doc: dict[str, Any], estimand: str, where: str) -> dict[str, Any]:
    """Readers call this. Refuses anything that is not a live artifact of the
    estimand asked for, and says which of the two went wrong.

    Failing closed on UNKNOWN is deliberate: every artifact written before F.8
    is untagged, and those are exactly the files that caused the incidents this
    amendment exists to prevent.
    """
    st = status_of(doc)
    if st != LIVE:
        raise ValueError(
            f"{where}: artifact status is {st!r}, not {LIVE!r}. "
            + ("It carries no F.8 tag, so it predates the amendment and may be "
               "a retired instrument's output; regenerate it or point at a "
               "post-F directory." if st == UNKNOWN else
               "Retired and diagnostic artifacts are kept for the record and "
               "are never computed on."))
    got = estimand_of(doc)
    if got != estimand:
        raise ValueError(
            f"{where}: artifact is {got!r} but {estimand!r} was required. "
            "The mixed-target and in-task regret series are the pair that has "
            "already been confused once; they are different estimands.")
    return doc
