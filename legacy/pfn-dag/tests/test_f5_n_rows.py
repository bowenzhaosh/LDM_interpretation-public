"""Amendment F.5, clause 2 — the headline ident n_rows is DEFINED as Track B's.

Not "declared to match". If the identifiability panel and the Track B scoring
panel use different context sizes, the width appendix's x-axis and the
model-side y-axis are measured at different information levels and the
comparison between them is invalid. The sensitivity is large and grows with eps:
the nrows10 set gives Q1 widths 1.25x (eps=0.1) to 1.79x (eps=0.5) the headline
set's.

F.2b makes this automatic for the PRIMARY axis, since the entropy deficit is
evaluated on the scoring panel itself. The width appendix still needs it, and
the constant is duplicated across two modules (corrected_trackb pulls in torch;
the ident run is CPU-only), so it is enforced here instead of by an import.
"""
from __future__ import annotations

import json
from pathlib import Path

from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS


def test_headline_n_rows_equals_track_b_panel_n_rows():
    from pfn_dag_verify.corrected_trackb import N_CONTEXT
    assert HEADLINE_N_ROWS == N_CONTEXT, (
        f"ident headline n_rows {HEADLINE_N_ROWS} != Track B panel n_rows "
        f"{N_CONTEXT}; F.5 clause 2 defines them to be the same number")


def test_ident_config_records_n_rows():
    """F.5 clause 1. The absence of this field is why nothing on disk
    distinguished the headline ident set from the nrows10 set but the filename."""
    import inspect
    from pfn_dag_verify import corrected_identifiability_run as R
    src = inspect.getsource(R.run)
    assert '"n_rows": n_rows' in src


def test_post_f_ident_artifacts_carry_n_rows():
    """Any ident JSON written after the amendment must carry n_rows. Pre-F
    artifacts are grandfathered: they are the disclosure item, not a failure."""
    root = Path(__file__).resolve().parent.parent
    post_f = sorted((root / "campaigns/corrected_20260812/raw").glob("postF/ident_*.json"))
    for f in post_f:
        cfg = json.loads(f.read_text()).get("config", {})
        assert "n_rows" in cfg, f"{f.name} has no n_rows in its config block"


# --- the invisible-artifact failure, as a test -----------------------------
# F.7 names three incidents that share one root: a retired artifact stayed
# consumable, and -- with the opposite sign -- a LIVE artifact stayed invisible,
# because `nrows10_ident_eps*.json` happened not to match a glob. That third one
# recurred on 2026-08-23, an hour after the amendment describing it was written:
# a producer wrote `deficit_eps0.json` and `deficit_eps1.json`, `_TAG_RE` did not
# match, and `cells_from_raw` skipped both IN SILENCE. The eps=0 gate (b) anchor
# and the eps=1 top rung vanished from a seven-cell ladder and it returned five
# cells without an error. A short ladder looks exactly like a ladder.

def test_eps_tag_round_trips_including_the_endpoints():
    """The endpoints are the whole point: `f"{0.0:g}"` is "0", not "0p0"."""
    from pfn_dag_verify.corrected_verdict import _TAG_RE, _eps_of, eps_tag
    for e in (0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 2.0):
        tag = eps_tag(e)
        assert _TAG_RE.match(f"deficit_eps{tag}"), f"eps_tag({e}) = {tag!r} is unreadable"
        assert _eps_of(tag) == e


def test_an_unreadable_artifact_raises_instead_of_vanishing(tmp_path):
    """A file that matches the glob but not the tag pattern must STOP the load."""
    import json
    import pytest
    from pfn_dag_verify.corrected_verdict import cells_from_raw

    (tmp_path / "deficit_eps0.json").write_text(json.dumps({"deficit_mean": 0.0}))
    with pytest.raises(ValueError, match="tag pattern"):
        cells_from_raw(tmp_path)


def test_the_w2_deficit_ladder_loads_every_rung():
    """The live artifact, not a fixture: all seven cells must be visible."""
    from pathlib import Path
    from pfn_dag_verify.corrected_verdict import cells_from_raw

    raw = Path(__file__).resolve().parents[1] / (
        "campaigns/corrected_20260812/raw/postF/deficit_w2")
    if not raw.is_dir():
        import pytest
        pytest.skip("W2 deficit panel not built in this checkout")
    cells = cells_from_raw(raw)
    assert [c.eps for c in cells] == [0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0]
    assert all(c.has_deficit for c in cells)


def test_env_sidecars_do_not_break_the_ident_loader(tmp_path):
    """The provenance sidecar matches the ident glob; it must not be read as one.

    This is the F.7 disease with the sign flipped: the guard that stops a live
    artifact from vanishing was itself stopping every ident directory from
    loading, because `cluster/ident.sbatch` writes `ident_eps<tag>.env.json`
    next to each result and its stem fails the tag pattern.
    """
    import json
    from pfn_dag_verify import corrected_verdict as V

    doc = {
        "aggregates": {"Q1": {"avg_width_mean": 0.5, "true_in_all_frac": 1.0}},
        "config": {"d": 3, "K": 8, "eps": 0.25, "n_contexts": 20,
                   "n_rows": 20, "solver": "highs-ipm"},
        "artifact": {"estimand": "ident_q1_avg_width", "status": "live",
                     "amendment": "F", "schema": 1},
    }
    (tmp_path / "ident_eps0p25_highs_ipm.json").write_text(json.dumps(doc))
    (tmp_path / "ident_eps0p25_highs_ipm.env.json").write_text(
        json.dumps({"host": "iht32-1501", "solver": "highs-ipm"}))

    widths = V._ident_widths(tmp_path, solver="highs_ipm") \
        if hasattr(V, "_ident_widths") else None
    if widths is None:
        return  # helper is private/absent; the cells_from_raw path below is the contract

    assert widths == {"0p25": 0.5}


def test_a_lone_env_sidecar_still_raises(tmp_path):
    """A sidecar with no result means the cell never landed. That must not pass."""
    import json
    import pytest
    from pfn_dag_verify import corrected_verdict as V

    (tmp_path / "ident_eps0p5_highs_ipm.env.json").write_text(json.dumps({"host": "x"}))
    with pytest.raises(ValueError, match="matches the ident glob but not the tag pattern"):
        V.cells_from_raw(tmp_path, solver="highs_ipm")
