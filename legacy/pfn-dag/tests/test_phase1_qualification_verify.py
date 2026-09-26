from pathlib import Path

import pytest

from pfn_dag_verify.phase1_qualification_verify import verify_qualification


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "campaigns" / "phase1_ordering_20260803" / "oracle_qualification"


def test_archived_qualification_recomputes_from_raw():
    result = verify_qualification(ARTIFACTS, ROOT)
    assert result["verification"] == "INDEPENDENT_RAW_RECOMPUTATION_PASS"
    assert result["decision"] == "QUALIFICATION_PASS"
    assert result["selected_truncation"] == 16384
    assert result["candidate_summary"]["8192"]["total_abs_logp_change_exceedances"] == 16
    assert result["candidate_summary"]["16384"]["total_abs_logp_change_exceedances"] == 0


def test_verifier_rejects_wrong_source_commit():
    with pytest.raises(RuntimeError, match="identity commit mismatch"):
        verify_qualification(ARTIFACTS, ROOT, "0" * 40)
