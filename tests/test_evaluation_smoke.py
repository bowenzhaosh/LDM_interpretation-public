from pathlib import Path

import numpy as np
import pytest

import pfn_dag_verify.evaluation as evaluation
from pfn_dag_verify.evaluation import generate_panel
from pfn_dag_verify.query_bank import FIXED_SENSITIVITY_BANK
from pfn_dag_verify.storage import load_numeric_npz


def test_tiny_panel_has_replay_complete_shapes(tmp_path: Path):
    banks = np.stack(
        [
            np.array([-4.0, -3.5, -3.0, -2.5, 2.5, 3.0, 3.5, 4.0]),
            FIXED_SENSITIVITY_BANK,
        ]
    )
    path = tmp_path / "panel.npz"
    metadata = generate_panel(
        commit_sha="0123456789abcdef" * 4,
        query_banks=banks,
        n_groups=2,
        n_continuations=2,
        out_path=path,
    )
    panel = load_numeric_npz(path)
    assert panel["core"].shape == (2, 20, 2)
    assert panel["reference"].shape == (2, 10, 2)
    assert panel["continuations"].shape == (2, 2, 10, 2)
    assert panel["f0_target"].shape == (2, 2, 2, 8, 100)
    assert panel["eligible_replace"].shape == (2, 2)
    assert metadata["panel_sha256"]


def test_direct_scientific_panel_api_fails_before_generation_when_lock_is_unverified(
    tmp_path: Path, monkeypatch
):
    banks = np.stack(
        [
            np.array([-4.0, -3.5, -3.0, -2.5, 2.5, 3.0, 3.5, 4.0]),
            FIXED_SENSITIVITY_BANK,
        ]
    )
    commit = "1" * 40
    monkeypatch.setattr(evaluation, "current_head", lambda root: commit)
    monkeypatch.setattr(
        evaluation,
        "verify_run_lock",
        lambda root: (_ for _ in ()).throw(RuntimeError("dirty or unlocked")),
    )
    out = tmp_path / "panel.npz"
    with pytest.raises(RuntimeError, match="dirty or unlocked"):
        generate_panel(
            commit_sha=commit,
            query_banks=banks,
            n_groups=256,
            n_continuations=8,
            out_path=out,
            interior_selected=True,
            scientific=True,
            run_lock_sha256="0" * 64,
        )
    assert not out.exists()
