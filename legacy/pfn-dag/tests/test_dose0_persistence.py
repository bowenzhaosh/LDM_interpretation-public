"""Amendment F.2c operational corollary — the dose-0 state must persist.

F.2c's y is (model - floor) / (ceiling - floor), and the floor is an untrained
checkpoint pushed through the same readout. Without a persisted dose-0 state
there is no floor, so no calibrated y, so no gate (c) at d=3.

The capture in ``train_pfn`` fires on ``(step + 1) in ckpt_steps``, which made
step 0 unreachable: requesting it was a silent no-op, so a fleet could report
having run with dose-0 and produce nothing. These tests exist so that cannot
recur quietly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pfn_dag_verify.corrected_models import ModelConfig, train_pfn
from pfn_dag_verify.corrected_world import make_world


def _tiny(tmp_path: Path, ckpt_steps):
    world = make_world(k=8, d=3, eps=1.0)      # eps=1 short-circuits: fast
    cfg = ModelConfig(name="t", d=3, d_model=16, d_ff=32, n_heads=2, n_layers=1)
    return train_pfn(world, steps=2, seed=0, cfg=cfg, n_ctx=6, n_query=2,
                     batch=4, ckpt_steps=ckpt_steps, outdir=tmp_path,
                     tag_prefix="t"), world


def test_dose0_is_captured_in_memory(tmp_path):
    res, _ = _tiny(tmp_path, (0, 2))
    assert 0 in res["checkpoints"], "step 0 (dose-0) was not captured"
    assert 2 in res["checkpoints"]


def test_dose0_reaches_disk(tmp_path):
    _tiny(tmp_path, (0, 2))
    assert (tmp_path / "t_ck0.pt").exists(), "no dose-0 checkpoint on disk"
    assert (tmp_path / "t_ck2.pt").exists()
    assert (tmp_path / "t.pt").exists()


def test_dose0_is_untrained(tmp_path):
    """The floor's whole meaning is that no gradient has touched it."""
    res, _ = _tiny(tmp_path, (0, 2))
    ck0, ck2 = res["checkpoints"][0], res["checkpoints"][2]
    same = all(torch.allclose(ck0[k], ck2[k]) for k in ck0)
    assert not same, "dose-0 and step-2 states are identical; training did nothing"
    # dose-0 must match a freshly seeded model, i.e. be genuinely pre-update
    torch.manual_seed(1000 * 0 + 7)
    from pfn_dag_verify.corrected_models import PFN, DEV
    fresh = PFN(ModelConfig(name="t", d=3, d_model=16, d_ff=32, n_heads=2, n_layers=1)).to(DEV)
    fs = {k: v.detach().cpu() for k, v in fresh.state_dict().items()}
    assert all(torch.allclose(ck0[k], fs[k]) for k in ck0), (
        "the dose-0 state is not the freshly initialised model")


def test_provenance_sidecar_pins_the_stack(tmp_path):
    _tiny(tmp_path, (0, 2))
    prov = json.loads((tmp_path / "t.provenance.json").read_text())
    for key in ("torch", "cuda", "device", "numpy", "amendment"):
        assert key in prov, f"provenance sidecar has no {key}"
