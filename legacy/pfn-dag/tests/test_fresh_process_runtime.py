import os
import subprocess
import sys
from pathlib import Path


def _run_fresh(code: str) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_derive_api_configures_runtime_before_verification_in_fresh_process():
    _run_fresh(
        """
from pathlib import Path
import torch
import pfn_dag_verify.analysis as analysis

def stop_after_asserting(**kwargs):
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.mha.get_fastpath_enabled()
    raise SystemExit(0)

analysis.verify_prediction_ledger = stop_after_asserting
analysis.derive_all(
    panel_path=Path("unused-panel"),
    prediction_dir=Path("unused-predictions"),
    out_dir=Path("unused-derived"),
)
"""
    )


def test_summarize_api_configures_runtime_before_verification_in_fresh_process():
    _run_fresh(
        """
from pathlib import Path
import torch
import pfn_dag_verify.analysis as analysis

def stop_after_asserting(path):
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.mha.get_fastpath_enabled()
    raise SystemExit(0)

analysis.load_numeric_npz = stop_after_asserting
analysis.summarize(run_dir=Path("unused-run"))
"""
    )


def test_legacy_compare_restores_locked_runtime_after_import_in_fresh_process():
    _run_fresh(
        """
import os
from pathlib import Path
import numpy as np
import torch
from pfn_dag_verify.legacy_compare import compare

os.environ["NTHREAD"] = "8"
result = compare(
    Path("artifacts/legacy/stage1_functional_law.py"),
    np.linspace(-4.0, 4.0, 8),
    n_contexts=1,
)
assert result["legacy_file"] == "artifacts/legacy/stage1_functional_law.py"
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
assert torch.are_deterministic_algorithms_enabled()
"""
    )
