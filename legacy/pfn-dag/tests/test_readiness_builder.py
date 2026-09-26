import hashlib
from types import SimpleNamespace

import pytest

import pfn_dag_verify.readiness_builder as readiness


def test_readiness_runs_pytest_and_derives_the_pass_count(monkeypatch, tmp_path):
    collect_output = "tests/test_one.py::test_one\n52 tests collected in 0.10s\n"
    run_output = "................................  [100%]\n52 passed in 5.00s\n"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        output = collect_output if "--collect-only" in command else run_output
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setenv("PYTEST_ADDOPTS", "-k one_test")
    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    collected, passed, digest = readiness._run_tests(tmp_path)
    assert collected == passed == 52
    assert calls[0][2:] == ["pytest", "--collect-only", "-q", "tests"]
    assert calls[1][2:] == ["pytest", "-q", "tests"]
    combined = collect_output + "\n--- pytest execution ---\n" + run_output
    assert digest == hashlib.sha256(combined.encode()).hexdigest()


def test_readiness_rejects_a_failed_test_process(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="collection failed", stderr=""))
    with pytest.raises(RuntimeError, match="collection failed"):
        readiness._run_tests(tmp_path)
