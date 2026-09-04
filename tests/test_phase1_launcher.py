import importlib.util
import json
from pathlib import Path
import re
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "cluster/submit_phase1_confirmation.py"
LAUNCHER_SHELL_PATH = ROOT / "cluster/submit_phase1_confirmation.sh"
WRAPPER_PATH = ROOT / "cluster/phase1_confirmation.sbatch"


def _launcher_module():
    spec = importlib.util.spec_from_file_location("phase1_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_freezes_resources_and_complete_dependency_dag(tmp_path):
    launcher = _launcher_module()
    attempt = (tmp_path / "attempt").resolve()
    plan = launcher.build_plan(ROOT.resolve(), attempt)
    assert [row["stage"] for row in plan] == [
        "panel",
        "pfn",
        "oracle0",
        "oracle1",
        "oracle2",
        "join",
        "verify",
    ]
    by_stage = {row["stage"]: row for row in plan}
    assert by_stage["panel"]["dependencies"] == []
    for stage in ("pfn", "oracle0", "oracle1", "oracle2"):
        assert by_stage[stage]["dependencies"] == ["panel"]
    assert by_stage["join"]["dependencies"] == [
        "pfn",
        "oracle0",
        "oracle1",
        "oracle2",
    ]
    assert by_stage["verify"]["dependencies"] == ["join"]
    for row in plan:
        arguments = row["sbatch_base_arguments"]
        assert arguments[0] == "/usr/bin/sbatch"
        assert "--hold" in arguments
        if row["stage"] == "verify":
            assert "--gres=gpu:a100:1" not in arguments
            assert "--cpus-per-task=2" in arguments
            assert "--mem=16G" in arguments
        else:
            assert "--gres=gpu:a100:1" in arguments
            assert "--cpus-per-task=8" in arguments
            assert "--mem=64G" in arguments
        assert "--partition=condo-cse5100" in arguments
        assert "--account=engr-acad-cse5100" in arguments
        assert "--export=NIL" in arguments
        assert "--no-requeue" in arguments
        assert Path(row["wrapper"]).is_absolute()
        assert row["wrapper_arguments"][0:2] == [
            "--source-root",
            str(ROOT.resolve()),
        ]
        assert Path(row["wrapper_arguments"][3]).is_absolute()
    output_paths = {
        argument
        for row in plan
        for argument in row["sbatch_base_arguments"]
        if argument.startswith("--output=")
    }
    assert len(output_paths) == 1


def test_shell_entrypoint_locks_the_production_launcher_interpreter():
    launcher = LAUNCHER_SHELL_PATH.read_text()
    assert launcher.startswith(
        "#!/usr/bin/env -S -i /bin/bash --noprofile --norc\n"
        "set -euo pipefail\numask 077\n"
    )
    assert LAUNCHER_SHELL_PATH.stat().st_mode & stat.S_IXUSR
    assert "/usr/bin/readlink -f" in launcher
    assert "/usr/bin/dirname" in launcher
    assert "exec /usr/bin/env -i" in launcher
    assert "SLURM_CONF=/project/compute/slurm/etc/slurm.conf" in launcher
    assert (
        "/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python -I -S -B" in launcher
    )
    assert '"${PHASE1_LAUNCHER_DIR}/submit_phase1_confirmation.py" "$@"' in launcher


def test_submission_receipt_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    launcher = _launcher_module()
    fsync_calls = []
    monkeypatch.setattr(launcher.os, "fsync", fsync_calls.append)
    path = tmp_path / "SUBMISSION.json"
    launcher._json_atomic(path, {"state": "SUBMITTING_HELD"})
    assert json.loads(path.read_text()) == {"state": "SUBMITTING_HELD"}
    assert len(fsync_calls) == 2


def test_launcher_clients_are_absolute_and_environment_is_allowlisted(monkeypatch):
    launcher = _launcher_module()
    assert launcher.LOCKED_PYTHON == Path(
        "/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"
    )
    assert launcher.GIT == "/usr/bin/git"
    assert launcher.SBATCH == "/usr/bin/sbatch"
    assert launcher.SCANCEL == "/usr/bin/scancel"
    assert launcher.SCONTROL == "/usr/bin/scontrol"
    assert launcher.SQUEUE == "/usr/bin/squeue"
    assert launcher.CLIENT_ENV == {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "TZ": "UTC",
        "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
    }
    assert not any(name.startswith("SBATCH_") for name in launcher.CLIENT_ENV)
    assert {name for name in launcher.CLIENT_ENV if name.startswith("SLURM_")} == {
        "SLURM_CONF"
    }

    observed = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher._run_client([launcher.SBATCH, "--version"])
    assert observed["arguments"] == ["/usr/bin/sbatch", "--version"]
    assert observed["env"] == launcher.CLIENT_ENV
    assert observed["check"] is True
    assert observed["capture_output"] is True


def test_all_jobs_remain_held_until_complete_submission_record_is_durable(
    tmp_path, monkeypatch
):
    launcher = _launcher_module()
    attempt = (tmp_path / "attempt").resolve()
    plan = launcher.build_plan(ROOT.resolve(), attempt)
    expected_stages = [row["stage"] for row in plan]
    submitted = []
    release_snapshot = {}

    def fake_run(arguments, *, cwd=None):
        assert cwd is None
        if arguments[0] == launcher.SBATCH:
            assert "--hold" in arguments
            stage = next(
                value.removeprefix("--job-name=p1-")
                for value in arguments
                if value.startswith("--job-name=p1-")
            )
            submitted.append(stage)
            job_id = str(41000 + len(submitted))
            return subprocess.CompletedProcess(
                arguments, 0, stdout=f"{job_id};washu\n", stderr=""
            )
        assert arguments[0:2] == [launcher.SCONTROL, "release"]
        release_snapshot.update(json.loads((attempt / "SUBMISSION.json").read_text()))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(launcher, "_run_client", fake_run)
    record = launcher.submit_plan(
        plan,
        attempt,
        {"commit": "a" * 40, "tag": "attempt", "config_sha256": "b" * 64},
    )

    assert submitted == expected_stages
    assert release_snapshot["state"] == "READY_TO_RELEASE"
    assert [row["stage"] for row in release_snapshot["jobs"]] == expected_stages
    assert len(release_snapshot["jobs"]) == 7
    assert all(row["submitted_held"] is True for row in release_snapshot["jobs"])
    job_ids = {row["stage"]: row["job_id"] for row in release_snapshot["jobs"]}
    assert release_snapshot["jobs"][-1]["dependencies"] == [job_ids["join"]]
    assert release_snapshot["release_arguments"] == [
        "/usr/bin/scontrol",
        "release",
        ",".join(job_ids[stage] for stage in expected_stages),
    ]
    assert record["state"] == "SUBMITTED"
    assert json.loads((attempt / "SUBMISSION.json").read_text()) == record


def test_release_interrupt_with_failed_cancel_is_recorded_unconfirmed(
    tmp_path, monkeypatch
):
    launcher = _launcher_module()
    attempt = (tmp_path / "attempt").resolve()
    plan = launcher.build_plan(ROOT.resolve(), attempt)
    submitted = []

    def fake_client(arguments, *, cwd=None):
        assert cwd is None
        if arguments[0] == launcher.SBATCH:
            submitted.append(arguments)
            return subprocess.CompletedProcess(
                arguments, 0, stdout=f"{42000 + len(submitted)};washu\n", stderr=""
            )
        raise KeyboardInterrupt

    def fake_unchecked(arguments, **kwargs):
        assert arguments[0] == launcher.SCANCEL
        assert kwargs["env"] == launcher.CLIENT_ENV
        return subprocess.CompletedProcess(
            arguments, 1, stdout="", stderr="injected cancellation failure"
        )

    monkeypatch.setattr(launcher, "_run_client", fake_client)
    monkeypatch.setattr(launcher.subprocess, "run", fake_unchecked)
    with pytest.raises(KeyboardInterrupt):
        launcher.submit_plan(
            plan,
            attempt,
            {"commit": "a" * 40, "tag": "attempt", "config_sha256": "b" * 64},
        )
    record = json.loads((attempt / "SUBMISSION.json").read_text())
    assert record["state"] == "SUBMISSION_FAILED_CANCEL_UNCONFIRMED"
    assert record["error_type"] == "KeyboardInterrupt"
    assert record["cancellation"]["confirmed_terminal"] is False
    assert record["cancellation"]["returncode"] == 1
    assert len(record["jobs"]) == 7


def test_cancellation_runs_even_if_cancelling_receipt_cannot_persist(
    tmp_path, monkeypatch
):
    launcher = _launcher_module()
    attempt = (tmp_path / "attempt").resolve()
    plan = launcher.build_plan(ROOT.resolve(), attempt)
    submitted = []

    def fake_client(arguments, *, cwd=None):
        assert cwd is None
        if arguments[0] == launcher.SBATCH:
            submitted.append(arguments)
            return subprocess.CompletedProcess(
                arguments, 0, stdout=f"{43000 + len(submitted)};washu\n", stderr=""
            )
        assert arguments[0:2] == [launcher.SCONTROL, "release"]
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    real_atomic = launcher._json_atomic

    def flaky_atomic(path, value):
        state = value["state"]
        if state == "SUBMITTED":
            real_atomic(path, value)
            raise OSError("injected post-rename fsync failure")
        if state == "CANCELLING":
            raise OSError("injected cancelling persistence failure")
        real_atomic(path, value)

    cancellation_calls = []

    def fake_cancel(job_ids):
        cancellation_calls.append(job_ids)
        return {"attempted": True, "confirmed_terminal": True, "jobs": job_ids}

    monkeypatch.setattr(launcher, "_run_client", fake_client)
    monkeypatch.setattr(launcher, "_json_atomic", flaky_atomic)
    monkeypatch.setattr(launcher, "_cancel_jobs", fake_cancel)
    with pytest.raises(OSError, match="post-rename"):
        launcher.submit_plan(
            plan,
            attempt,
            {"commit": "a" * 40, "tag": "attempt", "config_sha256": "b" * 64},
        )
    assert len(cancellation_calls) == 1
    assert len(cancellation_calls[0]) == 7
    record = json.loads((attempt / "SUBMISSION.json").read_text())
    assert record["state"] == "SUBMISSION_FAILED_CANCELLED"
    assert record["cancelling_persistence_error_type"] == "OSError"
    assert (attempt / "SUBMISSION.INVALID.json").is_file()


def test_cancel_is_confirmed_only_after_jobs_leave_the_queue(monkeypatch):
    launcher = _launcher_module()
    queue_polls = []

    def fake_run(arguments, **kwargs):
        assert kwargs["env"] == launcher.CLIENT_ENV
        if arguments[0] == launcher.SCANCEL:
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        assert arguments[0] == launcher.SQUEUE
        queue_polls.append(arguments)
        if len(queue_polls) == 1:
            return subprocess.CompletedProcess(
                arguments, 0, stdout="42001|CG\n", stderr=""
            )
        return subprocess.CompletedProcess(
            arguments,
            1,
            stdout="",
            stderr="slurm_load_jobs error: Invalid job id specified\n",
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    result = launcher._cancel_jobs(["42001"])
    assert result["confirmed_terminal"] is True
    assert result["confirmation_polls"] == 2


def test_dry_run_does_not_create_the_attempt_root(tmp_path):
    attempt = (tmp_path / "attempt").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--source-root",
            str(ROOT.resolve()),
            "--attempt-root",
            str(attempt),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"stage": "verify"' in completed.stdout
    assert not attempt.exists()


def test_plan_rejects_relative_and_in_source_attempt_roots(tmp_path):
    launcher = _launcher_module()
    try:
        launcher.build_plan(ROOT.resolve(), Path("relative"))
    except ValueError as error:
        assert "absolute" in str(error)
    else:
        raise AssertionError("relative attempt root was accepted")
    try:
        launcher.build_plan(ROOT.resolve(), ROOT / "attempt")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("in-source attempt root was accepted")


def test_wrapper_uses_isolated_no_site_python_and_derived_attempt_paths():
    wrapper = WRAPPER_PATH.read_text()
    assert (
        '"${PHASE1_PYTHON}" -I -S -X '
        '"pycache_prefix=${PHASE1_PYCACHE_PREFIX}" -c' in wrapper
    )
    assert '"site","sitecustomize","usercustomize"' in wrapper
    assert "unset LD_PRELOAD" in wrapper
    assert "PYTHONPATH:-" not in wrapper
    assert 'PHASE1_PANEL_DIR="${PHASE1_ATTEMPT_ROOT}/panel"' in wrapper
    assert "telemetry/${PHASE1_STAGE}-${SLURM_JOB_ID}" in wrapper
    assert 'PHASE1_PYCACHE_PREFIX="${PHASE1_LOG_DIR}/pycache"' in wrapper
    assert '/usr/bin/mkdir "${PHASE1_LOG_DIR}"' in wrapper
    assert 'state=="READY_TO_RELEASE"' in wrapper
    assert 'state=="SUBMITTED"' in wrapper
    assert 'row.get("stage")==stage' in wrapper
    assert 'str(rows[0].get("job_id"))==job' in wrapper
    assert '"${SLURM_JOB_ID}" "${PHASE1_PYCACHE_PREFIX}"' in wrapper
    assert wrapper.count("verify_submission_binding") == 3
    assert 'if [[ "${PHASE1_MODE}" != "verify" ]]' in wrapper
    assert "run_locked_module pip check" in wrapper
    assert "phase1_confirmation_verify" in wrapper
    config = json.loads((ROOT / "config/phase1_ordering_confirmation.json").read_text())
    tag_match = re.search(r"--attempt-tag\s+([^\s\\]+)", wrapper)
    assert tag_match is not None
    assert tag_match.group(1) == config["required_attempt_tag"]
    common = (ROOT / "src/pfn_dag_verify/phase1_confirm_common.py").read_text()
    assert '[sys.executable, "-m", "pip", "check"]' not in common


def test_wrapper_submission_binding_payload_is_valid_python():
    wrapper = WRAPPER_PATH.read_text()
    match = re.search(
        r"verify_submission_binding\(\) \{.*? -c \\\n\s+'(.*?)' \\\n",
        wrapper,
        flags=re.DOTALL,
    )
    assert match is not None
    compile(match.group(1), "submission-binding-payload", "exec")


def test_submission_requires_an_annotated_attempt_tag(tmp_path):
    launcher = _launcher_module()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    (tmp_path / "tracked").write_text("x\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(tmp_path), "tag", "lightweight", commit], check=True
    )
    with pytest.raises(RuntimeError, match="not annotated"):
        launcher._verify_annotated_tag(tmp_path, "lightweight", commit)
    subprocess.run(
        ["git", "-C", str(tmp_path), "tag", "-a", "annotated", "-m", "attempt", commit],
        check=True,
    )
    launcher._verify_annotated_tag(tmp_path, "annotated", commit)
