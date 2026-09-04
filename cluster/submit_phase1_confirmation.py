"""Submit the frozen Phase-1 confirmation DAG from an isolated interpreter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


LOCKED_PYTHON = Path("/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python")
GIT = "/usr/bin/git"
SBATCH = "/usr/bin/sbatch"
SCANCEL = "/usr/bin/scancel"
SCONTROL = "/usr/bin/scontrol"
SQUEUE = "/usr/bin/squeue"
CLIENT_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "TZ": "UTC",
    "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
}
PARTITION = "condo-cse5100"
ACCOUNT = "engr-acad-cse5100"
QOS = "normal"
GPU = "gpu:a100:1"
CPUS = 8
MEMORY = "64G"
STAGE_LIMITS = {
    "panel": "04:00:00",
    "pfn": "04:00:00",
    "oracle0": "24:00:00",
    "oracle1": "24:00:00",
    "oracle2": "24:00:00",
    "join": "08:00:00",
    "verify": "02:00:00",
}


def _fsync_directory(path: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _quarantine_receipt(path: Path) -> str | None:
    if not path.exists():
        return None
    invalid = path.with_name("SUBMISSION.INVALID.json")
    if invalid.exists():
        raise FileExistsError(
            f"quarantined submission receipt already exists: {invalid}"
        )
    os.replace(path, invalid)
    _fsync_directory(path.parent)
    return invalid.name


def _require_safe_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    raw = str(path)
    if "," in raw or "\n" in raw or "\r" in raw:
        raise ValueError(f"{label} contains a character unsafe for Slurm")
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _wrapper_arguments(source_root: Path, attempt_root: Path, stage: str) -> list[str]:
    mode = "oracle" if stage.startswith("oracle") else stage
    arguments = [
        "--source-root",
        str(source_root),
        "--attempt-root",
        str(attempt_root),
        "--mode",
        mode,
    ]
    if stage.startswith("oracle"):
        arguments.extend(["--bank", stage.removeprefix("oracle")])
    return arguments


def build_plan(source_root: Path, attempt_root: Path) -> list[dict[str, Any]]:
    source_root = _require_safe_absolute(source_root, "source root")
    attempt_root = _require_safe_absolute(attempt_root, "attempt root")
    if _is_within(attempt_root, source_root):
        raise ValueError("attempt root must be outside the source checkout")
    dependencies = {
        "panel": [],
        "pfn": ["panel"],
        "oracle0": ["panel"],
        "oracle1": ["panel"],
        "oracle2": ["panel"],
        "join": ["pfn", "oracle0", "oracle1", "oracle2"],
        "verify": ["join"],
    }
    stages = [
        "panel",
        "pfn",
        "oracle0",
        "oracle1",
        "oracle2",
        "join",
        "verify",
    ]
    wrapper = source_root / "cluster/phase1_confirmation.sbatch"
    slurm_dir = attempt_root / "slurm"
    plan = []
    for stage in stages:
        arguments = [
            SBATCH,
            "--parsable",
            "--hold",
            f"--job-name=p1-{stage}",
            f"--partition={PARTITION}",
            f"--account={ACCOUNT}",
            f"--qos={QOS}",
            "--nodes=1",
            "--ntasks=1",
            f"--time={STAGE_LIMITS[stage]}",
            "--kill-on-invalid-dep=yes",
            "--no-requeue",
            "--open-mode=append",
            f"--chdir={source_root}",
            f"--output={slurm_dir}/%x-%j.out",
            f"--error={slurm_dir}/%x-%j.err",
            "--export=NIL",
        ]
        if stage == "verify":
            arguments.extend(["--cpus-per-task=2", "--mem=16G"])
        else:
            arguments.extend(
                [f"--cpus-per-task={CPUS}", f"--mem={MEMORY}", f"--gres={GPU}"]
            )
        plan.append(
            {
                "stage": stage,
                "dependencies": dependencies[stage],
                "sbatch_base_arguments": arguments,
                "wrapper": str(wrapper),
                "wrapper_arguments": _wrapper_arguments(
                    source_root, attempt_root, stage
                ),
            }
        )
    return plan


def _run_client(
    arguments: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=CLIENT_ENV,
        check=True,
        text=True,
        capture_output=True,
    )


def _verify_annotated_tag(source_root: Path, tag: str, commit: str) -> None:
    tag_ref = f"refs/tags/{tag}"
    tag_type = _run_client(
        [GIT, "cat-file", "-t", tag_ref], cwd=source_root
    ).stdout.strip()
    if tag_type != "tag":
        raise RuntimeError(f"required attempt tag is not annotated: {tag}")
    tagged_commit = _run_client(
        [GIT, "rev-parse", "--verify", f"{tag_ref}^{{commit}}"], cwd=source_root
    ).stdout.strip()
    if tagged_commit != commit:
        raise RuntimeError(
            f"required attempt tag {tag} resolves to {tagged_commit}, not {commit}"
        )


def _verify_frozen_source(source_root: Path) -> dict[str, str]:
    config_path = source_root / "config/phase1_ordering_confirmation.json"
    config = json.loads(config_path.read_text())
    expected_tag = str(config["required_attempt_tag"])
    if config.get("cluster_launcher") != "cluster/submit_phase1_confirmation.sh":
        raise RuntimeError("production submission requires the isolated shell launcher")
    status = _run_client(
        [GIT, "status", "--porcelain", "--untracked-files=all"], cwd=source_root
    ).stdout
    if status:
        raise RuntimeError("submission requires a clean source checkout")
    head = _run_client([GIT, "rev-parse", "HEAD"], cwd=source_root).stdout.strip()
    _verify_annotated_tag(source_root, expected_tag, head)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {"commit": head, "tag": expected_tag, "config_sha256": config_sha256}


def _run_client_unchecked(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        env=CLIENT_ENV,
        check=False,
        text=True,
        capture_output=True,
    )


def _cancel_jobs(job_ids: list[str]) -> dict[str, Any]:
    if not job_ids:
        return {"attempted": False, "confirmed_terminal": True, "jobs": []}
    cancel_arguments = [SCANCEL, *job_ids]
    cancel = _run_client_unchecked(cancel_arguments)
    result: dict[str, Any] = {
        "attempted": True,
        "arguments": cancel_arguments,
        "returncode": int(cancel.returncode),
        "stdout": cancel.stdout,
        "stderr": cancel.stderr,
        "jobs": job_ids,
        "confirmed_terminal": False,
    }
    if cancel.returncode != 0:
        return result
    query_arguments = [
        SQUEUE,
        "--noheader",
        "--jobs=" + ",".join(job_ids),
        "--format=%i|%T",
    ]
    for poll_index in range(60):
        query = _run_client_unchecked(query_arguments)
        invalid_ids = (
            query.returncode != 0
            and not query.stdout.strip()
            and "Invalid job id specified" in query.stderr
        )
        if (query.returncode == 0 and not query.stdout.strip()) or invalid_ids:
            result.update(
                {
                    "confirmed_terminal": True,
                    "confirmation_arguments": query_arguments,
                    "confirmation_polls": poll_index + 1,
                    "confirmation_returncode": int(query.returncode),
                    "confirmation_stdout": query.stdout,
                    "confirmation_stderr": query.stderr,
                }
            )
            return result
        if query.returncode != 0:
            result.update(
                {
                    "confirmation_arguments": query_arguments,
                    "confirmation_polls": poll_index + 1,
                    "confirmation_returncode": int(query.returncode),
                    "confirmation_stdout": query.stdout,
                    "confirmation_stderr": query.stderr,
                }
            )
            return result
        if poll_index < 59:
            time.sleep(0.5)
    result.update(
        {
            "confirmation_arguments": query_arguments,
            "confirmation_polls": 60,
            "confirmation_returncode": int(query.returncode),
            "confirmation_stdout": query.stdout,
            "confirmation_stderr": query.stderr,
        }
    )
    return result


def submit_plan(
    plan: list[dict[str, Any]], attempt_root: Path, source: dict[str, str]
) -> dict[str, Any]:
    attempt_root.mkdir(mode=0o700)
    (attempt_root / "slurm").mkdir(mode=0o700)
    (attempt_root / "telemetry").mkdir(mode=0o700)
    record: dict[str, Any] = {
        "schema_version": 1,
        "source": source,
        "attempt_root": str(attempt_root),
        "client_environment": CLIENT_ENV,
        "plan": plan,
        "jobs": [],
        "state": "SUBMITTING_HELD",
    }
    record_path = attempt_root / "SUBMISSION.json"
    _json_atomic(record_path, record)
    job_ids: dict[str, str] = {}
    try:
        for row in plan:
            arguments = list(row["sbatch_base_arguments"])
            dependency_ids = [job_ids[name] for name in row["dependencies"]]
            if dependency_ids:
                arguments.append("--dependency=afterok:" + ":".join(dependency_ids))
            arguments.extend([row["wrapper"], *row["wrapper_arguments"]])
            completed = _run_client(arguments)
            job_id = completed.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    f"sbatch returned an invalid job id: {completed.stdout!r}"
                )
            job_ids[row["stage"]] = job_id
            record["jobs"].append(
                {
                    "stage": row["stage"],
                    "job_id": job_id,
                    "dependencies": dependency_ids,
                    "arguments": arguments,
                    "submitted_held": True,
                }
            )
            _json_atomic(record_path, record)
        release_arguments = [SCONTROL, "release", ",".join(job_ids.values())]
        record["release_arguments"] = release_arguments
        record["state"] = "READY_TO_RELEASE"
        _json_atomic(record_path, record)
        _run_client(release_arguments)
        record["state"] = "SUBMITTED"
        _json_atomic(record_path, record)
    except BaseException as error:
        record["error_type"] = type(error).__name__
        record["state"] = "CANCELLING"
        try:
            record["quarantined_receipt"] = _quarantine_receipt(record_path)
        except BaseException as quarantine_error:
            record["quarantine_error_type"] = type(quarantine_error).__name__
        try:
            _json_atomic(record_path, record)
        except BaseException as persistence_error:
            record["cancelling_persistence_error_type"] = type(
                persistence_error
            ).__name__
        try:
            cancellation = _cancel_jobs(list(job_ids.values()))
        except BaseException as cancellation_error:
            cancellation = {
                "attempted": True,
                "confirmed_terminal": False,
                "jobs": list(job_ids.values()),
                "error_type": type(cancellation_error).__name__,
            }
        record["cancellation"] = cancellation
        record["state"] = (
            "SUBMISSION_FAILED_CANCELLED"
            if cancellation["confirmed_terminal"]
            else "SUBMISSION_FAILED_CANCEL_UNCONFIRMED"
        )
        try:
            _json_atomic(record_path, record)
        except BaseException:
            pass
        raise
    return record


def _verify_launcher_runtime() -> None:
    if Path(sys.executable).resolve() != LOCKED_PYTHON.resolve():
        raise RuntimeError("production submission requires the locked interpreter")
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.no_user_site
        and sys.flags.dont_write_bytecode
    ):
        raise RuntimeError("production launcher must run with -I -S -B")
    forbidden = {"site", "sitecustomize", "usercustomize"} & set(sys.modules)
    if forbidden:
        raise RuntimeError(f"production launcher imported startup hooks: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = _require_safe_absolute(args.source_root, "source root")
    attempt_root = _require_safe_absolute(args.attempt_root, "attempt root")
    plan = build_plan(source_root, attempt_root)
    if args.dry_run:
        print(json.dumps({"schema_version": 1, "plan": plan}, indent=2, sort_keys=True))
        return 0
    _verify_launcher_runtime()
    source = _verify_frozen_source(source_root)
    record = submit_plan(plan, attempt_root, source)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
