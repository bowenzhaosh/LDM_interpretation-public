import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .model import configure_determinism
from .provenance import (
    EXPECTED_AUDIT_DISPOSITIONS,
    AUDIT_EVIDENCE_FILES,
    REQUIRED_VALIDATIONS,
    audit_readiness_subject_files,
    repository_root,
    validate_audit_evidence,
    validate_locked_validations,
    verify_runtime,
)
from .registry import sha256_file
from .storage import write_json_atomic


def _run_tests(root: Path) -> tuple[int, int, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")
    }
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    collect_output = collect.stdout + collect.stderr
    if collect.returncode != 0:
        raise RuntimeError("readiness test collection failed")
    collected_matches = re.findall(
        r"(?:^|\s)(\d+) tests? collected(?:\s|$)", collect_output
    )
    if len(collected_matches) != 1 or int(collected_matches[0]) < 1:
        raise RuntimeError("readiness could not verify the pytest collection count")
    tests_collected = int(collected_matches[0])
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError("readiness test suite failed")
    matches = re.findall(r"(?:^|\s)(\d+) passed(?:\s|$)", output)
    if len(matches) != 1 or int(matches[0]) != tests_collected:
        raise RuntimeError("readiness could not verify the pytest pass count")
    combined_output = collect_output + "\n--- pytest execution ---\n" + output
    return (
        tests_collected,
        int(matches[0]),
        hashlib.sha256(combined_output.encode("utf-8")).hexdigest(),
    )


def build_readiness(root: Path) -> dict:
    configure_determinism(0)
    verify_runtime(root)
    validate_locked_validations(root)
    if "Disposition: `READY_V3`" not in (root / "AUDIT.md").read_text():
        raise ValueError("AUDIT.md is not READY_V3")
    audit_evidence = validate_audit_evidence(root)
    tests_collected, tests_passed, test_output_sha256 = _run_tests(root)
    subject_hashes = {
        relative: sha256_file(root / relative)
        for relative in sorted(audit_readiness_subject_files(root))
    }
    return {
        "schema_version": 1,
        "protocol_version": 3,
        "disposition": "READY_V3",
        "scientific_outputs_observed": False,
        "test_command": "python -m pytest -q tests",
        "tests_collected": tests_collected,
        "tests_passed": tests_passed,
        "test_output_sha256": test_output_sha256,
        "required_validations": sorted(REQUIRED_VALIDATIONS),
        "audits": {
            lens: audit_evidence[lens]["disposition"]
            for lens in EXPECTED_AUDIT_DISPOSITIONS
        },
        "audit_evidence_sha256s": {
            lens: sha256_file(root / relative)
            for lens, relative in AUDIT_EVIDENCE_FILES.items()
        },
        "failed_stream": {
            "commit_sha": "d0b049d6241845e55443f4950e52b70644b2b1ab",
            "status": "BLOCKED_GUARD",
            "prediction_shards": 0,
            "scientific_metrics": 0,
        },
        "subject_sha256s": subject_hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=repository_root() / "config" / "audit_readiness.json"
    )
    args = parser.parse_args(argv)
    root = repository_root()
    result = build_readiness(root)
    write_json_atomic(args.out, result)
    print(json.dumps({"files": len(result["subject_sha256s"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
