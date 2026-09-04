"""Build the self-excluding Phase-1 qualification integrity manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns/phase1_ordering_20260803"
QUALIFICATION = CAMPAIGN / "oracle_qualification"
MANIFEST = QUALIFICATION / "integrity_manifest.json"
LEDGER = CAMPAIGN / "ledger.json"
VERIFIER = ROOT / "src/pfn_dag_verify/phase1_qualification_verify.py"
SOURCE_COMMIT = "cdd541c2ac7038b5cb8c7c6d3f1f6ac1811e4b88"
SOURCE_TAG = "phase1-ordering-qualification-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    ledger = json.loads(LEDGER.read_text())
    jobs = [
        row
        for row in ledger["jobs"]
        if row.get("job_id") in {"158858", "158859", "158860"}
        or row.get("stage") == "oracle_qualification_join"
    ]
    if len(jobs) != 4:
        raise RuntimeError("qualification ledger inventory is incomplete")
    verification_path = QUALIFICATION / "independent_verification.json"
    verification = json.loads(verification_path.read_text())
    if verification.get("verification") != "INDEPENDENT_RAW_RECOMPUTATION_PASS":
        raise RuntimeError("independent verification did not pass")
    if verification.get("verifier_source_sha256") != sha256_file(VERIFIER):
        raise RuntimeError("independent verifier source hash mismatch")
    tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", SOURCE_TAG],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tag_commit != SOURCE_COMMIT:
        raise RuntimeError("qualification source tag mismatch")
    tag_oid = subprocess.run(
        ["git", "rev-parse", SOURCE_TAG],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    entries = [
        record(path, QUALIFICATION)
        for path in sorted(QUALIFICATION.rglob("*"))
        if path.is_file() and path != MANIFEST
    ]
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    manifest = {
        "schema_version": 1,
        "stage": "phase1_ordering_cross_bank_qualification",
        "scope": "finite-panel 8192/16384 versus 32768 oracle truncation qualification",
        "verdict": "CAVEATED_QUALIFICATION_PASS",
        "selected_truncation": 16_384,
        "source": {
            "run_commit": SOURCE_COMMIT,
            "protocol_tag": {
                "name": SOURCE_TAG,
                "object_id": tag_oid,
                "signed": False,
            },
            "preservation": "run commit is an ancestor in this repository",
        },
        "executions": jobs,
        "ledger": record(LEDGER, ROOT),
        "verification": {
            "status": verification["verification"],
            "source": record(VERIFIER, ROOT),
            "runtime": verification["verifier_runtime"],
            "output": record(verification_path, ROOT),
        },
        "inventory": {
            "root": QUALIFICATION.relative_to(ROOT).as_posix(),
            "excluded": ["integrity_manifest.json"],
            "canonicalization": "UTF-8 JSON, sorted keys, separators comma/colon",
            "entries": entries,
            "files": len(entries),
            "bytes": sum(int(row["bytes"]) for row in entries),
            "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        },
        "known_limits": [
            "derived qualification metrics cannot be reconstructed from unarchived 100-bin predictions",
            "full 3M atom arrays were not retained",
            "per-job assigned GPU UUID was not captured",
            "join command, host, and timestamps were not captured prospectively",
        ],
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(payload)
    temporary.replace(MANIFEST)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "inventory_files": len(entries),
                "inventory_sha256": manifest["inventory"]["inventory_sha256"],
                "manifest_sha256": sha256_file(MANIFEST),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
