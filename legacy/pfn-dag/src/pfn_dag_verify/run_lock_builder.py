import argparse
import json
from pathlib import Path

from .provenance import (
    REQUIRED_VALIDATIONS,
    RUN_LOCK_CLAIM_SCOPE,
    RUN_LOCK_SETTINGS,
    expected_run_lock_files,
    repository_root,
    validate_audit_readiness,
    validate_locked_validations,
)
from .registry import sha256_file
from .storage import write_json_atomic


def build_run_lock(root: Path):
    validate_locked_validations(root)
    validate_audit_readiness(root)
    files = {}
    for relative in sorted(expected_run_lock_files(root)):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    return {
        "schema_version": 1,
        "files": files,
        "required_validations": sorted(REQUIRED_VALIDATIONS),
        "settings": RUN_LOCK_SETTINGS,
        "claim_scope": RUN_LOCK_CLAIM_SCOPE,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=repository_root() / "config/run_lock.json")
    args = parser.parse_args(argv)
    root = repository_root()
    lock = build_run_lock(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, lock)
    print(json.dumps({"files": len(lock["files"]), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
