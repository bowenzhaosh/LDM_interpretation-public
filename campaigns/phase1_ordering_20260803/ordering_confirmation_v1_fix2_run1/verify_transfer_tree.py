#!/usr/bin/env python3
"""Recompute a canonical content digest for a directory tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded(relative: str, exclusions: Iterable[str]) -> bool:
    for value in exclusions:
        normalized = value.strip("/")
        if relative == normalized or relative.startswith(normalized + "/"):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    records: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if is_excluded(relative, args.exclude):
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    records.sort(key=lambda value: str(value["path"]))

    digest = hashlib.sha256()
    lines: list[str] = []
    for record in records:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        digest.update(line.encode("utf-8") + b"\n")
        lines.append(line)
    if args.manifest is not None:
        args.manifest.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "algorithm": "sha256 of newline-delimited canonical JSON records",
                "exclusions": list(args.exclude),
                "file_count": len(records),
                "root": str(root),
                "schema_version": 1,
                "total_bytes": sum(int(value["size"]) for value in records),
                "tree_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
