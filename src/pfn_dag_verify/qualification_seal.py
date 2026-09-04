"""Create and independently replay a content-addressed mapping-qualification archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from .mapping_qualification import (
    ATTEMPT_TAG,
    qualification_run_directory,
    verify_qualification_artifact,
)
from .provenance import clean_head, repository_root
from .registry import sha256_file
from .storage import write_json_atomic


def _tree_hash(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _add_regular_file(archive: tarfile.TarFile, source: Path, member_name: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"qualification archive source is not a regular file: {source}")
    info = tarfile.TarInfo(member_name)
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def _verify_bundle(bundle_path: Path, *, root: Path, commit_sha: str) -> None:
    subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(bundle_path)],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    parsed = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0] for line in heads}
    if parsed.get("HEAD") != commit_sha:
        raise ValueError("qualification source bundle does not contain the exact HEAD")
    if f"refs/tags/{ATTEMPT_TAG}" not in parsed:
        raise ValueError("qualification source bundle omits the one-shot attempt tag")


def _replay_readme(run_name: str, commit_sha: str) -> str:
    return (
        "PFN-DAG native mapping qualification replay\n\n"
        "This archive contains the exact Git commit and attempt tag plus every raw "
        "prediction and guard tensor. The integrity replay does not rerun checkpoints; "
        "it recomputes all gates from the saved numeric tensors.\n\n"
        "From the extracted archive directory:\n\n"
        "  git clone replay/source.bundle restored\n"
        f"  mkdir -p restored/runs/{run_name}\n"
        f"  cp -R run/. restored/runs/{run_name}/\n"
        "  cd restored\n"
        "  python -m pip install -r environment/requirements-lock.txt\n"
        "  PYTHONPATH=src python -m pfn_dag_verify.mapping_qualification verify\n\n"
        f"Expected Git commit: {commit_sha}\n"
    )


def _static_verify_archive(archive_path: Path) -> dict:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise FileNotFoundError("qualification archive is missing")
    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "sealed_manifest.json" not in names:
            raise ValueError("qualification archive inventory is malformed")
        if any(
            not member.isfile()
            or member.name.startswith("/")
            or ".." in Path(member.name).parts
            for member in members
        ):
            raise ValueError("qualification archive contains an unsafe member")
        manifest_handle = archive.extractfile("sealed_manifest.json")
        if manifest_handle is None:
            raise ValueError("qualification archive manifest cannot be read")
        manifest = json.loads(manifest_handle.read())
        required = {
            "schema_version",
            "kind",
            "commit_sha",
            "attempt_tag",
            "decision",
            "run_name",
            "content_tree_sha256",
            "entries",
        }
        if set(manifest) != required or manifest.get("schema_version") != 1:
            raise ValueError("qualification archive manifest schema mismatch")
        entries = manifest["entries"]
        if not isinstance(entries, list) or _tree_hash(entries) != manifest["content_tree_sha256"]:
            raise ValueError("qualification archive content-tree hash mismatch")
        expected_names = {"sealed_manifest.json", *(entry["path"] for entry in entries)}
        if set(names) != expected_names:
            raise ValueError("qualification archive member inventory mismatch")
        for entry in entries:
            member = archive.getmember(entry["path"])
            if member.size != entry["size"]:
                raise ValueError(f"qualification archive size mismatch: {entry['path']}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"qualification archive member cannot be read: {entry['path']}")
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"qualification archive hash mismatch: {entry['path']}")
    return manifest


def verify_qualification_archive(archive_path: Path, *, semantic: bool = True) -> dict:
    archive_path = Path(archive_path).resolve()
    manifest = _static_verify_archive(archive_path)
    if not semantic:
        return manifest
    with tempfile.TemporaryDirectory(prefix="pfn-dag-map-replay-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive.getmembers():
                target = extracted / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"qualification archive member cannot be extracted: {member.name}")
                with target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        restored = extracted / "restored"
        subprocess.run(
            ["git", "clone", str(extracted / "replay" / "source.bundle"), str(restored)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        observed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=restored,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if observed_commit != manifest["commit_sha"]:
            raise ValueError("fresh replay clone checked out the wrong commit")
        restored_run = restored / "runs" / manifest["run_name"]
        shutil.copytree(extracted / "run", restored_run)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src"
        replay = subprocess.run(
            [sys.executable, "-m", "pfn_dag_verify.mapping_qualification", "verify"],
            cwd=restored,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if replay.returncode != 0:
            raise RuntimeError(
                "fresh qualification replay failed: "
                + (replay.stderr.strip() or replay.stdout.strip())
            )
        replay_summary = json.loads(replay.stdout)
        if replay_summary.get("decision") != manifest["decision"]:
            raise ValueError("fresh qualification replay decision mismatch")
    return manifest


def seal_qualification(*, root: Path | None = None) -> dict:
    root = repository_root() if root is None else Path(root).resolve()
    commit_sha = clean_head(root)
    summary = verify_qualification_artifact(root=root, commit_sha=commit_sha)
    run_dir = qualification_run_directory(commit_sha, root)
    bundle_dir = root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bundle_dir, prefix="map-seal-") as temporary:
        staging = Path(temporary)
        source_bundle = staging / "source.bundle"
        subprocess.run(
            [
                "git",
                "bundle",
                "create",
                str(source_bundle),
                "HEAD",
                f"refs/tags/{ATTEMPT_TAG}",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _verify_bundle(source_bundle, root=root, commit_sha=commit_sha)
        readme = staging / "README.txt"
        readme.write_text(_replay_readme(run_dir.name, commit_sha))
        sources = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                sources.append((f"run/{path.relative_to(run_dir).as_posix()}", path))
        sources.extend(
            [("replay/source.bundle", source_bundle), ("replay/README.txt", readme)]
        )
        entries = [
            {"path": name, "size": source.stat().st_size, "sha256": sha256_file(source)}
            for name, source in sources
        ]
        manifest = {
            "schema_version": 1,
            "kind": "pfn-dag-native-mapping-qualification",
            "commit_sha": commit_sha,
            "attempt_tag": ATTEMPT_TAG,
            "decision": summary["decision"],
            "run_name": run_dir.name,
            "content_tree_sha256": _tree_hash(entries),
            "entries": entries,
        }
        manifest_path = staging / "sealed_manifest.json"
        write_json_atomic(manifest_path, manifest)
        archive_path = bundle_dir / (
            f"mapping-qualification-{manifest['content_tree_sha256']}.tar"
        )
        temporary_archive = staging / "archive.tmp"
        with tarfile.open(temporary_archive, mode="w") as archive:
            _add_regular_file(archive, manifest_path, "sealed_manifest.json")
            for member_name, source in sources:
                _add_regular_file(archive, source, member_name)
        os.replace(temporary_archive, archive_path)
    verified = verify_qualification_archive(archive_path, semantic=True)
    receipt = {
        "schema_version": 1,
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "content_tree_sha256": verified["content_tree_sha256"],
        "commit_sha": verified["commit_sha"],
        "decision": verified["decision"],
        "same_runtime_fresh_source_replay_verified": True,
    }
    receipt_path = archive_path.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seal")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "seal":
        result = seal_qualification()
    else:
        manifest = verify_qualification_archive(args.archive, semantic=True)
        result = {
            "archive": str(args.archive.resolve()),
            "archive_sha256": sha256_file(args.archive),
            "commit_sha": manifest["commit_sha"],
            "decision": manifest["decision"],
            "same_runtime_fresh_source_replay_verified": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
