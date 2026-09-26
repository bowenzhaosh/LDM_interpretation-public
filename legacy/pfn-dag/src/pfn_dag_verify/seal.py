import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from .model import configure_determinism
from .integrity import EXPECTED_IDENTITIES, verify_prediction_ledger
from .provenance import (
    MEMORY_CAP_BYTES,
    RAW_CAP_BYTES,
    WALL_CAP_SECONDS,
    repository_root,
    require_scientific_run_path,
    verify_panel_lock,
    verify_run_lock,
)
from .registry import sha256_file
from .storage import load_numeric_npz, write_json_atomic


def _canonical_tree_hash(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tracked_files(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    values = [value.decode("utf-8") for value in output.split(b"\0") if value]
    if not values:
        raise ValueError("repository has no tracked replay files")
    return sorted(values)


def _safe_run_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError(f"run manifest path escapes the run directory: {relative}")
    return path


def _archive_member_name(entry: dict) -> str:
    return f"{entry['scope']}/{entry['path']}"


def _entry_source(entry: dict, *, root: Path, run_dir: Path) -> Path:
    if entry["scope"] == "repository":
        return (root / entry["path"]).resolve()
    if entry["scope"] == "run":
        return _safe_run_path(run_dir, entry["path"])
    if entry["scope"] == "replay":
        return _safe_run_path(run_dir, f"replay/{entry['path']}")
    raise ValueError("unknown archive entry scope")


def _verify_git_bundle(bundle_path: Path, *, root: Path, commit: str) -> None:
    subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(bundle_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not any(line.split()[0] == commit for line in heads if line.split()):
        raise ValueError("replay Git bundle does not contain the scientific commit")


def _create_replay_material(root: Path, run_dir: Path, commit: str) -> list[str]:
    replay_dir = run_dir / "replay"
    if replay_dir.exists():
        raise FileExistsError("replay material already exists")
    replay_dir.mkdir()
    bundle_path = replay_dir / "source.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    _verify_git_bundle(bundle_path, root=root, commit=commit)
    run_name = run_dir.name
    instructions = (
        "PFN-DAG sealed replay\n\n"
        "This recomputation is noncanonical: it writes replay_summary.json with "
        "status REPLAY_VERIFIED_NONCANONICAL and cannot issue LOCALLY_VERIFIED.\n\n"
        "From the extracted archive directory, run:\n\n"
        "  git clone replay/source.bundle restored\n"
        f"  mkdir -p restored/runs/{run_name}\n"
        f"  cp -R run/. restored/runs/{run_name}/\n"
        f"  cp -R replay restored/runs/{run_name}/replay\n"
        f"  cp sealed_manifest.json restored/runs/{run_name}/sealed_manifest.json\n"
        "  cd restored\n"
        f"  PYTHONPATH=src python -m pfn_dag_verify.analysis replay --run-dir runs/{run_name}\n\n"
        f"Expected Git commit: {commit}\n"
    )
    (replay_dir / "README.txt").write_text(instructions)
    return ["README.txt", "source.bundle"]


def _create_archive(
    *, manifest_path: Path, entries: list[dict], root: Path, run_dir: Path
) -> Path:
    bundle_dir = root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tree_hash = json.loads(manifest_path.read_text())["content_tree_sha256"]
    archive_path = bundle_dir / f"{tree_hash}.tar"
    with tempfile.NamedTemporaryFile(
        dir=bundle_dir, prefix=tree_hash + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, mode="w") as archive:
            sources = [("sealed_manifest.json", manifest_path)]
            sources.extend(
                (_archive_member_name(entry), _entry_source(entry, root=root, run_dir=run_dir))
                for entry in entries
            )
            for member_name, source in sources:
                info = tarfile.TarInfo(member_name)
                info.size = source.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with source.open("rb") as source_handle:
                    archive.addfile(info, source_handle)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, archive_path)
        descriptor = os.open(bundle_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive_path


def _write_manifest_and_archive(
    *, manifest_path: Path, manifest: dict, entries: list[dict], root: Path, run_dir: Path
) -> Path:
    """Converge the manifest's raw-byte total to the actual tar-file size."""
    for _ in range(4):
        write_json_atomic(manifest_path, manifest)
        archive_path = _create_archive(
            manifest_path=manifest_path,
            entries=entries,
            root=root,
            run_dir=run_dir,
        )
        actual_bytes = archive_path.stat().st_size
        if int(manifest["resource_totals"]["raw_bytes"]) == actual_bytes:
            return archive_path
        manifest["resource_totals"]["raw_bytes"] = actual_bytes
    raise RuntimeError("sealed archive size did not converge")


def _verify_archive(
    *, archive_path: Path, manifest_path: Path, entries: list[dict]
) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError("content-addressed replay archive is missing")
    expected = {"sealed_manifest.json": sha256_file(manifest_path)}
    expected.update({_archive_member_name(entry): entry["sha256"] for entry in entries})
    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != set(expected) or len(members) != len(expected):
            raise ValueError("replay archive member set is incomplete or duplicated")
        for member in members:
            if not member.isfile():
                raise ValueError(f"replay archive contains a non-file member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read replay archive member: {member.name}")
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected[member.name]:
                raise ValueError(f"replay archive member hash mismatch: {member.name}")


def _run_files(run_dir: Path) -> tuple[list[str], dict, dict]:
    panel_path = run_dir / "panel.npz"
    prediction_dir = run_dir / "predictions"
    derived_dir = run_dir / "derived"
    panel, prediction_ledger, _ = verify_prediction_ledger(
        panel_path=panel_path, prediction_dir=prediction_dir
    )
    verify_panel_lock(panel)
    from .analysis import verify_derived_ledger

    _, derived_ledger, _ = verify_derived_ledger(
        panel_path=panel_path,
        prediction_dir=prediction_dir,
        derived_dir=derived_dir,
    )
    if (
        derived_ledger.get("schema_version") != 1
        or derived_ledger.get("scientific") is not True
        or derived_ledger.get("panel_sha256") != sha256_file(panel_path)
        or derived_ledger.get("prediction_ledger_sha256")
        != sha256_file(prediction_dir / "prediction_ledger.json")
    ):
        raise ValueError("derived ledger identity mismatch during sealing")
    records = derived_ledger.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError("derived ledger must contain exactly 64 records")
    identities = {
        (int(record["seed"]), int(record["step"]), int(record["bank_index"]))
        for record in records
    }
    if identities != EXPECTED_IDENTITIES:
        raise ValueError("derived ledger identities are incomplete")
    derived_names = set()
    for record in records:
        path = derived_dir / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"derived file mismatch during sealing: {path.name}")
        derived_names.add(path.name)
    if {path.name for path in derived_dir.glob("derived_*.npz")} != derived_names:
        raise ValueError("derived directory contains a stale or missing shard")
    relative = [
        "panel.npz",
        "panel.json",
        "pre_score_guard.json",
        "score_progress.json",
        "derive_progress.json",
        "predictions/prediction_ledger.json",
        "derived/derived_ledger.json",
    ]
    relative.extend(
        f"predictions/{record['path']}" for record in prediction_ledger["records"]
    )
    relative.extend(f"derived/{record['path']}" for record in records)
    if len(relative) != 135 or len(set(relative)) != 135:
        raise AssertionError("sealed run file set is incomplete or duplicated")
    for item in relative:
        if not _safe_run_path(run_dir, item).is_file():
            raise FileNotFoundError(item)
    return sorted(relative), prediction_ledger, derived_ledger


def _validated_panel_resources(
    run_dir: Path, panel: dict, *, commit: str, run_lock_hash: str
) -> tuple[float, int]:
    path = run_dir / "panel.json"
    metadata = json.loads(path.read_text())
    expected_keys = {
        "schema_version",
        "producer_sha256",
        "commit_sha",
        "n_groups",
        "n_continuations",
        "eligible_replace_groups",
        "eligible_append_groups",
        "interior_selected",
        "scientific",
        "run_lock_sha256",
        "evaluation_root",
        "candidate_core_count",
        "candidate_block_count",
        "panel_sha256",
        "panel_bytes",
        "wall_seconds",
        "peak_rss_bytes",
    }
    panel_path = run_dir / "panel.npz"
    eligible_replace_groups = int(
        np.sum(np.sum(panel["eligible_replace"].astype(bool), axis=1) >= 4)
    )
    eligible_append_groups = int(
        np.sum(np.sum(panel["eligible_append"].astype(bool), axis=1) >= 4)
    )
    expected_values = {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__).with_name("evaluation.py")),
        "commit_sha": commit,
        "n_groups": int(panel["core"].shape[0]),
        "n_continuations": int(panel["continuations"].shape[1]),
        "eligible_replace_groups": eligible_replace_groups,
        "eligible_append_groups": eligible_append_groups,
        "interior_selected": bool(int(panel["selection_mode"]) == 1),
        "scientific": bool(int(panel["scientific"])),
        "run_lock_sha256": run_lock_hash,
        "evaluation_root": int(panel["evaluation_root"]),
        "candidate_core_count": int(len(panel["candidate_core_seed"])),
        "candidate_block_count": int(len(panel["candidate_block_seed"])),
        "panel_sha256": sha256_file(panel_path),
        "panel_bytes": panel_path.stat().st_size,
    }
    if set(metadata) != expected_keys or any(
        metadata.get(key) != value for key, value in expected_values.items()
    ):
        raise ValueError("panel metadata identity or producer mismatch")
    try:
        wall_seconds = float(metadata["wall_seconds"])
        peak_rss_bytes = int(metadata["peak_rss_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("panel metadata resource fields are invalid") from error
    if (
        not np.isfinite(wall_seconds)
        or wall_seconds < 0.0
        or peak_rss_bytes < 0
        or isinstance(metadata["peak_rss_bytes"], bool)
        or peak_rss_bytes != metadata["peak_rss_bytes"]
    ):
        raise ValueError("panel metadata resources must be finite and nonnegative")
    return wall_seconds, peak_rss_bytes


def seal_run(run_dir: Path, out_path: Path | None = None) -> dict:
    configure_determinism(0)
    root = repository_root()
    commit, run_lock_hash, _ = verify_run_lock(root)
    run_dir = run_dir.resolve()
    require_scientific_run_path(run_dir, commit_sha=commit, relative=".")
    out_path = run_dir / "sealed_manifest.json" if out_path is None else out_path.resolve()
    if out_path.parent != run_dir:
        raise ValueError("sealed manifest must live at the run root")
    run_files, prediction_ledger, derived_ledger = _run_files(run_dir)
    panel = load_numeric_npz(run_dir / "panel.npz")
    panel_commit, _ = verify_panel_lock(panel)
    if panel_commit != commit:
        raise ValueError("sealed run commit differs from clean HEAD")
    replay_files = _create_replay_material(root, run_dir, commit)

    panel_wall, panel_peak = _validated_panel_resources(
        run_dir, panel, commit=commit, run_lock_hash=run_lock_hash
    )
    wall_seconds = panel_wall
    wall_seconds += float(prediction_ledger["wall_seconds"])
    wall_seconds += float(derived_ledger["wall_seconds"])
    peak_rss_bytes = max(
        panel_peak,
        int(prediction_ledger["peak_rss_bytes"]),
        int(derived_ledger.get("peak_rss_bytes", 0)),
    )

    entries = []
    for relative in _tracked_files(root):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"tracked replay input is not a regular file: {relative}")
        entries.append(
            {
                "scope": "repository",
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for relative in run_files:
        path = _safe_run_path(run_dir, relative)
        entries.append(
            {
                "scope": "run",
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for relative in replay_files:
        path = _safe_run_path(run_dir, f"replay/{relative}")
        entries.append(
            {
                "scope": "replay",
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    entries.sort(key=lambda value: (value["scope"], value["path"]))
    member_bytes = sum(entry["size"] for entry in entries)
    if wall_seconds > WALL_CAP_SECONDS:
        raise RuntimeError("BLOCKED_COST: actual wall-clock limit exceeded")
    if peak_rss_bytes > MEMORY_CAP_BYTES:
        raise RuntimeError("BLOCKED_COST: actual memory limit exceeded")
    if member_bytes > RAW_CAP_BYTES:
        raise RuntimeError("BLOCKED_COST: actual raw-storage limit exceeded")
    manifest = {
        "schema_version": 1,
        "scientific": True,
        "commit_sha": commit,
        "run_lock_sha256": run_lock_hash,
        "panel_sha256": sha256_file(run_dir / "panel.npz"),
        "content_tree_sha256": _canonical_tree_hash(entries),
        "files": entries,
        "resource_totals": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "raw_bytes": member_bytes,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "raw_cap_bytes": RAW_CAP_BYTES,
        },
    }
    archive_path = _write_manifest_and_archive(
        manifest_path=out_path,
        manifest=manifest,
        entries=entries,
        root=root,
        run_dir=run_dir,
    )
    if archive_path.stat().st_size > RAW_CAP_BYTES:
        raise RuntimeError("BLOCKED_COST: actual replay archive exceeds storage limit")
    verify_sealed_manifest(out_path)
    return manifest


def verify_sealed_manifest(path: Path, *, require_archive: bool = True) -> dict:
    configure_determinism(0)
    root = repository_root()
    commit, run_lock_hash, _ = verify_run_lock(root)
    path = path.resolve()
    run_dir = path.parent
    require_scientific_run_path(run_dir, commit_sha=commit, relative=".")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scientific") is not True
        or manifest.get("commit_sha") != commit
        or manifest.get("run_lock_sha256") != run_lock_hash
        or manifest.get("panel_sha256") != sha256_file(run_dir / "panel.npz")
    ):
        raise ValueError("sealed manifest identity mismatch")
    expected_run_files, prediction_ledger, derived_ledger = _run_files(run_dir)
    expected_repository_files = _tracked_files(root)
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("sealed manifest file list is missing")
    observed_run = sorted(
        entry["path"] for entry in entries if entry.get("scope") == "run"
    )
    observed_repository = sorted(
        entry["path"] for entry in entries if entry.get("scope") == "repository"
    )
    observed_replay = sorted(
        entry["path"] for entry in entries if entry.get("scope") == "replay"
    )
    if (
        observed_run != expected_run_files
        or observed_repository != expected_repository_files
        or observed_replay != ["README.txt", "source.bundle"]
    ):
        raise ValueError("sealed manifest does not contain the exact replay tree")
    for entry in entries:
        if entry.get("scope") == "run":
            target = _safe_run_path(run_dir, entry["path"])
        elif entry.get("scope") == "repository":
            target = (root / entry["path"]).resolve()
            if not target.is_relative_to(root):
                raise ValueError("repository manifest path escapes the root")
        elif entry.get("scope") == "replay":
            target = _safe_run_path(run_dir, f"replay/{entry['path']}")
        else:
            raise ValueError("sealed manifest contains an unknown scope")
        if (
            not target.is_file()
            or target.stat().st_size != int(entry.get("size", -1))
            or sha256_file(target) != entry.get("sha256")
        ):
            raise ValueError(f"sealed content mismatch: {entry.get('path')}")
    _verify_git_bundle(run_dir / "replay" / "source.bundle", root=root, commit=commit)
    ordered = sorted(entries, key=lambda value: (value["scope"], value["path"]))
    if manifest.get("content_tree_sha256") != _canonical_tree_hash(ordered):
        raise ValueError("sealed content-tree hash mismatch")
    archive_path = root / "bundles" / f"{manifest['content_tree_sha256']}.tar"
    if require_archive:
        _verify_archive(
            archive_path=archive_path,
            manifest_path=path,
            entries=ordered,
        )
    totals = manifest.get("resource_totals", {})
    panel = load_numeric_npz(run_dir / "panel.npz")
    panel_wall, panel_peak = _validated_panel_resources(
        run_dir, panel, commit=commit, run_lock_hash=run_lock_hash
    )
    computed_wall = (
        panel_wall
        + float(prediction_ledger["wall_seconds"])
        + float(derived_ledger["wall_seconds"])
    )
    computed_peak = max(
        panel_peak,
        int(prediction_ledger["peak_rss_bytes"]),
        int(derived_ledger.get("peak_rss_bytes", 0)),
    )
    member_bytes = sum(int(entry["size"]) for entry in entries)
    try:
        recorded_raw = int(totals["raw_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("sealed raw-byte total is missing") from error
    if recorded_raw < member_bytes:
        raise ValueError("sealed raw-byte total is smaller than its members")
    computed_raw = archive_path.stat().st_size if require_archive else recorded_raw
    if (
        float(totals.get("wall_seconds", np.inf)) != computed_wall
        or int(totals.get("peak_rss_bytes", -1)) != computed_peak
        or recorded_raw != computed_raw
    ):
        raise ValueError("sealed resource totals do not match the replay tree")
    if (
        computed_wall > WALL_CAP_SECONDS
        or computed_peak > MEMORY_CAP_BYTES
        or computed_raw > RAW_CAP_BYTES
    ):
        raise RuntimeError("BLOCKED_COST: sealed resources exceed a scientific cap")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = seal_run(args.run_dir)
    print(
        json.dumps(
            {
                "content_tree_sha256": result["content_tree_sha256"],
                "files": len(result["files"]),
                "resource_totals": result["resource_totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
