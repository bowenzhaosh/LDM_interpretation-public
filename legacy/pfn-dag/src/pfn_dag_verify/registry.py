import hashlib
import json
from pathlib import Path

from .constants import BASE_MODEL_CONFIG


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(package.glob("*.py"), key=lambda value: value.name)
    }


def validation_input_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "config" / "checkpoint_registry.json"
    registry = json.loads(registry_path.read_text())
    relative_paths = {
        "config/query_bank.json",
        "config/checkpoint_registry.json",
        "artifacts/legacy/stage1_functional_law.py",
        "environment/runtime.json",
        "environment/requirements-lock.txt",
        "environment/installed-distributions.json",
        *(str(record["path"]) for record in registry.get("original_sources", [])),
    }
    return {
        relative: sha256_file(root / relative) for relative in sorted(relative_paths)
    }


def verify_file_record(record: dict, *, root: str | Path | None = None) -> None:
    required = {"path", "size", "sha256"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"registry record missing keys: {sorted(missing)}")
    path = Path(record.get("_resolved_path", record["path"]))
    if not path.is_absolute():
        if root is None:
            raise ValueError(f"relative registry path has no resolver root: {path}")
        path = Path(root) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"registered file does not exist: {path}")
    if path.stat().st_size != int(record["size"]):
        raise ValueError(f"size mismatch for {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"SHA-256 mismatch for {path}")


def load_checkpoint_registry(path: str | Path) -> dict:
    registry_path = Path(path).resolve()
    registry = json.loads(registry_path.read_text())
    repository_root = registry_path.parent.parent
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint registry schema")
    if registry.get("model_config") != BASE_MODEL_CONFIG:
        raise ValueError("registry model config mismatch")
    checkpoints = registry.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 32:
        raise ValueError("registry must contain exactly 32 checkpoints")
    identities = {(int(item["seed"]), int(item["step"])) for item in checkpoints}
    expected = {(seed, step) for seed in range(16) for step in (0, 12_000)}
    if identities != expected:
        raise ValueError("registry seed/step identities do not match the locked fleet")
    for item in checkpoints:
        verify_file_record(item, root=repository_root)
        item["_resolved_path"] = str((repository_root / item["path"]).resolve())
        durable_uri = item.get("durable_uri")
        if durable_uri != f"repo://{item['path']}":
            raise ValueError(f"checkpoint does not have its locked repository URI: {item['path']}")
    sources = registry.get("original_sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("registry must contain the four original source snapshots")
    for item in sources:
        verify_file_record(item, root=repository_root)
        if item.get("durable_uri") != f"repo://{item['path']}":
            raise ValueError(f"source does not have its locked repository URI: {item['path']}")
    by_identity = {(int(item["seed"]), int(item["step"])): item for item in checkpoints}
    for seed in range(16):
        if by_identity[(seed, 0)]["sha256"] == by_identity[(seed, 12_000)]["sha256"]:
            raise ValueError(f"seed {seed} has identical step-0 and step-12000 checkpoints")
    return registry


def expanded_checkpoint_record(registry: dict, seed: int, step: int) -> dict:
    matches = [
        item
        for item in registry["checkpoints"]
        if int(item["seed"]) == int(seed) and int(item["step"]) == int(step)
    ]
    if len(matches) != 1:
        raise ValueError(f"checkpoint identity is not unique: seed={seed}, step={step}")
    return {
        **matches[0],
        "model_config": registry["model_config"],
        "state_schema": registry["state_schema"],
    }
