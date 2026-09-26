"""Verify checksums of the public export and pinned scientific modules."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root / "manifests/public-export.json").read_text())
    failures = []
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            failures.append(name)
    scientific = json.loads((root / "manifests/scientific-provenance.json").read_text())
    for name, expected in scientific["portable_files_sha256"].items():
        relative = name if name.startswith("pfn_dag_verify/") else "pfn_dag_verify/" + name
        path = root / "src" / relative
        if not path.is_file() or digest(path) != expected:
            failures.append("src/" + relative)
    if failures:
        raise ValueError("Provenance verification failed: " + ", ".join(failures))
    return {"status": "PASS", "export_files": len(manifest["files_sha256"]),
            "pinned_scientific_modules": len(scientific["portable_files_sha256"])}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
