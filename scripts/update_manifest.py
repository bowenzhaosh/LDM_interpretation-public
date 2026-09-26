"""Refresh checksums after reviewing intentional current-package changes."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "manifests/public-export.json"
    manifest = json.loads(manifest_path.read_text())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    paths = sorted(name.decode() for name in tracked.split(b"\0") if name)
    paths = [name for name in paths if not name.startswith("legacy/")
             and name != "manifests/public-export.json"]
    generated = [name for name in paths
                 if name.startswith(("outputs/", ".venv/", "build/", "dist/"))
                 or any(part.endswith(".egg-info") or part == "__pycache__"
                        for part in Path(name).parts)]
    if generated:
        raise ValueError("Generated files are tracked; remove them from the index first: "
                         + ", ".join(generated))
    if not paths:
        raise ValueError("No tracked package files; stage the intended files first.")
    manifest["files_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated {len(paths)} current-package checksums; scientific pins unchanged.")


if __name__ == "__main__":
    main()
