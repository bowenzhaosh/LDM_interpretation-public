"""Fill Amendment F's F.0 code-state block and take its sha256.

Two rules this encodes, both learned the hard way.

The DIGEST IS TAKEN LAST. Per the turn-3 ordering constraint, any change to what
decides a gate goes into the verdict code BEFORE the lock, never after: taking
the digest first would have locked the width-primary implementation that F.2b
exists to replace, and the same argument applies to F.2d.1's slope.

The FILE LIST is what the amendment actually depends on. F.0 was written when
the gate lived in four files; F.2d.5 and F.8 put load-bearing behaviour in
split_panel.py, artifact_status.py and w2_run.py, and a code-state block that
omits them records less than it claims to.

Refuses to run on a dirty tree: a digest of files that are not what HEAD says
they are is a number with no referent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FILES = {
    "__D_VERDICT__": "src/pfn_dag_verify/corrected_verdict.py",
    "__D_ORACLE__": "src/pfn_dag_verify/corrected_oracle.py",
    "__D_MODELS__": "src/pfn_dag_verify/corrected_models.py",
    "__D_SEM__": "src/pfn_dag_verify/corrected_sem.py",
    "__D_SPLIT__": "src/pfn_dag_verify/split_panel.py",
    "__D_STATUS__": "src/pfn_dag_verify/artifact_status.py",
    "__D_DEFICIT__": "src/pfn_dag_verify/corrected_deficit_run.py",
    "__D_IDENT__": "src/pfn_dag_verify/corrected_identifiability_run.py",
    "__D_W2__": "src/pfn_dag_verify/w2_run.py",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amendment", type=Path, default=Path("AMENDMENT_F.md"))
    p.add_argument("--out", type=Path, default=Path("AMENDMENT_F.lock.json"))
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)

    root = Path.cwd()
    dirty = subprocess.run(["git", "status", "--porcelain=v1"], cwd=root,
                           check=True, text=True, capture_output=True).stdout
    if dirty.strip() and not a.allow_dirty:
        print("REFUSING: the working tree is dirty. A code-state digest of files "
              "that are not what HEAD says they are is a number with no "
              "referent.\n" + dirty, file=sys.stderr)
        return 2
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          text=True, capture_output=True).stdout.strip()

    text = a.amendment.read_text()
    digests = {}
    for token, rel in FILES.items():
        f = root / rel
        if not f.is_file():
            raise SystemExit(f"{rel} is missing; F.0 cannot record it")
        d = sha(f)
        digests[rel] = d
        if token in text:
            text = text.replace(token, d)
    text = text.replace("__HEAD__", head)
    left = [t for t in list(FILES) + ["__HEAD__"] if t in text]
    if left:
        raise SystemExit(f"unfilled placeholders remain: {left}")
    a.amendment.write_text(text)

    lock = {"amendment": a.amendment.name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "git_head": head, "code_state": digests,
            "n_lines": len(text.splitlines())}
    a.out.write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
