"""Amendment F — assemble the one directory the locked verdict reads.

`cells_from_raw` globs a SINGLE directory (corrected_verdict.py:1147+), and no
locked clause names an assembly step, so the convention is registered in
`.claude/FIDELITY_DECISIONS.md` decision F rather than invented at run time:

  * 21 `w2_eps{tag}_s{seed}.json`      from raw/postF/w2/        (gate (a), regret)
  * 7  `fidelity_eps{tag}.json`        from raw/postF/fidelity/  (the y-axis arrays)
  * 7  `ident_eps{tag}_highs_ipm.json` from raw/postF/ident_grid/ + .env.json
                                       (F.2b's DEMOTED secondary x-axis)

`deficit_eps{tag}.json` is **EXCLUDED**. Under B1 (owner, 2026-08-26)
`deficit_w2/` is not regenerated, and its `panel_sha256` differs from W2's at
every eps because the hash is a build fingerprint of a LAPACK-dependent draw
(errata 1), so `cells_from_raw` would refuse the pair. The consequence is
recorded rather than worked around: **the primary x-axis is absent from the
locked verdict run and the primary association reports not-evaluable.**

Assembly is by COPY, never by mixing in place: the producer directories are
untouched and stay the record, and `MANIFEST.json` carries every file's source
path and sha256 so the assembly is repeatable and auditable. The verdict output
is written BESIDE this directory, never into a globbed namespace.

Nothing here reads or computes y.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

W2_SEEDS = (0, 1, 2)
EPS_TAGS = ("0p0", "0p1", "0p25", "0p35", "0p5", "0p75", "1p0")
IDENT_SOLVER = "highs_ipm"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _plan(w2: Path, fid: Path, ident: Path, with_ident: bool):
    """(source, dest_name, group) for every file the verdict directory holds."""
    out = []
    for t in EPS_TAGS:
        for s in W2_SEEDS:
            n = f"w2_eps{t}_s{s}.json"
            out.append((w2 / n, n, "w2"))
        n = f"fidelity_eps{t}.json"
        out.append((fid / n, n, "fidelity"))
        if with_ident:
            n = f"ident_eps{t}_{IDENT_SOLVER}.json"
            out.append((ident / n, n, "ident"))
            e = f"ident_eps{t}_{IDENT_SOLVER}.env.json"
            if (ident / e).is_file():
                out.append((ident / e, e, "ident_env"))
    return out


def _dry_run(dest: Path, repo: Path) -> tuple[bool, str]:
    """Load the assembled directory through the LOCKED loader before reporting
    success. A directory that assembles but does not load is not an assembly."""
    code = (
        "import json,sys;from pathlib import Path;"
        "from pfn_dag_verify.corrected_verdict import cells_from_raw;"
        f"c=cells_from_raw(Path({str(dest)!r}));"
        "print(json.dumps({'n_cells':len(c),"
        "'eps':[x.eps for x in c],"
        "'has_fidelity':[bool(x.model_ctx is not None) for x in c],"
        "'has_regret':[bool(x.regret_seeds) for x in c],"
        "'has_width':[x.width_exact is not None for x in c],"
        "'has_deficit':[x.deficit_exact is not None for x in c]}))"
    )
    env = dict(os.environ, PYTHONPATH=str((repo / "src").resolve()))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env)
    return (r.returncode == 0, (r.stdout or "") + (r.stderr or ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--root", type=Path,
                   default=Path("campaigns/corrected_20260812/raw/postF"))
    p.add_argument("--dest", type=Path, default=None,
                   help="default: <root>/verdict_w2")
    p.add_argument("--force", action="store_true",
                   help="replace an existing destination directory")
    a = p.parse_args(argv)
    root = a.root
    dest = a.dest or (root / "verdict_w2")
    if dest.exists() and not a.force:
        raise SystemExit(f"{dest} exists. Pass --force to rebuild it; assembly "
                         "is a copy step and re-running it must be deliberate.")
    if dest.exists():
        shutil.rmtree(dest)

    with_ident = True
    for attempt in ("with_ident", "without_ident"):
        plan = _plan(root / "w2", root / "fidelity", root / "ident_grid", with_ident)
        missing = [str(s) for s, _, _ in plan if not s.is_file()]
        if missing:
            raise SystemExit("assembly refuses: these sources are absent — "
                             + ", ".join(missing[:10])
                             + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""))
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        manifest = {
            "amendment": "F",
            "decision": ".claude/FIDELITY_DECISIONS.md decision F",
            "deficit_excluded": (
                "B1 (owner, 2026-08-26): deficit_w2/ is NOT regenerated. Its "
                "panel_sha256 differs from W2's at every eps (errata 1), so "
                "cells_from_raw would refuse the pair. CONSEQUENCE: the primary "
                "x-axis is absent and the primary association is not-evaluable."),
            "ident_included": with_ident,
            "files": {},
        }
        for src, name, group in plan:
            shutil.copy2(src, dest / name)
            manifest["files"][name] = {"source": str(src), "sha256": _sha256(src),
                                       "group": group}
        (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

        ok, out = _dry_run(dest, a.repo)
        if ok:
            print(f"assembled {len(plan)} files into {dest} "
                  f"(ident_included={with_ident})")
            print(out.strip())
            return 0
        if attempt == "with_ident":
            # Decision F's pre-specified contingency, not a re-decision: an ident
            # file the loader refuses is dropped and the drop is recorded.
            print("dry-run load REFUSED an artifact with ident_grid present; "
                  "retrying without it per decision F's contingency:\n"
                  + out.strip(), file=sys.stderr)
            with_ident = False
            continue
        raise SystemExit("dry-run load failed even without ident_grid. The "
                         "assembly is not loadable and nothing downstream may "
                         f"run:\n{out.strip()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
