"""Fill Amendment G's G.0 code-state block and take its sha256.

Same two rules as scripts/lock_amendment.py (digest LAST; the file list is what
the amendment actually depends on), with G's own file list: the verifier and
its evaluator live in scripts/, the substrate and readout code in the F-locked
tree. Refuses a dirty tree unless --allow-dirty (untracked vendored artifact
directories are excluded via .git/info/exclude before locking, and the
exclusion is journaled).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FILES = {
    "__G_PREDGAIN__": "scripts/mech_predgain.py",          # the estimand producer
    "__G_GATES__": "scripts/mech_gates.py",                # the verdict
    "__G_PHASE1__": "scripts/mech_phase1.py",              # panel, exact posteriors, G4 projections
    "__G_INTERV__": "scripts/mech_interventions.py",       # panel replay with latents
    "__G_TRAIN__": "scripts/mech_train.py",                # the fleet's trainer
    "__G_TRAIN_SBATCH__": "cluster/mech_train.sbatch",
    "__G_SCORE_SBATCH__": "cluster/mech_score.sbatch",
    "__G_FIRE__": "cluster/fire_confirm_g.sh",
    "__G_SCORE__": "cluster/score_confirm_g.sh",                       # the pinned scoring driver
    "__G_READOUT_SBATCH__": "cluster/mech_readout_g.sbatch",           # G4's two stages
    "__G_LOCK__": "scripts/lock_amendment_g.py",           # this file, digested as it runs
    "__D_ORACLE__": "src/pfn_dag_verify/corrected_oracle.py",
    "__D_MODELS__": "src/pfn_dag_verify/corrected_models.py",
    "__D_SEM__": "src/pfn_dag_verify/corrected_sem.py",
    "__D_WORLD__": "src/pfn_dag_verify/corrected_world.py",
    "__D_SPLIT__": "src/pfn_dag_verify/split_panel.py",
    "__D_TOMO__": "src/pfn_dag_verify/corrected_tomography.py",     # G4 readout, unmodified
    "__D_IDENT__": "src/pfn_dag_verify/corrected_identifiability.py",  # Q2 panel, unmodified
    "__D_TRACKB__": "src/pfn_dag_verify/corrected_trackb.py",          # SCALES["base"]/["large"]: the trained arch
    "__D_VERDICT__": "src/pfn_dag_verify/corrected_verdict.py",        # eps_tag: every artifact filename and lookup key
    "__D_DEFICIT__": "src/pfn_dag_verify/corrected_deficit_run.py",    # PANEL_SEED / N_PER_HALF, the phase1 defaults
    "__D_IDENTRUN__": "src/pfn_dag_verify/corrected_identifiability_run.py",  # HEADLINE_N_ROWS, the MECH_N_ROWS default
    "__D_PILOT__": "src/pfn_dag_verify/pilot_shared.py",               # N_BINS + production_quadrature: EVERY exact score
    "__D_STATUS__": "src/pfn_dag_verify/artifact_status.py",           # imported by four G.0 modules (stamps/constants)
    "__D_W2__": "src/pfn_dag_verify/w2_run.py",            # recipe identity reference
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amendment", type=Path, default=Path("AMENDMENT_G.md"))
    p.add_argument("--out", type=Path, default=Path("AMENDMENT_G.lock.json"))
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "status", "--porcelain=v1"], cwd=root,
                           capture_output=True, text=True).stdout
    if dirty.strip() and not a.allow_dirty:
        print("REFUSING: the working tree is dirty. A code-state digest of files "
              "that are not what HEAD says they are is a number with no "
              "referent.\n" + dirty, file=sys.stderr)
        return 2
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    text = a.amendment.read_text()
    # A second run would freeze G.0 at the FIRST run's HEAD and digests (its tokens
    # are already consumed) while rewriting the .lock.json and .sha256 from the
    # current tree -- three artifacts, silently disagreeing, exit 0. Refuse.
    if not any(t in text for t in list(FILES) + ["__HEAD__"]):
        print("REFUSING: no unfilled tokens in "
              f"{a.amendment} — it is already locked. Re-locking would leave G.0 "
              "frozen at the first lock while the sidecars moved. To re-lock, restore "
              "the tokenised text (git checkout) and run once.", file=sys.stderr)
        return 4
    digests = {}
    for token, rel in FILES.items():
        d = sha(root / rel)
        digests[rel] = d
        if token in text:
            text = text.replace(token, d)
    text = text.replace("__HEAD__", head)
    left = [t for t in list(FILES) + ["__HEAD__"] if t in text]
    if left:
        print(f"REFUSING: unfilled tokens remain: {left}", file=sys.stderr)
        return 3
    a.amendment.write_text(text)
    lock = {"amendment": str(a.amendment), "head": head, "digests": digests,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    a.out.write_text(json.dumps(lock, indent=2))
    # The sha256 sidecar in Amendment F's prose format ("<name> sha256 <hex>" first
    # line), which cluster/fire_confirm_g.sh reads with head -1 | awk '{print $3}'.
    side = a.amendment.with_suffix(".sha256")
    side.write_text(f"{a.amendment.name} sha256 {lock['sha256']}\n"
                    f"locked by Bo (counter-signature in the amendment text)\nHEAD at lock {head}\n")
    print(f"{a.amendment} sha256 {lock['sha256']}\nHEAD {head}\nwrote {a.out} and {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
