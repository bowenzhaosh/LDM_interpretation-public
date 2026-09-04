#!/usr/bin/env python3
"""Fire the corrected campaign (Track A2 exact identifiability + Track B models)
to the WashU cluster.

- Track A2 (CPU): exact identifiability analysis per epsilon on general-cpu.
- Track B  (GPU): PFN saturation + order classifier per epsilon on condo-cse5100
  (owned priority), falling back to general-gpu.

The corrected-campaign namespace is NEW (never overwrites Branch B artifacts).

Usage:
  python3 cluster/fire_corrected.py [--launch-identifiability] [--launch-trackb]
                                    [--validate-only] [--repo PATH]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CLUSTER = "washu"
REMOTE_ROOT = "/engrfs/project/class/zhao.b/pfn-dag-corrected"
PY = "/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"
GPU_PARTITION = "condo-cse5100"
GPU_ACCOUNT = "engr-acad-cse5100"
# CPU lane: debug allows ALL accounts (general-cpu requires engr-research,
# which this user does not have). MaxTime 4h > the ~15 min identifiability jobs.
CPU_PARTITION = "debug"
CPU_ACCOUNT = "engr-class-any"

EPS_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
# Corrected in-task fleet: the disposition witness (eps1.0) + the low-eps cells
# where structure is poorly identified (w1 > 0.20) — the cells that decide
# whether the thesis is testable. eps0.5 skipped (least-pinned ident point).
IN_TASK_EPS = [0.0, 0.1, 0.25, 1.0]
D, K = 3, 8
N_IDENT_CONTEXTS = 20
IDENT_TOL = 1e-3


def _ssh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", CLUSTER, *args],
                          check=True, capture_output=True, text=True)


def _rsync(local: Path) -> None:
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude=.git", "--exclude=.pytest_cache", "--exclude=.ruff_cache",
        "--exclude=artifacts/checkpoints", "--exclude=campaigns/phase1_ordering_20260803",
        "--exclude=campaigns/branch_b_20260808", "--exclude=run", "--exclude=bundles",
        # Local experiment scratch: ~900MB of panel.npz / bundles, not needed by
        # the cluster campaign. Every fire previously re-synced it (10+ min).
        "--exclude=runs",
        str(local) + "/", f"{CLUSTER}:{REMOTE_ROOT}/",
    ]
    subprocess.run(cmd, check=True)
    print(f"synced -> {CLUSTER}:{REMOTE_ROOT}", flush=True)


def submit_identifiability(eps: float) -> int:
    tag = f"ident_eps{eps}".replace(".", "p")
    out = f"{REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.json"
    cmd = (f"sbatch --partition={CPU_PARTITION} --account={CPU_ACCOUNT} "
           f"--cpus-per-task=4 --mem=12G "
           f"--time=02:00:00 --parsable --job-name={tag} "
           f"--output={REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.out "
           f"--wrap=\"cd {REMOTE_ROOT} && PYTHONPATH=src {PY} -m "
           f"pfn_dag_verify.corrected_identifiability_run "
           f"--d {D} --K {K} --eps {eps} --n-contexts {N_IDENT_CONTEXTS} "
           f"--tol {IDENT_TOL} --out {out}\"")
    return int(_ssh(cmd).stdout.strip())


def submit_trackb(eps: float) -> int:
    tag = f"trackb_eps{eps}".replace(".", "p")
    out = f"{REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.json"
    cmd = (f"sbatch --partition={GPU_PARTITION} --account={GPU_ACCOUNT} --gres=gpu:1 "
           f"--cpus-per-task=8 --mem=32G --time=12:00:00 --parsable --job-name={tag} "
           f"--output={REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.out "
           f"--wrap=\"cd {REMOTE_ROOT} && PYTHONPATH=src {PY} -m "
           f"pfn_dag_verify.corrected_trackb "
           f"--d {D} --K {K} --eps {eps} --out {out}\"")
    return int(_ssh(cmd).stdout.strip())


def submit_tomography(eps: float) -> int:
    """Re-run the tomography with the FIXED code (the pre-fix runs had a
    one-sided L1 LP bug inflating residuals/tolerances). Merges into the
    trackb JSON."""
    tag = f"tomography_eps{eps}".replace(".", "p")
    trackb = f"{REMOTE_ROOT}/campaigns/corrected_20260812/raw/trackb_eps{eps}.json".replace(".json", "")
    cmd = (f"sbatch --partition={GPU_PARTITION} --account={GPU_ACCOUNT} --gres=gpu:1 "
           f"--cpus-per-task=8 --mem=32G --time=06:00:00 --parsable --job-name={tag} "
           f"--output={REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.out "
           f"--wrap=\"cd {REMOTE_ROOT} && PYTHONPATH=src {PY} -m "
           f"pfn_dag_verify.corrected_tomography_run "
           f"--d {D} --K {K} --eps {eps} "
           f"--trackb-json {trackb}.json --out {trackb}_tomography.json\"")
    return int(_ssh(cmd).stdout.strip())


def submit_intask(eps: float) -> int:
    """Corrected in-task trackb re-run: re-trains base-scale seed-0/1/2 and
    reports per-target + in-task (target=d-1) Bayes regret + in-task learning
    curve on the fixed mixed-target panel. Resolves the results-audit's
    eval-panel-mismatch finding (the stored regret mixes ~2/3 out-of-task
    queries)."""
    tag = f"intask_eps{eps}".replace(".", "p")
    out = f"{REMOTE_ROOT}/campaigns/corrected_20260812/raw/trackb_eps{eps}_corrected.json".replace(".json", "")
    cmd = (f"sbatch --partition={GPU_PARTITION} --account={GPU_ACCOUNT} --gres=gpu:1 "
           f"--cpus-per-task=8 --mem=32G --time=06:00:00 --parsable --job-name={tag} "
           f"--output={REMOTE_ROOT}/campaigns/corrected_20260812/raw/{tag}.out "
           f"--wrap=\"cd {REMOTE_ROOT} && PYTHONPATH=src {PY} -m "
           f"pfn_dag_verify.corrected_intask_run "
           f"--d {D} --K {K} --eps {eps} --out {out} --seeds 0,1,2\"")
    return int(_ssh(cmd).stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--launch-identifiability", action="store_true")
    parser.add_argument("--launch-trackb", action="store_true")
    parser.add_argument("--launch-tomography", action="store_true")
    parser.add_argument("--launch-intask", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    if not args.no_sync:
        _rsync(args.repo)
    _ssh(f"mkdir -p {REMOTE_ROOT}/campaigns/corrected_20260812/raw")

    if args.validate_only:
        j = submit_identifiability(0.0)
        print(f"validate ident job: {j}")
        return 0

    ids = []
    if args.launch_identifiability:
        for eps in EPS_VALUES:
            jid = submit_identifiability(eps)
            ids.append(jid)
            print(f"identifiability eps={eps}: job {jid}")
    if args.launch_trackb:
        for eps in EPS_VALUES:
            jid = submit_trackb(eps)
            ids.append(jid)
            print(f"trackb eps={eps}: job {jid}")

    if args.launch_tomography:
        for eps in EPS_VALUES:
            jid = submit_tomography(eps)
            ids.append(jid)
            print(f"tomography eps={eps}: job {jid}")

    if args.launch_intask:
        for eps in IN_TASK_EPS:
            jid = submit_intask(eps)
            ids.append(jid)
            print(f"intask eps={eps}: job {jid}")

    if not ids:
        print("nothing launched; use --launch-identifiability, --launch-trackb, "
              "--launch-tomography, --launch-intask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
