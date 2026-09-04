#!/usr/bin/env python3
"""Fire Branch B — exact finite-prior causal wind tunnel — to the WashU cluster.

Phase 1: generate K-atom finite libraries on CPU.
Phase 2: train PFN fleets on each library (GPU).
Phase 3: evaluate checkpoints against exact Bayesian posterior (GPU).
Phase 4: harvest results, generate the posterior-fidelity-vs-predictive-capture figure.

Usage:
  python3 cluster/fire_branch_b.py [--validate-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CLUSTER = "washu"
REMOTE_ROOT = "/engrfs/project/class/zhao.b/pfn-dag-branch-b"
PARTITION = "condo-cse5100"
ACCOUNT = "engr-acad-cse5100"
PY = "/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"

# Branch B config
K_ATOMS = 256
N_LIBRARIES = 3
LIBRARY_SEEDS = [889_100_000, 889_200_000, 889_300_000]
C_SEEDS = list(range(3))  # 3 C seeds per library
N_SEEDS = list(range(3))  # 3 N seeds per library
TRAIN_STEPS = 10000
CKPT_EVERY = 500  # 20 checkpoints per training
N_EVAL_CONTEXTS = 80  # per prior per library


def _ssh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", CLUSTER, *args],
                          check=True, capture_output=True, text=True)


def _rsync(local: Path, remote: str) -> None:
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude=.git", "--exclude=.pytest_cache", "--exclude=.ruff_cache",
        "--exclude=artifacts/checkpoints", "--exclude=campaigns",
        "--exclude=run", "--exclude=bundles",
        str(local) + "/", f"{CLUSTER}:{remote}/",
    ]
    subprocess.run(cmd, check=True)


def rust_library(seed: int, out_name: str) -> int:
    """Submit a library-generation job (CPU, fast). Returns job ID."""
    cmd = (f"sbatch --partition={PARTITION} --account={ACCOUNT} --cpus-per-task=4 "
           f"--mem=8G --time=00:15:00 --parsable --wrap=\""
           f"cd {REMOTE_ROOT} && {PY} -m pfn_dag_verify.branch_b_library "
           f"--k {K_ATOMS} --seed {seed} "
           f"--out {REMOTE_ROOT}/run/{out_name}.npz\"")
    return int(_ssh(cmd).stdout.strip())


def rust_train(library_name: str, prior: str, seed: int) -> int:
    """Submit a PFN training job (GPU). Returns job ID."""
    tag = f"{library_name}_{prior}_s{seed}"
    cmd = (f"sbatch --partition={PARTITION} --account={ACCOUNT} --gres=gpu:1 "
           f"--cpus-per-task=8 --mem=32G --time=06:00:00 --parsable --wrap=\""
           f"cd {REMOTE_ROOT} && {PY} -m pfn_dag_verify.branch_b_train "
           f"--library {REMOTE_ROOT}/run/{library_name}.npz "
           f"--prior {prior} --seed {seed} --steps {TRAIN_STEPS} "
           f"--ckpt-every {CKPT_EVERY} --out {REMOTE_ROOT}/run/\"")
    return int(_ssh(cmd).stdout.strip())


def rust_eval(library_name: str, prior: str, seed: int) -> int:
    """Submit an evaluation job (GPU). Returns job ID."""
    tag = f"bb_{prior}_s{seed}_st{TRAIN_STEPS}"
    ckpt_steps = ",".join(str(s) for s in range(CKPT_EVERY, TRAIN_STEPS + 1, CKPT_EVERY))
    cmd = (f"sbatch --partition={PARTITION} --account={ACCOUNT} --gres=gpu:1 "
           f"--cpus-per-task=8 --mem=48G --time=08:00:00 --parsable --wrap=\""
           f"cd {REMOTE_ROOT} && {PY} -m pfn_dag_verify.branch_b_eval "
           f"--library {REMOTE_ROOT}/run/{library_name}.npz "
           f"--ckpt-dir {REMOTE_ROOT}/run "
           f"--tag-prefix {tag} --prior {prior} "
           f"--n-contexts {N_EVAL_CONTEXTS} "
           f"--seed {889_500_000 + seed} "
           f"--ckpt-steps {ckpt_steps} "
           f"--out {REMOTE_ROOT}/run/eval_{library_name}_{prior}_s{seed}.json\"")
    return int(_ssh(cmd).stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()

    print("syncing repo...", flush=True)
    _rsync(repo, REMOTE_ROOT)
    _ssh(f"mkdir -p {REMOTE_ROOT}/run")
    print(f"synced -> {CLUSTER}:{REMOTE_ROOT}", flush=True)

    if args.validate_only:
        print("submitting one library + one training smoke...", flush=True)
        j_lib = rust_library(LIBRARY_SEEDS[0], "lib_smoke")
        print(f"  lib job: {j_lib}", flush=True)
        return 0

    # Phase 1: libraries
    print("\nPhase 1: generating libraries...", flush=True)
    lib_names = []
    for i, seed in enumerate(LIBRARY_SEEDS):
        name = f"lib_{i}"
        jid = rust_library(seed, name)
        lib_names.append(name)
        print(f"  {name} seed={seed}: job {jid}", flush=True)

    # Phase 2: training (submit held; release after libraries complete)
    print("\nPhase 2: submitting training jobs...", flush=True)
    train_jobs: dict[str, list[int]] = {}
    for lib_name in lib_names:
        train_jobs[lib_name] = []
        for prior in ("C", "N"):
            seeds = C_SEEDS if prior == "C" else N_SEEDS
            for s in seeds:
                jid = rust_train(lib_name, prior, s)
                train_jobs[lib_name].append(jid)
                print(f"  {lib_name} {prior} s{s}: job {jid}", flush=True)

    # Phase 3: evaluation (dependent on training — submit separately)
    # For now, submit evaluation as a post-hoc phase
    print("\nPhase 3: evaluation postponed (run after training completes)", flush=True)

    # Record submission
    record = {
        "schema_version": 1,
        "stage": "branch_b",
        "k_atoms": K_ATOMS,
        "n_libraries": N_LIBRARIES,
        "library_names": lib_names,
        "train_seeds": {"C": C_SEEDS, "N": N_SEEDS},
        "train_steps": TRAIN_STEPS,
        "ckpt_every": CKPT_EVERY,
        "train_jobs": train_jobs,
        "eval_contexts_per_prior": N_EVAL_CONTEXTS,
    }
    record_path = f"{REMOTE_ROOT}/run/BRANCH_B_SUBMISSION.json"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", CLUSTER, f"cat > {record_path}"],
        input=json.dumps(record, indent=2), text=True, check=True)
    print(f"\nsubmission record: {CLUSTER}:{record_path}", flush=True)
    total_jobs = sum(len(v) for v in train_jobs.values()) + len(lib_names)
    print(f"total: {len(lib_names)} lib + {total_jobs - len(lib_names)} train = {total_jobs} jobs submitted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
