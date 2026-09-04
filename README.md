# LDM_interpretation — public evaluation export

Source code, raw results, registered protocol documents and machine-written
verdict files for the PFN-DAG project, exported from the private working
repository at commit `397b3f8` (branch `paper-assembly`, 2026-09-03).

This export contains no worklogs, status notes, audits, reviews, reports, paper
prose or figures. Every file that was withheld is listed in `EXPORT_OMITTED.txt`.
Nothing that remains was edited, except `tests/test_retired_constants.py`, whose
allow-list lost four entries that named withheld files, and `.gitignore`.

## Layout

| path | content |
|---|---|
| `src/pfn_dag_verify/` | verification package (`pfn-dag-*` entry points, see `pyproject.toml`) |
| `scripts/` | campaign drivers, scorers, gate evaluators, figure generators |
| `cluster/` | Slurm submission scripts and cluster tests, as run on the WashU cluster |
| `tests/` | pytest suite |
| `config/`, `environment/` | run locks, checkpoint registries, query banks, pinned runtimes |
| `campaigns/<name>/` | one directory per campaign: raw arrays, checkpoints, result JSON, sealed manifests, terminal status |
| `artifacts/` | audit records, checkpoint tables, source snapshots, validation output |
| `paper/` | bibliography and style files only |
| `PREREG.md`, `PHASE1_*.md`, `ORACLE_*.md`, `MAPPING_*.md`, `AMENDMENT_{E,F,G}.md` | registered protocol documents, with `.sha256` / `.lock.json` sidecars where one was taken |
| `campaigns/*/PREREG-*.md`, `PRESPEC_*.md`, `SUPERSESSION.md`, `PROVENANCE.md` | per-campaign registrations and supersession records |

## Environment

Python 3.11 with the pins in `pyproject.toml` (numpy 1.26.4, scipy 1.12.0,
scikit-learn 1.6.1, torch 2.9.1, pytest 9.1.1).

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
PYTHONPATH=src python -m pytest tests -q
```

Tests known to fail in this export, with the reason, are listed in
`EXPECTED_TEST_FAILURES.txt`.

## Raw data shipped as release assets

The arrays read by `scripts/mech_gates.py` (campaign `mech_20260827`) and by
`scripts/mech_component_control.py` (campaign `mech_ext_20260902`) are too
large for the tree and are attached to the GitHub release
`raw-mech_20260827`. Hashes of the archives are in `RELEASE_ASSETS_SHA256SUMS.txt`;
hashes of the individual `.npz` files are in
`campaigns/mech_20260827/reported/MANIFEST.json`.

| archive | extracts to | size |
|---|---|---|
| `mech_20260827_predgain_confirm.tar` | `campaigns/mech_20260827/predgain_confirm/` | 1.0 GB |
| `mech_20260827_predgain_confirm_n40.tar` | `campaigns/mech_20260827/predgain_confirm_n40/` | 20 MB |
| `mech_20260827_predgain_select.tar` | `campaigns/mech_20260827/predgain_select/` | 198 MB |
| `mech_20260827_confirm_phase1.tar` | `campaigns/mech_20260827/confirm/phase1/` | 2.4 MB |
| `mech_ext_20260902_coarse.tar` | `campaigns/mech_ext_20260902/coarse/` | 143 MB |

```
cd campaigns && for t in ../release_assets/*.tar; do tar -xf "$t"; done
```

The committed verdict `campaigns/mech_20260827/confirm/AMENDMENT_G_VERDICT.json`
records in its `argv` field the exact invocation that produced it:

```
python scripts/mech_gates.py --fleet campaigns/mech_20260827/predgain_confirm \
  --grid campaigns/mech_20260827/predgain_confirm \
  --select campaigns/mech_20260827/predgain_select \
  --n40 campaigns/mech_20260827/predgain_confirm_n40/base_n40_lr0.001_d500000 \
  --readout campaigns/mech_20260827/confirm/phase1 \
  --seeds 3 4 5 --eps 0.5 0.75 1.0 --steps 100000 500000 --alpha 0.01 \
  --readout-step 500000 --out <path>
```

Raw arrays for the earlier campaigns (`phase1_ordering_20260803`,
`corrected_20260812`, `binary_duel_*`, `branch_b_20260808`) are in the tree.
Model checkpoints for the `mech_*` fleets are not included; the gate evaluators
compare checkpoint hashes recorded in the arrays and never load weights.

## Provenance notes

- Sealed manifests (`*.sha256`, `ARTIFACT_SHA256SUMS`, `integrity_manifest.json`,
  `run_lock.json`) were not edited. Where they name a withheld file,
  `sha256sum -c` reports it missing; `EXPORT_OMITTED.txt` lists which.
- Job logs and result JSON keep the cluster paths, hostnames and account name
  they were written with.
- Git history of the private repository is not included. Verifiers that call
  `git` for a HEAD hash (`phase1_*_verify`) will report the hash of this
  repository instead.
