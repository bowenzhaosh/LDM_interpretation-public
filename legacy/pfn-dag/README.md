# LDM_interpretation — public evaluation export

Source code, raw results, registered protocol documents and machine-written
verdict files for the PFN-DAG project, exported from the private working
repository at commit `397b3f8` (branch `paper-assembly`, 2026-09-03) and updated
on 2026-09-06 with the generalisation and internal-intervention results
(see *Second export* below).

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
| `campaigns/mech_int_20260905/` | internal-intervention campaign: pre-specification + digest, per-arm result JSON, pipeline checks |
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

## Second export (2026-09-06)

Added from the private repository between `397b3f8` and `43653e4`. Everything
below is REPORTED, never gated: the seven registered gates of `AMENDMENT_G.md`
and the verdict in `campaigns/mech_20260827/confirm/AMENDMENT_G_VERDICT.json`
are unchanged and none of the code they digest was touched.

**Generalisation (campaign `mech_ext_20260902`).** `scripts/mech_ext_report.py`
re-fits (a, b) per extension cell through the locked estimator and refuses to
write unless the 53 registered re-fits reproduce bit-for-bit.

| file | content |
|---|---|
| `reported/ext_cells.json` | 377 fits: replication seeds 6-11, three fresh atom libraries (`_ws11/12/13`), `d=4`, `K=2`, `eps 0.35`, context length 40 |
| `reported/ext_families.json`, `reported/ext_seed_summaries.json` | one-factor families; per-seed b and within-seed dose differences (random-effects reading beside the registered fixed-effects one) |
| `reported/cc_ws11,cc_ws12,cc_ws13,cc_dim4/component_control.json` | the size-matched non-order component control (`PRESPEC_component_control.md`) re-run in each fresh world and at `d=4` |
| `coarse_ws*/`, `coarse_dim4/*.catalogue.json` | the coarsened-oracle partition catalogues for those worlds (order partitions are sampled, not enumerated, at `O = 24`) |

**Internal intervention (campaign `mech_int_20260905`).** Pre-specified in
`PRESPEC_internal.md` (sha256 in `PRESPEC_internal.digest`, deviations appended
to `PRESPEC_internal.errata.md`), registered before any arm produced a number.

| arm | script | output | what it decides |
|---|---|---|---|
| R — representational | `scripts/mech_probe_acts.py`, `scripts/mech_probe_fit.py` | `reported/armR_*.json` | whether a fresh readout of the last-layer query state realises more order gain than the trained head (READOUT-LIMITED / HEAD-SATURATED) |
| C — causal | `scripts/mech_patch_build.py`, `mech_patch_run.py`, `mech_patch_analyse.py` | `reported/armC_*.json` | where the realised order information enters, beside a size-matched atom swap (routing verdicts C1-C3) |
| G — generalisation | `scripts/mech_k2_verdict.py` | `reported/armG_*.json` | at `K = 2, eps = 1` the order and atom components are matched in size; the joint fit decides order-specific (A) against smallest-component (B) |

Every arm re-derives the registered oracles and refuses to write unless they
equal the scored arrays: `S(full)`/`S(abl)`/`S(prior)` to 1e-9, the model
predictive to 1e-5, `S(model)` to 1e-4, and the checkpoint sha256 to the value
the registered scoring recorded. `campaigns/mech_int_20260905/checks/test_mech_probe.py`
holds the three pipeline checks (the uniform-p̂ hybrid equals the registered
order-ablated oracle; the S identity on registered contexts; shuffled targets
carry no information).

Raw arrays for this export's new results are **not** shipped: the Arm C patch
runs (about 750 MB each) and the Arm R activation and training sets are
regenerable from the code and the registered checkpoints, and every result JSON
records the sha256 of each input it read. The scored extension cells
(`campaigns/mech_ext_20260902/predgain/`) ship only their `world_index.json`,
`ext_provenance.json` and `summary_ck*.json` records for the same reason.

## Provenance notes

- Sealed manifests (`*.sha256`, `ARTIFACT_SHA256SUMS`, `integrity_manifest.json`,
  `run_lock.json`) were not edited. Where they name a withheld file,
  `sha256sum -c` reports it missing; `EXPORT_OMITTED.txt` lists which.
- Job logs and result JSON keep the cluster paths, hostnames and account name
  they were written with.
- Git history of the private repository is not included. Verifiers that call
  `git` for a HEAD hash (`phase1_*_verify`) will report the hash of this
  repository instead.
