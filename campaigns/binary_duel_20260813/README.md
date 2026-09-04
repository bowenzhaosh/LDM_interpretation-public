# binary_duel (2026-08-13)

Three pre-registered experiments on the d=2 evidence-integration substrate
(`M_base_AL40`, dose 12000, seeds 0–3). `PREREG-BINARY-DUEL.md` is the full
protocol: substrate, estimands, gates, and decision rules.

## Layout

- `PREREG-BINARY-DUEL.md` — locked pre-registration (sha256 + amendments A/B/C).
- `code/`
  - `core.py` — substrate constants, order-conditioned predictives, evidence-score and weight extraction, regression helpers.
  - `exp1_evidence_tracking.py` — Experiment 1.
  - `exp2_mislead.py` — Experiment 2.
  - `exp3_patch.py` — Experiment 3 (Phase A site screen).
  - `aggregate_results.py` — aggregates the per-seed result JSONs.
- `results/exp{1,2,3}/` — per-seed result JSONs, `*_s{0..3}_dose12000.json`.

## Provenance

Copied 2026-08-20 from `~/pfn-dag/experiments/binary-duel/`.

## Dependencies (not bundled)

The scripts import `config` (path setup, `~/pfn-dag/experiments/infra/`) and
`evidence_core` (substrate: `ℓ(D)`, oracle, model loading,
`~/pfn-dag/evidence-integration/`). `evidence_core` in turn imports `e21_fleet`,
`d5c_analyze`, and `d5c_gate0` from `~/pfn-dag/G-experiments/e18b-committed/`.
Model checkpoints are loaded from `dose_nets/*.pt`. None of these are included
here.
