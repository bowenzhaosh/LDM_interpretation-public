# Native-output mapping qualification preregistration

Status: locked before generating `mapping-calibration-v1` predictions.

## Reason for this qualification

The sealed v3 replace-10 run at commit `dd749cfa8bf75ab34a0f5c757c500516adc354d2` returned `INCONCLUSIVE_MAPPING`. The exact-mixture fixtures validated projection arithmetic, but native PFN outputs had not been required to lie near the oracle family-endpoint segment on a held-out stream before scientific scoring.

This qualification tests that missing bridge. It does not reanalyze or rescue the sealed v3 panel.

## Frozen inputs

- Existing two eight-query banks from `config/query_bank.json`.
- Existing 16 step-12,000 checkpoint files from `config/checkpoint_registry.json`.
- Current `GridOracle` estimand, unchanged for this qualification.
- Physical prediction batch size 64.
- Seed namespace `mapping-calibration-v1`, derived from the immutable seed root `dd749cfa8bf75ab34a0f5c757c500516adc354d2`. Later code commits do not change this draw.

Queries may not be optimized against either the failed scientific panel or this qualification stream. No affine rescaling, intercept correction, gate relaxation, or checkpoint exclusion is allowed after predictions exist.

## Qualification panel

Generate 64 groups. Each group has a 20-row core, one eligible 10-row reference block, and two eligible 10-row target blocks sampled under shared SEM parameters. Every length-30 base and target must satisfy, in both banks:

- exact posterior weight in `[0.05,0.95]`;
- endpoint Jensen-Shannon divergence at least `0.1`.

The stream must use seed labels prefixed by `mapping-calibration-v1:` and therefore be disjoint by construction from calibration, smoke, failed-v2, sealed-v3, and future scientific seed namespaces. The runner must recompute every recorded seed and explicitly check it against persisted prior-panel seeds and the fixed calibration/validation seeds.

This is a one-shot qualification. A tracked pre-draw attestation pins every load-bearing protocol file. Before panel generation, the runner creates the annotated Git tag `mapping-qualification-attempt-v1` at the clean protocol commit. If that tag or any `mapping-qualification-*` attempt directory already exists, a later invocation may not create another draw. Failed and interrupted tags remain terminal records. The query-bank and checkpoint-registry hashes are pinned to their sealed-v3 values.

## Per-checkpoint, per-bank gates

For each of 16 trained checkpoints and each bank, compute coordinate and independent KL projections for every base and target prediction. Require all of the following:

- boundary rate exactly `0`;
- coordinate/KL absolute log-odds disagreement: median at most `0.10`, p95 at most `0.30`;
- base mixture residual: median at most `0.10`, p95 at most `0.30`;
- target mixture residual: median at most `0.10`, p95 at most `0.30`;
For each checkpoint, compare fitted coordinate log odds across the two banks on matched base and target contexts. Require median absolute disagreement at most `0.10` and p95 at most `0.30`.

Before accepting a checkpoint-bank block, rerun its base and target predictions, reverse the physical batch order, and reverse context-row order. Direct replay and restored batch order must be byte-identical. The maximum row-permutation error must be at most `1e-6`.

All 32 checkpoint-bank blocks and all 16 cross-bank blocks must pass. One failure fails the qualification.

## Information barrier

The qualification code must not compute or report a regression of induced-coordinate change on exact evidence change, an evidence-composition slope, an exact-evidence reconstruction residual, or the previous scientific decision metrics. Exact evidence may be used only for the frozen interiority predicate. Mapping validity tests only whether native outputs admit a stable scalar coordinate under the frozen endpoints.

For future scientific runs, this document supersedes the old use of exact-evidence reconstruction as a mapping prerequisite in `PREREG.md`. Reconstruction depends on the unit-gain hypothesis itself, so it cannot decide whether the coordinate is measurable. It is a one-sided confirmation gate: failed reconstruction may block `COMPATIBLE_ON_TESTED_REGIME`, but it may not block or trigger `INCOMPATIBLE_ON_TESTED_REGIME` when both slope intervals already lie on the same side outside `[0.8,1.2]`.

## Decision rule

- `QUALIFIED`: every locked gate passes. Without changing source or configuration, the distinct default scientific seed domain may then be used for another replace-10 run. Scientific execution must verify this completed qualification first.
- `FAILED_NATIVE_MAPPING`: any gate fails. Do not run another induced-coordinate scientific stream with this readout.
- `INVALID_RUN`: provenance, completeness, finiteness, shape, hash, or stream-disjointness checks fail.

No other terminal label is permitted.

Every terminal artifact, including `FAILED_NATIVE_MAPPING`, must be sealed into a content-addressed tar containing the raw panel and prediction tensors, the annotated attempt tag, an exact-commit Git bundle with the registered checkpoints, environment fingerprints, and replay instructions. Fresh extraction must recompute the decision from raw arrays without access to the originating checkout.
