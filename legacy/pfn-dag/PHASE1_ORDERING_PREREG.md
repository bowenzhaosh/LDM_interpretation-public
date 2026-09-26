# Phase-1 ordering-use replication preregistration

Status: **QUALIFICATION V3 PASSED; ATTEMPTS 1 AND 2 FAILED BEFORE ENDPOINTS; FIX2 PRE-RUN AUDIT PASSED; FRESH RUN PENDING**

Pre-confirmation code audit found that the version-1 ordering ablation was
misimplemented after top-K truncation. Within-ordering weights were divided by
the full-bank normalizer, so unequal retained masses leaked into the final
ordering mixture. The calibration and qualification therefore compared two
truncations of the same wrong ablated estimator. Their artifacts remain sealed
as a historical flawed-estimator record, but they do not qualify the intended
uniform-ordering reference and cannot authorize confirmation.

The corrected version normalizes each retained within-ordering posterior to
one before mixing all 24 orderings uniformly. Qualification v2 was frozen but
blocked before execution when a numerical audit found that its inherited
quadrature was not sufficiently qualified. The prospective replacement is
frozen in `PHASE1_ORDERING_QUALIFICATION_V3_PREREG.md`. It uses another fresh
context namespace, the same three intended confirmatory atom banks, and an
explicit production-versus-reference quadrature axis. No PFN checkpoint or
scientific endpoint may be evaluated until qualification v3 passes.

Qualification v3 is the only admissible qualification for confirmation. Only
its verified artifact may select the truncation, satisfy the
predictive-truncation gate, or contribute a numerical allowance. The earlier
calibration and qualification artifacts below are retained only as historical
lineage and have no confirmatory force.

The registered calibration at commit `e679e74930fc990dc834e0a03aec0a10b582a862`
passed and selected `T_atom = 8,192` against the 32,768-atom reference. No PFN
checkpoint or scientific endpoint was evaluated during calibration.

The superseded three-bank qualification at commit
`cdd541c2ac7038b5cb8c7c6d3f1f6ac1811e4b88` rejected 8,192 because 16
individual held-out log-probability changes exceeded the frozen `0.009`-nat
boundary. It selected `T_atom = 16,384`, which had zero JS or log-probability
exceedances across all 48 registered families. An independent raw-array
recomputation reproduced that decision. Qualification also evaluated no PFN
checkpoint or scientific endpoint. These numerical statements describe only
the flawed version-1 estimator and carry no truncation-qualification force.

This protocol is for the first output-level claim that remains scientifically
independent of the failed Phase-2 induced-coordinate instrument. No projected
mixture weight, logit evidence coordinate, composition slope, probe, or
activation intervention appears in this experiment.

## Claim and scope

Primary claim:

> In the exact archived d=4 base fleet, the three causal-AL(r=4) models at
> 120,000 steps outperform the ordering-ablated causal oracle, and this direct
> advantage is larger than the matched generic output mismatch in the three
> independently trained Gaussian-control models.

The operational claim is restricted to the archived `nets4_xlong` checkpoint
fleet, context size 30, the native 100-bin output head, and the frozen d=4
synthetic generator. It is not a claim that the model explicitly represents a
Bayesian posterior over orderings. The six trained models are fixed objects;
the experiment supports inference over fresh contexts for this fleet, not over
the population of possible training runs. A population claim requires a later
prospective training replication with substantially more independent seeds.

Secondary claim:

> The ordering-specific output advantage improves between 20,000 and 120,000
> steps, so the early checkpoint can miss a capability present at the final
> checkpoint.

Scale, d=3/d=5 generalization, initialization, and Phase-2 composition are out
of scope for this attempt.

## Estimands

For prior `P` and checkpoint `t`, define on held-out outcomes

`deficit_P(t) = E[NLL_net,P(t) - NLL_ablated,P]`,

where `NLL_ablated` is the posterior predictive produced by retaining the
within-ordering covariance posterior and forcing the posterior over all 24
orderings to be uniform. The primary ordering-specific contrast is

`Delta(t) = deficit_C(t) - deficit_N(t)`.

Also define `gap_P(t) = E[NLL_net,P(t) - NLL_full,P]` and
`V_P = E[NLL_ablated,P - NLL_full,P]`. Before aggregation, every row, model,
and checkpoint must satisfy the float64 algebra identity
`deficit = gap - V` within absolute error `1e-12`.

For each prior, the deficit is first averaged equally over its three fixed
models and then over fresh held-out contexts. The `C` and `N` training seed
labels do not define paired randomizations and are never paired or resampled.

`N` is the matched Gaussian control for which ordering is non-identifiable and
`V_N = E[NLL_ablated,N - NLL_full,N]` is zero in the population. A negative
`Delta` means that the causal-prior PFN beats the ordering-blind reference by
more than the matched generic output mismatch observed under `N`.

`Delta` is not sufficient by itself: positive excess error in the independent
`N` fleet could make `Delta` negative even if the `C` fleet exactly matched its
ordering-ablated oracle. The primary decision therefore requires both a
negative direct `deficit_C` and a negative `Delta`, under the same frozen
effect floor and interval rule.

The confirmatory checkpoints are 20,000, 60,000, and final/120,000. The final
checkpoint is primary; 20,000 and the paired final-minus-early change are
secondary; 60,000 is descriptive.

## Frozen model fleet

- Architecture/scale: d=4 base PFN, native 100-bin head.
- Training length: 120,000 steps.
- Priors: `C` and `N`.
- Fixed training-seed labels: 0, 1, and 2 in each prior arm. The two arms were
  initialized and trained with prior-specific RNG streams, so equal labels are
  not treated as matched seeds.
- Checkpoint directory on WashU: `nets4_xlong`.
- Expected names: `M4_<prior>_s<seed>_st120000[_ck<step>].pt`.

Every checkpoint and the vendored fleet module must be SHA-256 registered
before confirmatory contexts are generated. Missing, additional, stale, or
silently skipped checkpoints halt the run.

The Phase-1 CUDA run is not installed from this repository's macOS mapping
requirements. It uses the separate WashU interpreter and package lock under
`environment/phase1-washu-*`. Before a production attempt, the runner verifies
the Python executable, NumPy and Torch payload trees, NumPy/Torch build
configuration, CPU model and instruction features, active BLAS runtime and
thread count, CUDA driver, GPU model and capability, deterministic settings,
and `pip check`. OpenBLAS is fixed to one thread. Each stage is bound to that
exact fingerprint and has one atomic writer lease. Stages are deliberately
fail-closed rather than partially resumable: an interrupted stage is retried
into a new empty output directory under the same immutable attempt identity,
while sealed upstream stages may be reused.

The isolated, source-bound launcher submits exactly seven held jobs in the
fixed graph `panel -> {pfn, oracle0, oracle1, oracle2} -> join -> verify`.
Before releasing any job, it durably records every job ID, exact argument
vector, dependency, and expected stage in the attempt's `SUBMISSION.json`.
The receipt enters `READY_TO_RELEASE` before release; wrappers wait for the
durable `SUBMITTED` transition and recheck it after their stage. A failure
enters `CANCELLING` before `scancel`, and the receipt reports cancellation only
after all recorded jobs leave the live queue. An unconfirmed cancellation is
an explicit failed attempt. A runnable receipt is quarantined first, and a
receipt-write failure cannot skip the scheduler cancellation attempt. Each
scientific stage receives one A100,
eight CPUs, 64 GB memory, an explicit time limit, and a job-ID-specific
telemetry directory, except that the final NumPy-only verifier requests two
CPUs and 16 GB without a GPU. The submission client receives only `PATH`,
`LC_ALL`, `TZ`, and the site-required frozen `SLURM_CONF` path. The locked
interpreter starts with `-I -S`, uses a per-job telemetry `pycache` prefix, and
admits only the frozen repository source and locked environment site-packages.
No other ambient Slurm user environment, inherited `PYTHONPATH`, user-site,
`.pth`, `sitecustomize`, or `LD_PRELOAD` startup path is allowed.

## Fresh sampling design

- Context size: 30.
- Priors: `C` and `N` only.
- Three independent evaluation draws.
- 1,067 contexts per prior per draw, giving 3,201 contexts per prior.
- Three independent 3,000,000-atom covariance banks.
- Within every evaluation draw, contexts are assigned evenly and
  deterministically across the three atom banks. This crosses evaluation and
  atom draws without multiplying the number of scientific contexts.
- Evaluation and calibration seed namespaces are disjoint from one another and
  from every persisted prior panel.
- All contexts, queries, outcomes, bins, full and ablated 100-bin predictive
  arrays, native PFN 100-bin log-probabilities, checkpoint hashes, and guard
  measurements are retained.

No confirmatory result may be read until all expected shards are present,
their hash inventory passes, and the required independent raw verification
stage agrees with the joined result.

## Qualification-v3-only truncation selection

Calibration uses a disjoint seed namespace and is excluded from every estimate
and verdict. It computes predictions at `T_atom` 8,192, 16,384, and the frozen
32,768 reference. Each candidate is compared directly with the 32,768 reference.
This prevents an apparently stable low-rung bridge from passing when a later
bridge still moves. Select the first candidate for which, under both `C` and `N`
and separately for full and ordering-ablated predictors:

- median Jensen-Shannon divergence is at most `1e-4`;
- p95 Jensen-Shannon divergence is at most `1e-3`;
- median absolute held-out-outcome log-probability change is at most `0.002`
  nats;
- p95 absolute held-out-outcome log-probability change is at most `0.01` nats;
- all probabilities are finite, normalized, and nonnegative; and
- no metric falls inside the frozen 10% numerical indifference band around a
  threshold. A borderline metric fails rather than selecting a device-dependent
  truncation.

Context-posterior retained mass and covariance-atom ESS are diagnostics only.
They cannot validate a conditional predictive because the query can reweight
atoms after context conditioning. Covariance-atom ESS is computed after
collapsing the 24 exactly enumerated ordering copies and is reported separately
for the full and ablated context posteriors.

The predictive quadrature is aligned to the native 100 output bins. Interior
bins use Gauss-Legendre quadrature, while the two clipped edge bins use a
semi-infinite change of variables. The conditional predictive is normalized
across the 100 bins, so normalization alone is not a convergence check.
Qualification v3 directly compares every production-grid vector with a
higher-order reference.

If neither candidate passes, stop. The historical version-1 calibration
retained only derived diagnostics. Qualification v3 instead retains native
predictive arrays, outcome bins, and explicit grid/truncation axes so an
independent implementation can reconstruct every numerical gate. It still
contains no PFN output and computes no scientific deficit. After qualification,
freeze the selected `T_atom`, checkpoint registry, source commit, exact seeds,
and all tolerances in a confirmatory attempt file and create an annotated local
attempt tag before generating confirmatory contexts.

The first calibration used 32 contexts per prior and one atom bank. Version 1
then used seeds `880903000..880913002`, but both used the subsequently rejected
ablated estimator. Version 2 was blocked before execution by the quadrature
audit. Qualification v3 uses 160 fresh contexts per prior and bank from C seeds
`880943000..880943002` and N seeds `880953000..880953002`; these contexts are
never reused in confirmation. Both 8,192 and 16,384 are compared directly with
32,768 on the 32/128 production grid. The lowest candidate must pass every
aggregate threshold above for every bank, prior, and predictor.
In addition, no individual context may exceed the strict p95 boundary of
`9e-4` JS or `0.009` nats absolute held-out log-probability change in any of the
48 candidate × bank × prior × predictor × metric families. With 160
zero-exceedance trials, the Bonferroni-adjusted one-sided 95%
Clopper-Pearson upper bound is 4.21% per family. Every full atom array is
content-hashed immediately after generation. All three jobs also regenerate a
common 4,096-atom canary from seed `881103999`; disagreement between canary
hashes invalidates the qualification. A resumed shard must reproduce its
recorded full-bank hash before any partial diagnostic is reused. If neither
candidate qualifies, stop without generating confirmatory contexts.
The separate v3 quadrature gate in
`PHASE1_ORDERING_QUALIFICATION_V3_PREREG.md` must also pass at all three
truncation levels against the 64/256 reference grid.

## Confirmatory validity gates

Every gate must pass before the primary contrast is interpreted.

1. **Completeness and provenance:** exact source commit/tag, config hash,
   checkpoint hashes, 18 expected prior × evaluation-draw × atom-bank shards,
   and no unexpected shard.
2. **Inference guards:** finite normalized probabilities; exact output shape;
   no fallback checkpoint load; and replay on every registered checkpoint.
   The replay panel is the first eight `shard_local_index` rows from each of the
   nine matching-prior draw × bank strata, giving 72 rows per checkpoint.
   Native 100-bin log-probabilities from singleton inference, production
   batches of 64, and reverse-batch-order inference must agree after restoring
   row order within max absolute error `1e-6`. A cyclic context-row permutation
   `roll(arange(30), 7)` must agree with the unpermuted output within max
   absolute log-probability error `1e-5`. The production scorer uses batch size
   64. Every stored log-probability vector must reconstruct normalization within
   `1e-6`.
3. **Predictive truncation:** the frozen calibration comparison passes for both
   full and ablated predictors under `C` and `N`; the v3 verified marker selects
   the lowest globally passing value from 8,192 or 16,384 against the 32,768
   reference, and every v3 quadrature gate passes. The independent v3 replay
   selected and froze 16,384 atoms before confirmatory contexts existed.
4. **Monte Carlo diagnostics:** collapsed covariance-atom ESS and context
   retained-mass distributions are reported. They are not substituted for the
   predictive-convergence gate.
5. **Ordering value:** with
   `V_P = E[NLL_ablated,P - NLL_full,P]`, the one-sided 95% context-bootstrap
   lower bound for `V_C` must exceed the total `0.001`-nat numerical clearance.
   For the Gaussian null, whose population ordering value is exactly zero, the
   entire
   two-sided 95% context-bootstrap CI for `V_N` must lie inclusively inside the
   frozen equivalence interval `[-1e-5, +1e-5]` nats: the lower endpoint is at
   least `-1e-5` and the upper endpoint is at most `+1e-5`. This margin is
   800-fold smaller than the primary effect floor and replaces the invalid
   “within two SE” rule.
6. **Oracle convergence:** the nested-half subset is exactly evaluation draw 0,
   original stream indices `0..199`, separately for each prior. This gives 200
   rows per prior, allocated across atom banks by the same `stream_index % 3`
   rule as the full panel. For this subset define
   `d_P = mean(NLL_ablated,3M - NLL_ablated,1.5M)`. The entire two-sided 95%
   bootstrap CI for direct `d_C` must lie strictly inside
   `[-0.0005, +0.0005]` nats, and the entire corresponding CI for the
   control-subtracted `d_C-d_N` must lie strictly inside that same interval.
   Also define
   `e_P = mean(NLL_full,3M - NLL_full,1.5M)` and
   `g_P = mean_models mean(NLL_net,final - NLL_full,3M)` on these same rows,
   with equal weight on the three fixed models. Separately for `C` and `N`, the
   full-predictive atom gate requires the one-sided 95% bootstrap lower bound
   for `g_P` to be positive. The entire two-sided 95% CI for `e_P` must lie
   strictly inside `[-0.0005, +0.0005]` nats, and its largest absolute endpoint
   must be less than 20% of that positive `g_P` lower bound. The 1.5M
   calculation uses the first 1,500,000 atoms of the same registered 3M bank
   and reselects top `T_atom` within that prefix.
7. **KL alarm:** for each prior, the final-checkpoint fixed-fleet mean
   `NLL_net - NLL_full` uses `-0.004` nats as the nominal alarm threshold. Its
   two-sided 95% CI upper endpoint must be at least `-0.003` to clear after
   numerical allowance. An upper endpoint below `-0.005` is an oracle alarm;
   an endpoint from `-0.005` inclusive to `-0.003` exclusive is numerically
   inconclusive and fails this validity gate. A stronger apparent improvement
   over the approximate full oracle is treated as an oracle alarm, not as
   model superiority.
8. **Fixed-fleet completeness:** all three registered models in each arm must
   contribute every checkpoint endpoint. Models cannot be dropped or replaced,
   and no gate depends on arbitrary cross-arm seed pairing.

The confirmatory tensor is stratified by prior, evaluation draw, and assigned
atom bank. The primary bootstrap uses 50,000 repetitions and RNG seed
`881003900`. It resamples contexts with replacement within each of the 18 fixed
prior × evaluation-draw × atom-bank strata, using weights proportional to the
registered stratum sizes. `C` and `N` context weights are drawn independently.
Within a prior and stratum, the same context weights are reused across all
three fixed models and all checkpoints. Evaluation draws, atom banks, and
training models are fixed and are not resampled. The three models in each arm
receive equal weight. Intervals are percentile intervals using NumPy's linear
quantile convention at `[0.025, 0.975]`; the `V_C` lower gate uses the 0.05
quantile. Each stratum has an independent `PCG64` generator initialized by
`SeedSequence([881003900, prior_code, evaluation_seed, atom_seed])`. Bootstrap
indices are generated in canonical replicate order in chunks of 256; the final
short chunk contains 80 replicates. The same within-stratum indices are reused
for every model and checkpoint. The SHA-256 of each generated index stream is
retained. Shard and draw display-label relabeling, file-discovery order, model
order, and row storage order must leave every estimate and decision unchanged.

The nested-half convergence gate uses a separate 50,000-replicate stratified
`PCG64` bootstrap. Its namespace is `1212238918`, and each generator is
initialized by `SeedSequence([881003900, 1212238918, prior_code, atom_seed])`.
The 200 rows per prior form atom-bank strata of 67, 67, and 66 rows. Indices are
generated in canonical replicate order in chunks of 256, reused for `d_P`,
`e_P`, and `g_P`, and content-hashed. It uses the same percentile intervals,
linear quantile convention, and one-sided 0.05 quantile specified above.

## Decision rules

If any validity gate fails, the result is `INCONCLUSIVE_PHASE1_INSTRUMENT` and
no ordering-use claim is made.

If all gates pass, the nominal primary effect floor remains `-0.008` nats. The
nominal diagnostic holds only when all of the following are true:

- `deficit_C(final) < -0.008` nats;
- the two-sided 95% fixed-stratum context-bootstrap CI for
  `deficit_C(final)` has upper bound below `-0.008` nats;
- `Delta(final) < -0.008` nats; and
- its two-sided 95% fixed-stratum context-bootstrap CI has upper bound below
  `-0.008` nats.

Qualification v3 supports a `0.0005`-nat top-K-plus-quadrature allowance. The
nested-half atom-bank gate contributes a separate `0.0005`-nat allowance, so
the registered total primary numerical clearance is `0.001` nats. For each of
the two final endpoints, direct `deficit_C` and control-subtracted `Delta`, the
decision endpoint is the larger of its point estimate and two-sided 95% CI
upper bound.

A positive call requires both decision endpoints to lie strictly below
`-0.009`. If either decision endpoint is at least `-0.007`, the result is
`NOT_REPLICATED_ORDERING_USE`. Every remaining combination lies in the
symmetric gray zone around the nominal `-0.008` floor and yields
`INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE`. That result requires the registered
32,768-atom, 64/256-node reference-oracle rerun under a fresh locked attempt
before any claim. The negative wording is specific to the frozen fleet and test
distribution and is not a claim that PFNs cannot use ordering information.

The secondary undertraining claim is supported only when the final checkpoint
passes the numerically cleared primary rule, at least one of the 20,000-step
direct or control-subtracted decision endpoints is at least `-0.007`,
`deficit_C(final) - deficit_C(20k) < -0.008`, and
`Delta(final) - Delta(20k) < -0.008`, with both corresponding 95% CI upper
bounds below `-0.008`. The clearance margin is not applied to these changes:
the same oracle row is subtracted at both checkpoints and cancels exactly under
the shared bootstrap indices.
The licensed wording is “undertraining can obscure the ordering-specific output
advantage in this fleet.” Failure of the early checkpoint alone is not evidence
of equivalence or absence.

## Information barrier

Calibration may expose only runtime, memory, normalization, retention, ESS,
and full-versus-half oracle differences. It must not score PFN checkpoints or
compute `deficit`, `Delta`, capture, or any scientific endpoint. Confirmatory
scripts may write raw arrays and mechanical integrity state, but the join step
is the only pipeline component allowed to compute the scientific decision.
The required seventh `verify` stage then independently reconstructs all point
estimates, bootstrap intervals, validity gates, and the decision from the raw
joined arrays and frozen configuration without importing a confirmation
pipeline module. It must agree exactly on the decision and within registered
numerical tolerances on every derived quantity before the result is licensed.
