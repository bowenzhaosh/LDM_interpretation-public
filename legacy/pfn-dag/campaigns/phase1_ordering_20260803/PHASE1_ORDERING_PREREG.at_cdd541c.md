# Phase-1 ordering-use replication preregistration

Status: **CALIBRATION COMPLETE; CONFIRMATORY PROTOCOL NOT YET LOCKED**

The registered calibration at commit `e679e74930fc990dc834e0a03aec0a10b582a862`
passed and selected `T_atom = 8,192` against the 32,768-atom reference. No PFN
checkpoint or scientific endpoint was evaluated during calibration.

This protocol is for the first output-level claim that remains scientifically
independent of the failed Phase-2 induced-coordinate instrument. No projected
mixture weight, logit evidence coordinate, composition slope, probe, or
activation intervention appears in this experiment.

## Claim and scope

Primary claim:

> In the exact archived d=4 base fleet, the three causal-AL(r=4) models at
> 120,000 steps have a larger ordering-specific output advantage than the three
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

For each prior, the deficit is first averaged equally over its three fixed
models and then over fresh held-out contexts. The `C` and `N` training seed
labels do not define paired randomizations and are never paired or resampled.

`N` is the matched Gaussian control for which ordering is non-identifiable and
`V_N = E[NLL_ablated,N - NLL_full,N]` is zero in the population. A negative
`Delta` means that the causal-prior PFN beats the ordering-blind reference by
more than the matched generic output mismatch observed under `N`.

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
and `pip check`. OpenBLAS is fixed to one thread. A resumable attempt is bound
to that exact fingerprint and has one atomic writer lease.

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

No confirmatory result may be read until all expected shards are present and
their hash inventory passes.

## Calibration-only truncation selection

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
semi-infinite change of variables. No missing tail mass is renormalized away.

If neither candidate passes, stop. Calibration artifacts persist only the
allowed convergence, retention, ESS, normalization, timing, and memory
diagnostics. They do not retain contexts, queries, outcomes, outcome bins,
ordering labels, or predictive arrays. After calibration, freeze the selected
`T_atom`, checkpoint registry, source commit, exact seeds, and all tolerances in
a confirmatory attempt file and create an annotated local attempt tag before
generating confirmatory contexts.

The first calibration used 32 contexts per prior and one atom bank. It selects
the candidate under the registered finite-sample rule, but it is not by itself
a population-tail or cross-bank guarantee. Before any confirmatory context is
generated, the selected ladder is therefore qualified on the three atom banks
that will be frozen for confirmation. Qualification uses 160 fresh contexts
per prior per bank from the explicit seeds `880903000..880903002` for `C` and
`880913000..880913002` for `N`; these contexts are never reused in confirmation.
Both 8,192 and 16,384 are compared directly with 32,768. The lowest candidate
must pass every aggregate threshold above for every bank, prior, and predictor.
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

## Confirmatory validity gates

Every gate must pass before the primary contrast is interpreted.

1. **Completeness and provenance:** exact source commit/tag, config hash,
   checkpoint hashes, 18 expected prior × evaluation-draw × atom-bank shards,
   and no unexpected shard.
2. **Inference guards:** finite normalized probabilities; exact output shape;
   batch-size replay within `1e-6`; context-row permutation replay within
   `1e-5`; no fallback checkpoint load.
3. **Predictive truncation:** the frozen calibration comparison passes for both
   full and ablated predictors under `C` and `N`; its verified marker selects
   `T_atom = 8,192` against the 32,768 reference.
4. **Monte Carlo diagnostics:** collapsed covariance-atom ESS and context
   retained-mass distributions are reported. They are not substituted for the
   predictive-convergence gate.
5. **Ordering value:** with
   `V_P = E[NLL_ablated,P - NLL_full,P]`, the one-sided 95% context-bootstrap
   lower bound for `V_C` is positive. For the Gaussian null, the entire
   two-sided 95% context-bootstrap CI for `V_N` must lie inside the frozen
   equivalence interval `[-1e-5, +1e-5]` nats. This equivalence margin is
   800-fold smaller than the primary effect floor and replaces the invalid
   “within two SE” rule.
6. **Oracle convergence:** on the frozen nested-half subset, the absolute
   full-bank versus half-bank change in the control-subtracted ablated NLL is
   below 0.004 nats. The full-predictive atom check must also satisfy the legacy
   20%-of-final-gap rule.
7. **KL alarm:** for each prior, the final-checkpoint fixed-fleet mean
   `NLL_net - NLL_full` must not have a two-sided 95% CI wholly below `-0.004`
   nats. A stronger apparent improvement over the approximate full oracle is
   treated as an oracle alarm, not as model superiority.
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
quantile. Shard and draw relabeling must leave every estimate and decision
unchanged.

## Decision rules

If any validity gate fails, the result is `INCONCLUSIVE_PHASE1_INSTRUMENT` and
no ordering-use claim is made.

If all gates pass, the primary claim is `REPLICATED_ORDERING_USE` only when:

- `Delta(final) < -0.008` nats; and
- its two-sided 95% fixed-stratum context-bootstrap CI has upper bound below
  `-0.008` nats.

Otherwise the primary claim is `NOT_REPLICATED_ORDERING_USE`. This wording is
specific to the frozen fleet and test distribution and is not a claim that PFNs
cannot use ordering information.

The secondary undertraining claim is supported only when the final checkpoint
passes the primary rule, the 20,000-step checkpoint does not pass that rule,
and `Delta(final) - Delta(20k) < -0.008` with a 95% CI upper bound below
`-0.008`.
The licensed wording is “undertraining can obscure the ordering-specific output
advantage in this fleet.” Failure of the early checkpoint alone is not evidence
of equivalence or absence.

## Information barrier

Calibration may expose only runtime, memory, normalization, retention, ESS,
and full-versus-half oracle differences. It must not score PFN checkpoints or
compute `deficit`, `Delta`, capture, or any scientific endpoint. Confirmatory
scripts may write raw arrays and mechanical integrity state, but the join step
is the only component allowed to compute the scientific decision.
