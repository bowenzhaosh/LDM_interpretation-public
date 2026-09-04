# Phase-1 ordering-use replication preregistration

Status: **CALIBRATION PROTOCOL, NOT YET LOCKED FOR CONFIRMATORY DATA**

This protocol is for the first output-level claim that remains scientifically
independent of the failed Phase-2 induced-coordinate instrument. No projected
mixture weight, logit evidence coordinate, composition slope, probe, or
activation intervention appears in this experiment.

## Claim and scope

Primary claim:

> By 120,000 training steps, base d=4 PFNs trained on the causal AL(r=4)
> prior exploit predictive information associated with causal ordering.

The operational claim is restricted to the archived `nets4_xlong` checkpoint
fleet, context size 30, the native 100-bin output head, and the frozen d=4
synthetic generator. It is not a claim that the model explicitly represents a
Bayesian posterior over orderings.

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
- Matched training seeds: 0, 1, and 2.
- Checkpoint directory on WashU: `nets4_xlong`.
- Expected names: `M4_<prior>_s<seed>_st120000[_ck<step>].pt`.

Every checkpoint and the vendored fleet module must be SHA-256 registered
before confirmatory contexts are generated. Missing, additional, stale, or
silently skipped checkpoints halt the run.

The Phase-1 CUDA run is not installed from this repository's macOS mapping
requirements. It uses the separate WashU interpreter and package lock under
`environment/phase1-washu-*`. Before a production attempt, the runner verifies
the Python executable, NumPy and Torch payload trees, NumPy/Torch build
configuration, CUDA driver, GPU model and capability, deterministic settings,
and `pip check`. A resumable attempt is bound to that exact fingerprint and has
one atomic writer lease.

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

## Confirmatory validity gates

Every gate must pass before the primary contrast is interpreted.

1. **Completeness and provenance:** exact source commit/tag, config hash,
   checkpoint hashes, 18 expected prior × evaluation-draw × atom-bank shards,
   and no unexpected shard.
2. **Inference guards:** finite normalized probabilities; exact output shape;
   batch-size replay within `1e-6`; context-row permutation replay within
   `1e-5`; no fallback checkpoint load.
3. **Predictive truncation:** the frozen calibration comparison passes for both
   full and ablated predictors under `C` and `N`.
4. **Monte Carlo diagnostics:** collapsed covariance-atom ESS and context
   retained-mass distributions are reported. They are not substituted for the
   predictive-convergence gate.
5. **Ordering value:** the 95% hierarchical-bootstrap lower bound for `V_C` is
   positive; measured `V_N` lies within plus or minus two bootstrap SE of zero.
6. **Oracle convergence:** on the frozen nested-half subset, the absolute
   full-bank versus half-bank change in the control-subtracted ablated NLL is
   below 0.004 nats. The full-predictive atom check must also satisfy the legacy
   20%-of-final-gap rule.
7. **KL alarm:** the final-checkpoint mean `NLL_net - NLL_full` 95% lower bound
   is at least `-1e-6` for both priors.
8. **Seed agreement:** all three per-training-seed final `Delta` estimates are
   negative.

The bootstrap resamples training seeds, evaluation draws, atom banks, and
contexts at their actual clustering levels. `C` and `N` contexts are resampled
independently; the same sampled training seeds are used for both priors.
Bootstrap repetitions and RNG seeds are frozen in the attempt file.

## Decision rules

If any validity gate fails, the result is `INCONCLUSIVE_PHASE1_INSTRUMENT` and
no ordering-use claim is made.

If all gates pass, the primary claim is `REPLICATED_ORDERING_USE` only when:

- `Delta(final) < -0.008` nats; and
- its two-sided 95% hierarchical-bootstrap CI has upper bound below zero.

Otherwise the primary claim is `NOT_REPLICATED_ORDERING_USE`. This wording is
specific to the frozen fleet and test distribution and is not a claim that PFNs
cannot use ordering information.

The secondary undertraining claim is supported only when the final checkpoint
passes the primary rule, the 20,000-step checkpoint does not pass that rule,
and `Delta(final) - Delta(20k) < -0.008` with a 95% CI upper bound below zero.
The licensed wording is “undertraining can obscure the ordering-specific output
advantage in this fleet.” Failure of the early checkpoint alone is not evidence
of equivalence or absence.

## Information barrier

Calibration may expose only runtime, memory, normalization, retention, ESS,
and full-versus-half oracle differences. It must not score PFN checkpoints or
compute `deficit`, `Delta`, capture, or any scientific endpoint. Confirmatory
scripts may write raw arrays and mechanical integrity state, but the join step
is the only component allowed to compute the scientific decision.
