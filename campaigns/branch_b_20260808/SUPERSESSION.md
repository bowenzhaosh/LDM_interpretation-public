# Branch B — supersession note (corrected campaign)

*Status: Branch B is SEALED and preserved unmodified for provenance. This note
records that a corrected re-implementation supersedes it scientifically. No
sealed Branch B artifact has been modified, deleted, rewritten, or regenerated.
All old artifacts (evaluation JSONs, libraries, model checkpoints,
TERMINAL_STATUS.json, REPORT.html) remain byte-identical and are NOT to be
re-interpreted as evidence for any corrected conclusion.*

## Corrected campaign identity

The corrected work lives under a NEW namespace:

- Campaign: `campaigns/corrected_20260812/`
- Source modules: `src/pfn_dag_verify/corrected_*.py`
- Test modules: `tests/test_corrected_*.py`
- Model/campaign artifacts: `campaigns/corrected_20260812/`

The old `branch_b_*` campaign identity is NOT reused and nothing is silently
"fixed" under it.

## Why Branch B is invalidated

The sealed Branch B conclusion `PREDICTIVE_CAPTURE_WITHOUT_POSTERIOR_FIDELITY`
cannot be taken as evidence about whether a PFN exploits causal-ordering
information, for four independent technical defects:

1. **SEM generation used the inverse triangular map where the frozen generator
   requires the forward map.**
   The frozen d=4 generator (`artifacts/phase1/d4_generator.py`, `gen_data`)
   generates contexts as `xpi = e @ L_unit.T` (forward map). Branch B's
   training and evaluation context generation used `xpi = e @ U.T` with
   `U = inv(L_unit)` (`branch_b_train.gen_batch_from_library`,
   `branch_b_eval.generate_contexts`). This changes the covariance of the
   generated data: `Cov(e @ U.T) != Cov(e @ L_unit.T)`, so the contexts did
   not come from the intended (sigma, ordering, prior) model, and the frozen
   SEM contract was not satisfied.

2. **The predictive oracle failed to condition covariance-atom weights on the
   context likelihood `p(D | k, o)`.**
   In `branch_b_oracle.order_predictive`, the order-conditioned predictive was
   computed as an unweighted mean over atoms (`log mean_k`), i.e. atom weights
   stayed at the uniform prior rather than being re-weighted by
   `p(k | D, o) = p(D | k, o) / sum_k' p(D | k', o)`. The order weights used the
   context but the atom weights did not. This is exactly the two-stage
   calculation the corrected posterior must not reproduce.

3. **The ordering-value functional was not the expected log-score / KL
   quantity.**
   `branch_b_oracle.full_and_ablated` defined
   `V = mean over the 100 predictive bins of (-log p_ablated + log p_full)`,
   an unweighted arithmetic average over log bin ratios. It is not an
   expectation of the log-score difference under the outcome distribution
   (which is an expected log-score / KL quantity) and is not guaranteed
   nonnegative. The corrected definition is the expected log-score difference.

4. **Reported PFN structural/order metrics were placeholders, not
   model-derived quantities.**
   In `branch_b_eval.evaluate_checkpoint`, `V_net` was set to
   `nll_ablated_exact - nll_full_exact` (the exact oracle's own value; the
   comment `# TODO: PFN ordering-specific output` marks it as unimplemented),
   and `order_js` was hardcoded to `0.0` for every context. Neither is a
   quantity derived from the PFN's own outputs, so the reported
   "posterior fidelity" and "ordering capture" metrics do not measure what
   their names claim.

## What the sealed numbers are allowed to mean

The sealed Branch B numbers remain a valid record of *what that (defective)
pipeline produced*. They are NOT licensed as evidence for or against any
claim about PFN posterior fidelity, ordering exploitation, or predictive-
versus-structural Bayes separation. Any such claim must be re-derived from the
corrected campaign under `campaigns/corrected_20260812/`.

## Gates for treating the corrected campaign as evidence

The corrected campaign is only interpretable after the mandatory invariant
test suite (`tests/test_corrected_*.py`) passes in full, covering SEM algebra,
exact-Bayes identities, proper scoring, numerical agreement
(scalar/vectorized, float64 golden fixtures), and provenance completeness.
Results are reported per fixed evaluation panel with hashes and config
recorded; no placeholder metrics and no silent exception swallowing.
