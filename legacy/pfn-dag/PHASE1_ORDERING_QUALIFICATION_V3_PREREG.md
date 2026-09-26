# Phase-1 corrected oracle qualification v3

Status: **FROZEN BEFORE EXECUTION**

Version 1 used a wrong ordering-ablation normalization. Version 2 corrected
that estimand but was blocked before execution when its 8-interior/32-tail
quadrature proved insufficiently qualified. Neither version-2 contexts nor
PFN outputs were generated. Version 3 therefore uses a fresh context namespace
and qualifies both top-K truncation and quadrature before confirmation.

## Corrected ordering ablation

For every ordering, retained top-K log likelihoods are normalized over their
own retained atoms before the 24 orderings are mixed uniformly. A positive
control with unequal retained masses must give every ordering equal total
weight. Each within-ordering log normalizer must equal zero within `1e-12`.

## Frozen sampling and numerical design

- Dimension 4, context size 30, priors `C` and `N`.
- Three 3,000,000-atom banks from seeds `881003101..881003103` and a common
  4,096-atom determinism canary from seed `881103999`.
- Fresh C context streams `880943000..880943002`; fresh N streams
  `880953000..880953002`.
- 160 contexts per prior and bank.
- Top-K candidates 8,192 and 16,384, with direct reference 32,768.
- Frozen production quadrature 32 interior nodes per finite bin and 128 nodes
  per infinite tail.
- Frozen reference quadrature 64 interior nodes and 256 tail nodes.
- Every production/reference grid comparison is evaluated at all three top-K
  levels, not only at the truncation reference.
- Oracle covariance transforms, contexts, queries, quadrature nodes, residuals,
  and likelihoods are evaluated in float64. Stored float64 outputs are not
  produced from hidden float32 residual arithmetic.

A read-only instrument pilot selected the production grid before this protocol
was frozen. It used 1,024 atoms, top-K 512, 20 contexts per prior, atom seed
`1803`, and context seeds `9821/9822`; none are reused here. Against 64/256,
8/32 shifted causal ordering value by up to `0.001382` nats. For 32/128, the
maximum causal observed-bin log-probability shift was `0.000108`, maximum
ordering-value shift was `0.000102`, and maximum JS was `2.40e-10`. On bins
with reference probability at least `1e-8`, maximum binwise log-probability
error was `0.000109`; the largest reference-weighted mean absolute binwise log
error was `0.0000348`. The 16/64 grid exceeded the prospective `0.0005`
binwise maximum in the pilot and was not selected.

## Quadrature gate

The raw schema has an explicit grid axis with values `[(32,128),(64,256)]` and
an explicit top-K axis `[8192,16384,32768]`. For every bank, prior, top-K level,
predictor, and context, compare the production and reference 100-bin vectors.
The gate is outcome-blind and requires:

- JS divergence at most `1e-7`;
- maximum absolute binwise log-probability change, restricted to bins whose
  reference probability is at least `1e-8`, at most `0.0005`;
- reference-weighted mean absolute binwise log-probability change at most
  `0.0001`; and
- maximum absolute binwise change in `log p_full - log p_ablated`, restricted
  to bins where both reference predictors are at least `1e-8`, at most
  `0.0005`.

All archived probabilities must be finite, strictly positive, and normalized
within `1e-8`. A failure at any top-K level stops the attempt. The final
renormalization is not treated as evidence of quadrature accuracy; accuracy is
established only by direct agreement with the higher-order conditional
predictive. A proposed pre-normalization mass check was rejected because the
integral over the response coordinate equals the query-covariate marginal,
not the context-weight normalizer.

## Truncation gate

Only after the quadrature gate passes, compare each top-K candidate directly
with 32,768 using the frozen 32/128 production grid. The version-1 aggregate
thresholds and 10% numerical-indifference rule remain unchanged. In addition,
there may be zero individual exceedances of `9e-4` JS or `0.009` nats absolute
observed-bin log-probability change in any of the 48 registered families. The
lowest globally passing candidate is selected. If none passes, confirmation
stops.

## Replay and information barrier

Each shard archives outcome bins, full and ablated 100-bin probabilities with
explicit grid and top-K axes, grid definitions, derived truncation diagnostics,
and derived quadrature diagnostics. It contains no PFN output and computes no
scientific deficit. A standalone verifier that imports none of the oracle,
metric, or join code must reconstruct every diagnostic and decision from the
native arrays, verify all file/source/config hashes, and verify its own source
against the tagged commit. Any disagreement invalidates qualification.

Version 3 receives an annotated source tag before submission. A separate result
tag is created only after three shards, the join, independent verification,
Slurm accounting, utilization records, and a local hash manifest are complete.
