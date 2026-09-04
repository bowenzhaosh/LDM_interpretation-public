# PFN-DAG essential verification plan, version 3

Date locked: 2026-08-03

Status: amended design artifact. The version-2 scientific stream stopped at a pre-score guard before any prediction shard or scientific metric existed. Version 3 must be committed with a new run lock before a fresh commit-derived panel is generated.

## 1. Questions and licensed claims

This run asks three ordered questions.

1. **Instrument validity:** Can an independently implemented AL40 oracle and frozen multi-query projection recover programmed two-family mixture weights on held-out contexts?
2. **Primary model question:** For the frozen fleet of 16 base AL40 checkpoints at step 12,000, do induced-coordinate differences between two length-30 contexts track the exact evidence differences produced by replacing a 10-row block while holding a 20-row core, graph, and latent SEM parameters fixed?
3. **Secondary descriptions:** How does the same fleet behave on the 20-to-30 append interface, and how do the projected metrics differ between paired step-0 and step-12,000 checkpoints?

The strongest licensed positive wording is:

> On identifiable AL40 contexts from this frozen evaluation design, the trained fleet's frozen induced-coordinate differences are compatible with exact evidence differences under a same-length replace-10 contrast.

The append result must name the tested 20-to-30 interface. The training result may only describe a paired step-0-to-step-12,000 change in specified projected metrics.

The run does **not** test, and may not claim, that a Transformer implements Bayes, performs a sequential internal computation, has a correct absolute posterior, treats evidence as sufficient, or has an architecture-specific advantage. Mean pooling is only a negative-control fixture. Causal intervention and matched trained architecture baselines are deferred.

## 2. Frozen substrate

- Family: d=2 binary graph family, equal graph prior, AL40 residuals, Fix-B latent covariance prior.
- Model: the existing base PFN architecture, evaluated without training or parameter changes.
- Checkpoints: seeds 0 through 15 at steps 0 and 12,000, paired by seed.
- Backend: CPU only, Python 3.11.7, PyTorch 2.9.1, NumPy 1.26.4, SciPy 1.12.0, one intra-op thread, one inter-op thread, deterministic algorithms enabled.
- Query banks: one calibration-selected symmetric eight-query bank and one preregistered fixed sensitivity bank `[-3.75, -2.25, -1.25, -0.25, 0.25, 1.25, 2.25, 3.75]`.

A committed 32-entry registry must contain checkpoint seed, step, size, SHA-256, state-dict key/shape schema, model config, source-code hashes, original local path, and a durable artifact URI if one exists. The loader must reject missing, extra, mismatched, or non-finite tensors. Because the original training-data lineage is incomplete, conclusions apply to these exact checkpoint files, not to an abstract training population.

## 3. Leakage firewall and seed derivation

Calibration, unit, smoke, statistical simulation, and scientific evaluation streams are disjoint.

- Unit stream: fixed labels beginning `unit-v2`.
- Query-bank calibration stream: fixed label `calibration-v2`; 512 natural contexts. It selects queries but never supplies a scientific row.
- Smoke stream: fixed label `smoke-v3`; 8 selected-interior groups by 8 continuations, both banks, and both checkpoint steps for one model seed. It may test runtime and schemas only.
- Statistical validation stream: fixed label `coverage-v2`; synthetic outcomes only.
- Production-shape validation stream: fixed seeds `860003` through `860007`; 64 length-20 core contexts, 64 length-30 contexts, and matched companion banks. It checks all 32 checkpoints and both query banks but never supplies a scientific row.
- Evaluation stream: derived only after the audited code, plan, resolved query bank, environment lock, and checkpoint registry are committed.

The evaluation root seed is the first 64 bits of `SHA256(commit_sha || "pfn-dag-essential-evaluation-v2")`. Child streams are derived by hashing the root with explicit labels for group, continuation, bootstrap, and permutation. The runner aborts if the worktree is dirty or the committed plan, registry, query bank, or code hashes differ.

Smoke output is structurally blinded: it prints guards, counts, timings, and hashes, but no PFN effect estimate. The scientific runner writes shards without reporting model metrics. Metrics are unblinded only after all shards finalize and the content tree verifies. Any result-relevant edit after unblinding invalidates the run and requires a new commit-derived evaluation stream.

## 4. Independent oracle and instrument

The new package may reuse documented equations and constants, but it must not import any legacy `experiments`, `oracle.py`, `metrics.py`, Stage-5, or report module. It contains two independent exact-evidence paths:

1. a clear scalar reference over the Fix-B quadrature grid;
2. a separately structured vectorized implementation used for the run.

On deterministic unit contexts, the two paths must agree to `1e-10` in log evidence and endpoint probabilities. They must fail on NaN, infinite, zero-normalization, or invalid-covariance inputs. No clipping, neutral replacement, stale cache, or silent fallback is permitted.

For prediction `p`, family endpoints `F0` and `F1`, and `d = F1 - F0`, the coordinate estimate is the bounded least-squares projection

`w_coord = clip(<p-F0,d>/<d,d>, 0, 1)`.

The unclipped value, denominator, boundary flag, and normalized residual `||p - ((1-w)F0+wF1)||2 / ||F1-F0||2` are retained. A separate KL estimate minimizes cross-entropy against the same mixture on `[0,1]` with a deterministic bounded scalar solver. Both operate on the identical flattened query-by-bin bank. Log odds are computed only after recording boundary status.

Endpoint identifiability is the mean Jensen-Shannon divergence between `F0` and `F1` across queries. An evaluation row is oracle-eligible only when both compared contexts have JS at least `0.1` and exact posterior weights in `[0.05,0.95]`. Eligibility is oracle-only and therefore common to every checkpoint and query-bank comparison.

## 5. Query-bank calibration

Candidate queries are the fixed grid `-4.0, -3.5, ..., 4.0`. On 512 calibration-only contexts, a deterministic greedy algorithm selects four positive magnitudes and mirrors them. At each step it maximizes the calibration 10th percentile of flattened endpoint squared distance after adding the candidate pair; ties choose the smaller magnitude. The selected bank and calibration input hash are committed before evaluation.

Selection fails if fewer than 50 percent of calibration contexts reach JS `0.1`. No evaluation result may influence the bank. The fixed sensitivity bank is never optimized.

## 6. Assert battery before model scoring

All assertions must pass before any scientific checkpoint is loaded.

1. Shapes, dtypes, normalization, covariance validity, finite values, and query-axis preservation.
2. Scientific inference fixes physical batch size at 64, and every scientific inference array must contain only complete batches. Commit-derived uniform samples of 64 length-20 cores and 64 length-30 targets are drawn without replacement across the full panel. On every one of the 32 checkpoints and both query banks, repeated batch-64 calls must be byte-identical. Permuting the batch axis and restoring it must be byte-identical. Every focal context in each length class must be byte-identical when the other 60 contexts are replaced at the same physical batch shape; 16 complementary batches cover four focal contexts each. Batch-1 versus batch-64 error is recorded descriptively and has no pass threshold because singleton inference is not part of the estimand. Context-row permutations remain subject to item 8.
3. Exact Bayes fixtures recover slope 1 and intercept 0; tempered fixtures recover `0.25`, `0.5`, and `0.75` within `0.02`.
4. Held-out planted weights `{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}` have coordinate max error `1e-4`, KL max error `1e-3`, coordinate/KL log-odds median disagreement at most `0.01`, and 95th percentile at most `0.05`.
5. Mean-pooling fixture has high free-slope fit and slope near `0.5`, demonstrating that free-slope R-squared is non-diagnostic.
6. Swapping graph labels maps `ell` and `g` to their negatives, preserves reconstructed predictions and residuals to `1e-10`, and swaps weights to `1-w`.
7. A 2,000-draw within-group permutation-null distribution has a 95 percent interval containing zero and lying within `[-0.15,0.15]`. Permutations include fixed points because a cyclic shift or forced derangement has negative expected covariance after within-group centering.
8. Context-row permutations change PFN probabilities by at most `1e-6`; otherwise all target contexts are explicitly counterbalanced and the design is revised before evaluation.
9. A wrong checkpoint hash, missing shard, repeated seed, all-zero arm, swapped query axis, and stale config each produce a nonzero exit.
10. Decision-rule tests cover every boundary and prove that mapping failure produces `INCONCLUSIVE_MAPPING`, never confirmation or refutation.

The crossed bootstrap implementation must be checked on synthetic crossed random-effects data at slopes `0.8`, `1.0`, and `1.2`. An initial calibration stream showed that ordinary 2.5/97.5 percentile intervals undercovered, reaching 0.916 at one boundary. Scientific intervals therefore use the more conservative 2/98 percentiles. On a fresh validation seed with at least 500 Monte Carlo datasets per slope, the observed coverage must be at least 0.93 and its Wilson 95 percent interval must contain 0.95. Scalar and vectorized oracles are compared on 64 deterministic unit contexts spanning both graphs and several context lengths.

## 7. Evaluation panel

Use a fixed panel of 256 **accepted identifiable-interior groups**. The calibration-only prevalence check found that only 8.8 percent of natural length-30 contexts have exact posterior weight in `[0.05,0.95]`; the earlier unselected panel would therefore have been practically empty under the repeated-continuation gate. This pre-evaluation amendment changes the target population explicitly rather than silently extending a failed natural panel.

Candidate generation is deterministic and oracle-only. Core candidate `j` and each of its block candidates use separately labeled child seeds derived from the locked commit. For each core candidate:

1. sample one valid latent covariance `Sigma_j`, graph `G_j`, and 20-row core `C_j`;
2. generate at most 512 independent 10-row block candidates under the same `Sigma_j` and `G_j`;
3. a block passes only when the length-30 context `C_j union block` has exact weight in `[0.05,0.95]` and endpoint JS at least `0.1` on both frozen query banks;
4. stop at the first nine passing blocks; the first is `A_j` and the next eight are `B_jm` in candidate order;
5. accept the core only if the eight continuation evidence values have standard deviation at least `0.25`.

The panel is the first 256 accepted cores among at most 2,000 core candidates. If 256 are not accepted under these caps, the run exits nonzero as `INCONCLUSIVE_IDENTIFIABILITY`; it may not extend the caps or analyze a partial panel. Every core and block candidate, child seed, context array, exact evidence, both-bank JS values, reason code, and acceptance rank is sealed before a checkpoint loads.

For each accepted group:

1. sample one valid latent covariance `Sigma_j` and graph `G_j`;
2. sample a 20-row core `C_j`;
3. sample one independent 10-row reference block `A_j`;
4. sample eight independent 10-row continuation blocks `B_jm`, `m=0,...,7`;
5. all rows in `C_j`, `A_j`, and every `B_jm` are iid conditional on the same `Sigma_j` and `G_j`.

No threshold or cap may change after candidate generation begins. Calibration rows never enter this panel. The scientific claim is conditional on this explicitly selected identifiable-interior population and says nothing about its natural prevalence.

The primary same-length contrast is

`D_base = C_j union A_j`, `D_target_m = C_j union B_jm`,

with `delta_ell_replace = ell(D_target_m) - ell(D_base)` and `delta_g_replace = g(D_target_m) - g(D_base)`.

The secondary append contrast is

`D_prefix = C_j`, `D_target_m = C_j union B_jm`,

with `delta_ell_append = ell(D_target_m) - ell(D_prefix)` and `delta_g_append = g(D_target_m) - g(D_prefix)`.

All 256 accepted groups must have all eight replace-10 continuations eligible on both query banks by construction. The runner verifies that the accepted IDs are exactly the first 256 passing the frozen predicate. Append remains secondary and uses the oracle-eligible subset of cores because core length 20 was not part of replace selection.

## 8. Estimands

The primary endpoint is the within-group centered slope for the replace-10 contrast. For every eligible group, subtract that group's mean from `delta_ell` and from each checkpoint's `delta_g`, then estimate the pooled unit-update slope across the frozen 16-model fleet. This forces identification from the realized differences among continuations under a fixed core, graph, and latent SEM.

The continuation-swap canary uses 2,000 commit-derived, independently uniform permutations of the eight exact-evidence labels in each group. The resulting permutation-null slope distribution must have a 95 percent interval containing zero and lying wholly inside `[-0.15,0.15]`. Permutations include fixed points so the centered null expectation is zero.

Secondary metrics are:

- zero-intercept identity NRMSE, normalized by the standard deviation of exact differences;
- identity NMAE, normalized by the exact-difference IQR;
- coordinate/KL log-odds disagreement and boundary rate;
- native mixture residual at base and target;
- native sequential-reconstruction residual, formed by updating fitted base odds with exact `delta_ell` and reconstructing the target prediction in the target endpoint frame;
- the same quantities for the append contrast;
- paired step-12,000 minus step-0 differences in identity NRMSE.

Free-slope R-squared is descriptive only and cannot enter a verdict.

## 9. Uncertainty

The sample size is fixed at the first 256 accepted groups by 8 continuations and 16 model seeds under the caps above. There is no effect-dependent stopping or post-hoc extension.

Use 10,000 deterministic crossed bootstrap replicates. Each replicate draws model-seed indices and group indices independently with replacement. The same selected group instances, including their full continuation clusters, are used across every model seed, step, and query bank. Paired checkpoint seeds remain paired for training contrasts. Every reported confirmatory interval uses the calibrated 2/98 percentiles from this procedure.

The population statement is limited to the frozen 16-checkpoint fleet and the generated AL40 panel. It does not estimate variation across architectures or unseen training runs.

## 10. Instrument and mapping gates

The scientific model verdict is reachable only if all fixture assertions pass and the scalar/vector oracle agreement passes.

For the trained step-12,000 fleet, each query bank must also meet all mapping gates on the common eligible panel:

- fitted-weight boundary rate exactly zero on the analyzed primary rows. A clipped boundary logit is an arbitrary epsilon-dependent number and may not enter a terminal slope verdict;
- coordinate/KL log-odds disagreement median at most `0.10` and 95th percentile at most `0.30`;
- normalized native mixture residual median at most `0.10` and 95th percentile at most `0.30`, separately at base and target;
- native sequential-reconstruction residual median at most `0.10` and 95th percentile at most `0.30`.

These bounds define adequacy relative to the full family-endpoint separation. Failure returns `INCONCLUSIVE_MAPPING`. It is not evidence for or against the induced-coordinate identity.

## 11. Primary decision rule

The claim margin is a preregistered 20 percent deviation from the exact unit gain: `[0.8,1.2]`. This is the only primary refutation endpoint.

- `COMPATIBLE_ON_TESTED_REGIME` requires: all instrument and mapping gates pass; the true-pair canary requirement passes; the 95 percent slope interval lies wholly within `[0.8,1.2]`; the identity NRMSE 95 percent upper bound is at most `0.35`; and both query banks meet these conditions.
- `INCOMPATIBLE_ON_TESTED_REGIME` requires: all instrument and mapping gates pass; the true-pair canary requirement passes; the 95 percent slope interval lies wholly outside `[0.8,1.2]`; and both banks agree on the side of the deviation.
- Every other outcome is `INCONCLUSIVE`, with a specific reason such as identifiability, mapping, query-bank sensitivity, canary failure, or precision.

NRMSE, reconstruction error, intercept, and individual-seed counts may block confirmation but cannot independently trigger incompatibility. This avoids a disjunctive refutation rule.

The append and step-0 comparisons receive estimates and 95 percent intervals but no confirmatory binary label in this first essential run. Append wording is limited to the 20-to-30 interface. The paired training metric is named exactly as a change in projected identity NRMSE; it cannot be described as learning Bayes or beating an architecture null.

## 12. Raw artifacts and replay

Every group shard is written atomically and incrementally. Raw artifacts include every accepted and rejected core and block candidate, candidate ordinal and child seed, rejection reason, contexts, graph label, covariance and SEM parameters, all exact evidence values, both query banks, both endpoint arrays, every checkpoint probability tensor, coordinate and KL fits, unclipped weights, residuals, eligibility masks, checkpoint registry, resolved config, logs, environment fingerprint, git revision, and seed derivation record. NPZ files use numeric arrays only and disallow pickle.

After the panel commit is decoded, rejected-stream denylist is checked, and the lexical commit-named path is verified, a `RUNNING` pre-score record is written before run-lock, registry, or checkpoint validation can abort. Normal completion replaces it with the full record. Any Python exception during that gate replaces it with an `ERROR` record before being re-raised. A failed or interrupted guard is terminal for that commit stream. A byte-identical `COMPLETE` and passing guard may support shard recovery, but it is re-adjudicated semantically and never rewritten.

Scoring and derivation each maintain an atomic cumulative attempt record. The score record starts with the persisted pre-score guard wall time and peak memory before any checkpoint attempt is appended, closing the crash window between guard completion and shard scoring. Every completed checkpoint pair or derived shard, and every caught exception, updates elapsed wall time, peak memory, and validated shard identities. A resumed invocation preserves prior attempt cost, so sealing cannot pass by reporting only the final invocation. Panel metadata is producer-bound and its resource fields must be finite and nonnegative. Finalization verifies every shard and both cumulative histories, writes file-level SHA-256 values, computes one content-tree hash, and creates a content-addressed archive with all raw material needed to recompute the summary. The archive includes a Git bundle for the exact scientific commit and restoration instructions for a Gitless extraction. Canonical summarization always verifies that content-addressed archive and is the only path that may issue `LOCALLY_VERIFIED`. A Gitless extraction uses the separate `analysis replay` command, writes `replay_summary.json`, and reports `REPLAY_VERIFIED_NONCANONICAL`; it cannot bypass the canonical archive gate. Compact code, plan, registry, hashes, and summary are committed locally. Publication of checkpoints and the raw archive to a durable external URI is a separate outward-facing action requiring user authorization. Until that happens, the status is `LOCALLY_VERIFIED`, never `INDEPENDENTLY_REPRODUCIBLE`.

## 13. Measured budget and stop conditions

Pre-plan measurements on this machine were approximately 11,237 oracle triplets per minute, 532 PFN contexts per second on CPU, and 1,069 contexts per second on MPS. Scientific inference uses CPU for determinism. The fixed panel requires 2,560 unique contexts per query bank and 32 checkpoints, or 163,840 model context-bank evaluations. The expected model pass is about 6 minutes before I/O and checks.

Before evaluation, one end-to-end smoke batch must run the same selected-interior path with 8 accepted groups, 8 continuations, both query banks, and both checkpoint steps for one model seed. It measures wall time, peak resident memory, compressed bytes, the exact current-commit Git-bundle size, and the tracked repository-tree size, then extrapolates panel size and the full 64-shard fleet with a 25 percent safety factor. The scientific cap is 45 wall-clock minutes, 16 GB peak RAM, and 2 GB sealed-archive storage. Final sealing records and enforces the actual tar-file size, including the manifest, repository members, replay material, and tar overhead. The run aborts with `BLOCKED_COST` before panel generation and again before unblinding if smoke extrapolation or measured resources exceed any cap. Missing checkpoints, failed guards, dirty code, hash mismatch, non-finite values, or incomplete shards stop the run immediately.

The smoke keeps its preregistered 8 by 8 selected-interior scoring panel and uses a separate deterministic 64-context bank solely to exercise both physical production shapes. Validation independently recomputes the wall, memory, and storage projections from persisted timing and byte components, including the measured replay Git bundle and duplicate tracked repository tree. Reported projections cannot certify themselves.

## 14. Execution order

1. Implement assertions and decision-rule tests.
2. Implement the independent oracle, instrument, design generator, model loader, storage, and crossed statistics.
3. Run unit tests and structural smoke only.
4. Complete the adversarial code audit and fix every blocker.
5. Select and commit the query bank, registry, environment, plan, and audited code.
6. Derive the one-shot evaluation stream from that commit and generate the fixed panel.
7. Run oracle and fixture gates. Stop if any fails.
8. Run checkpoint scoring without printing scientific estimates.
9. Verify and seal raw shards.
10. Unblind once, compute the preregistered summary, and write the honest claim ledger.

## 15. Version-3 amendment record

The immutable version-2 commit was `d0b049d6241845e55443f4950e52b70644b2b1ab`. Its oracle-only panel completed, but scoring stopped before creating a prediction directory. The fatal gate compared batch sizes 1 and 64 at an absolute probability tolerance of `1e-6`. One of 64 checkpoint-bank records measured `1.1324882507324219e-6`, at seed 4, step 12,000, sensitivity bank. Batch-64 replay was byte-identical and the fleet context-row permutation maximum was `5.066394805908203e-7`.

Forensic isolation found that batch size 1 differs while batch sizes 2, 4, 8, 16, 32, and 64 are byte-identical for the worst context. The first difference occurs in float32 `query_embed`; float64 reduces the probability discrepancy to `7.77e-16`. Replacing every companion context or moving the focal context at fixed physical batch size 64 changes its output by exactly zero. The old gate therefore tested physical-shape rounding outside the fixed-batch-64 estimand. Its cited calibration maximum was also absent from the locked validation artifacts.

The failed version-2 stream remains `BLOCKED_GUARD`. It produced no model outcome and licenses no scientific conclusion. Version 3 does not widen the observed threshold or reuse that panel. It replaces the non-estimand fatal comparison with the independently seeded fixed-production-shape validation in section 3, locks live PyTorch inference settings and the installed distribution payload trees, persists the pre-score guard before any run-lock or registry abort, cumulatively accounts for resumed scoring, seals an exact-commit Git bundle, and requires a new commit-derived evaluation stream.
