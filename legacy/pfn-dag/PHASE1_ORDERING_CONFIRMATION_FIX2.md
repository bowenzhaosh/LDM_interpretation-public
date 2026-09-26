# Phase-1 confirmation FIX2: bounded PFN replay and join parity

## Scope and status of attempt 2

Confirmation attempt 2 used source commit
`2157aa08e076c90af4b715efc1f89e3b3a053e58` and annotated tag
`phase1-ordering-confirmation-v1-fix1`. Its panel stage completed. The PFN
stage stopped on checkpoint `C`, seed 0, step 20,000 because the registered
batch replay tolerance was too strict. It stopped before writing any PFN
prediction shard. The oracle jobs were cancelled, the join and independent
verifier never ran, and no scientific endpoint or decision was computed.

The immutable attempt, failure trace, diagnostic sources, raw diagnostic
outputs, accounting, and checksum manifest are archived under
`campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix1_attempt2_failed/`.

## Label-free diagnosis

The replay diagnostics loaded only panel contexts and queries. They did not
load outcomes, outcome bins, covariance labels, oracle outputs, joined arrays,
or scientific endpoints.

Jobs `160038` and `160040` ran the same all-checkpoint diagnostic in separate
fresh processes after the production attempt-identity and locked-runtime
checks. Their outputs are byte-identical, with SHA-256
`d4f0ded169899b6cfc7054aeacdb7b05289f85b5220c5158ae102e4c05e3bc47`.
The runtime fingerprint was
`44394126ccf9b4d75eec6d1c3d691f0ab34fbbd50c579deed3abdcb06296636a`.

Job `160041` tested all 3,201 panel rows for each of all 18 registered
checkpoints. Its raw output SHA-256 is
`3fe0de767068bf233bb9443aa5b61705f65a849a05f8e7113744daff28d8cf23`.
Across 162 checkpoint-shards, repeat, reversed-row, same-shape block, and two
fixed random row-permutation comparisons were bit-identical. Fixed-shape
companion replacement, focal relocation, and remainder sizes 35, 36, and 43
were also bit-identical in the registered stress panel. This rules out detected
cross-row coupling and row-realignment errors under the locked runtime.

Changing the CUDA kernel shape to batch size 8 produced a maximum absolute
log-probability change of `3.4332275390625e-5`, maximum probability change of
`5.024417268906234e-7`, and maximum total variation of
`1.996038085179163e-6`. Rolling the 30 context rows produced maxima of
`1.9073486328125e-5`, `3.0763227687469197e-7`, and
`9.077932729430932e-7`, respectively. View and contiguous-copy context rolls
had the same recorded maxima. These observations are consistent with bounded float32
reduction and kernel-shape effects, not semantic dependence on companion
examples.

## Prospective replay bounds

Before any retry endpoint is computed, FIX2 freezes these fail-closed bounds:

- batch-shape maximum absolute log-probability change: `5e-5` nats;
- context-roll maximum absolute log-probability change: `3e-5` nats;
- combined context-roll plus batch-8 maximum absolute log-probability change:
  `8e-5` nats;
- maximum absolute probability change across every approximate comparison,
  including the combined comparison: `1e-6`;
- maximum total variation across every approximate comparison, including the
  combined comparison: `3e-6`;
- exact repeat, same-shape permutation, fixed companion replacement, and
  production-remainder checks: bit-identical.

The separate log-probability caps are prospective envelopes of 1.46 times the
observed batch-shape maximum and 1.57 times the observed context-roll maximum.
The combined cap is their sum. The earlier diagnostic did not measure the
combined transformation, so the production guard must measure it directly,
using both view and contiguous-copy rolls at batch size 8, on every panel row.
The probability bound is the next power-of-ten ceiling and is 1.99 times its
observed separate-comparison maximum. The total-variation bound is the next
`1e-6` grid value above its observed separate-comparison maximum and is 1.50
times that maximum. A combined comparison that exceeds any frozen cap aborts
the fleet before outcomes or scientific endpoints are opened.

## Decision-sensitivity propagation

The replay log-probability bound is propagated separately from the existing
oracle clearance. For a per-row PFN NLL bound `e = 8e-5`:

- a direct PFN-minus-oracle deficit or gap has replay bound `e`;
- a causal-minus-control deficit has replay bound `2e = 1.6e-4`;
- a direct final-minus-early checkpoint change has replay bound
  `2e = 1.6e-4`;
- the causal-minus-control final-minus-early contrast has replay bound
  `4e = 3.2e-4`.

Each decision endpoint must clear the scientific boundary by its own oracle
clearance plus its applicable replay bound. Checkpoint-change endpoints, for
which the identical oracle row cancels algebraically, must clear by the replay
bound alone. A value inside any resulting band produces an inconclusive
decision, not a positive or negative claim.

## Additional harness repairs

The panel's relative-only float64 covariance symmetry rule is shared with the
join so the join cannot reject an unchanged panel that the producer validly
sealed. The guard remains

`abs(Sigma[i,j] - Sigma[j,i]) <= 4 * eps64 * max(abs(Sigma[i,j]), abs(Sigma[j,i]))`,

with zero absolute tolerance. Positive-definiteness and the frozen fleet
validity-region checks remain unchanged.

The PFN unit fixture now uses bin-varying logits so row identity survives
`log_softmax`. Deliberately batch-coupled and context-order-sensitive fixtures
must be rejected by the replay guard.

## Retry identity

The repaired attempt must use annotated tag
`phase1-ordering-confirmation-v1-fix2`, a clean checkout at that tag, and a new
empty attempt root. Attempt 2's panel is diagnostic evidence only and cannot be
reused because its marker and every shard bind the FIX1 identity. FIX2
regenerates the panel from the unchanged scientific seeds. The prior panel may
be used only as a post-generation payload cross-check after excluding identity
bytes, never as a FIX2 input artifact.

This document takes precedence over the earlier confirmation amendments only
for the replay guard, replay decision-sensitivity bands, covariance validation
parity, regression fixtures, and retry identity. It changes no scientific
seed, context, generator formula, checkpoint, oracle, estimand, bootstrap,
effect floor, or original oracle numerical clearance.
