# Phase-1 ordering confirmation amendment

Status: prospective and locked before confirmatory panel generation or checkpoint
scoring.

## Qualification lineage

The original confirmation draft pointed to qualification v1. That result is
not admissible because its ablated ordering weights were divided by the
full-bank normalizer. Qualification v2 was frozen but deliberately not run
after its 8/32-node quadrature grid failed the pre-run numerical review.

The admissible qualification is protocol v3 at source commit
`fc0b8eb48c75f7c0d1dc208b1d344d663b16baf3`, annotated tag
`phase1-ordering-qualification-v3`. It uses within-ordering normalization,
three fresh atom banks and context streams, a 32/128-node production grid, a
64/256-node quadrature reference, and a 32,768-atom truncation reference. Its
joined artifacts are bound to the independent verifier repair at commit
`04429ae5713dff0f101d842b8e6b4890c7f4e668`, annotated tag
`phase1-ordering-qualification-v3-verifier-fix1`.

The independent raw replay selected 16,384 atoms. At 8,192 atoms it retained
12 maximum-bin absolute-log-probability exceedances and no JS exceedances. At
16,384 atoms it found zero exceedances of either kind under the registered
finite-panel gates. This establishes stability relative to the 32,768-atom
reference on the frozen v3 panel. It does not establish convergence to an
exact oracle or a population-wide error bound.

## Resolution

Qualification v3 supersedes v1 and the unexecuted v2 attempt for confirmation.
It is the only qualification result that can select the confirmation
truncation, satisfy the predictive-truncation gate, or contribute a numerical
allowance. Earlier qualification artifacts are historical records only.
Confirmatory Gate 3 is therefore read as:

> The frozen calibration comparison must pass for both full and ablated
> predictors under `C` and `N`, and its verified three-bank marker must select
> the truncation used by confirmation. For this attempt, that selected
> truncation is 16,384 atoms and the production quadrature grid is 32/128.

The earlier tagged attempts remain immutable. The qualification repair itself
does not change a confirmatory seed, panel size, checkpoint, bootstrap rule, or
effect floor. The prospective hardenings below were added during the final code
audit, still before any confirmatory context or checkpoint output existed. This
amendment and the complete confirmation configuration must be committed and
tagged before any confirmatory context is generated.

## Pre-run metric hardening

A final pre-run audit found that `Delta = deficit_C - deficit_N` alone could be
negative when `deficit_C = 0` and an independently trained `N` model merely had
positive generic excess NLL. No confirmation context had been generated. The
prospective primary rule now also requires `deficit_C(final)` and its 95% CI
upper bound to be below the unchanged `-0.008`-nat floor. The secondary rule
also requires the direct final-minus-20k `deficit_C` change and its CI upper
bound below that floor. `Delta` and its change remain required controls. Every
per-prior deficit and interval is written to the result summary.

Confirmation stages are deliberately fail-closed rather than partially
resumable. An interrupted stage is retried into a new empty output directory
under the same immutable attempt identity; a sealed upstream stage may be
reused. This replaces the earlier generic resumability sentence and prevents a
retry from accepting a mixture of stale and new leaf artifacts.

## Numerical clearance

Qualification v3 measures finite-panel stability, so a nominal result barely
past `-0.008` is not licensed as a positive confirmation. Its hashed raw array
has SHA-256
`a4bc89e5f22f41764ddefb310bc2fb67dc291c0d0156949c1840841549915317`.
A fixed 50,000-replicate `PCG64` bootstrap with seed `881003900`, generated in
bank-major and prior-minor order, gives a simultaneous one-sided 95% aggregate
ablated-oracle envelope of `0.0003433511679446371` nats. Adding the worst
observed reference-weighted ablated quadrature discrepancy,
`0.000029849800733846442`, gives `0.00037320096867848353`. The registered
top-K-plus-quadrature allowance rounds this upward to `0.0005` nats.

Confirmation also applies a separate nested-half atom-bank gate. A fixed
50,000-replicate stratified `PCG64` bootstrap uses namespace `1212238918` and
requires the entire two-sided 95% CI to lie strictly inside
`[-0.0005, +0.0005]` nats for the direct `C` ablated full-minus-half change, the
control-subtracted `C-N` ablated change, and each prior's full-predictive
full-minus-half change. Each full-predictive interval must also be smaller in
absolute value than 20% of the corresponding fixed-fleet final-gap one-sided
95% lower bound, which itself must be positive.

The registered total primary numerical clearance is therefore `0.001` nats:
the `0.0005` top-K-plus-quadrature allowance plus the `0.0005` nested-half
allowance. Around the nominal `-0.008`-nat floor, a positive primary result
requires both the point and CI upper bound of direct causal deficit and
control-subtracted `Delta` to lie strictly below `-0.009`. A result is clearly
negative if either endpoint reaches or exceeds `-0.007`. Every other endpoint
combination is the symmetric numerical gray zone and yields
`INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE`, not a positive or negative claim.
The predeclared next action for that gray zone is to recompute all confirmation
oracle rows at 32,768 atoms and 64/256-node quadrature under a fresh locked
attempt. The clearance does not apply to final-minus-20k changes because the
identical oracle row cancels exactly at the two checkpoints.

The same clearance protects two validity diagnostics. The `C` ordering-value
one-sided 95% lower bound must exceed `0.001` nats, while the Gaussian-null
ordering value remains subject to its exact/equivalence gate. For the KL alarm,
`-0.004` remains the nominal threshold: an upper CI endpoint below `-0.005`
alarms, an endpoint at or above `-0.003` clears, and an endpoint from `-0.005`
inclusive to `-0.003` exclusive is numerically inconclusive and fails the
validity gate.

## Runtime, checkpoint, and submission hardening

The production oracle is explicitly constructed with float64 compute and
aborts if that dtype is not retained. Static config checks may validate the
checkpoint registry without the WashU mount, but every production stage forces
byte and SHA-256 verification of all 18 checkpoints and six training sidecars
before any scientific work. A missing mount is an error, never a reason to skip
verification.

The source-bound launcher exclusively creates one attempt root and submits the
fixed seven-stage dependency graph
`panel -> {pfn, oracle0, oracle1, oracle2} -> join -> verify`. Every stage
receives one A100, eight CPUs, 64 GB memory, explicit time limits, and `afterok`
dependencies. The launcher submits every job held, durably records every job
ID, exact argument vector, dependency, and expected stage in `SUBMISSION.json`,
marks the receipt `READY_TO_RELEASE`, then releases the recorded jobs. A
released wrapper waits for the durable `SUBMITTED` transition before doing
science and checks the same binding again before exit. On any submission or
release failure, the receipt transitions to `CANCELLING` before `scancel`.
It says `SUBMISSION_FAILED_CANCELLED` only after every recorded job has left
the live queue; a failed or unconfirmed cancellation is recorded explicitly
and cannot license an output. The launcher first quarantines any runnable
receipt, and cancellation is attempted even if the replacement receipt cannot
be persisted.

The six scientific stages receive one A100, eight CPUs, 64 GB memory, and
explicit time limits. The final NumPy-only verifier requests two CPUs and 16 GB
without reserving an idle GPU.

Slurm exports no ambient user environment. The submission client receives only
`PATH`, `LC_ALL`, `TZ`, and the site-required frozen `SLURM_CONF` path. The
isolated launcher and wrapper use the locked interpreter; the wrapper derives
all result paths from the attempt root, exclusively creates a
job-ID-specific telemetry directory, and starts Python with `-I -S` plus a
per-job telemetry `pycache` prefix before adding only the frozen repository
source and locked environment site-packages. User-site, `.pth`,
`sitecustomize`, inherited `PYTHONPATH`, and `LD_PRELOAD` startup paths are
excluded. The required final `verify` stage independently
reconstructs the estimates, intervals, gates, and decision from the raw joined
arrays and configuration without importing the confirmation pipeline modules.
No confirmatory result is licensed unless this raw replay agrees and seals its
own verification artifact.

## Licensed interpretation

This amendment resolves document precedence only. It does not make the
qualification result an independent scientific claim and does not remove the
requirement that every confirmatory validity gate pass before the Phase-1
ordering-use contrast is interpreted.
