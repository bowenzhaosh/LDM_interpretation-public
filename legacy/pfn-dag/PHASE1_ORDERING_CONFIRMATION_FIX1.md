# Phase-1 ordering confirmation repair 1

Status: prospective and locked after attempt 1 failed, before retry panel
generation or any checkpoint scoring.

## Attempt 1 record

Confirmation attempt 1 used source commit
`38b6167bf97777e9eaf516cb972f923a4113157e` and annotated tag
`phase1-ordering-confirmation-v1`. The panel job, Slurm job `159868`, failed
after 45 seconds while validating the first generated context stream. The
other six held jobs were cancelled by the `afterok` dependency graph and used
zero runtime. No panel was sealed, no checkpoint was scored, no oracle output
was written, and no scientific endpoint was computed.

The immutable attempt receipt, partial panel lease, logs, device telemetry,
and Slurm accounting record are archived under
`campaigns/phase1_ordering_20260803/ordering_confirmation_v1_attempt1_failed/`.

## Root cause

The frozen generator constructs each covariance entry as

`Sigma[i,j] = sd[i] * corr[i,j] * sd[j]`.

For mirrored entries, the same floating-point factors are multiplied in a
different operand order. The analytical covariance is symmetric, but IEEE-754
roundoff can make the two stored float64 values differ by a few units in the
last place. Attempt 1 incorrectly required bitwise equality between every
mirrored pair.

The cheapest discriminating replay regenerated all six registered streams,
1,067 rows per stream. Across those streams, the maximum absolute asymmetry
was `4.440892098500626e-16` and the maximum relative asymmetry was
`3.250572244316946e-16`. These values are construction-level float64
roundoff. They are not material covariance asymmetry.

## Repair

The generator and every generated array remain unchanged. The validation
guard now accepts a mirrored pair only when

`abs(Sigma[i,j] - Sigma[j,i]) <= 4 * eps64 * max(abs(Sigma[i,j]), abs(Sigma[j,i]))`.

Here `4 * eps64 = 8.881784197001252e-16`, with zero absolute tolerance.
Positive-definiteness and the frozen fleet validity-region checks remain
mandatory. Each stream record now saves the maximum absolute and relative
asymmetry, plus the registered relative tolerance. A regression test accepts a
one-ULP construction difference without mutating the array, and a second test
rejects a materially asymmetric covariance.

This is a harness repair only. It changes no scientific seed, context,
generator formula, checkpoint, oracle, estimand, threshold, numerical
clearance, or decision rule.

## Retry identity

The repaired attempt must use annotated tag
`phase1-ordering-confirmation-v1-fix1` and a new empty attempt root. The failed
attempt directory cannot be resumed or reused. This repair document takes
precedence over the earlier confirmation amendment only for the covariance
symmetry validation guard and retry identity.
