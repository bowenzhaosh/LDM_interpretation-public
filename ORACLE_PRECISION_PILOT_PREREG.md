# Oracle precision pilot — preregistration (v1)

Status: **PROSPECTIVE.** Frozen before any 400-row-panel estimate is generated.
Do not edit after the first estimate.

## Purpose

Determine whether a posterior-targeted continuous-prior oracle can meet the
frozen ±0.0005-nat scientific gates on the archived 400-row nested panel. This
is instrument development; it cannot retroactively change the FIX2 verdict.

## Input panel (frozen, outcome-blind)

- `campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix2_run1/panel/inputs/C_d0_b*` + `N_d0_b*` (contexts, queries) and the corresponding `labels/*` (outcome bins only).
- The 400 rows are exactly those with `draw_index == 0` and
  `stream_index < 200`, per prior (C=200, N=200).
- The pilot never reads PFN checkpoints, PFN predictions, `nested_half_raw.npz`,
  `confirmatory_raw.npz`, any `oracle_raw.npz`, or any file matching the
  forbidden fragments in `pilot_shared.FORBIDDEN_PATH_FRAGMENTS`, and never
  computes deficit, gap, or Delta. This is enforced in code (path guard) and
  by the software guard against forbidden imports.

## Target

For each context D, prior P in {C,N}, and each of the 24 orderings o:
`pi_o(theta) = p(theta) p(D | theta, o)`, with theta the exact latent object
(sd, rho_mag, sign) of the frozen prior (uniform on the box in
(log_sd, rho, sign), intersected with PD and validity). Estimands:

- `Z_o(D)` (order marginal likelihood),
- `p(y | x_q, D, o)` (order-conditioned 100-bin predictive),
- `p_full`, `p_ablated` (full and uniformly-ablated mixtures),
- the order posterior `p(o|D) propto Z_o(D)`.

The ablated mixture is `(1/24) sum_o p(y|x_q,D,o)` with each order-conditioned
predictive normalized independently; retained-mass weighting across orderings
is prohibited.

## Estimators (two independent)

1. **Primary: annealed SMC** (`pilot_smc.run_smc_posterior`). Adaptive
   conditional-ESS temperature schedule, systematic resampling at ESS <
   `resample_frac * N`, adaptive-covariance MH + sign-flip rejuvenation,
   incremental-normalizer evidence, 100-bin predictive via the frozen
   production quadrature.
2. **Independent: MCMC + thermodynamic integration** (`pilot_mcmc`). Adaptive
   Metropolis-Hastings (multiple parallel chains, no resampling, no importance
   weights), beta=1 posterior predictive, evidence via the beta-ladder
   trapezoid (fine near beta=0), beta=0 chain initialized from the exact prior.

The defensive adaptive importance sampler (`pilot_ais`) is retained as a
diagnostic only; its documented heavy-tail importance-weight obstruction makes
it non-qualifying.

## Frozen configuration

`config/oracle_precision_pilot_v1.json`:

- SMC: `n_particles=16384`, `mh_steps=20`, `cess_target=0.5`,
  `resample_frac=0.5`, `max_temps=80`.
- MCMC: `n_chains=200`, `n_iter=3000`, `n_iter_per_beta=1500`,
  `betas=[0, .002, .005, .01, .02, .05, .1, .2, .35, .55, .8, 1]`.
- Seed root `886900000` (pilot namespace, disjoint from all frozen production
  seeds and from the development namespace `886000000`).
- Production quadrature 32/128.

## Ladder (convergence axis)

The fine/coarse convergence axis is the SMC particle-count ladder:
`{4096, 16384}` particles for the SMC (the coarse rung is the 4096 setting;
the frozen production rung is 16384). The MCMC ladder is the beta-ladder (the
production ladder is frozen; a coarser 6-beta ladder is used only to bound the
discretization error in development).

## Numerical gates (all must pass for QUALIFIED_ORACLE)

1. **SMC replica agreement:** 3 independent SMC runs (disjoint seeds) on a
   preregistered 40-row subset agree on the per-row full/ablated held-out NLL
   to within `1e-3` nat.
2. **SMC ladder convergence:** on all 400 rows, the mean over rows of the
   full and ablated held-out NLL changes between the 4096 and 16384 rungs have
   bootstrap CIs inside `[-0.0005, +0.0005]` nat (C full, C ablated, N full,
   N ablated, and the control-subtracted ablated change).
3. **MCMC replica agreement:** 3 independent MCMC runs on the 40-row subset
   agree within `1e-3` nat per row.
4. **SMC-vs-MCMC agreement:** on the 400 rows, per-row held-out NLL
   differences (full and ablated) have median absolute value below `0.002`
   nat and max below `0.02` nat.
5. **Order-posterior agreement:** the per-row order-posterior JS divergence
   between SMC and MCMC has median below `1e-4` and p95 below `1e-3`.
6. **No row-level catastrophe:** no single row has a full or ablated held-out
   NLL difference between SMC and MCMC above `0.5` nat.
7. Every 100-bin predictive is finite, nonnegative, and normalized to `1e-8`.

ESS and acceptance are reported as diagnostics; they do not substitute for
the prediction-level and endpoint-level gates.

## Difficult subset (frozen before new estimates)

The 40 rows with the lowest old FIX2 full-oracle ESS under C (rows selected
from the archived `confirmatory_raw.npz` `ess_full_atoms`, membership fixed in
this preregistration). Replica and ladder gates are also reported separately
for this subset.

## Terminal taxonomy

- `QUALIFIED_ORACLE`: all gates pass.
- `FAILED_ORACLE_PRECISION`: the ladder gates fail (the oracle cannot reach the
  frozen numerical standard).
- `FAILED_ORACLE_METHOD_AGREEMENT`: the two estimators disagree (order
  posterior or predictives).
- `INCONCLUSIVE_IMPLEMENTATION`: an estimator or verifier defect is found after
  freezing; the attempt is not a scientific verdict.
- `INTERRUPTED_ATTEMPT`: a resource/access failure prevents completion.

## Outputs (per attempt)

`summary.json`, raw numeric arrays (per-row full/ablated NLLs, order
posteriors, logZ, diagnostics for both estimators), `AUDIT_REPORT.md`,
`CLAIM_LEDGER.md`, a self-contained HTML report, resource accounting, and
replay instructions. The pilot's verdict is sealed under an annotated result
tag and cannot be revised by later code changes.

## Decision routing

- If `QUALIFIED_ORACLE` -> conditional branch A (development reanalysis of the
  archived FIX2 models, prospective power analysis, fresh-seed confirmatory
  campaign).
- Otherwise -> conditional branch B (exact finite-prior causal wind tunnel).
