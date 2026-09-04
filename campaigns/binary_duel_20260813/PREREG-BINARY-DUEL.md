# PREREG-BINARY-DUEL — Does the PFN exploit causal-order evidence in the context?

Locked BEFORE any result number exists. New experiment line on the d=2 evidence-integration
substrate (`dose_nets` AL40/AL20/N fleets). Extends the errata discipline: sha256-lock this
doc, draw a fresh eval-seed family, carry controls forward.

## 0. The question

A cross-entropy-trained PFN (predict `y | x` from a 30-row panel `D={(x_i,y_i)}`) is trained
under a prior over (Sigma, causal order). The exact order evidence `ℓ(D) = logit P(G1|D)` is
computable, and the two order-conditioned predictive distributions `p_G1`, `p_G2` are
computable with the (known) latent parameters. The experiment asks, in **prediction space**:

> Does the PFN's predictive distribution move monotonically between the two order-conditioned
> predictives as the context's order evidence moves, in a way that improves held-out prediction,
> and can a manipulated internal activation transfer the order without transferring the mechanism?

This is deliberately a *behavioral* protocol (Exps 1–2) before a *mechanistic* one (Exp 3).
The behavioral statement under test, to be granted only if the gates pass:

> **The PFN uses order-diagnostic information in the context to improve prediction.**

That statement is weaker than "the PFN holds an explicit internal order belief"; Exp 3 is the
only experiment that can approach the latter.

## 1. Substrate (fixed, shared by all experiments)

- **Task.** 2-var linear SEM, `N_CONTEXT=30`, `N_QUERY=7`, `N_BINS=100`, `BIN_EDGES=linspace(-8,8,101)`.
  Query = scalar `x*`; the model outputs `P(y* | x*, D)` over the 100 bins.
  Family `AL40` = asymmetric Laplace, skew r=4.0 (primary); `AL20` (r=2.0, weaker evidence) and
  `N` (Gaussian, order evidence identically zero) are the residual-family controls.
- **Generative process (identical to e18b/evidence_core).** Draw valid `Sigma` from the Fix-B
  prior (`σ1,σ2∈[0.6,1.5]`, `|ρ|∈[0.4,0.8]`, acceptance via `params_in_range`); draw latent
  order `cls∈{1,2}` (1 = G1: x→y; 2 = G2: y→x); params `(a,b1,b2)=sigma_to_params_G(cls)(Sigma)`.
  - G1: `x=e1`, `y=a·x+e2`, `e1~AL(0,b1)`, `e2~AL(0,b2)` (Var = 2b²).
  - G2: `y=e1`, `x=a·y+e2`, same parametrization under `sigma_to_params_G2`.
- **Evidence.** `ℓ(D) = logit P(G1|D) = EC.compute_ell(D, prior)` (exact 3097-point grid oracle).
  Agreement evidence `a(D) = ℓ(D)·(2·[cls==1]−1)`; `a>0` means the evidence agrees with the
  context's true order.
- **Order-conditioned predictives (known latent parameters, no continuous-prior integration).**
  For a context with known `(Sigma, cls)`:
  - `p_G1(D)`: predictive of `y|x*` under G1 with `(a,b1,b2)=sigma_to_params_G1(Sigma)`:
    `p(y*) ∝ f_AL(y* − a·x*; b2)` over bin centers, log-normalized.
  - `p_G2(D)`: predictive under G2 with `(a,b1,b2)=sigma_to_params_G2(Sigma)`:
    `p(y*) ∝ f_AL(x* − a·y*; b2)·f_AL(y*; b1)` over bin centers, log-normalized (1-D quadrature object).
  - `p_true(D)` = the one matching `cls`, `p_other(D)` = the other.
  - For N-family contexts both anchors collapse to the same Gaussian conditional (order-symmetric).
- **Model.** `EF.PFNModel` (`d_model=256,d_ff=512,n_heads=4,n_layers=2` base, ~1.1M params).
  Primary checkpoint: `M_base_AL40_s{seed}_dose12000.pt`. `dose` = training step.
- **Query bank.** Q0 = `linspace(-3,3,7)` (the training grid); robustness Q1 = `linspace(-4,4,9)`.
- **Model prediction-space order evidence.** For predictive `Q` (a 100-bin distribution at each query):
  - `s(Q) = Σ_b Q(b)[log p_G1(b) − log p_G2(b)]`  (the project's W3 score, single-Sigma analytic anchors).
  - `w(Q) = argmax_w Σ_b Q(b) log((1−w)·p_G1(b) + w·p_G2(b))`  (KL-mixture weight toward G2, 1-D concave fit).
  - **Per-context scalar (pinned):** for a context with the full `(n_q,100)` predictive, `s(D)` = mean over
    queries of `s(Q_q)`, and `w(D)` = argmax over the SUM `Σ_{q,b} Q_q(b) log((1−w)p_G1,q(b) + w p_G2,q(b))`
    (one global mixture weight over the query grid). Both are reported; `logit w(D)` is the E1a regressand.
  - Ideal Bayes (known params, exact order posterior): `s → ±KL` and `logit w → −ℓ(D)` as `ℓ → ±∞`; `logit w* = −ℓ`.

## 2. Experiment 1 — behavioral use of order evidence

Context banks (AL40 unless stated; fresh eval-seed family `R_EXP1 ∈ {101,102,103,104}`):
- Natural: N=1500 iid (Sigma, cls 50/50).
- Balanced: N=1500 stratified to cover `a(D)∈[−8,8]` densely (reuse `generate_fixed_panel` balanced machinery).
- Gaussian-null: N=1000 drawn under the N family (cls 50/50). ℓ(D)=0 identically.

Estimands (primary model: dose12000, 4 seeds s0–s3; each seed gets its own drawn bank):

- **E1a — Evidence tracking (monotonic movement).** Fit `logit w(Q(D)) ~ ℓ(D)` (Theil-Sen; and OLS).
  Report slope `β_w`, Spearman `ρ_w`, monotone-fraction (concordant pairs among ℓ-sorted contexts).
  GATE: bootstrap 95% CI of `β_w` excludes 0 AND `sign(ρ_w) = sign(β_w)` (tracking in the expected
  direction). Secondary: `s(Q)` slope `β_s` (GATE: CI excludes 0 AND `ρ_s > 0`).
  Direction: `β_w < 0` expected (`logit w* = −ℓ`); `β_s > 0` expected.
- **E1b — Cross-scoring on fresh outcomes (matched pair).** K=800 matched pairs: draw `(Sigma, cls)`;
  draw standardized residuals `z1,z2 ~ AL(0,1)`; `D_true` under true order (scaled by its b's);
  mirror `D_mirror` under the other order, same `z1,z2`, scaled by the other order's b's (same Sigma ⇒
  same covariance, matched residual draws). For each pair draw M=20 fresh outcomes from the true
  process (`x* = b1·z1*`, `y* = a·x* + b2·z2*` under G1). Estimand (true order A):
  `gain = mean_NLL(Q(D_mirror)) − mean_NLL(Q(D_true))` on A-outcomes (NLL = −log Q(y*|x*,D) at bin(y*)).
  GATE: paired `gain > 0` with Wilcoxon p<0.01, and the mirror (B-outcomes on the B-mirror pairs)
  gain also > 0. Expected sign: the context whose evidence matches the outcome process predicts better.
- **E1c — Selective controls** (each with the same estimands as E1a/E1b):
  1. Gaussian residuals: AL40-trained model on N-bank ⇒ ℓ≡0, anchors coincide ⇒ expect `β_w≈0`, no
     cross-scoring gain. Native control: `M_base_N` on N-bank (trained with no order signal).
  2. Initialization: `dose0` ⇒ expect no tracking, no cross-scoring gain.
  3. Emergence: trace `β_w` and cross-scoring `gain` along `dose ∈ {100,300,1000,3000,6000,12000}`
     (2 seeds). Expect monotone (non-decreasing) emergence; report trajectory.
  4. Seed survival: primary effect holds with 95% CI over the 4 seeds (jackknife over seeds).
  5. Mechanism survival: report E1a split by coefficient magnitude `|a|` bands and `|ρ|` bands
     (effect must not live only in a corner of the Sigma prior).
  6. Independent-context control: with `D'_same` an independent SAME-order context (same Sigma,
     fresh residual draws), the specific-wrong-order gain `g_spec = meanNLL(Q(D_mirror)) −
     meanNLL(Q(D_true))` must EXCEED the neutral gain `g_neut = meanNLL(Q(D'_same)) −
     meanNLL(Q(D_true))` (paired, same outcome draws). By exchangeability `g_neut ≈ 0`;
     the gate is `g_spec > g_neut` significantly. This separates "the specific wrong-order
     evidence hurts prediction" from "any context from the wrong-order family hurts equally."

**Failure rule (user-specified).** If E1a monotonicity FAILS AND E1b cross-scoring FAILS (across
seeds, after the controls are shown not to explain the pattern), the statement "the PFN exploits
causal order in prediction" is abandoned; any residual signal is attributed to sensitivity to other
non-Gaussian statistics.

## 3. Experiment 2 — deliberately mislead the PFN

- **E2a — Natural misleading contexts.** Draw N=30000 true-A (G1) contexts (valid draws from the A
  process). `ℓ(D)` computed for each. Misleading set = the K=300 with most negative `ℓ` (strongest
  B evidence despite true A); matched controls = K=300 with `ℓ` most positive, matched on `|ℓ|`.
  (N was raised from 8000 to 30000 in Amendment C so the K=300 tail is genuinely negative-ℓ:
  a calibration draw of 8000 showed the 300-most-negative tail has median ℓ ≈ +0.2 with only 45%
  negative, which would fail the w>0.5 gate by substrate weakness rather than by model behavior.)
  For each: `logit w(Q)` (expect > 0 = weight toward the wrong order G2), held-out NLL on fresh
  A-outcomes (expect DETERIORATE vs control), NLL on fresh B-outcomes (expect IMPROVE vs control),
  and scaling: `logit w` vs `ℓ` across the misleading tail (slope consistent with E1a).
  GATE: misleading `w`-weight significantly > 0.5, NLL-harm on A-outcomes significant vs control,
  and the ordering scales with `|ℓ|`.
- **E2b — Wrong-order evidence injection.** K=400 base A-contexts (G1, fixed Sigma). Mirror B-rows
  from G2(S) (matched standardized residuals). Dose `δ∈{0,0.10,0.25,0.50}`: replace a δ-fraction
  of rows with matched B rows. Row matching: nearest-neighbor on standardized (‖row‖, x·y,
  |x|, |y|). Same-order control: replace δ-fraction with matched fresh A rows (equal perturbation
  magnitude). Estimands: `logit w(Q(D_δ))` vs δ (expect monotone increase toward the clean-B
  predictive), NLL under the true A process vs δ (expect monotone harm), and the same-order control
  (expect flat). GATE: monotone wrong-order weight + monotone predictive harm, control flat.

## 4. Experiment 3 — internal causal-order interchange

Four-context grid `{mech1, mech2} × {orderA, orderB}` (Sigma1 ≠ Sigma2; order ∈ {G1,G2}):
Donor=(S1,B), Base=(S2,A), Desired=(S2,B), Control=(S1,A). Claim: an order-specific internal
state transfers order while preserving mechanism. A patched base should behave like **Desired
(mech2,B)**, NOT like Donor (mech1,B). The "not-donor" contrast is the critical control.

- **Phase A — component screen.** At each layer `ℓ∈{1,2}` and component `c ∈ {ctx-residual-stream,
  attn-output, mlp-output, query-residual-stream, per-head-output (×4)}`: run donor forward and
  capture; run base forward patched with donor activation on that component. Metric per site:
  `transfer = (KL(Q_base, p_Desired) − KL(Q_patch, p_Desired)) / KL(Q_base, p_Desired)` (normalized
  movement toward Desired), and the anti-donor check `KL(Q_patch, p_Desired) < KL(Q_patch, p_Donor)`.
  Averaged over ~200 fresh (donor,base) pairs drawn from held-out Sigma/context/residual draws and
  both directions (A→B, B→A). GATE for proceeding to Phase B: some site must show `transfer > 0`
  AND the anti-donor check (a positive-control test of the patching harness).
  Placebos: random-subspace patch and no-patch baseline (both must NOT transfer).
- **Phase B — distributed intervention subspace (DAS-lite).** At the best site, optimize a rank-r
  subspace `P=UU^T` (r∈{1,2,4,8,16}, orthogonal U) minimizing, over training triples, the KL from
  the patched predictive to `p_Desired`: patch = `z_base' = z_base + P(z_donor − z_base)`.
  Differentiable through the model. Test on held-out: unseen Sigma mechanisms, unseen contexts and
  residual draws, held-out query values (Q1), both intervention directions, separate model seeds
  (s0–s3). Report transfer on held-out pairs and the anti-donor check. Note: d=2 admits a single
  adjacent swap (G1↔G2), so "additional adjacent swaps" is vacuous here; both directions cover the
  swap space.

## 5. Model / checkpoint selection

- Primary: `M_base_AL40` at `dose12000`, seeds {0,1,2,3}.
- Emergence: dose ladder on seeds {0,1}.
- Scale check: `M_xl_AL40` `dose12000` (seeds available), same primary estimands.
- Gaussian null: `M_base_AL40` at `dose12000` on N-bank, plus `M_base_N` at `dose12000` (native).
- Optional cross-family: `M_base_AL40` on AL20-contexts (weaker evidence, effect should shrink).

## 6. Computational integrity (no-hallucination, pre-registered)

- Every number in the report is re-derived from result JSONs on disk; no prose/verdict strings are
  emitted by scripts (scripts emit numbers + gate booleans only).
- Fresh eval-seed families per run; results carry model/prior/seed/dose provenance.
- Per-context checkpointing with resume (no single end-of-run json.dump).
- The known silent-wrong-number bugs from the fact sheet are checked: `D_NETPFX`/`GAPV_STEPS` are
  N/A here (direct model loads); oracle-path optimizations are verified bit-identical to the stock
  `EC.compute_ell` path before use; query-subset oracle bias avoided (anchors are exact closed-form).
- Positive controls: (i) with known params, a synthetic "perfect Bayes" Q = σ(ℓ)·p_G1 + σ(−ℓ)·p_G2
  must achieve β_w = −1 exactly on E1a (harness check); (ii) the patching Phase-A harness must
  reproduce the donor when donor is patched into a base that equals the donor's own mechanism
  (self-identity), else the harness is broken.

## 7. Decision gates (summary)

- E1: PASS iff E1a monotonicity AND E1b cross-scoring PASS, with E1c controls consistent.
- E2: PASS iff E2a misleading weight/harm AND E2b injection monotonicity PASS with controls flat.
- E3: PASS iff Phase-A screen shows transfer at some site with anti-donor + placebo controls, and
  Phase-B held-out transfer holds. INCONCLUSIVE if Phase-A shows no site (harness positive controls
  must pass first).
- Claim granted only if E1 AND E2 pass: "the PFN uses order-diagnostic information in the context
  to improve prediction." Exp 3, if it passes, is required for any stronger internal-state claim.

sha256 of this document (lock): `306c5db453e79ee242166b90d8dbb1a57c66d9e0b25c94d03a19a083dbea22de`
Locked 2026-08-13, before any result number was produced. Checkpoint inventory at lock: 112 AL40-base
checkpoints (seeds 0-15 × doses {0,100,300,1000,3000,6000,12000}) present; evidence_core 3097-grid
faithfulness=True.
Amendment A (2026-08-13, pre-data): per-context scalar aggregation pinned (mean-of-s per query for s(D);
single global KL-mixture w over the query grid for w(D)); control 6 sharpened to g_spec > g_neut with a
matched-noise same-order context (exchangeability baseline). Re-hash: `9cbfd3db5c621ef0e8e0748e86656c05e287b48f4759d146b2d03cbacefa4f5e`.
Amendment B (2026-08-13, pre-data, sign fix only): the E1a GATE as written (`ρ_w > 0`) contradicts the
stated expected direction (`β_w < 0` via `logit w* = −ℓ`), so a correctly-tracking model could never pass.
Amended to `sign(ρ_w) = sign(β_w)` (tracking in the expected direction); the s-GATE is `ρ_s > 0` (unchanged).
No claim, sample size, or estimand changed. Re-hash: `8a133eb73b823107b0a84c819290c1d248aa23df716cf0ad6ddb3f13bf389238`.
Amendment C (2026-08-13, pre-data): E2a pool N raised 8000 → 30000 (see §3) so the K=300 most-negative
tail is genuinely misleading (median ℓ < 0). All gates and estimands unchanged. Re-hash: `acef625f2e84df475b99f4faa637ae6833e14d3017914fc17c0a1095979c29a0`.
