# PREREG-EXT3 — binary-duel: replication, d=3 readout port, Phase B

Extension of `PREREG-BINARY-DUEL.md` (lock `306c5db4…`, amendments A/B/C). Locked BEFORE any
result number for these three items exists. Fresh eval-seed families per item (never reuse a
seed family that already produced a number). Same no-hallucination rules: scripts emit numbers +
gate booleans only; every result JSON carries provenance; gates default to reject.

## 1. Seeds × scales replication (in-scope: original §5 "seed survival" + "scale check")

Re-run the two behavioral claim-granting estimands — **E1a** (evidence tracking `logit w(D) ~ ℓ(D)`)
and **E1b** (matched-pair cross-scoring on fresh outcomes) — with the E1c controls, and **E2a/E2b**,
across the full seed × scale grid on the d=2 `AL40` fleet at `dose12000`:

- base: seeds 0–3 already run (archive); **extend to 4–15** (12 new seed cells).
- mid, large, xl: seeds **0–11** (11 cells each; never run for binary-duel).

Eval-seed family: `R_REPL = {501, 502, …}` (fresh, per seed cell). Success = the primary effect
(E1a `β_w` negative with CI excluding 0; E1b `gain_spec > 0` significant with control-6) survives
with the same sign in ≥ 2/3 of the new seed cells AND on every scale (mid/large/xl). No new
estimand; failure mode = seed-limited or scale-limited effect (report which).

## 2. d=3 port of the evidence readout (NEW estimand scope)

Port the binary-duel E1a/E1b readout to the **d=3 substrate** (3-variable complete-DAG SEM,
AL skew r, `permuted-LDL` construction, 6 orderings; `d4plus_oracle.py` machinery used VERBATIM).

- **Duel pair.** `π_A = (0,1,2)` (x0 root, x1 middle, **x2 = sink**) vs `π_B = (0,2,1)` (x0 root,
  **x2 = middle**, x1 = sink). This is a single adjacent transposition (x1 ↔ x2) holding x0 as the
  shared root: it asks whether `x2` is downstream of `x1` (π_A) or upstream of `x1` (π_B). This is
  the d=3 analog of the d=2 G1↔G2 swap. (The orders `(0,1,2)` vs `(1,0,2)` are NOT used: both put
  x2 last, so their `p(x2|x0,x1)` conditionals coincide — no duel.)
- **Evidence.** `ℓ(D) = log P(π_A|D) − log P(π_B|D)` from the full 6-order MC posterior
  (`Oracle.posterior`), shared fixed atom set (M = 20000, seed 7005, matching `d4plus_oracle.py`
  `7002+D_DIM`), convergence-checked (two disjoint M/2 halves).
- **Order-conditioned predictives (analytic, closed-form).** Let `Lunit, U, b = params_for(S, π)`
  (`U = inv(Lunit)` unit-lower; `b[m]` = residual scale of the node at position m).
  - `p_πA(x2|x0,x1)` (x2 is last in the order → single AL):
    `loc = −U[2,0]·x0 − U[2,1]·x1`, scale `b[2]`.
  - `p_πB(x2|x0,x1)` (x2 is the middle → product of two ALs, the d=3 analog of d2's p_G2):
    `e1 = x2 − Lunit[1,0]·x0`, `e2 = x1 − Lunit[2,0]·x0 − Lunit[2,1]·e1`;
    density `∝ AL(e1; b[1]) · AL(e2; b[2])`.
  Both are 100-bin log-normalized distributions over `x2` at each query `(x0*,x1*)`. The two
  conditionals differ (single-AL vs product-AL) even at query `(0,0)`.
- **Estimands (identical form to d=2 E1a/E1b):**
  - `s(D) = mean_q Σ_b Q(b)[log p_{π_A}(b) − log p_{π_B}(b)]`; `logit w(D)` = global KL-mixture
    weight toward π_B over the query grid.
  - E1a: regress `logit w(D) ~ ℓ(D)` (Theil-Sen + OLS + Spearman + monotone-fraction + bootstrap CI).
    **Ideal Bayes `β_w = −1`** (logit w* = −ℓ). Direction: `β_w < 0`, `β_s > 0`.
  - E1b: matched pair — same Sigma, same AL residual draws `(e0,e1,e2)`, `D_true` under true order,
    `D_mirror` under the swapped order; fresh outcomes from the true process; `gain = NLL(mirror) − NLL(true)`.
- **Priors.** A = AL(r=2), C = AL(r=4) (both order-evidence families; A weaker, C stronger — the
  d=3 analog of d=2's AL20/AL40). N = Gaussian (order evidence identically zero; anchors coincide).
- **Models.** `nets3/`: base `M3_{A,C,N}_s{0..3}_st20000` at doses {1000, 3000, 6000, 12000, 20000}
  (dose ladder = emergence). Primary: `M3_C` (stronger evidence), dose 20000.
- **Controls.** (i) positive control: synthetic perfect-Bayes `Q = σ(ℓ)p_{π_A} + σ(−ℓ)p_{π_B}` must
  give `β_w = −1` exactly; (ii) Gaussian-null bank under N ⇒ `β_w ≈ 0`, no cross-scoring gain;
  (iii) dose ladder monotone emergence.
- **Gate.** E1a monotonic tracking AND E1b cross-scoring pass, with controls consistent.
- Eval-seed family: `R_D3 = {601, 602, …}`.

## 3. Patching Phase B — DAS-lite (in-scope: original §4)

Optimize a rank-r subspace `P = UUᵀ` (r ∈ {1,2,4,8,16}, U ∈ R^{d_model×r}) minimizing, over training
quadruples, the forward-KL from the patched predictive to `p_desired`:
`patch = h_base + P(h_donor − h_base)` at the intervention site, differentiable through the model.

- **Sites (corrected, pre-data).** Both resid-stream, both live in the post-LN model (whose final
  layer output reads only the query position):
  - `resid0` = residual stream at layer-0 output (post-norm2), ALL positions (NULL marker + context +
    query). The distributed representation that layer 1 reads via attention.
  - `query1` = residual stream at layer-1 output (post-norm2), query position only. The readout
    representation that `out_head` reads directly.
  The first-draft PRIMARY ("residual stream at layer L1, context + query") is DEAD (see erratum §4):
  patching layer-1 context positions has no effect because the post-LN output reads only the query
  position, so that site collapses to `query1`. The original §4 "at the best Phase-A head site" is
  DROPPED (see deviation note §4): Phase A's head sites are weak and seed-inconsistent, so the
  resid-stream DAS-lite is the canonical, more informative test.
- **Orthogonality.** U parametrized d_model×r; soft orthogonality penalty `‖UᵀU − I‖` added to the
  KL objective (weight 0.1), so P ≈ projection.
- **Train/test split.** Training quadruples: 400 (fresh Sigma/context/residual draws, both
  directions A→B and B→A). Held-out: 200 unseen quadruples, unseen query values (Q1), both
  directions, separate model seeds s0–s3.
- **Metrics (held-out).** `transfer = (KL(Q_base‖p_des) − KL(Q_patch‖p_des)) / KL(Q_base‖p_des)`;
  anti-donor `KL(Q_patch‖p_des) < KL(Q_patch‖p_don)`; report per rank r, per site.
- **Gates.** Some rank r shows held-out `transfer > 0` (signed-rank p<0.05) AND anti-donor rate > 0.5,
  and held-out transfer exceeds the no-patch / random-subspace placebo. INCONCLUSIVE if no r/site
  transfers (harness positive control: self-identity patch must reproduce the donor).
- **Eval-seed family:** `R_PB = {701, 702, …}`.
- **Note (documented risk).** Phase A transfer is weak and seed-inconsistent; Phase B may be a
  clean null (no low-rank order direction generalizes). That outcome is reportable, not papered over.

## 4. sha256 lock

Locked 2026-08-20, before any result number. Amendment B of the original prereg (sign fix) carries
forward. This amendment corrects the §3 intervention SITE (a pre-data factual correction + a
documented deviation); it changes no gate, metric, rank ladder, sample size, or eval-seed family.

Erratum (pre-data, §3 sites): the first-draft PRIMARY site ("residual stream at layer L1, context +
query positions") is DEAD — the post-LN model's final layer output is read only at the query
position, so patching layer-1 context positions cannot affect the predictive; that site collapses to
`query1`. Corrected to the two live resid-stream sites (`resid0` = layer-0 all-positions,
`query1` = layer-1 query position) before any number was produced.

Deviation note (pre-data, §3 sites): the original §4 "at the best Phase-A head site" is dropped.
Phase A's per-head-output screen was weak (transfer ≈ +0.02…+0.07) and seed-inconsistent (best site
varies L0.head2 / L1.head1 / L1.head3 / L1.head0), so "the best site" is ill-posed. The resid-stream
DAS-lite at `resid0` + `query1` is the canonical, more informative test of "is there a low-rank
subspace carrying the order direction". This is the only estimand-level change; it is recorded here
before any result number.

Erratum (pre-data, §2): the §2 duel pair was originally written as `(0,1,2)` vs `(1,0,2)` (both
x2-as-sink), which gives identical order-conditioned predictives and no duel. Corrected to `(0,1,2)`
vs `(0,2,1)` (x2 sink vs middle) before any number was produced. No result number exists yet;
nothing to retract.

sha256 of this document (lock): `f42a79c609ea6c2b1570f56373194d9d5923327d39e08fb2dbe624706ed013ad`

---

## Amendment D (pre-data): Phase B confirmation re-run

Locked 2026-08-21, after the §3 fleet ran but BEFORE any confirmation number exists. The §3
fleet (exp3b) returned held-out transfer at `resid0` (mean +0.18…+0.39) that survives the
random-subspace placebo, but a post-hoc integrity audit flagged two defects that make the
§3 claim over-reach: (i) the learned U's orthogonality was never measured, so "orthogonal
projection" is asserted but not established (a soft penalty weight 0.1 does not enforce
UᵀU=I, and U is returned un-re-orthonormalized, so the placebo — exactly orthonormal —
conflates direction with magnitude); (ii) the §3 eval query grid Q1=linspace(-4,4,9) is a
SUPERSET of the train grid Q0=linspace(-3,3,7), so "unseen query values" is false (only ±4
are new); generalization is established on the context/order axis but not the query axis.

This amendment re-runs exp3b with exactly two changes, in a SEPARATE result dir, to close
those two defects:

- **Eval query grid disjoint from train.** `qx_eval = linspace(-2.5, 3.5, 7)` (all 7 points
  half-integer, none in Q0's {-3..3}). Train grid unchanged (Q0). This tests genuine
  query-axis generalization: if the learned direction is query-agnostic, held-out transfer
  on disjoint queries stays > placebo; if it was query-position-specialized, it collapses.
- **Orthogonality logged.** Per rank r, save `u_orth_err = ‖UᵀU − I_r‖_F` and
  `u_colnorm = max_j ‖U[:,j]‖₂`. `u_orth_err ≈ 0` with `u_colnorm ≈ 1` ⇒ U is an actual
  orthogonal projection (the §3 "orthogonal subspace" reading is CONFIRMED). `u_colnorm ≫ 1`
  ⇒ scale confound (P=UUᵀ is a scaled projection; part of the transfer is magnitude-driven).
- **Everything else identical**: model `base`, prior AL40, seeds 0–3, sites {resid0, query1},
  dose 12000, ranks {1,2,4,8,16}, n_train=400, n_eval=200, n_steps=2000, lr=1e-2, ORTH_W=0.1,
  gates/metrics unchanged. Script `exp3b_phaseB_conf.py`, result dir `results/exp3b_conf/`.

- **Eval-seed family**: `R_PB2 = {801, 802, …}` (fresh, disjoint from R_PB={701…}).

**Decision rule (pre-registered).** The §3 resid0 direction-specific claim is CONFIRMED as an
*orthogonal* low-rank subspace only if (a) held-out transfer_mean on the DISJOINT eval grid
exceeds its placebo (signed-rank p<0.05) in the same rank/seed pattern as §3, AND (b)
`u_colnorm` is ≈1 (within ~1.5×) at the transferring ranks. If (a) holds but (b) fails, the
finding downgrades to "direction-specific low-rank (non-orthogonal) transfer". If (a) fails,
the §3 transfer was query-position-specialized (report that, do not paper over).

sha256 of THIS document including Amendment D (lock): `cfaac68b217724ad5a57f0d7b83554a3995f06edbb3789fd9934919c5ff14992`
