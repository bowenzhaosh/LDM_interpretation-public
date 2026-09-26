# PREREG AMENDMENT E — mixture convention, exact-expectation estimands, verdict rebuild

**Status:** LOCKED. Signed Bo, 2026-08-21. Supersedes the Aug-21 calendar where it conflicts.
**Scope:** the corrected campaign `campaigns/corrected_20260812/` (d=3, K=8, ε-dial) and the retrofits it
mandates onto the d=2 binary-duel line `campaigns/binary_duel_20260813/`.
**Chain:** extends `PREREG.md`, `MAPPING_QUALIFICATION_PREREG.md`, and the Branch-B supersession note
(`campaigns/branch_b_20260808/SUPERSESSION.md`).

This document is locked BEFORE any affected number exists. Every quantity it declares provisional is to be
regenerated after the code changes it mandates, and no pre-fix value of those quantities may be cited.

## E.0 Pre-fix code state being amended

Locked at repo HEAD `9eca9ac94d01b925b852c10e3f9f0ae599a784f1`, with these file digests (sha256):

| file | sha256 |
|---|---|
| `src/pfn_dag_verify/corrected_sem.py`     | `7277d76bc201801ad71d2042ae6dda7477996cbd8c2aa9b1e527dea6cff50b95` |
| `src/pfn_dag_verify/corrected_models.py`  | `521509caaef47d56b064e442d284ded79926da910114aaf80c9a0e093370977a` |
| `src/pfn_dag_verify/corrected_verdict.py` | `806500bcd5648f9f49117261dcfbce87b332942af28518acccce3de726b7d4f1` |

The three defects this amendment resolves, at source:

1. `corrected_sem.residual_logpdf` (L167-183) mixes the Gaussian and AL components **elementwise on the
   `(n, d)` residual array**, and `residual_logdensity_row` then sums over the coordinate axis. The oracle law
   is therefore `p(e) = prod_j [(1-eps) N(e_j) + eps AL(e_j)]`: one independent component draw per coordinate.
2. `corrected_sem.sample_residuals` (L417) draws `mask = rng.random(tuple(shape) + (1,)) < eps`, broadcasting
   over the coordinate axis: one component draw **per row**.
3. `corrected_models._vectorized_observational` (L153) draws `mask = rng.random((B, 1, 1)) < spec.eps`:
   one component draw **per training example**, shared across all `n_ctx + n_query` rows and all `d`
   coordinates. This entered at commit `3360b99` ("Vectorize Track B batch generation ... 1000x data-gen
   speedup"), whose message does not mention the mask.

At `eps <= 0` all three paths return the Gaussian branch and at `eps >= 1` all three return the AL branch, so
the endpoints are unaffected. The defect is confined to `eps in {0.1, 0.25, 0.5}`.

## E.1 (D1) Mixture convention: ELEMENTWISE everywhere

**Declared convention.** The residual law is a fixed per-coordinate density

    p_eps(z) = (1 - eps) * N(z; 0, (b sqrt2)^2) + eps * AL(z; b, r)

applied independently to each coordinate of each row. `sample_residuals` and `_vectorized_observational` are
unified to this law. **`residual_logpdf` is NOT changed**: the oracle already implements the declared
convention, so the exact-Bayes machinery, the query operators, and the identified-set LP are untouched.

**Scientific rationale (the reason this is the convention and not merely the cheaper fix).** Under the
elementwise law each coordinate has one fixed non-Gaussian density whose shape `eps` continuously deforms, so
`eps` is a **noise-density deformation dial** and the identifiability of causal direction varies continuously
with it, which is the quantity the paper's dial claims to control. Under a per-row or per-dataset indicator
the coordinates of a row share a component, so `eps` becomes the **mixing proportion of a population of two
residual families** and the manipulation becomes family-gating. That is a different scientific object: it
would make the dial a statement about dataset composition rather than about how much non-Gaussianity a single
SEM's noise carries, and the LiNGAM-style identifiability argument the paper leans on is stated for the
per-coordinate density.

**Consequence for training.** The eps in {0.1, 0.25, 0.5} models were trained under the per-dataset law and
evaluated under the per-row law while scored against the per-coordinate oracle. Those weights are retired.

## E.2 (D2) Provisional quantities, to be regenerated

Provisional and **not citable** until regenerated post-unification:

- the identifiability ladder at `eps in {0.1, 0.25, 0.5}` (`raw/ident_eps{0p1,0p25,0p5}.json`), because the
  20 evaluation contexts per cell are drawn through `sample_residuals`;
- every Track-B and in-task quantity at those three cells (regret, learning curves, classifier fidelity);
- the `nrows10_ident_eps*` companion set at those cells.

**Standing:** the endpoints `eps = 0` (Q1 width 1.0000) and `eps = 1.0` (Q1 width 0.1255) are unaffected by
the mixture convention and stand as measured. They remain provisional only with respect to E.3's estimand
change where a regret or fidelity number is involved.

The pre-fix ladder value at each provisional cell is recorded here so the post-fix regeneration is a
comparison and not a replacement made silently: Q1 width 0.4510 / 0.3232 / 0.2073 at eps 0.1 / 0.25 / 0.5.

## E.3 (D3) All behavioral estimands become exact expectations over outcomes

Every behavioral estimand is redefined as an **exact expectation over the outcome distribution**, computed by
summing over the 100 native bins against the reference predictive, replacing the current single-sampled-outcome
estimator (`corrected_models.evaluate_pfn_checkpoint` L345-347 scores one drawn bin per query).

For a query with reference (exact Bayes) predictive `p*` and model predictive `q`, the expected regret is

    R = sum_b p*(b) [ -log q(b) + log p*(b) ]  =  KL( p* || q )

and analogously for the E1b cross-score and the tracking regressands, each as a bin-sum against the exact
outcome law rather than against one draw. This removes outcome-sampling noise entirely, so **power planning
targets panel count (number of contexts and queries) only**.

**Sign discipline.** Gates and stop conditions evaluate the **prior-averaged** quantity with a
**panel-level** standard error over contexts. A negative regret on an individual panel or context is
legitimate and expected: the Bayes predictive hedges, so a model can beat it on particular realizations. Only
the prior-averaged expected regret is constrained to be non-negative, and a prior-averaged negative beyond
panel-level SE is an instrument failure, not a result.

**Retirement.** The pre-fix in-task regret values (eps 0.0/0.1/0.25/1.0 = -0.0058 / -0.0093 / +0.0114 /
+0.0124) were single-sampled-outcome means on a 33-44 query subset of one shared 120-query panel, quoted with
an across-seed SE (0.0009 at eps 0) that omits panel uncertainty. They are retired, not corrected.

## E.4 (D4) Verdict rebuild

`classify()` is **deleted**. `classify_intask` is **rebuilt from this spec** and renamed to the branch
vocabulary below; the old label strings are retired because `THESIS_SUPPORTED` previously denoted the
opposite branch.

**Per-eps input tuple.** For each eps cell: `(width_exact, pfn_fidelity +/- CI, expected_regret +/- CI)`.

- `width_exact`: the exact Q1 identified-set average width (substrate quantity, model-independent).
- `pfn_fidelity`: the PFN's own d=3 s/w tracking statistic per E.5, with a panel-level CI.
- `expected_regret`: E.3's prior-averaged exact expected regret, with a panel-level CI.

**BRANCH 1 — STRUCTURE_TRACKS_IDENTIFIABILITY.** Declared iff all three hold:
  (a) `expected_regret` is below the registered gate at **every** eps cell;
  (b) the `eps = 0` fidelity is **consistent with the null** (its CI covers the no-signal value);
  (c) the association between `(1 - width_exact)` and `pfn_fidelity` is **positive with a CI excluding zero**,
      measured over **at least 4 cells** by both an isotonic fit and a Spearman rank correlation, with the CI
      from a seed-and-panel jackknife. Both must agree in sign; the Spearman CI is the gate.

**BRANCH 2 — PREDICTIVE_CAPTURE_WITHOUT_STRUCTURAL_TRACKING.** Declared iff (a) holds and (c) fails.

**BRANCH 3 — PREDICTIVE_GATE_FAILS.** Declared iff (a) fails at any cell. Not a paper outcome; an instrument
or budget finding.

**BRANCH 4 — NOT_EVALUABLE.** Fewer than 4 cells carry a `pfn_fidelity` with a CI by the decision date.

**Power requirement (new, and the defect that motivated the rebuild).** The rebuilt classifier must be
falsifiable by the model it judges. A unit test `tests/test_verdict_power.py` asserts that substituting the
**exact Bayes oracle** for the PFN does **NOT** return BRANCH 1: the oracle attains regret 0 and leaves
`width_exact` unchanged, so under the old rule it passed the thesis gate. Requirement (c) is what makes the
rule model-sensitive, and the test enforces it.

**Written default (D4, binding).** If the verdict is not evaluable by **2026-09-04**, the d=2 headline is
**automatically selected** and the eps-dial ships as a pending section. This default is invoked without
further deliberation.

## E.5 (D5) The C5 model-side series

The model-side structural series is the **PFN's own d=3 s/w tracking per eps**, the same estimand family as
C1 at d=2, computed against the 6-ordering anchor set. Residual-stream probe R-squared is **secondary** and
may not carry the verdict.

The separately-trained supervised order classifier is **demoted** to a **data-difficulty context curve**: it
measures how recoverable the order is from the data by a model trained directly on the order label, and it is
reported as context for the substrate, never as the PFN's fidelity.

At `eps = 0` the classifier's KL to the exact posterior is `5e-6`. This is **reported as the model-side null**,
the correct behaviour demanded by identifiability theory, and it is **not** to be described as an artifact.
MAP accuracy is **undefined** at `eps = 0` because the exact order posterior is uniform there, so MAP measures
arbitrary tie-breaking; MAP is not reported at that cell. Every reported MAP cell carries an n=60 binomial CI,
and the pre-fix series (0.2722 / 0.1500 / 0.2778 / 0.6500 / 0.8722 against a 1/6 floor) is not to be described
as monotone.

## E.6 (D6) eps = 0.1 keep-or-cut, deferred

The `eps = 0.1` cell is plotted on the PFN-side overlay **iff** its post-fix calibrated bias floor is
**strictly below 0.5 times the exact signal** at that cell. This is a measurement, taken after E.1's
unification and the retrain, not a scheduling decision. The cell is not on the cut list.

## E.7 (D7) C2 at d=3: the efficiency estimand

C2's d=3 estimand is

    eta = (model matched-pair gain) / (exact-achievable matched-pair gain on the same pairs)

on identical matched pairs. The **mirror convention is preregistered here as the max-contrast ordering**: for
a context with true ordering `o`, the mirrored partner is the ordering in the 6-element set that maximizes the
exact achievable gain against `o`, computed by enumeration. `eta` is retrofitted to d=2, where the mirror is
the unique opposite ordering and the retrofit is definitional.

**No "C2 does not port" prose may be written until `eta` exists at d=3.** A low `eta` with a small
exact-achievable denominator is a statement about the substrate's headroom, not about the model.

## E.8 (D8) Claim-tier discipline

C6 is renamed **representational causal-use** and filed beside C4. The words **circuit, mechanism, head,
localized, localizable** are banned from C6 captions and claim sentences. The Phase-A component screen is
reported as a **seed-inconsistent null** in the same paragraph as the Phase-B result.

F4 is **layered**: the exact model-free ladders are computed and plotted first so the figure exists
independently of the PFN overlay. The **Q3 PFN panel is dropped**, with a caption stating that the vanilla PFN
has no intervention channel, so the Pearl L2 rung has no model-side counterpart **by construction**
(`corrected_tomography.pfn_panel_predictions` returns `None` for every interventional spec). This is a
statement about the model class, not a missing measurement.

## E.9 (D9) Standing invariance test: sampler <-> density

A permanent test in the suite, `tests/test_sampler_density_invariance.py`:

For every `eps in {0.0, 0.1, 0.25, 0.5, 1.0}` and each of the three generation paths
(`sample_residuals`, `_vectorized_observational`, and `generate_observational`), simulate N residual draws,
compute the mean of `residual_logpdf` over them, and compare to the **quadrature entropy** of the declared
oracle law, `-H(p_eps) = integral p_eps(z) log p_eps(z) dz`, evaluated by quadrature on the same grid family
the predictive uses.

If a path samples from the declared law, its mean log-likelihood converges to `-H(p_eps)`. If it samples from
a different law, Gibbs' inequality forces the mean strictly below `-H(p_eps)`. The test asserts agreement
within a Monte-Carlo tolerance derived from N and the sampled variance.

**Acceptance criterion, stated in advance:** this test **MUST FAIL** on the pre-fix code at
`eps in {0.1, 0.25, 0.5}` for the `sample_residuals` and `_vectorized_observational` paths, and **MUST PASS**
for every path at every eps post-fix. A test that passes on pre-fix code is not testing the defect and must be
strengthened before the fix lands. The pre-fix failure is to be captured and committed as evidence.

**Never-cut.** This test and the errata appendix are added to the never-cut set.

## E.10 (D10) Instrument-errata appendix

The paper carries an appendix disclosing, without euphemism:

1. the **Branch-B map defect** and its three companions, with the supersession note's four technical defects
   reproduced, and the statement that Branch B's sealed numbers license no conclusion;
2. the **mask defect**: the three-way mixing-granularity disagreement at `0 < eps < 1`, when it entered
   (commit `3360b99`), which cells it affected, and that the affected weights were retired and retrained;
3. the **invariance test** of E.9, its pre-fix failure and post-fix pass, as the standing guard;
4. the **verdict rebuild**, including that the superseded rule was passed by the exact Bayes oracle;
5. the **estimand change** of E.3, and the retirement of the single-sampled-outcome numbers.

Nothing is silently repaired. Where a number changed, both values are shown with the reason.

## E.11 Provenance discipline (unchanged, restated as binding)

Every claim carries a ledger tag: **verified** (re-derived from a committed artifact by someone other than its
producer), **red-teamed** (adversarially checked, not independently re-derived), or **unverified**. Artifacts
reachable only on WashU are **unverified** until mirrored into version control.

A stop-and-report condition means **stop and report**. It does not license improvising a repair, relaxing a
gate, or proceeding on the assumption that the condition was spurious.
