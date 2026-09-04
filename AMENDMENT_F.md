# Amendment F — the spec freeze before the retrain

**STATUS: NOT LOCKED. BOTH OPEN CLAUSES NOW HAVE ADJUDICATED TEXT AWAITING
COUNTER-SIGNATURE.**

1. **F.2d.4** failed the T3 validation applied to it: measured FPR 0.2160 against a
   nominal 0.05 under cell-level dispersion, 0.3352 at sd 0.10. ADJUDICATED to the
   conjunction repair. The measurement that ruling made a precondition of its own
   adoption has since been taken: worst null FPR 0.04648, power at beta = 0.2 of
   0.4604 and 0.1905, all clearing their stated thresholds.
2. **F.2c's "identical readout pipeline"** is not what the code does. ADJUDICATED to
   none of the three tolerances. The tolerance enters only the identified-set branch,
   and `tol_eff = max(tol, residual * 1.2)` restores a per-curve instrument whatever
   nominal value is shared, so the ladder moves to the projection branch, which takes
   no tolerance argument at all.

Both adjudications are recorded in place below with their exact operative text, their
stated costs, and what was known at the time of signing. Neither is a lock.

Everything else in this document is complete and its code state is ready to
digest. The lock and W2's launch are held on those two clauses. W2 itself is
unaffected by the second: it produces the regret and the persisted checkpoints
and computes no y.

Signed: Bo, 2026-08-22. Supersedes Amendment E only where stated; E's decisions
D1-D10 otherwise remain in force, as does its provenance discipline and its rule
that **stop conditions mean stop and report, never improvise**.

Written BEFORE the retrain it governs, and, per the ordering constraint in the
signing message, F.2b was implemented in the verdict code BEFORE the digest in
F.0 was taken. Taking the digest first would have locked the width-primary
implementation that F.2b exists to replace.

---

## F.0 — Code state at lock

Recorded so that anything done after this point is a change against a fixed
reference rather than a silent edit.

The file list is what the amendment actually depends on, which is wider than it
was when F.0 was drafted: F.2d.5 and F.8 put load-bearing behaviour in the split
panel, the status tags and the producers, and a code-state block that omits them
records less than it claims to.

- repo HEAD at lock: `b05e32093d93545b3d54c4ae66a4755595348e44`
- `src/pfn_dag_verify/corrected_verdict.py`            sha256 `c47dd3be482342ff3d02ea8ff0ac3c86f04f7faca55feebbe76fd9074aa0c14d`
- `src/pfn_dag_verify/corrected_oracle.py`             sha256 `0f1f8bf8377e20302c216dfb137c57b2ea45f33a7326d9eb3180f048d5393e3c`
- `src/pfn_dag_verify/corrected_models.py`             sha256 `d29e0325dff89bf50484bf73dda8b9a4139c2bad65dce07524c51baa44e5b9d9`
- `src/pfn_dag_verify/corrected_sem.py`                sha256 `424ed0a34e0683451c6b80773d4400f70d36605cd66cb7bec1ca16811b194668`
- `src/pfn_dag_verify/split_panel.py`                  sha256 `b0232ba8843d1e195e36ad9adb1be53cde7578e8f266a64056eb55d1b098b7dc`
- `src/pfn_dag_verify/artifact_status.py`              sha256 `18ec46c98d907cd39249111d717d8e64f2100fbacfdc62105f106192743ed200`
- `src/pfn_dag_verify/corrected_deficit_run.py`        sha256 `3340be106a797c46a91e0b66f4519b98a43712b93e0e42fbb09032b35f7cdedb`
- `src/pfn_dag_verify/corrected_identifiability_run.py` sha256 `293a1f87a8493b40f4aecaa8f02d8eadd43d7fefcb1cc6c2640f8bbb4907ad98`
- `src/pfn_dag_verify/w2_run.py`                       sha256 `160d892215f1ef1f8f8b29f96037b0378482602febaba5dcba21dee2237a5167`

Amendment E remains locked at sha256
`fed795cbf05ecd9d452a5594e4696562ffe29eb9b24e58dbd923bd9e4a4668c0`.

---

## F.1 — D3, exact-expectation estimands

Every behavioural estimand becomes an exact expectation over outcomes: the
bin-summed expected NLL, not the NLL of a single sampled outcome. This applies
to the regret, the E1b cross-score, and the tracking regressands.

The scorer to be replaced is the single-sampled-outcome path in
`corrected_models.evaluate_pfn_checkpoint`. Gates evaluate the PRIOR-AVERAGED
quantity with a panel-level SE. A per-panel negative remains legitimate — a
Bayes-optimal predictor hedges, and hedging shows up as a negative regret on
individual panels — so the regret gate is applied ONE-SIDED against the CI's
upper bound. This is already how `corrected_verdict.classify` reads it.

Power planning targets PANEL COUNT only. See F.2.

---

## F.2 — R5, the eval design

**Panel.** A target-fixed expected-regret panel. Panel contexts are iid draws
from the world prior: `make_eval_panel` samples `k ~ U(K)` and `o ~ U(O)` per
context and generates observationally from that latent state, which matches
`World.prior_lo`'s uniform product prior. This is confirmed here rather than
assumed, because it is what licenses the across-context SE as the SE of the
prior-averaged estimand rather than of a fixed finite population.

Panels are **per-eps** (the world carries eps, so a panel drawn against one eps
is not a panel for another) and **fixed across seeds and scales within an eps**,
so every model in a cell is scored on identical evaluation points.

**Context count — a preregistered RULE, not a preregistered number.** The count
is filled by measurement, not certified today. The rule:

> `n = max(500, smallest n clearing 2 SE on regret, smallest n giving a usable
> CI on the NORMALIZED fidelity at the weakest surviving cell)`, capped at 1500.

with **STOP-AND-REPORT** if even the cap fails any of the three. At that point
the mismatch is between the gate width and the instrument's noise floor, and it
is the GATE that needs re-examination, not `n`; do not raise the cap to make a
number work.

`n` is the count PER HALF. F.2d.5 splits each cell's panel into two disjoint
halves — the deficit on A, the readouts and the regret on B — so a cell draws
`2n` contexts and no context contributes to both axes. The halves are fixed by a
recorded seed, and the split is part of the panel artifact rather than a
re-derivation at read time.

**Post-W2 quality flag (reported, NOT a gate).** After the fleet lands: if the
weakest surviving cell's fidelity CI half-width exceeds half the cross-cell
range of `y`, the ladder is being read at a resolution the panel does not
support, and the report says so in the caption. This is deliberately not a gate.
It is a fact about precision that a reader is entitled to before they weigh the
slope, and turning it into a pass/fail after the sizing rule has already fired
would just be the cap argument again in a second costume.

The 500 floor is load-bearing and is NOT the regret gate's requirement. On the
corrected in-task numbers the regret term alone now selects `n = 60` —
technically compliant, zero margin — and collapsing the rule to that floor would
be a mistake, because the panel serves more than the regret gate:

- the fidelity `y` is redefined by F.2c below and its variance under the new
  definition is unmeasured;
- the eps=0.5 in-task cell has never been run, so its spread is unobserved;
- exact-scored contexts are cheap on an idle cluster.

The floor was always about margin and outcome noise. It stays.

Two caveats recorded now so neither is discovered later as a surprise:

- Gate D measures on a 100-step model. A model that far from convergence has a
  hotter context-to-context regret spread than a converged one, so the rule will
  size CONSERVATIVELY. That is the right direction of error and it is accepted.
- The variance components are lopsided, so ADDING SEEDS BUYS ALMOST NOTHING: all
  seeds share one panel, making the panel term a fixed effect common to them.
  Re-derived from `trackb_eps*_corrected.json` on the IN-TASK estimand, which is
  what gate (a) actually reads (base scale, n_contexts 60):

  | eps | panel SE | across-seed SE | ratio |
  |-----|----------|----------------|-------|
  | 0    | 0.0068  | 0.0011         | 6.4x  |
  | 0.1  | 0.0170  | 0.0040         | 4.3x  |
  | 0.25 | 0.0193  | 0.0015         | 12.5x |
  | 1.0  | 0.0238  | 0.0077         | 3.1x  |

  An earlier draft of this clause quoted 15x-36x from `trackb_eps*.json`. Those
  are the MIXED-TARGET panel's numbers, retired by E.3 and not the estimand the
  gate reads. The qualitative conclusion survives; the magnitude was inflated
  about threefold by out-of-task contamination.

- Consequently the 500-context floor is comfortable rather than thin. Scaling the
  worst cell (eps=1.0, SE 0.0238 at 60 contexts) as 1/sqrt(n) against the 0.05
  gate: 60 -> 2.10 SE, 500 -> 6.06 SE, 1500 -> 10.49 SE. The current 60-context
  panel ALREADY clears 2 SE. The rule and its cap stand as written, but on this
  evidence the stop-and-report clause is unlikely to fire, and Gate D's job is to
  confirm the post-D3 numbers rather than to rescue the count.

**The primary axis, measured on the W2 panel (half A, n = 500).** Recorded here
because the panel is what F.2's rule sizes and this is the first measurement at
that size. Artifacts `raw/postF/deficit_w2/`, manifest `raw/postF/w2_panels.json`:

| eps  | deficit  | SE      | gap to previous | z of gap |
|------|----------|---------|-----------------|----------|
| 0.00 | 0.000000 | 0.000000|                 |          |
| 0.10 | 0.035537 | 0.002658| +0.03554        | 13.4     |
| 0.25 | 0.155389 | 0.005960| +0.11985        | 18.4     |
| 0.35 | 0.257222 | 0.007608| +0.10183        | 10.5     |
| 0.50 | 0.415179 | 0.009791| +0.15796        | 12.7     |
| 0.75 | 0.627382 | 0.010017| +0.21220        | 15.1     |
| 1.00 | 0.766417 | 0.008575| +0.13904        | 10.5     |

Every adjacent pair separates at z >= 10.5 against the near-tie clause's
z = 1.96, so **the F.2b near-tie clause does not fire on the primary axis at
this panel size**. The near-tie the power simulation above measures is on the
SECONDARY axis, where it is real. This is stated now, before any y exists, so it
cannot later be presented as a discovery.

The 60-context deficits differ from these by up to 0.044, which is within about
1.5 SE of the old estimates and is what a different panel at a different size
looks like. They are retagged RETIRED with a reason under F.8 and the loader
refuses them; `deficit_w2/` is the live set.

### F.2a — the pre-flight, and what it gates

One GPU job, four gates, before the retrain fires:

- **Gate A, persistence.** `.pt` files exist on disk for the final model AND for
  every requested intermediate `ckpt_steps` value — the gate repaired under E
  was exactly the one that silently skipped `ckpt_steps` — and the
  `.provenance.json` sidecar carries torch, CUDA, numpy and device. **Including
  the dose-0 state**: F.2c's floor curve does not exist without it, so a fleet
  that trains without persisting dose-0 cannot produce a calibrated y at d=3 and
  would have to be re-run.
- **Gate B, functional reload.** Not state-dict equality. Load the persisted
  file in a fresh process, run the readout on a FIXED mini-batch through both
  the in-memory model and the reloaded one, and require agreement to float
  tolerance. Dtype, device and RNG surprises live precisely in the gap between
  "the state dict matches" and "the outputs match", and a state-dict check would
  pass while the reloaded model computed something else.
- **Gate C, the d=3 readout end to end.** The full s/w readout and probe path
  off the RELOADED checkpoint. Plumbing validity only; no numbers are claimed
  from a 100-step model.
- **Gate D, variance decomposition.** Per-context scores under both the old
  sampled scorer and the new exact one, which verifies F.1 and yields the
  panel/seed split that fills F.2's hole.

### F.2b — PRIMARY X-AXIS: exact average-case identifiability

The x-axis of the association gate becomes, per eps:

> the panel mean of the normalized order-marginal entropy deficit
> `1 - H(p(o | D)) / log O`, where `p(o | D)` is the order marginal of the exact
> latent posterior and `O = d!`, averaged over the SAME eval panel the PFN is
> scored on.

The LP identified-set width is **demoted to secondary**: computed and reported
in the appendix, never gating.

**Estimand convention.** The average is taken OVER contexts of the per-context
deficit, `E[1 - H/log O]`, and explicitly not the deficit of a pooled posterior.
The claim is about average-case identifiability — how identified a typical
dataset drawn from the prior leaves the ordering — which is the quantity the PFN
is scored against. Pooling first answers a different question.

**Normalization.** `World.__post_init__` builds `prior_order` uniform on `O`
orderings, so `log O` IS the prior's order entropy and the deficit is exactly the
fraction of it that the data removed. Deficit 0 means the data left the ordering
at its prior; deficit 1 means one ordering carries all the mass. Both axes are
oriented larger = more identified, so gate (c) is a positive-association test
under either.

**Why this is not gate-shopping.** Written when the gate was a rank statistic:
Spearman on five cells is invariant to any strictly monotone re-axing, so F.2b
could change the gate ONLY by reordering cells. Its value was validity and
near-tie diagnosis, not passing anything — which is precisely why it was costless
to adopt before the model-side series existed, and would have been fishing
afterwards. The empirical check of whether the two axes reorder any cell is
recorded in F.4.

F.2d.1 has since made the gate a weighted SLOPE, which is not rank-invariant, so
the argument no longer holds in that form and is not claimed. What holds instead
is the ordering: the re-axing was signed and implemented before any model-side y
existed, and the slope test was signed against the axis F.2b had already fixed.
Neither was chosen with a number in view. The two rungs F.6r adds are subject to
the same discipline and are recorded pre-data for the same reason.

**Two structural reasons for the swap**, both independent of any outcome:

1. The width ladder is computed on a separately configured context panel whose
   `n_rows` the ident artifacts never recorded (F.5), so x and y sat at
   different information levels. The deficit is measured on the scoring panel,
   making the alignment automatic rather than declared.
2. The deficit is a functional of the exact posterior and calls no linear
   program, so it cannot inherit an LP solver path. The eps=0.5 simplex cell
   that could not clear a 24-hour wall while interior-point cleared it in 16
   minutes is invisible to this axis.

**Rank tie handling (preregistered).** The gate is now the F.2d.1 slope and
does not rank anything, but the F.2d.2 permutation companion does. Ties take
MIDRANKS, formed explicitly via `scipy.stats.rankdata(..., method="average")`
rather than relying on `spearmanr`'s default, so the preregistered rule cannot
drift with a library version. If either rank vector is constant the association
is not evaluable — a constant series is an instrument failure, not a null
result.

**Near-tie clause (binding).** Adjacent rungs, ordered by the primary axis,
whose deficit CIs overlap at z = 1.96 are reported as UNRESOLVED. **An
unresolved adjacent pair does not fail gate (c) on its own.** With the rungs
ordered but two of them statistically indistinguishable, the F.2d.1 slope
remains the preregistered gate; the pair is a precision statement about what the
dial can separate, not evidence against tracking. This sentence is written
before the data so it cannot be composed afterwards to explain a result. It
holds a fortiori under the slope test, which uses the rung SPACING rather than
only the order and so degrades smoothly through a near-tie instead of hinging
on which side of it two cells fall.

Both axes are always computed. `association_secondary` carries the width-axis
answer and is NOT independent corroboration; the output must not read as though
it were. Under the rank gate the two blocks were bit-identical whenever the axes
ordered the cells alike. Under F.2d.1's slope they can differ numerically, which
is worse rather than better: two slopes on the same five y values against two
monotonically related x's are one measurement expressed in two units, and small
numerical disagreement invites reading them as a replication. `axis_comparison`
therefore reports the two x vectors, their rank orders, and whether those orders
are identical — the only channel through which a re-axing can change which cells
are being compared — and the evidence line says in as many words that the
secondary axis is a re-expression, not a second measurement.

### F.2c — CALIBRATED Y-AXIS: fraction of achievable

F.2b fixed the x. This fixes the y, and the two are the same problem.

**The measurement.** The y-instrument's GAIN is itself a function of eps. The
order-conditional predictive anchors coincide exactly at eps=0 (mean pairwise KL
0.000000) and separate as eps rises: 0.0031 / 0.0158 / 0.0588 / 0.3493 at eps
0.1 / 0.25 / 0.5 / 1.0, rank-identical to the primary axis at Spearman +1.000.
So part of any RAW fidelity increase is the measuring stick lengthening, with no
model in it.

**Two readings, both correct.** As a confound it must be removed: gate (c) could
otherwise pass for a model whose order sensitivity is constant in eps. As
physics it is the identifiability null appearing inside the instrument — at
eps=0 no readout can have gain because there is no order information to read,
which is consistency rather than defect. The standard calibration design serves
both.

**The design.** Three curves through the IDENTICAL readout pipeline:

| curve   | what goes through the readout        | true tracking |
|---------|--------------------------------------|---------------|
| ceiling | the exact Bayes posterior            | ideal         |
| floor   | a dose-0 / untrained checkpoint      | zero, by construction |
| model   | the trained PFN                      | to be measured |

and **y := (model − floor) / (ceiling − floor)**, the model's position in
[floor, ceiling]: fraction of achievable.

The Spearman +1.000 above establishes that the channel EXISTS; it does not give
its magnitude. **The floor curve is what quantifies the leakage**: if an
untrained model's raw readout rises with eps, raw-y associations are confounded
by exactly that much. `calibration_leakage()` reports the floor's movement
across the dial, its Spearman against the primary axis, and the share of the raw
rise it accounts for. A floor that rises monotonically is the finding, not a
nuisance.

**F.2c-tol — the shared readout tolerance. ADJUDICATED 2026-08-24 by a
cross-model (Fable) panel of three independent lenses plus a synthesis, on the
artifact alone. AWAITING BO'S COUNTER-SIGNATURE.**

SIGNED (F.2c-tol, adjudicated 2026-08-24): none of the three listed tolerances is adopted. The tolerance enters only the identified-set branch of the readout, and the measured record shows no value of it can serve the ladder. A tolerance the floor can satisfy is loose enough that the identified set is the whole simplex (id_avg_width 1.00000 at every panel at the derived floor tolerance 0.8355, run cancelled past 210 s), so the floor's identified-set readout is a constant and calibration_leakage has nothing to measure. A tighter tolerance triggers the feasibility clause tol_eff = max(tol, residual * 1.2) in tomography_context, which restores a per-curve effective tolerance without announcing it, so the pipeline is identical in name only. The fidelity ladder is therefore read on the projection branch, which takes no tolerance argument. Operative rules follow. (1) The per-context readout for each curve is the projection branch of tomography_context: the projection LP under mode "project" and the structural statistic it feeds. The readout value is r = 0 minus structural_js, computed for the curve's own predictions on the same half-B panel, the same contexts, and the same solver setting. The ceiling pushes the exact Bayes posterior's predictions through this path, the floor pushes the persisted dose-0 checkpoint's predictions, the model pushes the trained PFN's predictions. y and SE_y are computed from these per-context triples exactly as F.2d.6 specifies, and F.2d.7's gain test applies unchanged. (2) Identified-set LPs are not computed for the floor curve or the ceiling curve. On the model curve they remain the F.2b appendix diagnostic that never gates, run at the fixed nominal tolerance 1e-2, the pre-existing run_tomography default. The derivation tol = max(js_mean * 2.0, 1e-3) is retired for every new artifact; historical Track B artifacts keep their recorded tol and are not recomputed. (3) Every new tomography artifact persists, per context and per curve, the projection residual, the nominal tol, and tol_eff wherever an identified-set LP ran; tol_eff above tol on the model curve is recorded in the artifact and stops nothing. (4) Before the lock, one Mac measurement is taken and persisted: the dose-0 projection residual and the floor readout r at every surviving eps cell, one untrained seed, the fleet's panel size. This characterizes the floor's readout variance before any fleet depends on it. A cell whose measured ceiling minus floor gain falls below CALIB_MIN_GAIN exits the fidelity series through F.2d.7 and the exit is reported by survivorship. What this choice costs: the ladder carries no identified-set comparison between the three curves, so no whole-ladder width statement can ever be made; the floor's readout is an empirical per-context quantity rather than an exact zero, and its variance is unknown until the clause 4 measurement runs; and every F.2c gain and leakage number must be taken under this readout, since no number measured under the per-model tolerance carries over. Known at signing: derived tolerances 0.1979 (trained) and 0.8355 (dose-0) at eps=0.5, the 26.8 s and cancelled-past-210 s timings, the whole-simplex floor width, converged base js 0.046 to 0.107 across the dial, and the absence of any fidelity artifact on disk, so this choice precedes every number the gate will consume.

---

**Preregistered consequence 1 — eps=0 exits the fidelity series by
construction.** Fidelity is undefined where the ceiling equals the floor, and at
eps=0 the Gaussian order-invariance identity makes that exact. This is a fact
about eps, not a decision about the model, and `Cell.has_fidelity` returns False
there without anyone having to remember to exclude it. eps=0 remains an X-ONLY
anchor, plotted at exactly 0. It carries the **null-type claim** instead — the
model must reproduce the uniform order posterior, the reproduce-uniform genre of
the classifier's KL 5e-6 — which is a stronger statement than a fidelity value
there would have been. Gate (b) is rewritten accordingly: it can no longer be
satisfied by a broken instrument, a good model and a random model alike, and now
fails only if the eps=0 readout turns out to HAVE gain, which would mean the
instrument is not measuring what F.2c assumes.

**Preregistered consequence 2 — the cell-count collision, written before it can
bite.** Removing eps=0 leaves exactly four cells against E.4c's ">= 4 cells"
wording: zero slack, and D6 can still take eps=0.1. Discovering that on Sep 4 is
precisely the failure this discipline exists to prevent. Therefore:

> The association gate runs over **all surviving eps > 0 cells, minimum 3**
> (`MIN_CELLS = 3`). Below `MIN_CELLS_PREFERRED = 4` the verdict ships with the
> shortfall stated in the caption and in the evidence line; it is never upgraded
> silently. If fewer than 3 survive, the written default of E.4 applies.

F.2d.4 makes the sub-4 case operational rather than merely caveated: below four
surviving cells the slope is still computed and REPORTED, but it is descriptive,
and BRANCH 1 then requires the extreme-cell contrast instead. At three cells
`df = k - 2` is 1, which is a t test on one degree of freedom — a statistic
whose confidence interval is wide enough to be honest about, and not one to hang
the paper's affirmative branch on.

F.6r moves the expected count the other way. With eps = 0.35 and 0.75 added, six
cells survive eps=0 rather than four, `df` is 4 rather than 2, and the
`MIN_CELLS_PREFERRED` shortfall clause is two D6-style cuts away from firing
instead of one.

**Operational corollary, binding on W2.** The floor curve does not exist unless
the fleet persists dose-0 states, so **dose-0 checkpoint persistence is added to
the F.2a pre-flight assertion list (Gate A)**. A W2 fleet that trains without it
cannot produce a calibrated y at d=3 and would have to be re-run.

**SECOND OPEN CLAUSE (2026-08-23): "identical readout pipeline" is not what the
code does, and the floor curve is where it bites.** Found by the pre-flight,
which is what the pre-flight is for.

Every readout call site derives the identified-set tolerance from the model
being read out: `tol = max(js_mean * 2.0, 1e-3)`, in `corrected_trackb`,
`corrected_tomography_run` and the pre-flight alike. So a well-fit model is read
at a tight tolerance and a badly-fit one at a loose tolerance. The three F.2c
curves would therefore pass through three DIFFERENT instruments, and
`y = (model - floor)/(ceiling - floor)` would be a ratio of quantities measured
under different settings rather than a position within one.

The floor curve is the extreme case in both senses. Measured on the cluster at
eps=0.5, one tomography context, same context and same solver:

| curve         | js_mean | derived tol | wall     |
|---------------|---------|-------------|----------|
| trained (100 steps) | 0.0990 | 0.1979 | 26.8 s   |
| dose-0        | 0.4178  | 0.8355      | > 210 s, cancelled |

and at a tolerance that loose the identified set is the WHOLE simplex
(`id_avg_width` 1.00000 at every panel, measured directly). So the floor curve
is simultaneously the most expensive branch and the least informative one, and
at W2's scale -- seven eps by three seeds by a panel of contexts -- it is the
branch that would have consumed the fleet. Discovering that in W3 is precisely
the failure this pre-flight exists to prevent.

The repair is a SHARED tolerance across the three curves, which removes both
problems at once: it makes the pipeline identical as F.2c already claims, and it
sets the floor curve's tolerance to the trained model's tight one. **Choosing
that tolerance changes what the readout measures, so it is a signature and not
mine to make.** Options, none adopted:

1. **The model curve's tolerance**, applied to floor and ceiling. Keeps the
   headline curve's instrument exactly as it is today and re-reads the other two
   through it. Per-cell, so the tolerance still varies across eps.
2. **A single fixed tolerance for the whole ladder**, chosen pre-data (1e-2 is
   the module default). Makes every cell and every curve comparable, at the cost
   of a tolerance that fits no cell's predictive error in particular.
3. **The exact Bayes ceiling's tolerance**, which is the tightest of the three
   and the only one not set by a model's own error.

Until this is signed, the F.2c calibrated y cannot be computed. It does not
block W2, which produces the regret and the persisted checkpoints and computes
no y.

---

### F.2d — gate (c)'s test statistic (SIGNED, Bo, 2026-08-23)

E.4c's clause — "Spearman with a bootstrap CI excluding zero" — is retired with
cause and replaced. What follows is signed and binding; the numbered clauses are
the operative text.

**F.2d.1 — the gate is a one-sided weighted slope test.** With `y` the
fraction-of-achievable of F.2c and `x` the order-marginal entropy deficit of
F.2b, over the surviving cells (eps=0 is out by construction; D6 may still drop
eps=0.1): weighted least squares of `y` on `x` with weights `1 / SE_y^2`, the
per-cell SEs coming from iid contexts. `H0: slope <= 0`, one-sided p from the t
distribution with `df = k - 2`, alpha = 0.05.

Errors in `x` attenuate the slope toward zero. For a gate that fires only on a
POSITIVE slope this is conservative, so it is accepted rather than corrected —
recorded here as an accepted bias so that no one later reads the attenuation as
a discovery. Correcting it would loosen the gate, which is the direction that
needs a signature, not a footnote.

**F.2d.2 — the permutation test is a companion, not a conjunct.** The exact
permutation p over cell orderings is computed and reported beside the slope test
on every run. It does NOT gate. Making it a conjunct would reimport the
knife-edge that motivated it: at four cells the exact one-sided floor is
`1/4! = 0.0417`, so a conjunctive rule would turn a single adjacent swap into a
failed gate while a flawless ordering scrapes past alpha by 0.008.

**F.2d.3 — post-mortem on the bootstrap, retired with cause.** The E.4c
bootstrap was wrong in two independent ways, and it is recorded rather than
quietly swapped out:

1. it resampled the wrong unit. Seeds within a cell share one panel, and the
   dominant variance component is context-level (F.2 measures the panel SE at
   3.1x to 12.5x the across-seed SE), so resampling seeds reproduces almost none
   of the sampling variability that matters;
2. it had no null. The cell assignment never moved, so the statistic asked
   whether rho was stably positive GIVEN the assignment and never whether the
   assignment could have arisen by chance.

Measured false-positive rate under a null where fidelity is independent of the
axis, 400 trials per configuration:

| cells | within-cell sd 0.01 | 0.05 | 0.2 | nominal |
|-------|---------------------|------|-----|---------|
| 5     | 0.445               | 0.435| 0.403| 0.025   |
| 4     | 0.477               | 0.465| 0.430| 0.025   |

The clause deciding the paper's affirmative branch fired about half the time
under the null. The bootstrap path is DELETED from the verdict module, not
bypassed: a retired statistic that remains callable is the same failure mode as
a retired artifact that remains loadable (F.7/F.8).

**F.2d.4 — fallback when the ladder is short. ADJUDICATED 2026-08-24 by a
cross-model (Fable) panel of three independent lenses plus a synthesis, on the
artifact alone. AWAITING BO'S COUNTER-SIGNATURE.**

F.2d.4 Fallback ladder: if surviving cells < 4, the association demotes to descriptive and BRANCH 1's minimum evidence becomes a conjunction of two tests over the surviving cells. The first is the extreme-cell contrast as implemented: weakest versus strongest surviving cell by x, one-sided z on delta-y with the two endpoint SEs added in quadrature, passing when the 95% CI excludes zero. The second is the descriptive weighted slope of F.2d.1 computed on the same surviving cells with the same inverse-variance weights, passing when the slope is positive with one-sided p at or below 0.05 from t on k - 2 degrees of freedom. gate_passes in descriptive mode reads the conjunction, and both must pass. The slope conjunct carries the calibration: the contrast alone measured a false-positive rate of 0.33525 against a nominal 0.05 at cell-level dispersion sd 0.10, the slope stayed between 0.04675 and 0.05400 in the same rows, and a conjunction cannot fire more often than its calibrated conjunct, so the composite is bounded at or below the slope's measured rate under every null tested. The cost of this rule is power on one degree of freedom at k = 3, and an under-firing fallback is the accepted failure direction under this prereg. That power is now measured and persisted at campaigns/corrected_20260812/raw/postF/f2d4_option_characteristics.json, 40000 trials per row: the conjunction's worst null false-positive rate is 0.04648 against a nominal 0.05, so T3's stop condition does not fire, and its power at beta = 0.2 is 0.4604 with no added cell-level dispersion and 0.1905 at dispersion 0.05. Both clear the voiding floor below. This clause leaves F.2d.2 untouched: the permutation companion stays reported and never gates, and its knife-edge rationale is specific to the discrete permutation floor, which at three cells is 1/6 and could never clear alpha at all. The measurement that this clause made a precondition of its own adoption has been taken and persisted, at the trial counts recorded in that artifact, and T3's stop condition applies to the recorded rates. If the recorded power at beta = 0.2 with zero added cell-level dispersion is below 0.15, this fallback is void, fewer than 4 surviving cells is NOT EVALUABLE, and E.4's written default governs. The Sep 4 written default is unchanged in every case.

**F.2d.4 FAILED ITS OWN VALIDATION. THIS AMENDMENT IS NOT LOCKED.**

The T3 discipline that convicted the E.4c bootstrap was applied to F.2d.4's
contrast, on the post-D6 three-cell ladder where the clause is actually in
force, reading `gate_passes` -- the quantity `classify` consults -- rather than
`slope_passes`. Measured false-positive rate, 4000 trials per row:

| cell-level dispersion sd | descriptive slope | F.2d.4 contrast | THE GATE | nominal |
|--------------------------|-------------------|-----------------|----------|---------|
| 0.00 (calibrated)        | 0.04675           | 0.02825         | 0.02825  | 0.05    |
| 0.02                     | 0.05400           | 0.07725         | 0.07725  | 0.05    |
| 0.05                     | 0.04825           | 0.21600         | 0.21600  | 0.05    |
| 0.10                     | 0.05200           | 0.33525         | 0.33525  | 0.05    |

The cause is structural rather than a bug. The contrast reads only the two
endpoint SEs and has no residual-based scale estimate, so cell-level dispersion
the SEs do not know about passes straight through it. The slope test in the same
rows stays between 0.047 and 0.054 because its estimated-scale `s^2` absorbs that
dispersion -- which is the property F.2d.1 was signed for and F.2d.4 does not
share. Under overdispersion this is the same failure mode as the bootstrap,
smaller in magnitude and in a clause that fires less often.

**T3's stop condition is "FPR > 2x nominal after implementation".** It has
fired, on a SIGNED clause, so this is a stop-and-report rather than a fix: what
decides BRANCH 1 is not the implementer's to change. Recorded here, unresolved,
with the repair options measured but not adopted:

1. **Give the contrast a scale.** Inflate its SE by `sqrt(scale_s2)` from the
   descriptive fit -- available at k = 3, where `df` is 1 -- and refer the
   statistic to `t` rather than the normal. This gives the contrast the same
   protection the slope has, at the cost of a very wide reference distribution
   on one degree of freedom.
2. **Make the contrast necessary but not sufficient**, by additionally requiring
   the descriptive slope to be positive at alpha. The slope is calibrated under
   overdispersion, so the conjunction inherits its protection; the cost is the
   knife-edge argument that F.2d.2 refuses elsewhere.
3. **Leave it and disclose.** The clause fires only below four surviving cells,
   and F.6r puts six on the ladder, so reaching it needs two D6-style cuts.

Scope of what this does and does not block. The PRIMARY gate is unaffected:
F.2d.1's slope is at nominal across every null tested, on both ladders. Nothing
measured so far is invalidated. What is blocked is the LOCK, because locking an
amendment whose fallback has a measured 21% false-positive rate would preserve
exactly the defect this process exists to catch, and W2's launch, because the
data must not precede the frozen spec.

**F.2d.5 — split-panel independence.** Per cell, draw `2n` iid contexts and
split them into fixed disjoint halves; the assignment is seeded and recorded.
`x` — the entropy deficit — is computed on half A ONLY. `y` — the model, floor
and ceiling readouts — and the regret are computed on half B ONLY. `x` and `y`
are never computed from shared draws. Both are functions of the same contexts
otherwise, and a context that happens to be unusually informative would push the
deficit and the readout the same way, manufacturing exactly the association the
gate is testing for.

**F.2d.6 — y is a ratio of panel means.** `y_cell = (mean_model - mean_floor) /
(mean_ceiling - mean_floor)`, never the mean of per-context ratios. Per-context
ratios have unbounded variance wherever a context's gain is near zero, and their
mean estimates a different functional.

`SE_y` follows from the same per-context data by linearization: with
`a_i = model_i - floor_i` and `b_i = ceiling_i - floor_i`, the influence value is
`u_i = (a_i - y * b_i) / mean(b)` and `SE_y = sd(u) / sqrt(n)`. This is the
standard ratio-estimator SE, it uses the iid contexts F.2d.1 asks for, and it
carries the floor/ceiling correlation that treating the denominator as fixed
would drop. The across-SEED spread of `y` is reported beside it as a diagnostic;
the gate reads the context SE, per F.2d.1.

**F.2d.7 — cell survivorship, generalizing D6.** Any cell whose MEASURED
floor-ceiling separation fails its floor test (`ceiling - floor <
CALIB_MIN_GAIN`) exits the fidelity series. eps=0 is the case known in advance;
this clause covers the ones that are not. Every exit is surfaced in the report
with its measured gain. A cell never leaves the series silently.

**F.2d validation, run BEFORE this amendment locked.** The retired bootstrap
was convicted by a null simulation. Its replacement does not get to skip that
test, and a failure here was a reason to change the gate rather than a footnote
about it. `scripts/f2d_gate_validation.py`, 4000 trials per configuration,
artifact `raw/postF/f2d_gate_validation.json`.

FALSE-POSITIVE RATE under nulls where `y` carries no dependence on `x`, at the
measured per-cell SE PATTERN and the real rung spacing, on both ladders:

| null                                   | primary | secondary | nominal |
|----------------------------------------|---------|-----------|---------|
| calibrated (reported SE is the truth)  | 0.0505  | 0.0548    | 0.05    |
| SE understated 1.5x                    | 0.0437  | 0.0465    | 0.05    |
| SE overstated (true = 0.667x reported) | 0.0428  | 0.0495    | 0.05    |
| cell-level random effect, sd 0.02      | 0.0460  | 0.0495    | 0.05    |
| cell-level random effect, sd 0.05      | 0.0485  | 0.0495    | 0.05    |

The last two are the ones that matter. They add between-cell variance the
reported context SE knows nothing about, which is the structure that put the
E.4c bootstrap at 40-48%, and the slope test stays at nominal through it. That
is the estimated-scale convention doing its job: excess scatter about the line
inflates `s^2` and the t shrinks to match.

SCALE INVARIANCE, asserted rather than assumed, because it is what licenses
sweeping an SE scale that is not yet measured: multiplying every SE by 37 leaves
t unchanged to 1e-12 (9.726815532319 vs 9.726815532318). The weights enter only
through their ratios.

POWER, 1000 trials per slope. On the primary ladder: 0.234 / 0.540 / 0.945 /
1.000 at true slope 0.05 / 0.1 / 0.2 / 0.3. On the secondary ladder: 0.098 /
0.199 / 0.410 / 0.680 at the same slopes.

**A near-tie costs almost nothing, and an earlier draft of this clause said
otherwise.** That draft read the primary-versus-secondary gap as the price of a
near-tie. It is not: the two ladders differ in RANGE as well as in spacing, and a
compressed range lowers `Sxx` on its own, so the comparison confounded the two.
Tested properly -- x endpoints pinned, one interior rung sliding toward its
neighbour, so spacing is the only thing that moves -- power at slope 0.2 goes

| interior gap | 0.2598 | 0.1948 | 0.1299 | 0.0260 | 0.0026 |
|--------------|--------|--------|--------|--------|--------|
| power        | 0.940  | 0.931  | 0.923  | 0.931  | 0.918  |

which is flat to within simulation noise across a hundredfold collapse of the
gap. This is what F.2b's near-tie clause asserts a fortiori under a slope, now
measured rather than argued: two rungs at nearly the same x contribute almost
nothing to `Sxx`, so the fit is carried by the rungs the dial does separate and
degrades smoothly instead of hinging on which side of the tie they fall.

The correction is recorded rather than quietly applied because the retracted
version was the more flattering one: it made F.6r look necessary for a reason
that is not true. F.6r's case rests on the permutation floor and the degrees of
freedom, which are arithmetic, not on a power cost that does not exist.

**A heterogeneous SE misreport is the null that does move the gate**, which is
the reason the uniform rows above are kept only as the demonstration that they
cannot. Overstating one cell's confidence by 4x, at the high-leverage end:

| pattern (per cell)   | primary | secondary |
|----------------------|---------|-----------|
| 1 / 1 / 1 / 2        | 0.0757  | 0.0668    |
| 1 / 1 / 1 / 4        | 0.0965  | 0.0717    |
| 2 / 1 / 1 / 1        | 0.0467  | 0.0762    |
| 4 / 1 / 1 / 1        | 0.0320  | 0.0885    |
| 1 / 3 / 0.5 / 1      | 0.0293  | 0.0240    |

The worst is 0.0965, just under twice nominal and therefore short of T3's stop
condition, and it takes a FOURFOLD overstatement at the ladder's most
influential rung to get there. It is recorded because the eps=0.5 entry of the
SE pattern is interpolated -- that in-task cell has never been run -- and an
interpolated weight is exactly the one most likely to be wrong. Gate D of the
pre-flight is what replaces the interpolation with a measurement.

PERMUTATION FLOORS, exact:

| cells | floor = 1/k! | perfect ordering | clears 0.05 | one adjacent swap | clears 0.05 |
|-------|--------------|------------------|-------------|-------------------|-------------|
| 4     | 0.041667     | 0.041667         | yes         | 0.166667          | NO          |
| 5     | 0.008333     | 0.008333         | yes         | 0.041667          | yes         |
| 6     | 0.001389     | 0.001389         | yes         | 0.008333          | yes         |

At four cells a single adjacent swap puts the companion at 0.167. That is the
knife-edge F.2d.2 refuses to make a conjunct, and the row it lives on is the row
F.6r moves off.

NEGATIVE CONTROLS, both directions: a null-slope synthetic gives slope +0.0222,
p = 0.213, refused; a planted monotone gives slope +0.5690, p = 0.0016, passed.

**Still deliberately not decided: renaming the primary axis.** "Exact
average-case identifiability" overclaims — identifiability is asymptotic, this
is finite-n order evidence at the panel's `n_rows`, and it moves with that
parameter (measured at eps=0.5: 0.0925 / 0.2180 / 0.4606 / 0.5922 at `n_rows`
5 / 10 / 20 / 40). It is exactly `I(o;D)/log O`, the normalized mutual
information, which has a standard name, an unbiased sample mean and a
literature. Every caption would read "at n_rows = 20". Held for the writing
pass; it changes no number.

---

## F.3 — E.9.1 formalized

D9's signed statistic — mean log-likelihood against quadrature entropy — has
essentially no power against the defect it was written to catch. Per-coordinate,
per-row and per-dataset mixing induce IDENTICAL per-coordinate marginals, and
`residual_logpdf(...).sum(-1)` depends only on marginals. Measured at eps=0.25,
N = 4e5: -5.191943 versus -5.190646, against an MC SE of 0.0022.

Under D9's own strengthening clause the statistic is replaced by a joint
model-selection test: three normalized candidate densities that share those
marginals and differ only in granularity are scored over blocks of rows, and the
declared elementwise law must win by z > 6. The quadrature check is retained as
a normalization guard. Pre-fix the test fails 9 of 19, at exactly the three
generation paths crossed with the three mixed eps, with z from -9.5 to -25.9,
and that failure is committed as evidence. Post-fix it passes 19 of 19.

This moves the erratum from a test docstring into the preregistration proper.

---

## F.4 — D2 outcome, and one confounded rung

Post-fix Q1 `avg_width_mean`, against the pre-fix values recorded in E.2:

| eps  | pre-fix | post-fix Mac highs | post-fix Mac ipm | post-fix WashU highs | post-fix WashU ipm |
|------|---------|--------------------|------------------|----------------------|--------------------|
| 0.10 | 0.4510  | 0.437820           | 0.437814         | 0.437819             | 0.437814           |
| 0.25 | 0.3232  | 0.316973           | 0.316971         | 0.316973             | 0.316971           |
| 0.50 | 0.2073  | (see below)        | 0.260840         | (see below)          | 0.260840           |

The dial survives D1. The ladder is strictly monotone across the full range
(1.0000 / 0.4378 / 0.3170 / 0.2608 / 0.1255) and `true_in_all_frac` is 1.0 in
every cell at every evidence regime, so the exact set still covers the truth.

**Environment invariance** holds and was measured rather than assumed: two
different HiGHS builds (Mac scipy 1.12.0 / numpy 1.26.4 against WashU scipy
1.15.3 / numpy 2.2.6) agree to better than 5e-7 relative on the same cell and
solver, worst pair 4.4e-7 and best 9.1e-9.

**Solver inhomogeneity** — the unresolved finding in the committed audit — is
closed. `highs` and `highs-ipm` differ by ~1e-5 relative and differ by the SAME
amount in both environments, so the gap is a reproducible property of the two LP
paths rather than run-to-run noise, four orders of magnitude below the 0.12
spacing between rungs.

**The eps=0.5 rung was suspected confounded; the 2x2 clears it.** The committed
audit records that the published pre-fix `0.2073` came from one of two IPM runs,
that the config block does not record the `crossover` option, and that the
stored runtime (689.4 s) matches the run made BEFORE the code gained
`crossover: off`. That made the +0.0535 move potentially a mixture of D1 and a
code-version change, on the one rung whose movement flattens the top of the
ladder. Rather than assume the two effects were additive, the full 2x2 was run:

| SEM at eps=0.5 | crossover current | crossover legacy |
|----------------|-------------------|------------------|
| post-fix (D1)  | 0.260840          | 0.260840         |
| pre-fix        | 0.207325          | 0.207325         |

The crossover option has NO effect: -2.3e-8 on the post-fix row and exactly 0 on
the pre-fix row, both at solver tolerance. The interaction is +2.3e-8. So the
D1 effect is **+0.0535154**, clean and unconfounded, and `prefix/legacy`
reproduces the published 0.207325, which confirms the pre-fix number was a valid
measurement of the pre-fix instrument rather than an artifact of an
undocumented solver configuration.

The suspicion was worth testing and it did not survive contact. The rung is
citable as a clean D1 effect. The paper still cites only the post-fix
current-solver ladder; the 2x2 goes in the errata appendix as the evidence that
the attribution was checked rather than assumed.

**eps=0.5 x `highs` is an errata item, not a number.** It ran 3.9 h on the Mac
and past 21 h of a 24 h wall on the cluster without producing output, while
`highs-ipm` cleared the same cell in 20.8 min on the Mac and 15.5 min on WashU.

A conditioning explanation was floated and does NOT survive its own evidence, so
it is recorded here as refuted rather than quietly dropped. Q1 `cond_mean` across
the rungs: eps=0 3.535e11, 0.1 2.280e10, 0.25 2.005e10, 0.5 4.078e11, 1.0
3.714e11. The eps=1.0 cell is within 10% of the degenerate one and solved under
default `highs` in 784 s. An earlier draft compared eps=0.5 against only the two
low rungs, which made the contrast look decisive by omitting the two comparable
ones. **No cause is claimed for the degeneracy.** The cell is not restarted with
a longer wall; the ladder uses IPM and the solver-path note goes in the appendix.

**Axis reordering check (F.2b):** both axes rank all five cells identically
(deficit 0.0000 / 0.0355 / 0.1554 / 0.4152 / 0.7664 against 1-width 0 / 0.5622 /
0.6830 / 0.7392 / 0.8745), so the cross-axis rank correlation is 1.000 and the
re-axing cannot change the E.4 verdict on this substrate. Measured, not argued.

The deficit figures here are the LIVE `deficit_w2` set at 500 contexts per half.
An earlier draft of this paragraph quoted the retired 60-context set
(0.0354 / 0.1767 / 0.4588 / 0.7626), which E.4 superseded. The rank conclusion is
unchanged, since both series are strictly increasing; the digits were not.

---

## F.5 — n_rows disclosure

A second complete identifiability measurement set, `nrows10_ident_eps*.json`,
sits in the same `raw/` directory and is invisible to the report loader, whose
glob `ident_eps*.json` does not match it. Same seed, same `n_contexts`, same
tolerance; only `n_rows` differs. Its Q1 widths are higher everywhere and the
gap grows with eps. BOTH SIDES of that comparison are PRE-FIX, which makes it a
valid n_rows-only contrast but means the right-hand column is the ladder F.4
retires forty lines above: nrows10 0.5629 / 0.4399 / 0.3709 against the
pre-fix headline set, 1.25x rising to 1.79x. Against the POST-fix ladder the
ratios are 1.29 / 1.39 / 1.42. Either label both sides pre-fix, as done here, or
regenerate the nrows10 set post-fix before citing it.

The ident config block records `d, K, eps, n_contexts, tol, seed, solver` and
does NOT record `n_rows`. Track B's panel block does record it. So the ladder's
absolute widths depend on a panel parameter the ident artifacts never captured,
most strongly at the rung the stretch claim leans on.

Binding clauses:

1. `n_rows` is recorded in every ident config block from this point.
2. **The headline ident `n_rows` is DEFINED as Track B's panel `n_rows`, not
   merely declared to match it.** The two must be the same number. If they are
   not, the width appendix's x and the model-side y sit at different information
   levels and the appendix comparison is invalid. F.2b makes this automatic for
   the PRIMARY axis, since the deficit is evaluated on the scoring panel itself;
   the clause is stated because the width appendix still needs it.
3. The `nrows10` set and this sensitivity table go in the instrument-errata
   appendix under D10. It is disclosed, not quietly dropped.

---

## F.6 — Deferral register

Recorded BEFORE the reviews, so it reads as scope discipline rather than an
omission discovered later.

**A finer eps grid at 0.35 and 0.75 was DEFERRED.** Each identifiability cell now
costs about 15 minutes, so the grid is cheap on the exact side. Each additional
eps costs a full retrain fleet on the model side, which is the binding cost, and
the calendar to Sep 25 does not carry it. Held as rebuttal ammunition: if a
reviewer asks whether the dial's shape is an artifact of five coarse points, the
answer is a pre-planned pair of intermediate rungs, not a new idea.

### F.6r — the register re-opens for exactly one item (PRE-DATA, 2026-08-23)

**eps = 0.35 and 0.75 join the ident ladder and W2's fleet.** The deferral above
is reversed for these two rungs and for nothing else. The reason is that the
premise it was recorded under changed: F.2c's eps=0 exclusion altered the gate's
geometry AFTER the deferral was written, and the arithmetic that follows from
that exclusion is what the two rungs now buy.

- six fidelity cells move the exact permutation floor to `1/720`, with an
  adjacent swap at about `6/720`, against `1/24 = 0.0417` at four;
- the slope test gains degrees of freedom: `df = k - 2` goes from 2 to 4;
- one of the two sits INSIDE the region P3 certified as genuinely flattened, so
  the flattening is measured with an interior point rather than inferred from
  the ladder's endpoints.

This is the only scope expansion this cycle. It is recorded before any model-side
number exists, which is what separates it from widening a grid after seeing where
a curve went. The register otherwise stands as written.

**Both cells ran, and the ladder holds.** IPM, `n_rows` 20, seed 990200000,
n_contexts 20, tol 1e-3 -- configuration-identical to the five that preceded
them -- on WashU, tagged live, `true_in_all_frac` 1.0 in both:

| eps  | Q1 avg_width | gap to previous |
|------|--------------|-----------------|
| 0.00 | 1.000000     |                 |
| 0.10 | 0.437814     | 0.562186        |
| 0.25 | 0.316971     | 0.120844        |
| 0.35 | **0.279148** | 0.037823        |
| 0.50 | 0.260840     | 0.018308        |
| 0.75 | **0.156605** | 0.104235        |
| 1.00 | 0.125548     | 0.031057        |

Strictly monotone decreasing across all seven, so T4's stop condition did not
fire. All seven rungs are now measured under ONE solver (`highs-ipm`) at
`n_rows = 20`, each carrying an F.8 status tag, after the five cells that
predated F.8 were re-run on 2026-08-24. They reproduced their retired values to
nine significant figures, so nothing here moved on remeasurement; what changed is
that the ladder is solver-homogeneous and legible to the loader. The eps=1.00
rung is the one exception worth naming: it had been solved under default `highs`
and reads 0.125548 under IPM against 0.125550 before, a 1.5e-6 solver difference
that moves no digit this document cites. The eps=0.35 rung lands INSIDE the region P3 certified as genuinely
flattened and shows the flattening is smooth rather than a step: the top of the
ladder narrows through 0.0378 and then 0.0183 rather than jumping.

---

## F.7 — Provenance discipline

Unchanged from E. Verified / red-teamed / unverified ledger. Numbers come from
result JSONs on disk, never from report or memory prose. Nothing is silently
repaired; every instrument defect found is disclosed in the errata appendix.
Stop conditions mean stop and report.

**Retired-ARTIFACT status tags (F.8).** The tripwire below stops a dead NUMBER
reaching a document. This stops a dead FILE reaching a computation, which is the
same disease and has now cost this campaign three times:

- the power analysis in an earlier draft of F.2 was computed on
  `trackb_eps*.json`, the mixed-target panel E.3 retired, and reported as though
  it described the in-task gate;
- `cells_from_raw` globbed `ident_eps*.json` in whatever directory it was handed,
  and `raw/` still holds the widths E.2 retired under exactly those filenames;
- `nrows10_ident_eps*.json` is the same disease with the opposite sign: a live
  artifact no loader could see, because the glob happened not to match it.

Every result JSON carries `artifact: {estimand, status, amendment, schema}` with
status in {live, retired, diagnostic}. Writers stamp; readers call
`require_live(doc, estimand, where)` and refuse anything else. **An untagged
artifact is UNKNOWN, not live**, so everything written before this amendment
fails closed — which is correct, since those are exactly the files that caused
the incidents. The mixed-target and in-task regret series are named as separate
estimands rather than left to a filename convention, because they are the pair
already confused once.

**Retired-constant tripwire (P2).** `tests/test_retired_constants.py` fails when
any retired constant appears outside an explicit allow-list of files that
legitimately record history: the old ladder, the retired in-task regret values,
the old classifier series, and the misleading across-seed SE claim. Verified in
both directions — clean on the tree, and it fires when a retired number is
planted in a new file.

**The invisible-artifact failure recurred, one hour after this paragraph was
written.** It is recorded because a discipline that only catches the incidents
that predate it is a story rather than a mechanism. A producer tagged its output
with a hand-rolled `f"{eps:g}".replace(".", "p")`, which yields `"0"` and `"1"`
at the endpoints; `_TAG_RE` requires `\d+p\d+`; `cells_from_raw` hit
`if m is None: continue` and skipped both IN SILENCE. The eps=0 gate (b) anchor
and the eps=1 top rung dropped out of a seven-cell ladder and the loader
returned five cells without an error. A ladder short two rungs looks exactly like
a ladder.

Two changes, both binding. Any file matching a result glob but not the tag
pattern now RAISES rather than being skipped -- for ident, deficit and fidelity
alike. And `eps_tag()` is exported as the canonical inverse of `_eps_of`, with a
round-trip assertion, so producers stop rolling their own.
`tests/test_f5_n_rows.py` carries all three as regressions, including one that
loads the real seven-cell ladder off disk rather than a fixture.

It earned itself immediately. Two retired-number regressions had already reached
this document (F.2's SE ratios and F.5's unlabelled baseline, both corrected
above), and on first run the tripwire caught a third in a comment written the
same hour. The allow-list is what makes the next one have to be added
deliberately.

---

## F.9 — Signatures taken before the lock (2026-08-26)

F.2c-tol's rule (1) defines the floor readout `r = 0 - structural_js` on the
projection branch but does not name the query panel that supplies it, and rule
(4) does not name the solver, the seed, or a venue reachable today. Those gaps
were closed by measurement first and signature second. The clause-4 measurement
itself is recorded in F.9.2; it is what the signatures in F.9.1 rest on, and it
was taken on exact-posterior and dose-0 data only, with **no `fidelity_*`
artifact existing anywhere** — verified by search on the working tree and on the
cluster the same day — so nothing here could be fitted to an outcome.

### F.9.1 — U1, U3, U4, U5 (SIGNED, Bo, 2026-08-26)

**U1, the readout panel: Q2.** Q1 was signed first, on the morning of
2026-08-26, for consistency with the F.6r ladder, and is **WITHDRAWN** on the
clause-4 evidence. The Q1 projection readout is **refused as an axis anywhere**,
including as an appendix or robustness quantity, so the defect cannot re-enter
by another door. The withdrawal is not a matter of preference:

- Q1's joint operator is **rank 39-40 of 48** (condition 2.3e10 to 4.1e11)
  against Q2's **48 of 48** (condition 3.0e3 to 5.3e4), so Q1's projection
  attains zero residual on a manifold of at least nine dimensions.
- Pushing the EXACT Bayes posterior through the readout — the ceiling curve, which
  a panel that can represent it must return at exactly zero — gives **-0.070823,
  -0.073274, -0.078502, -0.083686, -0.090199, -0.108085** on Q1 across
  eps = 0.10 to 1.00, at projection residual exactly 0.000000. On Q2 it gives
  **-0.000000 to -0.000001**.
- Q1's ceiling carries standard deviation up to **0.216**, where a correct
  ceiling has standard deviation zero.
- Q1's ceiling is **not reproducible across environments**: between scipy 1.12.0
  and scipy 1.15.3 the floor curves agree to 1.1e-16 on both panels and Q2's
  ceiling to 1e-10, while Q1's ceiling differs by 1.6e-4 to 1.6e-3. Which point
  of the manifold the LP returns depends on the HiGHS build.
- **Every** projection LP failure observed — three of 12,000 in one venue, four
  of 12,000 in the other, at the same context indices — is on ceiling/Q1. None on
  any floor curve, none on Q2, none under highs-ipm.

The consistency argument that motivated Q1 does not transfer: F.6r's
"Q1 avg_width" is the IDENTIFIED-SET width, and rule (2) removes the
identified-set branch from the floor and ceiling curves entirely.

**U3, the solver: `highs`, the dual simplex.** Unchanged, and deliberately so.
Every tomography number on disk was computed under it. Its seven failures were a
symptom of Q1's degeneracy, not of the solver — conditional on U1 = Q2 it solved
12,000 of 12,000 in both venues, agreeing with highs-ipm to 1e-10. Changing
solvers would trade an evidenced setting for an unevidenced one in order to
repair a panel that is being removed.

**U4, the untrained seed: 0.** The fleet's first seed and the only one with a
persisted dose-0 anywhere. Both identity checks pass at every eps: the reloaded
`ck0` state equals a freshly seeded `PFN(cfg)` under `torch.manual_seed(1000*s+7)`,
and the six per-eps `ck0` states are tensor-identical to one another, since
initialization precedes any data draw.

**U5, the venue: WashU, substituting for the Mac.** Rule (4) says "one Mac
measurement". wenmacbook-air is reachable on the tailnet but refuses SSH, so the
substitution is signed rather than the clause silently violated. It is
numerically free: on every curve that is well posed, the two environments agree
to 1.1e-16 (floors, both panels) and 1e-10 (Q2 ceiling). The record gains no new
environment, WashU being the other one F.4's invariance table already
characterizes.

### F.9.2 — The clause-4 measurement (rule (4) discharged)

Six surviving cells, eps in {0.1, 0.25, 0.35, 0.5, 0.75, 1.0} — eps = 0 having
already exited the fidelity series by construction. 500 half-B contexts per cell
at the fleet's panel size, seed 0, floor and ceiling, Q1 and Q2, solver `highs`.

- LIVE: `campaigns/corrected_20260812/raw/postF/floor_calib/floor_calib_eps*.json`
  (WashU, Slurm job 226542, scipy 1.15.3 / numpy 2.2.6).
- DIAGNOSTIC venue comparison:
  `campaigns/corrected_20260812/raw/postF/floor_calib_bocomputer/` (scipy 1.12.0
  / numpy 1.26.4, the versions F.4's invariance table records for the Mac).
- Producer: `scripts/f2c_clause4_floor.py`. It calls the projection helper
  directly and recomputes the structural statistic verbatim, so **no
  identified-set LP runs on the floor or the ceiling**, per rule (2). Every
  per-context row carries the projection residual, a null nominal tol, a null
  tol_eff and `id_lp_ran: false`, per rule (3). LP failures are recorded per
  context with their message and excluded from summaries rather than substituted.

**What rule (4) was owed for is discharged, and favourably.** The concern on
record was that the floor's per-context projection variance had never been
observed and could leave the F.2d.1 slope underpowered. On Q2 the floor's
per-context standard deviation is **0.0605, 0.1187, 0.1521, 0.1825, 0.2144,
0.2387** across the ladder — large, and growing with eps, so the concern was well
founded as stated. At the fleet's panel size the mean's standard error is
**0.00270 to 0.01067** against a gain of **0.4592 to 0.5558**, i.e. gain over SE
of **169.9, 90.4, 71.9, 62.2, 56.7, 52.1**. The slope is not underpowered.

**No cell exits on gain.** Every measured ceiling-minus-floor gain is 0.4592 to
0.5558 on Q2 (0.3848 to 0.4428 on the withdrawn Q1) against
`CALIB_MIN_GAIN = 1e-6`. Survivorship reports six of six.

### F.9.3 — F.2d.4's conjunction implemented, then re-validated against the implementation

F.2d.4 registers that "gate_passes in descriptive mode reads the conjunction, and
both must pass". `corrected_verdict.py` did not do that: it set descriptive-mode
`gate_passes` from the extreme-cell contrast alone and discarded the slope. The
statistic the clause names was never implemented, and a code-state block that
digests that file while the clause describes it falsely records less than it
claims. Per the ordering constraint — any change to what decides a gate goes into
the verdict code BEFORE the lock, never after — the code was conformed to the
clause, on Bo's authorization of 2026-08-26, before any digest was taken. Both
conjuncts already existed and both already matched the clause verbatim; only the
line combining them read one of the two.

Re-validated by `scripts/f2d_gate_validation.py --trials 40000 --seed 20260826`,
which reads `gate_passes` off the real `association()` rather than a standalone
reimplementation — artifact
`campaigns/corrected_20260812/raw/postF/f2d_gate_validation_conjunction_20260826.json`:

| cell dispersion sd | descriptive slope | contrast alone | THE GATE | nominal |
|---|---|---|---|---|
| 0.00 | 0.04972 | 0.02395 | **0.00692** | 0.05 |
| 0.02 | 0.04955 | 0.07540 | **0.01698** | 0.05 |
| 0.05 | 0.04880 | 0.21575 | **0.03667** | 0.05 |
| 0.10 | 0.05155 | 0.33822 | **0.04718** | 0.05 |

Worst null gate false-positive rate **0.04718**; T3's stop condition is twice
nominal and **does not fire**. Power at beta = 0.1 / 0.2 / 0.3 / 0.5 is
**0.1681 / 0.4644 / 0.6536 / 0.8812**; the clause's self-voiding floor requires at
least 0.15 at beta = 0.2 with zero added dispersion and measures **0.4644**, so
the fallback stands.

Two things are disclosed rather than smoothed over. The implemented contrast uses
a two-sided 95% CI (`CONTRAST_Z = 1.96`) where `f2d4_option_sim.py` characterized
a one-sided alpha, so **the gate that will run is stricter than the one the
clause's recorded numbers describe** — the composite sits below nominal at every
row rather than at it. And the clause's own diagnosis reproduces independently
here: the contrast alone runs 0.02395 to 0.33822 across dispersion, against the
0.33525 F.2d.4 records at sd 0.10, while the slope holds 0.04880 to 0.05155.

### F.9.4 — What is NOT claimed here

No trained-model fidelity readout exists at the time of this lock. The only
trained-model tomography numbers ever taken are the 100-step pre-flight timing
smoke on one context and the DIAGNOSTIC-stamped 1000-step W2 smoke; neither is a
y value and neither is read by anything. W2 fires only after this lock, per the
written critical path, which Bo re-affirmed on 2026-08-26 when the option to
overlap training with the lock was offered and declined.
