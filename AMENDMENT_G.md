# Amendment G — confirmatory registration of the order-shortfall mechanism

**STATUS: DRAFT, NOT LOCKED. Written 2026-08-29 by the session that formed the
hypotheses it registers; amended 2026-08-30. It must be adjudicated by a cold
session (one that has seen only this draft, the code it names, and CLAUDE.md)
and counter-signed before `scripts/lock_amendment_g.py` fills G.0 and takes the
digest. Until then it binds nothing.** The file lives under `.claude/` as
`DRAFT_AMENDMENT_G.md` because `AMENDMENT*` is write-guarded; it is moved to
`AMENDMENT_G.md` at signing.

Supersedes nothing. Amendment F remains locked at sha256
`3d0e0f0af86c99601be8d99b34f778073fdd5fee9d8ef2b9a547cfe0eb8d8f98`; G re-measures
F's readout on new models (gate G4) and does not re-adjudicate F's verdict.

**Amendment history of this draft.** Rounds 1–3 of cold adjudication
(`.claude/G_ADJUDICATION.md`) closed at READY on 2026-08-29. Two things happened
after that and before any lock, both recorded in `WORKLOG.md`:

1. A delegated max-effort design review (`.claude/G_DESIGN_DECISION.md`,
   2026-08-30) ruled that **capacity must be confirmatory, not exploratory**. The
   exploratory capacity contrast was tuned asymmetrically — `large` was swept
   over three learning rates and its best arm taken while `base` and `small` ran
   at 1e-3 only — so "best-of-three large vs single-LR base" is not a capacity
   contrast. It also re-derived the motivating claim from the `.npz` and found it
   absent: the eps = 1 paired 500k→2M change in b is **−0.013 ± 0.027**, an
   unresolved null, not the "plateau" or "hardness premium" this amendment was
   originally drafted to confirm. G1–G4 are unchanged; G5a, G5b and G6 are new,
   and a symmetric learning-rate selection rule is registered in G.1.
2. A cold pre-flight of the execution path
   (`.claude/G_PREFLIGHT_20260830.md`, 30 confirmed defects, 16 blocking) found
   that the previously pinned commands could not run as written: two
   contradictory base conventions for the same paths, a G4 invocation that
   silently rebuilt the EXPLORATION panel, and pinned "commands" that were bare
   variable assignments with no executable. Every pinned invocation in G.2 is now
   a driver script under `cluster/`, digested in G.0.

---

## G.0 — Code state at lock

- repo HEAD at lock: `3f49c5120a534554274bc12647d29c132ca7787e`
- `scripts/mech_predgain.py`                          sha256 `c4733286822cdd751519bc158d67ab91f194b49930af590b29aa8d0da83b9c9c`
- `scripts/mech_gates.py`                             sha256 `eeb5a02362525119a876ce742088458c75ceb7a214fe63f29f74418429040346`
- `scripts/mech_phase1.py`                            sha256 `ee02fc990935889bd64be3766fa85ea27d37e72e4c12ecc33a21f406e7965151`
- `scripts/mech_interventions.py`                     sha256 `d9e3e75d64423675026c34afbbee6a62be608be95c9a1a51621e26d5e109aae0`
- `scripts/mech_train.py`                             sha256 `138ad6d09395feabd9d2d895534077bbb5b75ff78026f08a120cb83c47fe53f7`
- `cluster/mech_train.sbatch`                         sha256 `2cb185cd7e8942f01ffd7b5a841b4381a0a6201c6d67552012ed9d79c63b1500`
- `cluster/mech_score.sbatch`                         sha256 `31fb38649a9e0faeabd20058f185aaa2c6d5b722e9df612a220c8308a09017e0`
- `cluster/mech_readout_g.sbatch`                     sha256 `971351d3ab1c3163b341a10a03b41f1ea87d0cf387402ae64b0bbab6cf923aba`
- `cluster/fire_confirm_g.sh`                         sha256 `f235a31984ea54e1ef79c2a3ab4583b009dadda64c5ac4e5a6fb114e1fc207b3`
- `cluster/score_confirm_g.sh`                        sha256 `afe7402a672eb34057dbbeeb9923b0824beab6e201928e0a06c7b071b2b77274`
- `scripts/lock_amendment_g.py`                       sha256 `6b0967fae1aa19b6523dad25e1709bc97048d171ebe7a3cc8be9e28e4b45f9d7`
- `src/pfn_dag_verify/corrected_oracle.py`            sha256 `0f1f8bf8377e20302c216dfb137c57b2ea45f33a7326d9eb3180f048d5393e3c`
- `src/pfn_dag_verify/corrected_models.py`            sha256 `d29e0325dff89bf50484bf73dda8b9a4139c2bad65dce07524c51baa44e5b9d9`
- `src/pfn_dag_verify/corrected_sem.py`               sha256 `424ed0a34e0683451c6b80773d4400f70d36605cd66cb7bec1ca16811b194668`
- `src/pfn_dag_verify/corrected_world.py`             sha256 `294a6fc6b6ab3e3f1efde3f260f3c54d6ed592ac77388c32a5df76069444f5e4`
- `src/pfn_dag_verify/split_panel.py`                 sha256 `b0232ba8843d1e195e36ad9adb1be53cde7578e8f266a64056eb55d1b098b7dc`
- `src/pfn_dag_verify/corrected_tomography.py`        sha256 `36c90103d16efd2a59167a69e2e5c1ca0a174ce469b29fb5c1b0d9930371339e`
- `src/pfn_dag_verify/corrected_identifiability.py`   sha256 `201f92170ea632c301966b09d5526795402a6d754b6b15dd307530e27fe7489d`
- `src/pfn_dag_verify/corrected_trackb.py`            sha256 `b0b0dd8999e1fb14fb28226975a23133cc91307b2419c4099d240a52705ab61d`
- `src/pfn_dag_verify/corrected_verdict.py`           sha256 `c47dd3be482342ff3d02ea8ff0ac3c86f04f7faca55feebbe76fd9074aa0c14d`
- `src/pfn_dag_verify/corrected_deficit_run.py`       sha256 `3340be106a797c46a91e0b66f4519b98a43712b93e0e42fbb09032b35f7cdedb`
- `src/pfn_dag_verify/corrected_identifiability_run.py` sha256 `293a1f87a8493b40f4aecaa8f02d8eadd43d7fefcb1cc6c2640f8bbb4907ad98`
- `src/pfn_dag_verify/pilot_shared.py`                  sha256 `a31508273748212aaaa156252e667a6e2430ae999b744685025c1aeb5665d1c5`
- `src/pfn_dag_verify/artifact_status.py`               sha256 `18ec46c98d907cd39249111d717d8e64f2100fbacfdc62105f106192743ed200`
- `src/pfn_dag_verify/w2_run.py`                      sha256 `160d892215f1ef1f8f8b29f96037b0378482602febaba5dcba21dee2237a5167`

Six `src/` entries were added to this list during the 2026-08-30 amendment.
`corrected_trackb.py` defines `SCALES` — what `base` and `large` ARE;
`corrected_verdict.py` defines `eps_tag`, every artifact filename and lookup key;
`corrected_deficit_run.py` and `corrected_identifiability_run.py` carry the
`PANEL_SEED` / `N_PER_HALF` / `HEADLINE_N_ROWS` defaults; and `pilot_shared.py`
supplies `N_BINS` and `production_quadrature` — the bin edges and quadrature with
which *every* exact score here is computed (`S(q_full)`, `S(q_abl)`, `G_i`,
`regret_i`, `w_exact`, and the LP operator). `artifact_status.py` completes the
import closure of the four G.0 modules that import it. A change in any of them
moves a registered number silently, so G.0 digests them.

**Code identity on the cluster.** The fleet executes the WashU mirror's copy
while this block digests bo-computer's. `cluster/fire_confirm_g.sh` rsyncs every
file listed above to the mirror and then verifies each digest there with
`sha256sum -c`, refusing to submit on any mismatch. Code ships by rsync, never by
`git pull` on the mirror.

Recipe identity. `scripts/mech_train.py` calls
`corrected_models.train_pfn(world, steps, seed, cfg, n_ctx=n_rows, n_query=7,
peak_lr, ckpt_steps, outdir, tag_prefix)` — the call `w2_run.run` makes, with
`n_query=7`, batch 32, warmup 200 and the cosine schedule over the run's own
`steps` inherited unchanged from `train_pfn`'s defaults. It differs from
`w2_run` only in exposing `--scale`, `--n-rows`, `--peak-lr` and in performing
no in-run evaluation. `peak_lr` is tagged into the checkpoint filename only when
it differs from the registered 1e-3 (`mech_train.py:57`), so the registered arm's
checkpoints keep their `base_s{seed}` names.

## G.1 — Estimand, and the learning-rate selection rule

For a cell (arch, lr, dose, eps, seed) and the registered panel (G.2), score
every context i with `scripts/mech_predgain.py` (eight fresh query rows per
context from the context's generating latent, in-task target d−1 only, exact
expected log score S(q) = sum_b p*(b) log q(b)):

- regret_i = mean over the eight queries of S(q_full) − S(q_model);
- G_i      = mean over the eight queries of S(q_full) − S(q_abl), the
  order-information gain, where q_abl keeps the exact atom posterior p(k|o,D)
  and flattens the order marginal to the prior.

The **primary estimand** is the OLS fit regret_i = a + b·G_i over the 1000
half-B contexts: b is the order-specific shortfall fraction, a the generic floor.
The PRIMARY unit of every gate is the per-context regret averaged over the
three model seeds (Amendment F decision B's convention); per-seed values are
reported and a gate additionally requires every seed's contrast to have the
registered sign (consistency), without a per-seed z threshold. This form was
chosen on 2026-08-29 after a power check on the EXPLORATORY data (seeds 0–2):
the per-seed alpha = 0.01 form failed cells with effects of the expected size.

**Linearity of regret in G is an assumption of this estimand**, and a known
false one in detail: on the exploratory artifacts the binned ratio regret/G falls
monotonically with G (eps 0.75, 500k: 0.59 → 0.44 → 0.36 across quintiles), so b
is a G-weighted average slope, not a constant. Two consequences are registered
here rather than discovered later. (i) Every gate except G1 is a WITHIN-panel
contrast, holding each G_i fixed, so it contrasts the model's regret profile
alone. (ii) G1 compares worlds whose G distributions differ several-fold; its
licensed sentence is therefore the fact (b is smaller at eps = 1), never a
mechanism, and the matched-G binned comparison is reported beside it. The fit's
correlation r is reported for every cell.

eta_order = 1 − mean(regret)/mean(G) is reported as a summary and takes part in
no gate. It is not a measure of order knowledge on its own: eta_order =
1 − a/G − b, so it goes negative wherever G is small, which is a statement about
G rather than about the model.

Standard errors are context-level bootstrap (N_BOOT = 2000; each test's RNG is
seeded from 20260830 and the test's label, and the label excludes the directory,
so no result depends on evaluation order or on path spelling), paired wherever
the two cells share contexts (G2: two doses on one panel; G3: a panel and its
extension; G5a/G5b/G6: two architectures on one panel; the evaluator refuses a
pairing whose artifacts do not carry identical latents k, o).

Query rows: context i draws its eight queries with seed 440000000 +
1000·round(1000·eps) + i (`mech_predgain.Q_SEED_ROOT`); the n40 extension rows
with seed 560000000 + round(1000·eps) as `mech_interventions.py:133` computes it
(`N40_SEED_ROOT + int(round(eps * 1000))`; the code is the definition).

### Learning-rate selection (registered before any of it is scored)

The capacity gates compare architectures, so each architecture is given its own
best learning rate by a mechanical rule applied symmetrically to all of them.

- **Grid:** peak_lr in {3e-3, 1e-3, 3e-4, 1e-4}, seeds 3, 4, 5, all else the
  registered recipe.
- **Selection panel:** a THIRD panel, neither the exploration panel nor the gate
  panel — `MECH_PANEL_SEED=770000201`, `MECH_SPLIT_SEED=880000201`,
  `MECH_N_PER_HALF=500`, n_rows 20, m_q 8, `MECH_CONFIRM=1`, `MECH_LADDER=0`.
- **Selection statistic:** the mean over contexts of the seed-averaged regret_i
  — the model's own objective, in exact expectation. Not `final_loss`, which is
  the mean of the last 200 batch losses (SE ≈ 0.007) against LR differences of
  interest of 0.002–0.01.
- **LR\*(arch, dose, eps) = argmin** of that statistic. Ties (|Δ| < 1e-6) go to
  1e-3, then to the larger LR.
- **DIVERGED:** a cell that was trained, was scored, and whose selection
  statistic is not finite is excluded from selection and reported as DIVERGED.
  If every LR of an arm is DIVERGED, the gate that needs it is NOT EVALUABLE.
- **A missing or non-conforming selection cell is a REFUSAL, not a DIVERGED
  arm.** If the artifact is absent, or fails any panel / scale / seeds /
  `MECH_CONFIRM` check, `mech_gates.py` exits with an error rather than taking
  the argmin over whatever finished. A scoring job that died on its wall limit
  must not be able to move LR\* with exit 0; an execution error is a finding to
  report, not a gap to fill.
- The full selection table — all four regrets per (arch, dose, eps) — is part of
  the verdict JSON. **No gate reads a selection-panel artifact, and no gate reads
  the gate-panel artifact of an unselected LR.** An unselected LR's gate-panel
  cell, if it was scored, is left on disk unread; the verdict records b, a, SE
  and r for the cells each capacity gate actually used, under that gate's part
  (`gates.G5*.parts[0].cells`), with the full selection table at top level
  (`selection_table`).

G1–G4 keep the registered recipe 1e-3 for their base arm. They are within-recipe
direction tests; mixing per-eps LR selections into them would compare across
recipes.

## G.2 — Panel, fleet, and the pinned invocations

**Every path below is REPO-ROOT relative.** `cluster/mech_score.sbatch` cd's to
`$REPO` and passes `NETS_DIR` / `OUT_DIR` / `IV_DIR_VAL` through unchanged, and
`scripts/mech_gates.py` joins no base directory to its arguments, so a value
carries the `campaigns/mech_20260827/` prefix itself. The drivers below are the
only supported way to run any of it; they are digested in G.0.

- **Gate panel:** `MECH_PANEL_SEED=770000101`, `MECH_SPLIT_SEED=880000101`
  (fresh; the exploration used 770000000 / 880000000), half B of **1000**
  contexts per world, n_rows 20. The G3 panel is NOT a second draw: each
  n_rows-20 context is extended by 20 fresh rows from its own generating latent
  (`mech_interventions.py build_n40`), so the n20/n40 contrast is paired by
  context and, through the per-context query seed, by query row.
- **Selection panel:** 770000201 / 880000201, 500 per half, n_rows 20 (G.1).
- Worlds: `make_world(k=8, d=3, eps)` for eps in {0.5, 0.75, 1.0}.
- Every scored artifact records panel_seed, split_seed, n_per_half, n_rows,
  sha256_b, `scale`, the checkpoint digests and `confirm`; `mech_gates.py`
  refuses any artifact whose panel fields or `scale` differ from the cell it is
  being read as, or that was not produced under `MECH_CONFIRM=1` (which also
  forbids overwriting or silently skipping a cell). The G4 readout artifacts
  carry the same provenance, and their checkpoint digests are matched against the
  registered fleet's, so the instrument gate cannot be run on other models.

### Cell convention

One directory per (arch, n_rows, lr, dose), named `{arch}[_n{rows}]_lr{lr}_d{dose}`:

| what | where |
|---|---|
| checkpoints | `campaigns/mech_20260827/confirm/{cell}/nets/eps{tag}/` |
| gate-panel scores | `campaigns/mech_20260827/predgain_confirm/{cell}/` |
| G3 n40 scores | `campaigns/mech_20260827/predgain_confirm_n40/{cell}/` |
| selection-panel scores | `campaigns/mech_20260827/predgain_select/{cell}/` |
| G4 readout | `campaigns/mech_20260827/confirm/phase1/` |
| G3 paired panel | `campaigns/mech_20260827/confirm/iv/` |

`CKPTS='0'` everywhere: **only the fully decayed endpoint of each run is ever
compared.** A 100k number comes from a 100k run, never from a 500k run's
mid-cosine `_ck100000` — on the exploratory data those two states differ by 40%
in b, and comparing checkpoints across runs with different cosine `total`
doubled an apparent exponent on 2026-08-28.

### The fleet

Seeds **3, 4, 5** throughout (never trained before), n_query 7, batch 32,
warmup 200, cosine over the run's own `steps`. Wave 1 is everything a gate needs
plus the two cheap reported rungs; wave 2 is reported-only.

| # | arch | peak LR | n_rows | dose | eps | role | jobs |
|---|---|---|---|---|---|---|---|
| 1 | base | 1e-3 | 20 | 100k | .5 .75 1.0 | **G1 G2** | 3 |
| 2 | base | 1e-3 | 20 | 500k | .5 .75 1.0 | **G1 G2 G3 G4**, grid arm | 3 |
| 3 | base | 1e-3 | 40 | 500k | .75 | **G3** | 1 |
| 4 | base | 3e-3, 3e-4, 1e-4 | 20 | 500k | .75 1.0 | **G5a G5b** grid | 6 |
| 5 | large | 3e-3, 1e-3, 3e-4, 1e-4 | 20 | 500k | .75 1.0 | **G5a G5b G6** grid | 8 |
| 6 | base | 1e-3 | 20 | 2M | .5 .75 1.0 | **G6** grid (.75), dose law | 3 |
| 7 | base | 3e-3, 3e-4, 1e-4 | 20 | 2M | .75 | **G6** grid | 3 |
| 8 | base | 1e-3 | 20 | 10k, 25k | .5 .75 1.0 | reported | 6 |
| 9 | base (3e-3, 3e-4, 1e-4), large (grid ×4) | | 20 | 500k | .5 | reported (wave 2) | 7 |
| 10 | small | grid ×4 | 20 | 500k | .5 .75 1.0 | reported (wave 2) | 12 |
| 11 | large | 3e-4, 1e-4 | 20 | 2M | .75 | reported (wave 2) | 2 |

Wave 1 is rows 1–8 (33 jobs); wave 2 is rows 9–11 (21 jobs). Row 9 does NOT
include base at 1e-3: row 2 already trains that cell at eps 0.5, and it carries
G1 at both doses, G2 at eps 0.5 and G4's checkpoint cross-check. `fire()` also
refuses outright if the endpoint directory a row would write already exists on
the mirror, so no run of either wave can overwrite a trained cell. Three seeds share
one GPU per job. `--time` is set per row by the driver (2M rows 24 h, large 500k
12 h, large 2M 48 h).

### Pinned invocations

Run from the repo root on bo-computer, in this order. Each refuses rather than
improvises; none takes a free-text argument.

```
bash cluster/fire_confirm_g.sh check      # verify lock + mirror code, submit nothing
bash cluster/fire_confirm_g.sh wave1      # rows 1-8
bash cluster/score_confirm_g.sh n40       # build the G3 paired panel
bash cluster/score_confirm_g.sh select    # SELECTION panel, the LR grid  (G.1)
bash cluster/score_confirm_g.sh gate      # GATE panel, every gate-bearing cell
bash cluster/score_confirm_g.sh readout   # G4 oracles + projections
bash cluster/fire_confirm_g.sh wave2      # rows 9-11, reported only
bash cluster/score_confirm_g.sh report    # GATE panel, the wave-2 reported cells
```

Then pull the scored artifacts to bo-computer (rsync, never `git pull` on the
mirror), and run the verdict there from the repo root. Only these four trees are
needed, and they are small — the `.npz` files, not the checkpoints:

```
R=washu1:/engrfs/project/class/zhao.b/LDM_interpretation/campaigns/mech_20260827
rsync -a "$R"/{predgain_confirm,predgain_confirm_n40,predgain_select} \
         campaigns/mech_20260827/
rsync -a "$R"/confirm/phase1 campaigns/mech_20260827/confirm/
```
The second line's destination is `confirm/`, not the campaign root: `rsync`
copies a source with no trailing slash by BASENAME, so pulling `confirm/phase1`
into `campaigns/mech_20260827/` would land it at `phase1/` — which already holds
the EXPLORATORY panel's `eps*.npz` under the same filenames, and `-a` would
overwrite them (that tree is git-excluded, so the loss would be unrecoverable)
while leaving `--readout confirm/phase1` missing.

Before the verdict, verify that the code about to compute it is the code this
amendment sealed — the verdict runs on bo-computer, where the mirror's
`sha256sum -c` (G.0) does not apply:

```
python3 -c 'import json;d=json.load(open("AMENDMENT_G.lock.json"))["digests"];print("".join(f"{h}  {p}\n" for p,h in d.items()))' \
  | sha256sum -c --quiet -
```

The verdict, run from the repo root with the pinned interpreter (this box's
`python` is 3.14; the repo pins 3.11):

```
PYTHONPATH=src .venv/bin/python scripts/mech_gates.py \
  --fleet   campaigns/mech_20260827/predgain_confirm \
  --grid    campaigns/mech_20260827/predgain_confirm \
  --select  campaigns/mech_20260827/predgain_select \
  --n40     campaigns/mech_20260827/predgain_confirm_n40/base_n40_lr0.001_d500000 \
  --readout campaigns/mech_20260827/confirm/phase1 \
  --seeds 3 4 5 --eps 0.5 0.75 1.0 --steps 100000 500000 \
  --alpha 0.01 --readout-step 500000 \
  --out campaigns/mech_20260827/confirm/AMENDMENT_G_VERDICT.json
```

The verdict JSON echoes argv, the registered panels, the LR grid and the full
selection table. `MECH_SCALE` for each scored cell is the checkpoint prefix
`mech_train.py` wrote — `base`, `base_n40`, `large` at the registered 1e-3, and
`{arch}_lr{lr:g}` otherwise — and the evaluator asserts it against the cell's
directory name, so a mis-pointed directory is an error rather than a wrong number.
The Amendment F readout code is byte-identical to its state at F's lock HEAD
`b05e32093d93545b3d54c4ae66a4755595348e44` (`AMENDMENT_F.lock.json` does not
itself digest these two modules, so the claim is stated against that commit and
is checkable with `git cat-file`). It is digested here in G.0 and is unmodified:
(`corrected_tomography._lp_project_or_ranges`,
solver `highs`, mode `project`; `corrected_identifiability.build_query_panels`,
panel Q2, seed 990300001 — the seed the F.2c LIVE fidelity artifacts record under
`readout.query_panel_seed`; `build_panel_operator` default `coarse=True`) is
unmodified and is digested in G.0.

## G.3 — Evaluability

The PRIMARY cell of a gate (seed-averaged regret) is evaluable iff its bootstrap
SE(b) <= 0.15; a single-part gate whose primary cell is not evaluable is
**NOT EVALUABLE** — neither pass nor fail. Per-seed cells enter only through the
sign of their contrast and carry no evaluability threshold. A G4 cell is
evaluable iff at least 80% of contexts have a solved projection for every seed;
the statistic is the per-context JS difference averaged over seeds on those
contexts. A capacity gate whose LR selection is DIVERGED for either architecture
(G.1) is NOT EVALUABLE.

**Conjunctions.** G1, G2 and G4 are conjunctions over doses / over eps. A
conjunctive gate is **FAIL if any part FAILs**, otherwise **NOT EVALUABLE if any
part is not evaluable**, otherwise **PASS**. A definitely false conjunct
forecloses PASS whatever the unmeasurable parts would have said; reporting such
a gate as merely unmeasurable would let a real failure hide behind a missing
cell.

## G.4 — Gates

Each gate is one-sided at alpha = 0.01 on the seed-averaged primary unit, with
per-seed sign agreement required; seeds are fixed effects and inference is
conditional on seeds 3, 4, 5. `scripts/mech_gates.py` computes them and its JSON
output is the verdict.

- **G1 (eps-direction)**  b(eps=1.0) − b(eps=0.5) < 0 at each of the two doses.
  Unpaired (different worlds). Base arm at the registered lr 1e-3.
- **G2 (dose-direction)** b(500k) − b(100k) < 0 at each eps. Paired bootstrap
  (same panel at both doses). Base arm at 1e-3.
- **G3 (signal share)**   b(n_rows=40) − b(n_rows=20) < 0 at eps = 0.75, 500k.
  Paired (the n40 panel extends the n20 panel context by context).
- **G4 (instrument)**     mean_i [JS(w_model_i, uniform) − JS(w_model_i, w_exact_i)] < 0
  at eps in {0.5, 0.75}, 500k, where w_model is the Amendment F projection
  readout of the model: the projected order posterior is nearer uniform than
  the truth.
- **G5a (capacity, eps = 0.75)** b(large @ LR\*_large, 500k) −
  b(base @ LR\*_base, 500k) < 0 at eps = 0.75, each architecture at its own
  selected learning rate (G.1). Paired.
- **G5b (capacity at the dose-flat rung)** the same statistic at eps = 1.0.
  Paired.
- **G6 (capacity vs dose)** b(large @ LR\*_large, 500k) −
  b(base @ LR\*_base(2M), 2M) < 0 at eps = 0.75, where LR\*_base(2M) is selected
  among the base 2M runs by the same rule. Paired.

## G.5 — What each gate licenses in the paper, and what a failure removes

| Gate | Sentence licensed | On FAIL | On NOT EVALUABLE |
|---|---|---|---|
| G1 | The order-specific shortfall is smaller where the order signal is larger (eps). | Sentence cut; exploratory numbers stay labelled exploratory. | Sentence cut; reported as unmeasured, not as absent. |
| G2 | The shortfall decreases with training between 1e5 and 5e5 steps. | Sentence cut. | Sentence cut; reported as unmeasured. |
| G3 | The shortfall falls with more context rows at fixed architecture (the order signal's share, via context length, is a lever). | Sentence cut. | Sentence cut; reported as unmeasured. |
| G4 | The registered F.2c readout places the model nearer uniform than the truth, on models it never saw. | Instrument section reduced to the out-of-task-query fact and the conditioning measurement. | Same reduction; the LP solve rate is reported. |
| G5a | At each architecture's own best learning rate, the 4-layer d=256 network (2,136,676 parameters, 7.65× the base's 279,140) has a smaller order-specific shortfall fraction than the base model at the same dose. | Every capacity sentence is cut; the seeds-0–2 result goes to an appendix labelled exploratory-and-confounded. | Same, and the selection table is reported with the DIVERGED cells named. |
| G5b | The residual shortfall the base model retains at eps = 1 is not a floor of the task: a larger network at the same dose lowers it. | That sentence is cut; G5a's stands alone and the eps = 1 residual is described as unexplained. | Same. |
| G6 | A fourfold increase in training steps closes less of the order-specific shortfall than a 7.65× larger network trained for a quarter of the steps. | The sentence is cut; "capacity helps" (G5a) may still be said, "beats dose" may not. | Same. |

**Not registered, reported as exploratory or descriptive only:** every seeds-0–2
result including the 2e6 endpoints and the tuned-LR capacity arms; b at every
grid LR other than the selected one; `small` at every LR; `large` at eps 0.5 and
at 2M; the base 10k/25k endpoints; the n40@500k vs n20@2M comparison; the
matched-G binned diagnostic; the per-context tracking slope; the interventions.
The exploratory values of b (100k / 500k: 0.66 / 0.44, 0.49 / 0.39, 0.29 / 0.18
at eps 0.5 / 0.75 / 1.0; n40 0.26) are shown beside the confirmatory ones
**without adjudication** — no "met / not met". No point prediction is
adjudicated: two are already on record as failed (the 500k point prediction of
2026-08-29, and "capacity is not the lever at this budget" of the same day).

## G.6 — Order of operations

1. Cold-session adjudication of this draft and of the files in G.0 at their
   digests (do they compute what G.1–G.5 say, and nothing else). Rounds 1–3
   (2026-08-29) closed READY on the pre-amendment text. Round 4
   (`.claude/G_ADJUDICATION_20260830.md`) adjudicated the amended text and the
   G.2 drivers in a stubbed rig with no cluster contact: **NOT READY**, 2
   blocking / 2 major / 8 minor, all fixed in the commit this lock digests. A
   round 5 was CUT OFF mid-run, having filed one blocking and one major — both
   defects introduced by the round-4 fixes — which are fixed
   (`.claude/G_ADJUDICATION_20260830_R5.md`). **Round 6
   (`.claude/G_ADJUDICATION_20260830_R6.md`) is the adjudication of the state
   this lock seals: READY TO LOCK, 0 blocking / 0 major / 10 minor.**

   The ten minors are accepted and carried, not silently dropped. None can
   produce a wrong number; each is fail-closed or unreachable from the pinned
   commands. For the record: the non-`src/` G.0 files are not in the rig's own
   `protected` list, so the pre-verdict `sha256sum -c` above is what guards the
   verifier on bo-computer (m1); a mid-wave refusal must be finished by hand
   rather than by re-running the pinned driver (m2, m8); the two remote `squeue`
   probes report `wc`'s exit status rather than `squeue`'s (m3); the evaluator
   takes doses, eps, seeds and alpha from argv, which the pinned command supplies
   and the verdict echoes (m4); the selection cell's and gate cell's checkpoint
   digests are not cross-asserted, and agree because `fire()` forbids retraining
   over an existing cell (m5); `mech_score.sbatch` does not itself require
   `NETS_DIR`/`OUT_DIR` under `CONFIRM_VAL=1`, which every pinned job sets (m6);
   `report` mode reuses `gate`'s job names, which affects `sacct` journaling only
   (m7). Fixing any of them would change a G.0 digest and require another cold
   round; they are registered here instead and may be addressed by errata.
2. The gate-drift line for `.claude/protected.sha256` on record (journalled
   2026-08-30: a self-referential scan in the rig's `drift.py`, 120/120 sealed
   digests re-verified, nothing registered moved); vendored artifact directories
   added to `.git/info/exclude`, journaled; clean tree.
3. Counter-sign, then **commit the counter-signed, still-tokenised text** — the
   lock script refuses a dirty tree, so the signature must be committed before it
   runs — then `scripts/lock_amendment_g.py` (fills G.0, writes
   `AMENDMENT_G.lock.json` and the `AMENDMENT_G.sha256` sidecar in Amendment F's
   format — and refuses a second run, which would freeze G.0 at the first lock
   while the sidecars moved); commit the filled `AMENDMENT_G.md` + both sidecars.
4. `bash cluster/fire_confirm_g.sh check`, then `wave1`. The driver rsyncs and
   then verifies the G.0 code on the mirror before it submits anything. Journal
   the job IDs.
5. Score in the order G.2 pins: `n40`, `select`, `gate`, `readout`. The selection
   table must exist before the verdict is run.
6. `scripts/mech_gates.py` → `confirm/AMENDMENT_G_VERDICT.json`. Then `/redteam`
   blind re-derivation, from the raw `.npz` arrays, of four targets: b at
   (eps = 1, 500k, seed 3); the G3 contrast; the G5a contrast at the selected
   learning rates; and the selection table for (large, 500k, eps 0.75).
7. Only then do the confirmatory numbers enter `paper/`.

Stop conditions mean stop and report, never improvise (Amendment E, in force).

---

## Signature

Registered by Bowen Zhao (bzhao17@u.rochester.edu), 2026-08-30.

**Authority for the 2026-08-30 amendment is DELEGATED, and that is recorded here
rather than papered over.** Bowen lifted the GPU budget and delegated the
higher-level design decisions of this phase to a max-effort model session; that
session's ruling is `.claude/G_DESIGN_DECISION.md` and is the reason G5a, G5b, G6
and the learning-rate selection rule of G.1 exist. Bowen then authorised the lock
("locked, proceed", 2026-08-30) without personally re-deriving the contents. He
has not hand-checked every pinned invocation; five independent machine passes
have, and their reports are named below and are part of the record:

- `.claude/G_ADJUDICATION.md` — cold adjudication rounds 1–3 of the pre-amendment
  draft (2026-08-29), closing READY.
- `.claude/G_PREFLIGHT_20260830.md` — cold pre-flight of the execution path
  (2026-08-30): 30 confirmed defects, 16 blocking, every one of them missed by
  those three rounds. All fixed before this lock.
- `.claude/G_ADJUDICATION_20260830.md` (round 4) — NOT READY, 2 blocking /
  2 major / 8 minor, all fixed.
- `.claude/G_ADJUDICATION_20260830_R5.md` (round 5, cut off mid-run) — 1 blocking
  and 1 major, both defects introduced by the round-4 fixes, both fixed.
- `.claude/G_ADJUDICATION_20260830_R6.md` (round 6) — **READY TO LOCK**, 0
  blocking / 0 major / 10 accepted minors, adjudicating exactly the 25 G.0 files
  this block digests.

What this signature asserts is therefore narrow and exact: **the procedure above
is the one that will be run, the code digested in G.0 is the code that will run
it, and no confirmatory data existed when this file was sealed.** It does not
assert that a human has independently verified each statistic.

Anyone auditing this amendment should treat the reports above as part of
it. If the confirmatory verdict is later disputed, the dispute is settled by
re-running the pinned invocations against the G.0 digests — not by re-reading
this page.
