# PRESPEC_internal — the three internal/generalisation arms (REPORTED, never gated)

Registered 2026-09-04 (local 17:45) as the verbatim synthesis of the internal-intervention design workflow
(four proposals, eight critiques, one synthesis; session scratch internal_design/). Thresholds and decision rules
below were fixed before any arm produced a number. Digest sidecar: PRESPEC_internal.digest. Branch pushed as int-arms
for an external clock. Nothing in AMENDMENT_G.md moves; every arm imports the locked modules and never edits them.

---

# DESIGN_INTERNAL_20260904 — three arms, pre-registrable

Written 2026-09-04 from the four sibling designs (`internal_design/{probes,patching,circuits,generalization}/DESIGN.md`),
their eight critiques, the locked code and the artifacts cited below. Repo read-only. Every number is re-derived from an
artifact or from a local CPU check named in §7; no registered number is moved. Locked files (AMENDMENT_G.md:39-66) are
imported, never edited. Thresholds are fixed here, before any arm has produced a number.

## 0. Shared frame

**Anchors.** Redteam of reported numbers 09-10→09-14 (`.claude/CONSTRUCT_REVIEW_20260901.md:62`); abstract 09-18 (`:29`);
EXT freeze 09-15/16 (`.claude/EXT_PRESPEC.md:219`); paper 09-25. Owner steer: internal intervention + generalisation.

**Currency (all three arms).** Registered gate panel: `MECH_PANEL_SEED=770000101`, `MECH_SPLIT_SEED=880000101`,
`MECH_N_PER_HALF=1000`, n_rows 20 (`mech_gates.py:91`), replayed with latents by `mech_interventions.half_b_with_latents`
(`mech_interventions.py:61-83`, asserts bitwise equality with the registered split). Eight query rows per context from
`Q_SEED_ROOT + round(1000·eps)·1000 + i` (`mech_predgain.py:52,174`), target d−1 (`:90`). Exact predictive
`predictive(w) = (w@num)/(w@den)` (`corrected_oracle.py:340-345`) in the latent layout lo = k·O+o (`:14-15`);
S(p) = Σ_b p_true(b) log p(b), p_true = predictive(onehot(k,o)) (`mech_predgain.py:80-81,97,112`). Per context i:
regret_i = mean_q[S(full) − S(X)], G_i = mean_q[S(full) − S(abl)], abl = `ablated_weights(·,"order_ablated")`
(`corrected_oracle.py:302-308`); b^X = `np.polyfit` slope of regret_i on G_i (`mech_gates._ols`, `:169-180`).
Bootstrap: 2000 context resamples, `mech_gates._rng(label)` = `default_rng([20260830, crc32(label)])` (`:88-89,117-119`),
paired wherever two quantities share contexts. PRIMARY unit = per-context mean over seeds 3,4,5; per-seed sign agreement
required for any verdict (`mech_gates.gate`, `:277-288`; AMENDMENT_G.md:109-114). Evaluability SE ≤ 0.15 (`:87`).

**Registered comparators (read, not moved).** base@1e-3@500k seed-mean b = 0.5515 / 0.4224 / 0.2092 (SE 0.0387 / 0.0332 /
0.0384), Ḡ_order = 0.01630 / 0.04538 / 0.08048 at eps .5/.75/1 (`confirm/AMENDMENT_G_VERDICT.json` cells). Size-matched
atom slope b′ = +0.0650 / −0.0166 / −0.0187 (SE 0.018 / 0.009 / 0.011), Π* = `atom:01234526` / `atom:01234456` /
`atom:01202340`, Δ(Ḡ_order) = 0.470 / 0.320 / 0.138 with CIs excluding 0 (`mech_ext_20260902/reported/component_control.json`).
G5a −0.2255 (z −12.0), G5b −0.1368 (z −7.4), G6 −0.1590, G3 −0.1921 (verdict). Shrinkage μ = 0.47/0.67/0.80 vs
1−b = 0.45/0.58/0.79 (CONSTRUCT_REVIEW:59).

**Identity refusals, every arm, every run (fail closed, nothing written):** re-derived S(full/abl/prior) equal the registered
npz to 1e-9 (`mech_coarse_oracle.py:214-222`); re-derived S(model_s) equal the registered `model_s{s}` column to 1e-4 and
max|Δp| ≤ 1e-5 against the stored float32 `P` (local 2.2e-7 and 8.7e-7 across torch 2.4.0→2.9.1 — patching
`feas_check.log` §5, circuits `CRITIQUE_feasibility.md` §1; absorbs patch-ident m6, patch-feas F5); the loaded checkpoint is
`{scale}_s{s}_ck{step}.pt` and its sha256 equals the npz `ckpt_sha256[s]` (`mech_predgain.py:168,173`; the final `.pt` is
a different byte stream — probes-feas F2); env `MECH_PANEL_SEED/SPLIT_SEED/N_PER_HALF` set before `mech_phase1` is imported
(`mech_phase1.py:74-76`; `mech_coarse_oracle.py:261-264` pattern — patch-feas F6); eps = 0 refuses (order posterior
uniform; `check_hooks.log` last line).

**Blocking input (both internal arms).** The registered fleet is not on this box: `find campaigns/mech_20260827/confirm
-name '*.pt'` → 0; the npz `nets` field names `campaigns/mech_20260827/confirm/{cell}/nets/eps{tag}/` on the mirror
(AMENDMENT_G.md:207-216). Files needed (owner rsync, ≈ 60 MB, or a cluster CPU job on `debug`/`app-interactive`,
`~/.claude/washu-cluster.md:24`): `base_lr0.001_d500000/nets/eps{0p5,0p75,1p0}/base_s{3,4,5}_ck500000.pt` (9) +
`base_s{3,4,5}_ck0.pt` (3; ck0 depends only on (arch, seed): `corrected_models.py:249-251` — probes-feas F5);
`large_lr0.0003_d500000/nets/eps{0p75,1p0}/large_lr0.0003_s{3,4,5}_ck500000.pt` (6) + `_ck0` (3);
`base_lr0.001_d100000/.../base_s{3,4,5}_ck100000.pt` (9); K=2 `base_lr0.001_d500000_K2` eps 1p0 seeds 3–8 (6, once E3′
lands). Every pipeline is developed and dry-run first on the local exploratory fleet at the registered recipe
(`dose_ext/nets/eps*/base_s{0,1,2}_ck500000.pt`, provenance steps 500000 / peak_lr 1e-3 / n_ctx 20 / n_query 7 / batch 32;
`contrasts/nets/eps0p75/{large,small}_s*_ck500000.pt`) — labelled EXPLORATORY, decision never read (patch-feas F4,
circuits-feas MAJOR-2).

**New seed roots** (grep of `scripts src cluster` for `580_000_000|590_000_000` → 0 hits; taken: 440/550/560/660/770-772/
886/888/889/990 M): `PROBE_SEED_ROOT = 580_000_000` (Arm R training contexts), `OP_SEED_ROOT = 590_000_000` (Arm C partner
draws).

**Sites (architecture-relative; probes-feas F3, circuits-feas MAJOR-1).** Sequence `[null, row_1..row_20, query]`, T = 22,
`out_head` reads position −1 of the last layer only (`corrected_models.py:74-82`); post-LN, ReLU, dropout 0, no final norm
(`:68-72`). `last.out(q)` = output of `transformer.layers[-1]` at position −1 (the head's input); `logits` = `out_head`
output; `cut` = output of `layers[0]` at all positions (base: the only internal cut; large: `layers[0..2]`, reported).
Every position at every layer is query-dependent (one full pass per query, `:78-80`; gen-feas MINOR-5), so every
activation is stored per (context, query) (circuits-feas minor-2). A transparent re-implementation of the layer reproduces
`PFN.forward` to ≤ 7.6e-6 (patching `check_hooks.log`; circuits `feas_check.py`) and is the patch/ablation vehicle; hooks
are not relied on.

---

## 1. Arm R — REPRESENTATIONAL: is the readout the bottleneck?

**Claim.** R-A ("readout-limited"): the last-layer query state h_L carries order information that the linear `out_head`
does not convert into the predictive and that no recalibration of the logits recovers. R-0 ("head-saturated"): a fresh
nonlinear readout of h_L realises no more of the order gain than the trained head does — the shortfall is upstream of the
readout. Nichani/Reading B predicts the same probe signatures as R-A (probes-ident M4); this arm locates the shortfall, it
does not say why it is order.

**Estimand.** For a fresh readout X on frozen features at a site, on the registered contexts × queries:
Δ^X = b^model − b^X (paired). X ∈ {P0(last.out): MLP 128→256→100 softmax; P0(logits): MLP 100→256→100; P0lin(last.out):
linear 128→100; each × ck0 twin}. Every X is trained on the **exact soft target** q_full(·|D, x_q) = predictive(w_lo)
(`corrected_oracle.py:141-173,376-407`; probes-ident m3, probes-feas F4) with cross-entropy, on N = 25 000 fresh contexts
(k ~ U(K), o ~ U(O), `generate_observational`, `corrected_sem.py:425-440`, seed `PROBE_SEED_ROOT + round(1000·eps)·1000 +
train_seed`) × 8 queries, 10 % held out for early stopping, features z-scored on train, 5 train seeds (spread reported;
the same family as the model's predictor, so the free-intercept OLS is commensurable — probes-ident M1).
Primary statistic **T1 = Δ^{P0}(last.out) − Δ^{P0}(logits)** — the order gain present in h_L that the head destroys, net of
output recalibration (probes-ident M3). Atom side from the joint OLS regret = a + b·G + b′·G^{Π*} with Π* the registered
size-matched partition (`component_control.json` `joint.pistar`; `mech_component_control.joint_fit`, `:56-59`):
Δb′^{P0} = b′^model − b′^{P0} (probes-ident m5: atom-coarsened family, never the P2 uniform end).

**Intervention.** None inside the network (the arm reads states); the causal complement is Arm C.

**Controls / nulls.** (i) Convergence: b^{P0lin}(last.out) within ±0.05 of b^model, else REFUSE (the fresh linear head must
reproduce the trained one on exact targets). (ii) ck0 twins (`_ck0.pt`, random features): b^{P0,ck0}(last.out) ≥ b^model −
0.05, else NOT SEPARABLE (Hewitt–Liang; pilot: random features already decode 74 % of p(k|D) at L1.q — `probes/
probe_pilot_out.txt`; probes-ident m2). (iii) Δ^{P0}(logits) printed as "output-recalibration headroom" (subsumes any
temperature map, so no separate sharpening null is needed). (iv) Row-permutation invariance of every site (3.8e-6,
probes `feas_check.py`) and shuffled-target refit (b ≥ 0.95) as pipeline checks. (v) Specificity, REPORTED with its
expectation pre-stated: linear probes at last.out(q) for p(o|D) and for the size-matched atom target p(g_{Π*}(k)|D)
(coarsened by `averaging_matrix`, `mech_coarse_oracle.py:92-99,148`); trained−ck0 KL-fraction recovered, f_ord − f_atom*
with CI; both readings predict f_ord < f_atom*; no decision (probes-ident M4). (vi) Shrink map, DESCRIPTIVE: the linear
hybrid w = p(k|o,D)·p̂(o|D) from last.out(q), through-origin slope on G > 0, mapped to its μ-equivalent on the registered
shrink ladder (b = 0.610/0.444/0.290/0.178/0.091 at μ = .33/.5/.67/.8/.9, eps .75, `probes/readout_hybrid_eps0p75.json`)
beside μ_model = 0.67 (probes-ident M1 fix; never compared to b^model).

**Decision rule (decision cell base@1e-3@500k, eps 0.75; eps 0.5 and 1.0 must agree in sign or the verdict is MIXED).**
τ_eff = 0.10, τ_null = 0.05 (paired SE of a Δ on this panel ≈ 0.032, `readout_hybrid_eps0p75.json` `se_delta`).
- **READOUT-LIMITED** iff T1 ≥ τ_eff with 95 % CI excluding 0 in all 5 refits, per-seed sign agreement, (i) and (ii) hold,
  and |Δb′^{P0}| ≤ 0.05 (the fresh head does not move the atom slope).
- **HEAD-SATURATED** iff the 95 % CI upper bound of T1 < τ_null in all 5 refits and (i) holds.
- Else UNDECIDED; (ii) failing → NOT SEPARABLE. Reported beside, never gated: the same T1 on base@1e-3@100k (dose lever)
  and large@3e-4@500k (capacity lever; sites `layers[-1]`), the P1-hybrid site table, (v), (vi).
Licensed sentence on READOUT-LIMITED: "a fresh readout of the last query state recovers [T1] more of the order gain than
the trained head, beyond what any recalibration of the logits recovers, without moving the atom slope". On HEAD-SATURATED:
"no readout of the last query state realises more order gain than the trained head; the shortfall is upstream of the
readout". Neither sentence mentions cause.

**Cost.** CPU only. Per (cell, eps, seed): 200k forwards ≈ 30 s (1000 ctx × 1 q = 129 ms, gen-feas §3); exact targets
once per eps (25k posteriors × 2 ms + 200k operators × 15 ms ≈ 55 core-min, cached, shared by every cell/seed);
head training ≈ 20 s; scoring + 2000 bootstraps < 1 min. Decision cell + 2 lever cells + ck0 twins + 5 refits: **< 5 CPU-h**
(probes-feas F6). GPU 0.

**Calendar.** 09-05 `mech_probe_acts.py` + tests + prespec; 09-06 `mech_probe_fit.py`, exploratory dry-run (dose_ext s0-2,
eps 0 sanity on `w2/nets/eps0p0`); 09-08 registered run (≤ 1 h wall once the 30 files land); 09-09 report; redteam target
09-10→14: T1 at eps .75 re-derived blind from `probes_eps0p75.npz`.

**Files.** New: `scripts/mech_probe_acts.py` (panel replay, query replay, activations per (context, query) at last.out(q)
and logits, float32; targets `w_lo`; `stage train` builds the 25k-context set + exact q_full via the cached operators),
`scripts/mech_probe_fit.py` (heads, refits, hybrids, b^X via `mech_gates._ols`, paired resamples via `mech_gates._rng`),
`scripts/mech_probe_report.py`, `tests/test_mech_probe.py` (uniform-p̂ hybrid == `order_ablated` with `world.prior_order`,
bitwise — probes-feas §1; shuffled target b ≥ 0.95; S identity on 3 contexts), `campaigns/mech_int_20260905/PRESPEC_internal.md`
(+`.digest`). Locked imports: `mech_interventions.half_b_with_latents`, `mech_predgain.{Q_SEED_ROOT,_S}`,
`mech_gates.{_ols,_rng,_boot_idx,N_BOOT,SE_MAX}`, `mech_component_control.joint_fit`, `mech_coarse_oracle.averaging_matrix`,
`corrected_oracle.{exact_joint_posterior,ablated_weights,obs_query_operator}`, `corrected_models.{PFN,ModelConfig}`,
`corrected_trackb.SCALES`, `corrected_world.make_world`, `corrected_verdict.eps_tag`, `split_panel.contexts_sha256`.

**Critique findings absorbed.** probes-ident M1 (no hybrid vs b^model; within-family only), M2 (one pre-named site per
statistic; no max over 40 probes), M3 (P0(logits) null; T1), M4 (size-matched atom target as the specificity companion),
m1 (⊕ variant cut; routing goes to Arm C), m2 (ck0 conjunct; logZ dropped), m3 (exact soft targets), m4 (lever table
descriptive), m5 (Π* family for both atom cells), m6 (paired ordinal primary); probes-feas F1 (pull list §0), F2 (`_ck`
files), F3 (sites by n_layers), F4 (soft targets), F5 (3 ck0 files), F6 (compute), F7/F8 (tests, eps 0 sanity).

---

## 2. Arm C — CAUSAL: where does the realised order information enter, beside atom information at matched size?

**Claim.** C-routing: at the internal cut of the base model, the order information the model realises sits in the context-row
representations (not yet aggregated to the query/null positions), and it is read by last-layer attention through values.
Pre-stated mapping (patch-ident M5): SAME-ROUTE for order and atom ∧ identical head verdicts ⇒ "no order-specific routing
detected: the realised order information travels the atom pathway; the shortfall is not a routing failure" (reading-B
compatible, and the paper says so); LATER-THAN-ATOM ⇒ reported with the functional-role rival named first (atom information
sets coefficients that multiply x_q, requiring query-side computation; order information changes the residual shape,
additive). No clause of this arm speaks about cause; it describes the realised pathway.

**Paired sources (exact, model-blind, built once per eps).** For context A with latent (k,o): e = x_π U^T
(`corrected_sem.py:105-107,115-117`), sp = `params_for(Σ_k, π_o)` (`:70-102`). Standardised draws z = e / b_{k,o}
(b at `:100`). Every partner is z pushed through another latent with its own b — legitimate because the residual law is a
b-scale family (`_al_ac` `:149-151`, Gaussian sd √2·b `:169,410`, AL `:411-413`) — patch-ident M1 / patch-feas F1
(unrescaled partner collapses p(k_true|B) to mean 0.698, min 0.000; rescaled 1.000/0.999: `patching/crit_ident_check2.log` §3):
- order-paired B(o′): `from_permuted(forward_map(z·b_{k,o′}, sp′), sp′)`, o′ seeded uniform (`OP_SEED_ROOT + round(1000·eps)·1000 + i`)
  over the partners whose target conditional differs from o's (the partner sharing o's parent set of x_{d−1} has v_ord ≡ 0 and
  is refused — patch-ident m1; ≤ 5 order classes are stated); all visible o′ built and reported.
- atom-paired B′(k′): same o, all 7 k′ built; **primary partner = argmin_{k′} |‖v_atom(k′)‖² − ‖v_ord(o′)‖²| per context**,
  computed in the build (patch-ident M2, patch-feas F2: at the Ḡ-matched pairs ‖v_atom‖²/‖v_ord‖² is ×10 median,
  `crit_ident_check2.log`); the Ḡ-matched k′ from `coarse_eps{tag}.npz` `G_bar` (`mech_coarse_oracle.py:228`, 28 pair merges)
  reported beside for continuity with the component control.
- ordmean / atommean: the site's activations averaged over the visible order family / the 8 atoms (activation-level order /
  atom removal); resample: fresh z, same (k,o) (`mech_interventions.intervene` `:90-91`, the null spread); gaussianised:
  e_g = b√2·Φ⁻¹(F_mix(e/b)) in place, same draws (paired order removal at the input, patch-ident M3).

**Exact directions at A, per query, in 100-bin log-predictive space** (`ObsQueryOperator` `corrected_oracle.py:325-345`):
v_ord = log predictive(p(k|o,A)·p(o|B)) − log q_full(A); v_atom = log predictive(p(k|o,B′)·p(o|A)) − log q_full(A);
v_abl = log q_abl(A) − log q_full(A); v_atomabl likewise with `"atom_ablated"` (`:309-314`). Inner product weighted by
p_true and centred. Model movement Δ_s(C) = log p_patch(A; s ← C) − log p_model(A); INPUT = the full context swap.

**Estimands.** Pooled weighted LS over contexts × queries, (γ_ord, γ_atom)(s, C) = argmin Σ‖Δ_s(C) − γ_ord v_ord − γ_atom
v_atom‖² (2×2 normal equations per context, summed); the weighted R² and the off-span norm are stored per (site, source)
(patch-ident M4: local median R² 0.44 for the model order swap). Φ_ord = γ_ord(INPUT, B), Φ_atom = γ_atom(INPUT, B′).
Share ρ(s) = γ(s)/γ(INPUT), formed only where the CI of γ(INPUT) excludes 0 and reported with its own CI width
(≤ 0.3 for evaluability — patch-ident m8). Necessity ν̃_ord(s) = ν_ord(s)/ν_ord(cut) with ν the coefficient of
Δ_s(ordmean) on (v_abl, v_atom) and ν_ord(cut) the complete-cut ordmean patch (patch-ident M3). Exact leaks λ subtracted
(model-blind cross-axis coefficients of the exact swaps).

**Sites (base).** INPUT; `cut` (complete, identity: must equal INPUT ≤ 1e-6, `check_hooks.log` identity (b) 0.0);
`cut_c` / `cut_q` / `cut_t` (rows / query / null); `A2h@q` for h = 1..4 (last-layer head output at the query before
`out_proj`; the only position set that exists for the final layer — circuits-ident 1.7 / circuits-feas MAJOR-1);
`A2val`, `A2pat` (last-layer values at all positions / pattern at the query row); `M2q`. 12 sites × 16 sources. large:
`cut_l` for l = 0..2 and the last-layer sites, reported.

**Controls / nulls.** (1) Resample source: the null band for every coefficient, with the exact resample movement's γ printed
beside the model's (patch-ident m4). (2) **Temperature-matched null** (circuits-ident 1.1: at eps .5 a p^{0.5} flattening
of the model's own predictive reads as an order-specific loss, Δb +0.22, `identification/generic_damage_eps0p5.json`): for
every (site, source) the model's predictive re-tempered to the patched predictive's mean entropy gives γ_T; any positive
routing or head claim requires γ − γ_T with CI excluding 0. (3) Size: the pooled regression of γ on log‖v‖² with a family
indicator over the 5 o′ × 7 k′ swaps per context; Φ_ord vs Φ_atom read at matched ‖v‖² (family members at ‖v_atom‖²
1.3–2.5 lie inside the order range 0–3.9, `crit_ident_check2.log`). (4) Specificity in norm units (patch-ident m2):
‖(γ_atom(INPUT,B) − λ_atom^{ord})·v_atom‖ ≤ 0.5·‖γ_ord(INPUT,B)·v_ord‖, pooled; failure voids the order reading of γ_ord.
(5) eps = 0 refuses; P-identity, S-identity, checkpoint sha as §0. (6) Per-row patches `cut_c_j` reported as a
matched-magnitude curve only (the exact posterior saturates per row: 0.053 single-row sum vs 0.40 whole swap,
`check_hooks.log`).

**Decision rule (base@1e-3@500k, eps .5/.75/1, seeds 3-5; a verdict must be identical at all three eps, else MIXED —
patch-ident m5; ±50 % threshold sensitivity printed beside every verdict).**
- **C1 consistency (reported, not gated; patch-ident m7):** family coefficient Φ_ord − Φ_atom at matched ‖v‖² < 0 with CI
  excluding 0, and Φ_ord(1.0) − Φ_ord(0.5) > 0 (unpaired). If either fails the sentence "at matched movement size the
  activation-level transfer of order is lower than of atom" is not written; C2–C3 are still evaluated.
- **C2 layer-1 cut, per axis:** s_c = ρ(cut_c), s_q = ρ(cut_q), s_t = ρ(cut_t), ι = 1 − s_c − s_q − s_t. ROWS if s_c ≥ 0.8
  (CI lower ≥ 0.7); PRE-AGGREGATED if s_q + s_t ≥ 0.5 (CI lower ≥ 0.4); MIXED otherwise. Contrast: LATER-THAN-ATOM if
  s_c(ord) − s_c(atom) ≥ 0.2 with paired CI excluding 0 and γ − γ_T CI excluding 0; EARLIER if ≤ −0.2; SAME-ROUTE
  otherwise. Local n = 8 (exploratory): s_c(ord) 0.42–1.16 (plus one degenerate context at 2.8 with Φ_ord 0.07 — the CI guard
  on ρ excludes it) beside s_c(atom) 0.56–0.97 (`crit_ident_check2.log`) — the atom pathway is also ROWS; SAME-ROUTE is
  the expected outcome and is pre-mapped above.
- **C3 last-layer heads:** DEDICATED-ORDER-HEAD if ∃h: ρ_ord(h) ≥ 0.5 ∧ ν̃_ord(h) ≥ 0.5 ∧ ρ_ord(h) − ρ_atom(h) ≥ 0.3 (CI
  excluding 0; γ − γ_T CI excluding 0); DISTRIBUTED otherwise; the verdict requires agreement of the *count* across seeds,
  never of the index. CONTENT-ROUTED if ρ_ord(A2val) ≥ 0.8 ∧ ρ_ord(A2pat) ≤ 0.2; PATTERN-DEPENDENT if ρ_ord(A2pat) ≥ 0.3;
  MIXED else; the same for the atom axis; CONTENT-ROUTED for both is the null (iid-row additive evidence,
  `corrected_oracle.py:122-137`) and is stated as such.
Licensed sentence: "the order information the model uses enters the query readout at [sites], with [same/different]
routing from a size-matched atom swap". Nothing in C1–C3 can disconfirm "smallest component"; Arm G carries that.

**Cost.** CPU only, measured at 1 thread (`patching/feas_check.log` §4): patched manual forward batched over 8 queries
1.15 ms base / 6.98 ms large; posterior 1.9 ms; operator 13.4 ms. Build per eps: 1000 × (8 operators + ≈ 16 posteriors)
≈ 2.5 core-min. Run per (seed, eps): 1000 × 12 sites × 16 sources × 1.15 ms ≈ 0.06 core-h base → base 3 × 3 × 0.06 ≈
0.6 core-h; large (2 eps × 3 seeds, 3 cuts) ≈ 6 core-h (patch-feas F3). Bootstrap on stored per-context sums: seconds.
**≈ 8 CPU-h; ≈ 1 h wall on 14 workers.** Storage ≈ 25 MB per (cell, eps, seed). GPU 0.

**Calendar.** 09-05 `mech_patch_build.py` + tests (replay ≤ 1e-12, z-pair covariance identity and |p(k|B) − p(k|A)|_∞ small
on the panel, manual = module ≤ 1e-5, complete-cut identities ≤ 1e-6, eps 0 refusal, invisible-partner refusal); 09-06
`mech_patch_run.py` + `mech_patch_analyse.py`, dry-run on dose_ext s0-2 (exploratory); 09-08 registered run (base ≈ 10
min wall, large ≈ 1.2 h) once §0's files land; 09-09 report; redteam target 09-10→14: s_c(ord) − s_c(atom) at eps .75 and
one head's ρ from the raw `.npz`.

**Files.** New: `scripts/mech_patch_build.py` (sources, exact posteriors/predictives, directions, `‖v‖²`-matched partner,
provenance keys `panel_seed, split_seed, n_per_half, sha256_b, q_seed_root, op_seed_root, coarse_sha256`; refuses `--out`
inside `campaigns/mech_20260827` as `mech_coarse_oracle.py:259-260`), `scripts/mech_patch_run.py` (transparent forward with
a `patches` dict keyed (layer, kind, index, positions); writes per-context sums ⟨Δ, v⟩ and the 4×4 Gram, R², identities,
sha256s, torch/numpy versions), `scripts/mech_patch_analyse.py` (γ/ρ/Φ/ν̃/λ/γ_T, bootstrap, C1–C3, sensitivity band →
`campaigns/mech_int_20260905/reported/patching.json` + MANIFEST), `tests/test_mech_patch.py`. Locked imports:
`mech_interventions.{half_b_with_latents,intervene}`, `mech_predgain.{Q_SEED_ROOT,_S}`, `mech_gates.{_ols,_rng,_boot_idx,
N_BOOT,SE_MAX}`, `corrected_oracle.{exact_joint_posterior,ablated_weights,obs_query_operator}`,
`corrected_sem.{params_for,residuals_from_data,forward_map,from_permuted,to_permuted,sample_residuals,residual_logpdf}`,
`corrected_models.{PFN,ModelConfig}`, `corrected_trackb.SCALES`, `corrected_world.make_world`, `corrected_verdict.eps_tag`.

**Critique findings absorbed.** patch-ident M1 (standardised draws), M2 (‖v‖² matching + pooled size regression), M3
(ν normalised; paired gaussianised source), M4 (R², off-span, CI guard on ρ), M5 (pre-stated mapping), M6 (D5 cut, see §6),
m1 (invisible partner refused), m2 (norm-unit specificity), m3 (no symmetry test), m4 (exact resample γ printed), m5
(verdict identical across eps), m6 (1e-5), m7 (C1 consistency), m8 (ratio CI widths); patch-feas F1–F7; circuits-ident
1.1 (temperature-matched null imported as control (2)), 1.7 / circuits-feas MAJOR-1 (final-layer position set), minor-2
(per-forward storage).

---

## 3. Arm G — GENERALISATION: the size-matched world (K = 2), then the internal arms re-invoked there

**Claim.** The decisive control the literature names is the one that breaks the rank correlation between "is the order
component" and "is the smallest component" (Rahaman's amplitude-equalised control; `lit/learning_order.md` synthesis 3).
At K = 8 the atom component is 4–18× the order component (CONSTRUCT_REVIEW:67). At K = 2, eps = 1, the two components are
the same size. Reading A (order-specific) predicts the atom slope b′ stays near 0 there and the order slope b stays at the
registered-world level for its Ḡ; reading B (size-specific) predicts b ≈ b′ at matched Ḡ. **Re-derived here (§7, 200
contexts × 4 queries, n_rows 20, `make_world(k=2, d=3, eps)` = seed 990003002, `corrected_world.py:90`):** Ḡ_order /
Ḡ_atom = 0.0668 ± 0.0046 / 0.0616 ± 0.0040 at eps 1 (ratio 0.92), 0.0349 / 0.0489 at eps .75 (1.40), 0.0082 / 0.0426 at
eps .5 (5.2), with Ḡ_atom = mean_i[S(full) − S(atom_ablated)] (the one-block member of the registered atom family,
`corrected_oracle.py:309-314`, `mech_coarse_oracle.py:148`); the same script at K = 8 gives 14.4 / 6.4 / 4.8. The
generalization critique's [local] ratio 0.5 used S(abl) − S(prior), a different functional (0.034 at eps 1 here); the
component-control functional is the registered one and is what this arm uses. Predicted values from the K = 8 curve
(b = 0.55/0.42/0.21 at Ḡ 0.016/0.045/0.080): under B, b ≈ b′ ≈ 0.3 at K = 2, eps 1 (D_G ≈ 0); under A, b′ within
±0.05 of 0 as at K = 8 and D_G ≈ +0.25.

**Cells.** E3′: `base_lr0.001_d{100000,500000}_K2`, eps {.5, .75, 1}, seeds 3–8 (`cluster/fire_ext.sh:202-205`; fired
2026-09-03T08:01–08:05Z, `.claude/ext_jobs.tsv`; walls 2 h / 7 h). Scored by the EXT harvester on the registered gate panel
with `MECH_K=2`, `MECH_CONFIRM=1` (`cluster/mech_score_ext.sbatch:32,34,40`; `mech_predgain_ext.py:26-27`) into
`campaigns/mech_ext_20260902/predgain/base_lr0.001_d500000_K2/s3-8/`. **State 09-04 21:45Z:** no `_K2` directory under
`predgain/`, no K2 line in `.claude/harvest.log` — the cells are queued, not landed (gen-feas MAJOR-1). No new GPU is
asked. The rule below is written before any K = 2 `b` exists (the d=4 and ws cells were scored before their rule — disclosed,
gen DESIGN §5).

**Estimand (primary).** On the K = 2 gate panel (1000 contexts, same seeds/queries), the coarse pass `mech_coarse_oracle.py`
with `MECH_K=2` (`_patch_world`, `:105-117`; `partitions_for(2)` → {full, one-block}, `:84-89`; the `cc_dim4` precedent:
`reported/cc_dim4/component_control.MANIFEST.json` inputs `predgain/base_lr0.001_d500000_dim4/s3-8/*.npz`), then the joint
OLS regret_i = a + b·G_order,i + b′·G_atom,i (`mech_component_control.joint_fit`, `:56-59`), G_atom,i = S_i(full) −
S_i(atom:00). **D_G = b − b′**, one paired context-bootstrap stream (2000, `mech_gates._rng("k2:…")`), PRIMARY = seed-mean
over seeds 3–8, per-seed sign agreement on D_G required. Secondary, the mirror of the registered control:
Δ(Ḡ_atom) = median_{25 order partitions nearest Ḡ_atom} b^H − b^{atom:00} (the atom family has one member at K = 2;
`analyse` degenerates to it, `mech_component_control.py:113-119`).

**Decision rule (K = 2, base@1e-3@500k, eps 1.0 decision; eps 0.75 consistency — near-matched sizes; eps 0.5 reported —
ordered sizes).** Evaluable iff SE(b), SE(b′) ≤ 0.15 and provenance holds (`world_index.json` K = 2, `sha256_b` differs
from the registered cell's, `confirm` true, m_q 8; EXT_PRESPEC §5.6).
- **FAVOURS A** iff D_G > 0 with 95 % CI excluding 0 AND the upper 95 % CI of b′ ≤ 0.15, at eps 1, with the same sign of
  D_G at eps .75.
- **FAVOURS B** iff the 95 % CI of D_G lies within (−0.10, +0.10) AND the lower CI of b′ ≥ 0.10 (the atom component of
  matched size carries a shortfall of its own) at eps 1; or D_G < 0 with CI excluding 0.
- Else UNDECIDED (printed with both predicted values).
Direction replication (reported, no verdict label; Holm over the family, one rule: adjusted p < 0.05 one-sided, plus a
TOST-at-0 arm at δ = 0.10 so "no response" is reportable — gen-ident 12, 15): dose 100k→500k paired (`mech_gates.
contrast_paired`, `:250-274`, latents asserted `:261-262`) at eps .5/.75/1 in K = 2, d = 4 (base@1e-3 only — gen-ident 7)
and ws11/12/13; joint-OLS b beside b in every world (gen-ident 13). Family verdict rules of the sibling design are dropped
(gen-ident 5).

**Internal transfer (reported).** Arms R and C re-invoked verbatim on the K = 2 nets at eps 1 (base@1e-3@500k, seeds 3–5;
env `MECH_K=2` before import, the `mech_predgain_ext.py:26-27` mechanism; no hyper-parameter retuned): T1 and
s_c(ord) − s_c(atom). TRANSFERS if both have the registered-world sign with CIs excluding 0; PARTIAL if one; WORLD-SPECIFIC
if neither. This replaces the sibling's argmax-layer profile, which has no null at L = 2 (gen-ident 9).

**Cost.** GPU 0 new (E3′ in flight). CPU: coarse pass 3 eps × 1000 × 8 × 4 ms ≈ 2 min; verdict ≈ 1 min; transfer ≈ 30 min.

**Calendar.** Rule registered 09-05 in `PRESPEC_internal.md` (pushed — external clock, EXT_PRESPEC §9). Coarse pass + verdict
the day `predgain/base_lr0.001_d500000_K2/s3-8/predgain_eps1p0_ck500000.npz` is harvested (queue drains ≈ 09-07/08,
gen-feas MAJOR-1); transfer the same day if the 6 K = 2 checkpoints are pulled. Redteam target: D_G at eps 1 from the raw
`S` arrays. Descope: K = 2 500k not on this box by 09-12 → the primary is reported "not run" and the direction
replication (d = 4, ws, already on disk) stands alone; K = 2 100k only → dose replication reported, D_G at 100k labelled
EXPLORATORY.

**Files.** New: `scripts/mech_k2_verdict.py` (runs `mech_coarse_oracle.run_eps` with `MECH_K=2`, `--ref-dir` the ext cell,
`--out campaigns/mech_int_20260905/coarse_K2`; joint fit, D_G, Δ(Ḡ_atom), bootstrap, the rule, the direction table;
`--validate` reproduces `reported/component_control.json` b′ for base@1e-3@500k eps .75 to 1e-9 before any write),
`scripts/mech_transfer_k2.py` (sets `MECH_K`, imports and calls the Arm R/C drivers). Locked imports:
`mech_coarse_oracle.{run_eps,partitions_for,averaging_matrix}`, `mech_component_control.{joint_fit,slopes,analyse}`,
`mech_gates.{_ols,_rng,_boot_idx,contrast_paired,N_BOOT,SE_MAX,Z}`, `mech_predgain_ext` (pattern only),
`corrected_world.make_world`, `corrected_verdict.eps_tag`.

**Critique findings absorbed.** gen-ident 3 and 10 (K = 2 promoted from description to the registered A-vs-B rule; C2's
Ḡ-determined framing dropped), 5 (no ≥ 4-of-7 family verdict), 7 (no capacity conjunct at pinned LR in new worlds), 9
(profile argmax cut; transfer = frozen-procedure re-invocation of the internal arms), 12 (TOST-at-0 arm), 13 (joint-OLS b
per world), 15 (one rejection rule); gen-feas MAJOR-1 (no calendar claim ahead of the queue), MAJOR-2 (nets pulled or the
transfer is reported "not run"; the primary needs no checkpoint), MINOR-5 (site is the head's input, query-dependent by
construction); circuits-ident 1.2 fix ("the lever that breaks the order = smallest coupling is K = 2").

---

## 4. Calendar and compute

| date | step | needs |
|---|---|---|
| 09-05 | `PRESPEC_internal.md` (§1–§3 verbatim) + `.digest`, committed and pushed; owner rsync request (§0 list, ≈ 60 MB); `mech_patch_build.py`, `mech_probe_acts.py`, tests | nothing |
| 09-06/07 | `mech_probe_fit.py`, `mech_patch_run.py`, `mech_patch_analyse.py`; end-to-end dry-run on dose_ext s0-2 + contrasts large/small (EXPLORATORY; identity vs `predgain_doseext/predgain_eps0p75_ck500000.npz`) | local fleet |
| 09-08 | registered runs of Arm R (< 1 h wall) and Arm C (base 10 min, large 1.2 h); K = 2 coarse pass + verdict if harvested | the pulled files; K = 2 npz |
| 09-09 | reports, figure drafts, entries on the 09-10 redteam list | — |
| 09-10→14 | blind re-derivations: T1 (eps .75), s_c(ord) − s_c(atom) (eps .75), D_G (K = 2, eps 1) | — |
| 09-15 | freeze; 09-18 abstract; 09-22 last registered number; 09-25 paper | — |

Descope, in order: files not on this box by 09-09 → Arm R/C run on the cluster `debug` partition (owner submits; scripts are
venue-agnostic, outputs ≤ 25 MB per cell); not by 09-12 → Arm R/C enter the abstract as EXPLORATORY on seeds 0-2 and are
registered only if the run lands by 09-22. K = 2 not harvested by 09-12 → Arm G primary "not run".

| arm | CPU-h | GPU-h new | wall on 14 cores |
|---|---|---|---|
| R | < 5 | 0 | < 1 h |
| C | ≈ 8 (base 0.6, large 6, builds 0.2) | 0 | ≈ 1 h |
| G | < 1 | 0 (E3′ ≈ 6 jobs in flight since 09-03) | minutes |

## 5. What each arm licenses, and the one paragraph the paper carries

R: where the shortfall sits relative to the readout (READOUT-LIMITED / HEAD-SATURATED). C: the realised order pathway
beside a size-matched atom pathway (routing verdict; likely SAME-ROUTE, pre-mapped). G: whether the shortfall follows
the *kind* or the *size* of the component when the sizes are equalised (FAVOURS A / B). Only G touches cause; R and C
are read as "where", never "why". If G reads FAVOURS B, the paper's §3 pre-emptive sentence (CONSTRUCT_REVIEW:68) becomes
the claim, and R/C describe the internal locus of a smallest-component shortfall.

## 6. Cut table

| cut | from | why (fact, cited) |
|---|---|---|
| Δ^{P1} hybrid vs b^model (rules 2–3 of the probes design) | probes §2, §5 | free-intercept OLS is not family-invariant: a μ = 0.5 hybrid reads b_OLS 0.444 / b_origin 0.359 / ratio 0.277 on the same panel (`readout_hybrid_eps0p75.json`); the F.2c readout hybrid reads b = 1.045 while the output realises 0.58 (probes-ident M1) |
| max over ~40 P1 probes vs τ_null | probes §5 rule 3 | expected max of 40 zero-mean draws at SE 0.02–0.03 is 0.05–0.08 > τ_null; the rule fails under its own null (probes-ident M2) |
| ⊕ (query ⊕ null ⊕ ctx-mean) head; "A-route" at L1 sites | probes §2 P0 variant | width confound (2.5× first-layer parameters); no layer after L1 to route to (`corrected_models.py:68-80`; probes-ident m1). Routing is Arm C's C2 |
| logZ "cannot-need" calibration; P2 uniform end | probes §4 | no common unit (KL fraction vs R²); (1/K)p(o\|k,D) ≠ `atom_ablated` (`corrected_oracle.py:309-314`) (probes-ident m2, m5) |
| lever-table "predictions" (A-rep/A-read/A-route rows) | probes §3 | not a partition; fitted to the seven locked contrasts (probes-ident m4); the lever T1 values are reported descriptively |
| specificity index λ, M1 kind labels | circuits §2, §5 | λ does not measure kind: the exact `atom_ablated` oracle reads λ = −0.98 at eps 1 (above −log 4); a temperature flattening of the model's own predictive reads λ = 4.5, Δb +0.22 at eps .5 (`identification/generic_damage_eps0p5.json`; circuits-ident 1.1). The confound is imported as Arm C control (2) instead |
| emergence-dose ratio R (M4) | circuits §5 | the "R ≤ 2 → supports A" boundary coincides with the Saxe amplitude null (Ḡ ∝ eps^2.3 ⇒ R = 2.0); the null spans 2.0–4.9 across all three bins (circuits-ident 1.2) |
| greedy O\*/A\*, b_{O\*} ≥ 0.9, twin (gauss) patching | circuits §2, §3.2, M2–M3 | knife-edge target: a correct order removal landing on (1/O)p(k\|D) reads b = 0.851/0.903/0.937 (`identification/estimand_checks.json`); the gauss twin does not remove order under the training world (59 % of twins keep one order > 0.5); partial twin patches are off-manifold (circuits-ident 1.5, 1.6, 1.9) |
| Δb-sorted 32-neuron blocks | circuits §3.1 | kind-specific by sorting under a distributed continuum (circuits-ident 1.9) |
| E5 steering | circuits §3.5 | needs a sharpening null and a fitted direction; Arm R's Δ^{P0}(logits) already bounds the recalibration headroom exactly (circuits-ident 1.12) |
| cross-checkpoint patching D5 | patching §5 | its compatibility gate selects on the outcome (QUERY-SIDE cases fail it); only mid-cosine exploratory checkpoints exist (`CKPTS='0'`, AMENDMENT_G.md:218-222) (patch-ident M6) |
| per-row fidelity ratios; `A1c/A1h/M1c/M1q` sites | patching §3 | per-row exact posterior saturates (0.053 vs 0.40, `check_hooks.log`); site count trimmed to 12 for multiplicity (patch-ident m5) |
| residual-law worlds r ∈ {1,2} (≈ 40 GPU-h) | gen §4.1 | matched on the atom/order size ratio as well as on Ḡ, so C2 is reading B's prediction (gen-ident 3); the r ladder scales both cumulants (r=4 has 78 % more excess kurtosis than r=1; gen-ident 4a); TOST at δ = 0.10 needs SE < 0.030, no cell reaches it (gen-ident 1–2) |
| n80 (≈ 21 GPU-h) | gen §4.2 | raw paired Δb on an n-extension is not identified (G_i changes by 0.318, regret concave in G; EXT_PRESPEC §5.5); no `build_n40_eps*.npz` on this box (gen-ident 8, gen-feas MINOR-3) |
| wide/deep 2×2 (≈ 56–65 GPU-h) | gen §4.3 | R3 accepts the null (a width effect ≤ 0.10 is undetectable at SE 0.04–0.05); walls unmeasured (∝ layers^0.92 ⇒ TIMEOUT at 4.5 h) (gen-ident 6, gen-feas MINOR-4); queue ≥ 33 h behind (gen-feas MAJOR-1) |
| tilted `prior_order`; train-on-A/score-on-B | gen §4.4–4.5 | three new code paths for one exploratory point; the estimand is ill-posed (misspecification term in regret) (gen DESIGN §4.4–4.5) |
| C4 probe-profile argmax transfer; `locus.json`/`run_locus` contract | gen §6 | argmax over two layers is the last layer by construction; the contract is unilateral (`grep locus.json` over the sibling designs → 0) (gen-ident 9, gen-feas MAJOR-2) |
| family verdict "≥ 4 of 7 axes"; capacity conjuncts in new worlds | gen §5 | reachable from four author-controlled cells; LR moves b more than any gate effect (`refits.json`: base 0.55–0.90 across LR at eps .5) (gen-ident 5, 7) |

## 7. Local checks run for this document (CPU, `.venv` py3.11, torch 2.9.1, numpy 1.26.4)

- K = 2 vs K = 8 component sizes (§3): `make_world(k, d=3, eps)`, 200 (K=2) / 120 (K=8) contexts × 4 queries, n_rows 20,
  rng 123456; Ḡ_order, Ḡ_atom = mean[S(full) − S(atom_ablated)], corr, frac G ≤ 0 — printed in the session transcript
  (K=2 eps 1: 0.0668 ± 0.0046 / 0.0616 ± 0.0040, corr +0.45, frac 0.06; eps .75: 0.0349 / 0.0489; eps .5: 0.0082 / 0.0426;
  K=8: 0.0208 / 0.2999, 0.0528 / 0.3396, 0.0730 / 0.3532 — consistent with the registered 0.0163 / 0.0454 / 0.0805 at n = 120).
- Registered comparators read from `AMENDMENT_G_VERDICT.json` and `component_control.json` (values in §0).
- Seed roots 580M / 590M unused (`grep -rn` over `scripts src cluster` → 0).
- E3′ state: `.claude/ext_jobs.tsv` rows 242635–242640; `.claude/harvest.log` through 2026-09-04T21:45Z carries no K2
  score row; `ls campaigns/mech_ext_20260902/predgain/` has no `_K2` directory.
- Everything else cited is the sibling critics' logs in this tree (`patching/{check_axes,check_hooks,crit_ident_check,
  crit_ident_check2,feas_check}.log`, `probes/probe_pilot*_out.txt`, `probes/readout_hybrid_eps*.json`,
  `identification/*.json`), re-read today, not re-run.
