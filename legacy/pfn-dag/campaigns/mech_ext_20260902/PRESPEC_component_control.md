# Pre-specification — the size-matched non-order component control (REPORTED, never gated)

Written 2026-09-03 06:00 before `scripts/mech_coarse_oracle.py` / `scripts/mech_component_control.py`
exist in the repo. Design source: the 09-02 plan critique, alternatives lens §1 and identification lens
§2 (session scratch); those lenses ran a prototype and reported an outcome. This document fixes the
estimand and the decision rule so the repo run is graded against text written before it ran; a blind
`/redteam` re-derivation is owed on the numbers (09-10 list).

## Question
Reading A: the PFN's shortfall against the exact predictive is specific to the ORDERING component.
Reading B: it is specific to the SMALLEST component of the achievable gain, whatever its kind.

## Oracles (latent layout lo = k·O + o; W = w_lo.reshape(K, O), uniform prior)
- Atom-coarsened, for a set partition Π of the K=8 atoms: w_Π = (A_Π @ W).ravel(), A_Π[a,b] = 1/|g(a)| if
  g(a)=g(b) else 0. Nests `full` (singletons) and `atom_ablated` (one block). Leaves p(o|D) unchanged.
- Order-coarsened (mirror), for a set partition H of the O=6 orderings: w_H = (W @ B_H^T).ravel().
  Nests `full`; its one-block end is (1/O)·p(k|D), which is NOT the registered order_ablated
  (1/O)·p(k|o,D) — both are printed, neither substituted.
- All 4140 atom partitions and 203 order partitions are enumerated; a partition's component size is
  Ḡ^X = mean_i [S_i(full) − S_i(w_X)], an ORACLE quantity (model-blind).
- Contexts, queries, p_true and S_i(·) are exactly those of the registered scoring
  (`mech_predgain.py`: gate panel 770000101/880000101/1000, Q_SEED_ROOT + round(eps·1000)·1000 + i,
  m_q = 8). The run REFUSES unless its S_i(full), S_i(abl), S_i(prior) equal the registered npz's to 1e-9.

## Estimand
regret_i = S_i(full) − mean_seeds S_i(model_s) from the registered npz (unchanged).
b^X = OLS slope of regret_i on G^X_i = S_i(full) − S_i(w_X); intercept a^X. Same estimator as the spine
(`mech_gates._ols`), same bootstrap (2000 context resamples, `mech_gates._rng(label)`).

## Primary statistic
Δ(Ḡ*) = median_{H : 25 nearest Ḡ^H to Ḡ*} b^H − median_{Π : 25 nearest Ḡ^Π to Ḡ*} b^Π,
medians recomputed inside every bootstrap resample, at Ḡ* ∈ {Ḡ_order(eps), 0.02}.
Secondary: joint OLS regret_i = a + b·G_i + b'·G^Π*_i with Π* the single atom partition nearest
Ḡ_order; the family curve b^Π vs Ḡ^Π restricted to Ḡ^Π ≥ 0.005 (slopes below are unidentified);
the non-parametric share of contexts where the model outscores each size-matched oracle.

## Decision rule (fixed before the run)
- Favours A if Δ(Ḡ*) > 0 with the 95 % bootstrap CI excluding 0 at BOTH Ḡ* values, in
  base@1e-3@500k at all three eps (.5/.75/1); equivalently b' within ±0.05 of 0 while b reproduces
  its registered value.
- Favours B if b^Π rises as Ḡ^Π falls and meets the order curve at matched Ḡ: |Δ| < 0.05 with a CI
  containing 0, or b^Π(Ḡ*) ≥ b^H(Ḡ*).
- Anything else: undecided; report as such.
Cells: the 15 predgain_confirm base@1e-3 cells (5 doses × 3 eps) and large@3e-4@500k (.75/1) are all
reported; the decision reads base@1e-3@500k only. Confounds to print: a coarsened atom component is a
sub-component of an axis the model captures at 0.94–0.97 (l3 bands); the mirror family answers that.
