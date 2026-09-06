# PRESPEC_internal — errata and pre-named choices (append-only; the digested PRESPEC is never edited)

- 2026-09-04 18:50 (E1, before any Arm R number): §1 says "5 train seeds (spread reported)" with the training
  seed inside the data seed, while its cost line ("exact targets once per eps … cached, shared") assumes one
  training draw. Resolved toward the estimand text: the five refits are five independent training DRAWS
  (train_seed 0-4 → five fresh 25k-context sets with exact targets), head seed 1000·ts + s. Cost becomes
  ~15 builds × ~6 CPU-min; the decision rule's "in all 5 refits" is evaluated over draws.
- 2026-09-04 18:50 (E2, pre-named): the specificity probes (§1 (v)) are linear heads at last.out(q) trained on
  every (context, query) ROW of the 25k set with the row's context-level target (p(o|D) or the Π* group marginal
  from the saved w_lo); the context-level probe on the panel is the mean of its 8 query-row outputs.
- 2026-09-04 18:50 (E3, pre-named): the descriptive hybrid (§1 (vi)) uses the trained linear order probe's
  p̂(o|D) at last.out(q) (context-level, as in E2) inside w = p(k|o,D)·p̂(o|D); through-origin slope on G > 0.
- 2026-09-04 18:50 (E4): the per-seed sign conjunct is the REGISTERED sign (every model seed's own T1 > 0), as
  mech_gates.gate does, not mere agreement.
- 2026-09-04 19:10 (E5): the pipeline checks live at campaigns/mech_int_20260905/checks/test_mech_probe.py, not tests/ (the sealed set); same three checks.
