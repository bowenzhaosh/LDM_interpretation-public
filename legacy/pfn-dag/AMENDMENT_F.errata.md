# Amendment F — errata

Pinned to `AMENDMENT_F.md` sha256 `3d0e0f0af86c99601be8d99b34f778073fdd5fee9d8ef2b9a547cfe0eb8d8f98`, HEAD at lock
`b05e32093d93545b3d54c4ae66a4755595348e44`. Opened 2026-08-26.

**The locked document is not modified by anything here, and must not be.** The
digest is the lock. Every entry below corrects the *reading* of a statement in the
locked text, or records a fact the locked text asserts incorrectly. Where an entry
records a construction the locked text left open, it says so explicitly, so that a
later reader cannot mistake it for a free post-hoc choice.

Authority: nine items were adjudicated on 2026-08-26 — three by the repo owner
(B2, B1, and the handling of entry 2) and seven by a clean-context adjudication
working from the locked text and the code alone, with no y in existence. Full
reasoning in `.claude/W2_VERIFICATION_20260826.md` and
`.claude/FIDELITY_DECISIONS.md`.

---

## 1 — `panel_sha256` is a build fingerprint, not a draw fingerprint

`split_panel.py:39-42` hashes raw float64 bytes of a LAPACK/ULP-dependent
quantity. Four hashes are on record for the SAME eps=0.5 draw: `35f715a6`
(manifest + `deficit_w2/`), `45d7d02d` (WashU `floor_calib`), `cedec878`
(bo-computer), `3ce03717` (W2). Half A recomputes to relative difference 0.0 and
half B to 8-9 significant figures on independently rebuilt panels, so this is a
false positive of a bitwise identity check applied to a numerically reproducible
quantity. F.2d.5's "seeded and recorded" clause is implemented as a stricter check
than its text states, and is not satisfiable across venues as written.

Adjudicated: `deficit_w2/` is NOT regenerated. The verdict assembles from the
W2-panel-consistent set. **Consequence, recorded rather than worked around: the
primary x-axis is therefore absent from the locked verdict run and the primary
association reports not-evaluable.**

## 2 — The locked document states that it is not locked

At this digest, `AMENDMENT_F.md` carries "STATUS: NOT LOCKED" / "AWAITING BO'S
COUNTER-SIGNATURE" at lines 3, 328 and 475 — two of them inside operative clause
bodies — and line 479's stop-and-report block still says the lock and W2's launch
are blocked, and lists the F.2d.4 conjunction as "measured but not adopted".

All four were true when written and are false at this digest. The conjunction WAS
adopted (`corrected_verdict.py:799-800`; F.9.3) and re-validated at 40,000
trials/row before any digest was taken; W2 ran after the lock.

Cause: `scripts/lock_amendment.py` templates only the ten `__D_*__`/`__HEAD__`
tokens, so no check could catch stale prose, and F.9 was appended without reading
the body for contradictions. The text stands; this entry corrects the reading.

## 3 — F.1/E.3's exact-expectation estimand was never implemented

F.1 registers "the bin-summed expected NLL, not the NLL of a single sampled
outcome" and names the scorer to replace: the single-sampled-outcome path in
`corrected_models.evaluate_pfn_checkpoint`. It was never replaced. At this
digest, lines 433-435 still score one drawn bin (`p_pfn[ob]`), and that path
produced every number in W2's recorded regret table.

Gate (a)'s BRANCH_3 verdict **stands as taken under the locked instrument**.

Adjudicated (Bo, 2026-08-26): a NEW scorer
`scripts/f1_exact_expectation_rescore.py` reports the E.3 series beside the
recorded one; the locked tree is unmodified and its nine F.0 digests still verify.
**The ordering is disclosed and is not favourable:** the decision to compute the
registered estimand postdates knowledge both that the recorded series fails gate
(a) at eps=1.0 and that a rescored seed-0 cell clears it. A reader should weigh
the second series accordingly.

## 4 — The eps=0 ceiling is -0.4539, not 0

F.2c consequence 1 ("the identity makes that exact") and
`corrected_verdict.py:948-949` ("identically zero there for ANY model") are
descriptively false of the instrument. Both curves receive the same
degenerate-vertex reading at r ~ -0.4539: a 24-context diagnostic probe measures
floor -0.45391266, ceiling -0.45391251, gain 1.474e-7.

The eps=0 **exit stands**, on the measured gain against `CALIB_MIN_GAIN = 1e-6`
(margin 6.8x), re-measured at n=500 by the fidelity pass. What is retracted is the
stated *mechanism*: gate (b) is a finite-margin empirical gate, not a tautology.

## 5 — Gate (b) falsifiability

The locked text never instructs producing the eps=0 fidelity triple. Without it,
gate (b) takes an else-branch and cannot fail — the exact pathology F.2c's rewrite
existed to remove. The pass therefore produces the triple so the gain is measured.

Disclosed: `classify` returns at a gate-(a) failure before gate (b) is evaluated
(`corrected_verdict.py:929-939`), so under BRANCH_3 the measured eps=0 gain enters
the record through survivorship (F.2d.7) rather than through gate (b).

## 6 — F.2's panel sizing under-projected the realized SE

Realized worst-cell panel SE is 0.01494 against the 0.00824 that 1/sqrt(n) scaling
from `trackb_eps1p0_corrected.json` predicted — 1.81x — because the in-task
per-query SD nearly doubled (0.150 -> 0.277). F.2 sized the panel on the SE of an
artifact whose POINT ESTIMATES Amendment E retired; F.7's tripwire pins retired
constants but not the SEs printed beside them.

Gate (a)'s eps=1.0 failure is on interval width (CI upper +0.07663), not on the
point estimate (0.04734, below the 0.05 gate). F.2's own rule stands and is not
set aside here: it is the GATE that gets re-examined, not `n`.

## 7 — Common random numbers across the interior cells

`sample_residuals` (`corrected_sem.py:398-421`) consumes 1 / 4 / 2 RNG draws for
eps<=0 / interior / eps>=1, so the five interior cells share common random numbers
(context agreement 0.83 at delta-eps = 0.10) while eps=0 and eps=1.0 are
independent of them. Point estimates are unbiased; cross-cell variances are
mis-sized. F.2d.1's slope SE on an all-interior surviving set can be
ANTI-CONSERVATIVE; F.2d.4's extreme-cell contrast against eps=1.0 is correctly
sized. The series is a cross-world dose series — each eps is its own world and its
own panel — not a within-panel dose response.

## 8 — `js_mean` mislabelled in the working log

`WORKLOG.md` placed the retired mixed-target `js_mean` under a heading reading
"in-task, half B". 97.2-99.9% of its level and 91% of its 0.0665 -> 0.0976 rise
are out-of-task queries. The log is corrected in place by a later appended entry;
the artifact stands and is not rewritten.

## 9 — Constructions the locked text left open (recorded, not chosen freely)

The locked text names neither 100k checkpoint file and never says how seeds
combine. Registered here so that neither is later mistaken for a post-hoc choice:

- **The trained PFN** is the 100k state, read from
  `nets/eps{tag}/base_s{seed}_ck100000.pt` with `base_s{seed}.pt` asserted
  tensor-equal before use. Intermediate checkpoints (10k/25k/50k) carry **no y**;
  a per-checkpoint curve is a new, unregistered estimand, DIAGNOSTIC-only, and may
  never be named `fidelity_*` nor called y or fraction-of-achievable.
- **`model_ctx[i]`** is the per-context arithmetic mean over seeds {0,1,2}. The
  floor is one seed and the ceiling is model-free, so the denominator is common
  and the point estimate is provably identical to averaging three y's; the choice
  moves only SE_y. F.2d.6 assigns the gate's SE to the context level explicitly
  and demotes the across-seed spread to a diagnostic, and the digested docstring
  (`corrected_verdict.py:182-184`) says exactly this. Resolved-by-code.
