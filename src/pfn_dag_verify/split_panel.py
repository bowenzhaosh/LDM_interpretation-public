"""Amendment F.2d.5 — the split panel, and the only place a half is defined.

Per cell, draw ``2n`` iid contexts from the world prior and split them into two
FIXED DISJOINT halves. The entropy deficit (the gate's x) is computed on half A
only; the model, floor and ceiling readouts and the regret (the gate's y and
gate (a)) are computed on half B only. x and y are never computed from shared
draws.

The reason is not fastidiousness. Both quantities are functions of the same
contexts, and a context that happens to be unusually informative pushes the
deficit and the readout the same way. Sharing draws would manufacture exactly
the positive association gate (c) is testing for, and it would do it invisibly:
nothing in the resulting numbers would look wrong.

The assignment is seeded and RECORDED, not re-derived at read time. A split that
has to be recomputed to be checked is a split that can silently change when the
code around it changes.

Both halves of a cell report the SAME ``panel_sha256`` -- the hash of the full
``2n`` draw -- because they came from one panel. ``cells_from_raw`` checks that,
which is what stops half A of one panel being paired with half B of another.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .corrected_models import make_eval_panel_contexts

# Fixed by this amendment. The split is a property of the preregistration, not a
# per-run choice; passing a different one is possible and is recorded in the
# artifact, but the headline panels use this.
SPLIT_SEED = 880_000_000
HALF_A = "A"    # the deficit / primary x-axis
HALF_B = "B"    # the readouts (model, floor, ceiling) and the regret


def contexts_sha256(contexts) -> str:
    arr = np.ascontiguousarray(np.stack(contexts), dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class SplitPanel:
    """One cell's panel, already split. ``index_a``/``index_b`` are positions in
    the full 2n draw and are what gets written to disk."""

    eps: float
    n_rows: int
    n_per_half: int
    panel_seed: int
    split_seed: int
    n_query_per_context: int
    contexts: list          # the full 2n draw, in draw order
    index_a: list[int]
    index_b: list[int]
    panel_sha256: str
    sha256_a: str
    sha256_b: str

    def half(self, which: str) -> list:
        if which == HALF_A:
            return [self.contexts[i] for i in self.index_a]
        if which == HALF_B:
            return [self.contexts[i] for i in self.index_b]
        raise ValueError(f"half must be {HALF_A!r} or {HALF_B!r}, got {which!r}")

    def provenance(self, which: str) -> dict:
        """The block every artifact built on this half carries."""
        return {
            "half": which,
            "n_contexts": len(self.half(which)),
            "n_per_half": self.n_per_half,
            "n_rows": self.n_rows,
            "panel_seed": self.panel_seed,
            "split_seed": self.split_seed,
            "n_query_per_context": self.n_query_per_context,
            "panel_sha256": self.panel_sha256,
            "half_sha256": self.sha256_a if which == HALF_A else self.sha256_b,
            "index": list(self.index_a if which == HALF_A else self.index_b),
        }


def build_split_panel(world, n_per_half: int, n_rows: int, panel_seed: int,
                      split_seed: int = SPLIT_SEED,
                      n_query_per_context: int = 2) -> SplitPanel:
    """Draw 2 * n_per_half iid contexts and split them.

    ``make_eval_panel_contexts`` deduplicates the query multiplicity, so the
    returned list is DISTINCT contexts and the halves are disjoint in contexts
    rather than merely in panel rows -- which is the property F.2d.5 needs. A
    query-level split would leave two queries of one context on opposite sides
    and reintroduce exactly the dependence the clause removes.
    """
    if n_per_half < 2:
        raise ValueError("each half needs at least 2 contexts to carry an SE")
    total = 2 * n_per_half
    contexts = make_eval_panel_contexts(world, total, n_rows, seed=panel_seed,
                                        n_query_per_context=n_query_per_context)
    if len(contexts) != total:
        raise ValueError(
            f"asked for {total} distinct contexts, got {len(contexts)}; the "
            "panel builder deduplicated differently than F.2d.5 assumes")
    perm = np.random.default_rng(split_seed).permutation(total)
    ia = sorted(int(i) for i in perm[:n_per_half])
    ib = sorted(int(i) for i in perm[n_per_half:])
    assert not (set(ia) & set(ib)), "halves overlap"
    ca = [contexts[i] for i in ia]
    cb = [contexts[i] for i in ib]
    return SplitPanel(
        eps=float(world.spec.eps), n_rows=int(n_rows), n_per_half=int(n_per_half),
        panel_seed=int(panel_seed), split_seed=int(split_seed),
        n_query_per_context=int(n_query_per_context),
        contexts=list(contexts), index_a=ia, index_b=ib,
        panel_sha256=contexts_sha256(contexts),
        sha256_a=contexts_sha256(ca), sha256_b=contexts_sha256(cb))


def half_panel_entries(panel, split: SplitPanel, which: str = HALF_B) -> list:
    """The ``make_eval_panel`` entries belonging to one half, checked by VALUE.

    ``make_eval_panel`` appends the same context once per query, so entry ``i``
    belongs to distinct context ``i // n_query_per_context``. That is an
    assumption about the generator's loop order, not a guarantee, and it is the
    kind of assumption that stays true until someone reorders a loop and then
    silently scores half A's contexts under half B's label. So the mapping is
    used AND verified here: every selected entry's context must equal the
    half's context it was matched to.
    """
    import numpy as _np

    q = split.n_query_per_context
    if len(panel) != q * len(split.contexts):
        raise ValueError(
            f"panel has {len(panel)} entries for {len(split.contexts)} contexts "
            f"at {q} queries each; expected {q * len(split.contexts)}")
    keep = split.index_a if which == HALF_A else split.index_b
    keep_set = set(keep)
    out = []
    for i, entry in enumerate(panel):
        ci = i // q
        if ci not in keep_set:
            continue
        if not _np.array_equal(entry[0], split.contexts[ci]):
            raise ValueError(
                f"panel entry {i} does not carry context {ci}: the eval panel "
                "and the split panel disagree on ordering, so a half-B score "
                "would be computed on half-A contexts")
        out.append(entry)
    if len(out) != q * len(keep):
        raise ValueError(f"selected {len(out)} entries, expected {q * len(keep)}")
    return out
