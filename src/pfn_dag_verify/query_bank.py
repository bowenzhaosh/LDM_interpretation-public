from dataclasses import dataclass

import numpy as np

from .instrument import js_divergence


CANDIDATE_QUERIES = np.arange(-4.0, 4.0 + 0.25, 0.5, dtype=np.float64)
FIXED_SENSITIVITY_BANK = np.array(
    [-3.75, -2.25, -1.25, -0.25, 0.25, 1.25, 2.25, 3.75], dtype=np.float64
)


@dataclass(frozen=True)
class QuerySelection:
    queries: np.ndarray
    objective_trace: np.ndarray
    identifiable_fraction: float


def select_symmetric_query_bank(
    f0: np.ndarray,
    f1: np.ndarray,
    candidate_queries: np.ndarray = CANDIDATE_QUERIES,
    *,
    pairs: int = 4,
) -> QuerySelection:
    """Greedily maximize the 10th percentile cumulative endpoint distance."""
    f0 = np.asarray(f0, dtype=np.float64)
    f1 = np.asarray(f1, dtype=np.float64)
    candidates = np.asarray(candidate_queries, dtype=np.float64)
    if f0.shape != f1.shape or f0.ndim != 3 or f0.shape[1] != len(candidates):
        raise ValueError("endpoints must be contexts by candidate queries by bins")
    if not np.allclose(candidates, -candidates[::-1], atol=0, rtol=0):
        raise ValueError("candidate grid must be exactly symmetric")
    magnitudes = sorted(float(x) for x in candidates if x > 0)
    selected: list[float] = []
    trace: list[float] = []
    distance = np.sum((f1 - f0) ** 2, axis=2)
    for _ in range(pairs):
        best = None
        for magnitude in magnitudes:
            if magnitude in selected:
                continue
            trial = selected + [magnitude]
            indices = []
            for value in trial:
                indices.extend(
                    [
                        int(np.flatnonzero(candidates == -value)[0]),
                        int(np.flatnonzero(candidates == value)[0]),
                    ]
                )
            objective = float(np.quantile(np.sum(distance[:, indices], axis=1), 0.10))
            candidate = (objective, -magnitude, magnitude)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise RuntimeError("not enough candidate query pairs")
        selected.append(best[2])
        trace.append(best[0])
    bank = np.array(sorted([-x for x in selected] + selected), dtype=np.float64)
    bank_indices = [int(np.flatnonzero(candidates == value)[0]) for value in bank]
    identifiable = np.array(
        [js_divergence(f0[index, bank_indices], f1[index, bank_indices]) for index in range(len(f0))]
    )
    return QuerySelection(
        queries=bank,
        objective_trace=np.asarray(trace, dtype=np.float64),
        identifiable_fraction=float(np.mean(identifiable >= 0.1)),
    )
