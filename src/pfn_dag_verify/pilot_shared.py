"""Original quadrature, with the frozen bin edges supplied locally."""
from types import SimpleNamespace
import hashlib
import json
import numpy as np
N_BINS = 100

def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()

def fleet():
    # The original fleet provided these same frozen 101 bin edges.
    return SimpleNamespace(BIN_EDGES=np.linspace(-8, 8, N_BINS + 1))

def production_quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen production 100-bin quadrature: 32 interior / 128 tail nodes."""
    f = fleet()
    edges = np.asarray(f.BIN_EDGES, dtype=np.float64)[1:-1]
    xi, wi = np.polynomial.legendre.leggauss(32)
    xt, wt = np.polynomial.legendre.leggauss(128)
    values, weights, bins = [], [], []
    u = (xt + 1.0) / 2.0
    tail_w = (wt / 2.0) / ((1.0 - u) ** 2)
    values.append(edges[0] - u / (1.0 - u)); weights.append(tail_w)
    bins.append(np.zeros(128, dtype=np.int64))
    for b in range(1, N_BINS - 1):
        left, right = edges[b - 1], edges[b]
        values.append((right - left) * xi / 2.0 + (right + left) / 2.0)
        weights.append(wi * (right - left) / 2.0)
        bins.append(np.full(32, b, dtype=np.int64))
    values.append(edges[-1] + u / (1.0 - u)); weights.append(tail_w)
    bins.append(np.full(128, N_BINS - 1, dtype=np.int64))
    v = np.concatenate(values).astype(np.float64)
    w = np.concatenate(weights).astype(np.float64)
    b = np.concatenate(bins).astype(np.int64)
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise RuntimeError("production quadrature invalid")
    return v, b, np.log(w)
