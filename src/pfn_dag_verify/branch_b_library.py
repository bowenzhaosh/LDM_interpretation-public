"""Branch B: exact finite-prior causal wind tunnel — atom library.

Generates finite libraries of K valid covariance atoms from the frozen d=4
continuous prior. Each library is a set of (4,4) sigma matrices drawn by
accept-reject from the exact frozen prior (uniform in log_sd, rho, sign,
intersected with PD + validity). The training/evaluation prior is uniform over
these K atoms — an EXACT finite prior for which the exact posterior and
predictive can be computed by enumeration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .pilot_shared import (
    fleet,
    D_DIM,
    N_ORDERINGS,
    N_BINS,
    N_SIGN,
    LOG_SD_LO,
    LOG_SD_HI,
    RHO_LO,
    RHO_HI,
    sha256_file,
    sha256_array,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_ROOT = 888_000_000  # Branch B namespace, disjoint from all frozen seeds


def generate_library(
    k: int,
    seed: int,
    *,
    save_path: Path | None = None,
) -> np.ndarray:
    """Generate K valid covariance atoms by accept-reject from the frozen prior.

    Returns (K, 4, 4) float64 sigmas. The prior is uniform over these K atoms.
    """
    rng = np.random.default_rng(seed)
    lib = []
    total = 0
    # Measured acceptance rate ~7.3%; oversample by ~20×
    batch = max(k * 20, 4096)
    while len(lib) < k:
        log_sd = rng.uniform(LOG_SD_LO, LOG_SD_HI, (batch, D_DIM))
        rho_mag = rng.uniform(RHO_LO, RHO_HI, (batch, N_SIGN))
        sign = rng.choice([-1.0, 1.0], (batch, N_SIGN))
        S = np.empty((batch, D_DIM, D_DIM), dtype=np.float64)
        k_ij = 0
        for i in range(D_DIM):
            for j in range(i + 1, D_DIM):
                S[:, i, j] = S[:, j, i] = sign[:, k_ij] * rho_mag[:, k_ij] * np.exp(log_sd[:, i] + log_sd[:, j])
                k_ij += 1
        S[:, np.arange(D_DIM), np.arange(D_DIM)] = np.exp(2 * log_sd)
        ev = np.linalg.eigvalsh(S)
        pd = ev[:, 0] > 1e-6
        try:
            from .pilot_shared import _validity_batch
            valid = _validity_batch(S[pd])
        except Exception:
            continue
        keep = S[pd][valid]
        total += batch
        if len(lib) + len(keep) > k:
            keep = keep[:k - len(lib)]
        lib.append(keep)
    sigma = np.concatenate(lib, axis=0)[:k]
    if save_path is not None:
        np.savez(save_path, sigmas=sigma, seed=np.array([seed], dtype=np.int64),
                 k=np.array([k], dtype=np.int64))
    return sigma


def run_generation(k: int, seed: int, out: Path) -> dict[str, Any]:
    sigma = generate_library(k, seed, save_path=out)
    sha = sha256_array(sigma)
    return {
        "k": k,
        "seed": seed,
        "shape": list(sigma.shape),
        "dtype": str(sigma.dtype),
        "sha256": sha,
        "path": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_ROOT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    result = run_generation(args.k, args.seed, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
