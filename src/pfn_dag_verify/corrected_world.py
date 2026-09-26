"""Corrected tiny exact world (Track A2): d=3, K atoms.

The world is a finite prior over latent states z = (k, o): K covariance atoms
(correlations + scales drawn from the d=3 analog of the frozen d=4 prior,
accept-reject filtered by the frozen validity rule generalized to d=3) and
all 6 causal orders. The residual law is the epsilon mixture.

The atoms are deterministic for a given seed (registered namespace, disjoint
from Branch B's seeds), so golden fixtures are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .corrected_oracle import World
from .corrected_sem import (
    BETA_MAX,
    B_HI,
    B_LO,
    LOG_SD_HI,
    LOG_SD_LO,
    RHO_HI,
    RHO_LO,
    ResidualSpec,
    all_orderings,
    params_for,
)

# Registered seed namespace for corrected-world atom generation. Disjoint from
# Branch B seeds (889,xxx,xxx) and pilot seeds.
WORLD_SEED_ROOT = 990_000_000


def sample_valid_atoms(k: int, d: int, seed: int) -> np.ndarray:
    """Sample k valid (d, d) covariance atoms by accept-reject from the prior.

    Validity: for every causal order, the Cholesky parameters satisfy the
    frozen bounds (|beta| <= 1.5, b in [0.3, 1.3]).
    """
    rng = np.random.default_rng(seed)
    orderings = all_orderings(d)
    n_sign = d * (d - 1) // 2
    out: list[np.ndarray] = []
    batch = max(k * 40, 8192)
    while len(out) < k:
        log_sd = rng.uniform(LOG_SD_LO, LOG_SD_HI, (batch, d))
        rho_mag = rng.uniform(RHO_LO, RHO_HI, (batch, n_sign))
        sign = rng.choice([-1.0, 1.0], (batch, n_sign))
        S = np.zeros((batch, d, d), dtype=np.float64)
        ki = 0
        for i in range(d):
            for j in range(i + 1, d):
                S[:, i, j] = S[:, j, i] = (
                    sign[:, ki] * rho_mag[:, ki] * np.exp(log_sd[:, i] + log_sd[:, j]))
                ki += 1
        S[:, np.arange(d), np.arange(d)] = np.exp(2 * log_sd)
        ev = np.linalg.eigvalsh(S)
        keep = ev[:, 0] > 1e-6
        S_pd = S[keep]
        valid = np.ones(len(S_pd), dtype=bool)
        for pi in orderings:
            for idx in range(len(S_pd)):
                if not valid[idx]:
                    continue
                sp = params_for(S_pd[idx], pi)
                beta = -sp.U
                mask = np.tril(np.ones((d, d), dtype=bool), -1)
                amax = np.abs(beta[mask]).max()
                ok = (amax <= BETA_MAX) and np.all(sp.b >= B_LO) and np.all(sp.b <= B_HI)
                valid[idx] &= ok
        keep2 = S_pd[valid]
        if len(out) + len(keep2) > k:
            keep2 = keep2[: k - len(out)]
        out.append(keep2)
    return np.concatenate(out, axis=0)[:k]


def make_world(
    k: int,
    d: int = 3,
    eps: float = 0.0,
    seed: int | None = None,
) -> World:
    """Build a deterministic World (d, k, eps)."""
    if seed is None:
        seed = WORLD_SEED_ROOT + 1000 * d + k
    sigmas = sample_valid_atoms(k, d, seed)
    return World(sigmas=sigmas, spec=ResidualSpec(eps=eps))


def world_metadata(world: World, seed: int) -> dict:
    """Provenance metadata for a world: hashes, accepted-atom count, config.

    Records the exact accepted atom count (K), the library SHA-256, the source
    config (d, eps, r), and the seed. No blank SHA fields are permitted.
    """
    import hashlib
    import json
    from .pilot_shared import sha256_array

    sha = sha256_array(world.sigmas)
    config = {
        "d": world.d,
        "K": world.K,
        "O": world.O,
        "eps": world.spec.eps,
        "r": world.spec.r,
        "seed": seed,
        "orderings": [list(p) for p in world.orderings],
    }
    cfg_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "library_sha256": sha,
        "config_sha256": cfg_digest,
        "accepted_atom_count": int(world.K),
        "source": "corrected_world.sample_valid_atoms",
        "config": config,
    }



# -- tiny-world panel helpers ------------------------------------------------

def default_context(world: World, n_rows: int, seed: int) -> np.ndarray:
    """A fixed exact context: n_rows sampled from a randomly chosen (k, o)."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(world.K))
    o = int(rng.integers(world.O))
    from .corrected_sem import generate_observational
    return generate_observational(rng, world.sigmas[k], world.orderings[o],
                                  world.spec, n_rows)


def query_xq(world: World, seed: int, target: int = 2) -> np.ndarray:
    """A fixed observational query: observed covariates (d-1,) values."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, world.d - 1)

