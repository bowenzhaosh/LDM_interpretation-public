"""Frozen generator-only d=4 substrate for Phase-1 calibration.

This file contains no model class, checkpoint loader, result reader, or
scientific analysis.  Its generator algebra is copied from the archived
`d4_train_fleet.cluster.py` substrate and is hash-pinned before import.
"""

import math
from itertools import permutations

import numpy as np


D_DIM = 4
ORDERINGS = list(permutations(range(D_DIM)))
N_CONTEXT, N_QUERY, N_BINS = 30, 7, 100
BIN_EDGES = np.linspace(-8, 8, N_BINS + 1)
NULL_TOK = 2
R_OF = {"A": 2.0, "C": 4.0}


def params_for(S, pi):
    S = np.asarray(S)
    n, d = len(S), S.shape[1]
    Spi = S[:, pi][:, :, pi]
    Lch = np.linalg.cholesky(Spi)
    diag = np.diagonal(Lch, axis1=1, axis2=2)
    Lunit = Lch / diag[:, None, :]
    Dv = diag**2
    U = np.linalg.inv(Lunit)
    b = np.sqrt(np.maximum(Dv, 1e-12) / 2.0)
    return Lunit, U, b


def validity_keep(S):
    keep = np.ones(len(S), bool)
    for pi in ORDERINGS:
        _, U, b = params_for(S, pi)
        beta = -U
        mask = np.tril(np.ones((D_DIM, D_DIM)), -1).astype(bool)
        amax = np.abs(beta[:, mask]).max(1)
        keep &= (amax <= 1.5) & (b >= 0.3).all(1) & (b <= 1.3).all(1)
    return keep


def sample_Sigmas(rng, n):
    out = []
    while len(out) < n:
        m = 8192
        sd = np.exp(rng.uniform(math.log(0.6), math.log(1.5), (m, D_DIM)))
        R = np.eye(D_DIM)[None].repeat(m, 0)
        iu = np.triu_indices(D_DIM, 1)
        rho = rng.choice([-1.0, 1.0], (m, len(iu[0]))) * rng.uniform(
            0.3, 0.8, (m, len(iu[0]))
        )
        R[:, iu[0], iu[1]] = rho
        R[:, iu[1], iu[0]] = rho
        S = sd[:, :, None] * R * sd[:, None, :]
        ev = np.linalg.eigvalsh(S)
        S = S[ev[:, 0] > 1e-6]
        S = S[validity_keep(S)]
        out.extend(S)
    return np.array(out[:n])


def al_ac(b, r):
    c = np.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def al_sample(b, r, size, rng):
    a, c = al_ac(b, r)
    return rng.exponential(a, size) - rng.exponential(c, size) - (a - c)


def gen_data(S1, fam, r, k, rng, gaussian=False):
    pi = ORDERINGS[fam]
    Lunit, _, b = params_for(S1[None], pi)
    Lunit, b = Lunit[0], b[0]
    if gaussian:
        e = rng.normal(0, np.sqrt(2.0) * b[None, :], (k, D_DIM))
    else:
        e = np.stack(
            [al_sample(np.full(k, b[m]), r, k, rng) for m in range(D_DIM)], 1
        )
    xpi = e @ Lunit.T
    x = np.empty_like(xpi)
    x[:, list(pi)] = xpi
    return x


def bin_y(v):
    return np.searchsorted(BIN_EDGES[1:-1], v)
