from dataclasses import dataclass

import numpy as np

from .constants import (
    AL40_SKEW,
    A_VALID_HI,
    A_VALID_LO,
    B_VALID_HI,
    B_VALID_LO,
    RHO_MAG_HI,
    RHO_MAG_LO,
    SIGMA_HI,
    SIGMA_LO,
)


@dataclass(frozen=True)
class SEMParameters:
    beta: float
    b_root: float
    b_effect: float


@dataclass(frozen=True)
class ContextGroup:
    sigma: np.ndarray
    graph: int
    params: SEMParameters
    core: np.ndarray
    reference: np.ndarray
    continuations: np.ndarray


def build_sigma(sigma1: float, sigma2: float, rho: float) -> np.ndarray:
    return np.array(
        [
            [sigma1 * sigma1, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2 * sigma2],
        ],
        dtype=np.float64,
    )


def sigma_to_params(sigma: np.ndarray, graph: int) -> SEMParameters:
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (2, 2) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be a finite 2 by 2 matrix")
    sxx, sxy, syy = sigma[0, 0], sigma[0, 1], sigma[1, 1]
    det = sxx * syy - sxy * sxy
    if sxx <= 0 or syy <= 0 or det <= 0:
        raise ValueError("sigma must be positive definite")
    if graph == 1:  # G1: x -> y
        return SEMParameters(
            beta=float(sxy / sxx),
            b_root=float(np.sqrt(sxx / 2.0)),
            b_effect=float(np.sqrt(det / (2.0 * sxx))),
        )
    if graph == 0:  # G2: y -> x
        return SEMParameters(
            beta=float(sxy / syy),
            b_root=float(np.sqrt(syy / 2.0)),
            b_effect=float(np.sqrt(det / (2.0 * syy))),
        )
    raise ValueError("graph must be 0 (y to x) or 1 (x to y)")


def valid_sigma(sigma: np.ndarray) -> bool:
    try:
        params = (sigma_to_params(sigma, 0), sigma_to_params(sigma, 1))
    except ValueError:
        return False
    return all(
        A_VALID_LO <= p.beta <= A_VALID_HI
        and B_VALID_LO <= p.b_root <= B_VALID_HI
        and B_VALID_LO <= p.b_effect <= B_VALID_HI
        for p in params
    )


def sample_valid_sigma(rng: np.random.Generator, max_attempts: int = 100_000) -> np.ndarray:
    for _ in range(max_attempts):
        sigma1 = float(np.exp(rng.uniform(np.log(SIGMA_LO), np.log(SIGMA_HI))))
        sigma2 = float(np.exp(rng.uniform(np.log(SIGMA_LO), np.log(SIGMA_HI))))
        magnitude = float(rng.uniform(RHO_MAG_LO, RHO_MAG_HI))
        rho = magnitude if int(rng.integers(0, 2)) else -magnitude
        sigma = build_sigma(sigma1, sigma2, rho)
        if valid_sigma(sigma):
            return sigma
    raise RuntimeError("failed to sample a valid covariance within max_attempts")


def al_sample(
    rng: np.random.Generator, scale: float, size: int | tuple[int, ...], skew: float = AL40_SKEW
) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(skew) or skew <= 0:
        raise ValueError("scale and skew must be positive and finite")
    c = np.sqrt(2.0 * scale * scale / (1.0 + skew * skew))
    alpha = skew * c
    return rng.exponential(alpha, size) - rng.exponential(c, size) - (alpha - c)


def sample_context(
    rng: np.random.Generator,
    graph: int,
    params: SEMParameters,
    n_rows: int,
    skew: float = AL40_SKEW,
) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    root = al_sample(rng, params.b_root, n_rows, skew)
    effect = al_sample(rng, params.b_effect, n_rows, skew)
    if graph == 1:
        x = root
        y = params.beta * root + effect
    elif graph == 0:
        y = root
        x = params.beta * root + effect
    else:
        raise ValueError("graph must be 0 or 1")
    out = np.stack([x, y], axis=1).astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise FloatingPointError("generated a non-finite context")
    return out


def generate_group(
    rng: np.random.Generator,
    n_continuations: int = 8,
    core_rows: int = 20,
    block_rows: int = 10,
) -> ContextGroup:
    if n_continuations < 2:
        raise ValueError("repeated-continuation design requires at least two continuations")
    sigma = sample_valid_sigma(rng)
    graph = int(rng.integers(0, 2))
    params = sigma_to_params(sigma, graph)
    core = sample_context(rng, graph, params, core_rows)
    reference = sample_context(rng, graph, params, block_rows)
    continuations = np.stack(
        [sample_context(rng, graph, params, block_rows) for _ in range(n_continuations)], axis=0
    )
    return ContextGroup(
        sigma=sigma,
        graph=graph,
        params=params,
        core=core,
        reference=reference,
        continuations=continuations,
    )

