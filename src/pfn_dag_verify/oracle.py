from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp

from .constants import (
    AL40_SKEW,
    BIN_CENTERS,
    RHO_MAG_HI,
    RHO_MAG_LO,
    SIGMA_HI,
    SIGMA_LO,
)
from .generative import build_sigma, sigma_to_params, valid_sigma
from .instrument import js_divergence


@dataclass(frozen=True)
class OracleBundle:
    ell: float
    f0: np.ndarray
    f1: np.ndarray
    js: float
    w_star: float


def _grid(quadrature: int):
    if quadrature < 3:
        raise ValueError("quadrature must be at least three")
    sigmas = np.exp(np.linspace(np.log(SIGMA_LO), np.log(SIGMA_HI), quadrature))
    rho = np.concatenate(
        [
            np.linspace(-RHO_MAG_HI, -RHO_MAG_LO, quadrature // 2),
            np.linspace(RHO_MAG_LO, RHO_MAG_HI, quadrature - quadrature // 2),
        ]
    )
    rows = []
    for sigma1 in sigmas:
        for sigma2 in sigmas:
            for correlation in rho:
                sigma = build_sigma(float(sigma1), float(sigma2), float(correlation))
                if valid_sigma(sigma):
                    g1 = sigma_to_params(sigma, 1)
                    g0 = sigma_to_params(sigma, 0)
                    rows.append(
                        (
                            g1.beta,
                            g1.b_root,
                            g1.b_effect,
                            g0.beta,
                            g0.b_root,
                            g0.b_effect,
                        )
                    )
    if not rows:
        raise RuntimeError("quadrature grid has no valid covariance points")
    return np.asarray(rows, dtype=np.float64)


def _al_logpdf(residual: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    c = np.sqrt(2.0 * scale * scale / (1.0 + AL40_SKEW * AL40_SKEW))
    alpha = AL40_SKEW * c
    shifted = residual + (alpha - c)
    return -np.log(alpha + c) + np.where(shifted >= 0.0, -shifted / alpha, shifted / c)


def _validate_context(context: np.ndarray) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 2 or context.shape[1] != 2 or context.shape[0] < 1:
        raise ValueError("context must have shape (rows, 2) with at least one row")
    if not np.isfinite(context).all():
        raise ValueError("context must be finite")
    return context


class GridOracle:
    """Vectorized Fix-B AL40 evidence and family-predictive oracle."""

    def __init__(self, queries: np.ndarray, quadrature: int = 15):
        self.queries = np.asarray(queries, dtype=np.float64)
        if self.queries.ndim != 1 or self.queries.size < 2 or not np.isfinite(self.queries).all():
            raise ValueError("queries must be a finite one-dimensional bank")
        self.quadrature = int(quadrature)
        self.params = _grid(self.quadrature)
        self.a1, self.b11, self.b12, self.a0, self.b01, self.b02 = self.params.T
        self._cp1, self._cp0 = self._precompute_endpoint_logpmfs()

    @property
    def grid_size(self) -> int:
        return int(self.params.shape[0])

    def _precompute_endpoint_logpmfs(self):
        grid = self.grid_size
        query_count = self.queries.size
        cp1 = np.empty((grid, query_count, BIN_CENTERS.size), dtype=np.float64)
        cp0 = np.empty_like(cp1)
        for index, query in enumerate(self.queries):
            lp1 = _al_logpdf(
                BIN_CENTERS[None, :] - self.a1[:, None] * query,
                self.b12[:, None],
            )
            cp1[:, index, :] = lp1 - logsumexp(lp1, axis=1, keepdims=True)
            lp0 = _al_logpdf(BIN_CENTERS[None, :], self.b01[:, None])
            lp0 += _al_logpdf(
                query - self.a0[:, None] * BIN_CENTERS[None, :],
                self.b02[:, None],
            )
            cp0[:, index, :] = lp0 - logsumexp(lp0, axis=1, keepdims=True)
        return cp1, cp0

    def _context_loglik(self, context: np.ndarray):
        context = _validate_context(context)
        x = context[:, 0]
        y = context[:, 1]
        log_g1 = np.sum(_al_logpdf(x[None, :], self.b11[:, None]), axis=1)
        log_g1 += np.sum(
            _al_logpdf(y[None, :] - self.a1[:, None] * x[None, :], self.b12[:, None]),
            axis=1,
        )
        log_g0 = np.sum(_al_logpdf(y[None, :], self.b01[:, None]), axis=1)
        log_g0 += np.sum(
            _al_logpdf(x[None, :] - self.a0[:, None] * y[None, :], self.b02[:, None]),
            axis=1,
        )
        if not np.isfinite(log_g1).all() or not np.isfinite(log_g0).all():
            raise FloatingPointError("non-finite context likelihood")
        return log_g1, log_g0

    def evaluate(self, context: np.ndarray) -> OracleBundle:
        log_g1, log_g0 = self._context_loglik(context)
        marginal1 = float(logsumexp(log_g1))
        marginal0 = float(logsumexp(log_g0))
        ell = marginal1 - marginal0
        weight1 = np.exp(log_g1 - marginal1)
        weight0 = np.exp(log_g0 - marginal0)
        f1 = np.tensordot(weight1, np.exp(self._cp1), axes=(0, 0))
        f0 = np.tensordot(weight0, np.exp(self._cp0), axes=(0, 0))
        f1 /= f1.sum(axis=1, keepdims=True)
        f0 /= f0.sum(axis=1, keepdims=True)
        if not np.isfinite(ell) or not np.isfinite(f0).all() or not np.isfinite(f1).all():
            raise FloatingPointError("non-finite oracle output")
        return OracleBundle(
            ell=ell,
            f0=f0,
            f1=f1,
            js=js_divergence(f0, f1),
            w_star=float(expit(ell)),
        )

    def log_evidence(self, context: np.ndarray) -> float:
        log_g1, log_g0 = self._context_loglik(context)
        value = float(logsumexp(log_g1) - logsumexp(log_g0))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite log evidence")
        return value


class ScalarGridOracle:
    """Slow reference with scalar likelihood and independently built endpoint tables."""

    def __init__(self, queries: np.ndarray, quadrature: int = 15):
        self.queries = np.asarray(queries, dtype=np.float64)
        if self.queries.ndim != 1 or self.queries.size < 2 or not np.isfinite(self.queries).all():
            raise ValueError("queries must be a finite one-dimensional bank")
        self.quadrature = int(quadrature)
        self.params = _grid(self.quadrature)
        self._cp1, self._cp0 = self._build_scalar_endpoint_tables()

    @staticmethod
    def _single_logpdf(value: np.ndarray, scale: float):
        return _al_logpdf(np.asarray(value, dtype=np.float64), float(scale))

    def _build_scalar_endpoint_tables(self):
        cp1 = np.empty((len(self.params), len(self.queries), len(BIN_CENTERS)), dtype=np.float64)
        cp0 = np.empty_like(cp1)
        for gi, (a1, _b11, b12, a0, b01, b02) in enumerate(self.params):
            for qi, query in enumerate(self.queries):
                lp1 = self._single_logpdf(BIN_CENTERS - a1 * query, b12)
                cp1[gi, qi] = lp1 - logsumexp(lp1)
                lp0 = self._single_logpdf(BIN_CENTERS, b01)
                lp0 += self._single_logpdf(query - a0 * BIN_CENTERS, b02)
                cp0[gi, qi] = lp0 - logsumexp(lp0)
        return cp1, cp0

    def evaluate(self, context: np.ndarray) -> OracleBundle:
        context = _validate_context(context)
        x, y = context[:, 0], context[:, 1]
        log_g1 = np.empty(len(self.params), dtype=np.float64)
        log_g0 = np.empty_like(log_g1)
        for index, (a1, b11, b12, a0, b01, b02) in enumerate(self.params):
            log_g1[index] = np.sum(self._single_logpdf(x, b11))
            log_g1[index] += np.sum(self._single_logpdf(y - a1 * x, b12))
            log_g0[index] = np.sum(self._single_logpdf(y, b01))
            log_g0[index] += np.sum(self._single_logpdf(x - a0 * y, b02))
        marginal1 = float(logsumexp(log_g1))
        marginal0 = float(logsumexp(log_g0))
        ell = marginal1 - marginal0
        weight1 = np.exp(log_g1 - marginal1)
        weight0 = np.exp(log_g0 - marginal0)
        f1 = np.sum(weight1[:, None, None] * np.exp(self._cp1), axis=0)
        f0 = np.sum(weight0[:, None, None] * np.exp(self._cp0), axis=0)
        f1 /= f1.sum(axis=1, keepdims=True)
        f0 /= f0.sum(axis=1, keepdims=True)
        return OracleBundle(
            ell=ell,
            f0=f0,
            f1=f1,
            js=js_divergence(f0, f1),
            w_star=float(expit(ell)),
        )
