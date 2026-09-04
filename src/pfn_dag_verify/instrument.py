from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, xlogy


class UnidentifiableError(ValueError):
    pass


@dataclass(frozen=True)
class Projection:
    raw_w: float
    w: float
    g: float
    normalized_residual: float
    boundary: bool
    separation_squared: float


@dataclass(frozen=True)
class Reconstruction:
    updated_w: float
    prediction: np.ndarray
    normalized_residual: float


@dataclass(frozen=True)
class BatchProjection:
    raw_w: np.ndarray
    w: np.ndarray
    g: np.ndarray
    normalized_residual: np.ndarray
    boundary: np.ndarray
    separation_squared: np.ndarray


def _validated_triplet(p: np.ndarray, f0: np.ndarray, f1: np.ndarray):
    arrays = [np.asarray(v, dtype=np.float64) for v in (p, f0, f1)]
    if any(v.ndim != 2 for v in arrays):
        raise ValueError("prediction and endpoints must be query by bin matrices")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("prediction and endpoint shapes must match exactly")
    for name, value in zip(("prediction", "f0", "f1"), arrays):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        if not np.allclose(value.sum(axis=1), 1.0, atol=1e-8, rtol=0):
            raise ValueError(f"{name} rows must sum to one")
    return arrays


def _logit(w: float, eps: float = 1e-12) -> float:
    value = float(np.clip(w, eps, 1.0 - eps))
    return float(np.log(value) - np.log1p(-value))


def _residual(p: np.ndarray, mix: np.ndarray, delta: np.ndarray) -> float:
    denominator = float(np.linalg.norm(delta.ravel()))
    if denominator <= 1e-15:
        raise UnidentifiableError("endpoint segment has negligible separation")
    return float(np.linalg.norm((p - mix).ravel()) / denominator)


def project_coordinate(p: np.ndarray, f0: np.ndarray, f1: np.ndarray) -> Projection:
    p, f0, f1 = _validated_triplet(p, f0, f1)
    delta = f1 - f0
    separation_squared = float(np.dot(delta.ravel(), delta.ravel()))
    if separation_squared <= 1e-30:
        raise UnidentifiableError("endpoint segment has negligible separation")
    raw_w = float(np.dot((p - f0).ravel(), delta.ravel()) / separation_squared)
    w = float(np.clip(raw_w, 0.0, 1.0))
    mix = (1.0 - w) * f0 + w * f1
    return Projection(
        raw_w=raw_w,
        w=w,
        g=_logit(w),
        normalized_residual=_residual(p, mix, delta),
        boundary=bool(w <= 1e-6 or w >= 1.0 - 1e-6),
        separation_squared=separation_squared,
    )


def project_kl(p: np.ndarray, f0: np.ndarray, f1: np.ndarray) -> Projection:
    p, f0, f1 = _validated_triplet(p, f0, f1)
    delta = f1 - f0
    separation_squared = float(np.dot(delta.ravel(), delta.ravel()))
    if separation_squared <= 1e-30:
        raise UnidentifiableError("endpoint segment has negligible separation")

    def objective(w):
        mix = f0 + float(w) * delta
        return float(-np.sum(xlogy(p, np.clip(mix, np.finfo(np.float64).tiny, None))))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(0.0, 1.0),
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError("bounded KL projection failed")
    raw_w = float(result.x)
    w = float(np.clip(raw_w, 0.0, 1.0))
    mix = (1.0 - w) * f0 + w * f1
    return Projection(
        raw_w=raw_w,
        w=w,
        g=_logit(w),
        normalized_residual=_residual(p, mix, delta),
        boundary=bool(w <= 1e-6 or w >= 1.0 - 1e-6),
        separation_squared=separation_squared,
    )


def js_divergence(f0: np.ndarray, f1: np.ndarray) -> float:
    f0 = np.asarray(f0, dtype=np.float64)
    f1 = np.asarray(f1, dtype=np.float64)
    if f0.ndim != 2 or f0.shape != f1.shape:
        raise ValueError("endpoints must have identical query by bin shapes")
    _validated_triplet(f0, f0, f1)
    midpoint = 0.5 * (f0 + f1)
    js_per_query = 0.5 * np.sum(xlogy(f0, f0 / midpoint), axis=1)
    js_per_query += 0.5 * np.sum(xlogy(f1, f1 / midpoint), axis=1)
    value = float(np.mean(js_per_query))
    if not np.isfinite(value) or value < -1e-14:
        raise FloatingPointError("invalid Jensen-Shannon divergence")
    return max(0.0, value)


def reconstruct_updated_prediction(
    p_target: np.ndarray,
    f0_target: np.ndarray,
    f1_target: np.ndarray,
    *,
    w_base: float,
    delta_ell: float,
) -> Reconstruction:
    p_target, f0_target, f1_target = _validated_triplet(p_target, f0_target, f1_target)
    if not np.isfinite(w_base) or not 0.0 < w_base < 1.0:
        raise ValueError("base weight must be finite and interior")
    if not np.isfinite(delta_ell):
        raise ValueError("delta_ell must be finite")
    updated_w = float(expit(_logit(w_base) + delta_ell))
    prediction = (1.0 - updated_w) * f0_target + updated_w * f1_target
    residual = _residual(p_target, prediction, f1_target - f0_target)
    return Reconstruction(updated_w=updated_w, prediction=prediction, normalized_residual=residual)


def _validated_batch_triplet(p: np.ndarray, f0: np.ndarray, f1: np.ndarray):
    arrays = [np.asarray(value, dtype=np.float64) for value in (p, f0, f1)]
    if any(value.ndim != 3 for value in arrays):
        raise ValueError("batched predictions and endpoints must be context by query by bin")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("batched prediction and endpoint shapes must match")
    for name, value in zip(("prediction", "f0", "f1"), arrays):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        if not np.allclose(value.sum(axis=2), 1.0, atol=1e-6, rtol=0):
            raise ValueError(f"{name} rows must sum to one")
    return arrays


def _batch_projection_result(p, f0, f1, raw_w, w):
    delta = f1 - f0
    separation = np.sum(delta * delta, axis=(1, 2))
    if np.any(separation <= 1e-30):
        raise UnidentifiableError("at least one endpoint segment has negligible separation")
    mix = f0 + w[:, None, None] * delta
    residual = np.sqrt(np.sum((p - mix) ** 2, axis=(1, 2)) / separation)
    clipped = np.clip(w, 1e-12, 1.0 - 1e-12)
    return BatchProjection(
        raw_w=raw_w,
        w=w,
        g=np.log(clipped) - np.log1p(-clipped),
        normalized_residual=residual,
        boundary=(w <= 1e-6) | (w >= 1.0 - 1e-6),
        separation_squared=separation,
    )


def project_coordinate_batch(p: np.ndarray, f0: np.ndarray, f1: np.ndarray) -> BatchProjection:
    p, f0, f1 = _validated_batch_triplet(p, f0, f1)
    delta = f1 - f0
    separation = np.sum(delta * delta, axis=(1, 2))
    if np.any(separation <= 1e-30):
        raise UnidentifiableError("at least one endpoint segment has negligible separation")
    raw = np.sum((p - f0) * delta, axis=(1, 2)) / separation
    weight = np.clip(raw, 0.0, 1.0)
    return _batch_projection_result(p, f0, f1, raw, weight)


def project_kl_batch(
    p: np.ndarray, f0: np.ndarray, f1: np.ndarray, *, iterations: int = 40
) -> BatchProjection:
    """Independent convex KL projection using deterministic vectorized bisection."""
    p, f0, f1 = _validated_batch_triplet(p, f0, f1)
    if iterations < 20:
        raise ValueError("at least 20 bisection iterations are required")
    delta = f1 - f0
    separation = np.sum(delta * delta, axis=(1, 2))
    if np.any(separation <= 1e-30):
        raise UnidentifiableError("at least one endpoint segment has negligible separation")
    tiny = np.finfo(np.float64).tiny

    def gradient(weight):
        mix = np.clip(f0 + weight[:, None, None] * delta, tiny, None)
        return -np.sum(p * delta / mix, axis=(1, 2))

    zero = np.zeros(len(p), dtype=np.float64)
    one = np.ones(len(p), dtype=np.float64)
    gradient_zero = gradient(zero)
    gradient_one = gradient(one)
    at_zero = gradient_zero >= 0.0
    at_one = gradient_one <= 0.0
    interior = ~(at_zero | at_one)
    low = zero.copy()
    high = one.copy()
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        derivative = gradient(midpoint)
        move_low = (derivative < 0.0) & interior
        low = np.where(move_low, midpoint, low)
        high = np.where((~move_low) & interior, midpoint, high)
    weight = 0.5 * (low + high)
    weight[at_zero] = 0.0
    weight[at_one] = 1.0
    return _batch_projection_result(p, f0, f1, weight.copy(), weight)


def reconstruct_updated_batch(
    p_target: np.ndarray,
    f0_target: np.ndarray,
    f1_target: np.ndarray,
    *,
    w_base: np.ndarray,
    delta_ell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_target, f0_target, f1_target = _validated_batch_triplet(p_target, f0_target, f1_target)
    w_base = np.asarray(w_base, dtype=np.float64)
    delta_ell = np.asarray(delta_ell, dtype=np.float64)
    if w_base.shape != (len(p_target),) or delta_ell.shape != (len(p_target),):
        raise ValueError("base weights and evidence changes must match the context batch")
    if not np.isfinite(w_base).all() or np.any(w_base <= 0) or np.any(w_base >= 1):
        raise ValueError("all base weights must be finite and interior")
    if not np.isfinite(delta_ell).all():
        raise ValueError("all evidence changes must be finite")
    base_g = np.log(w_base) - np.log1p(-w_base)
    updated = expit(base_g + delta_ell)
    prediction = f0_target + updated[:, None, None] * (f1_target - f0_target)
    separation = np.sum((f1_target - f0_target) ** 2, axis=(1, 2))
    if np.any(separation <= 1e-30):
        raise UnidentifiableError("at least one target segment has negligible separation")
    residual = np.sqrt(np.sum((p_target - prediction) ** 2, axis=(1, 2)) / separation)
    return updated, residual
