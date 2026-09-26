import numpy as np


def permute_within_groups(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break continuation pairing with independent uniform permutations per group.

    Uniform permutations include fixed points. Excluding them creates a negative
    covariance after within-group centering and is therefore not a valid zero canary.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite groups by continuations matrix")
    out = np.empty_like(x)
    for group in range(x.shape[0]):
        out[group] = x[group, rng.permutation(x.shape[1])]
    return out


def permutation_null_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    out = np.empty(n_permutations, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != x.shape:
            raise ValueError("mask shape must match x")
    else:
        mask_array = None
    for index in range(n_permutations):
        if mask_array is None:
            permuted = permute_within_groups(x, rng)
        else:
            permuted = x.copy()
            for group in range(x.shape[0]):
                keep = np.flatnonzero(mask_array[group])
                permuted[group, keep] = x[group, keep[rng.permutation(len(keep))]]
        out[index] = within_group_slope(permuted, y, mask_array)
    return out


def _validated(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 3 or y.shape[1:] != x.shape:
        raise ValueError("x must be (groups, continuations) and y (seeds, groups, continuations)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("mask shape must match x")
    if np.any(mask.sum(axis=1) < 2):
        raise ValueError("every retained group needs at least two continuations")
    return x, y, mask


def within_group_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    x, y, mask = _validated(x, y, mask)
    numerator_by_seed_group = np.empty((y.shape[0], x.shape[0]), dtype=np.float64)
    denominator_by_group = np.empty(x.shape[0], dtype=np.float64)
    for group in range(x.shape[0]):
        keep = mask[group]
        centered_x = x[group, keep] - np.mean(x[group, keep])
        centered_y = y[:, group, keep] - np.mean(y[:, group, keep], axis=1, keepdims=True)
        numerator_by_seed_group[:, group] = np.sum(centered_y * centered_x[None, :], axis=1)
        denominator_by_group[group] = np.sum(centered_x * centered_x)
    numerator = float(np.sum(numerator_by_seed_group))
    denominator = float(y.shape[0] * np.sum(denominator_by_group))
    if denominator <= 1e-30:
        raise ValueError("within-group exact response has zero variation")
    return numerator / denominator


def crossed_resample_weights(
    *,
    n_boot: int,
    seeds: int,
    groups: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_boot < 1 or seeds < 1 or groups < 1:
        raise ValueError("bootstrap dimensions must be positive")
    seed_weights = rng.multinomial(seeds, np.full(seeds, 1.0 / seeds), size=n_boot)
    group_weights = rng.multinomial(groups, np.full(groups, 1.0 / groups), size=n_boot)
    return seed_weights, group_weights


def crossed_bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    resample_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    x, y, mask = _validated(x, y, mask)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    seeds, groups, _ = y.shape
    numerator = np.empty((seeds, groups), dtype=np.float64)
    denominator = np.empty(groups, dtype=np.float64)
    for group in range(groups):
        keep = mask[group]
        centered_x = x[group, keep] - np.mean(x[group, keep])
        centered_y = y[:, group, keep] - np.mean(y[:, group, keep], axis=1, keepdims=True)
        numerator[:, group] = np.sum(centered_y * centered_x[None, :], axis=1)
        denominator[group] = np.sum(centered_x * centered_x)
    if resample_weights is None:
        seed_weights, group_weights = crossed_resample_weights(
            n_boot=n_boot, seeds=seeds, groups=groups, rng=rng
        )
    else:
        seed_weights, group_weights = resample_weights
        if seed_weights.shape != (n_boot, seeds) or group_weights.shape != (n_boot, groups):
            raise ValueError("crossed bootstrap resample weights have the wrong shape")
    boot_numerator = np.einsum("bs,sg,bg->b", seed_weights, numerator, group_weights)
    boot_denominator = seeds * (group_weights @ denominator)
    if np.any(boot_denominator <= 1e-30):
        raise FloatingPointError("bootstrap produced zero exact-response variation")
    return boot_numerator / boot_denominator
