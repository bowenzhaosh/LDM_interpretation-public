"""Frozen scientific substrate for the oracle-precision pilot.

This module holds only frozen scientific definitions that BOTH pilot
estimators must agree on: the exact d=4 prior (native parameterization and its
log density), the validity/PD support indicators, the coordinate transforms,
the fleet loader, the seed registry, and guarded input/output helpers. It
contains no estimator implementation: no likelihood assembly, no proposal, no
evidence estimator, no predictive aggregation, and no decision code live here.
The SMC and AIS modules each implement those parts independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "artifacts" / "phase1" / "d4_generator.py"
GENERATOR_SHA256 = "1aa7652cad924c90f871309f860b9172e836898c5a1620c77fdd3196e70d291d"

# Native continuous-coordinate bounds (frozen prior box).
LOG_SD_LO, LOG_SD_HI = math.log(0.6), math.log(1.5)
RHO_LO, RHO_HI = 0.3, 0.8
D_DIM = 4
N_ORDERINGS = 24
N_BINS = 100
N_CONT = D_DIM + D_DIM * (D_DIM - 1) // 2  # 4 sd + 6 rho = 10 continuous
N_SIGN = D_DIM * (D_DIM - 1) // 2  # 6 sign bits

# Prior density value in native coords (uniform on the box, ignoring the
# validity/PD indicators which are tracked separately): the density is the
# product of the marginal densities of log_sd (uniform on [LOG_SD_LO, LOG_SD_HI]),
# rho_mag (uniform on [RHO_LO, RHO_HI]), and the Rademacher sign mass (1/2)^6.
NATIVE_LOG_DENSITY_CONST = (
    -4.0 * math.log(LOG_SD_HI - LOG_SD_LO)
    - 6.0 * math.log(RHO_HI - RHO_LO)
    - 6.0 * math.log(2.0)
)

# Normalizer of the frozen conditional prior. The prior is uniform on the box
# (in native coords) intersected with PD+validity; its normalizing constant is
# 1/P_valid. In z-coords the base density prod_j s(z_j)(1-s(z_j))*(1/2)^6
# integrates to P_valid over the valid subset, so the normalized log density is
# the base minus NEGLOG_P_VALID. Measured by MC on 2,000,000 raw draws with
# seed 886_500_000 (registered, not a panel draw). Frozen here.
P_VALID_MC = 0.076351
NEGLOG_P_VALID = 2.57242

# Reserved seed namespaces. All frozen production seeds and this pilot's
# development seeds are disjoint; the panel never shares a namespace with
# calibration. The forbidden seeds are the specific frozen values; the pilot
# seed roots (886/887 million) are in a distinct block. A pilot root within
# 1,000,000 of any forbidden seed is rejected.
PILOT_DEV_SEED_ROOT = 886_000_000
FORBIDDEN_SEED_NAMESPACES = (
    880_803_000,  # calibration seed root
    880_903_000, 880_923_000, 880_943_000, 880_953_000,  # qualification v1-v3
    881_003_000, 881_013_000,  # confirmation evaluation
    881_003_101, 881_003_102, 881_003_103,  # atom banks
    881_003_900,  # bootstrap master
    881_103_999,  # atom determinism canary
)

# Paths and import names that pilot code must never touch (outcome blindness).
FORBIDDEN_PATH_FRAGMENTS = (
    "nested_half_raw.npz",
    "confirmatory_raw.npz",
    "oracle_raw.npz",
    "oracle_half_raw.npz",
    "pfn_fleet_summary.json",
    "M4_",  # checkpoint file prefix
    "predictions",
    "phase1_checkpoint_registry.json",
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "phase1_pfn",
    "checkpoint",
    "torch.load",
    "phase1_join",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_fleet() -> Any:
    """Load the frozen d=4 generator, hash-pinned, and assert its contract."""
    import importlib.util

    if not GENERATOR.is_file():
        raise FileNotFoundError(GENERATOR)
    if sha256_file(GENERATOR) != GENERATOR_SHA256:
        raise RuntimeError("frozen generator module hash mismatch")
    spec = importlib.util.spec_from_file_location("phase1_frozen_d4_fleet", GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen fleet module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if int(module.D_DIM) != D_DIM or len(module.ORDERINGS) != N_ORDERINGS:
        raise RuntimeError("fleet module does not satisfy the d=4/24-order contract")
    if int(module.N_BINS) != N_BINS:
        raise RuntimeError("fleet module does not satisfy the 100-bin contract")
    return module


FLEET = None


def fleet() -> Any:
    global FLEET
    if FLEET is None:
        FLEET = load_fleet()
    return FLEET


# ---------------------------------------------------------------------------
# Coordinate transforms and parameter assembly
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return np.log(x) - np.log1p(-x)


def native_to_z(log_sd: np.ndarray, rho_mag: np.ndarray) -> np.ndarray:
    """Map native continuous coords (log_sd, rho_mag) to unbounded z.

    The frozen prior is uniform in LOG_sd (sd is log-uniform on [0.6,1.5]),
    so the z coordinate is the logit of the normalized LOG_sd, not of sd.
    """
    log_sd = np.asarray(log_sd, dtype=np.float64)
    rho_mag = np.asarray(rho_mag, dtype=np.float64)
    z_sd = _logit((log_sd - LOG_SD_LO) / (LOG_SD_HI - LOG_SD_LO))
    z_rho = _logit((rho_mag - RHO_LO) / (RHO_HI - RHO_LO))
    return np.concatenate([np.asarray(z_sd).ravel(), np.asarray(z_rho).ravel()])


def z_to_native(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of native_to_z; works on 1-D or trailing-axis arrays."""
    s = _sigmoid(z[..., :D_DIM])
    log_sd = LOG_SD_LO + (LOG_SD_HI - LOG_SD_LO) * s
    sd = np.exp(log_sd)
    r = _sigmoid(z[..., D_DIM:D_DIM + N_SIGN])
    rho_mag = RHO_LO + (RHO_HI - RHO_LO) * r
    return sd, rho_mag


def make_sigma(sd: np.ndarray, rho_mag: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Assemble the covariance matrix from native coords. Batch (...,4,4)."""
    sd = np.asarray(sd, dtype=np.float64)
    rho_mag = np.asarray(rho_mag, dtype=np.float64)
    sign = np.asarray(sign, dtype=np.float64)
    n = sd.shape[-1]
    if n != D_DIM:
        raise ValueError("make_sigma requires d=4")
    S = np.zeros(sd.shape[:-1] + (D_DIM, D_DIM), dtype=np.float64)
    k = 0
    for i in range(D_DIM):
        for j in range(i + 1, D_DIM):
            S[..., i, j] = S[..., j, i] = sign[..., k] * rho_mag[..., k] * sd[..., i] * sd[..., j]
            k += 1
    S[..., np.arange(D_DIM), np.arange(D_DIM)] = sd[..., :] ** 2
    return S


def log_prior_density_z(z: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Exact log prior density in (z, sign) coords.

    z is (..., 10) continuous in unbounded coords; sign is (..., 6) in {-1,+1}.
    Returns the log density (scalar per point). Invalid (non-PD or validity-
    violating) points get -inf. The density is
      prod_j s(z_j)(1-s(z_j)) * (1/2)^6 * 1[valid].
    The logistic factors are the exact push-forward of the uniform box prior.
    """
    z = np.asarray(z, dtype=np.float64)
    sign = np.asarray(sign, dtype=np.float64)
    s = _sigmoid(z)
    lp = np.sum(np.log(np.clip(s, 1e-300, 1.0)) + np.log(np.clip(1.0 - s, 1e-300, 1.0)), axis=-1)
    lp = lp + 6.0 * math.log(0.5) - NEGLOG_P_VALID
    sd, rho_mag = z_to_native(z)
    S = make_sigma(sd, rho_mag, sign)
    ev = np.linalg.eigvalsh(S)
    pd = ev[..., 0] > 1e-6
    valid = _validity_batch(S)
    ok = pd & valid
    out = np.full(lp.shape, -np.inf, dtype=np.float64)
    out[ok] = lp[ok]
    return out


def _validity_batch(S: np.ndarray) -> np.ndarray:
    """Batch validity_keep: for every ordering, Cholesky params satisfy bounds.

    Matrices that are not positive definite are returned as invalid (False)
    without running the Cholesky factorization.
    """
    f = fleet()
    shape = S.shape[:-2]
    n = int(np.prod(shape)) if shape else 1
    S2 = S.reshape(n, D_DIM, D_DIM)
    out = np.zeros(n, dtype=bool)
    ev = np.linalg.eigvalsh(S2)
    pd = ev[:, 0] > 1e-6
    idx = np.flatnonzero(pd)
    if len(idx) == 0:
        return out.reshape(shape)
    sub = np.ones(len(idx), dtype=bool)
    for pi in f.ORDERINGS:
        Spi = S2[idx][:, pi][:, :, pi]
        L = np.linalg.cholesky(Spi)
        diag = np.diagonal(L, axis1=1, axis2=2)
        Lunit = L / diag[:, None, :]
        U = np.linalg.inv(Lunit)
        b = np.sqrt(np.maximum(diag ** 2, 1e-12) / 2.0)
        beta = -U
        mask = np.tril(np.ones((D_DIM, D_DIM)), -1).astype(bool)
        amax = np.abs(beta[:, mask]).max(axis=1)
        keep = (amax <= 1.5) & (b >= 0.3).all(axis=1) & (b <= 1.3).all(axis=1)
        sub &= keep
    out[idx] = sub
    return out.reshape(shape)


def sample_prior_z(sign_shape: tuple[int, ...], n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n exact-prior points in (z, sign) coords by accept-reject."""
    sd_out = np.empty((n, D_DIM))
    rm_out = np.empty((n, N_SIGN))
    sg_out = np.empty((n, N_SIGN))
    got = 0
    while got < n:
        m = max(n * 4, 4096)
        log_sd = rng.uniform(LOG_SD_LO, LOG_SD_HI, (m, D_DIM))
        rho_mag = rng.uniform(RHO_LO, RHO_HI, (m, N_SIGN))
        sign = rng.choice([-1.0, 1.0], (m, N_SIGN))
        S = make_sigma(np.exp(log_sd), rho_mag, sign)
        ev = np.linalg.eigvalsh(S)
        pd = ev[:, 0] > 1e-6
        keep = pd & _validity_batch(S)
        if not keep.any():
            continue
        take = min(keep.sum(), n - got)
        idx = np.flatnonzero(keep)[:take]
        sd_out[got:got + take] = np.exp(log_sd[idx])
        rm_out[got:got + take] = rho_mag[idx]
        sg_out[got:got + take] = sign[idx]
        got += take
    z = np.stack([native_to_z(np.log(sd_out[i]), rm_out[i]) for i in range(n)])
    return z, sg_out


def make_sigma_torch(sd: torch.Tensor, rho_mag: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Torch batch make_sigma (...,4,4)."""
    S = torch.zeros(sd.shape[:-1] + (D_DIM, D_DIM), dtype=torch.float64, device=sd.device)
    k = 0
    for i in range(D_DIM):
        for j in range(i + 1, D_DIM):
            S[..., i, j] = S[..., j, i] = sign[..., k] * rho_mag[..., k] * sd[..., i] * sd[..., j]
            k += 1
    S[..., torch.arange(D_DIM), torch.arange(D_DIM)] = sd ** 2
    return S


def validity_torch(S: torch.Tensor) -> torch.Tensor:
    """Torch batch validity_keep; returns bool tensor. S (...,4,4) float64.

    Non-positive-definite matrices are returned as invalid without running the
    Cholesky factorization.
    """
    f = fleet()
    flat = S.reshape(-1, D_DIM, D_DIM)
    n = flat.shape[0]
    out = torch.zeros(n, dtype=torch.bool, device=S.device)
    ev = torch.linalg.eigvalsh(flat)
    pd = ev[:, 0] > 1e-6
    idx = torch.nonzero(pd, as_tuple=False).flatten()
    if idx.numel() == 0:
        return out.reshape(S.shape[:-2])
    sub = torch.ones(idx.numel(), dtype=torch.bool, device=S.device)
    for pi in f.ORDERINGS:
        Spi = flat[idx][:, pi][:, :, pi]
        L = torch.linalg.cholesky(Spi)
        diag = torch.diagonal(L, dim1=1, dim2=2)
        Lunit = L / diag[:, None, :]
        U = torch.linalg.inv(Lunit)
        b = torch.sqrt(torch.clamp(diag ** 2, min=1e-12) / 2.0)
        beta = -U
        tril = torch.tril(torch.ones(D_DIM, D_DIM, dtype=torch.bool, device=S.device), -1)
        amax = beta.abs()[:, tril].max(dim=1).values
        keep = (amax <= 1.5) & (b >= 0.3).all(dim=1) & (b <= 1.3).all(dim=1)
        sub &= keep
    out[idx] = sub
    return out.reshape(S.shape[:-2])


def log_prior_density_z_torch(z: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Torch exact log prior density in (z, sign) coords."""
    s = torch.sigmoid(z)
    lp = torch.sum(torch.log(torch.clamp(s, min=1e-300)) + torch.log(torch.clamp(1 - s, min=1e-300)), dim=-1)
    lp = lp + 6.0 * math.log(0.5) - NEGLOG_P_VALID
    sd, rho_mag = z_to_native_torch(z)
    S = make_sigma_torch(sd, rho_mag, sign)
    ev = torch.linalg.eigvalsh(S)
    pd = ev[..., 0] > 1e-6
    valid = validity_torch(S)
    ok = pd & valid
    out = torch.full_like(lp, -torch.inf)
    out[ok] = lp[ok]
    return out


def z_to_native_torch(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.sigmoid(z[..., :D_DIM])
    log_sd = LOG_SD_LO + (LOG_SD_HI - LOG_SD_LO) * s
    sd = torch.exp(log_sd)
    r = torch.sigmoid(z[..., D_DIM:D_DIM + N_SIGN])
    rho_mag = RHO_LO + (RHO_HI - RHO_LO) * r
    return sd, rho_mag


# ---------------------------------------------------------------------------
# Guards (outcome blindness)
# ---------------------------------------------------------------------------

def assert_path_allowed(path: Path) -> None:
    """Refuse paths that could leak PFN outputs or the old oracle results."""
    text = str(path)
    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise RuntimeError(f"pilot path guard refused: {path}")


def assert_no_forbidden_imports() -> None:
    """Refuse loading this package if any forbidden module is already imported."""
    forbidden = [name for name in sys.modules if any(
        frag in name for frag in FORBIDDEN_IMPORT_FRAGMENTS)]
    if forbidden:
        raise RuntimeError(f"pilot refuses a process with forbidden imports: {forbidden}")


import sys  # noqa: E402

# ---------------------------------------------------------------------------
# Quadrature (frozen production definition)
# ---------------------------------------------------------------------------

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
