"""Independent output-level Phase-1 ordering-use replication.

The first stage is an oracle-only calibration.  It is deliberately unable to
load a PFN checkpoint or compute a scientific deficit.  Confirmatory panel,
scoring, and join commands are added only after calibration freezes the
truncation setting in a locked attempt configuration.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
from copy import deepcopy
import csv
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from .storage import write_json_atomic, write_numeric_npz_atomic


HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


PRODUCTION_CALIBRATION_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "status": "calibration_only",
    "fleet_module": "artifacts/phase1/d4_generator.py",
    "fleet_sha256": "1aa7652cad924c90f871309f860b9172e836898c5a1620c77fdd3196e70d291d",
    "dimension": 4,
    "context_size": 30,
    "priors": ["C", "N"],
    "calibration_contexts_per_prior": 32,
    "calibration_seed_root": 880_803_000,
    "calibration_seed_namespace": "phase1-ordering-calibration-v1",
    "atom_count": 3_000_000,
    "atom_seed": 880_803_101,
    "reserved_confirmatory_seeds": [881_003_000, 881_003_101, 881_003_102, 881_003_103],
    "known_persisted_fixed_seeds": [
        810_000,
        810_099,
        810_101,
        810_777,
        820_001,
        850_002,
        860_003,
        860_004,
        860_005,
        860_006,
        860_007,
    ],
    "truncation_candidates": [8_192, 16_384],
    "reference_truncation": 32_768,
    "quadrature_interior_nodes": 8,
    "quadrature_tail_nodes": 32,
    "query_grid_chunk": 16,
    "context_atom_batch": 250_000,
    "require_clean_git": True,
    "environment_contract": "environment/phase1-washu-runtime.json",
    "requirements_lock": "environment/phase1-washu-requirements-lock.txt",
    "runtime_binary_inventory": "environment/phase1-washu-binary-inventory.json",
    "cluster_wrapper": "cluster/phase1_calibration.sbatch",
    "thresholds": {
        "median_js_max": 1e-4,
        "p95_js_max": 1e-3,
        "median_abs_logp_change_max": 0.002,
        "p95_abs_logp_change_max": 0.01,
        "numerical_indifference_fraction": 0.1,
        "probability_sum_atol": 1e-8,
    },
}


def production_qualification_protocol(bank_index: int) -> dict[str, Any]:
    if bank_index not in {0, 1, 2}:
        raise ValueError("qualification bank_index must be 0, 1, or 2")
    protocol = deepcopy(PRODUCTION_CALIBRATION_PROTOCOL)
    protocol.update(
        {
            "calibration_contexts_per_prior": 160,
            "calibration_seed_root": 880_903_000 + bank_index,
            "calibration_seed_namespace": (
                f"phase1-ordering-cross-bank-qualification-v1-bank{bank_index}"
            ),
            "atom_seed": 881_003_101 + bank_index,
            "atom_determinism_canary_seed": 881_103_999,
            "atom_determinism_canary_count": 4_096,
            "reserved_confirmatory_seeds": [
                881_003_000,
                881_003_001,
                881_003_002,
                881_013_000,
                881_013_001,
                881_013_002,
            ],
            "qualification_bank_index": bank_index,
            "qualification_zero_exceedance": {
                "familywise_alpha": 0.05,
                "families": 48,
                "contexts_per_family": 160,
                "js_strict_p95_max": 9e-4,
                "abs_logp_change_strict_p95_max": 0.009,
                "bonferroni_clopper_pearson_upper": 0.04201037701571053,
            },
        }
    )
    return protocol


def production_qualification_protocol_v2(bank_index: int) -> dict[str, Any]:
    """Corrected, replayable qualification for the uniform-ordering ablation.

    Version 1 was run before a pre-confirmation audit found that top-K retained
    mass leaked back into the ordering weights.  Version 2 keeps the same atom
    banks but uses fresh contexts and archives the native predictive arrays so
    an implementation-independent verifier can recompute every diagnostic.
    """

    if bank_index not in {0, 1, 2}:
        raise ValueError("qualification bank_index must be 0, 1, or 2")
    protocol = production_qualification_protocol(bank_index)
    protocol.update(
        {
            "qualification_protocol_version": 2,
            "calibration_seed_root": 880_923_000 + bank_index,
            "calibration_seed_namespace": (
                f"phase1-ordering-cross-bank-qualification-v2-bank{bank_index}"
            ),
            "ablation_weighting": (
                "uniform-over-orderings-after-within-order-topk-normalization-v1"
            ),
            "archive_predictive_arrays": True,
            "qualification_preregistration": (
                "PHASE1_ORDERING_QUALIFICATION_V2_PREREG.md"
            ),
        }
    )
    return protocol


def production_qualification_protocol_v3(bank_index: int) -> dict[str, Any]:
    """Prospective high-order quadrature qualification after v2 was blocked."""

    protocol = production_qualification_protocol_v2(bank_index)
    protocol.update(
        {
            "qualification_protocol_version": 3,
            "calibration_seed_root": 880_943_000 + bank_index,
            "calibration_seed_namespace": (
                f"phase1-ordering-cross-bank-qualification-v3-bank{bank_index}"
            ),
            "quadrature_interior_nodes": 32,
            "quadrature_tail_nodes": 128,
            "quadrature_reference_interior_nodes": 64,
            "quadrature_reference_tail_nodes": 256,
            "quadrature_qualification": {
                "js_max": 1e-7,
                "reference_probability_floor": 1e-8,
                "max_bin_abs_logp_change_max": 5e-4,
                "reference_weighted_abs_logp_change_max": 1e-4,
                "max_bin_abs_ordering_value_change_max": 5e-4,
            },
            "oracle_internal_dtype": "float64",
            "qualification_preregistration": (
                "PHASE1_ORDERING_QUALIFICATION_V3_PREREG.md"
            ),
            "required_source_tag": "phase1-ordering-qualification-v3",
        }
    )
    return protocol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _cpu_identity() -> dict[str, Any]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        raise RuntimeError("production runtime requires /proc/cpuinfo")
    for line in cpuinfo.read_text().splitlines():
        if not line.strip():
            break
        if ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        if name in {"vendor_id", "cpu family", "model", "model name", "stepping", "flags"}:
            fields[name] = value
    required = {"vendor_id", "cpu family", "model", "model name", "stepping", "flags"}
    if set(fields) != required:
        raise RuntimeError(f"incomplete CPU identity: {sorted(fields)}")
    fields["flags"] = " ".join(sorted(fields["flags"].split()))
    return fields


def _distribution_payload_fingerprint(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    files = distribution.files
    if files is None:
        raise RuntimeError(f"installed distribution has no file inventory: {name}")
    records = [item for item in files if str(item).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise RuntimeError(f"installed distribution has no unique RECORD: {name}")
    record_path = Path(distribution.locate_file(records[0])).resolve()
    entries: list[dict[str, Any]] = []
    payload_bytes = 0
    with record_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise RuntimeError(f"malformed installed distribution RECORD: {name}")
        relative, encoded_digest, encoded_size = row
        relative_path = Path(relative)
        if "__pycache__" in relative_path.parts or relative_path.suffix == ".pyc":
            continue
        path = Path(distribution.locate_file(relative)).resolve()
        if not path.is_file():
            raise RuntimeError(f"installed distribution payload missing: {name}:{relative}")
        size = path.stat().st_size
        if encoded_size and size != int(encoded_size):
            raise RuntimeError(f"installed distribution size mismatch: {name}:{relative}")
        digest = bytes.fromhex(_sha256_file(path))
        if encoded_digest:
            algorithm, encoded = encoded_digest.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError(
                    f"unsupported installed distribution digest: {name}:{relative}"
                )
            if digest != base64.urlsafe_b64decode(encoded + "=="):
                raise RuntimeError(
                    f"installed distribution payload mismatch: {name}:{relative}"
                )
        entries.append({"path": relative, "size": size, "sha256": digest.hex()})
        payload_bytes += size
    tree = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: value["path"]):
        tree.update(
            (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return {
        "version": distribution.version,
        "record_size": record_path.stat().st_size,
        "record_sha256": _sha256_file(record_path),
        "payload_files": len(entries),
        "payload_bytes": payload_bytes,
        "payload_tree_sha256": tree.hexdigest(),
    }


def build_runtime_binary_inventory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the Phase-1 production inventory must be built on CUDA")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    runtime_buffer = io.StringIO()
    with contextlib.redirect_stdout(runtime_buffer):
        np.show_runtime()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[0].strip()
    properties = torch.cuda.get_device_properties(device)
    executable = Path(sys.executable).resolve()
    inventory = {
        "schema_version": 1,
        "scope": "WashU Phase-1 CUDA calibration runtime",
        "python_executable": {
            "path": sys.executable,
            "resolved_path": str(executable),
            "size": executable.stat().st_size,
            "sha256": _sha256_file(executable),
        },
        "distributions": {
            name: _distribution_payload_fingerprint(name) for name in ("numpy", "torch")
        },
        "numpy_config": buffer.getvalue(),
        "numpy_runtime": runtime_buffer.getvalue(),
        "cpu_identity": _cpu_identity(),
        "torch_config": torch.__config__.show(),
        "platform": platform.platform(),
        "cuda_driver": driver,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory": int(properties.total_memory),
    }
    inventory["runtime_binary_fingerprint"] = _sha256_json(inventory)
    return inventory


def verify_runtime_binary_inventory(
    expected: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    observed = build_runtime_binary_inventory(device)
    if observed != expected:
        raise RuntimeError(
            "runtime binary fingerprint mismatch: "
            f"{observed.get('runtime_binary_fingerprint')} != "
            f"{expected.get('runtime_binary_fingerprint')}"
        )
    return observed


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def load_fleet_module(path: Path) -> ModuleType:
    """Load the frozen d=4 generator/model definition from an exact path."""

    if not path.is_file():
        raise FileNotFoundError(path)
    os.environ["D4_SCALE"] = "base"
    spec = importlib.util.spec_from_file_location("phase1_frozen_d4_fleet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load fleet module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "D_DIM",
        "ORDERINGS",
        "BIN_EDGES",
        "N_BINS",
        "R_OF",
        "params_for",
        "validity_keep",
        "gen_data",
        "bin_y",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AttributeError(f"fleet module is missing required names: {missing}")
    if int(module.D_DIM) != 4 or len(module.ORDERINGS) != math.factorial(4):
        raise ValueError("the Phase-1 protocol requires exactly 24 d=4 orderings")
    if int(module.N_BINS) != 100:
        raise ValueError("the Phase-1 protocol requires the native 100-bin head")
    return module


def configure_runtime(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False


def sample_sigmas_exact(fleet: ModuleType, rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorized equivalent of the frozen fleet's memory-heavy sampler.

    The legacy implementation extends a Python list with individual matrices.
    This implementation retains entire accepted batches and is bit-identical
    for a fixed RNG state while using bounded Python-object overhead.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    chunks: list[np.ndarray] = []
    accepted = 0
    d = int(fleet.D_DIM)
    while accepted < n:
        m = 8192
        sd = np.exp(rng.uniform(math.log(0.6), math.log(1.5), (m, d)))
        corr = np.repeat(np.eye(d)[None, :, :], m, axis=0)
        upper = np.triu_indices(d, 1)
        rho = rng.choice([-1.0, 1.0], (m, len(upper[0]))) * rng.uniform(
            0.3, 0.8, (m, len(upper[0]))
        )
        corr[:, upper[0], upper[1]] = rho
        corr[:, upper[1], upper[0]] = rho
        sigmas = sd[:, :, None] * corr * sd[:, None, :]
        eigenvalues = np.linalg.eigvalsh(sigmas)
        sigmas = sigmas[eigenvalues[:, 0] > 1e-6]
        sigmas = sigmas[np.asarray(fleet.validity_keep(sigmas), dtype=bool)]
        chunks.append(sigmas)
        accepted += len(sigmas)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0)[:n])


def _al_scales(b: torch.Tensor, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.sqrt(2.0 * b * b / (1.0 + r * r))
    return r * c, c


def _residual_logpdf(
    x: torch.Tensor, b: torch.Tensor, r: float, gaussian: bool
) -> torch.Tensor:
    if gaussian:
        scale = b * math.sqrt(2.0)
        return -0.5 * (x / scale) ** 2 - torch.log(scale) - HALF_LOG_2PI
    a, c = _al_scales(b, r)
    shifted = x + (a - c)
    return torch.where(shifted >= 0, -shifted / a, shifted / c) - torch.log(a + c)


@dataclass(frozen=True)
class OraclePrediction:
    full: np.ndarray
    ablated: np.ndarray
    ordering_posterior: np.ndarray
    keep_full: float
    keep_ablated: float


class OrderingOracle:
    """Monte Carlo posterior predictive over all 24 d=4 orderings."""

    def __init__(
        self,
        fleet: ModuleType,
        atoms: np.ndarray,
        device: torch.device,
        context_atom_batch: int = 250_000,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        if atoms.ndim != 3 or atoms.shape[1:] != (4, 4):
            raise ValueError(f"expected atoms with shape (M,4,4), got {atoms.shape}")
        self.fleet = fleet
        self.device = device
        self.atom_count = len(atoms)
        self.context_atom_batch = int(context_atom_batch)
        if self.context_atom_batch <= 0:
            raise ValueError("context_atom_batch must be positive")
        if compute_dtype not in {torch.float32, torch.float64}:
            raise ValueError("oracle compute_dtype must be float32 or float64")
        self.compute_dtype = compute_dtype
        numpy_dtype = np.float64 if compute_dtype == torch.float64 else np.float32
        orderings = np.asarray(fleet.ORDERINGS, dtype=np.int64)
        self.n_orderings = len(orderings)
        u = np.empty((self.n_orderings, self.atom_count, 4, 4), dtype=numpy_dtype)
        b = np.empty((self.n_orderings, self.atom_count, 4), dtype=numpy_dtype)
        for index, ordering in enumerate(fleet.ORDERINGS):
            _, ui, bi = fleet.params_for(atoms, ordering)
            u[index] = np.asarray(ui, dtype=numpy_dtype)
            b[index] = np.asarray(bi, dtype=numpy_dtype)
        self.u = torch.from_numpy(u).to(device)
        self.b = torch.from_numpy(b).to(device)
        self.orderings = torch.from_numpy(orderings).long().to(device)

    @torch.no_grad()
    def context_log_likelihood(
        self, context: np.ndarray, r: float, gaussian: bool
    ) -> torch.Tensor:
        data = torch.as_tensor(context, dtype=self.compute_dtype, device=self.device)
        if data.shape != (30, 4):
            raise ValueError(f"expected one (30,4) context, got {tuple(data.shape)}")
        result = torch.empty(
            (self.n_orderings, self.atom_count),
            dtype=torch.float64,
            device=self.device,
        )
        for ordering_index in range(self.n_orderings):
            permuted = data[:, self.orderings[ordering_index]]
            for start in range(0, self.atom_count, self.context_atom_batch):
                stop = min(self.atom_count, start + self.context_atom_batch)
                residual = torch.einsum(
                    "amj,kj->amk",
                    self.u[ordering_index, start:stop],
                    permuted,
                )
                log_density = _residual_logpdf(
                    residual,
                    self.b[ordering_index, start:stop, :, None],
                    r,
                    gaussian,
                )
                result[ordering_index, start:stop] = log_density.sum(
                    dim=(1, 2), dtype=torch.float64
                )
        return result

    @staticmethod
    def _deterministic_topk_rows(
        values: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select exact row-wise top-k with ascending-index cutoff tie-breaks."""

        if values.ndim != 2 or not 0 < count <= values.shape[1]:
            raise ValueError("invalid deterministic top-k request")
        kept_values: list[torch.Tensor] = []
        kept_indices: list[torch.Tensor] = []
        for row in values:
            if count == row.numel():
                indices = torch.arange(row.numel(), device=row.device)
            else:
                provisional = torch.topk(row, count, sorted=False).values
                cutoff = provisional.min()
                strict = torch.nonzero(row > cutoff, as_tuple=False).flatten()
                tied = torch.nonzero(row == cutoff, as_tuple=False).flatten()
                needed = count - strict.numel()
                if needed < 0 or tied.numel() < needed:
                    raise RuntimeError("deterministic top-k cutoff accounting failed")
                indices = torch.cat((strict, tied[:needed]))
            if indices.numel() != count:
                raise RuntimeError("deterministic top-k returned the wrong count")
            kept_indices.append(indices)
            kept_values.append(row[indices])
        return torch.stack(kept_values), torch.stack(kept_indices)

    @staticmethod
    def _uniform_ablated_log_weights(kept_log_likelihood: torch.Tensor) -> torch.Tensor:
        """Normalize each truncated ordering posterior before uniform mixing."""

        if kept_log_likelihood.ndim != 2 or kept_log_likelihood.shape[1] == 0:
            raise ValueError("ablated weights require a nonempty ordering-by-atom matrix")
        kept_log_z = torch.logsumexp(kept_log_likelihood.double(), dim=1, keepdim=True)
        weights = kept_log_likelihood.double() - kept_log_z
        normalization = torch.logsumexp(weights, dim=1)
        if not torch.allclose(
            normalization,
            torch.zeros_like(normalization),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("ordering-ablated weights are not normalized per ordering")
        return weights

    @torch.no_grad()
    def predict_from_log_likelihood(
        self,
        log_likelihood: torch.Tensor,
        query: np.ndarray,
        r: float,
        gaussian: bool,
        truncation: int,
        quadrature_values: np.ndarray,
        quadrature_bins: np.ndarray,
        quadrature_log_weights: np.ndarray,
        query_grid_chunk: int,
        probability_atol: float,
        atom_limit: int | None = None,
    ) -> OraclePrediction:
        limit = self.atom_count if atom_limit is None else int(atom_limit)
        if not 0 < limit <= self.atom_count:
            raise ValueError("atom_limit is outside the atom bank")
        ll = log_likelihood[:, :limit]
        t_keep = min(int(truncation), limit)
        if t_keep <= 0:
            raise ValueError("truncation must be positive")
        log_z = torch.logsumexp(ll, dim=1)
        ordering_posterior = torch.softmax(log_z, dim=0)
        kept_ll, kept_indices = self._deterministic_topk_rows(ll, t_keep)
        kept_log_z = torch.logsumexp(kept_ll, dim=1)
        retained_per_order = torch.exp(kept_log_z - log_z)
        keep_full = float((ordering_posterior * retained_per_order).sum())
        keep_ablated = float(retained_per_order.mean())

        ordering_rows = torch.arange(self.n_orderings, device=self.device)[:, None].expand(
            self.n_orderings, t_keep
        )
        kept_u = self.u[ordering_rows, kept_indices]
        kept_b = self.b[ordering_rows, kept_indices]
        full_weights = kept_ll
        ablated_weights = self._uniform_ablated_log_weights(kept_ll)
        n_values = len(quadrature_values)
        full_log_numerator = np.empty(n_values, dtype=np.float64)
        ablated_log_numerator = np.empty(n_values, dtype=np.float64)
        chunk_size = int(query_grid_chunk)
        if chunk_size <= 0:
            raise ValueError("query_grid_chunk must be positive")
        query_tensor = torch.as_tensor(
            query, dtype=self.compute_dtype, device=self.device
        )
        for start in range(0, n_values, chunk_size):
            stop = min(n_values, start + chunk_size)
            values = torch.as_tensor(
                quadrature_values[start:stop],
                dtype=self.compute_dtype,
                device=self.device,
            )
            points = torch.empty(
                (len(values), 4), dtype=self.compute_dtype, device=self.device
            )
            points[:, :3] = query_tensor[None, :]
            points[:, 3] = values
            permuted_points = points[:, self.orderings].permute(1, 0, 2)
            residual = torch.einsum("ftmj,fnj->ftmn", kept_u, permuted_points)
            log_joint = _residual_logpdf(
                residual, kept_b[:, :, :, None], r, gaussian
            ).sum(dim=2, dtype=torch.float64)
            full_chunk = torch.logsumexp(
                (full_weights[:, :, None].double() + log_joint).reshape(-1, len(values)),
                dim=0,
            )
            ablated_chunk = torch.logsumexp(
                (ablated_weights[:, :, None].double() + log_joint).reshape(
                    -1, len(values)
                ),
                dim=0,
            )
            full_log_numerator[start:stop] = full_chunk.cpu().numpy()
            ablated_log_numerator[start:stop] = ablated_chunk.cpu().numpy()
            del points, permuted_points, residual, log_joint, full_chunk, ablated_chunk

        def to_bins(log_numerator: np.ndarray) -> np.ndarray:
            # CUDA scatter_add is nondeterministic on some torch/CUDA pairs.
            # Aggregate weighted quadrature nodes deterministically on CPU.
            weighted = np.asarray(log_numerator, dtype=np.float64) + np.asarray(
                quadrature_log_weights, dtype=np.float64
            )
            shifted = weighted - np.max(weighted)
            density = np.exp(shifted)
            probability = np.bincount(
                np.asarray(quadrature_bins, dtype=np.int64),
                weights=density,
                minlength=int(self.fleet.N_BINS),
            ).astype(np.float64)
            probability = probability / probability.sum()
            return probability

        prediction = OraclePrediction(
            full=to_bins(full_log_numerator),
            ablated=to_bins(ablated_log_numerator),
            ordering_posterior=ordering_posterior.cpu().numpy().astype(np.float64),
            keep_full=keep_full,
            keep_ablated=keep_ablated,
        )
        validate_probability(prediction.full, "full", atol=probability_atol)
        validate_probability(prediction.ablated, "ablated", atol=probability_atol)
        return prediction

    @torch.no_grad()
    def collapsed_atom_ess(self, log_likelihood: torch.Tensor) -> tuple[float, float]:
        """Context-posterior ESS after collapsing exact ordering copies by atom."""

        log_z_per_order = torch.logsumexp(log_likelihood.double(), dim=1)
        full_atom_log_weight = torch.logsumexp(log_likelihood.double(), dim=0)
        ablated_atom_log_weight = torch.logsumexp(
            log_likelihood.double() - log_z_per_order[:, None], dim=0
        )
        full_probability = torch.softmax(full_atom_log_weight, dim=0)
        ablated_probability = torch.softmax(ablated_atom_log_weight, dim=0)
        full_ess = float(1.0 / torch.sum(full_probability * full_probability))
        ablated_ess = float(1.0 / torch.sum(ablated_probability * ablated_probability))
        return full_ess, ablated_ess


def validate_probability(value: np.ndarray, label: str, atol: float = 1e-6) -> None:
    probability = np.asarray(value, dtype=np.float64)
    if probability.shape != (100,):
        raise ValueError(f"{label} has wrong shape: {probability.shape}")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(probability < 0):
        raise ValueError(f"{label} contains negative probabilities")
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=atol):
        raise ValueError(f"{label} does not sum to one: {probability.sum()}")


def quadrature_grid(
    fleet: ModuleType, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned quadrature for the 98 finite and two semi-infinite output bins."""

    interior_nodes = int(config["quadrature_interior_nodes"])
    tail_nodes = int(config["quadrature_tail_nodes"])
    if interior_nodes <= 0 or tail_nodes <= 0:
        raise ValueError("quadrature node counts must be positive")
    edges = np.asarray(fleet.BIN_EDGES, dtype=np.float64)[1:-1]
    x_interior, w_interior = np.polynomial.legendre.leggauss(interior_nodes)
    x_tail, w_tail = np.polynomial.legendre.leggauss(tail_nodes)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    bins: list[np.ndarray] = []

    # Map u in (0,1) to the two infinite tails with dx/du=(1-u)^-2.
    u = (x_tail + 1.0) / 2.0
    base_tail_weight = (w_tail / 2.0) / ((1.0 - u) ** 2)
    values.append(edges[0] - u / (1.0 - u))
    weights.append(base_tail_weight)
    bins.append(np.zeros(tail_nodes, dtype=np.int64))

    for bin_index in range(1, int(fleet.N_BINS) - 1):
        left = edges[bin_index - 1]
        right = edges[bin_index]
        values.append((right - left) * x_interior / 2.0 + (right + left) / 2.0)
        weights.append(w_interior * (right - left) / 2.0)
        bins.append(np.full(interior_nodes, bin_index, dtype=np.int64))

    values.append(edges[-1] + u / (1.0 - u))
    weights.append(base_tail_weight)
    bins.append(np.full(tail_nodes, int(fleet.N_BINS) - 1, dtype=np.int64))
    value_array = np.concatenate(values).astype(np.float64)
    weight_array = np.concatenate(weights).astype(np.float64)
    bin_array = np.concatenate(bins).astype(np.int64)
    if np.any(weight_array <= 0) or not np.all(np.isfinite(weight_array)):
        raise RuntimeError("quadrature produced invalid weights")
    return value_array, bin_array, np.log(weight_array)


def generate_evaluation_stream(
    fleet: ModuleType, prior: str, count: int, context_size: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = float(fleet.R_OF.get(prior, 2.0))
    gaussian = prior == "N"
    contexts = np.empty((count, context_size, 4), dtype=np.float64)
    queries = np.empty((count, 3), dtype=np.float64)
    outcomes = np.empty(count, dtype=np.float64)
    sigmas = np.empty((count, 4, 4), dtype=np.float64)
    orderings = np.empty(count, dtype=np.int64)
    sampled_sigmas = sample_sigmas_exact(fleet, rng, count)
    for index in range(count):
        ordering = int(rng.integers(len(fleet.ORDERINGS)))
        block = fleet.gen_data(
            sampled_sigmas[index], ordering, r, context_size + 1, rng, gaussian=gaussian
        )
        contexts[index] = block[:context_size]
        queries[index] = block[context_size, :3]
        outcomes[index] = block[context_size, 3]
        sigmas[index] = sampled_sigmas[index]
        orderings[index] = ordering
    bins = np.asarray(fleet.bin_y(outcomes), dtype=np.int64)
    return {
        "contexts": contexts,
        "queries": queries,
        "outcomes": outcomes,
        "outcome_bins": bins,
        "sigmas": sigmas,
        "true_orderings": orderings,
    }


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left_probability = np.asarray(left, dtype=np.float64)
    right_probability = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left_probability + right_probability)
    left_mask = left_probability > 0
    right_mask = right_probability > 0
    left_kl = np.sum(
        left_probability[left_mask]
        * (np.log(left_probability[left_mask]) - np.log(middle[left_mask]))
    )
    right_kl = np.sum(
        right_probability[right_mask]
        * (np.log(right_probability[right_mask]) - np.log(middle[right_mask]))
    )
    return float(0.5 * (left_kl + right_kl))


def absolute_log_probability_change(
    candidate: np.ndarray, reference: np.ndarray, outcome_bin: int
) -> float:
    """Compare the native observed-bin probabilities without a silent floor."""

    candidate_probability = float(candidate[outcome_bin])
    reference_probability = float(reference[outcome_bin])
    if (
        not math.isfinite(candidate_probability)
        or not math.isfinite(reference_probability)
        or candidate_probability <= 0.0
        or reference_probability <= 0.0
    ):
        raise RuntimeError("observed-bin oracle probability must be finite and positive")
    return abs(math.log(candidate_probability) - math.log(reference_probability))


def _predictor_convergence_row(
    js_values: np.ndarray,
    logp_changes: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, float | bool]:
    row: dict[str, float | bool] = {
        "median_js": float(np.median(js_values)),
        "p95_js": float(np.quantile(js_values, 0.95)),
        "median_abs_logp_change": float(np.median(logp_changes)),
        "p95_abs_logp_change": float(np.quantile(logp_changes, 0.95)),
    }
    margin = float(thresholds["numerical_indifference_fraction"])
    if not 0.0 <= margin < 1.0:
        raise ValueError("numerical_indifference_fraction must be in [0,1)")
    comparisons = {
        "median_js": float(thresholds["median_js_max"]),
        "p95_js": float(thresholds["p95_js_max"]),
        "median_abs_logp_change": float(thresholds["median_abs_logp_change_max"]),
        "p95_abs_logp_change": float(thresholds["p95_abs_logp_change_max"]),
    }
    row["pass"] = bool(
        all(float(row[name]) <= threshold * (1.0 - margin) for name, threshold in comparisons.items())
    )
    row["borderline"] = bool(
        any(
            threshold * (1.0 - margin) < float(row[name]) <= threshold * (1.0 + margin)
            for name, threshold in comparisons.items()
        )
    )
    return row


def _calibration_candidate_passes(
    js_full: np.ndarray,
    js_ablated: np.ndarray,
    logp_full: np.ndarray,
    logp_ablated: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    full = _predictor_convergence_row(js_full, logp_full, thresholds)
    ablated = _predictor_convergence_row(js_ablated, logp_ablated, thresholds)
    passed = bool(full["pass"] and ablated["pass"])
    return passed, {"full": full, "ablated": ablated, "pass": passed}


def _validate_calibration_config(config: dict[str, Any], production: bool) -> None:
    if config.get("status") != "calibration_only":
        raise ValueError("calibrate accepts only a calibration_only config")
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported calibration schema_version")
    if (int(config.get("dimension", -1)), int(config.get("context_size", -1))) != (
        4,
        30,
    ):
        raise ValueError("the calibration protocol is fixed to d=4 and context size 30")
    priors = [str(value) for value in config.get("priors", [])]
    candidates = [int(value) for value in config.get("truncation_candidates", [])]
    reference = int(config.get("reference_truncation", -1))
    if production:
        for name, expected in PRODUCTION_CALIBRATION_PROTOCOL.items():
            if config.get(name) != expected:
                raise ValueError(
                    f"production calibration field {name!r} differs from the frozen protocol"
                )
    elif not priors or not candidates or reference <= candidates[-1]:
        raise ValueError("test calibration requires priors and an increasing reference")
    if candidates != sorted(set(candidates)) or reference <= candidates[-1]:
        raise ValueError("truncation ladder is not strictly increasing")
    if reference > int(config["atom_count"]):
        raise ValueError("reference_truncation exceeds atom_count")
    if float(config["thresholds"]["probability_sum_atol"]) <= 0:
        raise ValueError("probability_sum_atol must be positive")
    for name in ("query_grid_chunk", "context_atom_batch"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")


def _git_provenance(require_clean: bool) -> dict[str, Any]:
    root = _repo_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if require_clean and status:
        raise RuntimeError("production calibration refuses a dirty source tree")
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def _verify_annotated_tag(tag: str, commit: str) -> None:
    if not tag or any(character.isspace() for character in tag):
        raise RuntimeError("required source tag is invalid")
    tag_ref = f"refs/tags/{tag}"
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", tag_ref],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tag_type != "tag":
        raise RuntimeError(f"required source tag is not annotated: {tag}")
    tagged_commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tagged_commit != commit:
        raise RuntimeError(
            f"required source tag {tag} resolves to {tagged_commit}, not {commit}"
        )


def _source_inventory(
    config_path: Path, fleet_path: Path, extra_paths: tuple[Path, ...] = ()
) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "storage": Path(__file__).with_name("storage.py").resolve(),
        "preregistration": (_repo_root() / "PHASE1_ORDERING_PREREG.md").resolve(),
        "config": config_path.resolve(),
        "generator": fleet_path.resolve(),
    }
    for index, path in enumerate(extra_paths):
        paths[f"extra_{index}"] = path.resolve()
    return {str(path): _sha256_file(path) for path in paths.values()}


def _memory_preflight(
    device: torch.device,
    atom_count: int,
    truncation: int,
    grid_chunk: int,
    context_atom_batch: int,
    oracle_element_bytes: int,
) -> dict[str, int | float]:
    if oracle_element_bytes not in {4, 8}:
        raise ValueError("oracle_element_bytes must be four or eight")
    resident = 24 * atom_count * (4 * 4 + 4) * oracle_element_bytes
    likelihood = 24 * atom_count * 8
    predictive = 6 * 24 * truncation * 4 * grid_chunk * 8
    context = 3 * context_atom_batch * 4 * 30 * oracle_element_bytes
    estimate = resident + likelihood + predictive + context
    total = 0
    if device.type == "cuda":
        total = int(torch.cuda.get_device_properties(device).total_memory)
        if estimate >= int(0.80 * total):
            raise RuntimeError(
                f"estimated CUDA peak {estimate} exceeds 80% of device memory {total}"
            )
    return {
        "estimated_peak_bytes": int(estimate),
        "device_total_bytes": int(total),
        "estimated_fraction": float(estimate / total) if total else 0.0,
    }


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lease_is_active(lease: dict[str, Any]) -> bool | None:
    if lease.get("hostname") == socket.gethostname():
        try:
            os.kill(int(lease["pid"]), 0)
        except (ProcessLookupError, ValueError, KeyError):
            return False
        except PermissionError:
            return True
        return True
    slurm_job_id = str(lease.get("slurm_job_id", ""))
    if slurm_job_id and slurm_job_id != "null":
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", slurm_job_id],
                check=True,
                text=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return bool(result.stdout.strip())
    return None


def _acquire_attempt_lease(
    lease_path: Path, identity_sha256: str, recover_stale: bool
) -> dict[str, Any]:
    if lease_path.exists():
        if not recover_stale:
            raise FileExistsError(f"calibration attempt lease already exists: {lease_path}")
        recovery_mutex = lease_path.with_name("RECOVERY.lock")
        _write_exclusive_json(
            recovery_mutex,
            {
                "schema_version": 1,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "created_unix": time.time(),
            },
        )
        try:
            existing = _load_json(lease_path)
            active = _lease_is_active(existing)
            if active is not False:
                raise RuntimeError(
                    f"cannot prove existing calibration lease is stale (active={active})"
                )
            lease_path.unlink()
            return _acquire_attempt_lease(lease_path, identity_sha256, False)
        finally:
            recovery_mutex.unlink()
    lease = {
        "schema_version": 1,
        "attempt_identity_sha256": identity_sha256,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_unix": time.time(),
    }
    _write_exclusive_json(lease_path, lease)
    return lease


def _partial_arrays(
    level_count: int,
    candidate_count: int,
    context_count: int,
    identity_sha256: str,
    prediction_bin_count: int | None = None,
    archive_quadrature_reference: bool = False,
) -> dict[str, np.ndarray]:
    arrays = {
        "attempt_identity_sha256": np.frombuffer(
            bytes.fromhex(identity_sha256), dtype=np.uint8
        ).copy(),
        "completed": np.zeros(context_count, dtype=np.int8),
        "keep_full": np.full((level_count, context_count), np.nan, dtype=np.float64),
        "keep_ablated": np.full((level_count, context_count), np.nan, dtype=np.float64),
        "ess_full_atoms": np.full(context_count, np.nan, dtype=np.float64),
        "ess_ablated_atoms": np.full(context_count, np.nan, dtype=np.float64),
        "full_probability_sum_error": np.full(
            (level_count, context_count), np.nan, dtype=np.float64
        ),
        "ablated_probability_sum_error": np.full(
            (level_count, context_count), np.nan, dtype=np.float64
        ),
        "full_probability_minimum": np.full(
            (level_count, context_count), np.nan, dtype=np.float64
        ),
        "ablated_probability_minimum": np.full(
            (level_count, context_count), np.nan, dtype=np.float64
        ),
        "js_full": np.full((candidate_count, context_count), np.nan, dtype=np.float64),
        "js_ablated": np.full(
            (candidate_count, context_count), np.nan, dtype=np.float64
        ),
        "abs_logp_change_full": np.full(
            (candidate_count, context_count), np.nan, dtype=np.float64
        ),
        "abs_logp_change_ablated": np.full(
            (candidate_count, context_count), np.nan, dtype=np.float64
        ),
    }
    if prediction_bin_count is not None:
        if prediction_bin_count <= 1:
            raise ValueError("prediction_bin_count must exceed one")
        arrays.update(
            {
                "outcome_bins": np.full(context_count, -1, dtype=np.int64),
                "full_probability": np.full(
                    (level_count, context_count, prediction_bin_count),
                    np.nan,
                    dtype=np.float64,
                ),
                "ablated_probability": np.full(
                    (level_count, context_count, prediction_bin_count),
                    np.nan,
                    dtype=np.float64,
                ),
            }
        )
    if archive_quadrature_reference:
        if prediction_bin_count is None:
            raise ValueError("quadrature archival requires predictive-array archival")
        arrays.update(
            {
                "quadrature_grid_full_probability": np.full(
                    (2, level_count, context_count, prediction_bin_count),
                    np.nan,
                    dtype=np.float64,
                ),
                "quadrature_grid_ablated_probability": np.full(
                    (2, level_count, context_count, prediction_bin_count),
                    np.nan,
                    dtype=np.float64,
                ),
                "quadrature_js_full": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_js_ablated": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_max_bin_abs_logp_change_full": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_max_bin_abs_logp_change_ablated": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_reference_weighted_abs_logp_change_full": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_reference_weighted_abs_logp_change_ablated": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
                "quadrature_max_bin_abs_ordering_value_change": np.full(
                    (level_count, context_count), np.nan, dtype=np.float64
                ),
            }
        )
    return arrays


def _load_partial(path: Path, expected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not path.exists():
        return expected
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise RuntimeError(f"partial inventory mismatch: {path}")
        loaded = {name: archive[name].copy() for name in archive.files}
    for name, value in expected.items():
        if loaded[name].shape != value.shape or loaded[name].dtype != value.dtype:
            raise RuntimeError(f"partial shape/dtype mismatch for {name}: {path}")
    if not np.array_equal(
        loaded["attempt_identity_sha256"], expected["attempt_identity_sha256"]
    ):
        raise RuntimeError(f"partial attempt identity mismatch: {path}")
    return loaded


def run_calibration(
    config_path: Path,
    output_dir: Path,
    device_name: str,
    *,
    production_protocol: bool = True,
    recover_stale_lease: bool = False,
    frozen_protocol: dict[str, Any] | None = None,
    strict_runtime: bool | None = None,
    stage_name: str = "phase1_ordering_calibration",
) -> dict[str, Any]:
    configure_runtime(0)
    if strict_runtime is None:
        strict_runtime = production_protocol
    started = time.time()
    config_path = config_path.resolve()
    config_sha_start = _sha256_file(config_path)
    config = _load_json(config_path)
    qualification_protocol_version = int(
        config.get("qualification_protocol_version", 1)
    )
    if qualification_protocol_version == 2:
        raise RuntimeError(
            "qualification v2 is blocked-before-execution and superseded by v3"
        )
    if qualification_protocol_version == 3:
        bank_index = int(config.get("qualification_bank_index", -1))
        if bank_index not in {0, 1, 2}:
            raise RuntimeError(
                "production qualification v3 must use its frozen CLI execution path"
            )
        expected_protocol = production_qualification_protocol_v3(bank_index)
        if (
            config != expected_protocol
            or production_protocol
            or not strict_runtime
            or stage_name != "phase1_ordering_cross_bank_qualification_v3"
            or frozen_protocol != expected_protocol
        ):
            raise RuntimeError(
                "production qualification v3 must use its frozen CLI execution path"
            )
    _validate_calibration_config(config, production_protocol)
    if frozen_protocol is not None and config != frozen_protocol:
        raise ValueError("config differs from its frozen protocol")
    fleet_path = _resolve_repo_path(str(config["fleet_module"]))
    environment_contract_path = _resolve_repo_path(
        str(config.get("environment_contract", "environment/phase1-washu-runtime.json"))
    )
    environment_contract = _load_json(environment_contract_path)
    requirements_lock_path = _resolve_repo_path(str(config["requirements_lock"]))
    cluster_wrapper_path = _resolve_repo_path(
        str(config.get("cluster_wrapper", "cluster/phase1_calibration.sbatch"))
    )
    runtime_inventory_path: Path | None = None
    expected_runtime_inventory: dict[str, Any] | None = None
    if strict_runtime:
        runtime_inventory_path = _resolve_repo_path(str(config["runtime_binary_inventory"]))
        expected_runtime_inventory = _load_json(runtime_inventory_path)
    fleet_sha_start = _sha256_file(fleet_path)
    if fleet_sha_start != str(config["fleet_sha256"]):
        raise RuntimeError(
            f"generator hash mismatch before import: {fleet_sha_start} != {config['fleet_sha256']}"
        )
    if fleet_path.name != "d4_generator.py":
        raise RuntimeError("calibration requires the pinned generator-only module")
    qualification_preregistration_path: Path | None = None
    if "qualification_preregistration" in config:
        qualification_preregistration_path = _resolve_repo_path(
            str(config["qualification_preregistration"])
        )
    extra_source_paths = (
        environment_contract_path,
        requirements_lock_path,
        cluster_wrapper_path,
    )
    if runtime_inventory_path is not None:
        extra_source_paths += (runtime_inventory_path,)
    if qualification_preregistration_path is not None:
        extra_source_paths += (qualification_preregistration_path,)
    source_inventory_start = _source_inventory(config_path, fleet_path, extra_source_paths)
    git = _git_provenance(bool(config.get("require_clean_git", False)))
    if "required_source_tag" in config:
        _verify_annotated_tag(str(config["required_source_tag"]), git["commit"])
    if strict_runtime and sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError("production calibration supports only Python 3.10 or 3.11")
    if strict_runtime:
        observed_contract = {
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        expected_contract = {
            key: environment_contract[key]
            for key in (
                "interpreter",
                "python",
                "numpy",
                "torch",
                "torch_cuda",
                "cublas_workspace_config",
                "pythonhashseed",
                "omp_num_threads",
                "mkl_num_threads",
                "openblas_num_threads",
                "deterministic_algorithms",
                "float32_matmul_precision",
                "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
            )
        }
        if observed_contract != expected_contract:
            raise RuntimeError(
                f"environment contract mismatch: {observed_contract} != {expected_contract}"
            )
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            capture_output=True,
        )
        if pip_check.returncode != 0:
            raise RuntimeError(f"pip check failed: {pip_check.stdout}{pip_check.stderr}")

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration was requested but CUDA is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUDA calibration requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.cuda.reset_peak_memory_stats(device)
    if strict_runtime:
        if device.type != "cuda" or expected_runtime_inventory is None:
            raise RuntimeError("production calibration requires its locked CUDA runtime")
        observed_runtime_inventory = verify_runtime_binary_inventory(
            expected_runtime_inventory, device
        )
        contract_crosscheck = {
            "gpu_name": observed_runtime_inventory["gpu_name"],
            "gpu_capability": observed_runtime_inventory["gpu_capability"],
            "cuda_driver": observed_runtime_inventory["cuda_driver"],
            "runtime_binary_fingerprint": observed_runtime_inventory[
                "runtime_binary_fingerprint"
            ],
            "cpu_identity_sha256": _sha256_json(
                observed_runtime_inventory["cpu_identity"]
            ),
            "numpy_runtime_sha256": hashlib.sha256(
                observed_runtime_inventory["numpy_runtime"].encode()
            ).hexdigest(),
            "requirements_lock_sha256": _sha256_file(requirements_lock_path),
        }
        if any(
            environment_contract.get(name) != value
            for name, value in contract_crosscheck.items()
        ):
            raise RuntimeError("WashU runtime contract cross-check failed")
    else:
        observed_runtime_inventory = {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
            "device_request": device_name,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32
            if hasattr(torch.backends, "cuda")
            else None,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32
            if hasattr(torch.backends, "cudnn")
            else None,
        }
        observed_runtime_inventory["runtime_binary_fingerprint"] = _sha256_json(
            observed_runtime_inventory
        )
    execution_contract = {
        "runtime_binary_fingerprint": observed_runtime_inventory[
            "runtime_binary_fingerprint"
        ],
        "device_request": device_name,
        "gpu_name": observed_runtime_inventory.get("gpu_name"),
        "gpu_capability": observed_runtime_inventory.get("gpu_capability"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }

    priors = [str(value) for value in config["priors"]]
    candidates = [int(value) for value in config["truncation_candidates"]]
    reference = int(config["reference_truncation"])
    levels = candidates + [reference]
    calibration_stream_seeds = {
        prior: int(config["calibration_seed_root"]) + 10_000 * index
        for index, prior in enumerate(priors)
    }
    calibration_seeds = set(calibration_stream_seeds.values()) | {int(config["atom_seed"])}
    forbidden_seeds = {
        int(value)
        for value in config["reserved_confirmatory_seeds"]
        + config["known_persisted_fixed_seeds"]
    }
    overlap = sorted(calibration_seeds & forbidden_seeds)
    if overlap:
        raise RuntimeError(f"calibration seed namespace overlap: {overlap}")

    output_dir = output_dir.resolve()
    running_path = output_dir / "RUNNING.json"
    complete_path = output_dir / "COMPLETE.json"
    identity = {
        "config_sha256": config_sha_start,
        "fleet_sha256": fleet_sha_start,
        "source_inventory": source_inventory_start,
        "git_commit": git["commit"],
        "execution_contract": execution_contract,
    }
    identity_sha256 = _sha256_json(identity)
    running_identity = {"identity": identity, "identity_sha256": identity_sha256}
    if complete_path.exists():
        raise FileExistsError(f"completed calibration attempt already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    lease_path = output_dir / "ATTEMPT.lock"
    _acquire_attempt_lease(lease_path, identity_sha256, recover_stale_lease)
    existing_without_lease = [path for path in output_dir.iterdir() if path != lease_path]
    if existing_without_lease:
        if not running_path.exists() or _load_json(running_path) != running_identity:
            raise FileExistsError(f"nonmatching nonempty calibration directory: {output_dir}")
    else:
        write_json_atomic(running_path, running_identity)
    atom_count = int(config["atom_count"])
    oracle_dtype_name = str(config.get("oracle_internal_dtype", "float32"))
    oracle_dtype_by_name = {"float32": torch.float32, "float64": torch.float64}
    if oracle_dtype_name not in oracle_dtype_by_name:
        raise RuntimeError(f"unsupported oracle_internal_dtype: {oracle_dtype_name}")
    oracle_compute_dtype = oracle_dtype_by_name[oracle_dtype_name]
    memory = _memory_preflight(
        device,
        atom_count,
        reference,
        int(config["query_grid_chunk"]),
        int(config["context_atom_batch"]),
        8 if oracle_compute_dtype == torch.float64 else 4,
    )
    fleet = load_fleet_module(fleet_path)
    quadrature_values, quadrature_bins, quadrature_log_weights = quadrature_grid(
        fleet, config
    )
    archive_predictive_arrays = bool(config.get("archive_predictive_arrays", False))
    qualification_protocol_version = int(
        config.get("qualification_protocol_version", -1)
    )
    if archive_predictive_arrays and qualification_protocol_version not in {2, 3}:
        raise RuntimeError("predictive-array archival is restricted to qualification v2/v3")
    archive_quadrature_reference = qualification_protocol_version == 3
    prediction_bin_count = int(fleet.N_BINS) if archive_predictive_arrays else None
    reference_quadrature: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if archive_quadrature_reference:
        reference_quadrature = quadrature_grid(
            fleet,
            {
                "quadrature_interior_nodes": int(
                    config["quadrature_reference_interior_nodes"]
                ),
                "quadrature_tail_nodes": int(
                    config["quadrature_reference_tail_nodes"]
                ),
            },
        )
    atoms_started = time.time()
    atoms = sample_sigmas_exact(
        fleet, np.random.default_rng(int(config["atom_seed"])), atom_count
    )
    atom_bank = {
        "schema_version": 1,
        "seed": int(config["atom_seed"]),
        "count": atom_count,
        "shape": list(atoms.shape),
        "dtype": atoms.dtype.str,
        "sha256": _sha256_array(atoms),
    }
    canary_count = int(config.get("atom_determinism_canary_count", 0))
    if canary_count:
        canary_seed = int(config["atom_determinism_canary_seed"])
        canary_atoms = sample_sigmas_exact(
            fleet, np.random.default_rng(canary_seed), canary_count
        )
        atom_bank["determinism_canary"] = {
            "seed": canary_seed,
            "count": canary_count,
            "shape": list(canary_atoms.shape),
            "dtype": canary_atoms.dtype.str,
            "sha256": _sha256_array(canary_atoms),
        }
        del canary_atoms
    atom_bank_path = output_dir / "ATOM_BANK.json"
    if atom_bank_path.exists():
        if _load_json(atom_bank_path) != atom_bank:
            raise RuntimeError("regenerated atom bank differs from the recorded bank")
    else:
        if any((output_dir / f"partial_{prior}.npz").exists() for prior in priors):
            raise RuntimeError("partial diagnostics exist without an atom-bank identity")
        write_json_atomic(atom_bank_path, atom_bank)
    atom_build_seconds = time.time() - atoms_started
    oracle_started = time.time()
    oracle = OrderingOracle(
        fleet,
        atoms,
        device=device,
        context_atom_batch=int(config["context_atom_batch"]),
        compute_dtype=oracle_compute_dtype,
    )
    oracle_build_seconds = time.time() - oracle_started

    n_prior = len(priors)
    n_candidate = len(candidates)
    n_level = len(levels)
    n_context = int(config["calibration_contexts_per_prior"])
    combined = {
        name: np.empty((n_prior,) + value.shape, dtype=value.dtype)
        for name, value in _partial_arrays(
            n_level,
            n_candidate,
            n_context,
            identity_sha256,
            prediction_bin_count,
            archive_quadrature_reference,
        ).items()
    }
    probability_atol = float(config["thresholds"]["probability_sum_atol"])
    for prior_index, prior in enumerate(priors):
        partial_path = output_dir / f"partial_{prior}.npz"
        partial = _load_partial(
            partial_path,
            _partial_arrays(
                n_level,
                n_candidate,
                n_context,
                identity_sha256,
                prediction_bin_count,
                archive_quadrature_reference,
            ),
        )
        stream = generate_evaluation_stream(
            fleet, prior, n_context, 30, calibration_stream_seeds[prior]
        )
        if prior == "C":
            r = float(fleet.R_OF["C"])
            gaussian = False
        elif prior == "N":
            r = 2.0
            gaussian = True
        else:
            raise ValueError(f"unsupported prior arm: {prior}")
        for context_index in range(n_context):
            if partial["completed"][context_index] == 1:
                continue
            likelihood = oracle.context_log_likelihood(
                stream["contexts"][context_index], r, gaussian
            )
            ess_full, ess_ablated = oracle.collapsed_atom_ess(likelihood)
            partial["ess_full_atoms"][context_index] = ess_full
            partial["ess_ablated_atoms"][context_index] = ess_ablated
            predictions: list[OraclePrediction] = []
            for level_index, truncation in enumerate(levels):
                prediction = oracle.predict_from_log_likelihood(
                    likelihood,
                    stream["queries"][context_index],
                    r,
                    gaussian,
                    truncation,
                    quadrature_values,
                    quadrature_bins,
                    quadrature_log_weights,
                    int(config["query_grid_chunk"]),
                    probability_atol,
                )
                predictions.append(prediction)
                partial["keep_full"][level_index, context_index] = prediction.keep_full
                partial["keep_ablated"][level_index, context_index] = prediction.keep_ablated
                partial["full_probability_sum_error"][level_index, context_index] = abs(
                    float(prediction.full.sum()) - 1.0
                )
                partial["ablated_probability_sum_error"][level_index, context_index] = abs(
                    float(prediction.ablated.sum()) - 1.0
                )
                partial["full_probability_minimum"][level_index, context_index] = float(
                    prediction.full.min()
                )
                partial["ablated_probability_minimum"][level_index, context_index] = float(
                    prediction.ablated.min()
                )
            outcome_bin = int(stream["outcome_bins"][context_index])
            if archive_predictive_arrays:
                partial["outcome_bins"][context_index] = outcome_bin
                partial["full_probability"][:, context_index] = np.stack(
                    [prediction.full for prediction in predictions]
                )
                partial["ablated_probability"][:, context_index] = np.stack(
                    [prediction.ablated for prediction in predictions]
                )
            if archive_quadrature_reference:
                if reference_quadrature is None:
                    raise AssertionError("qualification v3 reference grid is missing")
                reference_grid_predictions = [
                    oracle.predict_from_log_likelihood(
                        likelihood,
                        stream["queries"][context_index],
                        r,
                        gaussian,
                        truncation,
                        reference_quadrature[0],
                        reference_quadrature[1],
                        reference_quadrature[2],
                        int(config["query_grid_chunk"]),
                        probability_atol,
                    )
                    for truncation in levels
                ]
                for prediction in predictions + reference_grid_predictions:
                    if np.any(prediction.full <= 0.0) or np.any(
                        prediction.ablated <= 0.0
                    ):
                        raise RuntimeError(
                            "qualification v3 quadrature probabilities must be strictly positive"
                        )
                partial["quadrature_grid_full_probability"][0, :, context_index] = (
                    np.stack([prediction.full for prediction in predictions])
                )
                partial["quadrature_grid_full_probability"][1, :, context_index] = (
                    np.stack(
                        [prediction.full for prediction in reference_grid_predictions]
                    )
                )
                partial["quadrature_grid_ablated_probability"][0, :, context_index] = (
                    np.stack([prediction.ablated for prediction in predictions])
                )
                partial["quadrature_grid_ablated_probability"][1, :, context_index] = (
                    np.stack(
                        [prediction.ablated for prediction in reference_grid_predictions]
                    )
                )
                quadrature_limits = config["quadrature_qualification"]
                probability_floor = float(
                    quadrature_limits["reference_probability_floor"]
                )
                for level_index, (
                    production_grid_prediction,
                    reference_grid_prediction,
                ) in enumerate(zip(predictions, reference_grid_predictions, strict=True)):
                    for predictor in ("full", "ablated"):
                        production_probability = getattr(
                            production_grid_prediction, predictor
                        )
                        reference_probability = getattr(
                            reference_grid_prediction, predictor
                        )
                        partial[f"quadrature_js_{predictor}"][
                            level_index, context_index
                        ] = jensen_shannon(
                            production_probability, reference_probability
                        )
                        log_change = np.abs(
                            np.log(production_probability)
                            - np.log(reference_probability)
                        )
                        active = reference_probability >= probability_floor
                        if not np.any(active):
                            raise RuntimeError(
                                "quadrature reference has no bin above its floor"
                            )
                        partial[f"quadrature_max_bin_abs_logp_change_{predictor}"][
                            level_index, context_index
                        ] = float(np.max(log_change[active]))
                        partial[
                            f"quadrature_reference_weighted_abs_logp_change_{predictor}"
                        ][level_index, context_index] = float(
                            np.sum(reference_probability * log_change)
                        )
                    production_value = np.log(production_grid_prediction.full) - np.log(
                        production_grid_prediction.ablated
                    )
                    reference_value = np.log(reference_grid_prediction.full) - np.log(
                        reference_grid_prediction.ablated
                    )
                    value_active = (
                        reference_grid_prediction.full >= probability_floor
                    ) & (reference_grid_prediction.ablated >= probability_floor)
                    if not np.any(value_active):
                        raise RuntimeError(
                            "quadrature ordering value has no bin above its floor"
                        )
                    partial["quadrature_max_bin_abs_ordering_value_change"][
                        level_index, context_index
                    ] = float(
                        np.max(np.abs(production_value[value_active] - reference_value[value_active]))
                    )
            for candidate_index in range(n_candidate):
                current = predictions[candidate_index]
                reference_prediction = predictions[-1]
                partial["js_full"][candidate_index, context_index] = jensen_shannon(
                    current.full, reference_prediction.full
                )
                partial["js_ablated"][candidate_index, context_index] = jensen_shannon(
                    current.ablated, reference_prediction.ablated
                )
                partial["abs_logp_change_full"][candidate_index, context_index] = (
                    absolute_log_probability_change(
                        current.full, reference_prediction.full, outcome_bin
                    )
                )
                partial["abs_logp_change_ablated"][candidate_index, context_index] = (
                    absolute_log_probability_change(
                        current.ablated, reference_prediction.ablated, outcome_bin
                    )
                )
            partial["completed"][context_index] = 1
            write_numeric_npz_atomic(partial_path, **partial)
            del likelihood, predictions
        if not np.all(partial["completed"] == 1):
            raise RuntimeError(f"prior arm did not complete: {prior}")
        for name in combined:
            combined[name][prior_index] = partial[name]

    raw_path = output_dir / "calibration_raw.npz"
    raw_metadata = {
        "prior_codes": np.arange(n_prior, dtype=np.int64),
        "candidates": np.asarray(candidates, dtype=np.int64),
        "reference_truncation": np.asarray([reference], dtype=np.int64),
    }
    if archive_quadrature_reference:
        raw_metadata.update(
            {
                "quadrature_grid_interior_nodes": np.asarray(
                    [
                        int(config["quadrature_interior_nodes"]),
                        int(config["quadrature_reference_interior_nodes"]),
                    ],
                    dtype=np.int64,
                ),
                "quadrature_grid_tail_nodes": np.asarray(
                    [
                        int(config["quadrature_tail_nodes"]),
                        int(config["quadrature_reference_tail_nodes"]),
                    ],
                    dtype=np.int64,
                ),
                "quadrature_truncation_levels": np.asarray(
                    levels, dtype=np.int64
                ),
            }
        )
    write_numeric_npz_atomic(raw_path, **raw_metadata, **combined)

    candidate_rows: list[dict[str, Any]] = []
    selected: int | None = None
    for candidate_index, candidate in enumerate(candidates):
        prior_rows: dict[str, Any] = {}
        passes = True
        for prior_index, prior in enumerate(priors):
            passed, row = _calibration_candidate_passes(
                combined["js_full"][prior_index, candidate_index],
                combined["js_ablated"][prior_index, candidate_index],
                combined["abs_logp_change_full"][prior_index, candidate_index],
                combined["abs_logp_change_ablated"][prior_index, candidate_index],
                config["thresholds"],
            )
            passes = passes and passed
            prior_rows[prior] = row
        candidate_rows.append(
            {
                "truncation": candidate,
                "compared_with": reference,
                "priors": prior_rows,
                "pass_all_priors": passes,
            }
        )
        if selected is None and passes:
            selected = candidate

    quadrature_qualification: dict[str, Any] | None = None
    if archive_quadrature_reference:
        quadrature_limits = config["quadrature_qualification"]
        quadrature_pass_all = True
        prior_quadrature_rows: dict[str, Any] = {}
        for prior_index, prior in enumerate(priors):
            predictor_rows: dict[str, Any] = {}
            for predictor in ("full", "ablated"):
                max_js = float(
                    np.max(combined[f"quadrature_js_{predictor}"][prior_index])
                )
                max_bin_logp = float(
                    np.max(
                        combined[
                            f"quadrature_max_bin_abs_logp_change_{predictor}"
                        ][prior_index]
                    )
                )
                max_weighted_logp = float(
                    np.max(
                        combined[
                            "quadrature_reference_weighted_abs_logp_change_"
                            f"{predictor}"
                        ][prior_index]
                    )
                )
                passed = bool(
                    max_js <= float(quadrature_limits["js_max"])
                    and max_bin_logp
                    <= float(quadrature_limits["max_bin_abs_logp_change_max"])
                    and max_weighted_logp
                    <= float(
                        quadrature_limits[
                            "reference_weighted_abs_logp_change_max"
                        ]
                    )
                )
                quadrature_pass_all = quadrature_pass_all and passed
                predictor_rows[predictor] = {
                    "max_js": max_js,
                    "max_bin_abs_logp_change": max_bin_logp,
                    "max_reference_weighted_abs_logp_change": max_weighted_logp,
                    "pass": passed,
                }
            max_ordering_value_change = float(
                np.max(
                    combined[
                        "quadrature_max_bin_abs_ordering_value_change"
                    ][prior_index]
                )
            )
            ordering_value_pass = bool(
                max_ordering_value_change
                <= float(
                    quadrature_limits[
                        "max_bin_abs_ordering_value_change_max"
                    ]
                )
            )
            quadrature_pass_all = quadrature_pass_all and ordering_value_pass
            prior_quadrature_rows[prior] = {
                "predictors": predictor_rows,
                "max_bin_abs_ordering_value_change": max_ordering_value_change,
                "ordering_value_pass": ordering_value_pass,
                "pass": bool(
                    ordering_value_pass
                    and all(row["pass"] for row in predictor_rows.values())
                ),
            }
        quadrature_qualification = {
            "production_grid": {
                "interior_nodes": int(config["quadrature_interior_nodes"]),
                "tail_nodes": int(config["quadrature_tail_nodes"]),
            },
            "reference_grid": {
                "interior_nodes": int(
                    config["quadrature_reference_interior_nodes"]
                ),
                "tail_nodes": int(config["quadrature_reference_tail_nodes"]),
            },
            "limits": quadrature_limits,
            "priors": prior_quadrature_rows,
            "pass": quadrature_pass_all,
        }
        if not quadrature_pass_all:
            selected = None

    if _sha256_file(config_path) != config_sha_start:
        raise RuntimeError("config changed during calibration")
    if _sha256_file(fleet_path) != fleet_sha_start:
        raise RuntimeError("generator changed during calibration")
    if _source_inventory(config_path, fleet_path, extra_source_paths) != source_inventory_start:
        raise RuntimeError("executable source inventory changed during calibration")
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_config": torch.__config__.show(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "environment_contract_name": environment_contract_path.name,
        "environment_contract_sha256": _sha256_file(environment_contract_path),
        "runtime_binary_fingerprint": observed_runtime_inventory[
            "runtime_binary_fingerprint"
        ],
        "execution_contract": execution_contract,
        "installed_distributions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
    }
    summary = {
        "schema_version": 1,
        "stage": stage_name,
        "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER",
        "scientific_endpoints_computed": False,
        "decision": "CALIBRATION_PASS" if selected is not None else "CALIBRATION_FAIL",
        "selected_truncation": selected,
        "candidate_results": candidate_rows,
        "quadrature_qualification": quadrature_qualification,
        "diagnostics": {
            prior: {
                "median_ess_full_atoms": float(
                    np.median(combined["ess_full_atoms"][prior_index])
                ),
                "median_ess_ablated_atoms": float(
                    np.median(combined["ess_ablated_atoms"][prior_index])
                ),
                "mean_keep_full_by_level": [
                    float(np.mean(row)) for row in combined["keep_full"][prior_index]
                ],
                "mean_keep_ablated_by_level": [
                    float(np.mean(row)) for row in combined["keep_ablated"][prior_index]
                ],
            }
            for prior_index, prior in enumerate(priors)
        },
        "prior_index_to_name": {str(index): prior for index, prior in enumerate(priors)},
        "truncation_levels": levels,
        "config_name": config_path.name,
        "config_sha256": config_sha_start,
        "fleet_name": fleet_path.name,
        "fleet_sha256": fleet_sha_start,
        "source_inventory": source_inventory_start,
        "attempt_identity_sha256": identity_sha256,
        "git": git,
        "calibration_seed_namespace": str(config["calibration_seed_namespace"]),
        "calibration_stream_seeds": calibration_stream_seeds,
        "atom_seed": int(config["atom_seed"]),
        "atom_bank": atom_bank,
        "reserved_confirmatory_seeds": [
            int(value) for value in config["reserved_confirmatory_seeds"]
        ],
        "raw_name": raw_path.name,
        "raw_sha256": _sha256_file(raw_path),
        "atom_count": atom_count,
        "oracle_internal_dtype": oracle_dtype_name,
        "atom_build_seconds": atom_build_seconds,
        "oracle_build_seconds": oracle_build_seconds,
        "wall_seconds": time.time() - started,
        "device": str(device),
        "memory_preflight": memory,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "environment": environment,
    }
    summary_path = output_dir / "calibration_summary.json"
    write_json_atomic(summary_path, summary)
    payload_paths = [raw_path, summary_path, running_path, atom_bank_path] + [
        output_dir / f"partial_{prior}.npz" for prior in priors
    ]
    complete = {
        "identity": identity,
        "identity_sha256": identity_sha256,
        "artifacts": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in payload_paths
        },
        "decision": summary["decision"],
    }
    write_json_atomic(complete_path, complete)
    lease_path.unlink()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="run the oracle-only information-barrier calibration"
    )
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    calibrate.add_argument("--recover-stale-lease", action="store_true")
    qualify = subparsers.add_parser(
        "qualify", help="run one frozen cross-bank oracle qualification shard"
    )
    qualify.add_argument("--config", type=Path, required=True)
    qualify.add_argument("--out", type=Path, required=True)
    qualify.add_argument("--device", default="cuda")
    qualify.add_argument("--recover-stale-lease", action="store_true")
    inventory = subparsers.add_parser(
        "runtime-inventory", help="fingerprint the exact WashU CUDA runtime"
    )
    inventory.add_argument("--out", type=Path, required=True)
    inventory.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "calibrate":
        summary = run_calibration(
            arguments.config,
            arguments.out,
            arguments.device,
            recover_stale_lease=arguments.recover_stale_lease,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        # A completed scientific calibration failure is not a process failure.
        # Exceptions and incomplete artifacts still exit nonzero.
        return 0
    if arguments.command == "qualify":
        registered = _load_json(arguments.config.resolve())
        bank_index = int(registered.get("qualification_bank_index", -1))
        protocol_version = int(registered.get("qualification_protocol_version", 1))
        if protocol_version == 1:
            frozen_protocol = production_qualification_protocol(bank_index)
            stage_name = "phase1_ordering_cross_bank_qualification"
        elif protocol_version == 2:
            raise RuntimeError(
                "qualification v2 is blocked-before-execution and superseded by v3"
            )
        elif protocol_version == 3:
            frozen_protocol = production_qualification_protocol_v3(bank_index)
            stage_name = "phase1_ordering_cross_bank_qualification_v3"
        else:
            raise ValueError(f"unsupported qualification protocol version: {protocol_version}")
        summary = run_calibration(
            arguments.config,
            arguments.out,
            arguments.device,
            production_protocol=False,
            recover_stale_lease=arguments.recover_stale_lease,
            frozen_protocol=frozen_protocol,
            strict_runtime=True,
            stage_name=stage_name,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if arguments.command == "runtime-inventory":
        configure_runtime(0)
        device = torch.device(arguments.device)
        inventory = build_runtime_binary_inventory(device)
        write_json_atomic(arguments.out.resolve(), inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())
