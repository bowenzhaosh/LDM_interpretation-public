import base64
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .registry import (
    load_checkpoint_registry,
    package_source_hashes,
    sha256_file,
    validation_input_hashes,
)
from .query_bank import FIXED_SENSITIVITY_BANK


REQUIRED_VALIDATIONS = {
    "artifacts/validation/instrument_primary.json",
    "artifacts/validation/instrument_sensitivity.json",
    "artifacts/validation/legacy_oracle_primary.json",
    "artifacts/validation/legacy_oracle_sensitivity.json",
    "artifacts/validation/bootstrap_coverage.json",
    "artifacts/validation/batch_shape.json",
    "artifacts/validation/smoke_budget.json",
}
RUN_LOCK_SETTINGS = {
    "groups": 256,
    "continuations": 8,
    "max_core_candidates": 2000,
    "max_blocks_per_core": 512,
    "min_within_group_sd": 0.25,
    "bootstrap_replicates": 10000,
    "permutation_replicates": 2000,
    "interval_quantiles": [0.02, 0.98],
    "model_batch_size": 64,
    "smoke_stream": {"label": "smoke-v3", "groups": 8, "continuations": 8},
}
RUN_LOCK_CLAIM_SCOPE = "selected identifiable-interior AL40 replace-10 regime"
RUN_LOCK_KEYS = {
    "schema_version",
    "files",
    "required_validations",
    "settings",
    "claim_scope",
}


def expected_run_lock_files(root: Path | None = None) -> set[str]:
    root = repository_root() if root is None else Path(root).resolve()
    fixed = {
        "PREREG.md",
        "MAPPING_QUALIFICATION_PREREG.md",
        "HARNESS.md",
        "AUDIT.md",
        "README.md",
        "SOURCE_ORIGIN.md",
        "pyproject.toml",
        "config/query_bank.json",
        "config/checkpoint_registry.json",
        "config/audit_readiness.json",
        "environment/requirements-lock.txt",
        "environment/runtime.json",
        "environment/installed-distributions.json",
        *REQUIRED_VALIDATIONS,
    }
    runtime_code = {
        path.relative_to(root).as_posix()
        for parent in (root / "src", root / "tests")
        for path in parent.rglob("*.py")
    }
    return fixed | runtime_code

EVALUATION_DOMAIN = "pfn-dag-essential-evaluation-v2"
WALL_CAP_SECONDS = 45 * 60
MEMORY_CAP_BYTES = 16 * 2**30
RAW_CAP_BYTES = 2 * 2**30

EXPECTED_AUDIT_DISPOSITIONS = {
    "metric_validity": "SOUND",
    "numerical_determinism": "SOUND",
    "configuration_plumbing": "SOUND",
    "replay_portability": "SOUND",
    "bias_audit": "AMENDMENT_JUSTIFIED",
}
AUDIT_EVIDENCE_FILES = {
    lens: f"artifacts/audit/{lens}.json" for lens in EXPECTED_AUDIT_DISPOSITIONS
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scientific_run_directory(
    commit_sha: str, repo: str | Path | None = None
) -> Path:
    root = repository_root() if repo is None else Path(repo).resolve()
    if len(commit_sha) < 7:
        raise ValueError("scientific run directory requires a full commit SHA")
    return root / "runs" / f"scientific-{commit_sha[:7]}"


def require_scientific_run_path(
    path: str | Path,
    *,
    commit_sha: str,
    relative: str | Path,
    repo: str | Path | None = None,
) -> Path:
    root = repository_root() if repo is None else Path(repo).resolve()
    expected = scientific_run_directory(commit_sha, root) / relative
    observed = Path(os.path.abspath(path))
    if observed != expected:
        raise ValueError(f"scientific path must be commit-named: {expected}")
    relative_observed = observed.relative_to(root)
    cursor = root
    for part in relative_observed.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"scientific path cannot traverse a symlink: {cursor}")
    return observed


def audit_readiness_subject_files(root: str | Path | None = None) -> set[str]:
    root_path = repository_root() if root is None else Path(root).resolve()
    fixed = {
        "PREREG.md",
        "MAPPING_QUALIFICATION_PREREG.md",
        "HARNESS.md",
        "AUDIT.md",
        "README.md",
        "SOURCE_ORIGIN.md",
        "pyproject.toml",
        "config/query_bank.json",
        "config/checkpoint_registry.json",
        "environment/requirements-lock.txt",
        "environment/runtime.json",
        "environment/installed-distributions.json",
        *AUDIT_EVIDENCE_FILES.values(),
        *REQUIRED_VALIDATIONS,
    }
    python_files = {
        path.relative_to(root_path).as_posix()
        for parent in (root_path / "src", root_path / "tests")
        for path in parent.rglob("*.py")
    }
    return fixed | python_files


def audit_review_subject_files(root: str | Path | None = None) -> set[str]:
    root_path = repository_root() if root is None else Path(root).resolve()
    fixed = {
        "PREREG.md",
        "MAPPING_QUALIFICATION_PREREG.md",
        "HARNESS.md",
        "README.md",
        "SOURCE_ORIGIN.md",
        "pyproject.toml",
        "config/query_bank.json",
        "config/checkpoint_registry.json",
        "environment/requirements-lock.txt",
        "environment/runtime.json",
        "environment/installed-distributions.json",
        *REQUIRED_VALIDATIONS,
    }
    python_files = {
        path.relative_to(root_path).as_posix()
        for parent in (root_path / "src", root_path / "tests")
        for path in parent.rglob("*.py")
    }
    return fixed | python_files


def validate_audit_evidence(repo: str | Path | None = None) -> dict[str, dict]:
    root = repository_root() if repo is None else Path(repo).resolve()
    reviewed_hashes = {
        relative: sha256_file(root / relative)
        for relative in sorted(audit_review_subject_files(root))
    }
    evidence = {}
    expected_keys = {
        "schema_version",
        "lens",
        "disposition",
        "reviewed_subject_sha256s",
        "blocking_findings",
        "major_findings",
        "summary",
        "reviewer_record",
    }
    for lens, relative in AUDIT_EVIDENCE_FILES.items():
        value = json.loads((root / relative).read_text())
        if (
            set(value) != expected_keys
            or value.get("schema_version") != 1
            or value.get("lens") != lens
            or value.get("disposition") != EXPECTED_AUDIT_DISPOSITIONS[lens]
            or value.get("reviewed_subject_sha256s") != reviewed_hashes
            or value.get("blocking_findings") != []
            or value.get("major_findings") != []
            or not isinstance(value.get("summary"), str)
            or not value["summary"].strip()
            or not isinstance(value.get("reviewer_record"), str)
            or not value["reviewer_record"].strip()
        ):
            raise ValueError(f"audit evidence mismatch: {lens}")
        evidence[lens] = value
    return evidence


def validate_audit_readiness(repo: str | Path | None = None) -> dict:
    root = repository_root() if repo is None else Path(repo).resolve()
    path = root / "config" / "audit_readiness.json"
    value = json.loads(path.read_text())
    expected_hashes = {
        relative: sha256_file(root / relative)
        for relative in sorted(audit_readiness_subject_files(root))
    }
    expected_failure = {
        "commit_sha": "d0b049d6241845e55443f4950e52b70644b2b1ab",
        "status": "BLOCKED_GUARD",
        "prediction_shards": 0,
        "scientific_metrics": 0,
    }
    audit_evidence = validate_audit_evidence(root)
    expected_audit_hashes = {
        lens: sha256_file(root / relative)
        for lens, relative in AUDIT_EVIDENCE_FILES.items()
    }
    if (
        value.get("schema_version") != 1
        or value.get("protocol_version") != 3
        or value.get("disposition") != "READY_V3"
        or value.get("scientific_outputs_observed") is not False
        or value.get("test_command") != "python -m pytest -q tests"
        or not isinstance(value.get("tests_collected"), int)
        or value.get("tests_collected", 0) < 1
        or not isinstance(value.get("tests_passed"), int)
        or value.get("tests_passed") != value.get("tests_collected")
        or not isinstance(value.get("test_output_sha256"), str)
        or len(value.get("test_output_sha256", "")) != 64
        or value.get("required_validations") != sorted(REQUIRED_VALIDATIONS)
        or value.get("audits")
        != {lens: record["disposition"] for lens, record in audit_evidence.items()}
        or value.get("audit_evidence_sha256s") != expected_audit_hashes
        or value.get("failed_stream") != expected_failure
        or value.get("subject_sha256s") != expected_hashes
    ):
        raise ValueError("audit readiness attestation mismatch")
    if "Disposition: `READY_V3`" not in (root / "AUDIT.md").read_text():
        raise ValueError("human audit disposition is not READY_V3")
    return value


def evaluation_root(commit_sha: str) -> int:
    """Return the preregistered unsigned 64-bit evaluation root."""
    if len(commit_sha) < 7:
        raise ValueError("commit_sha is required")
    payload = (commit_sha + EVALUATION_DOMAIN).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def derive_seed(commit_sha: str, label: str) -> int:
    """Derive a labeled unsigned 64-bit child from the locked root."""
    if not label:
        raise ValueError("seed label is required")
    root = evaluation_root(commit_sha).to_bytes(8, "big", signed=False)
    payload = root + label.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def current_head(repo: str | Path) -> str:
    repo = Path(repo)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clean_head(repo: str | Path) -> str:
    repo = Path(repo)
    head = current_head(repo)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("scientific evaluation requires a clean committed worktree")
    return head


def load_run_lock(repo: str | Path | None = None) -> tuple[Path, dict]:
    root = repository_root() if repo is None else Path(repo).resolve()
    path = root / "config" / "run_lock.json"
    if not path.is_file():
        raise FileNotFoundError("scientific run lock is missing")
    lock = json.loads(path.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported run-lock schema")
    return path, lock


def load_locked_query_banks(repo: str | Path | None = None) -> np.ndarray:
    root = repository_root() if repo is None else Path(repo).resolve()
    path = root / "config" / "query_bank.json"
    value = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "seed": 820001,
        "n_contexts": 512,
        "quadrature": 15,
        "oracle_grid_size": 3097,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"locked query-bank metadata mismatch: {key}")
    primary = np.asarray(value.get("selected_queries"), dtype=np.float64)
    candidate = np.asarray(value.get("candidate_queries"), dtype=np.float64)
    trace = np.asarray(value.get("objective_trace"), dtype=np.float64)
    if primary.shape != (8,) or candidate.shape != (17,) or trace.shape != (4,):
        raise ValueError("locked query-bank arrays have the wrong shape")
    if not np.isfinite(primary).all() or not np.isfinite(candidate).all() or not np.isfinite(trace).all():
        raise ValueError("locked query-bank metadata is non-finite")
    if not np.all(np.diff(trace) > 0):
        raise ValueError("locked query-bank objective trace is not strictly increasing")
    if value.get("identifiable_fraction") != 511 / 512:
        raise ValueError("locked query-bank calibration fraction mismatch")
    context_hash = value.get("context_sha256", "")
    if len(context_hash) != 64 or any(character not in "0123456789abcdef" for character in context_hash):
        raise ValueError("locked calibration context hash is malformed")
    sensitivity = np.asarray(FIXED_SENSITIVITY_BANK, dtype=np.float64)
    if sensitivity.shape != (8,) or np.array_equal(primary, sensitivity):
        raise ValueError("primary and sensitivity query-bank arms must be distinct")
    return np.stack([primary, sensitivity], axis=0)


def verify_runtime(repo: str | Path | None = None) -> dict:
    root = repository_root() if repo is None else Path(repo).resolve()
    expected = json.loads((root / "environment" / "runtime.json").read_text())
    actual = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "torch": importlib.metadata.version("torch"),
        "backend": "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads_locked": torch.get_num_threads(),
        "torch_num_interop_threads_locked": torch.get_num_interop_threads(),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
    }
    for key, observed in actual.items():
        if str(expected.get(key)) != str(observed):
            raise RuntimeError(f"runtime fingerprint mismatch for {key}: {observed}")
    if expected.get("backend") != "cpu" or expected.get("deterministic_algorithms") is not True:
        raise RuntimeError("runtime lock does not require deterministic CPU execution")
    installed = json.loads(
        (root / "environment" / "installed-distributions.json").read_text()
    )
    executable = installed.get("python_executable", {})
    if (
        Path(sys.executable).stat().st_size != executable.get("size")
        or sha256_file(sys.executable) != executable.get("sha256")
    ):
        raise RuntimeError("Python executable does not match the installed-runtime lock")
    distributions = installed.get("distributions", {})
    module_names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "torch": "torch",
    }
    for name in ("numpy", "scipy", "scikit-learn", "torch"):
        record = distributions.get(name, {})
        distribution = importlib.metadata.distribution(name)
        record_relative = next(
            item for item in distribution.files if str(item).endswith(".dist-info/RECORD")
        )
        record_path = Path(distribution.locate_file(record_relative))
        if (
            distribution.version != record.get("version")
            or record_path.stat().st_size != record.get("record_size")
            or sha256_file(record_path) != record.get("record_sha256")
        ):
            raise RuntimeError(f"installed distribution record mismatch: {name}")
        verified_paths = _verify_distribution_payloads(
            distribution, record_path, name, record
        )
        module = importlib.import_module(module_names[name])
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if module_path not in verified_paths:
            raise RuntimeError(f"imported module origin is outside locked distribution: {name}")
    return actual


def _verify_distribution_payloads(
    distribution, record_path: Path, name: str, locked_record: dict
) -> set[Path]:
    verified = set()
    payload_entries = []
    payload_bytes = 0
    with record_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise RuntimeError(f"malformed installed distribution RECORD: {name}")
        relative, encoded_digest, encoded_size = row
        path = Path(distribution.locate_file(relative)).resolve()
        if not path.is_file():
            raise RuntimeError(f"installed distribution payload missing: {name}:{relative}")
        relative_path = Path(relative)
        mutable_cache = "__pycache__" in relative_path.parts or relative_path.suffix == ".pyc"
        if mutable_cache:
            continue
        if encoded_size:
            try:
                expected_size = int(encoded_size)
            except ValueError as error:
                raise RuntimeError(
                    f"malformed installed distribution size: {name}:{relative}"
                ) from error
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"installed distribution size mismatch: {name}:{relative}")
        with path.open("rb") as payload:
            actual_digest = hashlib.file_digest(payload, "sha256").digest()
        if encoded_digest:
            try:
                algorithm, encoded = encoded_digest.split("=", 1)
            except ValueError as error:
                raise RuntimeError(
                    f"malformed installed distribution digest: {name}:{relative}"
                ) from error
            if algorithm != "sha256":
                raise RuntimeError(
                    f"unsupported installed distribution digest: {name}:{relative}"
                )
            expected_digest = base64.urlsafe_b64decode(encoded + "==")
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"installed distribution payload mismatch: {name}:{relative}"
                )
        payload_entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": actual_digest.hex(),
            }
        )
        payload_bytes += path.stat().st_size
        verified.add(path)
    if record_path.resolve() not in verified:
        raise RuntimeError(f"installed distribution RECORD is not self-listed: {name}")
    tree_digest = hashlib.sha256()
    for entry in sorted(payload_entries, key=lambda value: value["path"]):
        tree_digest.update(
            (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    if (
        locked_record.get("payload_files") != len(payload_entries)
        or locked_record.get("payload_bytes") != payload_bytes
        or locked_record.get("payload_tree_sha256") != tree_digest.hexdigest()
    ):
        raise RuntimeError(f"installed distribution payload tree mismatch: {name}")
    return verified


def _require_pair(values, name: str) -> tuple[float, float]:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must be a two-element list")
    low, high = (float(values[0]), float(values[1]))
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"{name} is not a finite ordered interval")
    return low, high


def validate_locked_validations(
    repo: str | Path | None = None,
    *,
    query_banks: np.ndarray | None = None,
) -> dict[str, dict]:
    root = repository_root() if repo is None else Path(repo).resolve()
    banks = load_locked_query_banks(root) if query_banks is None else np.asarray(query_banks)
    if banks.shape != (2, 8):
        raise ValueError("validation query banks must have shape (2, 8)")
    values = {}
    for relative in sorted(REQUIRED_VALIDATIONS):
        path = root / relative
        value = json.loads(path.read_text())
        if value.get("schema_version") != 1 or value.get("pass") is not True:
            raise ValueError(f"required validation did not pass: {relative}")
        values[relative] = value

    expected_implementation_hashes = package_source_hashes()
    expected_input_hashes = validation_input_hashes()
    load_checkpoint_registry(root / "config" / "checkpoint_registry.json")
    for relative, value in values.items():
        if value.get("input_sha256s") != expected_input_hashes:
            raise ValueError(f"validation input hash mismatch: {relative}")

    for index, role in enumerate(("primary", "sensitivity")):
        instrument = values[f"artifacts/validation/instrument_{role}.json"]
        if instrument.get("producer_sha256") != sha256_file(
            root / "src" / "pfn_dag_verify" / "validation.py"
        ):
            raise ValueError(f"instrument producer hash mismatch: {role}")
        if instrument.get("implementation_sha256s") != expected_implementation_hashes:
            raise ValueError(f"instrument implementation hash mismatch: {role}")
        if instrument.get("bank_role") != role:
            raise ValueError(f"instrument validation role mismatch: {role}")
        if instrument.get("quadrature") != 15 or instrument.get("unit_contexts") != 64:
            raise ValueError(f"instrument validation scope mismatch: {role}")
        np.testing.assert_array_equal(np.asarray(instrument.get("query_bank")), banks[index])
        if instrument.get("oracle_grid_size") != 3097:
            raise ValueError(f"instrument oracle grid mismatch: {role}")
        canary_low, canary_high = _require_pair(instrument.get("canary_interval"), "canary")
        numeric_limits = {
            "max_scalar_vector_ell_error": 1e-10,
            "max_scalar_vector_endpoint_error": 1e-10,
            "max_coordinate_weight_error": 1e-4,
            "max_kl_weight_error": 1e-3,
            "coordinate_kl_g_median": 0.01,
            "coordinate_kl_g_p95": 0.05,
            "max_tempered_slope_error": 0.02,
            "max_tempered_intercept_error": 0.02,
            "max_label_swap_error": 1e-10,
            "exact_reconstruction_residual": 1e-10,
        }
        numeric_pass = all(
            np.isfinite(float(instrument.get(key, np.inf)))
            and 0 <= float(instrument[key]) <= limit
            for key, limit in numeric_limits.items()
        )
        canary_pass = (
            canary_low >= -0.15
            and canary_high <= 0.15
            and canary_low <= 0 <= canary_high
        )
        recomputed_pass = bool(
            numeric_pass
            and canary_pass
            and instrument.get("degenerate_endpoint_fails_closed") is True
        )
        if not canary_pass:
            raise ValueError(f"instrument canary mismatch: {role}")
        if instrument.get("pass") is not recomputed_pass or not recomputed_pass:
            raise ValueError(f"instrument numeric thresholds failed: {role}")

        legacy = values[f"artifacts/validation/legacy_oracle_{role}.json"]
        if legacy.get("producer_sha256") != sha256_file(
            root / "src" / "pfn_dag_verify" / "legacy_compare.py"
        ):
            raise ValueError(f"legacy producer hash mismatch: {role}")
        if legacy.get("implementation_sha256s") != expected_implementation_hashes:
            raise ValueError(f"legacy implementation hash mismatch: {role}")
        if legacy.get("bank_role") != role or legacy.get("n_contexts") != 8:
            raise ValueError(f"legacy validation scope mismatch: {role}")
        np.testing.assert_array_equal(np.asarray(legacy.get("queries")), banks[index])
        expected_legacy_label = "artifacts/legacy/stage1_functional_law.py"
        if legacy.get("legacy_file") != expected_legacy_label:
            raise ValueError(f"legacy source path mismatch: {role}")
        legacy_path = root / expected_legacy_label
        if not legacy_path.is_file() or sha256_file(legacy_path) != legacy.get("legacy_sha256"):
            raise ValueError(f"legacy source snapshot mismatch: {role}")
        legacy_errors = np.asarray(
            [
                legacy.get("max_ell_error", np.inf),
                legacy.get("max_f0_error", np.inf),
                legacy.get("max_f1_error", np.inf),
            ],
            dtype=np.float64,
        )
        legacy_pass = bool(
            np.isfinite(legacy_errors).all()
            and np.all(legacy_errors >= 0)
            and np.max(legacy_errors) <= 1e-10
        )
        if legacy.get("pass") is not legacy_pass or not legacy_pass:
            raise ValueError(f"legacy numeric thresholds failed: {role}")

    coverage = values["artifacts/validation/bootstrap_coverage.json"]
    if coverage.get("producer_sha256") != sha256_file(
        root / "src" / "pfn_dag_verify" / "validation.py"
    ):
        raise ValueError("bootstrap producer hash mismatch")
    if coverage.get("implementation_sha256s") != expected_implementation_hashes:
        raise ValueError("bootstrap implementation hash mismatch")
    if (
        coverage.get("datasets_per_slope") != 500
        or coverage.get("bootstraps_per_dataset") != 1000
        or coverage.get("groups") != 256
        or coverage.get("percentile_interval") != [0.02, 0.98]
        or coverage.get("validation_seed") != 850002
        or set(coverage.get("results", {})) != {"0.8", "1.0", "1.2"}
    ):
        raise ValueError("bootstrap validation scope mismatch")
    coverage_pass = True
    for value in coverage["results"].values():
        low, high = _require_pair(value.get("coverage_wilson_95"), "coverage Wilson")
        estimate = float(value.get("coverage", np.nan))
        width = float(value.get("mean_interval_width", np.nan))
        coverage_pass = coverage_pass and bool(
            np.isfinite([estimate, width]).all()
            and 0 <= estimate <= 1
            and width > 0
            and estimate >= 0.93
            and low <= 0.95 <= high
        )
    if coverage.get("pass") is not coverage_pass or not coverage_pass:
        raise ValueError("bootstrap numeric thresholds failed")

    batch_shape = values["artifacts/validation/batch_shape.json"]
    if batch_shape.get("producer_sha256") != sha256_file(
        root / "src" / "pfn_dag_verify" / "validation.py"
    ):
        raise ValueError("batch-shape producer hash mismatch")
    if batch_shape.get("implementation_sha256s") != expected_implementation_hashes:
        raise ValueError("batch-shape implementation hash mismatch")
    expected_batch_shape = {
        "validation_kind": "fixed-production-shape-v2",
        "validation_seed": 860003,
        "companion_seed": 860004,
        "batch_permutation_seed": 860005,
        "row_permutation_seeds": {"core20": 860006, "length30": 860007},
        "contexts_per_kind": 64,
        "context_kinds": ["core20", "length30"],
        "context_rows": {"core20": 20, "length30": 30},
        "production_batch_size": 64,
        "identities_checked": 32,
        "banks_checked": 2,
        "registry_sha256": sha256_file(root / "config" / "checkpoint_registry.json"),
    }
    for key, expected in expected_batch_shape.items():
        if batch_shape.get(key) != expected:
            raise ValueError(f"batch-shape validation scope mismatch: {key}")
    np.testing.assert_array_equal(np.asarray(batch_shape.get("query_banks")), banks)
    bank_hash = hashlib.sha256(np.ascontiguousarray(banks).tobytes()).hexdigest()
    if batch_shape.get("query_banks_sha256") != bank_hash:
        raise ValueError("batch-shape query-bank hash mismatch")
    for key in ("sample_sha256s", "companion_sample_sha256s"):
        digests = batch_shape.get(key, {})
        if set(digests) != {"core20", "length30"}:
            raise ValueError(f"batch-shape context hashes incomplete: {key}")
        for digest in digests.values():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"batch-shape context hash malformed: {key}")
    records = batch_shape.get("records", [])
    registry_value = json.loads((root / "config" / "checkpoint_registry.json").read_text())
    checkpoint_hashes = {
        (int(record["seed"]), int(record["step"])): record["sha256"]
        for record in registry_value["checkpoints"]
    }
    expected_identities = {
        (seed, step, bank, context_kind)
        for seed in range(16)
        for step in (0, 12_000)
        for bank in (0, 1)
        for context_kind in ("core20", "length30")
    }
    observed_identities = {
        (
            record.get("seed"),
            record.get("step"),
            record.get("bank_index"),
            record.get("context_kind"),
        )
        for record in records
    }
    record_pass = bool(
        len(records) == 128
        and observed_identities == expected_identities
        and all(
            record.get("pass") is True
            and record.get("production_replay_byte_identical") is True
            and record.get("batch_axis_permutation_byte_identical") is True
            and record.get("companion_replacement_byte_identical") is True
            and record.get("checkpoint_sha256")
            == checkpoint_hashes.get((record.get("seed"), record.get("step")))
            and float(record.get("max_batch_axis_permutation_error", np.inf)) == 0.0
            and float(record.get("max_companion_replacement_error", np.inf)) == 0.0
            and 0.0 <= float(record.get("max_row_permutation_error", np.inf)) <= 1e-6
            and 0.0
            <= float(record.get("descriptive_max_batch_1_vs_64_error", np.inf))
            < np.inf
            and record.get("focal_contexts_checked") == 64
            and record.get("companion_variants") == 16
            and record.get("companion_group_max_errors") == [0.0] * 16
            and record.get("context_kind") in ("core20", "length30")
            for record in records
        )
    )
    record_maxima = {
        "max_batch_axis_permutation_error": max(
            float(record["max_batch_axis_permutation_error"]) for record in records
        ),
        "max_companion_replacement_error": max(
            float(record["max_companion_replacement_error"]) for record in records
        ),
        "max_row_permutation_error": max(
            float(record["max_row_permutation_error"]) for record in records
        ),
        "descriptive_max_batch_1_vs_64_error": max(
            float(record["descriptive_max_batch_1_vs_64_error"]) for record in records
        ),
    } if records else {}
    aggregate_pass = bool(
        batch_shape.get("production_replay_byte_identical") is True
        and batch_shape.get("batch_axis_permutation_byte_identical") is True
        and batch_shape.get("companion_replacement_byte_identical") is True
        and float(batch_shape.get("max_batch_axis_permutation_error", np.inf)) == 0.0
        and float(batch_shape.get("max_companion_replacement_error", np.inf)) == 0.0
        and 0.0 <= float(batch_shape.get("max_row_permutation_error", np.inf)) <= 1e-6
        and 0.0
        <= float(batch_shape.get("descriptive_max_batch_1_vs_64_error", np.inf))
        < np.inf
        and all(batch_shape.get(key) == value for key, value in record_maxima.items())
    )
    batch_shape_pass = bool(record_pass and aggregate_pass)
    if batch_shape.get("pass") is not batch_shape_pass or not batch_shape_pass:
        raise ValueError("batch-shape numeric thresholds failed")

    smoke = values["artifacts/validation/smoke_budget.json"]
    if smoke.get("producer_sha256") != sha256_file(
        root / "src" / "pfn_dag_verify" / "smoke.py"
    ):
        raise ValueError("smoke producer hash mismatch")
    if smoke.get("implementation_sha256s") != expected_implementation_hashes:
        raise ValueError("smoke implementation hash mismatch")
    expected_smoke = {
        "smoke_kind": "interior-selected-end-to-end-v3",
        "smoke_stream_label": "smoke-v3",
        "smoke_groups": 8,
        "smoke_continuations": 8,
        "max_core_candidates": 2000,
        "max_blocks_per_core": 512,
        "min_within_group_sd": 0.25,
        "guard_records": 4,
        "guard_sample_source": "dedicated-deterministic-smoke",
        "projected_scientific_guard_records": 128,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "memory_cap_bytes": MEMORY_CAP_BYTES,
        "raw_cap_bytes": RAW_CAP_BYTES,
    }
    for key, expected in expected_smoke.items():
        if smoke.get(key) != expected:
            raise ValueError(f"smoke-budget scope mismatch: {key}")
    enforce_cost_gate(smoke)
    multipliers = smoke.get("projection_multipliers", {})
    components = smoke.get("components", {})
    smoke_pass = bool(
        smoke.get("eligible_replace_rows") == 64
        and smoke.get("prediction_shards") == 4
        and multipliers
        == {
            "group_ratio": 32.0,
            "shard_ratio": 16.0,
            "guard_ratio": 32.0,
            "safety_factor": 1.25,
        }
        and float(components.get("guard_wall_seconds", -1.0)) > 0.0
        and float(components.get("non_guard_score_wall_seconds", -1.0)) > 0.0
        and float(smoke["projected_wall_seconds"]) <= WALL_CAP_SECONDS
        and int(smoke["projected_peak_rss_bytes"]) <= MEMORY_CAP_BYTES
        and int(smoke["projected_raw_bytes"]) <= RAW_CAP_BYTES
    )
    if smoke.get("pass") is not smoke_pass or not smoke_pass:
        raise ValueError("smoke numeric thresholds failed")
    return values


def recompute_smoke_projections(smoke: dict) -> tuple[float, int, int]:
    multipliers = smoke.get("projection_multipliers", {})
    components = smoke.get("components", {})
    required_multipliers = ("group_ratio", "shard_ratio", "guard_ratio", "safety_factor")
    required_components = (
        "panel_wall_seconds",
        "score_wall_seconds",
        "guard_wall_seconds",
        "non_guard_score_wall_seconds",
        "derive_wall_seconds",
        "panel_bytes",
        "prediction_shard_bytes",
        "prediction_metadata_bytes",
        "prediction_bytes",
        "derived_bytes",
        "replay_bundle_bytes",
        "tracked_repository_bytes",
        "measured_peak_rss_bytes",
    )
    try:
        numeric_multipliers = {
            key: float(multipliers[key]) for key in required_multipliers
        }
        numeric_components = {
            key: float(components[key]) for key in required_components
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("BLOCKED_COST: incomplete projection components") from error
    if not all(
        np.isfinite(value) and value > 0 for value in numeric_multipliers.values()
    ) or not all(
        np.isfinite(value) and value >= 0 for value in numeric_components.values()
    ):
        raise RuntimeError("BLOCKED_COST: non-finite or negative projection components")
    if (
        numeric_components["guard_wall_seconds"] <= 0
        or numeric_components["non_guard_score_wall_seconds"] <= 0
        or not np.isclose(
            numeric_components["score_wall_seconds"],
            numeric_components["guard_wall_seconds"]
            + numeric_components["non_guard_score_wall_seconds"],
            rtol=1e-12,
            atol=1e-9,
        )
        or int(numeric_components["prediction_bytes"])
        != int(numeric_components["prediction_shard_bytes"])
        + int(numeric_components["prediction_metadata_bytes"])
    ):
        raise RuntimeError("BLOCKED_COST: inconsistent measured timing or byte components")
    group_ratio = numeric_multipliers["group_ratio"]
    shard_ratio = numeric_multipliers["shard_ratio"]
    guard_ratio = numeric_multipliers["guard_ratio"]
    safety = numeric_multipliers["safety_factor"]
    inference_ratio = group_ratio * shard_ratio
    projected_wall = safety * (
        numeric_components["panel_wall_seconds"] * group_ratio
        + numeric_components["guard_wall_seconds"] * guard_ratio
        + numeric_components["non_guard_score_wall_seconds"] * inference_ratio
        + numeric_components["derive_wall_seconds"] * inference_ratio
    )
    projected_raw = int(
        safety
        * (
            numeric_components["panel_bytes"] * group_ratio
            + numeric_components["prediction_shard_bytes"] * inference_ratio
            + numeric_components["prediction_metadata_bytes"] * guard_ratio
            + numeric_components["derived_bytes"] * inference_ratio
            + numeric_components["replay_bundle_bytes"]
            + numeric_components["tracked_repository_bytes"]
            + 10 * 2**20
        )
    )
    projected_peak = int(
        max(
            numeric_components["measured_peak_rss_bytes"] * 4,
            numeric_components["panel_bytes"] * group_ratio * 3,
            512 * 2**20,
        )
    )
    return projected_wall, projected_peak, projected_raw


def enforce_cost_gate(smoke: dict) -> None:
    projected = (
        float(smoke.get("projected_wall_seconds", float("inf"))),
        int(smoke.get("projected_peak_rss_bytes", MEMORY_CAP_BYTES + 1)),
        int(smoke.get("projected_raw_bytes", RAW_CAP_BYTES + 1)),
    )
    recomputed = recompute_smoke_projections(smoke)
    if (
        not np.isfinite(projected[0])
        or projected[0] < 0
        or not np.isclose(projected[0], recomputed[0], rtol=1e-12, atol=1e-9)
        or projected[1] < 0
        or projected[1] != recomputed[1]
        or projected[2] < 0
        or projected[2] != recomputed[2]
    ):
        raise RuntimeError("BLOCKED_COST: reported projection does not match components")
    if projected[0] > WALL_CAP_SECONDS:
        raise RuntimeError("BLOCKED_COST: projected wall-clock limit exceeded")
    if projected[1] > MEMORY_CAP_BYTES:
        raise RuntimeError("BLOCKED_COST: projected memory limit exceeded")
    if projected[2] > RAW_CAP_BYTES:
        raise RuntimeError("BLOCKED_COST: projected raw-storage limit exceeded")


def validate_run_lock_metadata(lock: dict) -> None:
    if set(lock) != RUN_LOCK_KEYS or lock.get("schema_version") != 1:
        raise ValueError("run lock has an unexpected top-level schema")
    if lock.get("required_validations") != sorted(REQUIRED_VALIDATIONS):
        raise ValueError("run lock does not name the exact required validation set")
    if lock.get("settings") != RUN_LOCK_SETTINGS:
        raise ValueError("run-lock settings differ from the scientific specification")
    if lock.get("claim_scope") != RUN_LOCK_CLAIM_SCOPE:
        raise ValueError("run-lock claim scope differs from the scientific specification")


def verify_run_lock(repo: str | Path | None = None) -> tuple[str, str, dict]:
    root = repository_root() if repo is None else Path(repo).resolve()
    if root != repository_root():
        raise ValueError("scientific stages must use the repository containing the package")
    head = clean_head(root)
    lock_path, lock = load_run_lock(root)
    validate_run_lock_metadata(lock)
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("run lock has no file manifest")
    expected_files = expected_run_lock_files(root)
    if set(files) != expected_files:
        raise ValueError("run lock does not contain the exact audited source and guard set")
    for relative, expected_hash in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"run-lock file mismatch: {relative}")
    load_locked_query_banks(root)
    verify_runtime(root)
    validate_locked_validations(root)
    validate_audit_readiness(root)
    from .mapping_qualification import verify_completed_qualification

    verify_completed_qualification(root=root, commit_sha=head)
    return head, sha256_file(lock_path), lock


def decode_ascii(array) -> str:
    return bytes(array.tolist()).decode("ascii")


def verify_panel_lock(panel: dict, repo: str | Path | None = None) -> tuple[str, dict]:
    head, lock_hash, lock = verify_run_lock(repo)
    if decode_ascii(panel["commit_sha"]) != head:
        raise ValueError("panel commit does not match clean HEAD")
    if "run_lock_sha256" not in panel:
        raise ValueError("panel does not embed the run-lock hash")
    if bytes(panel["run_lock_sha256"].tolist()).hex() != lock_hash:
        raise ValueError("panel run-lock hash mismatch")
    if int(panel.get("scientific", np.asarray(0))) != 1:
        raise ValueError("panel is not marked scientific")
    settings = lock["settings"]
    expected_shape = (settings["groups"], settings["continuations"], 10, 2)
    if panel.get("continuations", np.empty(0)).shape != expected_shape:
        raise ValueError("scientific panel dimensions do not match the run lock")
    if int(panel.get("selection_mode", np.asarray(0))) != 1:
        raise ValueError("scientific panel did not use selected-interior sampling")
    namespace = bytes(panel.get("selection_seed_namespace", np.empty(0, dtype=np.uint8)).tolist()).decode(
        "utf-8"
    )
    seed_root = bytes(panel.get("selection_seed_root", np.empty(0, dtype=np.uint8)).tolist()).decode(
        "ascii"
    )
    if namespace or seed_root != head:
        raise ValueError("scientific panel seed domain does not match clean HEAD")
    if (
        int(panel.get("selection_max_core_candidates", -1))
        != settings["max_core_candidates"]
        or int(panel.get("selection_max_blocks_per_core", -1))
        != settings["max_blocks_per_core"]
        or float(panel.get("selection_min_within_group_sd", np.nan))
        != settings["min_within_group_sd"]
    ):
        raise ValueError("scientific selection settings do not match the run lock")
    if not np.array_equal(panel.get("query_banks"), load_locked_query_banks(repo)):
        raise ValueError("scientific panel query banks do not match the run lock")
    eligible = np.asarray(panel.get("eligible_replace"), dtype=bool)
    if eligible.shape != (settings["groups"], settings["continuations"]) or not eligible.all():
        raise ValueError("scientific replace panel is not fully eligible")
    reasons = np.asarray(panel.get("candidate_core_reason"))
    ranks = np.asarray(panel.get("candidate_core_acceptance_rank"))
    accepted_ids = np.flatnonzero(reasons == 0)
    if accepted_ids.size != settings["groups"] or not np.array_equal(
        ranks[accepted_ids], np.arange(settings["groups"])
    ):
        raise ValueError("scientific panel is not the first ranked accepted cohort")
    np.testing.assert_array_equal(panel["core"], panel["candidate_core_context"][accepted_ids])
    np.testing.assert_array_equal(panel["sigma"], panel["candidate_core_sigma"][accepted_ids])
    np.testing.assert_array_equal(panel["graph"], panel["candidate_core_graph"][accepted_ids])
    np.testing.assert_array_equal(panel["params"], panel["candidate_core_params"][accepted_ids])
    core_seeds = np.asarray(panel.get("candidate_core_seed"))
    if core_seeds.dtype != np.uint64 or not np.array_equal(
        core_seeds,
        np.asarray(
            [derive_seed(head, f"core-candidate:{index}") for index in range(len(core_seeds))],
            dtype=np.uint64,
        ),
    ):
        raise ValueError("scientific core child-seed record is invalid")
    block_core = np.asarray(panel.get("candidate_block_core_index"))
    block_index = np.asarray(panel.get("candidate_block_index"))
    block_seeds = np.asarray(panel.get("candidate_block_seed"))
    if block_seeds.dtype != np.uint64 or not np.array_equal(
        block_seeds,
        np.asarray(
            [
                derive_seed(head, f"core-candidate:{core}:block:{index}")
                for core, index in zip(block_core, block_index, strict=True)
            ],
            dtype=np.uint64,
        ),
    ):
        raise ValueError("scientific block child-seed record is invalid")
    block_reason = np.asarray(panel.get("candidate_block_reason"))
    block_rank = np.asarray(panel.get("candidate_block_eligible_rank"))
    block_context = np.asarray(panel.get("candidate_block_context"))
    for panel_index, core_id in enumerate(accepted_ids):
        rows = np.flatnonzero((block_core == core_id) & (block_reason == 0))
        order = rows[np.argsort(block_rank[rows])]
        if order.size != settings["continuations"] + 1 or not np.array_equal(
            block_rank[order], np.arange(settings["continuations"] + 1)
        ):
            raise ValueError("accepted core does not contain the first nine eligible blocks")
        np.testing.assert_array_equal(panel["reference"][panel_index], block_context[order[0]])
        np.testing.assert_array_equal(panel["continuations"][panel_index], block_context[order[1:]])
    return head, lock
