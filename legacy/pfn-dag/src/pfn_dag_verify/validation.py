import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .constants import (
    BATCH_COMPANION_SEED,
    BATCH_PERMUTATION_SEED,
    BATCH_VALIDATION_SEED,
    BOOTSTRAP_COVERAGE_SEED,
    CORE_ROW_PERMUTATION_SEED,
    LENGTH30_ROW_PERMUTATION_SEED,
    PERMUTATION_CANARY_SEED,
    PERMUTATION_FIXTURE_SEED,
    UNIT_CONTEXT_VALIDATION_SEED,
)
from .generative import generate_group
from .evaluation import production_shape_diagnostics
from .instrument import (
    UnidentifiableError,
    project_coordinate,
    project_kl,
    reconstruct_updated_prediction,
)
from .oracle import GridOracle, ScalarGridOracle
from .query_bank import FIXED_SENSITIVITY_BANK
from .model import configure_determinism, load_registered_checkpoint
from .provenance import verify_runtime
from .registry import (
    expanded_checkpoint_record,
    load_checkpoint_registry,
    package_source_hashes,
    sha256_file,
    validation_input_hashes,
)
from .statistics import crossed_bootstrap_slope, permutation_null_slopes


def _unit_contexts(count: int = 64):
    rng = np.random.default_rng(UNIT_CONTEXT_VALIDATION_SEED)
    contexts = []
    while len(contexts) < count:
        group = generate_group(rng, n_continuations=2)
        contexts.extend(
            [
                group.core,
                np.concatenate([group.core, group.reference], axis=0),
                np.concatenate([group.core, group.continuations[0]], axis=0),
                np.concatenate([group.core, group.continuations[1]], axis=0),
            ]
        )
    return contexts[:count]


def _batch_validation_contexts(seed: int, count: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    core_contexts = []
    target_contexts = []
    for _ in range(count):
        group = generate_group(rng, n_continuations=2)
        core_contexts.append(group.core)
        target_contexts.append(
            np.concatenate([group.core, group.continuations[0]], axis=0)
        )
    return {
        "core20": np.stack(core_contexts, axis=0),
        "length30": np.stack(target_contexts, axis=0),
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_batch_shape(
    *,
    query_banks: np.ndarray,
    registry_path: Path,
) -> dict:
    """Validate production-shape invariants on a fixed disjoint context stream."""

    started = time.perf_counter()
    configure_determinism(0)
    verify_runtime()
    query_banks = np.asarray(query_banks, dtype=np.float64)
    if query_banks.shape != (2, 8):
        raise ValueError("batch-shape validation requires two eight-query banks")
    validation_seed = BATCH_VALIDATION_SEED
    companion_seed = BATCH_COMPANION_SEED
    batch_permutation_seed = BATCH_PERMUTATION_SEED
    row_permutation_seeds = {
        "core20": CORE_ROW_PERMUTATION_SEED,
        "length30": LENGTH30_ROW_PERMUTATION_SEED,
    }
    samples = _batch_validation_contexts(validation_seed)
    companion_samples = _batch_validation_contexts(companion_seed)
    batch_permutation = np.random.default_rng(batch_permutation_seed).permutation(64)
    row_permutations = {
        kind: np.random.default_rng(row_permutation_seeds[kind]).permutation(
            sample.shape[1]
        )
        for kind, sample in samples.items()
    }
    registry = load_checkpoint_registry(registry_path)
    records = []
    for seed in range(16):
        for step in (0, 12_000):
            checkpoint = expanded_checkpoint_record(registry, seed, step)
            model = load_registered_checkpoint(checkpoint)
            for bank_index, queries in enumerate(query_banks):
                for context_kind, sample in samples.items():
                    records.append(
                        {
                            "seed": seed,
                            "step": step,
                            "bank_index": bank_index,
                            "context_kind": context_kind,
                            "checkpoint_sha256": checkpoint["sha256"],
                            **production_shape_diagnostics(
                                model,
                                sample,
                                companion_samples[context_kind],
                                queries,
                                batch_permutation=batch_permutation,
                                row_permutation=row_permutations[context_kind],
                            ),
                        }
                    )
            del model
    passed = bool(len(records) == 128 and all(record["pass"] for record in records))
    return {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__)),
        "implementation_sha256s": package_source_hashes(),
        "input_sha256s": validation_input_hashes(),
        "validation_kind": "fixed-production-shape-v2",
        "validation_seed": validation_seed,
        "companion_seed": companion_seed,
        "batch_permutation_seed": batch_permutation_seed,
        "row_permutation_seeds": row_permutation_seeds,
        "contexts_per_kind": 64,
        "context_kinds": list(samples),
        "context_rows": {kind: int(sample.shape[1]) for kind, sample in samples.items()},
        "production_batch_size": 64,
        "identities_checked": 32,
        "banks_checked": 2,
        "sample_sha256s": {
            kind: _array_sha256(sample) for kind, sample in samples.items()
        },
        "companion_sample_sha256s": {
            kind: _array_sha256(sample) for kind, sample in companion_samples.items()
        },
        "query_banks": query_banks.tolist(),
        "query_banks_sha256": _array_sha256(query_banks),
        "registry_sha256": sha256_file(registry_path),
        "production_replay_byte_identical": all(
            record["production_replay_byte_identical"] for record in records
        ),
        "batch_axis_permutation_byte_identical": all(
            record["batch_axis_permutation_byte_identical"] for record in records
        ),
        "companion_replacement_byte_identical": all(
            record["companion_replacement_byte_identical"] for record in records
        ),
        "max_batch_axis_permutation_error": max(
            record["max_batch_axis_permutation_error"] for record in records
        ),
        "max_companion_replacement_error": max(
            record["max_companion_replacement_error"] for record in records
        ),
        "max_row_permutation_error": max(
            record["max_row_permutation_error"] for record in records
        ),
        "descriptive_max_batch_1_vs_64_error": max(
            record["descriptive_max_batch_1_vs_64_error"] for record in records
        ),
        "records": records,
        "pass": passed,
        "wall_seconds": time.perf_counter() - started,
    }


def validate_instrument(queries: np.ndarray, quadrature: int = 15, bank_role: str = "primary"):
    configure_determinism(0)
    verify_runtime()
    started = time.perf_counter()
    vector = GridOracle(queries=queries, quadrature=quadrature)
    scalar = ScalarGridOracle(queries=queries, quadrature=quadrature)
    ell_errors = []
    endpoint_errors = []
    bundles = []
    for context in _unit_contexts(64):
        fast = vector.evaluate(context)
        slow = scalar.evaluate(context)
        bundles.append(fast)
        ell_errors.append(abs(fast.ell - slow.ell))
        endpoint_errors.extend(
            [
                float(np.max(np.abs(fast.f0 - slow.f0))),
                float(np.max(np.abs(fast.f1 - slow.f1))),
            ]
        )
    all_oracle_errors = np.asarray(ell_errors + endpoint_errors, dtype=np.float64)
    if all_oracle_errors.size == 0 or not np.isfinite(all_oracle_errors).all():
        raise FloatingPointError("oracle validation produced empty or non-finite errors")
    max_ell_error = float(np.max(ell_errors))
    max_endpoint_error = float(np.max(endpoint_errors))

    planted_weights = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    coordinate_errors = []
    kl_errors = []
    log_odds_disagreement = []
    label_swap_errors = []
    for bundle in bundles[:16]:
        for weight in planted_weights:
            prediction = (1.0 - weight) * bundle.f0 + weight * bundle.f1
            coordinate = project_coordinate(prediction, bundle.f0, bundle.f1)
            kl = project_kl(prediction, bundle.f0, bundle.f1)
            swapped = project_coordinate(prediction, bundle.f1, bundle.f0)
            coordinate_errors.append(abs(coordinate.w - weight))
            kl_errors.append(abs(kl.w - weight))
            log_odds_disagreement.append(abs(coordinate.g - kl.g))
            label_swap_errors.append(abs(swapped.w - (1.0 - coordinate.w)))
            label_swap_errors.append(abs(swapped.g + coordinate.g))

    ell_grid = np.linspace(-2.0, 2.0, 41)
    slope_errors = []
    intercept_errors = []
    fixture = bundles[0]
    for gain in (0.25, 0.50, 0.75, 1.0):
        recovered = []
        for ell in ell_grid:
            weight = 1.0 / (1.0 + np.exp(-gain * ell))
            prediction = (1.0 - weight) * fixture.f0 + weight * fixture.f1
            recovered.append(project_coordinate(prediction, fixture.f0, fixture.f1).g)
        slope, intercept = np.polyfit(ell_grid, recovered, 1)
        slope_errors.append(abs(float(slope) - gain))
        intercept_errors.append(abs(float(intercept)))

    base_weight = 0.37
    delta_ell = 0.6
    target_weight = 1.0 / (
        1.0 + np.exp(-(np.log(base_weight / (1.0 - base_weight)) + delta_ell))
    )
    target = bundles[1]
    target_prediction = (1.0 - target_weight) * target.f0 + target_weight * target.f1
    reconstruction = reconstruct_updated_prediction(
        target_prediction,
        target.f0,
        target.f1,
        w_base=base_weight,
        delta_ell=delta_ell,
    )

    rng = np.random.default_rng(PERMUTATION_FIXTURE_SEED)
    exact_change = rng.normal(size=(256, 8))
    model_change = np.broadcast_to(exact_change, (16, 256, 8)).copy()
    canary_draws = permutation_null_slopes(
        exact_change,
        model_change,
        n_permutations=2_000,
        rng=np.random.default_rng(PERMUTATION_CANARY_SEED),
    )
    canary_interval = np.quantile(canary_draws, [0.025, 0.975])

    degenerate_guard = False
    try:
        project_coordinate(fixture.f0, fixture.f0, fixture.f0)
    except UnidentifiableError:
        degenerate_guard = True

    disagreement = np.asarray(log_odds_disagreement)
    result = {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__)),
        "implementation_sha256s": package_source_hashes(),
        "input_sha256s": validation_input_hashes(),
        "quadrature": quadrature,
        "query_bank": np.asarray(queries).tolist(),
        "bank_role": bank_role,
        "oracle_grid_size": vector.grid_size,
        "unit_contexts": 64,
        "max_scalar_vector_ell_error": max_ell_error,
        "max_scalar_vector_endpoint_error": max_endpoint_error,
        "max_coordinate_weight_error": float(np.max(coordinate_errors)),
        "max_kl_weight_error": float(np.max(kl_errors)),
        "coordinate_kl_g_median": float(np.median(disagreement)),
        "coordinate_kl_g_p95": float(np.quantile(disagreement, 0.95)),
        "max_tempered_slope_error": float(np.max(slope_errors)),
        "max_tempered_intercept_error": float(np.max(intercept_errors)),
        "max_label_swap_error": float(np.max(label_swap_errors)),
        "exact_reconstruction_residual": reconstruction.normalized_residual,
        "canary_interval": canary_interval.tolist(),
        "degenerate_endpoint_fails_closed": degenerate_guard,
        "wall_seconds": time.perf_counter() - started,
    }
    result["pass"] = bool(
        max_ell_error <= 1e-10
        and max_endpoint_error <= 1e-10
        and result["max_coordinate_weight_error"] <= 1e-4
        and result["max_kl_weight_error"] <= 1e-3
        and result["coordinate_kl_g_median"] <= 0.01
        and result["coordinate_kl_g_p95"] <= 0.05
        and result["max_tempered_slope_error"] <= 0.02
        and result["max_tempered_intercept_error"] <= 0.02
        and result["max_label_swap_error"] <= 1e-10
        and result["exact_reconstruction_residual"] <= 1e-10
        and canary_interval[0] >= -0.15
        and canary_interval[1] <= 0.15
        and canary_interval[0] <= 0.0 <= canary_interval[1]
        and degenerate_guard
    )
    return result


def validate_bootstrap_coverage(
    *, datasets_per_slope: int = 500, bootstraps: int = 1_000, groups: int = 256
):
    configure_determinism(0)
    verify_runtime()
    started = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_COVERAGE_SEED)
    results = {}
    for beta in (0.8, 1.0, 1.2):
        covered = 0
        widths = []
        for _ in range(datasets_per_slope):
            x = rng.normal(size=(groups, 8))
            seed_gain = rng.normal(0.0, 0.15, size=(16, 1, 1))
            group_gain = rng.normal(0.0, 0.50, size=(1, groups, 1))
            noise = rng.normal(0.0, 0.10, size=(16, groups, 8))
            y = (beta + seed_gain + group_gain) * x[None, :, :] + noise
            draws = crossed_bootstrap_slope(x, y, n_boot=bootstraps, rng=rng)
            low, high = np.quantile(draws, [0.02, 0.98])
            covered += int(low <= beta <= high)
            widths.append(high - low)
        coverage = covered / datasets_per_slope
        z = 1.96
        center = (coverage + z * z / (2 * datasets_per_slope)) / (
            1 + z * z / datasets_per_slope
        )
        half_width = z * np.sqrt(
            coverage * (1 - coverage) / datasets_per_slope
            + z * z / (4 * datasets_per_slope * datasets_per_slope)
        ) / (1 + z * z / datasets_per_slope)
        results[str(beta)] = {
            "coverage": coverage,
            "coverage_wilson_95": [float(center - half_width), float(center + half_width)],
            "mean_interval_width": float(np.mean(widths)),
        }
    passed = (
        datasets_per_slope >= 500
        and bootstraps >= 1_000
        and groups == 256
        and all(
        value["coverage"] >= 0.93
        and value["coverage_wilson_95"][0] <= 0.95 <= value["coverage_wilson_95"][1]
        for value in results.values()
        )
    )
    return {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__)),
        "implementation_sha256s": package_source_hashes(),
        "input_sha256s": validation_input_hashes(),
        "datasets_per_slope": datasets_per_slope,
        "bootstraps_per_dataset": bootstraps,
        "groups": groups,
        "percentile_interval": [0.02, 0.98],
        "validation_seed": BOOTSTRAP_COVERAGE_SEED,
        "results": results,
        "pass": passed,
        "wall_seconds": time.perf_counter() - started,
    }


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    instrument = subparsers.add_parser("instrument")
    instrument.add_argument("--query-bank", type=Path, required=True)
    instrument.add_argument("--out", type=Path, required=True)
    instrument.add_argument("--bank", choices=("primary", "sensitivity"), default="primary")
    coverage = subparsers.add_parser("coverage")
    coverage.add_argument("--out", type=Path, required=True)
    coverage.add_argument("--datasets-per-slope", type=int, default=500)
    coverage.add_argument("--bootstraps", type=int, default=1_000)
    batch_shape = subparsers.add_parser("batch-shape")
    batch_shape.add_argument("--query-bank", type=Path, required=True)
    batch_shape.add_argument("--registry", type=Path, required=True)
    batch_shape.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "instrument":
        if args.bank == "primary":
            bank = json.loads(args.query_bank.read_text())["selected_queries"]
        else:
            bank = FIXED_SENSITIVITY_BANK
        result = validate_instrument(
            np.asarray(bank, dtype=np.float64), bank_role=args.bank
        )
    elif args.command == "coverage":
        result = validate_bootstrap_coverage(
            datasets_per_slope=args.datasets_per_slope,
            bootstraps=args.bootstraps,
        )
    else:
        primary = json.loads(args.query_bank.read_text())["selected_queries"]
        result = validate_batch_shape(
            query_banks=np.stack(
                [np.asarray(primary, dtype=np.float64), FIXED_SENSITIVITY_BANK]
            ),
            registry_path=args.registry,
        )
    _write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
