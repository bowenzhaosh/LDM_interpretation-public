from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import importlib.util
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pfn_dag_verify.phase1_ordering import (
    OrderingOracle,
    PRODUCTION_CALIBRATION_PROTOCOL,
    _acquire_attempt_lease,
    _calibration_candidate_passes,
    _partial_arrays,
    _sha256_json,
    _verify_annotated_tag,
    _validate_calibration_config,
    absolute_log_probability_change,
    generate_evaluation_stream,
    quadrature_grid,
    load_fleet_module,
    production_qualification_protocol,
    run_calibration,
    sample_sigmas_exact,
    validate_probability,
)


ROOT = Path(__file__).resolve().parents[1]
FLEET_PATH = ROOT / "artifacts" / "phase1" / "d4_generator.py"


def _fleet():
    return load_fleet_module(FLEET_PATH)


def test_observed_bin_log_probability_change_fails_closed_on_zero():
    candidate = np.array([0.0, 1.0], dtype=np.float64)
    reference = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(RuntimeError, match="finite and positive"):
        absolute_log_probability_change(candidate, reference, 0)
    assert absolute_log_probability_change(candidate, reference, 1) == 0.0


def test_required_source_tag_uses_exact_tag_namespace_and_commit_peel(monkeypatch):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        if arguments[1:3] == ["cat-file", "-t"]:
            return SimpleNamespace(stdout="tag\n")
        return SimpleNamespace(stdout="abc123\n")

    monkeypatch.setattr("pfn_dag_verify.phase1_ordering.subprocess.run", fake_run)
    _verify_annotated_tag("phase1-ordering-qualification-v3", "abc123")
    assert calls == [
        [
            "git",
            "cat-file",
            "-t",
            "refs/tags/phase1-ordering-qualification-v3",
        ],
        [
            "git",
            "rev-parse",
            "--verify",
            "refs/tags/phase1-ordering-qualification-v3^{commit}",
        ],
    ]


def test_vectorized_sigma_sampler_is_bit_identical_to_frozen_sampler():
    fleet = _fleet()
    expected = fleet.sample_Sigmas(np.random.default_rng(8123), 48)
    observed = sample_sigmas_exact(fleet, np.random.default_rng(8123), 48)
    np.testing.assert_array_equal(observed, expected)


def test_generator_only_module_is_bit_identical_to_archived_fleet_substrate():
    generator = _fleet()
    archived_path = ROOT / "artifacts" / "phase1" / "d4_train_fleet.py"
    spec = importlib.util.spec_from_file_location("archived_phase1_fleet", archived_path)
    assert spec is not None and spec.loader is not None
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    sigmas_generator = generator.sample_Sigmas(np.random.default_rng(441), 4)
    sigmas_archived = archived.sample_Sigmas(np.random.default_rng(441), 4)
    np.testing.assert_array_equal(sigmas_generator, sigmas_archived)
    for prior, r, gaussian in (("C", 4.0, False), ("N", 2.0, True)):
        for family in (0, 7, 23):
            observed = generator.gen_data(
                sigmas_generator[0], family, r, 31, np.random.default_rng(900 + family), gaussian
            )
            expected = archived.gen_data(
                sigmas_archived[0], family, r, 31, np.random.default_rng(900 + family), gaussian
            )
            np.testing.assert_array_equal(observed, expected, err_msg=prior)


def test_gaussian_ordering_null_is_exact_with_full_support():
    fleet = _fleet()
    atoms = sample_sigmas_exact(fleet, np.random.default_rng(17), 96)
    stream = generate_evaluation_stream(fleet, "N", 1, 30, 991)
    oracle = OrderingOracle(fleet, atoms, torch.device("cpu"), context_atom_batch=32)
    likelihood = oracle.context_log_likelihood(stream["contexts"][0], 2.0, True)
    config = {"quadrature_interior_nodes": 4, "quadrature_tail_nodes": 8}
    values, bins, log_weights = quadrature_grid(fleet, config)
    prediction = oracle.predict_from_log_likelihood(
        likelihood,
        stream["queries"][0],
        2.0,
        True,
        len(atoms),
        values,
        bins,
        log_weights,
        8,
        1e-8,
    )
    validate_probability(prediction.full, "full")
    validate_probability(prediction.ablated, "ablated")
    np.testing.assert_allclose(prediction.full, prediction.ablated, atol=2e-6, rtol=0)
    assert abs(prediction.keep_full - 1.0) < 2e-6
    assert abs(prediction.keep_ablated - 1.0) < 2e-6
    np.testing.assert_allclose(
        prediction.ordering_posterior,
        np.full(24, 1.0 / 24.0),
        atol=2e-6,
        rtol=0,
    )


def test_nested_half_bank_path_is_finite_and_normalized():
    fleet = _fleet()
    atoms = sample_sigmas_exact(fleet, np.random.default_rng(23), 128)
    stream = generate_evaluation_stream(fleet, "C", 1, 30, 992)
    oracle = OrderingOracle(fleet, atoms, torch.device("cpu"), context_atom_batch=64)
    likelihood = oracle.context_log_likelihood(stream["contexts"][0], 4.0, False)
    assert likelihood.dtype == torch.float64
    values, bins, log_weights = quadrature_grid(
        fleet,
        {"quadrature_interior_nodes": 4, "quadrature_tail_nodes": 8},
    )
    full = oracle.predict_from_log_likelihood(
        likelihood,
        stream["queries"][0],
        4.0,
        False,
        32,
        values,
        bins,
        log_weights,
        8,
        1e-8,
    )
    half = oracle.predict_from_log_likelihood(
        likelihood,
        stream["queries"][0],
        4.0,
        False,
        32,
        values,
        bins,
        log_weights,
        8,
        1e-8,
        atom_limit=64,
    )
    validate_probability(full.full, "full-bank full")
    validate_probability(full.ablated, "full-bank ablated")
    validate_probability(half.full, "half-bank full")
    validate_probability(half.ablated, "half-bank ablated")
    assert 0.0 < full.keep_full <= 1.000001
    assert 0.0 < full.keep_ablated <= 1.000001
    assert 0.0 < half.keep_full <= 1.000001
    assert 0.0 < half.keep_ablated <= 1.000001


def test_deterministic_topk_breaks_cutoff_ties_by_ascending_atom_index():
    values = torch.tensor(
        [[5.0, 4.0, 4.0, 4.0, 1.0], [1.0, 3.0, 2.0, 2.0, 2.0]],
        dtype=torch.float64,
    )
    first_values, first_indices = OrderingOracle._deterministic_topk_rows(values, 3)
    second_values, second_indices = OrderingOracle._deterministic_topk_rows(values, 3)
    torch.testing.assert_close(first_values, second_values)
    torch.testing.assert_close(first_indices, second_indices)
    assert first_indices.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_ablated_truncation_mixes_orderings_uniformly_despite_unequal_retained_mass():
    kept = torch.log(
        torch.tensor(
            [
                [0.50, 0.25, 0.25],
                [0.20, 0.05, 0.05],
            ],
            dtype=torch.float64,
        )
    )
    weights = OrderingOracle._uniform_ablated_log_weights(kept)
    per_order_mass = torch.exp(weights).sum(dim=1)
    torch.testing.assert_close(
        per_order_mass,
        torch.ones(2, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    global_order_mass = per_order_mass / per_order_mass.sum()
    torch.testing.assert_close(
        global_order_mass,
        torch.full((2,), 0.5, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_v2_partial_archive_uses_native_output_bin_shape():
    arrays = _partial_arrays(
        3, 2, 5, "00" * 32, 100, archive_quadrature_reference=True
    )
    assert arrays["outcome_bins"].shape == (5,)
    assert arrays["full_probability"].shape == (3, 5, 100)
    assert arrays["ablated_probability"].shape == (3, 5, 100)
    assert arrays["quadrature_grid_full_probability"].shape == (2, 3, 5, 100)
    assert arrays["quadrature_grid_ablated_probability"].shape == (2, 3, 5, 100)
    assert arrays["quadrature_max_bin_abs_ordering_value_change"].shape == (3, 5)


def test_every_production_calibration_field_is_frozen():
    _validate_calibration_config(deepcopy(PRODUCTION_CALIBRATION_PROTOCOL), True)
    for name, expected in PRODUCTION_CALIBRATION_PROTOCOL.items():
        mutated = deepcopy(PRODUCTION_CALIBRATION_PROTOCOL)
        if isinstance(expected, bool):
            mutated[name] = not expected
        elif isinstance(expected, int):
            mutated[name] = expected + 1
        elif isinstance(expected, float):
            mutated[name] = expected * 2.0
        elif isinstance(expected, str):
            mutated[name] = expected + ".changed"
        elif isinstance(expected, list):
            mutated[name] = expected + [999_999_999]
        elif isinstance(expected, dict):
            mutated[name][next(iter(expected))] *= 2.0
        else:
            raise AssertionError(type(expected))
        with pytest.raises((ValueError, KeyError)):
            _validate_calibration_config(mutated, True)


def test_registered_config_and_runtime_inventory_match_the_frozen_protocol():
    registered = json.loads(
        (ROOT / "config" / "phase1_ordering_calibration.json").read_text()
    )
    assert registered == PRODUCTION_CALIBRATION_PROTOCOL
    inventory = json.loads(
        (ROOT / "environment" / "phase1-washu-binary-inventory.json").read_text()
    )
    expected = inventory.pop("runtime_binary_fingerprint")
    assert _sha256_json(inventory) == expected
    for bank_index in range(3):
        qualification = json.loads(
            (
                ROOT
                / "config"
                / f"phase1_ordering_qualification_bank{bank_index}.json"
            ).read_text()
        )
        assert qualification == production_qualification_protocol(bank_index)
        assert qualification["calibration_contexts_per_prior"] == 160
        assert qualification["qualification_zero_exceedance"]["families"] == 48


def test_attempt_lease_allows_only_one_live_writer(tmp_path):
    lease_path = tmp_path / "ATTEMPT.lock"
    first = _acquire_attempt_lease(lease_path, "0" * 64, False)
    assert first["attempt_identity_sha256"] == "0" * 64
    with pytest.raises(FileExistsError):
        _acquire_attempt_lease(lease_path, "0" * 64, False)


def test_stale_lease_recovery_has_exactly_one_winner(tmp_path):
    lease_path = tmp_path / "ATTEMPT.lock"
    lease_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_identity_sha256": "0" * 64,
                "hostname": socket.gethostname(),
                "pid": 2_147_483_647,
                "slurm_job_id": None,
                "created_unix": 0.0,
            }
        )
    )

    def recover() -> bool:
        try:
            _acquire_attempt_lease(lease_path, "1" * 64, True)
        except (FileExistsError, RuntimeError):
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: recover(), range(2)))
    assert sum(outcomes) == 1
    assert not (tmp_path / "RECOVERY.lock").exists()


def test_calibration_gate_checks_both_predictors_and_indifference_band():
    thresholds = {
        "median_js_max": 0.01,
        "p95_js_max": 0.02,
        "median_abs_logp_change_max": 0.03,
        "p95_abs_logp_change_max": 0.04,
        "numerical_indifference_fraction": 0.1,
    }
    passed, row = _calibration_candidate_passes(
        np.full(100, 0.001),
        np.full(100, 0.002),
        np.full(100, 0.003),
        np.full(100, 0.004),
        thresholds,
    )
    assert passed
    borderline = np.full(100, 0.0095)
    failed, failed_row = _calibration_candidate_passes(
        borderline,
        np.full(100, 0.002),
        np.full(100, 0.003),
        np.full(100, 0.004),
        thresholds,
    )
    assert not failed
    assert failed_row["full"]["borderline"] is True
    assert row["pass"] is True


def test_collapsed_atom_ess_does_not_multiply_exact_ordering_copies():
    fleet = _fleet()
    atoms = sample_sigmas_exact(fleet, np.random.default_rng(30), 32)
    oracle = OrderingOracle(fleet, atoms, torch.device("cpu"), context_atom_batch=16)
    likelihood = torch.full((24, 32), -torch.inf)
    likelihood[:, 0] = 0.0
    full_ess, ablated_ess = oracle.collapsed_atom_ess(likelihood)
    assert abs(full_ess - 1.0) < 1e-12
    assert abs(ablated_ess - 1.0) < 1e-12


def test_quadrature_is_bin_aligned_for_constant_interior_density():
    fleet = _fleet()
    _, bins, log_weights = quadrature_grid(
        fleet, {"quadrature_interior_nodes": 8, "quadrature_tail_nodes": 16}
    )
    weights = np.exp(log_weights)
    interior = np.array([weights[bins == index].sum() for index in range(1, 99)])
    np.testing.assert_allclose(interior, np.full(98, 0.16), atol=1e-14, rtol=0)


def test_calibration_writes_raw_oracle_arrays_without_scientific_endpoint(tmp_path):
    config = {
        "schema_version": 1,
        "status": "calibration_only",
        "fleet_module": str(FLEET_PATH),
        "fleet_sha256": "1aa7652cad924c90f871309f860b9172e836898c5a1620c77fdd3196e70d291d",
        "dimension": 4,
        "context_size": 30,
        "priors": ["C", "N"],
        "calibration_contexts_per_prior": 2,
        "calibration_seed_root": 99800,
        "calibration_seed_namespace": "test-calibration",
        "atom_count": 96,
        "atom_seed": 99801,
        "reserved_confirmatory_seeds": [99900, 99901, 99902, 99903],
        "known_persisted_fixed_seeds": [810000],
        "truncation_candidates": [24, 48],
        "reference_truncation": 96,
        "quadrature_interior_nodes": 4,
        "quadrature_tail_nodes": 8,
        "query_grid_chunk": 8,
        "context_atom_batch": 32,
        "require_clean_git": False,
        "requirements_lock": "environment/requirements-lock.txt",
        "thresholds": {
            "median_js_max": 1000000000.0,
            "p95_js_max": 1000000000.0,
            "median_abs_logp_change_max": 1000000000.0,
            "p95_abs_logp_change_max": 1000000000.0,
            "numerical_indifference_fraction": 0.0,
            "probability_sum_atol": 1e-8,
        },
    }
    config_path = tmp_path / "calibration.json"
    config_path.write_text(json.dumps(config))
    summary = run_calibration(
        config_path, tmp_path / "out", "cpu", production_protocol=False
    )
    assert summary["decision"] == "CALIBRATION_PASS"
    assert summary["selected_truncation"] == 24
    assert summary["scientific_endpoints_computed"] is False
    # The sentinel is a substring scan for an endpoint NAME, and the summary
    # embeds `git status --porcelain` under "git". A dirty working tree that
    # touches src/pfn_dag_verify/corrected_deficit_run.py therefore matched
    # "deficit" and failed this test for a reason that had nothing to do with a
    # leaked endpoint. Scan everything EXCEPT the provenance block, whose whole
    # job is to carry file names.
    scanned = {k: v for k, v in summary.items() if k != "git"}
    assert "deficit" not in json.dumps(scanned).lower()
    assert "git" in summary, "the provenance block must still be present"
    assert summary["completion_state"] == "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER"
    assert [row["compared_with"] for row in summary["candidate_results"]] == [96, 96]
    complete_path = tmp_path / "out" / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    summary_path = tmp_path / "out" / "calibration_summary.json"
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == complete["artifacts"][
        "calibration_summary.json"
    ]["sha256"]
    assert not (tmp_path / "out" / "ATTEMPT.lock").exists()
    with np.load(tmp_path / "out" / "calibration_raw.npz", allow_pickle=False) as raw:
        forbidden = {
            "contexts",
            "queries",
            "outcomes",
            "outcome_bins",
            "true_orderings",
            "full_probability",
            "ablated_probability",
        }
        assert forbidden.isdisjoint(raw.files)
        assert raw["keep_full"].shape == (2, 3, 2)
        assert raw["full_probability_sum_error"].shape == (2, 3, 2)
        assert raw["js_full"].shape == (2, 2, 2)
        assert raw["attempt_identity_sha256"].shape == (2, 32)


def test_unregistered_v3_cannot_call_core_runner(tmp_path):
    config = {
        "schema_version": 1,
        "status": "calibration_only",
        "fleet_module": str(FLEET_PATH),
        "fleet_sha256": "1aa7652cad924c90f871309f860b9172e836898c5a1620c77fdd3196e70d291d",
        "dimension": 4,
        "context_size": 30,
        "priors": ["C", "N"],
        "calibration_contexts_per_prior": 1,
        "calibration_seed_root": 99_600,
        "calibration_seed_namespace": "test-qualification-v2",
        "atom_count": 96,
        "atom_seed": 99_601,
        "reserved_confirmatory_seeds": [99_700, 99_701],
        "known_persisted_fixed_seeds": [810_000],
        "truncation_candidates": [24, 48],
        "reference_truncation": 96,
        "quadrature_interior_nodes": 2,
        "quadrature_tail_nodes": 4,
        "quadrature_reference_interior_nodes": 4,
        "quadrature_reference_tail_nodes": 8,
        "quadrature_qualification": {
            "js_max": 1e9,
            "reference_probability_floor": 1e-8,
            "max_bin_abs_logp_change_max": 1e9,
            "reference_weighted_abs_logp_change_max": 1e9,
            "max_bin_abs_ordering_value_change_max": 1e9,
        },
        "query_grid_chunk": 8,
        "context_atom_batch": 32,
        "require_clean_git": False,
        "requirements_lock": "environment/requirements-lock.txt",
        "thresholds": {
            "median_js_max": 1e9,
            "p95_js_max": 1e9,
            "median_abs_logp_change_max": 1e9,
            "p95_abs_logp_change_max": 1e9,
            "numerical_indifference_fraction": 0.0,
            "probability_sum_atol": 1e-8,
        },
        "qualification_protocol_version": 3,
        "archive_predictive_arrays": True,
        "oracle_internal_dtype": "float64",
    }
    config_path = tmp_path / "qualification_v2.json"
    config_path.write_text(json.dumps(config))
    output_dir = tmp_path / "out-v3"
    with pytest.raises(RuntimeError, match="frozen CLI execution path"):
        run_calibration(
            config_path,
            output_dir,
            "cpu",
            production_protocol=False,
            strict_runtime=False,
            stage_name="phase1_ordering_cross_bank_qualification_v3",
        )
    assert not output_dir.exists()
