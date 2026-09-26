import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pfn_dag_verify.phase1_join import (
    _bootstrap,
    _decide,
    _nested_half_gate,
    _point_estimates,
    _validate_panel_covariances,
)


ROOT = Path(__file__).resolve().parents[1]


class _AlwaysValidFleet:
    @staticmethod
    def validity_keep(sigmas):
        return np.ones(len(sigmas), dtype=bool)


def test_join_accepts_registered_float64_covariance_roundoff_without_mutation():
    sigmas = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 2, axis=0)
    sigmas[0, 0, 1] = 0.2
    sigmas[0, 1, 0] = np.nextafter(0.2, np.inf)
    before = sigmas.copy()

    _validate_panel_covariances(_AlwaysValidFleet(), sigmas, "fixture")

    np.testing.assert_array_equal(sigmas, before)


def test_join_rejects_material_or_zero_to_nonzero_covariance_asymmetry():
    material = np.eye(4, dtype=np.float64)[None, :, :]
    material[0, 0, 1] = 0.2
    material[0, 1, 0] = 0.2000001
    with pytest.raises(RuntimeError, match="symmetry roundoff bound"):
        _validate_panel_covariances(_AlwaysValidFleet(), material, "material")

    zero_to_nonzero = np.eye(4, dtype=np.float64)[None, :, :]
    zero_to_nonzero[0, 0, 1] = np.nextafter(0.0, np.inf)
    with pytest.raises(RuntimeError, match="symmetry roundoff bound"):
        _validate_panel_covariances(
            _AlwaysValidFleet(), zero_to_nonzero, "zero-to-nonzero"
        )


def _config():
    return json.loads((ROOT / "config/phase1_ordering_confirmation.json").read_text())


def _constant_rows():
    config = _config()
    values = {
        "row_id": [],
        "prior_code": [],
        "evaluation_seed": [],
        "atom_seed": [],
        "ordering_value": [],
        "deficit": [],
        "gap": [],
    }
    row_id = 0
    for prior_code, prior in enumerate(("C", "N")):
        for evaluation_seed in config["evaluation_seeds"][prior]:
            for atom in config["atom_banks"]:
                n = 356 if atom["bank_index"] < 2 else 355
                for _ in range(n):
                    values["row_id"].append(row_id)
                    values["prior_code"].append(prior_code)
                    values["evaluation_seed"].append(evaluation_seed)
                    values["atom_seed"].append(atom["seed"])
                    values["ordering_value"].append(0.2 if prior_code == 0 else 0.0)
                    cell = np.empty((3, 3), dtype=np.float64)
                    for model in range(3):
                        cell[model] = (
                            prior_code + model * 0.01 + np.array([0.1, 0.2, 0.3])
                        )
                    values["deficit"].append(cell)
                    values["gap"].append(cell + 0.5)
                    row_id += 1
    return {name: np.asarray(value) for name, value in values.items()}


def test_bootstrap_is_constant_and_storage_order_invariant():
    config = _config()
    rows = _constant_rows()
    first, first_hashes = _bootstrap(rows, config)
    assert len(first_hashes) == 18
    assert all(len(value) == 64 for value in first_hashes.values())
    assert np.allclose(first["ordering_value"][0], 0.2)
    assert np.allclose(first["ordering_value"][1], 0.0)
    assert np.allclose(first["delta"], -1.0)

    permutation = np.random.default_rng(44).permutation(len(rows["row_id"]))
    shuffled = {name: value[permutation] for name, value in rows.items()}
    second, second_hashes = _bootstrap(shuffled, config)
    assert first_hashes == second_hashes
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_point_estimate_uses_context_weights_and_equal_model_weights():
    points = _point_estimates(_constant_rows())
    assert np.allclose(points["deficit"][0], [0.11, 0.21, 0.31])
    assert np.allclose(points["deficit"][1], [1.11, 1.21, 1.31])
    assert np.allclose(points["delta"], [-1.0, -1.0, -1.0])
    assert np.allclose(points["deficit_change_final_minus_early"], [0.2, 0.2])
    assert np.isclose(points["delta_change_final_minus_early"], 0.0)


def _decision_inputs(
    final=-0.010,
    early=0.0,
    change=-0.010,
    causal_final=-0.010,
    causal_early=0.0,
    causal_change=-0.010,
):
    causal_deficit = np.array([causal_early, -0.002, causal_final])
    delta = np.array([early, -0.002, final])
    control_deficit = causal_deficit - delta
    points = {
        "ordering_value": np.array([0.1, 0.0]),
        "gap_final": np.array([0.1, 0.1]),
        "deficit": np.vstack((causal_deficit, control_deficit)),
        "deficit_change_final_minus_early": np.array(
            [causal_change, causal_change - change]
        ),
        "delta": delta,
        "delta_change_final_minus_early": np.array(change),
    }
    bootstrap = {
        "ordering_value": np.vstack((np.full(100, 0.1), np.zeros(100))),
        "gap_final": np.full((2, 100), 0.1),
        "deficit": np.stack(
            (
                np.repeat(causal_deficit[:, None], 100, axis=1),
                np.repeat(control_deficit[:, None], 100, axis=1),
            )
        ),
        "deficit_change_final_minus_early": np.vstack(
            (np.full(100, causal_change), np.full(100, causal_change - change))
        ),
        "delta": np.vstack(
            (np.full(100, early), np.full(100, -0.002), np.full(100, final))
        ),
        "delta_change_final_minus_early": np.full(100, change),
    }
    return points, bootstrap


def test_decision_truth_table_and_strict_effect_boundary():
    config = _config()
    mechanical = {"mechanical": True}
    points, bootstrap = _decision_inputs()
    result = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert result["decision"] == "REPLICATED_ORDERING_USE"
    assert (
        result["secondary"] == "SUPPORTED_UNDERTRAINING_CAN_OBSCURE_ORDERING_ADVANTAGE"
    )

    points, bootstrap = _decision_inputs(
        final=-0.007,
        change=-0.008,
        causal_final=-0.007,
        causal_change=-0.008,
    )
    boundary = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert boundary["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"

    invalid = _decide(points, bootstrap, {"pass": False}, config, mechanical)
    assert invalid["decision"] == "INCONCLUSIVE_PHASE1_INSTRUMENT"
    assert invalid["primary"] == "NOT_EVALUATED"


def test_decision_stops_inside_the_registered_numerical_clearance_band():
    config = _config()
    mechanical = {"mechanical": True}
    points, bootstrap = _decision_inputs(
        final=-0.0081,
        change=-0.0081,
        causal_final=-0.0081,
        causal_change=-0.0081,
    )
    result = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert result["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"
    assert result["numerical_clearance"] == {
        "effect_floor": -0.008,
        "oracle_clearance_nats": 0.001,
        "clearance_boundary": -0.009000000000000001,
        "rejection_boundary": -0.007,
        "pfn_replay_bounds": {
            "single_pfn_nll": 8e-5,
            "direct_deficit_or_gap": 8e-5,
            "causal_minus_control_delta": 1.6e-4,
            "direct_checkpoint_change": 1.6e-4,
            "delta_difference_in_differences": 3.2e-4,
        },
        "borderline_action": (
            "stop_without_claim_and_increase_numeric_fidelity_or_replay_stability"
        ),
    }
    assert result["delta_by_checkpoint"]["120000"]["passes_effect_floor"] is True
    assert result["delta_by_checkpoint"]["120000"]["passes_rule"] is False

    points, bootstrap = _decision_inputs(
        final=-0.009,
        change=-0.009,
        causal_final=-0.009,
        causal_change=-0.009,
    )
    strict_boundary = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert strict_boundary["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"

    points, bootstrap = _decision_inputs(
        final=-0.0079,
        change=-0.010,
        causal_final=-0.0079,
        causal_change=-0.010,
    )
    near_negative = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert near_negative["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"

    points, bootstrap = _decision_inputs(
        final=-0.0068,
        change=-0.010,
        causal_final=-0.0068,
        causal_change=-0.010,
    )
    clear_negative = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert clear_negative["decision"] == "NOT_REPLICATED_ORDERING_USE"


def test_primary_requires_endpoint_specific_replay_margin():
    config = _config()
    mechanical = {"mechanical": True}

    points, bootstrap = _decision_inputs(
        final=-0.00912,
        change=-0.010,
        causal_final=-0.00912,
        causal_change=-0.010,
    )
    delta_inside_replay_band = _decide(
        points, bootstrap, {"pass": True}, config, mechanical
    )
    assert (
        delta_inside_replay_band["decision"]
        == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"
    )
    assert (
        delta_inside_replay_band["deficit_by_prior_and_checkpoint"]["C"]["120000"][
            "passes_direct_rule"
        ]
        is True
    )
    assert (
        delta_inside_replay_band["delta_by_checkpoint"]["120000"]["passes_rule"]
        is False
    )

    points, bootstrap = _decision_inputs(
        final=-0.00918,
        change=-0.010,
        causal_final=-0.00918,
        causal_change=-0.010,
    )
    beyond_replay_band = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert beyond_replay_band["decision"] == "REPLICATED_ORDERING_USE"

    points, bootstrap = _decision_inputs(
        final=-0.006975,
        change=-0.010,
        causal_final=-0.006975,
        causal_change=-0.010,
    )
    negative_inside_replay_band = _decide(
        points, bootstrap, {"pass": True}, config, mechanical
    )
    assert (
        negative_inside_replay_band["decision"]
        == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"
    )


def test_secondary_reports_replay_sensitive_checkpoint_change():
    config = _config()
    points, bootstrap = _decision_inputs(
        final=-0.010,
        early=0.0,
        change=-0.00815,
        causal_final=-0.010,
        causal_early=0.0,
        causal_change=-0.00805,
    )
    result = _decide(points, bootstrap, {"pass": True}, config, {"mechanical": True})
    assert result["primary"] == "REPLICATED_ORDERING_USE"
    assert result["secondary"] == "INCONCLUSIVE_UNDERTRAINING_REPLAY_SENSITIVITY"
    assert result["deficit_change_final_minus_early"]["C"]["replay_sensitive"]
    assert result["delta_change_final_minus_early"]["replay_sensitive"]


def test_primary_requires_both_direct_and_control_subtracted_clearance():
    config = _config()
    points, bootstrap = _decision_inputs(
        final=-0.0081,
        causal_final=-0.010,
        change=-0.010,
        causal_change=-0.010,
    )
    delta_unclear = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert delta_unclear["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"

    points, bootstrap = _decision_inputs(
        final=-0.010,
        causal_final=-0.0081,
        change=-0.010,
        causal_change=-0.010,
    )
    direct_unclear = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert direct_unclear["decision"] == "INCONCLUSIVE_PHASE1_NUMERICAL_CLEARANCE"


def test_decision_remains_positive_when_comfortably_beyond_clearance():
    config = _config()
    points, bootstrap = _decision_inputs(
        final=-0.010,
        change=-0.010,
        causal_final=-0.010,
        causal_change=-0.010,
    )
    result = _decide(points, bootstrap, {"pass": True}, config, {"mechanical": True})
    assert result["decision"] == "REPLICATED_ORDERING_USE"
    assert result["delta_by_checkpoint"]["120000"]["passes_rule"] is True


def test_registered_numerical_clearance_exceeds_v3_aggregate_envelope():
    raw_path = (
        ROOT
        / "campaigns/phase1_ordering_20260803/oracle_qualification_v3/joined"
        / "qualification_raw.npz"
    )
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == (
        "a4bc89e5f22f41764ddefb310bc2fb67dc291c0d0156949c1840841549915317"
    )
    replicates = 50_000
    chunk = 500
    generator = np.random.Generator(np.random.PCG64(881_003_900))
    bootstrap = np.empty((3, 2, replicates), dtype=np.float64)
    index_digest = hashlib.sha256()
    with np.load(raw_path, allow_pickle=False) as raw:
        assert raw["candidates"].tolist() == [8192, 16384]
        bins = raw["outcome_bins"]
        probabilities = raw["ablated_probability"]
        for bank in range(3):
            for prior in range(2):
                outcome = bins[bank, prior]
                selected = probabilities[bank, prior, 1, np.arange(160), outcome]
                reference = probabilities[bank, prior, 2, np.arange(160), outcome]
                absolute_nll_error = np.abs(np.log(reference) - np.log(selected))
                for start in range(0, replicates, chunk):
                    stop = min(replicates, start + chunk)
                    indices = generator.integers(
                        0,
                        160,
                        size=(stop - start, 160),
                        dtype=np.int64,
                    )
                    index_digest.update(
                        indices.astype("<i8", copy=False).tobytes(order="C")
                    )
                    bootstrap[bank, prior, start:stop] = absolute_nll_error[
                        indices
                    ].mean(axis=1)
        quadrature = raw["quadrature_reference_weighted_abs_logp_change_ablated"][
            :, :, 1, :
        ]
        worst_quadrature = float(quadrature.max())
    envelope = bootstrap[:, 0, :].max(axis=0) + bootstrap[:, 1, :].max(axis=0)
    q95 = float(np.quantile(envelope, 0.95, method="linear"))
    assert index_digest.hexdigest() == (
        "19761f6579c48ca233303bb8fa5cf87a4a8a2dab9df550f65ac163ced68efb41"
    )
    assert np.isclose(q95, 0.0003433511679446371, rtol=0.0, atol=1e-15)
    assert np.isclose(worst_quadrature, 0.000029849800733846442)
    gates = _config()["gates"]
    assert q95 + worst_quadrature < gates["qualified_topk_quadrature_clearance"]
    assert (
        gates["qualified_topk_quadrature_clearance"]
        + gates["nested_half_control_subtracted_ablated_abs_max"]
        == gates["primary_numerical_clearance"]
    )


def test_decision_rejects_control_only_negative_delta():
    config = _config()
    mechanical = {"mechanical": True}
    points, bootstrap = _decision_inputs(
        final=-0.02,
        early=0.0,
        change=-0.02,
        causal_final=0.0,
        causal_early=0.0,
        causal_change=0.0,
    )
    result = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert result["delta_by_checkpoint"]["120000"]["passes_rule"] is True
    assert (
        result["deficit_by_prior_and_checkpoint"]["C"]["120000"]["passes_direct_rule"]
        is False
    )
    assert result["decision"] == "NOT_REPLICATED_ORDERING_USE"
    assert result["secondary"] == "NOT_SUPPORTED_UNDERTRAINING_CLAIM"


def test_undertraining_requires_direct_causal_deficit_change():
    config = _config()
    mechanical = {"mechanical": True}
    points, bootstrap = _decision_inputs(
        final=-0.010,
        early=0.0,
        change=-0.010,
        causal_final=-0.010,
        causal_early=-0.010,
        causal_change=0.0,
    )
    result = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert result["decision"] == "REPLICATED_ORDERING_USE"
    assert result["secondary"] == "NOT_SUPPORTED_UNDERTRAINING_CLAIM"


def test_undertraining_uses_nominal_early_and_change_rules():
    config = _config()
    mechanical = {"mechanical": True}
    points, bootstrap = _decision_inputs(
        final=-0.010,
        early=-0.0081,
        change=-0.010,
        causal_final=-0.010,
        causal_early=-0.0081,
        causal_change=-0.010,
    )
    nominal_early = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert nominal_early["decision"] == "REPLICATED_ORDERING_USE"
    assert nominal_early["secondary"] == "NOT_SUPPORTED_UNDERTRAINING_CLAIM"

    points, bootstrap = _decision_inputs(
        final=-0.010,
        early=0.0,
        change=-0.008,
        causal_final=-0.010,
        causal_early=0.0,
        causal_change=-0.008,
    )
    strict_change = _decide(points, bootstrap, {"pass": True}, config, mechanical)
    assert strict_change["decision"] == "REPLICATED_ORDERING_USE"
    assert strict_change["secondary"] == "NOT_SUPPORTED_UNDERTRAINING_CLAIM"


def _nested_half_rows(
    d_c: float, d_n: float, fixed_fleet_gap: float = 0.1
) -> dict[str, np.ndarray]:
    config = _config()
    prior_code = np.repeat(np.arange(2, dtype=np.int64), 200)
    stream_index = np.tile(np.arange(200, dtype=np.int64), 2)
    atom_seeds = np.asarray(
        [row["seed"] for row in config["atom_banks"]], dtype=np.int64
    )
    atom_seed = atom_seeds[stream_index % 3]
    d = np.where(prior_code == 0, d_c, d_n).astype(np.float64)
    oracle_half_full = np.zeros(400, dtype=np.float64)
    oracle_full = np.zeros(400, dtype=np.float64)
    return {
        "row_id": np.arange(400, dtype=np.int64),
        "prior_code": prior_code,
        "atom_seed": atom_seed,
        "oracle_half_ablated_nll": np.zeros(400, dtype=np.float64),
        "oracle_ablated_nll": d,
        "oracle_half_full_nll": oracle_half_full,
        "oracle_full_nll": oracle_full,
        "pfn_final_nll": np.full((400, 3), fixed_fleet_gap, dtype=np.float64),
    }


def test_nested_half_gate_bounds_direct_and_control_subtracted_bank_error():
    config = _config()
    passing = _nested_half_gate(_nested_half_rows(0.0001, 0.0), config)
    assert passing["pass"] is True
    assert passing["causal_ablated_change_pass"] is True
    assert passing["ablated_change_difference_pass"] is True
    assert len(passing["bootstrap_index_stream_sha256"]) == 6

    shared_bias = _nested_half_gate(_nested_half_rows(0.0039, 0.0039), config)
    assert shared_bias["causal_ablated_change_pass"] is False
    assert shared_bias["pass"] is False

    control_subtracted_bias = _nested_half_gate(_nested_half_rows(0.0039, 0.0), config)
    assert control_subtracted_bias["ablated_change_difference_pass"] is False
    assert control_subtracted_bias["pass"] is False

    replay_sensitive_gap = _nested_half_gate(
        _nested_half_rows(0.0001, 0.0, fixed_fleet_gap=4e-5), config
    )
    assert (
        replay_sensitive_gap["priors"]["C"]["fixed_fleet_final_gap_one_sided_95_lower"]
        > 0.0
    )
    assert (
        replay_sensitive_gap["priors"]["C"]["replay_robust_gap_one_sided_95_lower"]
        < 0.0
    )
    assert replay_sensitive_gap["pass"] is False


def test_oracle_dependent_validity_boundaries_use_numerical_clearance():
    config = _config()
    points, bootstrap = _decision_inputs()
    bootstrap["gap_final"][0] = -0.0039
    points["gap_final"][0] = -0.0039
    kl_borderline = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert kl_borderline["decision"] == "INCONCLUSIVE_PHASE1_INSTRUMENT"
    assert kl_borderline["kl_alarm"]["C"]["numerically_borderline"] is True
    assert kl_borderline["validity_gates"]["kl_alarm_clear"] is False

    points, bootstrap = _decision_inputs()
    bootstrap["gap_final"][0] = -0.00296
    points["gap_final"][0] = -0.00296
    replay_borderline = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert replay_borderline["kl_alarm"]["C"]["clear"] is False
    assert replay_borderline["kl_alarm"]["C"]["numerically_borderline"] is True

    points, bootstrap = _decision_inputs()
    bootstrap["gap_final"][0] = -0.00290
    points["gap_final"][0] = -0.00290
    replay_cleared = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert replay_cleared["kl_alarm"]["C"]["clear"] is True

    points, bootstrap = _decision_inputs()
    bootstrap["ordering_value"][0] = 0.0009
    points["ordering_value"][0] = 0.0009
    ordering_borderline = _decide(
        points, bootstrap, {"pass": True}, config, {"mechanical": True}
    )
    assert ordering_borderline["decision"] == "INCONCLUSIVE_PHASE1_INSTRUMENT"
    assert ordering_borderline["validity_gates"]["ordering_value"] is False
