import ast
import json
from pathlib import Path

import numpy as np

from pfn_dag_verify.phase1_confirmation_verify import (
    _bootstrap as independent_bootstrap,
    _decide as independent_decide,
    _point_estimates as independent_points,
)
from pfn_dag_verify.phase1_join import (
    _bootstrap as joined_bootstrap,
    _decide as joined_decide,
    _point_estimates as joined_points,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((ROOT / "config/phase1_ordering_confirmation.json").read_text())


def _rows():
    config = _config()
    values = {
        name: [] for name in ("row_id", "prior_code", "evaluation_seed", "atom_seed")
    }
    values.update({name: [] for name in ("ordering_value", "deficit", "gap")})
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


def test_independent_verifier_reproduces_join_statistics_after_storage_relabel():
    config = _config()
    rows = _rows()
    permutation = np.random.default_rng(91).permutation(len(rows["row_id"]))
    shuffled = {name: value[permutation] for name, value in rows.items()}
    joined_estimates = joined_points(rows)
    independent = independent_points(shuffled)
    for name in joined_estimates:
        assert np.allclose(joined_estimates[name], independent[name], rtol=0, atol=0)
    joined_boot, joined_hashes = joined_bootstrap(rows, config)
    independent_boot, independent_hashes = independent_bootstrap(shuffled, config)
    assert joined_hashes == independent_hashes
    for name in joined_boot:
        assert np.array_equal(joined_boot[name], independent_boot[name])


def _decision_case(final, change, causal_final, causal_change):
    causal_deficit = np.array([0.0, -0.002, causal_final])
    delta = np.array([0.0, -0.002, final])
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
        "delta": np.vstack((np.zeros(100), np.full(100, -0.002), np.full(100, final))),
        "delta_change_final_minus_early": np.full(100, change),
    }
    return points, bootstrap


def test_independent_verifier_reproduces_replay_sensitive_decisions():
    config = _config()
    mechanical = {
        "completeness_and_provenance": True,
        "inference_guards": True,
        "predictive_truncation": True,
        "monte_carlo_diagnostics_reported": True,
        "fixed_fleet_completeness": True,
    }
    cases = (
        (-0.010, -0.010, -0.010, -0.010),
        (-0.009075, -0.010, -0.009075, -0.010),
        (-0.006975, -0.010, -0.006975, -0.010),
        (-0.010, -0.00815, -0.010, -0.00805),
    )
    for case in cases:
        points, bootstrap = _decision_case(*case)
        joined = joined_decide(points, bootstrap, {"pass": True}, config, mechanical)
        independent = independent_decide(points, bootstrap, {"pass": True}, config)
        assert independent == joined


def test_independent_verifier_has_no_pipeline_imports():
    path = ROOT / "src/pfn_dag_verify/phase1_confirmation_verify.py"
    tree = ast.parse(path.read_text())
    forbidden = {
        "phase1_confirm_common",
        "phase1_join",
        "phase1_oracle_confirm",
        "phase1_pfn",
    }
    imported = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported.isdisjoint(forbidden)
