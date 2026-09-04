import numpy as np

from pfn_dag_verify.analysis import _mapping_block
from pfn_dag_verify.decision import DecisionInputs, decide_primary
from pfn_dag_verify.statistics import (
    crossed_bootstrap_slope,
    permutation_null_slopes,
    within_group_slope,
)


def _synthetic(beta=1.0, seeds=16, groups=128, continuations=8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(groups, continuations))
    group_noise = rng.normal(scale=0.4, size=(groups, 1))
    seed_gain = rng.normal(beta, 0.025, size=(seeds, 1, 1))
    eps = rng.normal(scale=0.08, size=(seeds, groups, continuations))
    y = seed_gain * x[None, :, :] + group_noise[None, :, :] + eps
    return x, y


def test_within_group_slope_recovers_realized_continuation_gain():
    x, y = _synthetic(beta=0.75)
    assert abs(within_group_slope(x, y) - 0.75) <= 0.03


def test_crossed_bootstrap_resamples_shared_groups_and_model_seeds():
    x, y = _synthetic(beta=1.0)
    draws = crossed_bootstrap_slope(x, y, n_boot=500, rng=np.random.default_rng(9))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    assert lo < 1.0 < hi
    assert hi - lo < 0.15


def test_cyclic_swap_canary_collapses_realized_pairing():
    x, y = _synthetic(beta=1.0, groups=256, seed=12)
    draws = permutation_null_slopes(
        x, y, n_permutations=500, rng=np.random.default_rng(13)
    )
    lo, hi = np.quantile(draws, [0.025, 0.975])
    assert -0.15 <= lo <= 0 <= hi <= 0.15


def _decision(**overrides):
    values = dict(
        instrument_pass=True,
        identifiable=True,
        mapping_pass=True,
        reconstruction_confirmation_pass=True,
        canary_intervals=((-0.04, 0.04), (-0.05, 0.03)),
        slope_intervals=((0.90, 1.08), (0.88, 1.10)),
        nrmse_upper=(0.20, 0.22),
    )
    values.update(overrides)
    return decide_primary(DecisionInputs(**values))


def test_decision_rules_are_fail_closed():
    assert _decision().code == "COMPATIBLE_ON_TESTED_REGIME"
    assert _decision(slope_intervals=((0.40, 0.60), (0.45, 0.62))).code == "INCOMPATIBLE_ON_TESTED_REGIME"
    assert _decision(mapping_pass=False).code == "INCONCLUSIVE_MAPPING"
    assert _decision(reconstruction_confirmation_pass=False).code == "INCONCLUSIVE_RECONSTRUCTION"
    assert _decision(
        reconstruction_confirmation_pass=False,
        slope_intervals=((0.40, 0.60), (0.45, 0.62)),
    ).code == "INCOMPATIBLE_ON_TESTED_REGIME"
    assert _decision(identifiable=False).code == "INCONCLUSIVE_IDENTIFIABILITY"
    assert _decision(canary_intervals=((-0.2, 0.03), (-0.05, 0.03))).code == "INCONCLUSIVE_CANARY"
    assert _decision(slope_intervals=((0.75, 0.95), (0.90, 1.10))).code == "INCONCLUSIVE"


def test_future_mapping_gate_does_not_prejudge_tempered_evidence_gain():
    groups, continuations = 4, 4
    panel = {"eligible_replace": np.ones((groups, continuations), dtype=np.uint8)}
    shard = {
        "boundary_base": np.zeros(groups, dtype=np.uint8),
        "boundary_target": np.zeros((groups, continuations), dtype=np.uint8),
        "g_base": np.linspace(-0.4, 0.4, groups),
        "kl_g_base": np.linspace(-0.4, 0.4, groups),
        "g_target": np.zeros((groups, continuations)),
        "kl_g_target": np.zeros((groups, continuations)),
        "mix_residual_base": np.zeros(groups),
        "mix_residual_target": np.zeros((groups, continuations)),
        # A tempered on-segment readout can fail exact unit-gain reconstruction.
        "reconstruction_replace": np.full((groups, continuations), 0.2),
    }
    block = _mapping_block(panel, [shard], bank_index=0, contrast="replace")
    assert block["pass"] is True
    assert block["identity_reconstruction_residual"]["median"] == 0.2
