from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pfn_dag_verify.phase1_oracle_confirm import (
    ORACLE_ARRAYS,
    _legacy_raw_array_sha256,
    _make_oracle,
    _score_shard,
)
from pfn_dag_verify.phase1_ordering import OraclePrediction
from pfn_dag_verify.phase1_panel import ROW_KEYS


class _FakeOracle:
    fleet = SimpleNamespace(R_OF={"C": 2.0})

    def context_log_likelihood(self, context, r, gaussian):
        return torch.zeros((24, 4), dtype=torch.float64)

    def collapsed_atom_ess(self, likelihood):
        return 1.0, 2.0

    def predict_from_log_likelihood(self, *args, atom_limit=None, **kwargs):
        probability = np.full(100, 0.01, dtype=np.float64)
        if atom_limit is not None:
            probability = probability.copy()
            probability[0] += 0.001
            probability[1:] -= 0.001 / 99.0
        return OraclePrediction(
            full=probability,
            ablated=probability[::-1].copy(),
            ordering_posterior=np.full(24, 1.0 / 24.0),
            keep_full=0.9,
            keep_ablated=0.8,
        )


def _shard(n=66):
    values = {name: np.zeros(n, dtype=np.int64) for name in ROW_KEYS}
    values["row_id"] = np.arange(n, dtype=np.int64)
    values["stream_index"] = np.arange(2, 2 + 3 * n, 3, dtype=np.int64)
    values["atom_bank_index"][:] = 2
    values["shard_local_index"] = np.arange(n, dtype=np.int64)
    return {
        **values,
        "row_key_sha256": np.zeros((n, 32), dtype=np.uint8),
        "input_row_sha256": np.ones((n, 32), dtype=np.uint8),
        "nested_half_mask": np.ones(n, dtype=np.int8),
        "contexts": np.zeros((n, 30, 4), dtype=np.float64),
        "queries": np.zeros((n, 3), dtype=np.float64),
    }


def test_legacy_hash_is_raw_bytes_only():
    import hashlib

    value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert (
        _legacy_raw_array_sha256(value) == hashlib.sha256(value.tobytes()).hexdigest()
    )


def test_confirmation_oracle_is_constructed_in_float64():
    expected = SimpleNamespace(compute_dtype=torch.float64)
    with patch(
        "pfn_dag_verify.phase1_oracle_confirm.OrderingOracle", return_value=expected
    ) as constructor:
        observed = _make_oracle(
            SimpleNamespace(),
            np.zeros((4, 4, 4), dtype=np.float64),
            torch.device("cpu"),
            {"context_atom_batch": 2, "oracle_compute_dtype": "float64"},
        )
    assert observed is expected
    assert constructor.call_args.kwargs["compute_dtype"] == torch.float64


def test_score_shard_keeps_half_prefix_outputs_separate():
    config = {
        "selected_truncation": 16_384,
        "probability_sum_atol": 1e-8,
        "query_grid_chunk": 16,
        "nested_half_atom_count": 1_500_000,
        "atom_count": 4,
    }
    identity = "00" * 32
    raw, half = _score_shard(
        _FakeOracle(),
        _shard(),
        "C",
        config,
        (np.zeros(1), np.zeros(1, dtype=np.int64), np.zeros(1)),
        identity,
    )
    assert tuple(raw) == ORACLE_ARRAYS
    assert raw["full_probability"].shape == (66, 100)
    assert np.all(raw["ess_full_atoms"] == 1.0)
    assert half is not None
    assert half["row_id"].tolist() == list(range(66))
    assert half["full_probability"].shape == (66, 100)
    assert not np.array_equal(raw["full_probability"], half["full_probability"])
