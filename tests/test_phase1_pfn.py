import numpy as np
import pytest
import torch

from pfn_dag_verify.phase1_pfn import _infer, _replay_guard


_BIN_INDEX = np.arange(100, dtype=np.float32)
_BIN_SLOPE = (_BIN_INDEX - np.float32(49.5)) / np.float32(50.0)
_BIN_OFFSET = ((_BIN_INDEX % 11) - np.float32(5.0)) / np.float32(100.0)


class _RowIndependentSetPFN(torch.nn.Module):
    @staticmethod
    def _row_signal(context, query, token):
        return context.sum(dim=(1, 2)) + 0.5 * query.sum(dim=(1, 2)) + token.float()

    def forward(self, context, query, token):
        signal = self._row_signal(context, query, token)
        bin_index = torch.arange(100, dtype=context.dtype, device=context.device)
        bin_slope = (bin_index - 49.5) / 50.0
        bin_offset = ((bin_index.remainder(11)) - 5.0) / 100.0
        return (
            signal[:, None, None] * bin_slope[None, None, :] + bin_offset[None, None, :]
        )


class _BatchCoupledPFN(_RowIndependentSetPFN):
    @staticmethod
    def _row_signal(context, query, token):
        independent = _RowIndependentSetPFN._row_signal(context, query, token)
        return independent + independent.mean()


class _ContextRowOrderSensitivePFN(_RowIndependentSetPFN):
    @staticmethod
    def _row_signal(context, query, token):
        position = torch.arange(
            1, context.shape[1] + 1, dtype=context.dtype, device=context.device
        )
        return (
            (context[:, :, 0] * position[None, :]).sum(dim=1)
            + 0.5 * query.sum(dim=(1, 2))
            + token.float()
        )


class _OutputRowRotatingPFN(_RowIndependentSetPFN):
    @staticmethod
    def _row_signal(context, query, token):
        independent = _RowIndependentSetPFN._row_signal(context, query, token)
        return torch.roll(independent, shifts=1, dims=0)


class _CombinedContextBatchSensitivePFN(_RowIndependentSetPFN):
    @staticmethod
    def _row_signal(context, query, token):
        independent = _RowIndependentSetPFN._row_signal(context, query, token)
        if context.shape[0] != 8:
            return independent
        return independent + 0.1 * context[:, 0, 0]


def _manual_log_probability(context, query):
    context32 = np.asarray(context, dtype=np.float32)
    query32 = np.asarray(query, dtype=np.float32)
    signal = np.float32(
        context32.sum(dtype=np.float32)
        + np.float32(0.5) * query32.sum(dtype=np.float32)
        + np.float32(2.0)
    )
    logits = signal * _BIN_SLOPE + _BIN_OFFSET
    shifted = logits.astype(np.float64) - float(np.max(logits))
    return shifted - np.log(np.exp(shifted).sum())


def _replay_shards():
    rng = np.random.default_rng(20260804)
    shards = []
    for shard_index in range(9):
        contexts = rng.normal(scale=0.25, size=(8, 30, 4))
        queries = rng.normal(scale=0.25, size=(8, 3))
        contexts[:, 0, 0] += shard_index + np.arange(8) / 8.0
        shards.append({"contexts": contexts, "queries": queries})
    return shards


def _replay_config():
    return {
        "pfn_replay_rows_per_stratum": 8,
        "pfn_batch_size": 64,
        "pfn_context_permutation_roll": 7,
        "pfn_batch_logp_atol": 5e-5,
        "pfn_combined_context_batch_logp_atol": 8e-5,
        "pfn_context_permutation_logp_atol": 3e-5,
        "pfn_replay_probability_atol": 1e-6,
        "pfn_replay_total_variation_atol": 3e-6,
        "pfn_replay_permutation_seeds": [1212240001, 1212240002],
    }


def test_native_scorer_matches_manual_per_row_probabilities():
    rng = np.random.default_rng(9)
    contexts = rng.normal(size=(7, 30, 4))
    queries = rng.normal(size=(7, 3))
    observed = _infer(
        _RowIndependentSetPFN().eval(),
        contexts,
        queries,
        4,
        torch.device("cpu"),
    )
    expected = np.stack(
        [
            _manual_log_probability(contexts[row], queries[row])
            for row in range(len(contexts))
        ]
    )
    # The scorer evaluates log_softmax in float32; the independent calculation
    # above normalizes the same float32 logits in float64.
    np.testing.assert_allclose(observed, expected, atol=6e-6, rtol=0)
    assert np.max(np.abs(expected[0] - expected[1])) > 1.0


def test_replay_guard_accepts_row_independent_set_model():
    config = _replay_config()
    result = _replay_guard(
        _RowIndependentSetPFN().eval(),
        _replay_shards(),
        config,
        torch.device("cpu"),
    )
    expected_comparisons = {
        "stress_repeat",
        "stress_singleton",
        "stress_batch_8",
        "stress_reverse",
        "stress_context_roll_view",
        "stress_context_roll_contiguous",
        "stress_context_roll_view_batch_8",
        "stress_context_roll_contiguous_batch_8",
        "stress_same_shape_block_permutation",
        "stress_fixed_shape_companion_replacement",
        "stress_fixed_shape_focal_relocation",
        "stress_remainder_35",
        "stress_remainder_36",
        "stress_remainder_43",
        "full_panel_repeat",
        "full_panel_reverse",
        "full_panel_same_shape_block_permutation",
        "full_panel_random_permutation_1",
        "full_panel_random_permutation_2",
        "full_panel_batch_8",
        "full_panel_context_roll_view",
        "full_panel_context_roll_contiguous",
        "full_panel_context_roll_view_batch_8",
        "full_panel_context_roll_contiguous_batch_8",
    }
    exact_controls = {
        "stress_repeat",
        "stress_same_shape_block_permutation",
        "stress_fixed_shape_companion_replacement",
        "stress_fixed_shape_focal_relocation",
        "stress_remainder_35",
        "stress_remainder_36",
        "stress_remainder_43",
        "full_panel_repeat",
        "full_panel_reverse",
        "full_panel_same_shape_block_permutation",
        "full_panel_random_permutation_1",
        "full_panel_random_permutation_2",
    }
    assert result["stress_rows"] == 72
    assert result["full_panel_rows"] == 72
    assert set(result["comparisons"]) == expected_comparisons
    assert result["exact_controls_pass"] is True
    assert result["pass"] is True
    for name in exact_controls:
        assert result["comparisons"][name] == {
            "bit_identical": True,
            "max_abs_logp_error": 0.0,
            "max_abs_probability_error": 0.0,
            "max_total_variation": 0.0,
        }
    assert result["batch_max_abs_logp_error"] <= config["pfn_batch_logp_atol"]
    assert (
        result["context_max_abs_logp_error"]
        <= config["pfn_context_permutation_logp_atol"]
    )
    assert (
        result["combined_max_abs_logp_error"]
        <= config["pfn_combined_context_batch_logp_atol"]
    )
    assert (
        result["approximate_max_abs_probability_error"]
        <= config["pfn_replay_probability_atol"]
    )
    assert (
        result["approximate_max_total_variation"]
        <= config["pfn_replay_total_variation_atol"]
    )


def test_replay_guard_rejects_batch_coupling():
    config = _replay_config()
    result = _replay_guard(
        _BatchCoupledPFN().eval(),
        _replay_shards(),
        config,
        torch.device("cpu"),
    )
    assert result["pass"] is False
    assert (
        result["comparisons"]["stress_singleton"]["max_abs_logp_error"]
        > config["pfn_batch_logp_atol"]
    )
    assert (
        result["comparisons"]["stress_fixed_shape_companion_replacement"][
            "bit_identical"
        ]
        is False
    )


def test_replay_guard_rejects_context_row_order_dependence():
    config = _replay_config()
    result = _replay_guard(
        _ContextRowOrderSensitivePFN().eval(),
        _replay_shards(),
        config,
        torch.device("cpu"),
    )
    view = result["comparisons"]["stress_context_roll_view"]
    contiguous = result["comparisons"]["stress_context_roll_contiguous"]
    assert result["pass"] is False
    assert result["exact_controls_pass"] is True
    for comparison in (view, contiguous):
        assert (
            comparison["max_abs_logp_error"]
            > config["pfn_context_permutation_logp_atol"]
        )
    assert view["max_abs_logp_error"] == contiguous["max_abs_logp_error"]
    assert view["max_abs_probability_error"] == contiguous["max_abs_probability_error"]
    assert view["max_total_variation"] == pytest.approx(
        contiguous["max_total_variation"], abs=2e-8, rel=0
    )


def test_replay_guard_rejects_combined_context_and_batch_interaction():
    shards = []
    for shard_index in range(9):
        contexts = np.zeros((16, 30, 4), dtype=np.float64)
        queries = np.zeros((16, 3), dtype=np.float64)
        contexts[:, 23, 0] = 1.0
        if shard_index == 8:
            contexts[:8, 23, 0] = 0.0
        shards.append({"contexts": contexts, "queries": queries})
    config = _replay_config()
    result = _replay_guard(
        _CombinedContextBatchSensitivePFN().eval(),
        shards,
        config,
        torch.device("cpu"),
    )
    assert result["pass"] is False
    assert result["batch_max_abs_logp_error"] <= config["pfn_batch_logp_atol"]
    assert (
        result["context_max_abs_logp_error"]
        <= config["pfn_context_permutation_logp_atol"]
    )
    assert (
        result["combined_max_abs_logp_error"]
        > config["pfn_combined_context_batch_logp_atol"]
    )


def test_replay_guard_rejects_output_row_rotation():
    result = _replay_guard(
        _OutputRowRotatingPFN().eval(),
        _replay_shards(),
        _replay_config(),
        torch.device("cpu"),
    )
    assert result["pass"] is False
    assert result["exact_controls_pass"] is False
    assert result["comparisons"]["stress_repeat"]["bit_identical"] is True
    assert (
        result["comparisons"]["stress_same_shape_block_permutation"]["bit_identical"]
        is False
    )
    assert (
        result["comparisons"]["full_panel_random_permutation_1"]["bit_identical"]
        is False
    )


def test_native_scorer_rejects_nonfinite_inputs():
    contexts = np.zeros((1, 30, 4), dtype=np.float64)
    contexts[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite input"):
        _infer(
            _RowIndependentSetPFN(),
            contexts,
            np.zeros((1, 3), dtype=np.float64),
            1,
            torch.device("cpu"),
        )
