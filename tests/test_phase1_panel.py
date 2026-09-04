from pathlib import Path

import numpy as np
import pytest

from pfn_dag_verify.phase1_panel import (
    COVARIANCE_SYMMETRY_RTOL,
    _validate_stream,
    split_stream,
)
from pfn_dag_verify.phase1_ordering import (
    generate_evaluation_stream,
    load_fleet_module,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeFleet:
    @staticmethod
    def validity_keep(sigmas: np.ndarray) -> np.ndarray:
        return np.ones(len(sigmas), dtype=bool)

    @staticmethod
    def bin_y(outcomes: np.ndarray) -> np.ndarray:
        return np.arange(len(outcomes), dtype=np.int64) % 100


def _fake_stream(count: int) -> dict[str, np.ndarray]:
    return {
        "contexts": np.arange(count * 30 * 4, dtype=np.float64).reshape(count, 30, 4),
        "queries": np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        "outcomes": np.arange(count, dtype=np.float64),
        "outcome_bins": np.arange(count, dtype=np.int64) % 100,
        "sigmas": np.tile(np.eye(4, dtype=np.float64), (count, 1, 1)),
        "true_orderings": np.arange(count, dtype=np.int64) % 24,
    }


def test_full_stream_is_split_only_after_generation_and_reconstructs_exactly():
    stream = _fake_stream(11)
    shards = split_stream(
        stream,
        prior_code=1,
        draw_index=2,
        evaluation_seed=881013002,
        atom_seeds=[881003101, 881003102, 881003103],
        identity_sha256="ab" * 32,
        nested_half_draw=0,
        nested_half_stop=4,
    )
    assert [len(inputs["row_id"]) for inputs, _ in shards] == [4, 4, 3]
    for bank, (inputs, labels) in enumerate(shards):
        assert np.all(inputs["stream_index"] % 3 == bank)
        assert np.array_equal(inputs["stream_index"] // 3, inputs["shard_local_index"])
        assert np.array_equal(inputs["row_key_sha256"], labels["row_key_sha256"])
        assert not np.any(inputs["nested_half_mask"])
    reconstructed = np.empty_like(stream["contexts"])
    for inputs, _ in shards:
        reconstructed[inputs["stream_index"]] = inputs["contexts"]
    np.testing.assert_array_equal(reconstructed, stream["contexts"])


def test_nested_half_mask_is_frozen_by_draw_and_original_stream_index():
    shards = split_stream(
        _fake_stream(11),
        prior_code=0,
        draw_index=0,
        evaluation_seed=881003000,
        atom_seeds=[881003101, 881003102, 881003103],
        identity_sha256="cd" * 32,
        nested_half_draw=0,
        nested_half_stop=5,
    )
    marked = sorted(
        int(index)
        for inputs, _ in shards
        for index in inputs["stream_index"][inputs["nested_half_mask"] == 1]
    )
    assert marked == [0, 1, 2, 3, 4]


def test_stream_validation_accepts_only_float64_symmetry_roundoff_without_mutation():
    stream = _fake_stream(2)
    stream["sigmas"][0, 0, 1] = 0.2
    stream["sigmas"][0, 1, 0] = np.nextafter(0.2, np.inf)
    before = stream["sigmas"].copy()

    diagnostics = _validate_stream(_FakeFleet(), stream, 2)

    np.testing.assert_array_equal(stream["sigmas"], before)
    assert diagnostics["covariance_symmetry_rtol"] == COVARIANCE_SYMMETRY_RTOL
    assert diagnostics["covariance_max_abs_asymmetry"] > 0.0
    assert (
        diagnostics["covariance_max_relative_asymmetry"]
        <= COVARIANCE_SYMMETRY_RTOL
    )


def test_stream_validation_rejects_material_covariance_asymmetry():
    stream = _fake_stream(2)
    stream["sigmas"][0, 0, 1] = 0.2
    stream["sigmas"][0, 1, 0] = 0.2000001

    with pytest.raises(
        RuntimeError,
        match="exceeds the float64 symmetry roundoff bound",
    ):
        _validate_stream(_FakeFleet(), stream, 2)


@pytest.mark.parametrize(
    ("prior", "seed"),
    [
        ("C", 881003000),
        ("C", 881003001),
        ("C", 881003002),
        ("N", 881013000),
        ("N", 881013001),
        ("N", 881013002),
    ],
)
def test_registered_confirmation_streams_satisfy_symmetry_roundoff_bound(
    prior: str, seed: int
):
    fleet = load_fleet_module(ROOT / "artifacts/phase1/d4_generator.py")
    stream = generate_evaluation_stream(fleet, prior, 1067, 30, seed)

    diagnostics = _validate_stream(fleet, stream, 1067)

    assert diagnostics["covariance_max_abs_asymmetry"] <= np.finfo(float).eps * 2
    assert (
        diagnostics["covariance_max_relative_asymmetry"]
        <= COVARIANCE_SYMMETRY_RTOL
    )
