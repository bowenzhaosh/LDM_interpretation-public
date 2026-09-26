import numpy as np
import pytest

from pfn_dag_verify.generative import generate_group
from pfn_dag_verify.oracle import GridOracle, ScalarGridOracle


QUERIES = np.array([-3.75, -2.25, -1.25, -0.25, 0.25, 1.25, 2.25, 3.75])


def test_scalar_and_vector_oracles_agree_and_endpoints_normalize():
    rng = np.random.default_rng(810001)
    group = generate_group(rng, n_continuations=2)
    scalar = ScalarGridOracle(queries=QUERIES, quadrature=5)
    vector = GridOracle(queries=QUERIES, quadrature=5)
    for context in (group.core, np.concatenate([group.core, group.reference])):
        a = scalar.evaluate(context)
        b = vector.evaluate(context)
        assert abs(a.ell - b.ell) <= 1e-10
        np.testing.assert_allclose(a.f0, b.f0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(a.f1, b.f1, atol=1e-10, rtol=0)
        np.testing.assert_allclose(b.f0.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(b.f1.sum(axis=1), 1.0, atol=1e-12, rtol=0)


def test_reverse_endpoint_is_query_conditioned_not_broadcast():
    rng = np.random.default_rng(810002)
    group = generate_group(rng, n_continuations=2)
    bundle = GridOracle(queries=QUERIES, quadrature=5).evaluate(group.core)
    assert np.max(np.abs(bundle.f0[0] - bundle.f0[-1])) > 1e-4
    assert np.max(np.abs(bundle.f1[0] - bundle.f1[-1])) > 1e-4


def test_invalid_context_fails_closed():
    oracle = GridOracle(queries=QUERIES, quadrature=5)
    bad = np.zeros((20, 2), dtype=float)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        oracle.evaluate(bad)
