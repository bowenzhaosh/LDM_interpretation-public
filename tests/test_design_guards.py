import numpy as np

from pfn_dag_verify.generative import generate_group


def test_group_is_deterministic_and_has_repeated_shared_latent_continuations():
    a = generate_group(np.random.default_rng(101), n_continuations=8)
    b = generate_group(np.random.default_rng(101), n_continuations=8)
    np.testing.assert_array_equal(a.sigma, b.sigma)
    np.testing.assert_array_equal(a.core, b.core)
    np.testing.assert_array_equal(a.reference, b.reference)
    np.testing.assert_array_equal(a.continuations, b.continuations)
    assert a.core.shape == (20, 2)
    assert a.reference.shape == (10, 2)
    assert a.continuations.shape == (8, 10, 2)
    assert a.graph in (0, 1)
    assert np.unique(a.continuations.reshape(8, -1), axis=0).shape[0] == 8

