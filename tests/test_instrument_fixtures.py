import numpy as np

from pfn_dag_verify.instrument import (
    js_divergence,
    project_coordinate,
    project_coordinate_batch,
    project_kl,
    project_kl_batch,
    reconstruct_updated_prediction,
)


def _endpoints(seed=7, q=8, bins=100):
    rng = np.random.default_rng(seed)
    f0 = rng.gamma(2.0, 1.0, size=(q, bins))
    f1 = rng.gamma(2.0, 1.0, size=(q, bins))
    f0 /= f0.sum(axis=1, keepdims=True)
    f1 /= f1.sum(axis=1, keepdims=True)
    return f0, f1


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_exact_and_tempered_mixtures_recover_programmed_gain():
    f0, f1 = _endpoints()
    ell = np.linspace(-2.0, 2.0, 41)
    for tau in (0.25, 0.5, 0.75, 1.0):
        recovered = []
        for value in ell:
            w = _sigmoid(tau * value)
            p = (1.0 - w) * f0 + w * f1
            coord = project_coordinate(p, f0, f1)
            kl = project_kl(p, f0, f1)
            assert abs(coord.w - w) <= 1e-4
            assert abs(kl.w - w) <= 1e-3
            recovered.append(coord.g)
        slope, intercept = np.polyfit(ell, recovered, 1)
        assert abs(slope - tau) <= 0.02
        assert abs(intercept) <= 0.02


def test_label_swap_preserves_prediction_and_flips_log_odds():
    f0, f1 = _endpoints()
    w = 0.73
    p = (1.0 - w) * f0 + w * f1
    original = project_coordinate(p, f0, f1)
    swapped = project_coordinate(p, f1, f0)
    assert abs(swapped.w - (1.0 - original.w)) <= 1e-12
    assert abs(swapped.g + original.g) <= 1e-10
    np.testing.assert_allclose(
        (1.0 - original.w) * f0 + original.w * f1,
        (1.0 - swapped.w) * f1 + swapped.w * f0,
        atol=1e-12,
        rtol=0,
    )
    assert abs(swapped.normalized_residual - original.normalized_residual) <= 1e-12


def test_mean_pooling_is_a_high_fit_non_bayesian_negative_control():
    rng = np.random.default_rng(91)
    g1 = rng.normal(size=400)
    g2 = rng.normal(size=400)
    g12 = 0.5 * (g1 + g2)
    slope, intercept = np.polyfit(g1 + g2, g12, 1)
    pred = intercept + slope * (g1 + g2)
    r2 = 1.0 - np.sum((g12 - pred) ** 2) / np.sum((g12 - g12.mean()) ** 2)
    assert abs(slope - 0.5) <= 1e-12
    assert r2 >= 0.999999999


def test_native_reconstruction_is_exact_for_bayesian_update():
    f0_base, f1_base = _endpoints(seed=11)
    f0_target, f1_target = _endpoints(seed=12)
    w_base = 0.35
    delta_ell = 0.8
    base_odds = np.log(w_base / (1.0 - w_base))
    w_target = _sigmoid(base_odds + delta_ell)
    p_target = (1.0 - w_target) * f0_target + w_target * f1_target
    out = reconstruct_updated_prediction(
        p_target, f0_target, f1_target, w_base=w_base, delta_ell=delta_ell
    )
    assert abs(out.updated_w - w_target) <= 1e-12
    assert out.normalized_residual <= 1e-10


def test_orthogonal_native_error_is_not_hidden_by_coordinate_recovery():
    f0, f1 = _endpoints(seed=81)
    w = 0.4
    base = (1.0 - w) * f0 + w * f1
    rng = np.random.default_rng(82)
    perturbation = rng.normal(size=base.shape)
    perturbation -= perturbation.mean(axis=1, keepdims=True)
    delta = f1 - f0
    perturbation -= np.sum(perturbation * delta) / np.sum(delta * delta) * delta
    scale = 1e-4
    perturbed = base + scale * perturbation
    while np.any(perturbed <= 0):
        scale *= 0.5
        perturbed = base + scale * perturbation
    perturbed /= perturbed.sum(axis=1, keepdims=True)
    recovered = project_coordinate(perturbed, f0, f1)
    assert abs(recovered.w - w) <= 1e-8
    assert recovered.normalized_residual > 0


def test_js_is_symmetric_and_zero_only_for_equal_endpoints():
    f0, f1 = _endpoints()
    assert abs(js_divergence(f0, f1) - js_divergence(f1, f0)) <= 1e-12
    assert js_divergence(f0, f1) > 0
    assert js_divergence(f0, f0) <= 1e-15


def test_batch_coordinate_and_kl_match_scalar_solvers():
    f0, f1 = _endpoints(seed=107)
    weights = np.array([0.1, 0.3, 0.6, 0.9])
    f0_batch = np.broadcast_to(f0, (len(weights),) + f0.shape).copy()
    f1_batch = np.broadcast_to(f1, (len(weights),) + f1.shape).copy()
    predictions = f0_batch + weights[:, None, None] * (f1_batch - f0_batch)
    coordinate = project_coordinate_batch(predictions, f0_batch, f1_batch)
    kl = project_kl_batch(predictions, f0_batch, f1_batch)
    np.testing.assert_allclose(coordinate.w, weights, atol=1e-12, rtol=0)
    np.testing.assert_allclose(kl.w, weights, atol=1e-9, rtol=0)
    for index in range(len(weights)):
        assert abs(project_kl(predictions[index], f0, f1).w - kl.w[index]) <= 1e-6


def test_batch_projection_accepts_normal_float32_softmax_roundoff():
    f0, f1 = _endpoints(seed=207)
    weights = np.array([0.2, 0.8], dtype=np.float32)
    f0_batch = np.broadcast_to(f0.astype(np.float32), (2,) + f0.shape).copy()
    f1_batch = np.broadcast_to(f1.astype(np.float32), (2,) + f1.shape).copy()
    prediction = f0_batch + weights[:, None, None] * (f1_batch - f0_batch)
    result = project_coordinate_batch(prediction, f0_batch, f1_batch)
    np.testing.assert_allclose(result.w, weights, atol=1e-6, rtol=0)
