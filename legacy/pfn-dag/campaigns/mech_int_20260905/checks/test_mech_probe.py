"""Arm R pipeline checks (PRESPEC_internal §1 Files): (1) the uniform-p̂ hybrid equals the registered
order_ablated oracle bitwise; (2) a head trained on shuffled targets scores no better than uniform-ish
(b >= 0.95 on a synthetic regret/G construction is a pipeline property, checked in the fit run itself);
(3) the S identity of the activation stage on 3 registered contexts. CPU, seconds."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _world(eps=0.75):
    os.environ.update(MECH_PANEL_SEED="770000101", MECH_SPLIT_SEED="880000101", MECH_N_PER_HALF="1000", MECH_LADDER="0")
    for k in ("MECH_K", "MECH_D", "MECH_WORLD_SEED"):
        os.environ.pop(k, None)
    from pfn_dag_verify.corrected_world import make_world
    return make_world(k=8, d=3, eps=eps)


def test_uniform_phat_hybrid_equals_order_ablated():
    from pfn_dag_verify.corrected_oracle import ablated_weights, exact_joint_posterior
    from pfn_dag_verify.corrected_sem import generate_observational
    from mech_probe_fit import hybrid_weights
    world = _world()
    rng = np.random.default_rng(3)
    for _ in range(5):
        k, o = int(rng.integers(world.K)), int(rng.integers(world.O))
        ctx = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, 20)
        post = exact_joint_posterior(world, ctx)
        w_h = hybrid_weights(post["w_lo"], np.asarray(world.prior_order, float), world.K, world.O)
        w_a = ablated_weights(world, post, "order_ablated")
        assert np.array_equal(w_h, w_a) or np.max(np.abs(w_h - w_a)) < 1e-15


@pytest.mark.skipif(not (ROOT / "campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets/eps0p75/base_s3_ck500000.pt").is_file(),
                    reason="registered checkpoint not on this box")
def test_S_identity_three_contexts():
    import mech_predgain as PG
    from mech_interventions import half_b_with_latents
    from mech_probe_acts import forward_sites, load_model
    from pfn_dag_verify.corrected_sem import generate_observational
    world = _world()
    z = np.load(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000/predgain_eps0p75_ck500000.npz")
    names = list(z["names"])
    ctxs, ks, os_ = half_b_with_latents(world)
    m = load_model("base", 3, ROOT / "campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets/eps0p75/base_s3_ck500000.pt")
    for i in (0, 500, 999):
        rng = np.random.default_rng(PG.Q_SEED_ROOT + 750 * 1000 + i)
        rows = generate_observational(rng, world.sigmas[ks[i]], world.orderings[os_[i]], world.spec, 8)
        _, _, P = forward_sites(m, np.asarray(ctxs[i], float)[None], rows[None])
        assert np.max(np.abs(P[0] - z["P"][i, :, names.index("model_s3"), :])) < 1e-5


def test_shuffled_targets_carry_no_information():
    from mech_probe_fit import train_head
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 16)).astype(np.float32)
    W = rng.normal(size=(16, 100)); L = X @ W
    T = np.exp(L - L.max(1, keepdims=True)); T /= T.sum(1, keepdims=True); T = T.astype(np.float32)
    pred, _ = train_head(X, T[rng.permutation(len(T))], "mlp", seed=1, epochs=10)
    P = pred(X[:500])
    ce = -(T[:500] * np.log(P)).sum(1).mean()
    H = -(T[:500] * np.log(T[:500])).sum(1).mean()
    assert ce > H + 0.5          # far from the target: the head learned nothing usable
