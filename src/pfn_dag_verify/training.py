"""Run the original finite-study scientific code in a small portable CPU example."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pfn_dag_verify.corrected_models import PFN, ModelConfig, gen_batch, lr_at
from pfn_dag_verify.corrected_oracle import exact_joint_posterior, obs_query_operator
from pfn_dag_verify.corrected_world import make_world
from .target_construction import (
    coupled_row, forward, input_key, loss_for, mixture, query_components,
)

ARMS = ("Q", "qz", "signed", "FreshY1")
API = SimpleNamespace(exact_joint_posterior=exact_joint_posterior,
                      obs_query_operator=obs_query_operator, mixture=mixture)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(config):
    """Use the original support seed, pilot inputs, and coupled target sampler."""
    world = make_world(k=8, d=3, eps=.75, seed=config["world_seed"])
    assert world.K == 8 and world.O == 6
    key, count = input_key("pilot", 0)
    inputs = gen_batch(world, np.random.default_rng(np.random.SeedSequence(key)),
                       count, 20, 7)[0].astype(np.float32)
    inputs = inputs[:config["train_rows"]]
    rows = [coupled_row(API, world, x, "pilot", 0, i)
            for i, x in enumerate(inputs)]
    Q = np.stack([r["Q"] for r in rows])
    # This is the original TargetStore's one-time FP32 normalization.
    q = np.stack([r["chosen"] for r in rows])
    q = q / q.sum(-1, keepdims=True)
    y = np.stack([r["labels"] for r in rows])[..., 0]
    key, count = input_key("panel", 0)
    evaluation = gen_batch(world, np.random.default_rng(np.random.SeedSequence(key)),
                          count, 20, 7)[0].astype(np.float32)
    evaluation = evaluation[:config["evaluation_rows"]]
    Q_eval = np.stack([np.stack([p[2] for p in query_components(API, world, x)[1]])
                       for x in evaluation])
    return inputs, Q, q, y, evaluation, Q_eval, world


def state_sha(model):
    return array_sha(torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy())


def train_arms(config, prepared):
    X, Q, q, y, evaluation, Q_eval, world = prepared
    seed = config["optimization_seed"]
    seed_index = seed - 71
    results = {}
    models = {}
    for arm in ARMS:
        torch.manual_seed(1000 * seed + 7)
        model = PFN(ModelConfig(name="base", d=3)).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999),
                                     eps=1e-8, weight_decay=0, amsgrad=False,
                                     maximize=False, foreach=None, capturable=False,
                                     differentiable=False, fused=None)
        initial = state_sha(model)
        rng = np.random.default_rng(10000 + seed)
        perm, pos, visit = rng.permutation(len(X)), 0, 0
        trace, losses = [], []
        for step in range(config["updates"]):
            batch_size = config["batch_size"]
            if pos + batch_size > len(X):
                perm, pos, visit = rng.permutation(len(X)), 0, visit + 1
            if visit >= 7:
                raise ValueError("The authenticated target sampler has seven visits; reduce updates.")
            indices = perm[pos:pos + batch_size]
            pos += batch_size
            trace.append({"indices": indices.tolist(), "visit": visit})
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, .001, 200000, 200)
            targets = (Q[indices], q[indices, seed_index, visit],
                       y[indices, seed_index, visit])
            loss = loss_for(forward(model, X[indices], "cpu"), targets, arm)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logp = forward(model, evaluation, "cpu").double().log_softmax(-1).numpy()
        positive = Q_eval > 0
        logq = np.zeros_like(Q_eval)
        np.log(Q_eval, out=logq, where=positive)
        risk = float(np.mean(np.sum(Q_eval * (logq - logp), axis=-1)))
        results[arm] = {"source_KL": risk, "initial_parameter_sha256": initial,
                        "final_parameter_sha256": state_sha(model),
                        "traversal": trace, "training_losses": losses}
        models[arm] = model
    assert len({r["initial_parameter_sha256"] for r in results.values()}) == 1
    assert all(r["traversal"] == results["Q"]["traversal"] for r in results.values())
    assert len({r["final_parameter_sha256"] for r in results.values()}) == 4
    return {"scope": "Execution example only; not a reproduction of the paper's effect size or sign.",
            "config": config, "versions": {"numpy": np.__version__, "torch": torch.__version__},
            "support": {"covariance_atoms": world.K, "orderings": world.O,
                        "joint_components": world.K * world.O,
                        "sigmas_sha256": array_sha(world.sigmas)},
            "training_inputs_sha256": array_sha(X),
            "evaluation_inputs_sha256": array_sha(evaluation),
            "arms": results,
            "Fresh_minus_qz_source_KL": results["FreshY1"]["source_KL"] - results["qz"]["source_KL"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert 1 <= config["train_rows"] <= 64 and 1 <= config["evaluation_rows"] <= 4096
    assert 1 <= config["batch_size"] <= config["train_rows"]
    assert 71 <= config["optimization_seed"] <= 75 and config["updates"] >= 1
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    result = train_arms(config, prepare(config))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
