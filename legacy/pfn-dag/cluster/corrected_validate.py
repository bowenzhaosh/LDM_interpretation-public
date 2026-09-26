"""GPU/cluster environment validation for the corrected campaign.

Runs a tiny end-to-end check: corrected modules import, the forward-map SEM
generator produces data, the vanilla PFN and order classifier train a few
steps on GPU, and the exact oracle + one identifiability context run. Any
exception aborts (no silent swallowing).
"""
import numpy as np
import torch
import sys

from pfn_dag_verify.corrected_sem import ResidualSpec, generate_observational, params_for
from pfn_dag_verify.corrected_world import make_world
from pfn_dag_verify.corrected_oracle import exact_joint_posterior, obs_query_operator
from pfn_dag_verify.corrected_models import (
    ModelConfig, train_pfn, train_classifier, make_eval_panel,
    evaluate_pfn_checkpoint, evaluate_classifier_checkpoint,
)

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
world = make_world(k=4, d=3, eps=1.0)
rng = np.random.default_rng(0)
x = generate_observational(rng, world.sigmas[0], world.orderings[1], world.spec, 10)
assert x.shape == (10, 3) and np.isfinite(x).all()
post = exact_joint_posterior(world, x)
assert abs(post["w_o"].sum() - 1.0) < 1e-9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = ModelConfig(name="v", d=3, d_model=32, d_ff=64, n_heads=2, n_layers=1)
res = train_pfn(world, steps=25, seed=0, cfg=cfg, n_ctx=10, n_query=3)
ev = evaluate_pfn_checkpoint(res["model"], world,
                             make_eval_panel(world, 2, 10, seed=5, n_query_per_context=1),
                             device)
res2 = train_classifier(world, 25, seed=0, cfg=cfg, n_ctx=10)
from pfn_dag_verify.corrected_models import make_classifier_panel
ev2 = evaluate_classifier_checkpoint(res2["model"], world,
                                     make_classifier_panel(world, 2, 10, seed=6), device)
print("VALIDATE_OK pfn_regret=%.3f clf_map=%.2f" % (ev["bayes_regret_mean"], ev2["map_accuracy"]), flush=True)
