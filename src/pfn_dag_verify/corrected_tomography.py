"""PFN posterior tomography (Track A+B connection).

For a trained vanilla PFN, collect its predictions on the exact nested query
panels Q1/Q2/Q3. For each held-out context:

  (a) Projection: find the simplex latent posterior w minimizing the L1
      divergence between the PFN's predictions and the exact predictions
      generated from w through the exact query operators. The minimum is the
      "coherence residual" (how far the PFN's outputs are from ANY exact
      latent posterior).
  (b) Partial identification: with a tolerance set to the PFN's measured
      predictive error, compute min/max compatible order probability
      (identified-set widths) and compare with the exact p(o | D).

For each context we record: PFN Bayes regret, exact structural posterior,
projected structural posterior, structural JS/TV, identified-set widths,
query rank/effective rank, condition number, coherence residual.

Both (a) and (b) are LPs (L1 linearized); (a) minimizes the residual, (b)
fixes it at the tolerance and optimizes each order marginal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .corrected_identifiability import (
    PanelOperator,
    build_panel_operator,
    build_query_panels,
    operator_identifiability,
)
from .corrected_oracle import World, exact_joint_posterior
from .pilot_shared import N_BINS


def pfn_panel_predictions(model, world: World, panel, contexts, device):
    """PFN predictions on every query of `panel` for each context.

    Returns a list (per context) of dicts matching PanelOperator.exact_predictive:
      obs: [{'pred': (B,)} per obs query], int: [{'pred': (B,)} per int query].
    """
    from .corrected_models import PFN
    import torch

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    out = []
    for ctx in contexts:
        ctx_t = torch.tensor(ctx, dtype=torch.float32, device=torch_device).unsqueeze(0)
        tok = torch.full((1,), 2, dtype=torch.long, device=torch_device)
        obs_preds = []
        for spec in panel.obs:
            qxy = torch.tensor(spec.x_q, dtype=torch.float32,
                               device=torch_device).reshape(1, 1, -1)
            p = model.predict_bin_probs(ctx_t, qxy, tok)[0, 0]
            obs_preds.append({"pred": p})
        int_preds = []
        for spec in panel.int:
            # Interventional queries are NOT part of the PFN's training
            # interface; the vanilla PFN has no intervention channel. Mark as
            # unavailable (the tomography uses obs panels for the PFN and the
            # exact interventional oracle only for the identifiability side).
            int_preds.append(None)
        out.append({"obs": obs_preds, "int": int_preds})
    return out


# D2 uses interior point on the ident ladder with crossover off; the same
# options are used here so the two LP paths are configured identically.
_LP_OPTS = {"highs-ipm": {"crossover": "off"}}


def _lp_project_or_ranges(world: World, op: PanelOperator, target: dict,
                          mode: str, tol: float, which_orders=None,
                          solver: str = "highs"):
    """Shared LP machinery for projection and partial identification.

    mode='project': minimize sum of L1 predictive residual over the simplex.
    mode='id': fix residual <= tol, return min/max per-order marginals.

    ``solver`` is the scipy method. The DEFAULT IS UNCHANGED at "highs" (dual
    simplex), because every tomography number already on disk was computed with
    it and flipping the default would silently move measurements nobody
    re-derived. Callers that need the interior-point path opt in.

    Why the knob exists. The identified-set LP fixes the residual at
    ``tol_eff`` and then optimises each order marginal, so the feasible region
    GROWS with the tolerance -- and ``run_tomography``'s caller derives the
    tolerance from the model's own predictive error. For a converged model that
    is a tight constraint. For an UNTRAINED one it is not: at 100 steps the
    measured js_mean is about 0.38, giving tol around 0.76, at which point the
    identified set is the WHOLE SIMPLEX (measured id_avg_width 1.00000). That
    matters for F.2c, whose floor curve pushes a dose-0 checkpoint through this
    readout by construction, so an uninformative identified set is the floor
    curve's normal condition rather than an edge case.

    The knob is here so this path can be configured like D2's ident ladder,
    which runs interior point with crossover off. It is NOT offered as a fix for
    the pre-flight that burned 2h42m of CPU on the cluster without finishing:
    measured on the Mac at tol 0.76, dose-0, one context, the two solvers are
    the same speed (19.2 s interior point against 18.1 s dual simplex) and both
    return the same width. THE CLUSTER SLOWDOWN IS NOT YET ATTRIBUTED and no
    cause is claimed for it; the pre-flight is instrumented per gate to find out
    rather than to guess.
    """
    from scipy.optimize import linprog

    n_lo = world.n_latent
    O = world.O
    B = N_BINS

    # build linear rows (obs queries with den scaling, int queries direct)
    rows: list[np.ndarray] = []
    for i in range(len(op.num)):
        num_i, den_i = op.num[i], op.den[i]
        t = target["obs"][i]
        den_star = max(t.get("den", 1.0), 1e-12)
        for b in range(B):
            rows.append((num_i[:, b] - t["pred"][b] * den_i) / den_star)
    for j in range(len(op.op)):
        op_j = op.op[j]
        t = target["int"][j]
        if t is None:
            continue
        for b in range(B):
            rows.append(op_j[:, b] - t["pred"][b])

    n_rows = len(rows)
    n_var = n_lo + 2 * n_rows  # w + slack+ + slack-

    A_ub, b_ub = [], []
    # Both |r_m| inequalities (see identified_set_ranges): r w <= s+_m and
    # -r w <= s-_m.
    for m in range(n_rows):
        row1 = np.zeros(n_var)
        row1[:n_lo] = rows[m]
        row1[n_lo + m] = -1.0
        A_ub.append(row1)
        b_ub.append(0.0)
        row2 = np.zeros(n_var)
        row2[:n_lo] = -rows[m]
        row2[n_lo + n_rows + m] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)
    A_eq = np.zeros((1, n_var)); A_eq[0, :n_lo] = 1.0; b_eq = [1.0]
    bounds = [(0, None)] * n_var

    if mode == "project":
        c = np.zeros(n_var)
        c[n_lo:n_lo + 2 * n_rows] = 1.0  # minimize total slack
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver,
                      options=_LP_OPTS.get(solver))
        if not res.success:
            raise RuntimeError(f"projection LP failed: {res.message}")
        residual = res.fun
        w = res.x[:n_lo]
        w = w / max(w.sum(), 1e-12)
        w_o = np.array([sum(w[k * O + o] for k in range(world.K)) for o in range(O)])
        return {"residual": float(residual), "w_lo": w, "w_o": w_o}

    # mode == "id": fix residual <= tol, optimize each order marginal
    budget = np.zeros(n_var); budget[n_lo:n_lo + 2 * n_rows] = 1.0
    A_ub.append(budget); b_ub.append(tol)
    orders = which_orders if which_orders is not None else range(O)
    ranges = {}
    for o in orders:
        c = np.zeros(n_var)
        for k in range(world.K):
            c[k * O + o] = 1.0
        opts = _LP_OPTS.get(solver)
        rmin = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                       A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver,
                       options=opts)
        rmax = linprog(-c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                       A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver,
                       options=opts)
        if not rmin.success or not rmax.success:
            raise RuntimeError("identified-set LP failed")
        ranges[o] = (float(rmin.fun), float(-rmax.fun))
    return {"ranges": ranges}


def tomography_context(
    world: World,
    op: PanelOperator,
    panel,
    ctx: np.ndarray,
    pfn_preds: dict,
    exact_post: dict,
    tol: float,
    solver: str = "highs",
) -> dict[str, Any]:
    """Full tomography record for one context and one panel."""
    A = op.joint_operator()
    idf = operator_identifiability(A)
    # projection (free simplex): the minimum residual ANY latent posterior can
    # achieve against the PFN's predictions
    proj = _lp_project_or_ranges(world, op, pfn_preds, "project", tol,
                                 solver=solver)
    # partial identification around the PFN's predictions. The identified set
    # is only non-empty when the tolerance is at least the projection residual,
    # so use the effective tolerance tol_eff = max(tol, residual * 1.2).
    tol_eff = max(tol, proj["residual"] * 1.2)
    id_ = _lp_project_or_ranges(world, op, pfn_preds, "id", tol_eff,
                                solver=solver)

    # structural comparison: exact vs projected
    w_o_exact = exact_post["w_o"]
    w_o_proj = proj["w_o"]
    p_e = np.maximum(w_o_exact, 1e-300)
    p_p = np.maximum(w_o_proj, 1e-300)
    m = 0.5 * (p_e + p_p)
    js = 0.5 * np.sum(p_e * np.log(p_e / m)) + 0.5 * np.sum(p_p * np.log(p_p / m))
    tv = 0.5 * np.sum(np.abs(p_e - p_p))
    widths = [id_["ranges"][o][1] - id_["ranges"][o][0] for o in range(world.O)]
    return {
        "operator": idf,
        "projection_residual": proj["residual"],
        "projected_w_o": w_o_proj.tolist(),
        "structural_js": float(js),
        "structural_tv": float(tv),
        "id_ranges": {str(o): list(id_["ranges"][o]) for o in range(world.O)},
        "id_avg_width": float(np.mean(widths)),
        "id_max_width": float(np.max(widths)),
        "true_in_id": bool(all(id_["ranges"][o][0] - 1e-9 <= w_o_exact[o] <= id_["ranges"][o][1] + 1e-9
                               for o in range(world.O))),
    }


def run_tomography(
    world: World,
    model,
    contexts: list[np.ndarray],
    panel_names: tuple[str, ...] = ("Q1", "Q2", "Q3"),
    tol: float = 1e-2,
    device: str = "cuda",
    seed: int = 990_300_000,
    solver: str = "highs",
) -> dict[str, Any]:
    """Run the full tomography over contexts and panels.

    ``solver`` defaults to the dual simplex every existing tomography number was
    computed with. Pass "highs-ipm" for an untrained or poorly-fit model, where
    the tolerance derived from the model's own predictive error leaves the
    identified set nearly the whole simplex; see ``_lp_project_or_ranges``.
    """
    panels = build_query_panels(world, seed=seed + 1)
    ops = {pname: build_panel_operator(world, panels[pname]) for pname in panel_names}
    results = {"world": {"d": world.d, "K": world.K, "O": world.O,
                         "eps": world.spec.eps},
               "lp_solver": solver, "tol": float(tol), "contexts": []}
    for ci, ctx in enumerate(contexts):
        exact_post = exact_joint_posterior(world, ctx)
        row = {"context_index": ci, "true_w_o": exact_post["w_o"].tolist(),
               "entropy_o": float(-np.sum(exact_post["w_o"] * np.log(np.maximum(exact_post["w_o"], 1e-300))))}
        for pname in panel_names:
            pfn_preds = pfn_panel_predictions(model, world, panels[pname], [ctx], device)[0]
            rec = tomography_context(world, ops[pname], panels[pname], ctx,
                                     pfn_preds, exact_post, tol, solver=solver)
            row[pname] = rec
        results["contexts"].append(row)
    return results


def aggregate_tomography(results: dict[str, Any]) -> dict[str, Any]:
    agg = {}
    for pname in ("Q1", "Q2", "Q3"):
        if pname not in results["contexts"][0]:
            continue
        keys = ["projection_residual", "structural_js", "structural_tv", "id_avg_width", "id_max_width"]
        agg[pname] = {k: float(np.mean([c[pname][k] for c in results["contexts"]]))
                      for k in keys}
        agg[pname]["true_in_id_frac"] = float(np.mean([c[pname]["true_in_id"] for c in results["contexts"]]))
        agg[pname]["rank_mean"] = float(np.mean([c[pname]["operator"]["rank"] for c in results["contexts"]]))
        agg[pname]["cond_mean"] = float(np.mean([c[pname]["operator"]["condition_struct"] for c in results["contexts"]]))
    return agg
