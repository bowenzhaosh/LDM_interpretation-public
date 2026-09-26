"""Track A2 — tiny exact identifiability (A5 query panels, A6 query operators,
A7 identified sets).

Everything is exact on the finite latent space z = (k, o). For a query panel Q
we build the exact latent query operator (the joint numerator/denominator for
observational queries, the interventional operator for do() queries), analyze
its rank / singular spectrum / conditioning, and solve an L1 LP for the
identified set of latent posteriors compatible with a target predictive within
a tolerance. The primary scientific output is the per-order posterior range
(min/max compatible p(o | D)), and how it shrinks from Q1 -> Q2 -> Q3.

Convention for the tolerance: the constraint is
    sum_{i,b} |num_i[b](w) - t_i[b] * den_i(w)| / den_i^star
  + sum_{j,b} |op_j[b](w) - t_j[b]|   <=  tol
where den_i^star is the target's denominator (the observed-x evidence under
the target posterior). The first term is a scaled predictive-L1 distance; the
second is directly predictive-L1. All rows are L1-linearized into an LP via
scipy.optimize.linprog (HiGHS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .corrected_oracle import (
    World,
    ablated_weights,
    exact_joint_posterior,
    int_query_operator,
    obs_query_operator,
)
from .pilot_shared import N_BINS


# ---------------------------------------------------------------------------
# A5 — query panels (nested: Q1 <= Q2 <= Q3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsSpec:
    """Observational query: observe x_q (d-1 covariates), predict `target`."""
    target: int
    x_q: np.ndarray


@dataclass(frozen=True)
class IntSpec:
    """Interventional query: do(x_intervened = value), predict `target`."""
    intervened: int
    value: float
    target: int


@dataclass(frozen=True)
class QueryPanel:
    name: str
    obs: tuple[ObsSpec, ...] = ()
    int: tuple[IntSpec, ...] = ()


def build_query_panels(world: World, seed: int = 990_100_000) -> dict[str, QueryPanel]:
    """Nested query panels for the tiny world.

    Q1: one observational query (target 2, centered x_q).
    Q2: several observational queries (all three targets, a few x_q).
    Q3: Q2 plus single-node interventions with multiple values/targets.
    Panels are nested as sets of query specs.
    """
    rng = np.random.default_rng(seed)
    d = world.d

    # Q1: a single observational query.
    xq_0 = np.array([0.0] * (d - 1), dtype=np.float64)
    q1_spec = ObsSpec(target=d - 1, x_q=xq_0)
    q1 = QueryPanel(name="Q1", obs=(q1_spec,))

    # Q2: several observational queries across targets and covariate configs.
    # Nested: Q1's exact query is INCLUDED, so Q1 -> Q2 is a pure panel-growth
    # comparison (adding queries), not a query change.
    obs_q2 = [q1_spec]
    for target in range(d):
        for _ in range(2):
            x_q = rng.normal(0.0, 1.0, d - 1)
            obs_q2.append(ObsSpec(target=target, x_q=x_q))
    q2 = QueryPanel(name="Q2", obs=tuple(obs_q2))

    # Q3: Q2 plus single-node interventions (multiple values and targets).
    int_q3 = []
    for iv in range(d):
        for val in (-1.2, 0.6, 1.8):
            for target in range(d):
                if target != iv:
                    int_q3.append(IntSpec(intervened=iv, value=val, target=target))
    q3 = QueryPanel(name="Q3", obs=tuple(obs_q2), int=tuple(int_q3))

    return {"Q1": q1, "Q2": q2, "Q3": q3}


# ---------------------------------------------------------------------------
# A6 — exact latent query operator
# ---------------------------------------------------------------------------

@dataclass
class PanelOperator:
    """Exact linear query operator for a panel.

    For the panel (obs queries, int queries), the operator maps a latent
    posterior weight vector w (K*O,) to the joint observables:
      - for obs i: num_i[w] (B,), den_i[w] (scalar)
      - for int j: op_j[w] (B,)
    The stacked joint operator matrix A has rows
        [num_i (B rows), den_i (1 row)] per obs query, then [op_j (B rows)]
    per int query. Its rank / singular spectrum characterizes how much
    independent linear information the queries carry about the latent
    posterior.
    """
    world: World
    panel: QueryPanel
    num: list[np.ndarray]        # per obs query, (K*O, B)
    den: list[np.ndarray]        # per obs query, (K*O,)
    op: list[np.ndarray]         # per int query, (K*O, B)

    def joint_operator(self) -> np.ndarray:
        rows = []
        for i in range(len(self.num)):
            rows.append(self.num[i].T)           # (B, K*O)
            rows.append(self.den[i][None, :])    # (1, K*O)
        for j in range(len(self.op)):
            rows.append(self.op[j].T)            # (B, K*O)
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, self.world.n_latent))

    def exact_predictive(self, w_lo: np.ndarray) -> dict[str, Any]:
        """The panel response under posterior weights w_lo (for the LP)."""
        out = {"obs": [], "int": []}
        for i in range(len(self.num)):
            num = w_lo @ self.num[i]
            den = float(w_lo @ self.den[i])
            out["obs"].append({"num": num, "den": den, "pred": num / den})
        for j in range(len(self.op)):
            out["int"].append({"op": w_lo @ self.op[j], "pred": w_lo @ self.op[j]})
        return out


def build_panel_operator(world: World, panel: QueryPanel, coarse: bool = True) -> PanelOperator:
    num, den, op = [], [], []
    for spec in panel.obs:
        qop = obs_query_operator(world, spec.x_q, spec.target)
        num.append(qop.num)
        den.append(qop.den)
    for spec in panel.int:
        qop = int_query_operator(world, spec.intervened, spec.value, spec.target,
                                 coarse=coarse)
        op.append(qop.op)
    return PanelOperator(world=world, panel=panel, num=num, den=den, op=op)


def operator_identifiability(A: np.ndarray) -> dict[str, Any]:
    """Rank / effective-rank / singular-spectrum / conditioning of A.

    A is (n_rows, n_latent). Returns full SVD summary.
    """
    n_latent = A.shape[1]
    sv = np.linalg.svd(A, compute_uv=False)
    sv = sv[sv > 0.0]
    rho = 1e-12 * max(sv.max(), 1.0) if sv.size else 1e-12
    eff_rank = int(np.sum(sv > rho))
    # structural condition number on the subspace of singular vectors with
    # singular value above rho
    if eff_rank >= 2:
        cond_struct = float(sv[:eff_rank].max() / sv[eff_rank - 1])
    elif eff_rank == 1:
        cond_struct = 1.0
    else:
        cond_struct = float("inf")
    return {
        "n_rows": int(A.shape[0]),
        "n_latent": int(n_latent),
        "full_rank_possible": bool(A.shape[0] >= n_latent),
        "rank": int(eff_rank),
        "rank_deficit": int(max(n_latent - eff_rank, 0)),
        "effective_rank": eff_rank,
        "singular_values": sv.tolist(),
        "sv_max": float(sv[0]) if sv.size else 0.0,
        "sv_min_struct": float(sv[eff_rank - 1]) if eff_rank else 0.0,
        "condition_struct": cond_struct,
    }


# ---------------------------------------------------------------------------
# A7 — identified set (L1 LP)
# ---------------------------------------------------------------------------

def identified_set_ranges(
    world: World,
    panel: QueryPanel,
    op: PanelOperator,
    target_predictive: dict[str, Any],
    tol: float,
    solver: str = "highs",
    crossover: str | None = None,
) -> dict[str, Any]:
    """Min/max compatible per-order posterior probability within tolerance.

    Solves, for each order o, the LP
        min / max  sum_k w[k, o]
    s.t. w >= 0, sum w = 1, and the L1 predictive residual (scaled) <= tol.

    Returns per-order (min, max), interval widths, diameter, and whether the
    true posterior lies inside each interval.
    """
    from scipy.optimize import linprog

    n_lo = world.n_latent
    O = world.O
    B = N_BINS

    # --- build linear residual rows r_{m} @ w - slack <= target_m, etc. ----
    # L1 linearization: introduce one slack per (row, sign). We use a single
    # L1 budget: sum of all slacks <= tol.
    rows_lin: list[np.ndarray] = []     # each row: coefficients on w
    rhs_lin: list[float] = []
    # slack index bookkeeping
    def add_linear(a, b):
        rows_lin.append(np.asarray(a, dtype=np.float64))
        rhs_lin.append(float(b))

    # obs queries: |num[w] - t * den[w]| / den_star, per bin
    n_slack = 0
    for i in range(len(op.num)):
        num_i = op.num[i]                 # (K*O, B)
        den_i = op.den[i]                 # (K*O,)
        t = target_predictive["obs"][i]
        den_star = max(t["den"], 1e-12)
        for b in range(B):
            # row coefficient: num_i[:, b] - t["pred"][b] * den_i
            a = num_i[:, b] - t["pred"][b] * den_i
            a = a / den_star
            add_linear(a, 0.0)
            n_slack += 1
    for j in range(len(op.op)):
        op_j = op.op[j]                   # (K*O, B)
        t = target_predictive["int"][j]
        for b in range(B):
            a = op_j[:, b] - t["pred"][b]
            add_linear(a, 0.0)
            n_slack += 1

    # --- LP variables: w (n_lo), slack+ (n_slack), slack- (n_slack) ---
    n_w = n_lo
    n_var = n_w + 2 * n_slack
    # Constraints:
    #   sum w = 1
    #   for each row m: r_m @ w - s_m+ + s_m- <= 0   (and >= via the pair)
    #   sum_m (s_m+ + s_m-) <= tol
    #   w >= 0, s >= 0
    A_ub = []
    b_ub = []
    # |r_m(w)| <= tol via two slacks per row: r w <= s+_m and -r w <= s-_m,
    # with the L1 budget summing (s+_m + s-_m). (One-sided linearization would
    # let the negative half of each residual run free and halve the effective
    # tolerance; both inequalities are required.)
    for m in range(n_slack):
        row1 = np.zeros(n_var)
        row1[:n_w] = rows_lin[m]
        row1[n_w + m] = -1.0
        A_ub.append(row1)
        b_ub.append(0.0)
        row2 = np.zeros(n_var)
        row2[:n_w] = -rows_lin[m]
        row2[n_w + n_slack + m] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)
    # L1 budget: sum(s+) + sum(s-) <= tol
    budget = np.zeros(n_var)
    budget[n_w:n_w + 2 * n_slack] = 1.0
    A_ub.append(budget)
    b_ub.append(tol)
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_w] = 1.0
    b_eq = [1.0]

    # For interior point we only need the optimal objective, never a basic
    # solution, so skip HiGHS's simplex-based crossover: on degenerate LPs in
    # the mixed-noise regime the crossover phase can cycle (dual-simplex Phase-2
    # rebuild loop) even after IPM itself has converged. The objective is
    # identical with crossover off (verified numerically).
    # ``crossover`` exists ONLY for the Amendment F.4 attribution experiment,
    # which needs to reproduce the solver configuration in force before this
    # option was added. Default None = current behaviour, bit-identical to the
    # code above this parameter's introduction; "legacy" passes no options at
    # all. Nothing in the shipped pipeline sets it.
    if crossover == "legacy":
        lp_options: dict = {}
    else:
        lp_options = {"crossover": "off"} if solver == "highs-ipm" else {}
    ranges = {}
    for o in range(O):
        c = np.zeros(n_var)
        for k in range(world.K):
            c[k * O + o] = 1.0
        res_min = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                          A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n_var,
                          method=solver, options=lp_options)
        if not res_min.success:
            raise RuntimeError(f"LP min failed for order {o}: {res_min.message}")
        wmin = res_min.fun
        res_max = linprog(-c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                          A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n_var,
                          method=solver, options=lp_options)
        if not res_max.success:
            raise RuntimeError(f"LP max failed for order {o}: {res_max.message}")
        ranges[o] = (float(wmin), float(-res_max.fun))

    return ranges


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def analyze_world(
    world: World,
    n_contexts: int = 40,
    n_rows: int = 20,
    seed: int = 990_200_000,
    tol: float = 1e-3,
    panels: tuple[str, ...] = ("Q1", "Q2", "Q3"),
    solver: str = "highs",
    crossover: str | None = None,
) -> dict[str, Any]:
    """Full identifiability analysis over fixed contexts.

    n_rows defaults to 20 to match the Track B model context size, so the
    exact identifiability is evaluated at the same posterior sharpness the
    models see. Returns per-panel operator identifiability, identified-set
    widths, and whether the true order posterior lies inside each set.
    """
    from .corrected_world import default_context

    rng = np.random.default_rng(seed)
    panel_map = build_query_panels(world, seed=seed + 1)
    # Query operators depend only on (world, panel), not the context: build once.
    ops = {pname: build_panel_operator(world, panel_map[pname]) for pname in panels}
    results: dict[str, Any] = {"world": _world_summary(world), "contexts": []}

    for c in range(n_contexts):
        ctx = default_context(world, n_rows=n_rows, seed=seed + 1000 + c)
        post = exact_joint_posterior(world, ctx)
        ctx_row = {"context_index": c, "true_w_o": post["w_o"].tolist(),
                   "entropy_o": float(-np.sum(post["w_o"] * np.log(np.maximum(post["w_o"], 1e-300))))}
        for pname in panels:
            op = ops[pname]
            A = op.joint_operator()
            idf = operator_identifiability(A)
            targ = op.exact_predictive(post["w_lo"])
            ranges = identified_set_ranges(world, panel_map[pname], op, targ, tol,
                                           solver=solver, crossover=crossover)
            widths = [ranges[o][1] - ranges[o][0] for o in range(world.O)]
            contains_true = [ranges[o][0] - 1e-9 <= post["w_o"][o] <= ranges[o][1] + 1e-9
                             for o in range(world.O)]
            ctx_row[pname] = {
                "operator": idf,
                "ranges": {str(o): list(ranges[o]) for o in range(world.O)},
                "avg_width": float(np.mean(widths)),
                "max_width": float(np.max(widths)),
                "diameter": float(np.max(widths) - np.min(widths)),
                "contains_true": contains_true,
                "true_in_all": bool(all(contains_true)),
            }
        results["contexts"].append(ctx_row)

    # aggregate
    for pname in panels:
        agg = {"avg_width_mean": float(np.mean([c[pname]["avg_width"] for c in results["contexts"]])),
               "avg_width_max": float(np.max([c[pname]["avg_width"] for c in results["contexts"]])),
               "max_width_mean": float(np.mean([c[pname]["max_width"] for c in results["contexts"]])),
               "rank_mean": float(np.mean([c[pname]["operator"]["rank"] for c in results["contexts"]])),
               "cond_mean": float(np.mean([c[pname]["operator"]["condition_struct"] for c in results["contexts"]])),
               "true_in_all_frac": float(np.mean([c[pname]["true_in_all"] for c in results["contexts"]]))}
        results.setdefault("aggregates", {})[pname] = agg
    return results


def _world_summary(world: World) -> dict[str, Any]:
    return {"d": world.d, "K": world.K, "O": world.O,
            "n_latent": world.n_latent, "eps": world.spec.eps,
            "r": world.spec.r}
