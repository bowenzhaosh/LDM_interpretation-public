"""Arm C (PRESPEC_internal §2), stage 1 — MODEL-BLIND build: paired source contexts and exact directions.

Per eps, on the registered gate panel (half_b_with_latents; queries as the registered scoring):
  context A with latent (k, o), rows X in original coordinates:
    sp = params_for(Sigma_k, pi_o);  e = residuals_from_data(to_permuted(X, sp), sp);  z = e / sp.b   (standardised draws)
  order-paired partners  B(o')  = from_permuted(forward_map(z * sp'.b, sp'), sp'),  sp' = params_for(Sigma_k, pi_o')
        for every VISIBLE o' (the parent set of the target x_{d-1} differs from o's; same-class partners have
        v_ord == 0 and are refused);  PRIMARY o' drawn uniformly from the visible set with
        rng = default_rng([OP_SEED_ROOT, round(1000*eps), i]).
  atom-paired partners   B'(k') = from_permuted(forward_map(z * sp''.b, sp''), sp''), sp'' = params_for(Sigma_k', pi_o),
        all k' != k;  PRIMARY k' = argmin_k' | ||v_atom(k')||^2 - ||v_ord(o'_primary)||^2 |  (size-matched);
        the Gbar-matched k' (from the coarse pass) is recorded beside.
  resample: fresh residuals, same (k, o)  (next draw of the same rng);  gaussianised: e_g = b*sqrt(2)*Phi^{-1}(F_mix(e/b))
        in place (paired order removal at the input; F_mix is the exact standardised residual CDF).
  Exact directions at A, per query, in 100-bin log-predictive space (p_true-weighted, centred):
    v_ord(o')  = log pred(p(k|o,A) * p(o|B(o')))  - log q_full(A)
    v_atom(k') = log pred(p(k|o,B'(k')) * p(o|A)) - log q_full(A)
    v_abl      = log q_abl(A) - log q_full(A)        (ablated_weights "order_ablated")
    v_atomabl  = log pred(atom_ablated(A)) - log q_full(A)
    exact swap movements  D_exact(B) = log q_full(B) - log q_full(A)  for every partner (for the leak coefficients).
  Written: patch/build_eps{tag}_{world}.npz  — contexts of every source, p_true, log q_full(A), the direction
  vectors, ||v||^2 per partner, primary indices, provenance (panel, sha256_b, q/op seed roots, coarse sha).
Refuses eps = 0 and any --out inside the registered campaign. Everything digested by G.0 is imported, never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from mech_probe_acts import INT, _env_panel, _world, world_tag  # noqa: E402

OP_SEED_ROOT = 590_000_000


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ residual law helpers
def _al_consts(r: float):
    """standardised AL (b = 1): a = r c, c = sqrt(2 / (1 + r^2)); the variate is Exp(a) - Exp(c) - (a - c)."""
    c = math.sqrt(2.0 / (1.0 + r * r))
    return r * c, c


def F_mix(u: np.ndarray, eps: float, r: float) -> np.ndarray:
    """CDF of the standardised residual law (b = 1): (1 - eps) N(0, 2) + eps AL(r), elementwise."""
    from scipy.stats import norm
    a, c = _al_consts(r)
    x = u + (a - c)                                   # un-shift: X = Exp(a) - Exp(c)
    F_al = np.where(x >= 0, 1.0 - (a / (a + c)) * np.exp(-np.maximum(x, 0) / a), (c / (a + c)) * np.exp(np.minimum(x, 0) / c))
    F_n = norm.cdf(u / math.sqrt(2.0))
    return (1.0 - eps) * F_n + eps * F_al


def gaussianise(e: np.ndarray, b: np.ndarray, eps: float, r: float) -> np.ndarray:
    from scipy.stats import norm
    u = e / b[None, :]
    F = np.clip(F_mix(u, eps, r), 1e-12, 1 - 1e-12)
    return b[None, :] * math.sqrt(2.0) * norm.ppf(F)


def parent_class(perm, target: int):
    """frozenset of variables before `target` in the ordering (the target's parent set)."""
    p = list(perm)
    return frozenset(p[:p.index(target)])


# ------------------------------------------------------------------ worker
_W: dict = {}


def _init(eps):
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    _, world = _world(eps)
    _W["world"] = world
    _W["eps"] = eps


def _log(p):
    return np.log(np.maximum(p, 1e-300))


def _one(job):
    from pfn_dag_verify.corrected_oracle import ablated_weights, exact_joint_posterior, obs_query_operator
    from pfn_dag_verify.corrected_sem import (forward_map, from_permuted, generate_observational, params_for,
                                              residuals_from_data, sample_residuals, to_permuted)
    ci, ctx, rows, k, o, q_seed = job
    world, eps = _W["world"], _W["eps"]
    K, O, d = world.K, world.O, world.d
    target = d - 1; m_q = rows.shape[0]
    rng = np.random.default_rng([OP_SEED_ROOT, int(round(eps * 1000)), ci])
    sp = params_for(world.sigmas[k], world.orderings[o], r=world.spec.r)
    e = residuals_from_data(to_permuted(ctx, sp), sp)
    z = e / sp.b[None, :]
    # ---- visible order partners
    cls_o = parent_class(world.orderings[o], target)
    visible = [op for op in range(O) if op != o and parent_class(world.orderings[op], target) != cls_o]
    o_primary = int(visible[rng.integers(len(visible))])
    B_ord = np.zeros((O, ctx.shape[0], d)); B_ord[:] = np.nan
    for op in visible:
        spp = params_for(world.sigmas[k], world.orderings[op], r=world.spec.r)
        B_ord[op] = from_permuted(forward_map(z * spp.b[None, :], spp), spp)
    # ---- atom partners (all k' != k), same order
    B_atom = np.zeros((K, ctx.shape[0], d)); B_atom[:] = np.nan
    for kp in range(K):
        if kp == k:
            continue
        spq = params_for(world.sigmas[kp], world.orderings[o], r=world.spec.r)
        B_atom[kp] = from_permuted(forward_map(z * spq.b[None, :], spq), spq)
    # ---- resample and gaussianised
    e_res = sample_residuals(rng, sp.b, world.spec, (ctx.shape[0],))
    B_res = from_permuted(forward_map(e_res, sp), sp)
    B_gau = from_permuted(forward_map(gaussianise(e, sp.b, eps, world.spec.r), sp), sp)
    # ---- exact posteriors
    post_A = exact_joint_posterior(world, ctx); WA = post_A["w_lo"].reshape(K, O)
    pkA_given_o = WA / np.maximum(WA.sum(0)[None, :], 1e-300); poA = WA.sum(0)
    w_abl = ablated_weights(world, post_A, "order_ablated"); w_atomabl = ablated_weights(world, post_A, "atom_ablated")
    w_full_A = post_A["w_lo"]
    w_ord = {}; w_full_B = {}
    for op in visible:
        pB = exact_joint_posterior(world, B_ord[op]); WB = pB["w_lo"].reshape(K, O)
        w = pkA_given_o * WB.sum(0)[None, :]; w_ord[op] = w.ravel() / w.sum(); w_full_B[("ord", op)] = pB["w_lo"]
    w_atom = {}
    for kp in range(K):
        if kp == k:
            continue
        pB = exact_joint_posterior(world, B_atom[kp]); WB = pB["w_lo"].reshape(K, O)
        pk_given_o_B = WB / np.maximum(WB.sum(0)[None, :], 1e-300)
        w = pk_given_o_B * poA[None, :]; w_atom[kp] = w.ravel() / w.sum(); w_full_B[("atom", kp)] = pB["w_lo"]
    w_full_res = exact_joint_posterior(world, B_res)["w_lo"]; w_full_gau = exact_joint_posterior(world, B_gau)["w_lo"]
    # ---- per query: p_true, log q_full(A), directions, exact swap movements, ||v||^2
    p_true = np.zeros((m_q, 100)); lqA = np.zeros((m_q, 100))
    v_ord = np.zeros((m_q, O, 100)); v_atom = np.zeros((m_q, K, 100)); v_abl = np.zeros((m_q, 100)); v_atomabl = np.zeros((m_q, 100))
    D_ord = np.zeros((m_q, O, 100)); D_atom = np.zeros((m_q, K, 100)); D_res = np.zeros((m_q, 100)); D_gau = np.zeros((m_q, 100))
    w_true = np.zeros(K * O); w_true[k * O + o] = 1.0
    for q in range(m_q):
        qop = obs_query_operator(world, rows[q, :target], target)
        pt = qop.predictive(w_true); p_true[q] = pt
        lq = _log(qop.predictive(w_full_A)); lqA[q] = lq
        v_abl[q] = _log(qop.predictive(w_abl)) - lq; v_atomabl[q] = _log(qop.predictive(w_atomabl)) - lq
        for op in visible:
            v_ord[q, op] = _log(qop.predictive(w_ord[op])) - lq
            D_ord[q, op] = _log(qop.predictive(w_full_B[("ord", op)])) - lq
        for kp in range(K):
            if kp == k:
                continue
            v_atom[q, kp] = _log(qop.predictive(w_atom[kp])) - lq
            D_atom[q, kp] = _log(qop.predictive(w_full_B[("atom", kp)])) - lq
        D_res[q] = _log(qop.predictive(w_full_res)) - lq; D_gau[q] = _log(qop.predictive(w_full_gau)) - lq

    def wnorm2(v):   # p_true-weighted, centred squared norm summed over queries: (n_partners,)
        vc = v - (p_true[:, None, :] * v).sum(-1, keepdims=True)
        return (p_true[:, None, :] * vc * vc).sum(-1).sum(0)
    n2_ord = wnorm2(v_ord); n2_atom = wnorm2(v_atom)
    n2_ord[[op for op in range(O) if op not in visible]] = np.nan; n2_atom[k] = np.nan
    k_primary = int(np.nanargmin(np.abs(n2_atom - n2_ord[o_primary])))
    return (ci, B_ord, B_atom, B_res, B_gau, np.array(visible), o_primary, k_primary, p_true, lqA,
            v_ord, v_atom, v_abl, v_atomabl, D_ord, D_atom, D_res, D_gau, n2_ord, n2_atom)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--out", default=str(INT / "patch"))
    p.add_argument("--scored", default=str(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000"))
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--coarse", default=str(ROOT / "campaigns/mech_ext_20260902/coarse"))
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--world-seed", type=int, default=None)
    a = p.parse_args()
    if a.eps <= 0:
        raise SystemExit("eps = 0 refused")
    out = Path(a.out)
    if "campaigns/mech_20260827" in str(out.resolve()):
        raise SystemExit("REFUSING: --out inside the registered campaign")
    _env_panel(a)
    M, world = _world(a.eps)
    import mech_predgain as PG
    from mech_interventions import half_b_with_latents
    from pfn_dag_verify.corrected_sem import generate_observational
    from pfn_dag_verify.corrected_verdict import eps_tag
    from pfn_dag_verify.split_panel import contexts_sha256
    tag = eps_tag(a.eps); wt = world_tag(world, a)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"build_eps{tag}_{wt}.npz"
    if dest.is_file():
        raise SystemExit(f"{dest} exists; written once")
    scored = Path(a.scored) / f"predgain_eps{tag}_ck{a.step}.npz"
    z = np.load(scored); names = list(z["names"]); m_q = int(z["m_q"])
    ctxs, ks, os_ = half_b_with_latents(world)
    if list(z["k"]) != ks or list(z["o"]) != os_:
        raise SystemExit("panel latents differ from the scored cell")
    d = world.d
    rows = np.zeros((len(ctxs), m_q, d))
    for i in range(len(ctxs)):
        rng = np.random.default_rng(PG.Q_SEED_ROOT + int(round(a.eps * 1000)) * 1000 + i)
        rows[i] = generate_observational(rng, world.sigmas[ks[i]], world.orderings[os_[i]], world.spec, m_q)
    coarse = Path(a.coarse) / f"coarse_eps{tag}.npz"
    coarse_sha = _sha(coarse) if coarse.is_file() else ""
    t0 = time.time()
    jobs = [(i, np.asarray(ctxs[i], float), rows[i], ks[i], os_[i], PG.Q_SEED_ROOT + int(round(a.eps * 1000)) * 1000 + i) for i in range(len(ctxs))]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(a.eps,)) as pool:
        res = list(pool.map(_one, jobs, chunksize=4))
    res.sort(key=lambda r: r[0])
    n = len(res)
    stack = lambda j, dt=np.float32: np.stack([r[j] for r in res]).astype(dt)
    # identity: S(full) from our p_true/log q_full must equal the scored npz per query
    S_full = (stack(8, np.float64) * stack(9, np.float64)).sum(-1)
    dev = float(np.max(np.abs(S_full - z["S"][:, :, names.index("full")])))
    if dev > 1e-9:
        raise SystemExit(f"REFUSING: S(full) differs from the scored cell by {dev:.2e}")
    visible = np.zeros((n, world.O), bool)
    for r in res:
        visible[r[0], r[5]] = True
    np.savez(dest, ctx_A=np.stack([np.asarray(c, np.float32) for c in ctxs]), rows=rows.astype(np.float32),
             k=np.array(ks), o=np.array(os_), B_ord=stack(1), B_atom=stack(2), B_res=stack(3), B_gau=stack(4),
             visible=visible, o_primary=np.array([r[6] for r in res]), k_primary=np.array([r[7] for r in res]),
             p_true=stack(8), log_q_full_A=stack(9), v_ord=stack(10), v_atom=stack(11), v_abl=stack(12), v_atomabl=stack(13),
             D_ord=stack(14), D_atom=stack(15), D_res=stack(16), D_gau=stack(17), n2_ord=stack(18, np.float64), n2_atom=stack(19, np.float64),
             eps=a.eps, step=a.step, m_q=m_q, K=world.K, O=world.O, d=d, world_seed=a.world_seed if a.world_seed is not None else -1,
             sigmas_sha256=hashlib.sha256(np.ascontiguousarray(world.sigmas, dtype=np.float64).tobytes()).hexdigest(),
             panel_seed=a.panel_seed, split_seed=a.split_seed, n_per_half=a.n_per_half, sha256_b=contexts_sha256(ctxs),
             q_seed_root=PG.Q_SEED_ROOT, op_seed_root=OP_SEED_ROOT, scored_file=str(scored), scored_sha256=_sha(scored),
             coarse_sha256=coarse_sha, identity_dev=dev, prespec_sha256=_sha(INT / "PRESPEC_internal.md"))
    n_vis = visible.sum(1)
    print(f"[patch-build] eps {a.eps} {wt}: {n} contexts, visible order partners {n_vis.min()}-{n_vis.max()} (mean {n_vis.mean():.2f}), "
          f"S(full) identity {dev:.1e}, median ||v_ord||^2 primary {np.nanmedian([r[18][r[6]] for r in res]):.3f} vs matched atom "
          f"{np.nanmedian([r[19][r[7]] for r in res]):.3f} -> {dest.name} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
