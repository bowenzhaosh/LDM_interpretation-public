"""Coarsened-latent oracle families on the registered gate panel (REPORTED, never gated).

Pre-specification: campaigns/mech_ext_20260902/PRESPEC_component_control.md (digested).

For each eps, replay the registered scoring's contexts and queries exactly
(mech_interventions.half_b_with_latents on the gate panel, query rows from
Q_SEED_ROOT + round(eps*1000)*1000 + i, m_q queries), and score every oracle in two
families of coarsened exact posteriors, in the latent layout lo = k*O + o:

  atom family   w_Pi = (A_Pi @ W).ravel()   for every set partition Pi of the K atoms
                (A_Pi[a,b] = 1/|g(a)| if g(a) == g(b) else 0); leaves p(o | D) untouched
  order family  w_H  = (W @ B_H).ravel()    for every set partition H of the O orderings

plus the registered full / order-ablated / prior oracles, which are re-derived and
REQUIRED to equal the registered npz's S arrays (query by query) to 1e-9 — otherwise
the run refuses to write. Output per eps: the per-context query-mean exact log score
S_i(w_X) of every oracle (n_ctx x n_oracle, float64), the partition catalogue, and
the component sizes Gbar^X = mean_i [S_i(full) - S_i(w_X)] (model-blind).

Nothing digested is edited; the locked modules are imported. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def set_partitions(n: int):
    """All set partitions of range(n) as restricted-growth strings (Bell(n) of them)."""
    out = []

    def rec(i, cur, m):
        if i == n:
            out.append(tuple(cur))
            return
        for b in range(m + 1):
            cur.append(b)
            rec(i + 1, cur, max(m, b + 1) if b == m else m)
            cur.pop()

    rec(0, [], 0)
    return out


MAX_ENUM = 5000     # enumerate set partitions only when Bell(n) <= this; otherwise sample merge chains


def sample_partitions(n: int, n_samples: int = 1200, seed: int = 20260904):
    """Set partitions of range(n) sampled by random merge chains (singletons -> one block),
    one partition per merge step, de-duplicated; used when Bell(n) is too large (d=4: O=24).
    Deterministic given seed. Always includes the singleton (full) and one-block partitions."""
    rng = np.random.default_rng(seed)
    seen = {tuple(range(n)), tuple([0] * n)}
    out = [tuple(range(n)), tuple([0] * n)]
    while len(out) < n_samples:
        blocks = [[i] for i in range(n)]
        while len(blocks) > 1:
            a, b = rng.choice(len(blocks), 2, replace=False)
            blocks[a].extend(blocks.pop(b)) if a < b else blocks[b].extend(blocks.pop(a))
            lab = [0] * n
            for j, blk in enumerate(sorted(blocks, key=min)):
                for i in blk:
                    lab[i] = j
            t = tuple(lab)
            if t not in seen:
                seen.add(t); out.append(t)
                if len(out) >= n_samples:
                    break
    return out


def partitions_for(n: int):
    """All set partitions when feasible, else a fixed random sample (recorded in the npz)."""
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    if n < len(bell) and bell[n] <= MAX_ENUM:
        return set_partitions(n), "enumerated"
    return sample_partitions(n), "sampled"


def averaging_matrix(rgs, n):
    A = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if rgs[a] == rgs[b]:
                A[a, b] = 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A


_G: dict = {}


def _patch_world(M) -> None:
    """World variants (EXT cells): MECH_D / MECH_K override the locked module constants;
    MECH_WORLD_SEED routes every make_world call (mech_phase1, mech_interventions and
    this file) through one partial, exactly as scripts/mech_predgain_ext.py does."""
    import functools
    import mech_interventions as MI
    from pfn_dag_verify.corrected_world import make_world
    M.D = int(os.environ.get("MECH_D", M.D))
    M.K = int(os.environ.get("MECH_K", M.K))
    ws = os.environ.get("MECH_WORLD_SEED")
    mw = functools.partial(make_world, seed=int(ws)) if ws else make_world
    M.make_world = mw
    MI.make_world = mw


def _init(eps, m_q, ref_names):
    import threadpoolctl  # noqa
    try:
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    import mech_phase1 as M
    _patch_world(M)
    world = M.make_world(k=M.K, d=M.D, eps=eps)
    K, O = world.K, world.O
    atom_p, _ = partitions_for(K)
    ord_p, _ = partitions_for(O)
    A = np.stack([averaging_matrix(p, K) for p in atom_p])      # (nPi, K, K)
    B = np.stack([averaging_matrix(p, O) for p in ord_p])       # (nH, O, O), symmetric
    _G.update(world=world, A=A, B=B, m_q=m_q, eps=eps, ref_names=ref_names)


def _one(job):
    from pfn_dag_verify.corrected_oracle import ablated_weights, exact_joint_posterior, obs_query_operator
    from pfn_dag_verify.corrected_sem import generate_observational
    ci, ctx, k, o, seed = job
    world, A, B, m_q = _G["world"], _G["A"], _G["B"], _G["m_q"]
    K, O, d = world.K, world.O, world.d
    target = d - 1
    rng = np.random.default_rng(seed)
    rows = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, m_q)
    post = exact_joint_posterior(world, ctx)
    W = post["w_lo"].reshape(K, O)
    w_atom = np.einsum("pab,bo->pao", A, W).reshape(len(A), K * O)      # (nPi, K*O)
    w_ord = np.einsum("ko,hop->hkp", W, B).reshape(len(B), K * O)       # (nH, K*O)
    w_reg = np.stack([ablated_weights(world, post, m) for m in ("full", "order_ablated", "prior")])
    Wall = np.concatenate([w_reg, w_atom, w_ord], axis=0)               # (3 + nPi + nH, K*O)
    w_true = np.zeros(K * O); w_true[k * O + o] = 1.0
    S = np.zeros((m_q, Wall.shape[0]))
    for q in range(m_q):
        qop = obs_query_operator(world, rows[q, :target], target)
        p_true = qop.predictive(w_true)
        num = Wall @ qop.num                                            # (n_or, B)
        den = Wall @ qop.den                                            # (n_or,)
        if np.any(den <= 0) or not np.all(np.isfinite(den)):
            raise FloatingPointError(f"context {ci} query {q}: nonpositive denominator")
        P = num / den[:, None]
        S[q] = (p_true[None, :] * np.log(np.maximum(P, 1e-300))).sum(axis=1)
    return ci, S


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_eps(eps: float, ref_dir: Path, step: int, out: Path, jobs: int, kind: str) -> None:
    import mech_phase1 as M
    _patch_world(M)
    from mech_interventions import half_b_with_latents
    from pfn_dag_verify.corrected_verdict import eps_tag
    from pfn_dag_verify.split_panel import contexts_sha256
    import mech_predgain as PG
    make_world = M.make_world
    tag = eps_tag(eps)
    suffix = "" if kind == "orig" else f"_{kind}"
    ref = (ref_dir / f"predgain_eps{tag}_ck{step}{suffix}.npz").resolve()
    dest = out / f"coarse_eps{tag}{suffix}.npz"
    if dest.is_file():
        raise SystemExit(f"{dest} exists; written once. Delete deliberately and journal.")
    z = np.load(ref)
    for key, want in (("panel_seed", M.PANEL_SEED_EFF), ("split_seed", M.SPLIT_SEED_EFF), ("n_per_half", M.N_PER_HALF_EFF)):
        if int(z[key]) != want:
            raise SystemExit(f"{ref}: {key}={int(z[key])} != {want} (this process's panel)")
    if not bool(z["confirm"]):
        raise SystemExit(f"{ref}: not scored under MECH_CONFIRM=1")
    m_q = int(z["m_q"])
    names = list(z["names"])
    world = make_world(k=M.K, d=M.D, eps=eps)
    ctxs, ks, os_ = half_b_with_latents(world)
    if kind == "n40ext":
        b = np.load(Path(os.environ["MECH_IV_DIR"]) / f"build_n40_eps{tag}.npz")
        assert list(b["k"]) == ks and list(b["o"]) == os_, "latent replay mismatch"
        ctxs = list(b["X_n40ext"])
    if list(z["k"]) != ks or list(z["o"]) != os_:
        raise SystemExit(f"{ref}: latent (k, o) sequence differs from the replayed panel")
    sha_b = contexts_sha256(ctxs)
    sha_match = str(z["sha256_b"]) == sha_b
    if not sha_match:
        # The hash is numpy-version sensitive across venues; the binding identity check
        # is the per-query equality of the registered oracles' S below (1e-9).
        print(f"[eps={eps}] WARNING sha256_b {z['sha256_b']} != replayed {sha_b}; relying on the S check", flush=True)
    K, O = world.K, world.O
    (atom_p, atom_mode), (ord_p, ord_mode) = partitions_for(K), partitions_for(O)
    jobs_list = [(i, ctxs[i], ks[i], os_[i], PG.Q_SEED_ROOT + int(round(eps * 1000)) * 1000 + i) for i in range(len(ctxs))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(eps, m_q, names)) as pool:
        res = list(pool.map(_one, jobs_list, chunksize=8))
    res.sort(key=lambda r: r[0])
    S = np.stack([r[1] for r in res])                                    # (n_ctx, m_q, n_or)
    # ---- the registered oracles must reproduce, query by query
    S_ref = z["S"]
    worst = 0.0
    for j, nm in enumerate(("full", "abl", "prior")):
        dv = float(np.max(np.abs(S[:, :, j] - S_ref[:, :, names.index(nm)])))
        worst = max(worst, dv)
        print(f"[eps={eps}] {nm}: max |S - S_ref| = {dv:.3e}", flush=True)
    if worst > 1e-9:
        raise SystemExit(f"REFUSING to write: registered oracles differ from {ref} by {worst:.3e} > 1e-9")
    S_mean = S.mean(axis=1)                                              # (n_ctx, n_or) query means
    n_reg = 3
    catalogue = ([{"family": "registered", "name": n, "rgs": None, "n_blocks": None} for n in ("full", "abl", "prior")]
                 + [{"family": "atom", "name": f"atom:{''.join(map(str, p))}", "rgs": list(p), "n_blocks": max(p) + 1} for p in atom_p]
                 + [{"family": "order", "name": f"order:{''.join(map(str, p))}", "rgs": list(p), "n_blocks": max(p) + 1} for p in ord_p])
    G_bar = (S_mean[:, [0]] - S_mean).mean(axis=0)                       # (n_or,), 0 for full
    out.mkdir(parents=True, exist_ok=True)
    np.savez(dest, S_mean=S_mean, names=np.array([c["name"] for c in catalogue]),
             family=np.array([c["family"] for c in catalogue]),
             n_blocks=np.array([-1 if c["n_blocks"] is None else c["n_blocks"] for c in catalogue]),
             G_bar=G_bar, eps=eps, step=step, kind=kind, m_q=m_q, n_ctx=len(ctxs),
             K=K, O=O, d=world.d, world_seed=int(os.environ.get("MECH_WORLD_SEED", -1)), n_reg=n_reg,
             atom_partitions=atom_mode, order_partitions=ord_mode, n_atom=len(atom_p), n_order=len(ord_p),
             panel_seed=M.PANEL_SEED_EFF, split_seed=M.SPLIT_SEED_EFF, n_per_half=M.N_PER_HALF_EFF,
             sha256_b=sha_b, sha256_b_ref=str(z["sha256_b"]), sha256_b_match=sha_match, ref_file=str(ref.relative_to(ROOT.resolve())), ref_sha256=_sha(ref),
             check_max_abs_dev=worst, q_seed_root=PG.Q_SEED_ROOT, k=np.array(ks), o=np.array(os_),
             prespec_sha256=_sha(ROOT / "campaigns/mech_ext_20260902/PRESPEC_component_control.md"))
    (out / f"coarse_eps{tag}{suffix}.catalogue.json").write_text(json.dumps(catalogue))
    print(f"[eps={eps}] {len(catalogue)} oracles x {len(ctxs)} contexts -> {dest.name}; "
          f"Gbar_order {G_bar[1]:.5f}; atom family Gbar range [{G_bar[n_reg:n_reg+len(atom_p)].min():.5f}, "
          f"{G_bar[n_reg:n_reg+len(atom_p)].max():.5f}]; order family [{G_bar[n_reg+len(atom_p):].min():.5f}, "
          f"{G_bar[n_reg+len(atom_p):].max():.5f}]  ({time.time() - t0:.0f}s)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    p.add_argument("--ref-dir", type=Path, default=ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000")
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--out", type=Path, default=ROOT / "campaigns/mech_ext_20260902/coarse")
    p.add_argument("--kind", default="orig", choices=["orig", "n40ext"])
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    a = p.parse_args()
    if "campaigns/mech_20260827" in str(a.out.resolve()):
        raise SystemExit("REFUSING: --out inside the registered campaign")
    os.environ["MECH_PANEL_SEED"] = str(a.panel_seed)
    os.environ["MECH_SPLIT_SEED"] = str(a.split_seed)
    os.environ["MECH_N_PER_HALF"] = str(a.n_per_half)
    os.environ.setdefault("MECH_LADDER", "0")
    for e in a.eps:
        run_eps(e, a.ref_dir, a.step, a.out, a.jobs, a.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
