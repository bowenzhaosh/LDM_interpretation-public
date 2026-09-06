"""Arm R (PRESPEC_internal §1), stage 1: activations and exact targets. REPORTED, never gated.

  build  --eps E        the fresh training set for the readout heads: N contexts (k ~ U(K), o ~ U(O),
                        generate_observational, seed PROBE_SEED_ROOT + round(1000*eps)*1000 + train_seed),
                        8 query rows each, and the EXACT soft target q_full(.|D, x_q) = predictive(w_lo)
                        for every (context, query)  ->  probes/train_eps{tag}_ts{train_seed}.npz
  acts   --cell --eps --seed --step
                        the registered gate panel replayed (half_b_with_latents; queries exactly as the
                        registered scoring), the checkpoint {prefix}_s{seed}_ck{step}.pt loaded and
                        sha-verified against the scored npz, and per (context, query): last.out(q) (the
                        head's input: output of transformer.layers[-1] at the query position), the logits,
                        and the model predictive. REFUSES unless S(model) equals the registered model_s
                        column to 1e-4 and max|dp| <= 1e-5 against the stored P.  Also the ck0 twin.
                        -> probes/acts_{cell}_eps{tag}_s{seed}_ck{step}.npz

Sites are architecture-relative: nn.TransformerEncoder has no final norm here, so layers[-1]'s output
at position -1 is exactly what out_head reads (corrected_models.py PFN.forward). Everything digested by
AMENDMENT_G.md G.0 is imported, never edited.
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

PROBE_SEED_ROOT = 580_000_000
INT = ROOT / "campaigns/mech_int_20260905"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _env_panel(a):
    os.environ["MECH_PANEL_SEED"] = str(a.panel_seed)
    os.environ["MECH_SPLIT_SEED"] = str(a.split_seed)
    os.environ["MECH_N_PER_HALF"] = str(a.n_per_half)
    os.environ.setdefault("MECH_LADDER", "0")
    for k in ("MECH_K", "MECH_D", "MECH_WORLD_SEED"):     # never inherit a world from the shell
        os.environ.pop(k, None)
    if a.K is not None:
        os.environ["MECH_K"] = str(a.K)
    if a.d is not None:
        os.environ["MECH_D"] = str(a.d)
    if a.world_seed is not None:
        os.environ["MECH_WORLD_SEED"] = str(a.world_seed)


def world_tag(world, a) -> str:
    """World identity in every filename: K, d and the atom-library seed ('reg' = the registered library)."""
    return f"K{world.K}_d{world.d}_w{a.world_seed if a.world_seed is not None else 'reg'}"


def _world(eps):
    import mech_phase1 as M
    from mech_coarse_oracle import _patch_world
    _patch_world(M)
    return M, M.make_world(k=M.K, d=M.D, eps=eps)


# ------------------------------------------------------------------ exact targets (workers)
_W: dict = {}


def _init_w(eps):
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    _, world = _world(eps)
    _W["world"] = world


def _targets_one(job):
    """Exact q_full for each query of one context; also p_true and S(full) for bookkeeping."""
    from pfn_dag_verify.corrected_oracle import exact_joint_posterior, obs_query_operator
    ci, ctx, rows, k, o = job
    world = _W["world"]
    K, O, d = world.K, world.O, world.d
    target = d - 1
    post = exact_joint_posterior(world, ctx)
    w_full = post["w_lo"]
    w_true = np.zeros(K * O); w_true[k * O + o] = 1.0
    qf = np.zeros((rows.shape[0], 100)); pt = np.zeros_like(qf)
    for q in range(rows.shape[0]):
        qop = obs_query_operator(world, rows[q, :target], target)
        qf[q] = qop.predictive(w_full)
        pt[q] = qop.predictive(w_true)
    return ci, qf, pt, w_full


def build(a) -> None:
    from pfn_dag_verify.corrected_sem import generate_observational
    from pfn_dag_verify.corrected_verdict import eps_tag
    if a.eps <= 0:
        raise SystemExit("eps = 0 refused: the order posterior is uniform there")
    M, world = _world(a.eps)
    tag = eps_tag(a.eps)
    out = INT / "probes"; out.mkdir(parents=True, exist_ok=True)
    dest = out / f"train_eps{tag}_{world_tag(world, a)}_ts{a.train_seed}.npz"
    if dest.is_file():
        raise SystemExit(f"{dest} exists; written once")
    K, O, d = world.K, world.O, world.d
    rng = np.random.default_rng(PROBE_SEED_ROOT + int(round(a.eps * 1000)) * 1000 + a.train_seed)
    ctxs = np.zeros((a.n_train, a.n_rows, d)); rows = np.zeros((a.n_train, a.m_q, d))
    ks = np.zeros(a.n_train, int); os_ = np.zeros(a.n_train, int)
    for i in range(a.n_train):
        k = int(rng.integers(K)); o = int(rng.integers(O))
        ctxs[i] = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, a.n_rows)
        rows[i] = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, a.m_q)
        ks[i], os_[i] = k, o
    t0 = time.time()
    jobs = [(i, ctxs[i], rows[i], int(ks[i]), int(os_[i])) for i in range(a.n_train)]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_w, initargs=(a.eps,)) as pool:
        res = list(pool.map(_targets_one, jobs, chunksize=16))
    res.sort(key=lambda r: r[0])
    q_full = np.stack([r[1] for r in res]).astype(np.float32)
    p_true = np.stack([r[2] for r in res]).astype(np.float32)
    w_lo = np.stack([r[3] for r in res]).astype(np.float32)
    np.savez(dest, ctx=ctxs.astype(np.float32), rows=rows.astype(np.float32), k=ks, o=os_,
             q_full=q_full, p_true=p_true, w_lo=w_lo, eps=a.eps, train_seed=a.train_seed, seed_root=PROBE_SEED_ROOT,
             n_rows=a.n_rows, m_q=a.m_q, K=K, O=O, d=d, world_seed=a.world_seed if a.world_seed is not None else -1,
             sigmas_sha256=hashlib.sha256(np.ascontiguousarray(world.sigmas, dtype=np.float64).tobytes()).hexdigest())
    print(f"[build] eps {a.eps}: {a.n_train} contexts x {a.m_q} queries, exact targets -> {dest.name} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ activations
def load_model(scale_prefix: str, d: int, ck: Path):
    import torch
    from pfn_dag_verify.corrected_models import PFN, ModelConfig
    from pfn_dag_verify.corrected_trackb import SCALES
    cfg = ModelConfig(name=scale_prefix, d=d, **SCALES[scale_prefix.split("_")[0]])
    m = PFN(cfg)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m.eval()
    return m


def forward_sites(m, ctx: np.ndarray, rows: np.ndarray, batch: int = 250):
    """Per (context, query): last.out(q) (n, m_q, d_model), logits (n, m_q, 100), p_model (n, m_q, 100)."""
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    d = ctx.shape[-1]; target = d - 1
    n, m_q = rows.shape[0], rows.shape[1]
    d_model = m.cfg.d_model
    H = np.zeros((n, m_q, d_model), np.float32); L = np.zeros((n, m_q, 100), np.float32)
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = out[:, -1, :].detach()
    handle = m.transformer.layers[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, n, batch):
                c = torch.tensor(ctx[s:s + batch], dtype=torch.float32)
                qxy = torch.tensor(rows[s:s + batch, :, :target], dtype=torch.float32)
                tok = torch.full((c.shape[0],), 2, dtype=torch.long)
                for q in range(m_q):
                    # replicate PFN.forward exactly for one query (corrected_models.py:74-82)
                    ce = m.point_embed(c); te = m.token_embed(tok).unsqueeze(1)
                    qe = m.query_embed(qxy[:, q, :]).unsqueeze(1)
                    out = m.transformer(torch.cat([te, ce, qe], 1))
                    logits = m.out_head(out[:, -1, :])
                    H[s:s + batch, q] = captured["h"].numpy()
                    L[s:s + batch, q] = logits.numpy()
    finally:
        handle.remove()
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    return H, L, P.astype(np.float32)


def acts(a) -> None:
    import mech_predgain as PG
    from mech_interventions import half_b_with_latents
    from pfn_dag_verify.corrected_sem import generate_observational
    from pfn_dag_verify.corrected_verdict import eps_tag
    from pfn_dag_verify.split_panel import contexts_sha256
    if a.eps <= 0:
        raise SystemExit("eps = 0 refused")
    M, world = _world(a.eps)
    tag = eps_tag(a.eps)
    scored = Path(a.scored) / f"predgain_eps{tag}_ck{a.step}.npz"
    z = np.load(scored)
    names = list(z["names"]); m_q = int(z["m_q"])
    if a.exploratory:
        print(f"[acts] EXPLORATORY: {scored} is not a registered cell; panel/confirm/sha asserts relaxed", flush=True)
    else:
        import mech_gates as MG
        seeds_in = [int(x) for x in z["seeds"]]
        MG._check_panel(z, scored, seeds_in, n_rows_expected=int(z["n_rows"]) if "n_rows" in z.files else None,
                        scale_expected=None, panel={"panel_seed": a.panel_seed, "split_seed": a.split_seed, "n_per_half": a.n_per_half})
        if int(z["n_rows"]) != a.n_rows_expected:
            raise SystemExit(f"{scored}: n_rows {int(z['n_rows'])} != {a.n_rows_expected}")
    if f"model_s{a.seed}" not in names:
        raise SystemExit(f"{scored}: no model_s{a.seed} column")
    ckpt_sha = json.loads(str(z["ckpt_sha256"])) if "ckpt_sha256" in z.files else {}
    prefix = str(z["scale"])
    ck = Path(a.nets) / f"eps{tag}" / f"{prefix}_s{a.seed}_ck{a.step}.pt"
    ck0 = Path(a.nets) / f"eps{tag}" / f"{prefix}_s{a.seed}_ck0.pt"
    if not ck.is_file():
        raise SystemExit(f"MISSING checkpoint {ck}")
    if not a.exploratory:
        if _sha(ck) != ckpt_sha.get(str(a.seed)):
            raise SystemExit(f"{ck}: sha256 differs from the registered scoring's ckpt_sha256")
        if not ck0.is_file():
            raise SystemExit(f"MISSING ck0 twin {ck0} (required for the (ii) conjunct; PRESPEC §1)")
    ctxs, ks, os_ = half_b_with_latents(world)
    if list(z["k"]) != ks or list(z["o"]) != os_:
        raise SystemExit("latent replay differs from the scored cell")
    d = world.d; target = d - 1
    rows = np.zeros((len(ctxs), m_q, d))
    for i in range(len(ctxs)):
        rng = np.random.default_rng(PG.Q_SEED_ROOT + int(round(a.eps * 1000)) * 1000 + i)
        rows[i] = generate_observational(rng, world.sigmas[ks[i]], world.orderings[os_[i]], world.spec, m_q)
    ctx_arr = np.stack([np.asarray(c, float) for c in ctxs])
    t0 = time.time()
    m = load_model(prefix, d, ck)
    H, L, P = forward_sites(m, ctx_arr, rows)
    # ---- identity refusals against the registered scoring
    P_ref = z["P"][:, :, names.index(f"model_s{a.seed}"), :]
    dp = float(np.max(np.abs(P.astype(np.float64) - P_ref.astype(np.float64))))
    p_true = z["P"][:, :, names.index("full"), :]  # not p_true; recompute S with the exact p_true below
    # S(model) from stored p_true is not in the npz; re-derive p_true from the oracle for a strict check
    from pfn_dag_verify.corrected_oracle import obs_query_operator
    S_model = np.zeros((len(ctxs), m_q))
    K, O = world.K, world.O
    for i in range(len(ctxs)):
        w_true = np.zeros(K * O); w_true[ks[i] * O + os_[i]] = 1.0
        for q in range(m_q):
            qop = obs_query_operator(world, rows[i, q, :target], target)
            pt = qop.predictive(w_true)
            S_model[i, q] = float(np.sum(pt * np.log(np.maximum(P[i, q].astype(np.float64), 1e-300))))
    S_ref = z["S"][:, :, names.index(f"model_s{a.seed}")]
    dS = float(np.max(np.abs(S_model - S_ref)))
    if "sha256_b" in z.files and str(z["sha256_b"]) != contexts_sha256(ctxs):
        print("[acts] WARNING sha256_b differs from the scored npz (numpy-version sensitive); the S/P identity below is the binding check", flush=True)
    print(f"[acts] {a.cell} eps {a.eps} s{a.seed} ck{a.step}: max|dp| {dp:.2e}, max|dS| {dS:.2e} ({time.time() - t0:.0f}s)", flush=True)
    if dp > 1e-5 or dS > 1e-4:
        raise SystemExit(f"REFUSING: identity with the registered scoring fails (dp {dp:.2e}, dS {dS:.2e})")
    H0 = L0 = P0 = None
    if ck0.is_file():
        m0 = load_model(prefix, d, ck0)
        H0, L0, P0 = forward_sites(m0, ctx_arr, rows)
    out = INT / "probes"; out.mkdir(parents=True, exist_ok=True)
    dest = out / f"acts_{a.cell}_eps{tag}_{world_tag(world, a)}_s{a.seed}_ck{a.step}{'_EXPLORATORY' if a.exploratory else ''}.npz"
    if dest.is_file():
        raise SystemExit(f"{dest} exists; written once")
    np.savez(dest, last_out=H, logits=L, p_model=P, S_model=S_model,
             last_out_ck0=H0 if H0 is not None else np.zeros(0, np.float32),
             logits_ck0=L0 if L0 is not None else np.zeros(0, np.float32),
             p_ck0=P0 if P0 is not None else np.zeros(0, np.float32),
             k=np.array(ks), o=np.array(os_), sha256_b=contexts_sha256(ctxs), sha256_b_ref=str(z["sha256_b"]) if "sha256_b" in z.files else "",
             exploratory=a.exploratory, rows=rows.astype(np.float32),
             cell=a.cell, eps=a.eps, seed=a.seed, step=a.step, scale=prefix, ckpt=str(ck), ckpt_sha256=_sha(ck),
             ck0=str(ck0) if ck0.is_file() else "", ck0_sha256=_sha(ck0) if ck0.is_file() else "",
             scored_file=str(scored), scored_sha256=_sha(scored), identity_dp=dp, identity_dS=dS,
             panel_seed=a.panel_seed, split_seed=a.split_seed, n_per_half=a.n_per_half, m_q=m_q,
             d_model=H.shape[-1], K=world.K, O=world.O, d=d,
             world_seed=a.world_seed if a.world_seed is not None else -1,
             prespec_sha256=_sha(INT / "PRESPEC_internal.md"), torch=__import__("torch").__version__, numpy=np.__version__)
    print(f"[acts] -> {dest.name}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "acts"])
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=25_000)
    p.add_argument("--n-rows", type=int, default=20)
    p.add_argument("--m-q", type=int, default=8)
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--cell", default="base_lr0.001_d500000")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--scored", default=str(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000"))
    p.add_argument("--nets", default=str(ROOT / "campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets"))
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--world-seed", type=int, default=None)
    p.add_argument("--n-rows-expected", type=int, default=20)
    p.add_argument("--exploratory", action="store_true", help="non-registered nets/npz: relax the registered-only asserts, label the output")
    a = p.parse_args()
    _env_panel(a)
    if a.stage == "build":
        build(a)
    else:
        acts(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
