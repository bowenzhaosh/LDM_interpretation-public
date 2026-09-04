"""Mechanism phase 1d — projection-free order-information readout. DIAGNOSTIC.

The PFN is a predictor of x_{d-1} given the rest (in-task target = d-1, the only
coordinate it was trained on). Its native currency is the log score of that
predictive. So measure order use in that currency, with exact expectations:

  For half-B context i with generating latent (k, o), draw M fresh query rows
  from (k, o); for each query the TRUE bin law p_true = predictive(one-hot(k,o))
  is exact, and S(p) = sum_b p_true(b) log p(b) is the exact expected log score.

  Oracles on the same query (all exact, no LP):
    full   : w_lo = p(k,o | D)                         the ceiling
    abl    : w_lo = p_order(o) p(k | o, D)              order information removed
    prior  : w_lo = 1/(K O)                             everything removed
    Fk     : w_lo = w_hat_o(k-th ladder) x p(k | o, D)  restricted-order oracles
                                                         (phase 1's fitted ladder)
  Estimands, per eps, averaged over contexts x queries:
    G_order(p)  = S(p) - S(abl)        gain from order information
    eta_order   = G_order(model) / G_order(full)
    G_atom      = S(abl) - S(prior)    gain from atom (covariance) information
    eta_total   = (S(model) - S(prior)) / (S(full) - S(prior))
    regret_E3   = KL(full || model)    the registered E.3 formula, for the record

Nothing here reads the LP readout; a disagreement between this eta and F.2c's y
is a statement about the readout, not about the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mech_phase1 as M  # noqa: E402
from mech_interventions import half_b_with_latents  # noqa: E402
from pfn_dag_verify.corrected_oracle import ablated_weights, exact_joint_posterior, obs_query_operator  # noqa: E402
from pfn_dag_verify.corrected_sem import generate_observational  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402
from pfn_dag_verify.corrected_world import make_world  # noqa: E402

# MECH_OUT overrides the output dir (pair it with MECH_NETS for a different fleet).
OUT = Path(os.environ.get("MECH_OUT", str(ROOT / "campaigns/mech_20260827/predgain")))
Q_SEED_ROOT = 440_000_000
LADDER = ("F2", "F3", "F4", "Fres", "F4+res")
_G: dict = {}


def _init(eps: float, ckpts: list[str], ladder: dict, seeds=(0, 1, 2)) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    import torch
    from pfn_dag_verify.corrected_models import PFN, ModelConfig
    from pfn_dag_verify.corrected_trackb import SCALES
    torch.set_num_threads(1)
    world = make_world(k=M.K, d=M.D, eps=eps)
    cfg = ModelConfig(name=M.SCALE, d=M.D, **SCALES[M.SCALE.split("_")[0]])
    models = []
    for ck in ckpts:
        m = PFN(cfg)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        models.append(m)
    _G.update(world=world, models=models, ladder=ladder, model_seeds=list(seeds))


def _S(p_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(p_true * np.log(np.maximum(p, 1e-300))))


def _one(job):
    import torch
    ci, ctx, k, o, m_q, seed = job
    world, models, ladder = _G["world"], _G["models"], _G["ladder"]
    model_seeds = _G["model_seeds"]
    K, O, d = world.K, world.O, world.d
    target = d - 1
    rng = np.random.default_rng(seed)
    rows = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, m_q)
    post = exact_joint_posterior(world, ctx)
    w_full = post["w_lo"]
    w_abl = ablated_weights(world, post, "order_ablated")
    w_prior = ablated_weights(world, post, "prior")
    w_true = np.zeros(K * O); w_true[k * O + o] = 1.0
    w_ladder = {}
    for name, w_hat_o in ladder.items():
        w = post["w_k_given_o"] * w_hat_o[ci][None, :]
        w_ladder[name] = w.ravel() / w.sum()
    ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
    qxy = torch.tensor(rows[:, :target], dtype=torch.float32).unsqueeze(0)
    tok = torch.full((1,), 2, dtype=torch.long)
    p_models = [m.predict_bin_probs(ctx_t, qxy, tok)[0] for m in models]      # (m_q, B) each
    names = ["full", "abl", "prior"] + list(w_ladder) + [f"model_s{s}" for s in model_seeds]   # by ACTUAL seed
    S = np.zeros((m_q, len(names)))
    kl_e3 = np.zeros((m_q, len(models)))
    P = np.zeros((m_q, len(names), p_models[0].shape[-1]), dtype=np.float32)
    for q in range(m_q):
        qop = obs_query_operator(world, rows[q, :target], target)
        p_true = qop.predictive(w_true)
        preds = [qop.predictive(w_full), qop.predictive(w_abl), qop.predictive(w_prior)]
        preds += [qop.predictive(w_ladder[n]) for n in w_ladder]
        preds += [pm[q] for pm in p_models]
        for j, p in enumerate(preds):
            S[q, j] = _S(p_true, p)
            P[q, j] = p
        pf = preds[0]
        for s, pm in enumerate(p_models):
            kl_e3[q, s] = float(np.sum(pf * (np.log(np.maximum(pf, 1e-300)) - np.log(np.maximum(pm[q], 1e-300)))))
    return ci, names, S, kl_e3, P


def _dest(eps: float, step: int, kind: str) -> Path:
    tag = eps_tag(eps)
    return OUT / (f"predgain_eps{tag}_ck{step}.npz" if kind == "orig"
                  else f"predgain_eps{tag}_ck{step}_{kind}.npz")


def run_eps(eps: float, step: int, seeds: list[int], m_q: int, jobs: int, kind: str = "orig") -> None:
    tag = eps_tag(eps)
    dest = _dest(eps, step, kind)
    confirm = os.environ.get("MECH_CONFIRM", "0") == "1"
    if dest.is_file():
        if confirm:
            raise SystemExit(f"{dest} exists. Under MECH_CONFIRM=1 a registered cell is scored "
                             "exactly once; delete deliberately and journal, never overwrite or skip.")
        print(f"[eps={eps}] {dest.name} exists, skip", flush=True); return
    world = make_world(k=M.K, d=M.D, eps=eps)
    ctxs, ks, os_ = half_b_with_latents(world)
    from pfn_dag_verify.split_panel import contexts_sha256
    sha_b = contexts_sha256(ctxs)
    if kind == "n40ext":
        # Amendment G G3: the n20 contexts extended by 20 rows from the same latent
        # (mech_interventions.py build_n40); same (k, o), same query seed -> paired.
        b = np.load(Path(os.environ.get("MECH_IV_DIR", str(ROOT / "campaigns/mech_20260827/interventions")))
                    / f"build_n40_eps{tag}.npz")
        assert list(b["k"]) == ks and list(b["o"]) == os_, "latent replay mismatch"
        ctxs = list(b["X_n40ext"])
    elif kind != "orig":
        # intervened contexts from mech_interventions' build; queries still come
        # from the ORIGINAL (k, o) law with the same seed -> identical query rows
        b = np.load(Path(os.environ.get("MECH_IV_DIR", str(ROOT / "campaigns/mech_20260827/interventions")))
                    / f"build_eps{tag}.npz")
        assert list(b["k"]) == ks and list(b["o"]) == os_, "latent replay mismatch"
        ctxs = list(b[f"X_{kind}"])
    # The fitted ladders are indexed by CONTEXT on the exploration's half-B panel;
    # on any other panel (MECH_LADDER=0, a fresh MECH_PANEL_SEED, or an intervened
    # context set) they must not be used.
    ph1_path = M.OUT / f"eps{tag}.npz"
    use_ladder = (kind == "orig" and os.environ.get("MECH_LADDER", "1") != "0"
                  and "MECH_PANEL_SEED" not in os.environ and ph1_path.is_file())
    ladder = {}
    if use_ladder:
        ph1 = np.load(ph1_path)
        ladder = {n: ph1[f"w_{n}"] for n in LADDER if f"w_{n}" in ph1.files}
    ckpts = [str(M.NETS / f"eps{tag}" / f"{M.SCALE}_s{s}_ck{step}.pt") for s in seeds]
    for c in ckpts:
        if not Path(c).is_file():
            raise SystemExit(f"MISSING checkpoint {c}")
    import hashlib
    ckpt_sha = {str(s): hashlib.sha256(Path(c).read_bytes()).hexdigest() for s, c in zip(seeds, ckpts)}
    jobs_list = [(i, ctxs[i], ks[i], os_[i], m_q, Q_SEED_ROOT + int(round(eps * 1000)) * 1000 + i)
                 for i in range(len(ctxs))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                             initargs=(eps, ckpts, ladder, list(seeds))) as pool:
        res = list(pool.map(_one, jobs_list, chunksize=4))
    res.sort(key=lambda r: r[0])
    names = res[0][1]
    S = np.stack([r[2] for r in res])          # (n_ctx, m_q, n_pred)
    kl = np.stack([r[3] for r in res])         # (n_ctx, m_q, n_seed)
    P = np.stack([r[4] for r in res])          # (n_ctx, m_q, n_pred, B) float32
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(dest, names=np.array(names), S=S, kl_e3=kl, P=P, k=np.array(ks), o=np.array(os_),
             step=step, seeds=np.array(seeds), m_q=m_q, kind=kind,
             # panel identity (Amendment G G.2): verifiable, never assumed
             panel_seed=M.PANEL_SEED_EFF, split_seed=M.SPLIT_SEED_EFF, n_per_half=M.N_PER_HALF_EFF,
             n_rows=int(np.asarray(ctxs[0]).shape[0]), sha256_b=sha_b, scale=M.SCALE,
             nets=str(M.NETS), ckpt_sha256=json.dumps(ckpt_sha), confirm=confirm)
    summ = summarise_one(eps, step, names, S, kl)
    print(f"[eps={eps}] ck{step} {kind}: {json.dumps(summ, default=float)}  ({time.time() - t0:.0f}s)", flush=True)


def _kl(p, q):
    p = np.maximum(p.astype(np.float64), 1e-300); q = np.maximum(q.astype(np.float64), 1e-300)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def stage_responses(step: int) -> None:
    """Intervention responses in the predictive currency (needs orig + kind files)."""
    rows = []
    for eps in M.EPS_GRID:
        f0 = _dest(eps, step, "orig")
        if not f0.is_file():
            continue
        z0 = np.load(f0)
        n0 = list(z0["names"]); P0 = z0["P"]
        i_full = n0.index("full"); i_models = [i for i, n in enumerate(n0) if n.startswith("model_s")]
        row = {"eps": eps, "step": step, "orig": {
            "fidelity_kl_full_model": float(np.mean([_kl(P0[:, :, i_full], P0[:, :, i]).mean() for i in i_models]))}}
        for kind in ("resample", "gauss", "symlap", "flip"):
            f = _dest(eps, step, kind)
            if not f.is_file():
                continue
            z = np.load(f); n1 = list(z["names"]); P1 = z["P"]
            j_full = n1.index("full"); j_models = [i for i, n in enumerate(n1) if n.startswith("model_s")]
            resp_exact = _kl(P0[:, :, i_full], P1[:, :, j_full])                      # (n_ctx, m_q)
            resp_model = np.mean([_kl(P0[:, :, i], P1[:, :, j]) for i, j in zip(i_models, j_models)], axis=0)
            fid = np.mean([_kl(P1[:, :, j_full], P1[:, :, j]).mean() for j in j_models])
            summ = summarise_one(eps, step, n1, z["S"], z["kl_e3"])
            row[kind] = {"response_exact": float(resp_exact.mean()), "response_model": float(resp_model.mean()),
                         "corr_response_ctx": float(np.corrcoef(resp_exact.mean(1), resp_model.mean(1))[0, 1]),
                         "fidelity_kl_full_model": float(fid),
                         "G_order_full": summ["G_order_full"], "eta_order_model": summ["eta_order"]["model"]}
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    (OUT / f"responses_ck{step}.json").write_text(json.dumps(rows, indent=2, default=float))


def summarise_one(eps, step, names, S, kl) -> dict:
    names = list(names)
    Sm = S.mean(axis=1)                        # per-context mean over queries (n_ctx, n_pred)
    n = Sm.shape[0]
    col = {nm: Sm[:, i] for i, nm in enumerate(names)}
    model_cols = [nm for nm in names if nm.startswith("model_s")]
    model = np.mean([col[nm] for nm in model_cols], axis=0)
    g_full = col["full"] - col["abl"]
    g_atom = col["abl"] - col["prior"]
    out = {"eps": eps, "step": step, "n_ctx": n,
           "G_order_full": float(g_full.mean()), "G_order_full_se": float(g_full.std(ddof=1) / np.sqrt(n)),
           "G_atom": float(g_atom.mean()),
           "eta_order": {}, "eta_total": {}}
    def eta(x, y):
        return float(x.mean() / y.mean())
    out["eta_order"]["model"] = eta(model - col["abl"], g_full)
    out["eta_order"]["model_se_ctx"] = float(np.std((model - col["abl"]) / g_full.mean(), ddof=1) / np.sqrt(n))
    for nm in model_cols:
        out["eta_order"][nm] = eta(col[nm] - col["abl"], g_full)
    for nm in names:
        if nm in LADDER:
            out["eta_order"][nm] = eta(col[nm] - col["abl"], g_full)
    tot = col["full"] - col["prior"]
    out["eta_total"]["model"] = eta(model - col["prior"], tot)
    out["eta_total"]["abl_only(atom)"] = eta(col["abl"] - col["prior"], tot)
    out["regret_E3_mean"] = float(kl.mean())
    out["regret_true_expect"] = float((col["full"] - model).mean())
    out["S"] = {nm: float(col[nm].mean()) for nm in names}
    out["S"]["model_mean"] = float(model.mean())
    return out


def stage_summarise(step: int) -> None:
    rows = []
    for eps in M.EPS_GRID:
        f = OUT / f"predgain_eps{eps_tag(eps)}_ck{step}.npz"
        if not f.is_file():
            continue
        z = np.load(f)
        rows.append(summarise_one(eps, step, z["names"], z["S"], z["kl_e3"]))
    (OUT / f"summary_ck{step}.json").write_text(json.dumps(rows, indent=2, default=float))
    for r in rows:
        print(f"eps={r['eps']}: G_order_full={r['G_order_full']:.4f}  eta_order model={r['eta_order']['model']:.3f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in r['eta_order'].items() if k in LADDER)
              + f"  | eta_total model={r['eta_total']['model']:.3f} atom-only={r['eta_total']['abl_only(atom)']:.3f}"
              + f"  | E3 regret={r['regret_E3_mean']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["run", "summarise", "responses"])
    p.add_argument("--eps", type=float, nargs="*", default=list(M.EPS_GRID))
    p.add_argument("--step", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--m-q", type=int, default=8)
    p.add_argument("--kind", default="orig")
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = p.parse_args()
    if a.stage == "run":
        for e in a.eps:
            run_eps(e, a.step, a.seeds, a.m_q, a.jobs, a.kind)
    elif a.stage == "responses":
        stage_responses(a.step)
    else:
        stage_summarise(a.step)


if __name__ == "__main__":
    main()
