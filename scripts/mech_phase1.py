"""Mechanism phase 1 — DIAGNOSTIC, nothing registered is read for a number.

Question: which restricted oracle does the PFN's LP-projected order posterior
agree with, per eps, on the half-B panel (500 contexts in the exploration;
Amendment G registers 1000 fresh contexts via MECH_N_PER_HALF)?

Ladder (all in the F.2c readout's units, r = -JS(w_exact_o, w_o)):
  uniform   : any reader limited to <=2nd-order statistics. EXACT — the
              order-invariance identity says 2nd-order stats carry no order
              information at any eps (they are the Gaussian sufficient stats).
  F3 / F4   : p(o | raw sample moments of order <=3 / <=4), fitted by gradient
              boosting on fresh draws from the same world (hard labels).
  Fres      : p(o | per-ordering OLS-residual moments: skew, kurt, mean/sd) —
              the "LiNGAM implementation" feature set (a rational function of
              moments <=4, but far easier to fit).
  exact     : the ceiling, r = 0.
  model     : the PFN, per seed, w_o = LP projection of its Q2 predictive.

Stages
  oracles   : panel + exact w_o + uniform + fitted ladders  -> eps{tag}.npz
  project   : model / dose-0 projections per checkpoint     -> proj_*.npz
  summarise : the table                                       -> summary.json

Usage
  PYTHONPATH=src .venv/bin/python scripts/mech_phase1.py oracles --eps 1.0
  PYTHONPATH=src .venv/bin/python scripts/mech_phase1.py project --eps 1.0 --steps 100000
  PYTHONPATH=src .venv/bin/python scripts/mech_phase1.py summarise
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED  # noqa: E402
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS  # noqa: E402
from pfn_dag_verify.corrected_models import gen_classifier_batch  # noqa: E402
from pfn_dag_verify.corrected_oracle import exact_joint_posterior  # noqa: E402
from pfn_dag_verify.corrected_sem import residual_logpdf  # noqa: E402
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402
from pfn_dag_verify.corrected_world import make_world  # noqa: E402
from pfn_dag_verify.split_panel import HALF_B, SPLIT_SEED, build_split_panel, contexts_sha256  # noqa: E402

D, K = 3, 8
EPS_GRID = (0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0)
OUT = ROOT / "campaigns/mech_20260827/phase1"
FID = ROOT / "campaigns/corrected_20260812/raw/postF/fidelity"
# MECH_NETS overrides the checkpoint root (e.g. the dose-extension nets); the
# default is W2's registered fleet.
NETS = Path(os.environ.get("MECH_NETS", str(ROOT / "campaigns/corrected_20260812/raw/postF/w2/nets")))
QUERY_PANEL_SEED = 990_300_001      # f2c_fidelity_pass.QUERY_PANEL_SEED
PANEL = "Q2"
SOLVER = "highs"
# MECH_SCALE / MECH_N_ROWS: the contrast fleets (scripts/mech_train.py) vary the
# architecture and the context length; the registered defaults are base / 20.
SCALE = os.environ.get("MECH_SCALE", "base")
N_ROWS = int(os.environ.get("MECH_N_ROWS", HEADLINE_N_ROWS))
# Amendment G: the confirmatory panel is a FRESH draw. MECH_PANEL_SEED /
# MECH_SPLIT_SEED override the exploration's seeds (770000000 / 880000000);
# MECH_PHASE1_OUT redirects this file's outputs so fresh-panel files never
# overwrite the exploratory ones.
PANEL_SEED_EFF = int(os.environ.get("MECH_PANEL_SEED", PANEL_SEED))
SPLIT_SEED_EFF = int(os.environ.get("MECH_SPLIT_SEED", SPLIT_SEED))
N_PER_HALF_EFF = int(os.environ.get("MECH_N_PER_HALF", N_PER_HALF))   # G.2 registers 1000
OUT = Path(os.environ.get("MECH_PHASE1_OUT", str(OUT)))
TRAIN_SEED_ROOT = 660_000_000       # fresh namespace for the oracle fits


# --------------------------------------------------------------------------- #
# exact posterior, vectorised over contexts (cross-checked vs the oracle)
# --------------------------------------------------------------------------- #
def exact_posterior_batch(world, X: np.ndarray) -> np.ndarray:
    """w_o (N, O) for X (N, n, d) float64, exactly as exact_joint_posterior."""
    c = world.params_cache()
    Kk, O = world.K, world.O
    N = X.shape[0]
    ll = np.empty((N, Kk * O), dtype=np.float64)
    for k in range(Kk):
        for o in range(O):
            perm = c["perms"][o]
            e = X[:, :, perm] @ c["U"][k, o].T
            ll[:, k * O + o] = residual_logpdf(e, c["b"][k, o], world.spec).sum(axis=(1, 2))
    logp = ll + np.log(world.prior_lo())[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    w = np.exp(logp)
    w /= w.sum(axis=1, keepdims=True)
    return w.reshape(N, Kk, O).sum(axis=1)


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise JS, natural log, 1e-300 clamp — corrected_tomography's."""
    p = np.maximum(p, 1e-300)
    q = np.maximum(q, 1e-300)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


# --------------------------------------------------------------------------- #
# feature ladders
# --------------------------------------------------------------------------- #
def monomial_feats(X: np.ndarray, max_order: int) -> np.ndarray:
    N, n, d = X.shape
    cols = [X.mean(1)]
    idx = [(i,) for i in range(d)]
    for order in range(2, max_order + 1):
        idx = [t + (j,) for t in idx for j in range(t[-1], d)]
        prod = np.ones((N, n))
        block = []
        for t in idx:
            prod = np.ones((N, n))
            for i in t:
                prod = prod * X[:, :, i]
            block.append(prod.mean(1))
        cols.append(np.stack(block, 1))
    return np.concatenate(cols, 1)


def resid_feats(X: np.ndarray, orderings) -> np.ndarray:
    N, n, d = X.shape
    cols = []
    for perm in orderings:
        Xp = X[:, :, list(perm)]
        for j in range(d):
            y = Xp[:, :, j]
            if j == 0:
                r = y
            else:
                A = Xp[:, :, :j]
                AtA = np.einsum("nij,nik->njk", A, A)
                Aty = np.einsum("nij,ni->nj", A, y)
                beta = np.linalg.solve(AtA, Aty[..., None])[..., 0]
                r = y - np.einsum("nij,nj->ni", A, beta)
            m1, m2 = r.mean(1), (r ** 2).mean(1)
            m3, m4 = (r ** 3).mean(1), (r ** 4).mean(1)
            sd = np.sqrt(np.maximum(m2, 1e-12))
            cols += [m1 / sd, m3 / sd ** 3, m4 / sd ** 4, np.abs(r).mean(1) / sd]
    return np.stack(cols, 1)


FEATURE_SETS = {
    "F2": lambda X, w: monomial_feats(X, 2),
    "F3": lambda X, w: monomial_feats(X, 3),
    "F4": lambda X, w: monomial_feats(X, 4),
    "Fres": lambda X, w: resid_feats(X, w.orderings),
    "F4+res": lambda X, w: np.concatenate([monomial_feats(X, 4), resid_feats(X, w.orderings)], 1),
}


def fit_restricted(name, world, Xtr, otr, Xva, ova, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    f = FEATURE_SETS[name]
    Ftr, Fva = f(Xtr, world), f(Xva, world)
    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=seed)
    t0 = time.time()
    clf.fit(Ftr, otr)
    pva = clf.predict_proba(Fva)
    return clf, {"n_feat": int(Ftr.shape[1]), "iters": int(clf.n_iter_),
                 "val_logloss": float(log_loss(ova, pva, labels=list(range(world.O)))),
                 "fit_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# stage: oracles
# --------------------------------------------------------------------------- #
def half_b(world):
    split = build_split_panel(world, N_PER_HALF_EFF, N_ROWS, PANEL_SEED_EFF,
                              split_seed=SPLIT_SEED_EFF, n_query_per_context=N_QUERY_PER_CONTEXT)
    ctxs = list(split.half(HALF_B))
    return ctxs, split.sha256_b


def stage_oracles(eps: float, n_train: int, sets: list[str]) -> None:
    tag = eps_tag(eps)
    if os.environ.get("MECH_CONFIRM", "0") == "1" and (OUT / f"eps{tag}.npz").is_file():
        raise SystemExit(f"{OUT / f'eps{tag}.npz'} exists; under MECH_CONFIRM=1 the panel file is written once.")
    world = make_world(k=K, d=D, eps=eps)
    ctxs, sha_b = half_b(world)
    X = np.stack([np.asarray(c, dtype=np.float64) for c in ctxs])          # (500, 20, 3)
    fresh = "MECH_PANEL_SEED" in os.environ
    if fresh:
        # Amendment G: the exploration's F.2c artifact is NOT read on a fresh panel.
        fid = {"half_sha256": None, "model_ctx": [], "floor_ctx": [], "ceiling_ctx": []}
        sha_match = False
    else:
        fid = json.loads((FID / f"fidelity_eps{tag}.json").read_text())
        sha_match = (fid["half_sha256"] == sha_b)
    print(f"[eps={eps}] half-B sha matches fidelity artifact: {sha_match} (fresh panel: {fresh})", flush=True)

    w_exact = exact_posterior_batch(world, X)
    # cross-check against the registered oracle on 5 contexts
    for i in (0, 1, 2, 250, 499):
        ref = exact_joint_posterior(world, X[i])["w_o"]
        assert np.allclose(ref, w_exact[i], atol=1e-12), (i, ref, w_exact[i])
    uniform = np.full_like(w_exact, 1.0 / world.O)

    out = {"eps": eps, "sha_b_matches_fidelity": bool(sha_match),
           "panel_seed": PANEL_SEED_EFF, "split_seed": SPLIT_SEED_EFF, "n_per_half": N_PER_HALF_EFF,
           "n_rows": N_ROWS, "sha256_b": sha_b, "fresh_panel": fresh,
           # G.2 provenance: mech_gates refuses a readout not built under MECH_CONFIRM=1.
           # stage_oracles loads no checkpoint, so the "checkpoint digests" clause of
           # G.2 is vacuous for this artifact family and is deliberately not faked.
           "confirm": os.environ.get("MECH_CONFIRM", "0") == "1",
           "w_exact": w_exact, "w_uniform": uniform,
           "r_model_artifact": np.array(fid["model_ctx"], float),
           "r_floor_artifact": np.array(fid["floor_ctx"], float),
           "r_ceiling_artifact": np.array(fid["ceiling_ctx"], float),
           "X": X}
    meta = {"eps": eps, "n_train": n_train, "sets": {}}

    if eps > 0 and sets:
        rng = np.random.default_rng(TRAIN_SEED_ROOT + int(round(eps * 1000)))
        Xtr, _, otr = gen_classifier_batch(world, rng, n_train, HEADLINE_N_ROWS)
        Xva, _, ova = gen_classifier_batch(world, rng, 20_000, HEADLINE_N_ROWS)
        Xtr, Xva = Xtr.astype(np.float64), Xva.astype(np.float64)
        # the exact predictor's log-loss on the same validation labels: the floor
        # no restricted oracle can beat; the gap to it is information lost + fit error
        w_va = exact_posterior_batch(world, Xva)
        meta["val_logloss_exact"] = float(-np.log(np.maximum(w_va[np.arange(len(ova)), ova], 1e-300)).mean())
        meta["val_logloss_uniform"] = float(math.log(world.O))
        for name in sets:
            clf, info = fit_restricted(name, world, Xtr, otr, Xva, ova, seed=0)
            w_hat = clf.predict_proba(FEATURE_SETS[name](X, world))
            out[f"w_{name}"] = w_hat
            info["r_mean_halfB"] = float(-js(w_exact, w_hat).mean())
            meta["sets"][name] = info
            print(f"[eps={eps}] {name}: {info}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"eps{tag}.npz", **out)
    (OUT / f"eps{tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[eps={eps}] uniform r_mean {-js(w_exact, uniform).mean():.4f}; "
          f"model r_mean (artifact) {out['r_model_artifact'].mean():.4f}; "
          f"floor {out['r_floor_artifact'].mean():.4f}", flush=True)


# --------------------------------------------------------------------------- #
# stage: project (model / dose-0 checkpoints through the registered readout)
# --------------------------------------------------------------------------- #
_G: dict = {}


def _init(eps: float, ckpt: str) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    import torch
    from pfn_dag_verify.corrected_identifiability import build_panel_operator, build_query_panels
    from pfn_dag_verify.corrected_models import PFN, ModelConfig
    from pfn_dag_verify.corrected_trackb import SCALES
    torch.set_num_threads(1)
    world = make_world(k=K, d=D, eps=eps)
    panels = build_query_panels(world, seed=QUERY_PANEL_SEED)
    # MECH_SCALE may carry an LR tag from mech_train.py (e.g. "large_lr0.0003"):
    # the architecture is the part before "_lr", the full string is the file prefix.
    cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE.split("_")[0]])   # "base_n40", "large_lr0.0003" -> arch
    model = PFN(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    _G.update(world=world, panel=panels[PANEL],
              op=build_panel_operator(world, panels[PANEL]), model=model)


def _one(job):
    from pfn_dag_verify.corrected_tomography import _lp_project_or_ranges, pfn_panel_predictions
    ci, ctx = job
    preds = pfn_panel_predictions(_G["model"], _G["world"], _G["panel"], [ctx], "cpu")[0]
    try:
        proj = _lp_project_or_ranges(_G["world"], _G["op"], preds, "project", 0.0, solver=SOLVER)
    except Exception as exc:
        return ci, None, None, str(exc)[:200]
    return ci, proj["w_o"], float(proj["residual"]), None


def stage_project(eps: float, steps: list[int], seeds: list[int], jobs: int) -> None:
    tag = eps_tag(eps)
    world = make_world(k=K, d=D, eps=eps)
    ctxs, _ = half_b(world)
    ex = np.load(OUT / f"eps{tag}.npz")
    w_exact = ex["w_exact"]
    for s in seeds:
        for st in steps:
            ck = NETS / f"eps{tag}" / f"{SCALE}_s{s}_ck{st}.pt"
            dest = OUT / f"proj_eps{tag}_s{s}_ck{st}.npz"
            confirm = os.environ.get("MECH_CONFIRM", "0") == "1"
            if dest.is_file():
                if confirm:
                    raise SystemExit(f"{dest} exists; under MECH_CONFIRM=1 nothing is re-projected silently.")
                continue
            if not ck.is_file():
                raise SystemExit(f"MISSING checkpoint {ck}")
            ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                                     initargs=(eps, str(ck))) as pool:
                rows = list(pool.map(_one, list(enumerate(ctxs)), chunksize=4))
            rows.sort(key=lambda r: r[0])
            fails = [r for r in rows if r[1] is None]
            w_proj = np.full((len(ctxs), world.O), np.nan)
            resid = np.full(len(ctxs), np.nan)
            for ci, w, res, _ in rows:
                if w is not None:
                    w_proj[ci], resid[ci] = w, res
            r = -js(w_exact, np.nan_to_num(w_proj, nan=1.0 / world.O))
            np.savez(dest, w_proj=w_proj, residual=resid, r=r, n_fail=len(fails),
                     fail_msgs=np.array([f[3] for f in fails][:20]),
                     panel_seed=PANEL_SEED_EFF, split_seed=SPLIT_SEED_EFF, n_per_half=N_PER_HALF_EFF,
                     n_rows=N_ROWS, sha256_b=ex["sha256_b"] if "sha256_b" in ex.files else "",
                     seed=s, step=st, scale=SCALE, solver=SOLVER, panel=PANEL, query_panel_seed=QUERY_PANEL_SEED,
                     # G.2 provenance: which checkpoint, from which fleet, under which mode.
                     # mech_gates.g4_cell matches ckpt_sha256 against the registered fleet's
                     # predgain artifact, so G4 cannot be run on models outside that fleet.
                     nets=str(NETS), ckpt=str(ck), ckpt_sha256=ck_sha, confirm=confirm)
            print(f"[eps={eps}] s{s} ck{st}: r_mean {np.nanmean(r):.4f}, "
                  f"{len(fails)} LP failures, {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# stage: summarise
# --------------------------------------------------------------------------- #
def stage_summarise() -> None:
    rows = []
    for eps in EPS_GRID:
        tag = eps_tag(eps)
        f = OUT / f"eps{tag}.npz"
        if not f.is_file():
            continue
        ex = np.load(f)
        w_exact = ex["w_exact"]
        floor = float(ex["r_floor_artifact"].mean())
        gain = 0.0 - floor
        row = {"eps": eps, "floor": floor, "gain": gain,
               "sha_b_matches_fidelity": bool(ex["sha_b_matches_fidelity"])}

        def y_of(r_ctx):
            return float((np.mean(r_ctx) - floor) / gain) if gain > 1e-6 else None

        ladders = {"uniform": ex["w_uniform"]}
        for name in FEATURE_SETS:
            if f"w_{name}" in ex.files:
                ladders[name] = ex[f"w_{name}"]
        row["y_ladder"] = {k: y_of(-js(w_exact, w)) for k, w in ladders.items()}
        row["y_model_artifact"] = y_of(ex["r_model_artifact"])
        # model projections at 100k, per seed, and which oracle they sit nearest
        projs = sorted(OUT.glob(f"proj_eps{tag}_s*_ck100000.npz"))
        if projs:
            W = [np.load(p)["w_proj"] for p in projs]
            ok = np.all([np.isfinite(w).all(1) for w in W], axis=0)
            r_model = np.mean([-js(w_exact[ok], w[ok]) for w in W], axis=0)
            row["y_model_reproj"] = y_of(r_model)
            row["n_ok"] = int(ok.sum())
            agree = {}
            for k, w in ladders.items():
                agree[k] = float(np.mean([js(wm[ok], w[ok]).mean() for wm in W]))
            agree["exact"] = float(np.mean([js(wm[ok], w_exact[ok]).mean() for wm in W]))
            row["js_model_to"] = agree
            # per-context: does the model's error track the ladder's error?
            err_model = np.mean([js(w_exact[ok], wm[ok]) for wm in W], axis=0)
            row["corr_err_model_vs_ladder_err"] = {
                k: float(np.corrcoef(err_model, js(w_exact[ok], w[ok]))[0, 1])
                for k, w in ladders.items()}
        rows.append(row)
        print(json.dumps(row, indent=None, default=float), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, default=float))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["oracles", "project", "summarise"])
    p.add_argument("--eps", type=float, nargs="*", default=list(EPS_GRID))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--sets", nargs="*", default=["F2", "F3", "F4", "Fres", "F4+res"])
    p.add_argument("--steps", type=int, nargs="*", default=[100_000])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = p.parse_args()
    # Amendment G G.2: under MECH_CONFIRM=1 the registered panel must be given
    # EXPLICITLY. Without this guard a G4 invocation that omits the panel env
    # silently builds the EXPLORATION panel (770000000 / 880000000 / 500) and
    # exits 0 -- the quiet failure, not the loud one.
    if os.environ.get("MECH_CONFIRM", "0") == "1":
        missing = [v for v in ("MECH_PANEL_SEED", "MECH_SPLIT_SEED", "MECH_N_PER_HALF")
                   if v not in os.environ]
        if missing:
            raise SystemExit("MECH_CONFIRM=1 requires the registered panel explicitly; "
                             f"missing: {' '.join(missing)}")
        if "MECH_PHASE1_OUT" not in os.environ:
            raise SystemExit("MECH_CONFIRM=1 requires MECH_PHASE1_OUT (confirmatory outputs "
                             "must not land in the exploratory directory)")
    if a.stage == "oracles":
        for e in a.eps:
            stage_oracles(e, a.n_train, a.sets)
    elif a.stage == "project":
        for e in a.eps:
            stage_project(e, a.steps, a.seeds, a.jobs)
    else:
        stage_summarise()


if __name__ == "__main__":
    main()
