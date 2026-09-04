"""Mechanism phase 1c — causal input interventions on the half-B panel. DIAGNOSTIC.

For each half-B context we know the generating (k, o) (replayed from the panel
RNG), so we can recover the residuals e = x_pi U^T exactly and replace them:

  resample : fresh draw from the TRAINING law, same (k, o)      (control: natural
             context-to-context variability of every metric below)
  gauss    : Gaussian residuals, same b  -> variance 2b^2 unchanged, all odd
             cumulants and excess kurtosis removed
  symlap   : symmetric Laplace (r = 1), same b -> variance unchanged, skew
             removed, kurtosis kept: the ordering evidence that survives lives
             in EVEN-order statistics only
  flip     : e -> -e, deterministic: every even moment kept, every odd moment
             sign-flipped

All three residual laws share Var = 2 b^2 by construction (AL: a^2 + c^2 =
c^2 (r^2 + 1) = 2 b^2), so the second-order picture of a context is matched
across interventions in expectation.

Target for every intervened context is the exact posterior UNDER THE TRAINING
WORLD (the model's own prior), which is what a posterior-matching model should
output whatever the data's true law. Metrics (summarise):
  fidelity   JS(w_model', w_exact')            — does the model still track?
  response   JS(w_exact', w_exact) vs JS(w_model', w_model)
  to-uniform JS(w', uniform) for exact and model
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
from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED  # noqa: E402
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS  # noqa: E402
from pfn_dag_verify.corrected_models import bin_y  # noqa: E402
from pfn_dag_verify.corrected_sem import (  # noqa: E402
    ResidualSpec, forward_map, from_permuted, generate_observational, params_for,
    residuals_from_data, sample_residuals, to_permuted,
)
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402
from pfn_dag_verify.corrected_world import make_world  # noqa: E402
from pfn_dag_verify.split_panel import SPLIT_SEED  # noqa: E402

OUT = Path(os.environ.get("MECH_IV_DIR", str(ROOT / "campaigns/mech_20260827/interventions")))
INTERVENTIONS = ("resample", "gauss", "symlap", "flip")
IV_SEED_ROOT = 550_000_000


def half_b_with_latents(world):
    """Replay make_eval_panel's RNG verbatim, recording (k, o); then split as
    build_split_panel does. Asserts bitwise equality with the registered path."""
    rng = np.random.default_rng(M.PANEL_SEED_EFF)
    total = 2 * M.N_PER_HALF_EFF
    ctxs, ks, os_ = [], [], []
    for _ in range(total):
        k = int(rng.integers(world.K))
        o = int(rng.integers(world.O))
        ctx = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                     world.spec, M.N_ROWS)
        for _q in range(N_QUERY_PER_CONTEXT):
            target = int(rng.integers(world.d))
            qrow = generate_observational(rng, world.sigmas[k], world.orderings[o],
                                          world.spec, 1)[0]
            bin_y(qrow[target])
        ctxs.append(ctx); ks.append(k); os_.append(o)
    perm = np.random.default_rng(M.SPLIT_SEED_EFF).permutation(total)
    ib = sorted(int(i) for i in perm[M.N_PER_HALF_EFF:])
    ref, _ = M.half_b(world)
    got = [ctxs[i] for i in ib]
    assert all(np.array_equal(a, b) for a, b in zip(ref, got)), "panel replay drifted"
    return got, [ks[i] for i in ib], [os_[i] for i in ib]


def intervene(world, ctx, k, o, kind, rng):
    sp = params_for(world.sigmas[k], world.orderings[o], r=world.spec.r)
    e = residuals_from_data(to_permuted(ctx, sp), sp)
    n = e.shape[0]
    if kind == "resample":
        e2 = sample_residuals(rng, sp.b, world.spec, (n,))
    elif kind == "gauss":
        e2 = sample_residuals(rng, sp.b, ResidualSpec(eps=0.0, r=world.spec.r), (n,))
    elif kind == "symlap":
        e2 = sample_residuals(rng, sp.b, ResidualSpec(eps=1.0, r=1.0), (n,))
    elif kind == "flip":
        e2 = -e
    else:
        raise ValueError(kind)
    return from_permuted(forward_map(e2, sp), sp)


def stage_build(eps: float) -> None:
    tag = eps_tag(eps)
    world = make_world(k=M.K, d=M.D, eps=eps)
    ctxs, ks, os_ = half_b_with_latents(world)
    OUT.mkdir(parents=True, exist_ok=True)
    X0 = np.stack([np.asarray(c, float) for c in ctxs])
    doc = {"eps": eps, "k": np.array(ks), "o": np.array(os_), "X_orig": X0,
           "w_exact_orig": M.exact_posterior_batch(world, X0)}
    for kind in INTERVENTIONS:
        rng = np.random.default_rng(IV_SEED_ROOT + int(round(eps * 1000)) + INTERVENTIONS.index(kind))
        X = np.stack([intervene(world, c, k, o, kind, rng) for c, k, o in zip(ctxs, ks, os_)])
        doc[f"X_{kind}"] = X
        doc[f"w_exact_{kind}"] = M.exact_posterior_batch(world, X)
        u = np.full_like(doc[f"w_exact_{kind}"], 1.0 / world.O)
        print(f"[eps={eps}] {kind}: exact JS to orig {M.js(doc[f'w_exact_{kind}'], doc['w_exact_orig']).mean():.4f}, "
              f"exact JS to uniform {M.js(doc[f'w_exact_{kind}'], u).mean():.4f}", flush=True)
    np.savez(OUT / f"build_eps{tag}.npz", **doc)


N40_SEED_ROOT = 560_000_000


def stage_build_n40(eps: float, extra_rows: int = 20) -> None:
    """Amendment G G.2/G3: the n_rows-40 panel is the n_rows-20 panel with each
    context EXTENDED by `extra_rows` fresh rows from its own generating latent,
    so the n20/n40 contrast is paired by context (and, through mech_predgain's
    per-context query seed, by query row as well)."""
    tag = eps_tag(eps)
    world = make_world(k=M.K, d=M.D, eps=eps)
    ctxs, ks, os_ = half_b_with_latents(world)
    rng = np.random.default_rng(N40_SEED_ROOT + int(round(eps * 1000)))
    X40 = []
    for c, k, o in zip(ctxs, ks, os_):
        extra = generate_observational(rng, world.sigmas[k], world.orderings[o], world.spec, extra_rows)
        X40.append(np.concatenate([np.asarray(c, float), extra], axis=0))
    X40 = np.stack(X40)
    assert X40.shape[1] == M.N_ROWS + extra_rows
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MECH_CONFIRM", "0") == "1" and (OUT / f"build_n40_eps{tag}.npz").is_file():
        raise SystemExit(f"{OUT / f'build_n40_eps{tag}.npz'} exists; under MECH_CONFIRM=1 it is built once.")
    np.savez(OUT / f"build_n40_eps{tag}.npz", eps=eps, k=np.array(ks), o=np.array(os_),
             X_n40ext=X40, X_orig=np.stack([np.asarray(c, float) for c in ctxs]),
             panel_seed=M.PANEL_SEED_EFF, split_seed=M.SPLIT_SEED_EFF, n_per_half=M.N_PER_HALF_EFF)
    print(f"[eps={eps}] n40ext panel: {X40.shape} -> build_n40_eps{tag}.npz", flush=True)


def stage_project(eps: float, seeds: list[int], step: int, kinds: list[str], jobs: int) -> None:
    tag = eps_tag(eps)
    b = np.load(OUT / f"build_eps{tag}.npz")
    world = make_world(k=M.K, d=M.D, eps=eps)
    for kind in kinds:
        X = b[f"X_{kind}"]
        for s in seeds:
            dest = OUT / f"proj_eps{tag}_{kind}_s{s}_ck{step}.npz"
            if dest.is_file():
                continue
            ck = M.NETS / f"eps{tag}" / f"{M.SCALE}_s{s}_ck{step}.pt"
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=jobs, initializer=M._init,
                                     initargs=(eps, str(ck))) as pool:
                rows = list(pool.map(M._one, list(enumerate(list(X))), chunksize=4))
            rows.sort(key=lambda r: r[0])
            w = np.full((len(X), world.O), np.nan); res = np.full(len(X), np.nan)
            for ci, wo, r, _ in rows:
                if wo is not None:
                    w[ci], res[ci] = wo, r
            nfail = int(np.isnan(w[:, 0]).sum())
            np.savez(dest, w_proj=w, residual=res, n_fail=nfail)
            print(f"[eps={eps}] {kind} s{s}: JS to exact' {np.nanmean(M.js(b[f'w_exact_{kind}'], np.nan_to_num(w, nan=1/world.O))):.4f}, "
                  f"{nfail} fails, {time.time() - t0:.0f}s", flush=True)


def stage_summarise(step: int) -> None:
    rows = []
    for eps in M.EPS_GRID:
        tag = eps_tag(eps)
        f = OUT / f"build_eps{tag}.npz"
        if not f.is_file():
            continue
        b = np.load(f)
        O = b["w_exact_orig"].shape[1]
        u = np.full((len(b["X_orig"]), O), 1.0 / O)
        base = sorted(M.OUT.glob(f"proj_eps{tag}_s*_ck{step}.npz"))
        if not base:
            continue
        Wb = [np.load(p)["w_proj"] for p in base]
        row = {"eps": eps, "orig": {
            "fidelity_model": float(np.mean([M.js(w, b["w_exact_orig"]).mean() for w in Wb])),
            "exact_to_uniform": float(M.js(b["w_exact_orig"], u).mean()),
            "model_to_uniform": float(np.mean([M.js(w, u).mean() for w in Wb]))}}
        for kind in INTERVENTIONS:
            projs = sorted(OUT.glob(f"proj_eps{tag}_{kind}_s*_ck{step}.npz"))
            if not projs:
                continue
            W = [np.load(p)["w_proj"] for p in projs]
            we = b[f"w_exact_{kind}"]
            ok = np.all([np.isfinite(w).all(1) for w in W + Wb], axis=0)
            resp_exact = M.js(we[ok], b["w_exact_orig"][ok])
            resp_model = np.mean([M.js(w[ok], wb[ok]) for w, wb in zip(W, Wb)], axis=0)
            row[kind] = {
                "n_ok": int(ok.sum()),
                "fidelity_model": float(np.mean([M.js(w[ok], we[ok]).mean() for w in W])),
                "exact_to_uniform": float(M.js(we[ok], u[ok]).mean()),
                "model_to_uniform": float(np.mean([M.js(w[ok], u[ok]).mean() for w in W])),
                "response_exact": float(resp_exact.mean()),
                "response_model": float(resp_model.mean()),
                "corr_response": float(np.corrcoef(resp_exact, resp_model)[0, 1]),
            }
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    (OUT / f"summary_ck{step}.json").write_text(json.dumps(rows, indent=2, default=float))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "build_n40", "project", "summarise"])
    p.add_argument("--eps", type=float, nargs="*", default=[e for e in M.EPS_GRID if e > 0])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--step", type=int, default=100_000)
    p.add_argument("--kinds", nargs="*", default=list(INTERVENTIONS))
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = p.parse_args()
    # Amendment G G.2, same guard as mech_phase1.py: under MECH_CONFIRM=1 the
    # registered panel must be given EXPLICITLY. build_n40 defines the G3 pairing,
    # so a silent fallback to the exploration panel would produce an n40 panel that
    # does not extend the n20 one and a paired bootstrap that is not paired.
    if os.environ.get("MECH_CONFIRM", "0") == "1":
        missing = [v for v in ("MECH_PANEL_SEED", "MECH_SPLIT_SEED", "MECH_N_PER_HALF",
                               "MECH_IV_DIR") if v not in os.environ]
        if missing:
            raise SystemExit("MECH_CONFIRM=1 requires the registered panel explicitly; "
                             f"missing: {' '.join(missing)}")
    if a.stage == "build":
        for e in a.eps:
            stage_build(e)
    elif a.stage == "build_n40":
        for e in a.eps:
            stage_build_n40(e)
    elif a.stage == "project":
        for e in a.eps:
            stage_project(e, a.seeds, a.step, a.kinds, a.jobs)
    else:
        stage_summarise(a.step)


if __name__ == "__main__":
    main()
