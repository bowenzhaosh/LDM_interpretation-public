"""Amendment F.2c-tol rule (4) — the pre-lock floor readout calibration.

Rule (4): "Before the lock, one Mac measurement is taken and persisted: the
dose-0 projection residual and the floor readout r at every surviving eps cell,
one untrained seed, the fleet's panel size."

WHY THIS IS A NEW SCRIPT AND NOT A CALL INTO corrected_tomography
-----------------------------------------------------------------
Rule (2) forbids identified-set LPs on the floor and ceiling curves.
`tomography_context` runs one unconditionally (`tol_eff = max(tol, residual*1.2)`
then mode "id"), and `run_tomography` always calls it. `src/pfn_dag_verify/**` is
write-protected because editing the verifier IS editing the gate. So the
projection branch is called directly here and the structural statistic is
recomputed verbatim from corrected_tomography.py:213-220.

WHAT IS UNDERDETERMINED, AND HOW THIS SCRIPT HANDLES IT
-------------------------------------------------------
The signed text does not pin four things (see .claude/CLAUSE4_SPEC.md):
  U1 which query panel supplies r      -> SIGNED Q1 (Bo, 2026-08-26); Q2 measured
                                          alongside as a robustness companion
  U2 whether the ceiling is computed   -> --curves floor,ceiling (superset; the
                                          text is silent and measuring both is cheap)
  U3 the solver setting                -> SIGNED "highs", the dual simplex (Bo)
  U4 the seed identity                 -> 0, the fleet's first and the only seed
                                          with a persisted dose-0 anywhere
A superset costs one run and lets any later signature be consumed without a
re-measurement. It does NOT make the choice: the lock text still must pin them.

  U5 the VENUE. Rule (4) says "one Mac measurement". The Mac refuses SSH, so Bo
  signed the substitution to WashU on 2026-08-26 -- the other environment F.4's
  invariance table already characterizes, so no new environment enters the record.
  --status still defaults to DIAGNOSTIC: anywhere but WashU, this is a smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from pfn_dag_verify.artifact_status import DIAGNOSTIC, LIVE, stamp
from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED
from pfn_dag_verify.corrected_identifiability import (
    build_panel_operator, build_query_panels,
)
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS
from pfn_dag_verify.corrected_models import ModelConfig, PFN, train_pfn
from pfn_dag_verify.corrected_oracle import exact_joint_posterior
from pfn_dag_verify.corrected_tomography import (
    _lp_project_or_ranges, pfn_panel_predictions,
)
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT, SCALES
from pfn_dag_verify.corrected_verdict import eps_tag
from pfn_dag_verify.corrected_world import make_world
from pfn_dag_verify.split_panel import HALF_B, SPLIT_SEED, build_split_panel

ESTIMAND = "f2c_floor_readout_calibration"
CALIB_MIN_GAIN = 1e-6          # mirrors corrected_verdict.CALIB_MIN_GAIN
SURVIVING_EPS = (0.1, 0.25, 0.35, 0.5, 0.75, 1.0)   # eps=0 exits by construction
QUERY_PANEL_SEED = 990_300_001                       # run_tomography's seed + 1
SCALE = "base"
D, K = 3, 8

_G: dict[str, Any] = {}


def _structural_js(w_exact: np.ndarray, w_proj: np.ndarray) -> float:
    """Verbatim from corrected_tomography.py:213-220. Natural log, 1e-300 clamp."""
    p_e = np.maximum(w_exact, 1e-300)
    p_p = np.maximum(w_proj, 1e-300)
    m = 0.5 * (p_e + p_p)
    return float(0.5 * np.sum(p_e * np.log(p_e / m))
                 + 0.5 * np.sum(p_p * np.log(p_p / m)))


def _strip_to_pipeline(preds: dict) -> dict:
    """The ceiling gets the SAME instrument as the floor and the model, no more.

    The LP scales rows by ``t.get("den", 1.0)``, so leaving the exact predictive's
    `den` in would hand the ceiling a normalizer the other two curves never see.
    Interventional predictions are None because the PFN has no interventional
    channel (corrected_tomography.py:63-69), so the ceiling must not have one
    either or the readout pipeline is not identical.
    """
    return {"obs": [{"pred": np.asarray(t["pred"])} for t in preds["obs"]],
            "int": [None] * len(preds.get("int", []))}


def _init(eps: float, seed: int, ckpt: str, panels_wanted: tuple) -> None:
    # One BLAS thread per worker. Unpinned, N workers x 16 threads oversubscribes
    # the box and the measured cost went from 0.1 s/LP to minutes; the LPs and the
    # operator build are the matrix-heavy parts and they thrash.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    torch.set_num_threads(1)
    world = make_world(k=K, d=D, eps=eps)
    panels = build_query_panels(world, seed=QUERY_PANEL_SEED)
    cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE])
    model = PFN(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    _G.update(world=world, panels=panels,
              ops={p: build_panel_operator(world, panels[p])
                   for p in panels_wanted},
              model=model)


def _one(job: tuple) -> dict:
    ci, ctx, pname, solver, curve = job
    world, op, panel = _G["world"], _G["ops"][pname], _G["panels"][pname]
    exact = exact_joint_posterior(world, ctx)
    if curve == "floor":
        preds = pfn_panel_predictions(_G["model"], world, panel, [ctx], "cpu")[0]
    else:
        preds = _strip_to_pipeline(op.exact_predictive(exact["w_lo"]))
    t0 = time.time()
    # An LP that will not solve is a FINDING about the readout, not an exception to
    # swallow and not a value to substitute. It is recorded per context with its
    # message and excluded from the summary, so a panel whose readout is only
    # partly computable reports that fact instead of averaging over the gaps.
    # No per-context solver fallback: rule (1) requires ONE solver setting across
    # the curves, so a fallback would silently break the comparison it exists to make.
    try:
        proj = _lp_project_or_ranges(world, op, preds, "project", 0.0, solver=solver)
    except Exception as exc:
        return {"context_index": ci, "panel": pname, "solver": solver, "curve": curve,
                "lp_failed": True, "lp_error": str(exc)[:400],
                "projection_residual": None, "structural_js": None, "r": None,
                "nominal_tol": None, "tol_eff": None, "id_lp_ran": False,
                "lp_wallclock_s": round(time.time() - t0, 4)}
    js = _structural_js(exact["w_o"], proj["w_o"])
    return {"context_index": ci, "panel": pname, "solver": solver, "curve": curve,
            "lp_failed": False,
            "projection_residual": float(proj["residual"]),
            "structural_js": js, "r": float(0.0 - js),
            "nominal_tol": None, "tol_eff": None, "id_lp_ran": False,
            "lp_wallclock_s": round(time.time() - t0, 4)}


def _summary(rows: list[dict]) -> dict:
    ok = [x for x in rows if not x.get("lp_failed")]
    failed = [x for x in rows if x.get("lp_failed")]
    if not ok:
        return {"n_contexts": len(rows), "n_solved": 0,
                "n_lp_failed": len(failed),
                "failed_context_indices": [x["context_index"] for x in failed],
                "solve_rate": 0.0, "r_mean": None}
    r = np.array([x["r"] for x in ok], float)
    res = np.array([x["projection_residual"] for x in ok], float)
    n = len(r)
    return {"n_contexts": len(rows), "n_solved": n, "n_lp_failed": len(failed),
            "failed_context_indices": [x["context_index"] for x in failed][:50],
            "solve_rate": round(n / max(len(rows), 1), 6),
            "r_mean": float(r.mean()), "r_sd": float(r.std(ddof=1)) if n > 1 else 0.0,
            "r_se": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "r_min": float(r.min()), "r_max": float(r.max()),
            "residual_mean": float(res.mean()), "residual_sd": float(res.std(ddof=1)) if n > 1 else 0.0,
            "residual_min": float(res.min()), "residual_max": float(res.max())}


def persist_dose0(eps: float, seed: int, netdir: Path) -> Path:
    """Write the step-0 state through the fleet's own persistence path.

    steps=0 passes the reachability check and the capture fires before the (empty)
    loop, which is exactly the Gate-A property F.2c's floor depends on.
    """
    netdir.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE])
    train_pfn(make_world(k=K, d=D, eps=eps), steps=0, seed=seed, cfg=cfg,
              n_ctx=HEADLINE_N_ROWS, n_query=7, ckpt_steps=(0,),
              outdir=netdir, tag_prefix=f"{SCALE}_s{seed}")
    ck = netdir / f"{SCALE}_s{seed}_ck0.pt"
    if not ck.is_file():
        raise SystemExit(f"dose-0 was requested and is absent at {ck}. Step 0 "
                         "used to be a SILENT no-op; that is why this is checked "
                         "here rather than discovered when the floor curve is built.")
    return ck


def identity_checks(ck: Path, seed: int, others: list[Path]) -> dict:
    """Gate-A's identity property, plus the cross-eps invariance the spec predicts."""
    cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE])
    torch.manual_seed(1000 * seed + 7)
    fresh = PFN(cfg).state_dict()
    got = torch.load(ck, map_location="cpu")
    same_fresh = all(torch.equal(fresh[k], got[k]) for k in fresh)
    same_cross = True
    for o in others:
        st = torch.load(o, map_location="cpu")
        if not all(torch.equal(st[k], got[k]) for k in got):
            same_cross = False
    return {"identity_check_fresh_seed": bool(same_fresh),
            "identity_check_cross_eps": bool(same_cross)}


def run_eps(eps: float, a) -> dict:
    t0 = time.time()
    world = make_world(k=K, d=D, eps=eps)
    split = build_split_panel(world, a.n_per_half, HEADLINE_N_ROWS, PANEL_SEED,
                              split_seed=SPLIT_SEED,
                              n_query_per_context=N_QUERY_PER_CONTEXT)
    prov = split.provenance(HALF_B)
    ctxs = split.half(HALF_B) if hasattr(split, "half") else None
    if ctxs is None:
        raise SystemExit("split panel exposes no half(); adapt before measuring")
    ctxs = list(ctxs)[: a.limit] if a.limit else list(ctxs)

    tag = eps_tag(eps)
    netdir = Path(a.outdir) / "nets" / f"eps{tag}"
    ck = persist_dose0(eps, a.seed, netdir)

    jobs = [(ci, c, p, s, cu)
            for ci, c in enumerate(ctxs)
            for p in a.panels for s in a.solvers for cu in a.curves]
    print(f"[eps={eps}] {len(ctxs)} contexts x {len(a.panels)} panels x "
          f"{len(a.solvers)} solvers x {len(a.curves)} curves = {len(jobs)} LPs",
          flush=True)

    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init,
                             initargs=(eps, a.seed, str(ck),
                                       tuple(a.panels))) as ex:
        rows = list(ex.map(_one, jobs, chunksize=4))

    curves: dict[str, Any] = {}
    for cu in a.curves:
        curves[cu] = {}
        for p in a.panels:
            for s in a.solvers:
                sel = [r for r in rows if r["curve"] == cu and r["panel"] == p
                       and r["solver"] == s]
                curves[cu].setdefault(p, {})[s] = {
                    "per_context": sorted(sel, key=lambda r: r["context_index"]),
                    "summary": _summary(sel)}

    gain = {}
    if "ceiling" in a.curves and "floor" in a.curves:
        for p in a.panels:
            for s in a.solvers:
                cm = curves["ceiling"][p][s]["summary"]["r_mean"]
                fm = curves["floor"][p][s]["summary"]["r_mean"]
                if cm is None or fm is None:
                    gain.setdefault(p, {})[s] = {"ceiling_minus_floor": None,
                                                 "calib_min_gain": CALIB_MIN_GAIN,
                                                 "below_min_gain": None,
                                                 "reason": "a curve had no solvable context"}
                    continue
                g = cm - fm
                gain.setdefault(p, {})[s] = {
                    "ceiling_minus_floor": g, "calib_min_gain": CALIB_MIN_GAIN,
                    "below_min_gain": bool(g < CALIB_MIN_GAIN)}

    doc: dict[str, Any] = {
        "amendment": "F.2c-tol rule (4)",
        "world": {"d": D, "K": K, "O": world.O, "eps": eps,
                  "spec_r": getattr(world.spec, "r", None)},
        "panel": prov, "n_contexts_measured": len(ctxs),
        "query_panel_seed": QUERY_PANEL_SEED,
        "solvers": list(a.solvers), "panels": list(a.panels), "curves": list(a.curves),
        "readout_panel_signed": a.readout_panel,
        "signatures": {"U1_readout_panel": a.readout_panel, "U3_solver": a.solvers[0],
                       "U4_seed": a.seed, "U5_venue": a.venue},
        "dose0": {"seed": a.seed, "checkpoint": str(ck.relative_to(a.outdir)),
                  "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
                  **identity_checks(ck, a.seed, a.prior_ck)},
        "curves_by_name": curves,
        "gain": gain,
        "underdetermined_not_decided_here": ["U1 panel", "U2 ceiling", "U3 solver",
                                             "U4 seed", "U5 venue"],
        "environment": {"host": socket.gethostname(), "platform": platform.platform(),
                        "python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__},
        "wallclock_s": round(time.time() - t0, 2),
    }
    stamp(doc, ESTIMAND, LIVE if a.status == "live" else DIAGNOSTIC)
    doc["artifact"]["n_rows"] = int(HEADLINE_N_ROWS)
    doc["artifact"]["half"] = HALF_B
    doc["artifact"]["n_per_half"] = int(a.n_per_half)
    out = Path(a.outdir) / f"floor_calib_eps{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=float))
    a.prior_ck.append(ck)
    print(f"[eps={eps}] wrote {out} in {doc['wallclock_s']}s", flush=True)
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", type=float, nargs="*", default=list(SURVIVING_EPS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panels", nargs="*", default=["Q1", "Q2"],
                   help="Q3 is deliberately absent: the PFN has no interventional "
                        "channel, so Q3 collapses to Q2 for the floor and the model "
                        "(measured: identical projection residual 0.353582) while the "
                        "ceiling WOULD get its 18 interventional rows. Including it "
                        "hands one curve an instrument the others cannot have.")
    p.add_argument("--solvers", nargs="*", default=["highs"])
    p.add_argument("--curves", nargs="*", default=["floor", "ceiling"])
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--limit", type=int, default=0,
                   help="measure only the first N half-B contexts (smoke only)")
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--readout-panel", default="Q1",
                   help="U1, signed by Bo 2026-08-26: Q1, the headline panel")
    p.add_argument("--venue", default="washu",
                   help="U5, signed by Bo 2026-08-26: WashU substitutes for the Mac")
    p.add_argument("--status", choices=["diagnostic", "live"], default="diagnostic",
                   help="LIVE only once the venue (U5) and U1/U3/U4 are signed")
    p.add_argument("--outdir", type=Path,
                   default=Path("campaigns/corrected_20260812/raw/postF/floor_calib"))
    a = p.parse_args(argv)
    a.prior_ck = []
    if a.status == "live" and a.limit:
        raise SystemExit("REFUSING: a --limit run is a smoke, not the measurement.")
    for e in a.eps:
        run_eps(e, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
