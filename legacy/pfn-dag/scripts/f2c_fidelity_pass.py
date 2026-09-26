"""Amendment F.2c / F.2c-tol rule (1) — the three-curve fidelity pass.

Extends ``scripts/f2c_clause4_floor.py`` with the MODEL curve and emits the
artifact ``corrected_verdict.cells_from_raw`` consumes. **Nothing here computes
y**: ``Cell.fidelity_norm`` is the only implementation (F.2d.6), and writing a
``y`` field would create the second path that docstring exists to prevent.

WHY THIS IS A NEW SCRIPT
------------------------
``f2c_clause4_floor.py`` is the producer of a LIVE pre-lock artifact and is not
modified in place. ``src/pfn_dag_verify/**`` is write-protected because editing
the verifier IS editing the gate, and every F.0 digest must keep verifying, so
the model curve is added here rather than in ``corrected_tomography``. Rule (2)
forbids identified-set LPs on the floor and ceiling, and ``tomography_context``
runs one unconditionally — which is why the projection branch is called directly
and ``structural_js`` is recomputed verbatim from corrected_tomography.py:213-220.

THE POST-LOCK CONSTRUCTION DECISIONS THIS IMPLEMENTS (.claude/FIDELITY_DECISIONS.md)
-----------------------------------------------------------------------------------
  A  the trained PFN is the 100k pair read via ``base_s{seed}_ck100000.pt`` with
     ``torch.equal`` asserted against ``base_s{seed}.pt``; 10k/25k/50k carry NO y
  B  ``model_ctx[i]`` = the per-context arithmetic mean over seeds {0,1,2}; the
     per-seed panel means go to ``raw_panel_mean_seeds`` (diagnostic only)
  C  ZERO identified-set appendix LPs: every row carries ``id_lp_ran: false``
  D  eps=0 IS measured, full triple at n=500, so gate (b) is falsifiable; the
     cell then exits the fidelity series on its own MEASURED gain (F.2d.7)
  E  the eps>0 ceiling-at-zero assertion is EXEMPT at eps=0, where both curves
     receive the same degenerate-vertex reading at r ~ -0.4539 (errata 4)
  G  venue is W2's own node (a60-2209, condo-cse5100, the tidpo interpreter);
     the MANDATORY pre-flight is that the rebuilt panel hash equals every
     ``w2_eps{tag}_s*.json`` ``panel.panel_sha256``. Mismatch is a STOP.

SIGNED (F.9.1): readout panel Q2 (Q1 REFUSED as an axis anywhere), solver
``highs``, floor seed 0, venue WashU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from pfn_dag_verify.artifact_status import DIAGNOSTIC, EST_FIDELITY_NORM, LIVE, stamp
from pfn_dag_verify.corrected_deficit_run import N_PER_HALF, PANEL_SEED
from pfn_dag_verify.corrected_identifiability import (
    build_panel_operator, build_query_panels,
)
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS
from pfn_dag_verify.corrected_models import ModelConfig, PFN
from pfn_dag_verify.corrected_oracle import exact_joint_posterior
from pfn_dag_verify.corrected_tomography import (
    _lp_project_or_ranges, pfn_panel_predictions,
)
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT, SCALES
from pfn_dag_verify.corrected_verdict import eps_tag
from pfn_dag_verify.corrected_world import make_world
from pfn_dag_verify.split_panel import (
    HALF_B, SPLIT_SEED, build_split_panel, contexts_sha256,
)

ESTIMAND_OUT = EST_FIDELITY_NORM        # artifact_status.py:44
PANEL = "Q2"                            # F.9.1 U1, SIGNED. Q1 is REFUSED.
SOLVER = "highs"                        # F.9.1 U3, SIGNED
FLOOR_SEED = 0                          # F.9.1 U4, SIGNED
MODEL_SEEDS = (0, 1, 2)
EPS_GRID = (0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0)
MODEL_STEP = 100_000
DOSE0_STEP = 0
QUERY_PANEL_SEED = 990_300_001          # run_tomography's seed + 1
CALIB_MIN_GAIN = 1e-6                   # mirrors corrected_verdict.CALIB_MIN_GAIN
MODEL_NOMINAL_TOL = 1e-2                # the registered nominal that WOULD govern
SCALE = "base"
D, K = 3, 8

CEILING_ZERO_TOL = 1e-5                 # spec check 7, eps>0 only
CLAUSE4_DELTA_TOL = 1e-6                # spec check 4 / the mandatory cross-check

# F.9.2's LIVE clause-4 floor readout, Q2 / highs / seed 0 / 500 half-B contexts,
# re-derived here from the artifacts themselves
# (campaigns/corrected_20260812/raw/postF/floor_calib/floor_calib_eps*.json,
# curves_by_name.floor.Q2.highs.summary.r_mean), NOT from any prose table.
# Those ran on a40-2205; this pass runs on W2's a6000 node, so the comparison is
# cross-venue and the tolerance is CLAUSE4_DELTA_TOL, not bitwise.
# eps=0 has no clause-4 reference: rule (4) measured only the surviving cells.
CLAUSE4_FLOOR = {
    0.1:  -0.45924796668263074,
    0.25: -0.4799204976766899,
    0.35: -0.48917087565771816,
    0.5:  -0.5075518363899336,
    0.75: -0.5433020861083544,
    1.0:  -0.5558292942319566,
}
CLAUSE4_GAIN = {
    0.1:  0.4592479484357116,
    0.25: 0.4799204860477641,
    0.35: 0.4891707909450456,
    0.5:  0.5075517545225902,
    0.75: 0.543301477258973,
    1.0:  0.555828665344458,
}

_G: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Reused VERBATIM from f2c_clause4_floor.py. Not re-derived: the floor this pass
# must reproduce to 1e-6 was produced by these exact functions.
# --------------------------------------------------------------------------- #
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
            "residual_mean": float(res.mean()),
            "residual_sd": float(res.std(ddof=1)) if n > 1 else 0.0,
            "residual_min": float(res.min()), "residual_max": float(res.max())}


def identity_checks(ck: Path, seed: int, others: list[Path]) -> dict:
    """Gate-A's identity property, plus the cross-eps invariance the spec predicts.

    Verbatim from f2c_clause4_floor.py:191-204. This is spec check 5: a dose-0
    that is not the untrained state rescales every y, and nothing downstream
    would look wrong.
    """
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


# --------------------------------------------------------------------------- #
# 1. pre-flight, before any LP (B1 / decision G)
# --------------------------------------------------------------------------- #
def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def w2_docs(w2_dir: Path, eps: float) -> list[tuple[str, dict]]:
    tag = eps_tag(eps)
    out = []
    for s in MODEL_SEEDS:
        f = w2_dir / f"w2_eps{tag}_s{s}.json"
        if not f.is_file():
            raise SystemExit(
                f"eps={eps}: {f} is absent. The pre-flight compares this node's "
                "panel to what W2 actually scored; without W2's artifact there "
                "is nothing to compare to and the venue is unverifiable.")
        out.append((f.name, json.loads(f.read_text())))
    return out


def preflight(eps: float, split, w2_dir: Path, deficit_dir: Path | None) -> dict:
    """Decision G's operative venue criterion. Refuse on any W2 mismatch.

    F.2d.5's identity check compares BITWISE hashes of a LAPACK-dependent draw
    (errata 1), so "WashU" is not specific enough: four hashes are on record for
    the same eps=0.5 draw. Wherever the hashes match IS the venue.
    """
    mine = split.panel_sha256
    docs = w2_docs(w2_dir, eps)
    mismatched = [(n, d["panel"]["panel_sha256"]) for n, d in docs
                  if d["panel"]["panel_sha256"] != mine]
    if mismatched:
        raise SystemExit(
            f"eps={eps}: this node builds panel {mine[:16]} but W2 scored "
            f"{mismatched[0][1][:16]} in {mismatched[0][0]} "
            f"({len(mismatched)}/{len(docs)} artifacts disagree). Decision G's "
            "falsifier: no venue reproduces W2's bytes. STOP and report — do "
            "not proceed, and do not regenerate a registered artifact.")
    # Recorded, NOT refused: B1 is adjudicated, deficit_w2/ is not regenerated
    # and its hash differs at every eps by construction.
    dfc_hash = None
    if deficit_dir is not None:
        dfc_f = deficit_dir / f"deficit_eps{eps_tag(eps)}.json"
        if dfc_f.is_file():
            dfc_hash = json.loads(dfc_f.read_text()).get("panel_sha256")
    return {"panel_hash_matches_w2": True,
            "panel_sha256_here": mine,
            "w2_artifacts": [n for n, _ in docs],
            "w2_panel_sha256": docs[0][1]["panel"]["panel_sha256"],
            "deficit_panel_sha256": dfc_hash,
            "panel_hash_matches_deficit": (None if dfc_hash is None
                                           else bool(dfc_hash == mine)),
            "deficit_mismatch_is_adjudicated": "B1 (owner, 2026-08-26)"}


# --------------------------------------------------------------------------- #
# 2. workers: one _init per (eps, curve-instance)
# --------------------------------------------------------------------------- #
def _init(eps: float, ckpt: str | None) -> None:
    # One BLAS thread per worker. Unpinned, N workers x 16 threads oversubscribes
    # the box and the measured cost went from 0.1 s/LP to minutes. The PANEL was
    # already built in the parent under W2's own thread budget (decision G); the
    # LPs cannot touch panel bytes, so pinning here is free.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    try:                                    # best effort; already-loaded BLAS
        import threadpoolctl                # ignores the env var above
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    torch.set_num_threads(1)
    world = make_world(k=K, d=D, eps=eps)
    panels = build_query_panels(world, seed=QUERY_PANEL_SEED)
    model = None
    if ckpt is not None:
        cfg = ModelConfig(name=SCALE, d=D, **SCALES[SCALE])
        model = PFN(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
    _G.update(world=world, panels=panels,
              op=build_panel_operator(world, panels[PANEL]), model=model)


def _one(job: tuple) -> dict:
    ci, ctx, curve, seed = job
    world, op, panel = _G["world"], _G["op"], _G["panels"][PANEL]
    exact = exact_joint_posterior(world, ctx)
    if curve == "ceiling":
        preds = _strip_to_pipeline(op.exact_predictive(exact["w_lo"]))
        # spec check 10: the ceiling must not keep `den` and must not gain an
        # interventional channel the PFN does not have. Asserted, not assumed.
        assert all(t is None for t in preds["int"]), "ceiling kept an int channel"
        assert all("den" not in t for t in preds["obs"]), "ceiling kept a den"
    else:
        # floor and model take the SAME branch: both are pfn_panel_predictions.
        preds = pfn_panel_predictions(_G["model"], world, panel, [ctx], "cpu")[0]
    # Rule (3): the record fields. C — no identified-set LP runs in this pass, so
    # id_lp_ran is false everywhere and tol_eff is null everywhere; the model
    # curve carries the registered nominal that WOULD govern if one ran.
    tol_fields = {"nominal_tol": (MODEL_NOMINAL_TOL if curve == "model" else None),
                  "tol_eff": None, "id_lp_ran": False}
    t0 = time.time()
    # An LP that will not solve is a FINDING about the readout, not an exception
    # to swallow and not a value to substitute. Recorded per context with its
    # message and excluded from the summary; run_eps then REFUSES to write the
    # cell, so a partly computable panel never ships as a fully computed one.
    # No per-context solver fallback: rule (1) requires ONE solver across curves.
    try:
        proj = _lp_project_or_ranges(world, op, preds, "project", 0.0, solver=SOLVER)
    except Exception as exc:
        return {"context_index": ci, "curve": curve, "seed": seed,
                "panel": PANEL, "solver": SOLVER,
                "lp_failed": True, "lp_error": str(exc)[:400],
                "projection_residual": None, "structural_js": None, "r": None,
                **tol_fields, "lp_wallclock_s": round(time.time() - t0, 4)}
    js = _structural_js(exact["w_o"], proj["w_o"])
    return {"context_index": ci, "curve": curve, "seed": seed,
            "panel": PANEL, "solver": SOLVER, "lp_failed": False,
            "projection_residual": float(proj["residual"]),
            "structural_js": js, "r": float(0.0 - js),
            **tol_fields, "lp_wallclock_s": round(time.time() - t0, 4)}


def run_curve(curve: str, ckpt: Path | None, ctxs: list, eps: float,
              jobs: int, seed: int | None = None) -> list[dict]:
    jl = [(ci, c, curve, seed) for ci, c in enumerate(ctxs)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                             initargs=(eps, None if ckpt is None else str(ckpt))) as ex:
        rows = list(ex.map(_one, jl, chunksize=4))
    label = curve if seed is None else f"{curve}[s{seed}]"
    print(f"[eps={eps}] {label}: {len(rows)} LPs in {time.time() - t0:.1f}s",
          flush=True)
    return rows


# --------------------------------------------------------------------------- #
# checkpoint resolution — decision A
# --------------------------------------------------------------------------- #
def _provenance(ck: Path) -> dict | None:
    for cand in (ck.with_name(ck.stem + ".provenance.json"),
                 ck.with_name(ck.name.replace(".pt", ".provenance.json"))):
        if cand.is_file():
            return json.loads(cand.read_text())
    return None


def load_trained(netdir: Path, seed: int) -> dict:
    """Decision A. The two 100k files are ONE state written twice; a disagreement
    between them is a stop, not a choice."""
    ck = netdir / f"{SCALE}_s{seed}_ck{MODEL_STEP}.pt"
    fin = netdir / f"{SCALE}_s{seed}.pt"
    for f in (ck, fin):
        if not f.is_file():
            raise SystemExit(f"{f} is absent; decision A reads BOTH 100k files "
                             "and asserts they agree before either is used.")
    a = torch.load(ck, map_location="cpu")
    b = torch.load(fin, map_location="cpu")
    if set(a) != set(b) or not all(torch.equal(a[k], b[k]) for k in a):
        raise SystemExit(
            f"{ck.name} and {fin.name} are both the 100k state and they differ. "
            "The locked text names neither file (errata 9); a disagreement "
            "between them is a stop, not a choice. Provenance forensics.")
    prov = _provenance(fin)
    steps = None if prov is None else prov.get("steps")
    if steps is not None and int(steps) != MODEL_STEP:
        raise SystemExit(f"{fin.name}'s provenance says steps={steps}, not "
                         f"{MODEL_STEP}. A 50k state read as 100k makes y low "
                         "by an unknown amount and monotone-looking.")
    return {"file": str(ck), "sha256": _sha256(ck),
            "final_pt": str(fin), "final_pt_sha256": _sha256(fin),
            "steps": steps, "equals_final_pt": True, "path": ck}


def load_dose0(netdir: Path, seed: int) -> dict:
    """The floor state: W2's OWN persisted dose-0, not a regenerated one.

    F.2c's operational corollary is why W2 persisted it: regenerating here would
    put a second untrained state on the record that only the identity checks
    could tell apart from the fleet's.
    """
    ck = netdir / f"{SCALE}_s{seed}_ck{DOSE0_STEP}.pt"
    if not ck.is_file():
        raise SystemExit(
            f"dose-0 is absent at {ck}. Step 0 used to be a SILENT no-op, which "
            "is why its absence is checked rather than discovered when the floor "
            "curve is built.")
    prov = _provenance(ck) or {}
    if prov.get("steps_taken") not in (0, None) or prov.get("dose0") is False:
        raise SystemExit(
            f"{ck.name}'s provenance says steps_taken={prov.get('steps_taken')} "
            f"dose0={prov.get('dose0')}. The floor is the UNTRAINED state; a "
            "partly trained one shrinks the gain and rescales every y.")
    return {"file": str(ck), "sha256": _sha256(ck),
            "steps_taken": prov.get("steps_taken"),
            "dose0": prov.get("dose0"), "path": ck}


# --------------------------------------------------------------------------- #
# 3. per eps
# --------------------------------------------------------------------------- #
def run_eps(eps: float, a) -> dict:
    t0 = time.time()
    tag = eps_tag(eps)
    world = make_world(k=K, d=D, eps=eps)
    # Built in the PARENT, under whatever thread budget the launcher set — the
    # panel bytes are the venue test (decision G) and the workers pin threads
    # only after this returns.
    split = build_split_panel(world, a.n_per_half, HEADLINE_N_ROWS, PANEL_SEED,
                              split_seed=SPLIT_SEED,
                              n_query_per_context=N_QUERY_PER_CONTEXT)
    prov = split.provenance(HALF_B)
    pf = preflight(eps, split, a.w2_dir, a.deficit_dir)
    print(f"[eps={eps}] pre-flight OK: panel {split.panel_sha256[:16]} == W2's",
          flush=True)
    if a.preflight_only:
        return {"eps": eps, "preflight": pf}

    ctxs = list(split.half(HALF_B))
    # spec check 3: half-A leakage manufactures exactly the association gate (c)
    # tests for. The half is checked by recomputing its digest, not by trusting
    # the label that came with it.
    if contexts_sha256(ctxs) != split.sha256_b:
        raise SystemExit(f"eps={eps}: the contexts handed to the readout do not "
                         "digest to half B. F.2d.5 scores y on half B only.")
    if a.limit:
        ctxs = ctxs[: a.limit]

    netdir = Path(a.netroot) / f"eps{tag}"
    dose0 = load_dose0(netdir, FLOOR_SEED)
    trained = {s: load_trained(netdir, s) for s in MODEL_SEEDS}
    ident = identity_checks(dose0["path"], FLOOR_SEED, a.prior_ck)
    # spec check 5. The clause-4 LIVE artifact records True/True at all six
    # surviving eps, so a False here is drift, not a known tolerance.
    if not ident["identity_check_fresh_seed"]:
        raise SystemExit(
            f"eps={eps}: {dose0['file']} is not the fresh-seed untrained state "
            f"(torch.manual_seed({1000 * FLOOR_SEED + 7}) -> PFN(cfg)). The "
            "floor would be a different model and y would rescale silently.")
    if not ident["identity_check_cross_eps"]:
        print(f"[eps={eps}] WARNING: dose-0 differs across eps; the untrained "
              "state should not depend on the world. Recorded, not refused.",
              flush=True)

    rows: list[dict] = []
    rows += run_curve("ceiling", None, ctxs, eps, a.jobs)
    rows += run_curve("floor", dose0["path"], ctxs, eps, a.jobs)
    for s in MODEL_SEEDS:
        rows += run_curve("model", trained[s]["path"], ctxs, eps, a.jobs, seed=s)

    # spec check 8. Measured Q2 failure rate is 0/6000, so any failure at all is
    # news; a cell with one is written to a name the loader's glob cannot see.
    failures = [r for r in rows if r["lp_failed"]]
    if failures:
        fail_doc = {"eps": eps, "n_failed": len(failures), "rows": failures[:200],
                    "note": "NOT a fidelity artifact. Deliberately named outside "
                            "the loader's fidelity_eps*.json glob."}
        fp = Path(a.raw_dir) / f"LPFAIL_eps{tag}.json"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(fail_doc, indent=2, default=float))
        raise SystemExit(
            f"eps={eps}: {len(failures)} LP(s) failed out of {len(rows)}. The "
            f"measured Q2 failure rate is 0/6000, so this is news, not noise. "
            f"Details at {fp}; no fidelity artifact written for this cell.")

    # --- 4. the three aligned arrays. Alignment is ASSERTED, not assumed. -----
    def _sel(curve, seed=None):
        s = [r for r in rows if r["curve"] == curve
             and (seed is None or r["seed"] == seed)]
        return sorted(s, key=lambda r: r["context_index"])

    def _idx(sel):
        return [r["context_index"] for r in sel]

    def _r(sel):
        return [r["r"] for r in sel]

    want = list(range(len(ctxs)))
    ceiling_rows, floor_rows = _sel("ceiling"), _sel("floor")
    per_seed_rows = {s: _sel("model", s) for s in MODEL_SEEDS}
    for name, sel in [("ceiling", ceiling_rows), ("floor", floor_rows)] + \
                     [(f"model[s{s}]", per_seed_rows[s]) for s in MODEL_SEEDS]:
        if _idx(sel) != want:
            raise SystemExit(
                f"eps={eps}: the {name} curve's context indices are not the "
                "half-B index range. Cell._ctx checks SHAPE, not identity, so a "
                "misalignment here yields a perfectly plausible y.")

    ceiling_ctx = _r(ceiling_rows)
    floor_ctx = _r(floor_rows)
    per_seed = {s: _r(per_seed_rows[s]) for s in MODEL_SEEDS}
    # Decision B: seed-average PER CONTEXT (corrected_verdict.py:182-184). The
    # denominator is common across seeds, so the point estimate equals averaging
    # three y's; the choice moves only SE_y, which F.2d.6 puts at context level.
    model_ctx = [float(v) for v in
                 np.mean(np.vstack([per_seed[s] for s in MODEL_SEEDS]), axis=0)]
    assert len(model_ctx) == len(floor_ctx) == len(ceiling_ctx) == len(ctxs)

    ceil_mean = float(np.mean(ceiling_ctx))
    ceil_sd = float(np.std(ceiling_ctx, ddof=1))
    floor_mean = float(np.mean(floor_ctx))
    gain = ceil_mean - floor_mean

    # spec check 7. Exempt at eps=0 (decision E / errata 4): there both curves
    # receive the same degenerate-vertex reading at r ~ -0.4539, which is the
    # instrument's known behaviour, not a failure. A ceiling away from zero at
    # eps>0 is the Q1 defect appearing on the signed panel and inflates every y.
    if eps > 0 and (abs(ceil_mean) > CEILING_ZERO_TOL or ceil_sd > CEILING_ZERO_TOL):
        raise SystemExit(
            f"eps={eps}: Q2 ceiling is {ceil_mean:.3e} +/- {ceil_sd:.3e}, not 0. "
            "A panel that can represent the exact posterior must return it at "
            "exactly zero residual and exactly zero JS; this is the Q1 defect "
            "F.9.1 withdrew Q1 for, appearing on the signed panel.")

    # The mandatory cross-check: this pass recomputes the floor F.9.2 already
    # measured LIVE at 500 contexts. A pass that cannot reproduce the pre-lock
    # floor is not measuring the same instrument, and that is discoverable
    # before a single model number is believed.
    c4_ref = CLAUSE4_FLOOR.get(eps)
    c4_delta = None if c4_ref is None else abs(floor_mean - c4_ref)
    if c4_delta is not None and c4_delta > CLAUSE4_DELTA_TOL and not a.limit:
        raise SystemExit(
            f"eps={eps}: floor r_mean {floor_mean!r} does not reproduce F.9.2's "
            f"LIVE clause-4 value {c4_ref!r} (|delta| {c4_delta:.3e} > "
            f"{CLAUSE4_DELTA_TOL:g}). This is a different instrument.")

    # --- 5. the artifact. TOP-LEVEL half/n_rows/panel_sha256 (loader:1298-1315).
    doc: dict[str, Any] = {
        **prov,
        "eps": float(eps), "d": D, "K": K, "O": world.O,
        "amendment": "F.2c/F.2c-tol/F.2d.5/F.2d.6",
        "decisions": ".claude/FIDELITY_DECISIONS.md A-G; errata 4,5,9",
        "readout": {"panel": PANEL, "solver": SOLVER,
                    "query_panel_seed": QUERY_PANEL_SEED,
                    "branch": "projection",
                    "statistic": "r = 0 - structural_js(exact_w_o, projected_w_o)"},
        "n_contexts_measured": len(ctxs),
        "model_ctx": model_ctx, "floor_ctx": floor_ctx, "ceiling_ctx": ceiling_ctx,
        "raw_panel_mean_seeds": [float(np.mean(per_seed[s])) for s in MODEL_SEEDS],
        "curves": {
            "model": {
                "seeds": list(MODEL_SEEDS),
                "seed_combination": "per-context arithmetic mean (decision B)",
                "checkpoint": {str(s): {k: v for k, v in trained[s].items()
                                        if k != "path"} for s in MODEL_SEEDS},
                "per_seed": {str(s): {"per_context": per_seed_rows[s],
                                      "summary": _summary(per_seed_rows[s])}
                             for s in MODEL_SEEDS},
            },
            "floor": {
                "seed": FLOOR_SEED,
                "checkpoint": {**{k: v for k, v in dose0.items() if k != "path"},
                               **ident},
                "per_context": floor_rows, "summary": _summary(floor_rows),
            },
            "ceiling": {
                "source": "exact_joint_posterior -> op.exact_predictive "
                          "-> _strip_to_pipeline",
                "per_context": ceiling_rows, "summary": _summary(ceiling_rows),
            },
        },
        "gain": {"ceiling_minus_floor": gain, "calib_min_gain": CALIB_MIN_GAIN,
                 "below_min_gain": bool(gain < CALIB_MIN_GAIN),
                 "clause4_reference": CLAUSE4_GAIN.get(eps),
                 "clause4_floor_reference": c4_ref,
                 "clause4_abs_delta": c4_delta,
                 "exit_note": ("eps=0 exits the fidelity series on this MEASURED "
                               "gain (F.2d.7 / decision D); it is not exempt from "
                               "measurement, only from y.") if eps == 0 else None},
        "lp_failures": [],
        "id_lp_policy": ("decision C: zero identified-set appendix LPs in this "
                         "pass; rule (3)'s tol_eff clause is conditional and is "
                         "satisfied vacuously"),
        "preflight": pf,
        "environment": {
            "host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
            "threads": {v: os.environ.get(v) for v in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS")},
            "git_head": a.git_head,
            "amendment_f_sha256": a.amendment_sha256,
        },
        "wallclock_s": round(time.time() - t0, 2),
    }
    # stamp OVERWRITES doc["artifact"] (artifact_status.py:49-56): call it LAST,
    # and never move half/n_rows/panel_sha256 inside that block — the loader
    # reads fidelity provenance at TOP LEVEL, unlike W2's nested layout.
    stamp(doc, ESTIMAND_OUT, LIVE if a.status == "live" else DIAGNOSTIC)
    out = Path(a.raw_dir) / f"fidelity_eps{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # A LIVE artifact is the registered per-context record. The sbatch defaults
    # STATUSVAL to diagnostic, so a single accidental resubmit would otherwise
    # pass every in-script check and replace the registered set with DIAGNOSTIC
    # twins -- detectable downstream only because require_live refuses them, by
    # which point the per-context rows are gone. Refuse instead of overwriting.
    if out.exists():
        try:
            prior = json.loads(out.read_text()).get("artifact", {}).get("status")
        except Exception:
            prior = "unreadable"
        if prior == LIVE and a.status != "live":
            raise SystemExit(
                f"REFUSING: {out} is stamped LIVE and this run is "
                f"--status {a.status}. Overwriting a registered artifact with a "
                f"diagnostic one destroys the per-context record. Move the LIVE "
                f"file aside deliberately, or pass --status live.")
        if not a.overwrite:
            raise SystemExit(
                f"REFUSING: {out} already exists (status {prior}). Pass "
                f"--overwrite to replace it deliberately.")
    out.write_text(json.dumps(doc, indent=2, default=float))
    a.prior_ck.append(dose0["path"])
    print(f"[eps={eps}] wrote {out} in {doc['wallclock_s']}s "
          f"(floor {floor_mean:.9f}, ceiling {ceil_mean:.3e}, gain {gain:.9f})",
          flush=True)
    return doc


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _amendment_sha(repo: Path) -> str | None:
    f = repo / "AMENDMENT_F.md"
    return _sha256(f) if f.is_file() else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", type=float, nargs="*", default=list(EPS_GRID))
    p.add_argument("--n-per-half", type=int, default=N_PER_HALF)
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--limit", type=int, default=0,
                   help="measure only the first N half-B contexts (smoke only)")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--w2-dir", type=Path,
                   default=Path("campaigns/corrected_20260812/raw/postF/w2"))
    p.add_argument("--netroot", type=Path, default=None,
                   help="default: <w2-dir>/nets")
    p.add_argument("--deficit-dir", type=Path,
                   default=Path("campaigns/corrected_20260812/raw/postF/deficit_w2"),
                   help="recorded for the B1 comparison; never refuses")
    p.add_argument("--raw-dir", type=Path,
                   default=Path("campaigns/corrected_20260812/raw/postF/fidelity"))
    p.add_argument("--preflight-only", action="store_true",
                   help="rebuild each panel and compare to W2's hash; no LPs")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing artifact; refused by default")
    p.add_argument("--status", choices=["diagnostic", "live"], default="diagnostic",
                   help="LIVE only on a node whose pre-flight matches W2's bytes")
    a = p.parse_args(argv)
    if a.netroot is None:
        a.netroot = a.w2_dir / "nets"
    if a.status == "live" and a.limit:
        raise SystemExit("REFUSING: a --limit run is a smoke, not the measurement.")
    a.prior_ck = []
    a.git_head = _git_head(a.repo)
    a.amendment_sha256 = _amendment_sha(a.repo)
    for e in a.eps:
        run_eps(e, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
