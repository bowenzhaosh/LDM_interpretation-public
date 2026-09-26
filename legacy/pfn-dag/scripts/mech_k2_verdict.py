"""Arm G (PRESPEC_internal §3): the size-matched world K = 2. REPORTED, never gated.

On the K = 2 gate panel (same seeds/queries as the registered panel; world make_world(k=2, d=3, eps)):
  coarse pass  mech_coarse_oracle.run_eps with MECH_K=2 (partitions_for(2) -> {full, one block}), --ref-dir the
               harvested ext cell  campaigns/mech_ext_20260902/predgain/base_lr0.001_d500000_K2/s3-8/
  joint OLS    regret_i = a + b*G_order,i + b'*G_atom,i,  G_atom,i = S_i(full) - S_i(atom:00)
  D_G = b - b', one paired context bootstrap (2000, mech_gates._rng("k2:...")); PRIMARY = seed-mean over seeds 3-8;
  per-seed sign agreement on D_G required.  Secondary: Delta(Gbar_atom) = median_{25 order partitions nearest
  Gbar_atom} b^H - b^{atom:00}.
  Decision (eps 1.0 decides; eps .75 consistency; eps .5 reported), evaluable iff SE(b), SE(b') <= SE_MAX and the
  provenance holds (world_index K = 2, sha256_b differs from the registered cell's, confirm true, m_q 8):
    FAVOURS A  iff D_G > 0 with 95 % CI excluding 0 AND the upper 95 % CI of b' <= 0.15 at eps 1, same sign at eps .75
    FAVOURS B  iff the 95 % CI of D_G lies within (-0.10, +0.10) AND the lower CI of b' >= 0.10 at eps 1; or D_G < 0 with CI excluding 0
    else UNDECIDED (both predicted values printed: under B b ~ b' ~ 0.3, under A b' within ±0.05 of 0 and D_G ~ +0.25).
--validate first reproduces the registered decision cell's b' (base@1e-3@500k eps .75) from reported/component_control.json
to 1e-9 through the same joint_fit before anything is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import mech_gates as MG  # noqa: E402  (locked)
from mech_component_control import joint_fit, slopes  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

INT = ROOT / "campaigns/mech_int_20260905"
EXT = ROOT / "campaigns/mech_ext_20260902"
B_PRIME_MAX_A, DG_NULL, B_PRIME_MIN_B = 0.15, 0.10, 0.10
PRED = {"under_B": {"b": 0.3, "b_prime": 0.3, "D_G": 0.0}, "under_A": {"b_prime": "within ±0.05 of 0", "D_G": 0.25}}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate() -> None:
    cc = json.load(open(EXT / "reported/component_control.json"))
    r = next(x for x in cc["results"] if x["cell"] == "base_lr0.001_d500000" and x["unit"] == "mean" and abs(x["eps"] - 0.75) < 1e-9)
    z = np.load(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000/predgain_eps0p75_ck500000.npz")
    zc = np.load(EXT / "coarse/coarse_eps0p75.npz")
    n = list(z["names"]); S = z["S"].mean(1); full = S[:, n.index("full")]
    reg = full - S[:, [i for i, k in enumerate(n) if k.startswith("model_s")]].mean(1)
    G = full - S[:, n.index("abl")]; cn = list(zc["names"]); Gp = zc["S_mean"][:, 0] - zc["S_mean"][:, cn.index(r["joint"]["pistar"])]
    bp = joint_fit(reg, G, Gp)[2]
    if abs(bp - r["joint"]["b_prime_atom"]) > 1e-9:
        raise SystemExit(f"validate: b' {bp} != registered {r['joint']['b_prime_atom']}")
    print(f"[validate] registered b' reproduced: {bp:.6f}")


def fit(cell_dir: Path, coarse_dir: Path, eps: float, step: int, seeds, unit):
    tag = eps_tag(eps)
    z = np.load(cell_dir / f"predgain_eps{tag}_ck{step}.npz"); zc = np.load(coarse_dir / f"coarse_eps{tag}.npz")
    n = list(z["names"]); S = z["S"].mean(1); full = S[:, n.index("full")]; abl = S[:, n.index("abl")]
    if float(np.max(np.abs(zc["S_mean"][:, 0] - full))) > 1e-9:
        raise SystemExit("coarse pass and scored cell disagree on S(full)")
    cols = [n.index(f"model_s{s}") for s in seeds]
    model = S[:, cols].mean(1) if unit == "mean" else S[:, n.index(f"model_s{unit}")]
    reg = full - model; G = full - abl
    cn = list(zc["names"]); fam = np.array([str(f) for f in zc["family"]]); Gbar = (zc["S_mean"][:, [0]] - zc["S_mean"]).mean(0)
    i_atom = [i for i in range(len(cn)) if fam[i] == "atom" and int(zc["n_blocks"][i]) == 1][0]     # the one-block atom partition
    G_atom = zc["S_mean"][:, 0] - zc["S_mean"][:, i_atom]
    a0, b0, bp0 = joint_fit(reg, G, G_atom)
    lab = f"k2:{cell_dir.name}:{eps}:{step}:{unit}"
    rng = MG._rng(lab); N = len(reg)
    bs = np.array([joint_fit(reg[i], G[i], G_atom[i]) for i in (MG._boot_idx(N, rng) for _ in range(MG.N_BOOT))])
    dg = bs[:, 1] - bs[:, 2]
    # secondary: order family nearest Gbar_atom vs the atom one-block slope
    order = np.where((fam == "order") & (Gbar >= 0.005))[0]
    o25 = order[np.argsort(np.abs(Gbar[order] - Gbar[i_atom]))[:25]]
    b_ord25 = float(np.median(slopes(reg, zc["S_mean"][:, [0]] - zc["S_mean"][:, o25])))
    b_atom_simple = float(slopes(reg, G_atom[:, None])[0])
    return {"unit": unit, "n_ctx": int(N), "a": float(a0), "b": float(b0), "b_prime": float(bp0),
            "se_b": float(bs[:, 1].std(ddof=1)), "se_b_prime": float(bs[:, 2].std(ddof=1)),
            "b_ci": [float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))],
            "b_prime_ci": [float(np.quantile(bs[:, 2], .025)), float(np.quantile(bs[:, 2], .975))],
            "D_G": float(b0 - bp0), "D_G_ci": [float(np.quantile(dg, .025)), float(np.quantile(dg, .975))],
            "Gbar_order": float(G.mean()), "Gbar_atom": float(G_atom.mean()), "corr_G_Gatom": float(np.corrcoef(G, G_atom)[0, 1]),
            "b_simple_order": float(MG._ols(reg, G)[0]), "b_simple_atom_oneblock": b_atom_simple,
            "delta_Gbar_atom_order25_minus_atom": b_ord25 - b_atom_simple, "order25_Gbar_range": [float(Gbar[o25].min()), float(Gbar[o25].max())],
            "boot_label": lab}


def decide(r1, r075):
    ev = r1["se_b"] <= MG.SE_MAX and r1["se_b_prime"] <= MG.SE_MAX
    if not ev:
        return "NOT_EVALUABLE"
    if r1["D_G"] > 0 and r1["D_G_ci"][0] > 0 and r1["b_prime_ci"][1] <= B_PRIME_MAX_A and (r075 is None or np.sign(r075["D_G"]) == np.sign(r1["D_G"])):
        return "FAVOURS A"
    if (-DG_NULL < r1["D_G_ci"][0] and r1["D_G_ci"][1] < DG_NULL and r1["b_prime_ci"][0] >= B_PRIME_MIN_B) or (r1["D_G"] < 0 and r1["D_G_ci"][1] < 0):
        return "FAVOURS B"
    return "UNDECIDED"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="base_lr0.001_d500000_K2")
    p.add_argument("--seedtag", default="s3-8")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    p.add_argument("--coarse", default=str(INT / "coarse_K2"))
    p.add_argument("--out", default=str(INT / "reported"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    validate()
    cell_dir = EXT / "predgain" / a.cell / a.seedtag
    idx = json.load(open(cell_dir / "world_index.json")) if (cell_dir / "world_index.json").is_file() else {}
    reg_sha = str(np.load(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000/predgain_eps1p0_ck500000.npz")["sha256_b"])
    results = {}
    for eps in a.eps:
        tag = eps_tag(eps); f = cell_dir / f"predgain_eps{tag}_ck{a.step}.npz"
        if not f.is_file():
            print(f"[k2] eps {eps}: {f.name} not harvested yet"); continue
        z = np.load(f)
        w = idx.get(f.name, {})
        prov_ok = (w.get("K") == 2 and bool(z["confirm"]) and int(z["m_q"]) == MG.M_Q and str(z["sha256_b"]) != reg_sha
                   and [int(s) for s in z["seeds"]] == a.seeds)
        if not prov_ok:
            raise SystemExit(f"{f.name}: provenance check failed (world_index {w}, confirm {bool(z['confirm'])}, m_q {int(z['m_q'])}, seeds {list(z['seeds'])})")
        if not (Path(a.coarse) / f"coarse_eps{tag}.npz").is_file():
            raise SystemExit(f"coarse pass missing: run mech_coarse_oracle.py with MECH_K=2 --ref-dir {cell_dir} --out {a.coarse} --eps {eps}")
        results[str(eps)] = {"mean": fit(cell_dir, Path(a.coarse), eps, a.step, a.seeds, "mean"),
                             "per_seed": {str(s): fit(cell_dir, Path(a.coarse), eps, a.step, a.seeds, s) for s in a.seeds}}
        r = results[str(eps)]["mean"]
        print(f"[k2] eps {eps}: b {r['b']:.3f}±{r['se_b']:.3f} b' {r['b_prime']:.3f}±{r['se_b_prime']:.3f} D_G {r['D_G']:+.3f} {r['D_G_ci']} | Gbar order/atom {r['Gbar_order']:.4f}/{r['Gbar_atom']:.4f} corr {r['corr_G_Gatom']:.2f} | per-seed D_G signs {[np.sign(results[str(eps)]['per_seed'][str(s)]['D_G']) for s in a.seeds]}")
    verdict = None
    if "1.0" in results:
        r1 = results["1.0"]["mean"]; r075 = results.get("0.75", {}).get("mean")
        signs = [np.sign(results["1.0"]["per_seed"][str(s)]["D_G"]) for s in a.seeds]
        verdict = decide(r1, r075)
        if verdict == "FAVOURS A" and not all(x > 0 for x in signs):
            verdict = "UNDECIDED (per-seed D_G signs disagree)"
    doc = {"status": "REPORTED — never gated", "cell": a.cell, "seeds": a.seeds, "results": results, "verdict_eps1": verdict,
           "predicted": PRED, "thresholds": {"b_prime_max_A": B_PRIME_MAX_A, "D_G_null": DG_NULL, "b_prime_min_B": B_PRIME_MIN_B, "se_max": MG.SE_MAX},
           "meta": {"prespec_sha256": _sha(INT / "PRESPEC_internal.md"), "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED,
                    "inputs_sha256": {**{f"scored_{e}": _sha(cell_dir / f"predgain_eps{eps_tag(float(e))}_ck{a.step}.npz") for e in results},
                                      **{f"coarse_{e}": _sha(Path(a.coarse) / f"coarse_eps{eps_tag(float(e))}.npz") for e in results}}}}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dest = out / f"armG_{a.cell}_{a.seedtag}.json"
    if dest.is_file() and not a.force:
        raise SystemExit(f"{dest} exists; written once")
    dest.write_text(json.dumps(doc, indent=1, allow_nan=False, default=float))
    print(f"[k2] verdict (eps 1 decides): {verdict} -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
