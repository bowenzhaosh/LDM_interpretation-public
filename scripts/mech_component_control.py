"""Size-matched non-order component control — analysis (REPORTED, never gated).

Pre-specification: campaigns/mech_ext_20260902/PRESPEC_component_control.md (digested;
its sha256 is recorded in every output). Inputs: the coarse-oracle pass
(scripts/mech_coarse_oracle.py -> coarse_eps{tag}[_n40ext].npz) and the registered
scored cells (predgain_eps{tag}_ck{step}.npz: S for full/abl/prior/model seeds).

Per (cell, eps):
  regret_i = S_i(full) - mean_s S_i(model_s)          from the registered npz
  G^X_i    = S_i(full) - S_i(w_X)                      for every oracle X in the coarse pass
  b^X      = OLS slope of regret_i on G^X_i           (closed form; = mech_gates._ols's slope)
  Delta(Gbar*) = median_{25 order partitions nearest Gbar*} b^H
               - median_{25 atom  partitions nearest Gbar*} b^Pi,
    Gbar* in {Gbar_order(eps), 0.02}; 2000 paired context resamples, mech_gates._rng(label),
    medians recomputed inside every resample.
  joint OLS regret = a + b G + b' G^{Pi*}, Pi* = the atom partition nearest Gbar_order.
  family curves b^X vs Gbar^X restricted to Gbar^X >= 0.005; binned medians.
  non-parametric: share of contexts with S_i(model mean) > S_i(w_X) for X = abl and Pi*.
Decision (pre-specified) read on base@1e-3@500k only; every other cell is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import mech_gates as MG  # noqa: E402  (locked; _rng / N_BOOT / BOOT_SEED)
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

N_NEAREST = 25
G_MIN = 0.005
G_STAR_FIXED = 0.02
BINS = [0.005, 0.01, 0.03, 0.06, 0.12, 0.40]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def slopes(reg: np.ndarray, G: np.ndarray) -> np.ndarray:
    """OLS slope of reg on each column of G (n, m) -> (m,)."""
    Gc = G - G.mean(axis=0, keepdims=True)
    rc = reg - reg.mean()
    with np.errstate(divide="ignore", invalid="ignore"):          # a zero-variance column (G == 0) -> nan, excluded by G_MIN
        return (Gc * rc[:, None]).sum(axis=0) / (Gc * Gc).sum(axis=0)


def joint_fit(reg, g1, g2):
    X = np.column_stack([np.ones_like(g1), g1, g2])
    beta, *_ = np.linalg.lstsq(X, reg, rcond=None)
    return beta  # a, b, b'


def analyse(cell_dir: Path, cell: str, eps: float, step: int, coarse: Path, kind: str, unit: str = "mean") -> dict:
    tag = eps_tag(eps)
    suffix = "" if kind == "orig" else f"_{kind}"
    zr = np.load(cell_dir / f"predgain_eps{tag}_ck{step}{suffix}.npz")
    zc = np.load(coarse / f"coarse_eps{tag}{suffix}.npz")
    names = list(zr["names"])
    Sr = zr["S"].mean(axis=1)                                    # (n_ctx, n_pred)
    full, abl = Sr[:, names.index("full")], Sr[:, names.index("abl")]
    cols = [i for i, n in enumerate(names) if n.startswith("model_s")]
    model = Sr[:, cols].mean(axis=1) if unit == "mean" else Sr[:, names.index(f"model_s{unit}")]
    Sc = zc["S_mean"]                                            # (n_ctx, n_or)
    if Sc.shape[0] != Sr.shape[0]:
        raise SystemExit(f"{cell} eps {eps}: {Sc.shape[0]} coarse contexts vs {Sr.shape[0]} scored")
    dev = float(np.max(np.abs(Sc[:, 0] - full)))
    if dev > 1e-9 or float(np.max(np.abs(Sc[:, 1] - abl))) > 1e-9:
        raise SystemExit(f"{cell} eps {eps}: coarse pass and scored cell disagree on full/abl ({dev:.2e})")
    for key in ("panel_seed", "split_seed", "n_per_half"):
        if int(zr[key]) != int(zc[key]):
            raise SystemExit(f"{cell} eps {eps}: {key} differs between the scored cell and the coarse pass")
    reg = full - model
    Gall = Sc[:, [0]] - Sc                                       # (n_ctx, n_or); col 0 == 0
    Gbar = Gall.mean(axis=0)
    fam = np.array([str(f) for f in zc["family"]])
    n_reg = int(zc["n_reg"])
    G_order = Gall[:, 1]
    b_reg = float(slopes(reg, G_order[:, None])[0])
    Gbar_order = float(Gbar[1])
    atom = np.where((fam == "atom") & (Gbar >= G_MIN))[0]
    order = np.where((fam == "order") & (Gbar >= G_MIN))[0]
    b_all = slopes(reg, Gall[:, 1:]); b_all = np.concatenate([[np.nan], b_all])   # aligned to columns
    out = {"cell": cell, "eps": eps, "step": step, "kind": kind, "unit": unit, "n_ctx": int(len(reg)),
           "b_order_registered_estimator": b_reg, "Gbar_order": Gbar_order,
           "coarse_file": str(coarse / f"coarse_eps{tag}{suffix}.npz"), "coarse_sha256": _sha(coarse / f"coarse_eps{tag}{suffix}.npz"),
           "scored_file": str((cell_dir / f"predgain_eps{tag}_ck{step}{suffix}.npz").relative_to(ROOT)),
           "n_atom_used": int(len(atom)), "n_order_used": int(len(order)), "G_min": G_MIN,
           "atom_family": {"Gbar_range": [float(Gbar[atom].min()), float(Gbar[atom].max())],
                           "b_median": float(np.median(b_all[atom])), "b_q05": float(np.quantile(b_all[atom], .05)),
                           "b_q95": float(np.quantile(b_all[atom], .95)), "b_min": float(b_all[atom].min()), "b_max": float(b_all[atom].max()),
                           "binned_median_b": {}},
           "order_family": {"Gbar_range": [float(Gbar[order].min()), float(Gbar[order].max())] if len(order) else None,
                            "b_median": float(np.median(b_all[order])) if len(order) else None,
                            "binned_median_b": {}},
           "delta": {}, "joint": {}, "nonparametric": {}}
    for lo, hi in zip(BINS, BINS[1:]):
        for name, idx in (("atom_family", atom), ("order_family", order)):
            m = idx[(Gbar[idx] >= lo) & (Gbar[idx] < hi)]
            out[name]["binned_median_b"][f"[{lo},{hi})"] = {"n": int(len(m)), "b_median": float(np.median(b_all[m])) if len(m) else None}
    # ---- Delta at the two Gbar* values, paired bootstrap with medians inside every resample
    n = len(reg)
    for gl, gstar in (("Gbar_order", Gbar_order), ("0.02", G_STAR_FIXED)):
        a25 = atom[np.argsort(np.abs(Gbar[atom] - gstar))[:N_NEAREST]]
        o25 = order[np.argsort(np.abs(Gbar[order] - gstar))[:N_NEAREST]] if len(order) else np.array([], int)
        if len(o25) == 0:
            out["delta"][gl] = {"Gstar": gstar, "not_evaluable": "order family empty above G_min"}; continue
        sel = np.concatenate([a25, o25]); Gsel = Gall[:, sel]
        b_sel = slopes(reg, Gsel)
        point = float(np.median(b_sel[len(a25):]) - np.median(b_sel[:len(a25)]))
        rng = MG._rng(f"cc:{cell}:{eps}:{step}:{kind}:{unit}:delta:{gl}")
        bs = np.empty(MG.N_BOOT)
        for t in range(MG.N_BOOT):
            i = MG._boot_idx(n, rng)
            bb = slopes(reg[i], Gsel[i])
            bs[t] = np.median(bb[len(a25):]) - np.median(bb[:len(a25)])
        out["delta"][gl] = {"Gstar": gstar, "delta": point, "se": float(bs.std(ddof=1)),
                            "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                            "z": point / float(bs.std(ddof=1)),
                            "atom25_Gbar_range": [float(Gbar[a25].min()), float(Gbar[a25].max())], "atom25_b_median": float(np.median(b_sel[:len(a25)])),
                            "order25_Gbar_range": [float(Gbar[o25].min()), float(Gbar[o25].max())], "order25_b_median": float(np.median(b_sel[len(a25):])),
                            "order25_reaches_Gstar": bool(Gbar[o25].max() >= gstar * 0.8)}
    # ---- joint fit with the single atom partition nearest Gbar_order
    pistar = int(atom[np.argmin(np.abs(Gbar[atom] - Gbar_order))])
    Gp = Gall[:, pistar]
    a0, b0, bp0 = joint_fit(reg, G_order, Gp)
    rng = MG._rng(f"cc:{cell}:{eps}:{step}:{kind}:{unit}:joint")
    bj = np.array([joint_fit(reg[i], G_order[i], Gp[i]) for i in (MG._boot_idx(n, rng) for _ in range(MG.N_BOOT))])
    out["joint"] = {"pistar": str(zc["names"][pistar]), "Gbar_pistar": float(Gbar[pistar]),
                    "a": float(a0), "b_order": float(b0), "se_b_order": float(bj[:, 1].std(ddof=1)),
                    "b_prime_atom": float(bp0), "se_b_prime": float(bj[:, 2].std(ddof=1)),
                    "b_prime_simple": float(b_all[pistar]), "corr_G_Gpi": float(np.corrcoef(G_order, Gp)[0, 1])}
    out["nonparametric"] = {"share_model_beats_abl": float((model > Sc[:, 1]).mean()),
                            "share_model_beats_pistar": float((model > Sc[:, pistar]).mean())}
    return out


def decide(r: dict) -> str:
    d = r["delta"]
    ok = all(("delta" in d[k]) and d[k]["delta"] > 0 and d[k]["ci95"][0] > 0 for k in ("Gbar_order", "0.02"))
    if ok:
        return "A"
    if any(("delta" in d[k]) and abs(d[k]["delta"]) < 0.05 and d[k]["ci95"][0] <= 0 <= d[k]["ci95"][1] for k in d):
        return "B-compatible"
    return "undecided"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", type=Path, default=ROOT / "campaigns/mech_ext_20260902/coarse")
    p.add_argument("--confirm-root", type=Path, default=ROOT / "campaigns/mech_20260827/predgain_confirm")
    p.add_argument("--out", type=Path, default=ROOT / "campaigns/mech_ext_20260902/reported")
    p.add_argument("--cells", nargs="*", default=["base_lr0.001_d10000:10000", "base_lr0.001_d25000:25000", "base_lr0.001_d100000:100000",
                                                   "base_lr0.001_d500000:500000", "base_lr0.001_d2000000:2000000", "large_lr0.0003_d500000:500000"])
    p.add_argument("--eps", type=float, nargs="*", default=[0.5, 0.75, 1.0])
    p.add_argument("--per-seed", action="store_true")
    p.add_argument("--n40", action="store_true", help="also the paired n40 extension cell (kind n40ext, eps 0.75)")
    a = p.parse_args()
    if "campaigns/mech_20260827" in str(a.out.resolve()):
        raise SystemExit("REFUSING: --out inside the registered campaign")
    t0 = time.time()
    results = []
    for spec in a.cells:
        cell, step = spec.split(":"); step = int(step)
        d = a.confirm_root / cell
        for e in a.eps:
            f = d / f"predgain_eps{eps_tag(e)}_ck{step}.npz"
            if not f.is_file():
                print(f"  skip {cell} eps {e}: no scored file", flush=True); continue
            units = ["mean"] + ([int(s) for s in np.load(f)["seeds"]] if a.per_seed else [])
            for u in units:
                r = analyse(d, cell, e, step, a.coarse, "orig", "mean" if u == "mean" else str(u))
                r["decision_rule_readout"] = decide(r)
                results.append(r)
                dd = r["delta"]
                print(f"  {cell} eps {e} unit {u}: b_order {r['b_order_registered_estimator']:.4f} | Delta(Gbar_order) "
                      f"{dd['Gbar_order'].get('delta', float('nan')):+.3f} [{dd['Gbar_order'].get('ci95', [np.nan, np.nan])[0]:+.3f},{dd['Gbar_order'].get('ci95', [np.nan, np.nan])[1]:+.3f}] | "
                      f"Delta(0.02) {dd['0.02'].get('delta', float('nan')):+.3f} | joint b' {r['joint']['b_prime_atom']:+.4f}±{r['joint']['se_b_prime']:.4f} | "
                      f"atom-family median b {r['atom_family']['b_median']:+.3f} | {r['decision_rule_readout']}", flush=True)
    if a.n40:
        d = a.confirm_root.parent / "predgain_confirm_n40" / "base_n40_lr0.001_d500000"
        r = analyse(d, "base_n40_lr0.001_d500000", 0.75, 500000, a.coarse, "n40ext")
        r["decision_rule_readout"] = decide(r); results.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    primary = [r for r in results if r["cell"] == "base_lr0.001_d500000" and r["unit"] == "mean"]
    doc = {"status": "REPORTED — never gated; Amendment G verdict untouched",
           "prespec": "campaigns/mech_ext_20260902/PRESPEC_component_control.md",
           "prespec_sha256": _sha(ROOT / "campaigns/mech_ext_20260902/PRESPEC_component_control.md"),
           "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED, "n_nearest": N_NEAREST, "G_min": G_MIN,
           "decision_cell": "base_lr0.001_d500000", "decision_per_eps": {str(r["eps"]): r["decision_rule_readout"] for r in primary},
           "decision": ("A" if primary and all(r["decision_rule_readout"] == "A" for r in primary) else
                        ("B-compatible" if primary and all(r["decision_rule_readout"] == "B-compatible" for r in primary) else "undecided")),
           "results": results, "wall_s": time.time() - t0}
    (a.out / "component_control.json").write_text(json.dumps(doc, indent=1, allow_nan=False))
    man = {"script": "scripts/mech_component_control.py", "script_sha256": _sha(Path(__file__).resolve()),
           "mech_gates_sha256": _sha(ROOT / "scripts/mech_gates.py"),
           "inputs_sha256": {**{r["scored_file"]: _sha(ROOT / r["scored_file"]) for r in results},
                             **{r["coarse_file"]: r["coarse_sha256"] for r in results}},
           "outputs_sha256": {"component_control.json": _sha(a.out / "component_control.json")},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (a.out / "component_control.MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"[cc] decision on {doc['decision_cell']}: {doc['decision']} ({doc['decision_per_eps']}); {len(results)} analyses -> {a.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
