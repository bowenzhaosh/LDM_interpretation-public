"""Arm C (PRESPEC_internal §2), stage 3 — analysis. REPORTED, never gated.

Inputs: patch/build_eps{tag}_{world}.npz (directions v_ord, v_atom, v_abl, v_atomabl; exact swap movements;
||v||^2; primary partners) and patch/run_{cell}_eps{tag}_{world}_s{seed}_ck{step}.npz per seed (Delta per
site x source x query; log p_model). Inner products are p_true-weighted and centred, summed over queries.

  gamma(s, C) = argmin sum_i || Delta_i(s,C) - gamma_ord v_ord,i - gamma_atom v_atom,i ||^2   (2x2 normal equations,
                per-context Gram and rhs summed; weighted R^2 and the off-span fraction stored)
  Phi_ord = gamma_ord(INPUT, B(o'*)),  Phi_atom = gamma_atom(INPUT, B'(k'*))            (primary partners)
  rho_ord(s) = gamma_ord(s, B(o'*)) / Phi_ord   (formed only when the CI of Phi excludes 0; CI width <= 0.3 for evaluability)
  nu_ord(s)  = coefficient of Delta_s(ordmean) on v_abl in the (v_abl, v_atom) regression;  nu~_ord(s) = nu_ord(s) / nu_ord(cut)
  lambda     = cross-axis coefficients of the EXACT swap movements (model-blind leaks), subtracted in the specificity check
  gamma_T    = the same coefficients for the model's own predictive re-tempered to the patched predictive's mean entropy
  C1 (reported): family regression of the per-swap single-direction coefficient on log ||v||^2 with a family indicator
  C2 (layer-1 cut): s_c, s_q, s_t = rho_ord(cut_c/q/t); ROWS / PRE-AGGREGATED / MIXED; contrast s_c(ord) - s_c(atom)
     -> LATER-THAN-ATOM / EARLIER / SAME-ROUTE (thresholds 0.2, paired CI, gamma - gamma_T CI)
  C3 (last layer): DEDICATED-ORDER-HEAD / DISTRIBUTED; CONTENT-ROUTED / PATTERN-DEPENDENT / MIXED (both axes)
Bootstrap: 2000 context resamples of the per-context sums (mech_gates._rng labels). PRIMARY unit = per-context
mean over seeds of Delta; per-seed verdict counts reported. Sensitivity: every threshold at x0.5 and x1.5.
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
import mech_gates as MG  # noqa: E402  (locked)
from mech_probe_acts import INT, _env_panel, _world, world_tag  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

TH = {"rows": 0.8, "rows_ci": 0.7, "preagg": 0.5, "preagg_ci": 0.4, "contrast": 0.2, "head_rho": 0.5, "head_nu": 0.5,
      "head_diff": 0.3, "content_val": 0.8, "content_pat": 0.2, "pattern": 0.3, "rho_ci_width": 0.3, "spec": 0.5}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ weighted centred inner products
def wip(p, u, v):
    """sum over queries of the p-weighted centred inner product; u, v, p: (..., m_q, B) -> (...,)"""
    uc = u - (p * u).sum(-1, keepdims=True); vc = v - (p * v).sum(-1, keepdims=True)
    return (p * uc * vc).sum(-1).sum(-1)


def gram_rhs_all(P, Delta, V):
    """batched over contexts: P, Delta (n, m_q, B); V list of (n, m_q, B) -> Gs (n, k, k), rs (n, k), dd (n,)."""
    k = len(V)
    Gs = np.zeros((Delta.shape[0], k, k)); rs = np.zeros((Delta.shape[0], k))
    for a_ in range(k):
        rs[:, a_] = wip(P, Delta, V[a_])
        for b_ in range(a_, k):
            Gs[:, a_, b_] = Gs[:, b_, a_] = wip(P, V[a_], V[b_])
    return Gs, rs, wip(P, Delta, Delta)


def solve(Gs, rs, dd):
    """pooled LS from summed Gram/rhs: coefficients, weighted R^2, off-span fraction."""
    G, r, D = Gs.sum(0), rs.sum(0), dd.sum()
    coef = np.linalg.solve(G + 1e-12 * np.eye(len(r)), r)
    explained = coef @ r
    return coef, float(explained / max(D, 1e-300)), float(1 - explained / max(D, 1e-300))


def boot(fn, n, label, n_boot=MG.N_BOOT):
    rng = MG._rng(label)
    vals = np.array([fn(MG._boot_idx(n, rng)) for _ in range(n_boot)])
    return vals


def ci(vals):
    return [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))]


def temper_all(lp, target_H):
    """batched: lp (n, m_q, B), target_H (n,) -> log p^{1/T_i} normalised, T_i s.t. mean_q H_i = target_H_i (bisection)."""
    def H_of(T):
        l = lp / T[:, None, None]; l = l - l.max(-1, keepdims=True); l = l - np.log(np.exp(l).sum(-1, keepdims=True))
        return -(np.exp(l) * l).sum(-1).mean(-1), l
    lo = np.full(lp.shape[0], 0.05); hi = np.full(lp.shape[0], 20.0)
    for _ in range(50):
        mid = np.sqrt(lo * hi); H, _ = H_of(mid)
        lo = np.where(H < target_H, mid, lo); hi = np.where(H < target_H, hi, mid)
    return H_of(np.sqrt(lo * hi))[1]


# ------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="base_lr0.001_d500000")
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--build", default=str(INT / "patch"))
    p.add_argument("--out", default=str(INT / "reported"))
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--world-seed", type=int, default=None)
    p.add_argument("--suffix", default="", help="run-file suffix (e.g. _n20 for a dry run)")
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    _env_panel(a)
    M, world = _world(a.eps)
    tag = eps_tag(a.eps); wt = world_tag(world, a)
    t0 = time.time()
    b = np.load(Path(a.build) / f"build_eps{tag}_{wt}.npz")
    runs = {}
    for s in a.seeds:
        f = Path(a.build) / f"run_{a.cell}_eps{tag}_{wt}_s{s}_ck{a.step}{a.suffix}{'_EXPLORATORY' if a.exploratory else ''}.npz"
        runs[s] = np.load(f)
        if str(runs[s]["build_sha256"]) != _sha(Path(a.build) / f"build_eps{tag}_{wt}.npz"):
            raise SystemExit(f"{f.name}: built from a different build file")
    sites = list(runs[a.seeds[0]]["sites"]); sources = list(runs[a.seeds[0]]["sources"])
    n = int(runs[a.seeds[0]]["n_ctx"]); m_q = b["p_true"].shape[1]
    for s in a.seeds:
        if list(runs[s]["sites"]) != sites or list(runs[s]["sources"]) != sources or int(runs[s]["n_ctx"]) != n:
            raise SystemExit("run files disagree on sites/sources/n")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"armC_{a.cell}_eps{tag}_{wt}_ck{a.step}{a.suffix}{'_EXPLORATORY' if a.exploratory else ''}.json"
    if dest.is_file() and not a.force:
        raise SystemExit(f"{dest} exists; written once")
    p_true = b["p_true"][:n].astype(np.float64)
    op = b["o_primary"][:n]; kp = b["k_primary"][:n]
    V = {"ord": np.stack([b["v_ord"][i, :, op[i]] for i in range(n)]).astype(np.float64),
         "atom": np.stack([b["v_atom"][i, :, kp[i]] for i in range(n)]).astype(np.float64),
         "abl": b["v_abl"][:n].astype(np.float64), "atomabl": b["v_atomabl"][:n].astype(np.float64)}
    src_ord = [f"ord:{int(o)}" for o in op]; src_atom = [f"atom:{int(k)}" for k in kp]

    # Delta lives in a compressed npz: every runs[s]["Delta"][...] re-reads the whole
    # 750 MB array (the 09-04 analysis timed out / OOM-killed on exactly that).  Read it
    # once per seed, keep only what the estimands need, then free it.
    DELTA: dict = {}
    for s in a.seeds:
        Dfull = runs[s]["Delta"]                                   # (n, sites, sources, m_q, B) float32
        for site in sites:
            si = sites.index(site)
            DELTA[(site, "ord", s)] = np.stack([Dfull[i, si, sources.index(src_ord[i])] for i in range(n)])
            DELTA[(site, "atom", s)] = np.stack([Dfull[i, si, sources.index(src_atom[i])] for i in range(n)])
            for fixed in ("ordmean:0", "atommean:0", "res:0", "gau:0"):
                DELTA[(site, fixed, s)] = Dfull[:n, si, sources.index(fixed)]
        DELTA[("INPUT", "family", s)] = Dfull[:n, 0]                # all sources at INPUT, for C1
        del Dfull
    MEAN: dict = {}

    def Delta_of(site, src_per_ctx, unit):
        """(n, m_q, B) movement for a site and a per-context source name; unit 'mean' or a seed."""
        key = "ord" if src_per_ctx is src_ord else ("atom" if src_per_ctx is src_atom else src_per_ctx)
        seeds = a.seeds if unit == "mean" else [unit]
        ck = (site, key, unit)
        if ck not in MEAN:
            MEAN[ck] = np.mean([DELTA[(site, key, s)].astype(np.float64) for s in seeds], axis=0)
        return MEAN[ck]

    def coefs(D, dirs, label):
        """pooled coefficients of D on dirs (names), point + bootstrap over contexts; per-context sums cached."""
        Gs, rs, dd = gram_rhs_all(p_true, D, [V[x] for x in dirs])
        pt, r2, off = solve(Gs, rs, dd)
        bs = boot(lambda idx: solve(Gs[idx], rs[idx], dd[idx])[0], n, label)
        return {"coef": pt.tolist(), "ci": [ci(bs[:, j]) for j in range(len(dirs))], "r2": r2, "off_span": off, "_bs": bs, "_sums": (Gs, rs, dd)}

    results = {}
    for unit in ["mean"] + list(a.seeds):
        U = {}
        lab = f"armC:{a.cell}:{a.eps}:{a.step}:{wt}:{unit}"
        # INPUT swaps on the primary partners
        D_in_ord = Delta_of("INPUT", src_ord, unit); D_in_atom = Delta_of("INPUT", src_atom, unit)
        g_in_ord = coefs(D_in_ord, ["ord", "atom"], f"{lab}:INPUT:ord"); g_in_atom = coefs(D_in_atom, ["ord", "atom"], f"{lab}:INPUT:atom")
        Phi_ord, Phi_atom = g_in_ord["coef"][0], g_in_atom["coef"][1]
        # exact leaks (model-blind): the exact swap movements on the same directions
        Dx_ord = np.stack([b["D_ord"][i, :, op[i]] for i in range(n)]).astype(np.float64)
        Dx_atom = np.stack([b["D_atom"][i, :, kp[i]] for i in range(n)]).astype(np.float64)
        lam_ord = coefs(Dx_ord, ["ord", "atom"], f"{lab}:exact:ord"); lam_atom = coefs(Dx_atom, ["ord", "atom"], f"{lab}:exact:atom")
        # specificity in norm units: ||(gamma_atom(INPUT,B) - lambda_atom^ord) v_atom|| <= 0.5 ||gamma_ord v_ord||
        n_atom = np.sqrt(np.sum([wip(p_true[i], V["atom"][i], V["atom"][i]) for i in range(n)])); n_ord = np.sqrt(np.sum([wip(p_true[i], V["ord"][i], V["ord"][i]) for i in range(n)]))
        spec_ratio = abs(g_in_ord["coef"][1] - lam_ord["coef"][1]) * n_atom / max(abs(Phi_ord) * n_ord, 1e-300)
        U["INPUT"] = {"Phi_ord": Phi_ord, "Phi_ord_ci": g_in_ord["ci"][0], "gamma_atom_on_ord_swap": g_in_ord["coef"][1], "r2_ord": g_in_ord["r2"], "off_span_ord": g_in_ord["off_span"],
                      "Phi_atom": Phi_atom, "Phi_atom_ci": g_in_atom["ci"][1], "gamma_ord_on_atom_swap": g_in_atom["coef"][0], "r2_atom": g_in_atom["r2"],
                      "lambda_atom_of_ord_swap": lam_ord["coef"][1], "lambda_ord_of_atom_swap": lam_atom["coef"][0],
                      "specificity_ratio": float(spec_ratio), "specificity_ok": bool(spec_ratio <= TH["spec"])}
        phi_ok = g_in_ord["ci"][0][0] > 0 or g_in_ord["ci"][0][1] < 0
        phi_atom_ok = g_in_atom["ci"][1][0] > 0 or g_in_atom["ci"][1][1] < 0
        # per-site shares rho and the temperature-matched null
        rho = {}
        lp_model = np.mean([runs[s]["log_p_model"][:n].astype(np.float64) for s in (a.seeds if unit == "mean" else [unit])], axis=0)
        for site in sites[1:]:
            for axis, srcs, gidx, Phi, ok, bs_in in (("ord", src_ord, 0, Phi_ord, phi_ok, g_in_ord["_bs"][:, 0]), ("atom", src_atom, 1, Phi_atom, phi_atom_ok, g_in_atom["_bs"][:, 1])):
                D = Delta_of(site, srcs, unit)
                g = coefs(D, ["ord", "atom"], f"{lab}:{site}:{axis}")
                # temperature-matched null: re-temper the model's predictive to the patched predictive's mean entropy
                lpp = lp_model + D; Hp = -(np.exp(lpp) * lpp).sum(-1).mean(-1)
                DT = temper_all(lp_model, Hp) - lp_model
                gT = coefs(DT, ["ord", "atom"], f"{lab}:{site}:{axis}:T")
                diffT = g["_bs"][:, gidx] - gT["_bs"][:, gidx]
                r = {"gamma": g["coef"][gidx], "gamma_ci": g["ci"][gidx], "r2": g["r2"], "gamma_T": gT["coef"][gidx],
                     "gamma_minus_T_ci": ci(diffT), "gamma_minus_T_excludes_0": bool(ci(diffT)[0] > 0 or ci(diffT)[1] < 0)}
                if ok:
                    rb = g["_bs"][:, gidx] / bs_in
                    r.update({"rho": g["coef"][gidx] / Phi, "rho_ci": ci(rb), "rho_evaluable": bool(ci(rb)[1] - ci(rb)[0] <= TH["rho_ci_width"])})
                else:
                    r.update({"rho": None, "rho_ci": None, "rho_evaluable": False})
                rho[(site, axis)] = r
        # necessity: ordmean source on (v_abl, v_atom); nu~ = nu(site)/nu(cut)
        nu = {}
        for site in sites[1:]:
            g = coefs(Delta_of(site, "ordmean:0", unit), ["abl", "atom"], f"{lab}:{site}:ordmean")
            nu[site] = {"nu_ord": g["coef"][0], "ci": g["ci"][0], "r2": g["r2"], "_bs": g["_bs"][:, 0]}
        nu_cut = nu["cut"]["nu_ord"]
        for site in nu:
            nu[site]["nu_tilde"] = nu[site]["nu_ord"] / nu_cut if abs(nu_cut) > 1e-12 else None
            nu[site]["nu_tilde_ci"] = ci(nu[site]["_bs"] / nu["cut"]["_bs"]) if abs(nu_cut) > 1e-12 else None
        # ---- C2: layer-1 cut shares
        def sh(site, axis):
            return rho[(site, axis)]
        C2 = {}
        for axis in ("ord", "atom"):
            sc, sq, st = sh("cut_c", axis), sh("cut_q", axis), sh("cut_t", axis)
            if sc["rho"] is None:
                C2[axis] = {"verdict": "NOT_EVALUABLE"}; continue
            s_sum = (sq["rho"] or 0) + (st["rho"] or 0)
            if sc["rho"] >= TH["rows"] and sc["rho_ci"][0] >= TH["rows_ci"]:
                v = "ROWS"
            elif s_sum >= TH["preagg"]:
                v = "PRE-AGGREGATED"
            else:
                v = "MIXED"
            C2[axis] = {"s_c": sc["rho"], "s_c_ci": sc["rho_ci"], "s_q": sq["rho"], "s_t": st["rho"], "iota": 1 - sc["rho"] - s_sum, "verdict": v}
        if C2["ord"].get("s_c") is not None and C2["atom"].get("s_c") is not None:
            d_sc = C2["ord"]["s_c"] - C2["atom"]["s_c"]
            bs_d = rho[("cut_c", "ord")]["_bs"] if False else None
            # paired CI of s_c(ord) - s_c(atom): recompute from the stored bootstrap streams (same resample indices by label? no — different labels)
            # -> use a dedicated paired bootstrap over contexts on the per-context sums
            Gs_o, rs_o, _ = coefs(Delta_of("cut_c", src_ord, unit), ["ord", "atom"], f"{lab}:pair:cut_c:ord")["_sums"]
            Gs_a, rs_a, _ = coefs(Delta_of("cut_c", src_atom, unit), ["ord", "atom"], f"{lab}:pair:cut_c:atom")["_sums"]
            Gi_o, ri_o, _ = g_in_ord["_sums"]; Gi_a, ri_a, _ = g_in_atom["_sums"]
            def paired(idx):
                so = np.linalg.solve(Gs_o[idx].sum(0), rs_o[idx].sum(0))[0] / np.linalg.solve(Gi_o[idx].sum(0), ri_o[idx].sum(0))[0]
                sa = np.linalg.solve(Gs_a[idx].sum(0), rs_a[idx].sum(0))[1] / np.linalg.solve(Gi_a[idx].sum(0), ri_a[idx].sum(0))[1]
                return so - sa
            bsd = boot(paired, n, f"{lab}:pair:s_c")
            tnull = rho[("cut_c", "ord")]["gamma_minus_T_excludes_0"]
            if d_sc >= TH["contrast"] and ci(bsd)[0] > 0 and tnull:
                cv = "LATER-THAN-ATOM"
            elif d_sc <= -TH["contrast"] and ci(bsd)[1] < 0 and tnull:
                cv = "EARLIER-THAN-ATOM"
            else:
                cv = "SAME-ROUTE"
            C2["contrast"] = {"s_c_ord_minus_atom": float(d_sc), "ci": ci(bsd), "gamma_minus_T_excludes_0": tnull, "verdict": cv}
        # ---- C3: last-layer heads and routing
        head_sites = [s for s in sites if "h" in s and "@q" in s]
        heads = {}
        n_ded = 0
        for hs in head_sites:
            ro, ra = rho[(hs, "ord")], rho[(hs, "atom")]
            nt = nu[hs]["nu_tilde"]
            dedicated = (ro["rho"] is not None and ra["rho"] is not None and nt is not None and ro["rho"] >= TH["head_rho"] and nt >= TH["head_nu"]
                         and (ro["rho"] - ra["rho"]) >= TH["head_diff"] and ro["gamma_minus_T_excludes_0"])
            heads[hs] = {"rho_ord": ro["rho"], "rho_atom": ra["rho"], "nu_tilde": nt, "dedicated": bool(dedicated)}
            n_ded += int(dedicated)
        val_site = [s for s in sites if s.endswith("val")][0]; pat_site = [s for s in sites if s.endswith("pat")][0]
        routing = {}
        for axis in ("ord", "atom"):
            rv, rp = rho[(val_site, axis)]["rho"], rho[(pat_site, axis)]["rho"]
            if rv is None or rp is None:
                routing[axis] = "NOT_EVALUABLE"
            elif rv >= TH["content_val"] and rp <= TH["content_pat"]:
                routing[axis] = "CONTENT-ROUTED"
            elif rp >= TH["pattern"]:
                routing[axis] = "PATTERN-DEPENDENT"
            else:
                routing[axis] = "MIXED"
        C3 = {"heads": heads, "n_dedicated_order_heads": n_ded, "verdict": "DEDICATED-ORDER-HEAD" if n_ded > 0 else "DISTRIBUTED", "routing": routing}
        # ---- C1: family regression of single-direction coefficients on log ||v||^2 with a family indicator
        ys, xs, fam = [], [], []
        FAM = np.mean([DELTA[("INPUT", "family", s)].astype(np.float64) for s in (a.seeds if unit == "mean" else [unit])], axis=0)
        for i in range(n):
            for o in np.where(b["visible"][i])[0]:
                v = b["v_ord"][i, :, o].astype(np.float64); D = FAM[i, sources.index(f"ord:{int(o)}")]
                vv = wip(p_true[i], v, v)
                if vv > 1e-12:
                    ys.append(wip(p_true[i], D, v) / vv); xs.append(np.log(vv)); fam.append(1.0)
            for k in range(int(b["K"])):
                if k == int(b["k"][i]):
                    continue
                v = b["v_atom"][i, :, k].astype(np.float64); D = FAM[i, sources.index(f"atom:{k}")]
                vv = wip(p_true[i], v, v)
                if vv > 1e-12:
                    ys.append(wip(p_true[i], D, v) / vv); xs.append(np.log(vv)); fam.append(0.0)
        X = np.column_stack([np.ones(len(ys)), xs, fam]); y = np.array(ys)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        C1 = {"n_swaps": len(ys), "coef_intercept_logn2_family": beta.tolist(), "family_effect_ord_minus_atom_at_matched_norm": float(beta[2]),
              "note": "single-direction coefficient per swap; CI not formed here (swaps are not independent within a context)"}
        U.update({"rho": {f"{k[0]}:{k[1]}": {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rho.items()},
                  "nu": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in nu.items()},
                  "C1": C1, "C2": C2, "C3": C3})
        results[str(unit)] = U
        print(f"[armC] {a.cell} eps {a.eps} unit {unit}: Phi_ord {Phi_ord:+.3f} {g_in_ord['ci'][0]} Phi_atom {Phi_atom:+.3f} | "
              f"C2 ord {C2['ord'].get('verdict')} (s_c {C2['ord'].get('s_c', float('nan')):.2f}) atom {C2['atom'].get('verdict')} (s_c {C2['atom'].get('s_c', float('nan')):.2f}) "
              f"contrast {C2.get('contrast', {}).get('verdict')} | C3 {C3['verdict']} n_ded {n_ded} routing {routing} | spec {spec_ratio:.2f} ({time.time() - t0:.0f}s)", flush=True)
    doc = {"status": "REPORTED — never gated" + (" — EXPLORATORY" if a.exploratory or a.suffix else ""), "cell": a.cell, "eps": a.eps, "step": a.step, "world": wt,
           "seeds": a.seeds, "n_ctx": n, "sites": sites, "sources": sources, "thresholds": TH, "results": results,
           "decision_scope": "this eps; the arm verdict requires identical C2/C3 verdicts at eps .5/.75/1 (else MIXED); per-seed verdict counts in results[seed]",
           "meta": {"prespec_sha256": _sha(INT / "PRESPEC_internal.md"), "build_sha256": _sha(Path(a.build) / f"build_eps{tag}_{wt}.npz"),
                    "runs_sha256": {str(s): str(runs[s]["ckpt_sha256"]) for s in a.seeds}, "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED, "wall_s": time.time() - t0}}
    dest.write_text(json.dumps(doc, indent=1, allow_nan=False, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)))
    print(f"[armC] -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
