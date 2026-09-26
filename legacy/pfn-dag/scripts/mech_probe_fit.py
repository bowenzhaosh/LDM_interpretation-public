"""Arm R (PRESPEC_internal §1), stage 2: fresh readout heads, T1, decision. REPORTED, never gated.

For one (cell, eps, step) and model seeds S, and n_refits TRAINING DRAWS ts = 0..n-1 (each a fresh
25k-context set with exact soft targets, mech_probe_acts build; PRESPEC §1 "5 train seeds"):
  per (seed s, draw ts): heads trained with cross-entropy on the EXACT soft target q_full:
      P0(last.out)  MLP d_model -> 256 -> 100     P0(logits)  MLP 100 -> 256 -> 100
      P0lin(last.out)  linear d_model -> 100      each x the ck0 twin's features
      features z-scored on train; 10 % held out for early stopping; head seed 1000*ts + s.
  On the registered panel: S_i(X) = mean_q sum_b p_true log p_X; regret^X_i = S_i(full) - S_i(X).
  PRIMARY unit = per-context mean over seeds of S_i(X) (the registered model column's convention).
  b^X = mech_gates._ols slope on the registered G_i; Delta^X = b^model - b^X, paired bootstrap
  (2000 resamples, mech_gates._rng(label)); T1 = Delta^{P0}(last.out) - Delta^{P0}(logits)
  = b^{P0}(logits) - b^{P0}(last.out).  Atom side: joint OLS regret = a + b G + b' G^{Pi*}.
  Decision THIS EPS (the arm verdict additionally needs eps .5/.75/1 sign agreement, else MIXED):
    READOUT-LIMITED iff T1 >= 0.10 with 95 % CI excluding 0 in every draw, every model seed's own
        T1 > 0, (i) |b^{P0lin} - b^model| <= 0.05, (ii) b^{P0,ck0}(last.out) >= b^model - 0.05,
        |Delta b'| <= 0.05, all draws;
    HEAD-SATURATED iff the upper 95 % CI of T1 < 0.05 in every draw and (i) holds;
    NOT SEPARABLE if (ii) fails; NOT_EVALUABLE if any SE(T1) > mech_gates.SE_MAX; else UNDECIDED.
  Refusals (nothing written): panel identity (mech_gates._check_panel), world identity of every input
  (K, O, d, atom-library sha), m_q, shuffled-target pipeline check b >= 0.95, output exists.
Reported beside, no decision: specificity linear probes at last.out(q) for p(o|D) and the Pi* group
marginal (trained on the 25k set, never the panel; f_ord - f_atom* with CI); the descriptive hybrid
shrink map (through-origin slope of the p(k|o,D)*p_hat(o|D) hybrid on G > 0, beside the mu ladder).
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
from mech_component_control import joint_fit  # noqa: E402
from mech_probe_acts import INT, _env_panel, _world, forward_sites, load_model, world_tag  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

TAU_EFF, TAU_NULL, TOL_LIN, TOL_CK0, TOL_ATOM, SHUFFLE_MIN = 0.10, 0.05, 0.05, 0.05, 0.05, 0.95
MU_LADDER = {"mu": [0.33, 0.5, 0.67, 0.8, 0.9], "b_ols_eps0p75": [0.610, 0.444, 0.290, 0.178, 0.091],
             "source": ".claude/internal_design/probes/readout_hybrid_eps0p75.json (exploratory)"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hybrid_weights(w_lo: np.ndarray, p_hat_o: np.ndarray, K: int, O: int) -> np.ndarray:
    """w(k,o) = p(k | o, D) * p_hat(o), the formula of corrected_oracle.ablated_weights('order_ablated')
    with p_hat in place of the order prior (bitwise equal to it when p_hat = prior_order)."""
    W = np.asarray(w_lo, np.float64).reshape(K, O)
    w_o = W.sum(0)
    w_k_given_o = W / np.maximum(w_o[None, :], 1e-300)
    out = w_k_given_o * np.asarray(p_hat_o, np.float64)[None, :]
    return out.ravel() / out.sum()


# ------------------------------------------------------------------ heads
def train_head(X: np.ndarray, T: np.ndarray, kind: str, seed: int, epochs: int = 60, patience: int = 5,
               batch: int = 1024, lr: float | None = None):
    """kind: 'mlp' (in -> 256 -> 100) or 'lin' (in -> out); CE against soft targets T; z-scored features."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if lr is None:
        lr = 1e-2 if kind == "lin" else 3e-3      # synthetic check 09-04: lin reaches H(target) at 1e-2; mlp within 0.02 at 3e-3
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = ((X - mu) / sd).astype(np.float32)
    n = len(Xz); rng = np.random.default_rng(seed)
    perm = rng.permutation(n); n_val = n // 10
    iv, it = perm[:n_val], perm[n_val:]
    xt, tt = torch.tensor(Xz[it]), torch.tensor(T[it])
    xv, tv = torch.tensor(Xz[iv]), torch.tensor(T[iv])
    din, dout = X.shape[1], T.shape[1]
    net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, dout)) if kind == "mlp" else nn.Linear(din, dout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(xt))
        for s in range(0, len(xt), batch):
            b = idx[s:s + batch]
            loss = -(tt[b] * torch.log_softmax(net(xt[b]), -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(-(tv * torch.log_softmax(net(xv), -1)).sum(-1).mean())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()

    def predict(Xn: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            lp = torch.log_softmax(net(torch.tensor(((Xn - mu) / sd).astype(np.float32))), -1)
        return np.exp(lp.numpy().astype(np.float64))
    return predict, {"val_ce": best, "epochs": ep + 1, "kind": kind, "seed": seed}


def S_of(p: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    return (p_true * np.log(np.maximum(p, 1e-300))).sum(-1).mean(1)


def boot_diff(reg_a, reg_b, G, label, n_boot=MG.N_BOOT):
    """paired bootstrap of b(reg_a) - b(reg_b) on shared contexts -> (point, ci95, se)."""
    rng = MG._rng(label); n = len(G)
    point = MG._ols(reg_a, G)[0] - MG._ols(reg_b, G)[0]
    bs = np.empty(n_boot)
    for t in range(n_boot):
        i = MG._boot_idx(n, rng)
        bs[t] = MG._ols(reg_a[i], G[i])[0] - MG._ols(reg_b[i], G[i])[0]
    return point, [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], float(bs.std(ddof=1))


def boot_se(reg, G, label, n_boot=MG.N_BOOT):
    rng = MG._rng(label); n = len(G)
    return float(np.std([MG._ols(reg[i], G[i])[0] for i in (MG._boot_idx(n, rng) for _ in range(n_boot))], ddof=1))


def kl_fraction_rows(probe_p, target):
    """per-row 1 - (CE(probe) - H) / (CE(uniform) - H); returns the row vector (for bootstrapping)."""
    u = np.full_like(target, 1.0 / target.shape[1])
    H = -(target * np.log(np.maximum(target, 1e-300))).sum(-1)
    ce_p = -(target * np.log(np.maximum(probe_p, 1e-300))).sum(-1)
    ce_u = -(target * np.log(np.maximum(u, 1e-300))).sum(-1)
    return 1 - (ce_p - H) / np.maximum(ce_u - H, 1e-12)


def check_world(name: str, zf, world, a, eps: float) -> None:
    sig = hashlib.sha256(np.ascontiguousarray(world.sigmas, dtype=np.float64).tobytes()).hexdigest()
    for key, want in (("K", world.K), ("O", world.O), ("d", world.d)):
        if key in zf.files and int(zf[key]) != want:
            raise SystemExit(f"{name}: {key}={int(zf[key])} != this world's {want}")
    if "eps" in zf.files and abs(float(zf["eps"]) - eps) > 1e-9:
        raise SystemExit(f"{name}: eps {float(zf['eps'])} != {eps}")
    if "sigmas_sha256" in zf.files and str(zf["sigmas_sha256"]) != sig:
        raise SystemExit(f"{name}: atom library differs from this world")
    ws = a.world_seed if a.world_seed is not None else -1
    if "world_seed" in zf.files and int(zf["world_seed"]) != ws:
        raise SystemExit(f"{name}: world_seed {int(zf['world_seed'])} != {ws}")


# ------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="base_lr0.001_d500000")
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--scored", default=str(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000"))
    p.add_argument("--nets", default=str(ROOT / "campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets"))
    p.add_argument("--coarse", default=str(ROOT / "campaigns/mech_ext_20260902/coarse"))
    p.add_argument("--component-control", default=str(ROOT / "campaigns/mech_ext_20260902/reported/component_control.json"))
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=str(INT / "reported"))
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--world-seed", type=int, default=None)
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing output (journal it)")
    a = p.parse_args()
    _env_panel(a)
    M, world = _world(a.eps)
    tag = eps_tag(a.eps); wt = world_tag(world, a)
    K, O, d = world.K, world.O, world.d; target = d - 1
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"armR_{a.cell}_eps{tag}_{wt}_ck{a.step}{'_EXPLORATORY' if a.exploratory else ''}.json"
    if dest.is_file() and not a.force:
        raise SystemExit(f"{dest} exists; written once (use --force and journal)")
    # ---- registered scored cell (panel identity via the locked checker)
    scored = Path(a.scored) / f"predgain_eps{tag}_ck{a.step}.npz"
    z = np.load(scored)
    if not a.exploratory:
        MG._check_panel(z, scored, [int(x) for x in z["seeds"]], n_rows_expected=int(z["n_rows"]), scale_expected=None,
                        panel={"panel_seed": a.panel_seed, "split_seed": a.split_seed, "n_per_half": a.n_per_half})
    names = list(z["names"]); Sr = z["S"].mean(1); m_q = int(z["m_q"])
    if m_q != MG.M_Q:
        raise SystemExit(f"{scored}: m_q {m_q} != {MG.M_Q}")
    S_full, S_abl = Sr[:, names.index("full")], Sr[:, names.index("abl")]
    G = S_full - S_abl
    model_cols = [names.index(f"model_s{s}") for s in a.seeds]
    reg_model = S_full - Sr[:, model_cols].mean(1)
    b_model = MG._ols(reg_model, G)[0]
    se_b_model = boot_se(reg_model, G, f"armR:{a.cell}:{a.eps}:{a.step}:{wt}:b_model")
    # ---- size-matched atom component (coarse pass + the registered Pi*)
    zc = np.load(Path(a.coarse) / f"coarse_eps{tag}.npz")
    if float(np.max(np.abs(zc["S_mean"][:, 0] - S_full))) > 1e-9:
        raise SystemExit("coarse pass and scored cell disagree on S(full)")
    cc = json.load(open(a.component_control))
    pistar = next((r["joint"]["pistar"] for r in cc["results"]
                   if r["cell"] == a.cell and r["unit"] == "mean" and abs(r["eps"] - a.eps) < 1e-9 and r["step"] == a.step), None)
    if pistar is None:
        raise SystemExit(f"no registered Pi* for {a.cell} eps {a.eps} in {a.component_control}")
    cn = list(zc["names"]); G_pi = zc["S_mean"][:, 0] - zc["S_mean"][:, cn.index(pistar)]
    b_model_joint = joint_fit(reg_model, G, G_pi)
    rgs = [int(c) for c in pistar.split(":")[1]]; groups = sorted(set(rgs))
    # ---- activations (registered panel) per seed, with world checks
    acts = {}
    for s in a.seeds:
        f = INT / "probes" / f"acts_{a.cell}_eps{tag}_{wt}_s{s}_ck{a.step}{'_EXPLORATORY' if a.exploratory else ''}.npz"
        acts[s] = np.load(f); check_world(f.name, acts[s], world, a, a.eps)
        if str(acts[s]["scored_sha256"]) != _sha(scored) or int(acts[s]["m_q"]) != m_q:
            raise SystemExit(f"{f.name} does not match the scored cell")
        if acts[s]["last_out_ck0"].size == 0 and not a.exploratory:
            raise SystemExit(f"{f.name}: no ck0 twin activations (required, PRESPEC §1 (ii))")
    rows = acts[a.seeds[0]]["rows"]; ks = list(acts[a.seeds[0]]["k"]); os_ = list(acts[a.seeds[0]]["o"])
    for s in a.seeds:
        if list(acts[s]["k"]) != ks:
            raise SystemExit("acts files disagree on the panel latents")
    # ---- panel exact quantities: p_true per query, p(o|D), p(k|D), group marginal, operators for the hybrid
    from pfn_dag_verify.corrected_oracle import exact_joint_posterior, obs_query_operator
    from mech_interventions import half_b_with_latents
    ctxs, ks2, os2 = half_b_with_latents(world)
    if list(ks2) != ks or list(os2) != os_:
        raise SystemExit("panel replay differs from the acts files")
    n = len(ctxs)
    p_true = np.zeros((n, m_q, 100)); w_panel = np.zeros((n, K * O)); ops = [[None] * m_q for _ in range(n)]
    for i in range(n):
        w_panel[i] = exact_joint_posterior(world, ctxs[i])["w_lo"]
        w_true = np.zeros(K * O); w_true[ks[i] * O + os_[i]] = 1.0
        for q in range(m_q):
            ops[i][q] = obs_query_operator(world, rows[i, q, :target], target)
            p_true[i, q] = ops[i][q].predictive(w_true)
    Wp = w_panel.reshape(n, K, O); p_o_panel = Wp.sum(1); p_k_panel = Wp.sum(2)      # layout lo = k*O + o: sum over K -> p(o), over O -> p(k)
    p_g_panel = np.stack([p_k_panel[:, [i for i in range(K) if rgs[i] == g]].sum(1) for g in groups], 1)
    # ---- training draws (each with world checks)
    draws = {}
    for ts in a.train_seeds:
        f = INT / "probes" / f"train_eps{tag}_{wt}_ts{ts}.npz"
        tr = np.load(f); check_world(f.name, tr, world, a, a.eps)
        if int(tr["m_q"]) != m_q or int(tr["n_rows"]) != int(rows.shape[1] if False else z["n_rows"]):
            raise SystemExit(f"{f.name}: m_q/n_rows differ from the panel")
        draws[ts] = tr
    # ---- per (seed, draw): heads -> panel scores
    S_sum: dict = {}; per_seed: dict = {s: {} for s in a.seeds}; spec_rows: dict = {}
    hyb_S = {s: {} for s in a.seeds}
    for s in a.seeds:
        ck = Path(a.nets) / f"eps{tag}" / f"{acts[s]['scale']}_s{s}_ck{a.step}.pt"
        if _sha(ck) != str(acts[s]["ckpt_sha256"]):
            raise SystemExit(f"{ck}: sha differs from the acts file")
        m = load_model(str(acts[s]["scale"]), d, ck)
        m0 = load_model(str(acts[s]["scale"]), d, Path(str(acts[s]["ck0"]))) if acts[s]["last_out_ck0"].size else None
        panel_feats = {"last.out": acts[s]["last_out"].reshape(-1, acts[s]["last_out"].shape[-1]), "logits": acts[s]["logits"].reshape(-1, 100)}
        if m0 is not None:
            panel_feats["last.out.ck0"] = acts[s]["last_out_ck0"].reshape(-1, acts[s]["last_out_ck0"].shape[-1])
            panel_feats["logits.ck0"] = acts[s]["logits_ck0"].reshape(-1, 100)
        for ts, tr in draws.items():
            T = tr["q_full"].reshape(-1, 100).astype(np.float32)
            Htr, Ltr, _ = forward_sites(m, tr["ctx"], tr["rows"])
            feats = {"last.out": Htr.reshape(-1, Htr.shape[-1]), "logits": Ltr.reshape(-1, 100)}
            if m0 is not None:
                H0, L0, _ = forward_sites(m0, tr["ctx"], tr["rows"])
                feats["last.out.ck0"] = H0.reshape(-1, H0.shape[-1]); feats["logits.ck0"] = L0.reshape(-1, 100)
            hseed = 1000 * ts + s
            for site, kind in (("last.out", "mlp"), ("logits", "mlp"), ("last.out", "lin"),
                               ("last.out.ck0", "mlp"), ("logits.ck0", "mlp"), ("last.out.ck0", "lin")):
                if site not in feats:
                    continue
                pred, info = train_head(feats[site], T, kind, seed=hseed)
                Sx = S_of(pred(panel_feats[site]).reshape(n, m_q, 100), p_true)
                S_sum[(site, kind, ts)] = S_sum.get((site, kind, ts), 0) + Sx
                per_seed[s][f"b:{site}:{kind}:ts{ts}"] = float(MG._ols(S_full - Sx, G)[0])
                per_seed[s][f"head:{site}:{kind}:ts{ts}"] = info
            # shuffled-target pipeline check (draw 0 only), permutation from the arm's label namespace
            if ts == a.train_seeds[0]:
                perm = MG._rng(f"armR:{a.cell}:{a.eps}:{a.step}:{wt}:shuffle:s{s}").permutation(len(T))
                pred, _ = train_head(feats["last.out"], T[perm], "mlp", seed=hseed + 7)
                Sx = S_of(pred(panel_feats["last.out"]).reshape(n, m_q, 100), p_true)
                S_sum[("last.out", "shuffled", ts)] = S_sum.get(("last.out", "shuffled", ts), 0) + Sx
            # specificity probes: linear at last.out(q) rows, trained on the 25k set from its w_lo, scored on the panel
            if ts == a.train_seeds[0]:
                Wt = tr["w_lo"].reshape(len(tr["k"]), K, O); po_t = Wt.sum(1); pk_t = Wt.sum(2)
                pg_t = np.stack([pk_t[:, [i for i in range(K) if rgs[i] == g]].sum(1) for g in groups], 1)
                for name, tgt_t, tgt_p in (("order", po_t, p_o_panel), ("atom_pistar", pg_t, p_g_panel)):
                    Ttr = np.repeat(tgt_t, m_q, axis=0).astype(np.float32)          # per (context, query) row carries its context's target
                    for lab2, Xtr, Xp in (("trained", feats["last.out"], panel_feats["last.out"]),
                                          ("ck0", feats.get("last.out.ck0"), panel_feats.get("last.out.ck0"))):
                        if Xtr is None:
                            continue
                        pred, _ = train_head(Xtr, Ttr, "lin", seed=hseed + 11)
                        pp = pred(Xp).reshape(n, m_q, -1).mean(1)                     # context-level probe = mean over its query rows
                        spec_rows[(s, name, lab2)] = kl_fraction_rows(pp, tgt_p)
                        if name == "order" and lab2 == "trained":
                            # descriptive hybrid: w = p(k|o,D) * p_hat(o|D); its exact predictive on the panel
                            Sh = np.zeros((n, m_q))
                            for i in range(n):
                                wh = hybrid_weights(w_panel[i], pp[i], K, O)
                                for q in range(m_q):
                                    Sh[i, q] = float(np.sum(p_true[i, q] * np.log(np.maximum(ops[i][q].predictive(wh), 1e-300))))
                            hyb_S[s]["hybrid"] = Sh.mean(1)
        per_seed[s]["b_model"] = float(MG._ols(S_full - acts[s]["S_model"].mean(1), G)[0])
        print(f"[fit] seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    # ---- primary unit (per-context mean over seeds) and the statistics per draw
    S_h = {k: v / len(a.seeds) for k, v in S_sum.items()}
    lab = f"armR:{a.cell}:{a.eps}:{a.step}:{wt}"
    rows_out = []
    for ts in a.train_seeds:
        rP0h = S_full - S_h[("last.out", "mlp", ts)]; rP0l = S_full - S_h[("logits", "mlp", ts)]; rlin = S_full - S_h[("last.out", "lin", ts)]
        dH = boot_diff(reg_model, rP0h, G, f"{lab}:dH:ts{ts}"); dL = boot_diff(reg_model, rP0l, G, f"{lab}:dL:ts{ts}")
        t1 = boot_diff(rP0l, rP0h, G, f"{lab}:T1:ts{ts}")                     # = b^{P0}(logits) - b^{P0}(last.out)
        bj = joint_fit(rP0h, G, G_pi)
        row = {"train_seed": ts, "b_P0_lastout": float(MG._ols(rP0h, G)[0]), "b_P0_logits": float(MG._ols(rP0l, G)[0]),
               "b_P0lin_lastout": float(MG._ols(rlin, G)[0]), "delta_P0_lastout": dH[0], "delta_P0_lastout_ci": dH[1],
               "delta_P0_logits": dL[0], "delta_P0_logits_ci": dL[1], "T1": t1[0], "T1_ci": t1[1], "T1_se": t1[2],
               "b_prime_P0": float(bj[2]), "delta_bprime": float(b_model_joint[2] - bj[2]),
               "per_model_seed_T1": {str(s): per_seed[s][f"b:logits:mlp:ts{ts}"] - per_seed[s][f"b:last.out:mlp:ts{ts}"] for s in a.seeds}}
        for site, kind, key in (("last.out.ck0", "mlp", "b_P0_ck0_lastout"), ("logits.ck0", "mlp", "b_P0_ck0_logits"), ("last.out.ck0", "lin", "b_P0lin_ck0_lastout")):
            if (site, kind, ts) in S_h:
                row[key] = float(MG._ols(S_full - S_h[(site, kind, ts)], G)[0])
        rows_out.append(row)
    b_shuf = float(MG._ols(S_full - S_h[("last.out", "shuffled", a.train_seeds[0])], G)[0])
    if b_shuf < SHUFFLE_MIN:
        raise SystemExit(f"REFUSING: shuffled-target head gives b {b_shuf:.3f} < {SHUFFLE_MIN} (pipeline leaks target information)")
    # ---- controls, evaluability, decision for THIS eps
    lin_ok = all(abs(r["b_P0lin_lastout"] - b_model) <= TOL_LIN for r in rows_out)
    ck0_ok = all(("b_P0_ck0_lastout" in r) and r["b_P0_ck0_lastout"] >= b_model - TOL_CK0 for r in rows_out)
    atom_ok = all(abs(r["delta_bprime"]) <= TOL_ATOM for r in rows_out)
    seeds_pos = all(v > 0 for r in rows_out for v in r["per_model_seed_T1"].values())     # the registered sign of T1
    evaluable = all(r["T1_se"] <= MG.SE_MAX for r in rows_out) and se_b_model <= MG.SE_MAX
    rl = all(r["T1"] >= TAU_EFF and r["T1_ci"][0] > 0 for r in rows_out) and seeds_pos and lin_ok and ck0_ok and atom_ok
    hs = all(r["T1_ci"][1] < TAU_NULL for r in rows_out) and lin_ok
    if not evaluable:
        verdict = "NOT_EVALUABLE"
    elif rl:
        verdict = "READOUT-LIMITED"
    elif hs:
        verdict = "HEAD-SATURATED"
    elif not ck0_ok:
        verdict = "NOT SEPARABLE"
    else:
        verdict = "UNDECIDED"
    # ---- specificity: f_ord - f_atom* (trained - ck0), pooled over seeds, context bootstrap
    spec = {}
    for s in a.seeds:
        d_ord = spec_rows.get((s, "order", "trained")); d_atm = spec_rows.get((s, "atom_pistar", "trained"))
        c_ord = spec_rows.get((s, "order", "ck0")); c_atm = spec_rows.get((s, "atom_pistar", "ck0"))
        if d_ord is None or d_atm is None:
            continue
        f_ord = float(d_ord.mean() - (c_ord.mean() if c_ord is not None else 0)); f_atm = float(d_atm.mean() - (c_atm.mean() if c_atm is not None else 0))
        diff_rows = (d_ord - (c_ord if c_ord is not None else 0)) - (d_atm - (c_atm if c_atm is not None else 0))
        rng = MG._rng(f"{lab}:spec:s{s}")
        bs = np.array([diff_rows[MG._boot_idx(n, rng)].mean() for _ in range(MG.N_BOOT)])
        spec[str(s)] = {"f_ord_trained": float(d_ord.mean()), "f_atom_trained": float(d_atm.mean()),
                        "f_ord_ck0": float(c_ord.mean()) if c_ord is not None else None, "f_atom_ck0": float(c_atm.mean()) if c_atm is not None else None,
                        "f_ord_minus_f_atom_trained_minus_ck0": float(diff_rows.mean()),
                        "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    hyb = {}
    for s in a.seeds:
        if "hybrid" in hyb_S[s]:
            reg_h = S_full - hyb_S[s]["hybrid"]; m_ = G > 0
            hyb[str(s)] = {"b_origin_Gpos": float((reg_h[m_] * G[m_]).sum() / (G[m_] ** 2).sum()), "b_ols": float(MG._ols(reg_h, G)[0])}
    res = {"status": "REPORTED — never gated" + (" — EXPLORATORY nets" if a.exploratory else ""),
           "meta": {"cell": a.cell, "eps": a.eps, "step": a.step, "seeds": a.seeds, "train_seeds": a.train_seeds, "world": wt,
                    "n_ctx": n, "m_q": m_q, "unit": "per-context mean over model seeds", "label_root": lab,
                    "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED, "prespec_sha256": _sha(INT / "PRESPEC_internal.md"),
                    "inputs_sha256": {"scored": _sha(scored), "coarse": _sha(Path(a.coarse) / f"coarse_eps{tag}.npz"),
                                      **{f"acts_s{s}": str(acts[s]["ckpt_sha256"]) for s in a.seeds},
                                      **{f"train_ts{ts}": _sha(INT / "probes" / f"train_eps{tag}_{wt}_ts{ts}.npz") for ts in a.train_seeds}},
                    "wall_s": time.time() - t0},
           "b_model": b_model, "se_b_model": se_b_model, "b_model_joint": {"a": float(b_model_joint[0]), "b": float(b_model_joint[1]), "b_prime": float(b_model_joint[2])},
           "pistar": pistar, "Gbar_order": float(G.mean()), "Gbar_pistar": float(G_pi.mean()),
           "draws": rows_out, "b_shuffled_target": b_shuf,
           "controls": {"linear_head_converges": lin_ok, "ck0_twin_not_better": ck0_ok, "atom_slope_unmoved": atom_ok,
                        "per_model_seed_T1_positive": seeds_pos, "evaluable": evaluable, "shuffled_ok": True,
                        "row_permutation_invariance": "established in .claude/internal_design/probes/feas_check.py (3.8e-6); not re-run here"},
           "thresholds": {"tau_eff": TAU_EFF, "tau_null": TAU_NULL, "tol_lin": TOL_LIN, "tol_ck0": TOL_CK0, "tol_atom": TOL_ATOM, "se_max": MG.SE_MAX, "shuffle_min": SHUFFLE_MIN},
           "verdict_this_eps": verdict,
           "decision_scope": "single eps; the arm verdict requires eps .5/.75/1 sign agreement of T1 on the decision cell (else MIXED); eps .75 is the decision eps",
           "specificity_linear_probes": spec, "hybrid_shrink_map": {"per_seed": hyb, "mu_ladder": MU_LADDER, "note": "descriptive; never compared to b_model"},
           "per_seed": {str(s): v for s, v in per_seed.items()}}
    dest.write_text(json.dumps(res, indent=1, allow_nan=False))
    mT1 = np.mean([r["T1"] for r in rows_out]); r0 = rows_out[0]
    print(f"[armR] {a.cell} eps {a.eps} {wt}: b_model {b_model:.3f}±{se_b_model:.3f} | b_P0(last.out) {np.mean([r['b_P0_lastout'] for r in rows_out]):.3f} | "
          f"b_P0(logits) {np.mean([r['b_P0_logits'] for r in rows_out]):.3f} | b_lin {np.mean([r['b_P0lin_lastout'] for r in rows_out]):.3f} | "
          f"ck0 {np.mean([r.get('b_P0_ck0_lastout', np.nan) for r in rows_out]):.3f} | T1 {mT1:+.3f} (ts0 CI {r0['T1_ci'][0]:+.3f},{r0['T1_ci'][1]:+.3f}) | "
          f"d_b' {np.mean([r['delta_bprime'] for r in rows_out]):+.3f} | shuffled {b_shuf:.3f} -> {verdict} ({time.time() - t0:.0f}s) -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
