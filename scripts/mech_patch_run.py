"""Arm C (PRESPEC_internal §2), stage 2 — patched forwards. REPORTED, never gated.

A transparent re-implementation of PFN.forward (nn.TransformerEncoder of post-LN layers, ReLU, dropout 0,
no final norm; corrected_models.py) that exposes, per layer, the residual stream at every position, the
per-head attention outputs before out_proj, the values, the attention pattern and the MLP output, and lets
any of them be replaced by the same quantity taken from a SOURCE context's own forward on the same query.

Sites (base: 2 layers):  INPUT (the whole context replaced),
  cut / cut_c / cut_q / cut_t   = output of layers[0] at all / row / query / null positions,
  A2h{h}@q  = last-layer head h output at the query position,  A2val = last-layer values at all positions,
  A2pat     = last-layer attention pattern at the query row,   M2q   = last-layer MLP output at the query.
Sources (from mech_patch_build): every visible order partner B(o'), every atom partner B'(k'), ordmean and
atommean (the site's activations averaged over the visible order family / all K atoms including A), resample,
gaussianised.  For each (site, source, query): Delta = log p_patch - log p_model  (100 bins, float32).

Identity refusals (per seed, before anything is written): transparent forward == PFN.predict_bin_probs to 1e-5;
the complete-cut patch from A itself == A (1e-6); the complete-cut patch from B == INPUT(B) (1e-6);
S(model) equals the registered column to 1e-4.  Output: patch/run_{cell}_eps{tag}_{world}_s{seed}_ck{step}.npz
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
from mech_probe_acts import INT, _env_panel, _world, load_model, world_tag  # noqa: E402

SOURCE_KINDS = ("ord", "atom", "ordmean", "atommean", "res", "gau")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ transparent forward (numpy, float64)
class Net:
    """Weights of a PFN as numpy arrays; a forward that records and optionally patches."""

    def __init__(self, m):
        import torch
        g = lambda t: t.detach().numpy().astype(np.float64)
        self.d_model = m.cfg.d_model; self.n_heads = m.cfg.n_heads
        self.W_pe, self.b_pe = g(m.point_embed.weight), g(m.point_embed.bias)
        self.W_qe, self.b_qe = g(m.query_embed.weight), g(m.query_embed.bias)
        self.tok = g(m.token_embed.weight)
        self.W_out, self.b_out = g(m.out_head.weight), g(m.out_head.bias)
        self.layers = []
        for L in m.transformer.layers:
            assert not L.norm_first and L.self_attn.batch_first
            self.layers.append(dict(
                W_in=g(L.self_attn.in_proj_weight), b_in=g(L.self_attn.in_proj_bias),
                W_o=g(L.self_attn.out_proj.weight), b_o=g(L.self_attn.out_proj.bias),
                W1=g(L.linear1.weight), b1=g(L.linear1.bias), W2=g(L.linear2.weight), b2=g(L.linear2.bias),
                n1w=g(L.norm1.weight), n1b=g(L.norm1.bias), n2w=g(L.norm2.weight), n2b=g(L.norm2.bias), eps=L.norm1.eps))
        assert m.transformer.norm is None

    @staticmethod
    def _ln(x, w, b, eps):
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    def embed(self, ctx, xq):
        """ctx (n_rows, d), xq (m_q, d-1) -> tokens (m_q, T, D): [null, rows, query_q]."""
        ce = ctx @ self.W_pe.T + self.b_pe                                  # (n_rows, D)
        qe = xq @ self.W_qe.T + self.b_qe                                   # (m_q, D)
        te = self.tok[2][None, :]
        T = np.concatenate([np.broadcast_to(te, (xq.shape[0], 1, self.d_model)),
                            np.broadcast_to(ce[None], (xq.shape[0],) + ce.shape), qe[:, None, :]], axis=1)
        return T

    def forward(self, x, patches=None, record=False):
        """x (m_q, T, D). patches: {(layer, kind, index, positions): array}; kinds: 'resid' (layer OUTPUT at
        positions), 'head' (per-head attn output at positions, index=h), 'val' (values at positions, all heads),
        'pat' (attention pattern rows at positions), 'mlp' (MLP output at positions). Returns logits (m_q, 100)
        and, if record, the dict of recorded activations."""
        patches = patches or {}; rec = {}
        H, D = self.n_heads, self.d_model; dh = D // H
        for li, L in enumerate(self.layers):
            qkv = x @ L["W_in"].T + L["b_in"]
            q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
            key = (li, "val", 0)
            for (pl, pk, pi, pos), val in patches.items():
                if pl == li and pk == "val":
                    v = v.copy(); v[:, pos, :] = val
            mq, T = x.shape[0], x.shape[1]
            qh = q.reshape(mq, T, H, dh).transpose(0, 2, 1, 3); kh = k.reshape(mq, T, H, dh).transpose(0, 2, 1, 3); vh = v.reshape(mq, T, H, dh).transpose(0, 2, 1, 3)
            sc = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(dh)
            sc = sc - sc.max(-1, keepdims=True); pat = np.exp(sc); pat /= pat.sum(-1, keepdims=True)   # (mq, H, T, T)
            for (pl, pk, pi, pos), val in patches.items():
                if pl == li and pk == "pat":
                    pat = pat.copy(); pat[:, :, pos, :] = val
            heads = pat @ vh                                                  # (mq, H, T, dh)
            for (pl, pk, pi, pos), val in patches.items():
                if pl == li and pk == "head":
                    heads = heads.copy(); heads[:, pi, pos, :] = val
            attn = heads.transpose(0, 2, 1, 3).reshape(mq, T, D)
            a_out = attn @ L["W_o"].T + L["b_o"]
            x1 = self._ln(x + a_out, L["n1w"], L["n1b"], L["eps"])
            mlp = np.maximum(x1 @ L["W1"].T + L["b1"], 0) @ L["W2"].T + L["b2"]
            for (pl, pk, pi, pos), val in patches.items():
                if pl == li and pk == "mlp":
                    mlp = mlp.copy(); mlp[:, pos, :] = val
            x = self._ln(x1 + mlp, L["n2w"], L["n2b"], L["eps"])
            for (pl, pk, pi, pos), val in patches.items():
                if pl == li and pk == "resid":
                    x = x.copy(); x[:, pos, :] = val
            if record:
                rec[(li, "resid")] = x.copy(); rec[(li, "head")] = heads.copy(); rec[(li, "val")] = v.copy()
                rec[(li, "pat")] = pat.copy(); rec[(li, "mlp")] = mlp.copy()
        logits = x[:, -1, :] @ self.W_out.T + self.b_out
        return (logits, rec) if record else logits


def logsoftmax(l):
    l = l - l.max(-1, keepdims=True)
    return l - np.log(np.exp(l).sum(-1, keepdims=True))


def sites_for(net: Net, n_rows: int):
    """(name, layer, kind, index, positions) — positions in the [null, rows, query] token sequence."""
    last = len(net.layers) - 1
    rows = list(range(1, n_rows + 1)); qpos = [n_rows + 1]; tpos = [0]; allp = tpos + rows + qpos
    s = [("cut", 0, "resid", 0, allp), ("cut_c", 0, "resid", 0, rows), ("cut_q", 0, "resid", 0, qpos), ("cut_t", 0, "resid", 0, tpos)]
    for h in range(net.n_heads):
        s.append((f"A{last + 1}h{h}@q", last, "head", h, qpos))
    s += [(f"A{last + 1}val", last, "val", 0, allp), (f"A{last + 1}pat", last, "pat", 0, qpos), (f"M{last + 1}q", last, "mlp", 0, qpos)]
    if last > 0:                                                             # large: the other cuts, reported
        for l in range(1, last):
            s.append((f"cut{l}", l, "resid", 0, allp))
    return s


# ------------------------------------------------------------------ worker
_G: dict = {}


def _init(ck_path, scale, d, bld_path):
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    m = load_model(scale, d, Path(ck_path))
    _G["net"] = Net(m); _G["bld"] = np.load(bld_path)


def _one(i):
    net, b = _G["net"], _G["bld"]
    K, O = int(b["K"]), int(b["O"]); n_rows = b["ctx_A"].shape[1]; d = b["ctx_A"].shape[2]; target = d - 1
    xq = b["rows"][i, :, :target].astype(np.float64)
    A = b["ctx_A"][i].astype(np.float64)
    sites = sites_for(net, n_rows)
    # sources: contexts
    src_ctx = {}
    for op in np.where(b["visible"][i])[0]:
        src_ctx[("ord", int(op))] = b["B_ord"][i, op].astype(np.float64)
    for kp in range(K):
        if kp != int(b["k"][i]):
            src_ctx[("atom", int(kp))] = b["B_atom"][i, kp].astype(np.float64)
    src_ctx[("res", 0)] = b["B_res"][i].astype(np.float64); src_ctx[("gau", 0)] = b["B_gau"][i].astype(np.float64)
    # forwards with recording
    logit_A, rec_A = net.forward(net.embed(A, xq), record=True)
    lp_A = logsoftmax(logit_A)
    recs = {}; lp_input = {}
    for key, C in src_ctx.items():
        lg, rc = net.forward(net.embed(C, xq), record=True)
        recs[key] = rc; lp_input[key] = logsoftmax(lg)
    # family means (over the visible order partners; over all K atoms including A)
    fam = {"ordmean": [k for k in recs if k[0] == "ord"], "atommean": [k for k in recs if k[0] == "atom"]}
    for name, keys in fam.items():
        mean_rec = {}
        for site in rec_A:
            arrs = [recs[k][site] for k in keys] + ([rec_A[site]] if name == "atommean" else [])
            mean_rec[site] = np.mean(arrs, axis=0)
        recs[(name, 0)] = mean_rec
    # patched forwards — FIXED source layout (absent partners = NaN) so every context shares one index
    src_keys = [("ord", o) for o in range(O)] + [("atom", k) for k in range(K)] + [("ordmean", 0), ("atommean", 0), ("res", 0), ("gau", 0)]
    n_sites = len(sites) + 1
    Delta = np.full((n_sites, len(src_keys), xq.shape[0], 100), np.nan, np.float32)
    for si, key in enumerate(src_keys):
        if key not in recs:
            continue
        Delta[0, si] = lp_input[key] - lp_A if key in lp_input else np.nan     # INPUT (means have no input form)
        for sj, (name, layer, kind, idx, pos) in enumerate(sites, start=1):
            src = recs[key][(layer, kind)]
            if kind == "head":
                val = src[:, idx, pos, :]
            elif kind == "pat":
                val = src[:, :, pos, :]
            else:
                val = src[:, pos, :]
            lg = net.forward(net.embed(A, xq), patches={(layer, kind, idx, tuple(pos)): val})
            Delta[sj, si] = logsoftmax(lg) - lp_A
    # identities on this context: complete cut from A == A; complete cut from a source == its INPUT
    cutA = net.forward(net.embed(A, xq), patches={(0, "resid", 0, tuple(sites[0][4])): rec_A[(0, "resid")][:, sites[0][4], :]})
    id_a = float(np.max(np.abs(logsoftmax(cutA) - lp_A)))
    k0 = [k for k in src_keys if k[0] == "ord" and k in recs][0]
    id_b = float(np.max(np.abs(Delta[1, src_keys.index(k0)] - Delta[0, src_keys.index(k0)])))
    return i, Delta, lp_A, [f"{k[0]}:{k[1]}" for k in src_keys], [s[0] for s in sites], id_a, id_b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="base_lr0.001_d500000")
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--scored", default=str(ROOT / "campaigns/mech_20260827/predgain_confirm/base_lr0.001_d500000"))
    p.add_argument("--nets", default=str(ROOT / "campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets"))
    p.add_argument("--build", default=str(INT / "patch"))
    p.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--n-ctx", type=int, default=None, help="limit contexts (dry runs only; output labelled)")
    p.add_argument("--panel-seed", type=int, default=770000101)
    p.add_argument("--split-seed", type=int, default=880000101)
    p.add_argument("--n-per-half", type=int, default=1000)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--world-seed", type=int, default=None)
    p.add_argument("--exploratory", action="store_true")
    a = p.parse_args()
    _env_panel(a)
    M, world = _world(a.eps)
    from pfn_dag_verify.corrected_verdict import eps_tag
    tag = eps_tag(a.eps); wt = world_tag(world, a)
    bld_path = Path(a.build) / f"build_eps{tag}_{wt}.npz"
    b = np.load(bld_path)
    scored = Path(a.scored) / f"predgain_eps{tag}_ck{a.step}.npz"
    z = np.load(scored); names = list(z["names"])
    if str(b["scored_sha256"]) != _sha(scored):
        raise SystemExit("build file does not belong to this scored cell")
    prefix = str(z["scale"]); ck = Path(a.nets) / f"eps{tag}" / f"{prefix}_s{a.seed}_ck{a.step}.pt"
    if not a.exploratory and _sha(ck) != json.loads(str(z["ckpt_sha256"]))[str(a.seed)]:
        raise SystemExit(f"{ck}: sha differs from the registered scoring")
    d = world.d
    # identity 1: transparent forward == module, on 5 contexts
    m = load_model(prefix, d, ck); net = Net(m)
    import torch
    worst = 0.0
    for i in (0, 1, 2, 500, 999):
        xq = b["rows"][i, :, :d - 1].astype(np.float64)
        lp = logsoftmax(net.forward(net.embed(b["ctx_A"][i].astype(np.float64), xq)))
        ref = m.predict_bin_probs(torch.tensor(b["ctx_A"][i][None], dtype=torch.float32), torch.tensor(b["rows"][i][None, :, :d - 1], dtype=torch.float32), torch.full((1,), 2, dtype=torch.long))[0]
        worst = max(worst, float(np.max(np.abs(np.exp(lp) - ref))))
    if worst > 1e-5:
        raise SystemExit(f"REFUSING: transparent forward differs from PFN.forward by {worst:.2e}")
    n = int(b["ctx_A"].shape[0]) if a.n_ctx is None else min(a.n_ctx, int(b["ctx_A"].shape[0]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(str(ck), prefix, d, str(bld_path))) as pool:
        res = list(pool.map(_one, range(n), chunksize=4))
    res.sort(key=lambda r: r[0])
    src_names = res[0][3]; site_names = ["INPUT"] + res[0][4]
    for r in res:
        if r[3] != src_names or r[4] != res[0][4]:
            raise SystemExit("source/site lists differ across contexts")
    Delta = np.stack([r[1] for r in res]); lp_A = np.stack([r[2] for r in res])
    id_a = max(r[5] for r in res); id_b = max(r[6] for r in res)
    # identity 2: S(model) from lp_A equals the registered column
    S_model = (b["p_true"][:n].astype(np.float64) * lp_A).sum(-1)
    dS = float(np.max(np.abs(S_model - z["S"][:n, :, names.index(f"model_s{a.seed}")])))
    print(f"[patch-run] {a.cell} eps {a.eps} s{a.seed}: module identity {worst:.1e}, cut(A)==A {id_a:.1e}, cut(B)==INPUT(B) {id_b:.1e}, S(model) {dS:.1e}; "
          f"{n} contexts x {len(site_names)} sites x {len(src_names)} sources ({time.time() - t0:.0f}s)", flush=True)
    if id_a > 1e-6 or id_b > 1e-6 or dS > 1e-4:
        raise SystemExit("REFUSING: an identity check failed")
    out = Path(a.build); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"run_{a.cell}_eps{tag}_{wt}_s{a.seed}_ck{a.step}{'_n' + str(n) if a.n_ctx else ''}{'_EXPLORATORY' if a.exploratory else ''}.npz"
    if dest.is_file():
        raise SystemExit(f"{dest} exists; written once")
    np.savez(dest, Delta=Delta, log_p_model=lp_A.astype(np.float32), sites=np.array(site_names), sources=np.array(src_names),
             n_ctx=n, cell=a.cell, eps=a.eps, seed=a.seed, step=a.step, scale=prefix, ckpt_sha256=_sha(ck), build_sha256=_sha(bld_path),
             scored_sha256=_sha(scored), identity_module=worst, identity_cutA=id_a, identity_cutB=id_b, identity_S=dS,
             K=int(b["K"]), O=int(b["O"]), d=d, world_seed=int(b["world_seed"]), exploratory=a.exploratory,
             torch=torch.__version__, numpy=np.__version__, prespec_sha256=_sha(INT / "PRESPEC_internal.md"))
    print(f"[patch-run] -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
