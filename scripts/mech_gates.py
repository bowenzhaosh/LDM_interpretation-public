"""Amendment G gate evaluator. The verdict is what this file prints — no session
reads summaries and grades by eye.

Primary estimand (G.1). For one cell (arch, lr, dose, eps, seed) scored by
scripts/mech_predgain.py on the registered fresh panel: per context i,
    regret_i = mean_q [ S(q_full) - S(q_model) ]      (exact expected log score)
    G_i      = mean_q [ S(q_full) - S(q_abl)   ]      (order-information gain)
and the OLS fit  regret_i = a + b * G_i  over the 1000 contexts gives
    b = order-specific shortfall fraction,  a = generic floor.
Context-level bootstrap (N_BOOT resamples, paired wherever two cells share
contexts) gives the SE of b and of every contrast below.

Directory convention (G.2). Every scored cell lives in its own directory named
`{arch}_lr{lr:g}_d{dose}` under a root, holding `predgain_eps{tag}_ck{step}.npz`
for each eps. `scale` recorded in the artifact must equal the cell's scale tag
(`{arch}` at the registered lr 1e-3, `{arch}_lr{lr:g}` otherwise — the naming
scripts/mech_train.py gives the checkpoints), so a mis-pointed directory is an
error rather than a silently wrong number.

Learning-rate selection (G.1). Capacity gates compare architectures, so each
architecture is given its own best learning rate by a registered, mechanical
rule applied symmetrically: for every LR in LR_GRID, score seeds on the
SELECTION panel (REG_SELECT_PANEL — neither the exploration panel nor the gate
panel), take the mean over contexts of the seed-averaged regret_i, and select
the argmin. Ties within SEL_TIE go to 1e-3, then to the larger LR. A cell whose
selection statistic is not finite is DIVERGED: excluded from selection and
reported; if every LR of an arm is DIVERGED the gate that needs it is NOT
EVALUABLE. A selection cell that is ABSENT, or that fails any provenance check,
is a REFUSAL (SystemExit) and not a DIVERGED arm -- otherwise a job that died on
its wall limit would silently move LR* to whichever arm happened to finish.
The whole selection table is written into the verdict. No gate ever reads a
selection-panel artifact, and no gate reads the gate-panel artifact of an
unselected LR.

Evaluability (G.3). The PRIMARY cell (per-context regret averaged over the
model seeds) is evaluable iff SE(b) <= SE_MAX; per-seed cells enter only through
the sign of their contrast. A single-part gate whose primary cell is not
evaluable is NOT EVALUABLE — never pass, never fail. A CONJUNCTIVE gate follows
Kleene strong conjunction: FAIL if any part FAILs, else NOT_EVALUABLE if any
part is not evaluable, else PASS — a definitely false conjunct forecloses PASS
whatever the unmeasurable ones would have said.

Gates (G.4), each one-sided at alpha on the primary unit, PLUS sign agreement
of every per-seed contrast (no per-seed z threshold):
    G1  b(eps=1.0) - b(eps=0.5) < 0            at each dose      (unpaired: different worlds)
    G2  b(500k)    - b(100k)    < 0            at each eps       (paired: same panel)
    G3  b(n_rows=40) - b(n_rows=20) < 0        at eps 0.75, 500k (paired: the n40 panel
                                                extends the n20 panel context by context)
    G4  mean_i [ JS(w_model_i, uniform) - JS(w_model_i, w_exact_i) ] < 0
        at eps 0.5 and 0.75, 500k; the per-context statistic is averaged over the
        model seeds on the contexts where every seed's LP solved (>= 80% required);
        Amendment F readout code unmodified (scripts/mech_phase1.py project stage).
    G5a b(large @ LR*_large, 500k) - b(base @ LR*_base, 500k) < 0   at eps 0.75 (paired)
    G5b the same at eps 1.0                                                     (paired)
    G6  b(large @ LR*_large, 500k) - b(base @ LR*_base(2M), 2M) < 0 at eps 0.75 (paired)
G1-G4 keep the registered recipe lr 1e-3 for the base arm: they are
within-recipe direction tests, and mixing per-eps LR selections into them would
compare across recipes.

Panel identity is asserted from the artifacts (registered seeds, n_per_half,
scale, and equal k/o for paired contrasts), never assumed. Bootstrap RNGs are
seeded per test from BOOT_SEED and the test's label — a label that excludes the
directory, so path spelling cannot move a resample.
eta_order is reported as a summary only and takes part in no gate.

Usage
  python scripts/mech_gates.py --fleet ROOT_n20 --n40 DIR_n40 --readout DIR_phase1 \
      --grid ROOT_grid --select ROOT_select \
      --seeds 3 4 5 --out campaigns/.../AMENDMENT_G_VERDICT.json
Every path argument is taken as given (this file joins no base directory): run
the verdict from the repo root, the convention G.2 pins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402

ALPHA = 0.01
SE_MAX = 0.15
N_BOOT = 2000
BOOT_SEED = 20260830
# Registered gate panel (G.2). Every gate artifact must carry these.
REG_PANEL = {"panel_seed": 770000101, "split_seed": 880000101, "n_per_half": 1000}
# Registered SELECTION panel (G.1). No gate reads an artifact carrying these.
REG_SELECT_PANEL = {"panel_seed": 770000201, "split_seed": 880000201, "n_per_half": 500}
# Registered learning-rate grid, applied symmetrically to every architecture.
LR_GRID = (3e-3, 1e-3, 3e-4, 1e-4)
REG_LR = 1e-3                    # the registered recipe; G1-G4's base arm
SEL_TIE = 1e-6                   # |delta| below this is a tie
M_Q = 8                          # queries per context (G.1); asserted on every cell
DOSE_MAIN = 500_000              # the dose the capacity gates compare at
DOSE_LONG = 2_000_000            # G6's base dose

Z = {0.01: 2.3263478740408408, 0.05: 1.6448536269514722}

_SEEDS_EXPECTED: list = []


def scale_tag(arch: str, lr: float) -> str:
    """The net prefix / MECH_SCALE mech_train.py gives this (arch, lr).
    mech_train.py:57 tags the filename only when lr differs from the registered 1e-3."""
    return arch if lr == REG_LR else f"{arch}_lr{lr:g}"


def cell_dir(root: Path, arch: str, lr: float, dose: int) -> Path:
    return root / f"{arch}_lr{lr:g}_d{dose}"


def _rng(label: str):
    import zlib
    return np.random.default_rng([BOOT_SEED, zlib.crc32(label.encode())])


def _check_panel(z, path: Path, seeds, n_rows_expected: int | None = None,
                 scale_expected: str | None = None, panel: dict | None = None) -> None:
    for k, v in (panel or REG_PANEL).items():
        if k not in z.files or int(z[k]) != v:
            raise SystemExit(f"{path}: {k}={z[k] if k in z.files else 'ABSENT'} != registered {v}")
    if "seeds" not in z.files or [int(s) for s in z["seeds"]] != [int(s) for s in seeds]:
        raise SystemExit(f"{path}: seeds {z['seeds'] if 'seeds' in z.files else 'ABSENT'} != {list(seeds)}")
    if "confirm" not in z.files or not bool(z["confirm"]):
        raise SystemExit(f"{path}: not scored under MECH_CONFIRM=1 (key absent or false)")
    if n_rows_expected is not None and ("n_rows" not in z.files or int(z["n_rows"]) != n_rows_expected):
        raise SystemExit(f"{path}: n_rows={z['n_rows'] if 'n_rows' in z.files else 'ABSENT'} != {n_rows_expected}")
    if "m_q" not in z.files or int(z["m_q"]) != M_Q:
        raise SystemExit(f"{path}: m_q={z['m_q'] if 'm_q' in z.files else 'ABSENT'} != registered {M_Q}")
    if scale_expected is not None and ("scale" not in z.files or str(z["scale"]) != scale_expected):
        raise SystemExit(f"{path}: scale={z['scale'] if 'scale' in z.files else 'ABSENT'} "
                         f"!= {scale_expected} (the directory does not hold the cell it names)")


def _load_raw(d: Path, eps: float, step: int, kind: str, scale_expected: str | None = None,
              panel: dict | None = None):
    suffix = "" if kind == "orig" else f"_{kind}"
    path = d / f"predgain_eps{eps_tag(eps)}_ck{step}{suffix}.npz"
    if not path.is_file():
        raise SystemExit(f"MISSING scored cell {path}")
    z = np.load(path)
    _check_panel(z, path, _SEEDS_EXPECTED, n_rows_expected=(40 if kind == "n40ext" else 20),
                 scale_expected=scale_expected, panel=panel)
    return z


def _load_cell(d: Path, eps: float, step: int, seed, kind: str = "orig",
               scale_expected: str | None = None, panel: dict | None = None):
    """regret_i, G_i for one cell. seed may be an int (one model seed) or the
    string 'mean' (per-context mean over all model seeds in the file — the F.2c
    decision-B convention, the PRIMARY form of every gate)."""
    z = _load_raw(d, eps, step, kind, scale_expected, panel)
    n = list(z["names"])
    S = z["S"].mean(axis=1)                                   # (n_ctx, n_pred)
    full, abl = S[:, n.index("full")], S[:, n.index("abl")]
    if seed == "mean":
        cols = [i for i, k in enumerate(n) if k.startswith("model_s")]
        model = S[:, cols].mean(axis=1)
    else:
        model = S[:, n.index(f"model_s{seed}")]
    return full - model, full - abl                           # regret_i, G_i


def _ols(reg, G):
    """OLS slope/intercept. A non-finite input silently poisons np.polyfit
    (it returns nan without raising), and a nan b would flow into a contrast
    and be reported as a number; refuse instead."""
    if not (np.isfinite(reg).all() and np.isfinite(G).all()):
        raise SystemExit(f"non-finite regret/G in an OLS cell "
                         f"({int((~np.isfinite(reg)).sum())} bad regret, "
                         f"{int((~np.isfinite(G)).sum())} bad G)")
    b, a = np.polyfit(G, reg, 1)
    if not (np.isfinite(b) and np.isfinite(a)):
        raise SystemExit("OLS returned a non-finite fit")
    return float(b), float(a)


def _boot_idx(n, rng):
    return rng.integers(0, n, n)


def cell(d: Path, eps: float, step: int, seed, rng, kind: str = "orig",
         scale_expected: str | None = None, label: str | None = None) -> dict:
    rng = _rng(label or f"cell:{eps}:{step}:{seed}:{kind}")
    reg, G = _load_cell(d, eps, step, seed, kind, scale_expected)
    b, a = _ols(reg, G)
    bs = np.array([_ols(reg[i], G[i])[0] for i in (_boot_idx(len(G), rng) for _ in range(N_BOOT))])
    se = float(bs.std(ddof=1))
    return {"eps": eps, "step": step, "seed": seed, "kind": kind, "n_ctx": int(len(G)),
            "scale": scale_expected, "dir": str(d),
            "b": b, "a": a, "se_b": se, "evaluable": bool(se <= SE_MAX),
            "G_order_mean": float(G.mean()), "regret_mean": float(reg.mean()),
            "eta_order_summary": float(1 - reg.mean() / G.mean()) if G.mean() > 0 else None,
            "r_regret_G": float(np.corrcoef(reg, G)[0, 1])}


# ---------------------------------------------------------------- LR selection

def select_lr(select_root: Path, arch: str, dose: int, eps: float, seeds) -> dict:
    """Registered selection rule (G.1). Returns the selection table for one
    (arch, dose, eps) and the selected LR, or lr=None if every arm DIVERGED."""
    table = []
    for lr in LR_GRID:
        d = cell_dir(select_root, arch, lr, dose)
        row = {"lr": lr, "dir": str(d), "scale": scale_tag(arch, lr)}
        # A MISSING selection cell is a refusal, not a DIVERGED arm: a job that died
        # on its wall limit would otherwise silently move LR* to whichever arm
        # finished, with exit 0. Provenance failures (wrong panel, wrong scale, no
        # MECH_CONFIRM=1) must likewise propagate -- swallowing them here would let a
        # gate run at an LR the registered rule did not select.
        reg, _G = _load_cell(d, eps, dose, "mean", "orig",
                             scale_expected=scale_tag(arch, lr), panel=REG_SELECT_PANEL)
        stat = float(np.mean(reg))
        # DIVERGED keeps its registered meaning: the arm trained, was scored, and
        # its statistic is not finite.
        row.update(regret_mean=stat, diverged=not np.isfinite(stat), n_ctx=int(len(reg)))
        table.append(row)
    live = [r for r in table if not r["diverged"]]
    if not live:
        return {"arch": arch, "dose": dose, "eps": eps, "lr": None,
                "reason": "every LR DIVERGED", "table": table}
    best = min(r["regret_mean"] for r in live)
    tied = [r for r in live if abs(r["regret_mean"] - best) < SEL_TIE]
    if any(r["lr"] == REG_LR for r in tied):
        pick = REG_LR
    else:
        pick = max(r["lr"] for r in tied)
    return {"arch": arch, "dose": dose, "eps": eps, "lr": pick,
            "selection_statistic": "mean over contexts of seed-averaged regret_i, selection panel",
            "n_tied": len(tied), "table": table}


# ---------------------------------------------------------------- contrasts

def contrast_unpaired(c_lo: dict, c_hi: dict, alpha: float) -> dict:
    """Test c_hi.b - c_lo.b < 0 (one-sided), independent panels/worlds."""
    if not (c_lo["evaluable"] and c_hi["evaluable"]) and (c_lo["seed"] == "mean" or c_hi["seed"] == "mean"):
        return {"evaluable": False, "passes": None}
    diff = c_hi["b"] - c_lo["b"]
    se = float(np.hypot(c_lo["se_b"], c_hi["se_b"]))
    z = diff / se
    return {"evaluable": True, "diff": diff, "se": se, "z": z, "passes": bool(z < -Z[alpha])}


def contrast_paired(lo: tuple, hi: tuple, seed, rng, alpha: float, c_lo: dict, c_hi: dict,
                    label: str | None = None) -> dict:
    """Test b(hi) - b(lo) < 0 with the SAME context indices resampled jointly.
    lo/hi = (dir, eps, step, kind, scale). Pairing is by context index: the two
    cells must come from the same panel (same latents), asserted below."""
    if not (c_lo["evaluable"] and c_hi["evaluable"]) and seed == "mean":
        return {"evaluable": False, "passes": None}
    # label excludes the directory: path spelling must not move a resample
    rng = _rng(label or f"paired:{lo[1:4]}:{hi[1:4]}:{seed}")
    z0 = _load_raw(lo[0], lo[1], lo[2], lo[3], lo[4])
    z1 = _load_raw(hi[0], hi[1], hi[2], hi[3], hi[4])
    if not (np.array_equal(z0["k"], z1["k"]) and np.array_equal(z0["o"], z1["o"])):
        raise SystemExit(f"paired contrast {lo[1:]} vs {hi[1:]}: the two cells do not share latents (k, o)")
    r0, G0 = _load_cell(lo[0], lo[1], lo[2], seed, lo[3], lo[4])
    r1, G1 = _load_cell(hi[0], hi[1], hi[2], seed, hi[3], hi[4])
    if len(G0) != len(G1):
        raise SystemExit("paired contrast requires the same contexts")
    diffs = []
    for _ in range(N_BOOT):
        i = _boot_idx(len(G0), rng)
        diffs.append(_ols(r1[i], G1[i])[0] - _ols(r0[i], G0[i])[0])
    diff = float(_ols(r1, G1)[0] - _ols(r0, G0)[0])
    se = float(np.std(diffs, ddof=1))
    z = diff / se
    return {"evaluable": True, "diff": diff, "se": se, "z": z, "passes": bool(z < -Z[alpha])}


def gate(primary: dict, per_seed: list) -> dict:
    """A gate PASSES iff the primary (seed-averaged) test passes AND every
    per-seed difference has the registered sign (consistency criterion; the
    per-seed z is reported but not thresholded)."""
    # Evaluability is decided on the PRIMARY cell only (G.3); per-seed cells enter
    # through the sign of their contrast, which exists whenever both cells loaded.
    if primary.get("passes") is None:
        return {"result": "NOT_EVALUABLE", "primary": primary, "per_seed": per_seed}
    signs_ok = all(("diff" in t) and t["diff"] < 0 for t in per_seed)
    return {"result": "PASS" if (primary["passes"] and signs_ok) else "FAIL",
            "primary_passes": primary["passes"], "sign_agreement": signs_ok,
            "primary": primary, "per_seed": per_seed}


def not_evaluable(reason: str) -> dict:
    return {"result": "NOT_EVALUABLE", "reason": reason,
            "primary": {"passes": None}, "per_seed": []}


def _js(p, q):
    p = np.maximum(p, 1e-300); q = np.maximum(q, 1e-300); m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


def g4_cell(readout: Path, eps: float, step: int, seeds, rng, alpha: float,
            fleet_cell: Path | None = None, min_frac: float = 0.8) -> dict:
    """seeds: a list -> per-context mean of the statistic over those seeds (primary);
    a single int -> that seed alone (consistency).

    The readout artifacts carry the same provenance the predgain artifacts do
    (G.2): the registered panel, MECH_CONFIRM=1, and the checkpoint digest. The
    digest is checked against the fleet cell the projections are pinned to read,
    so an instrument gate cannot be run on models other than the registered
    fleet's."""
    rng = _rng(f"g4:{eps}:{step}:{seeds}")
    ex_path = readout / f"eps{eps_tag(eps)}.npz"
    if not ex_path.is_file():
        raise SystemExit(f"MISSING G4 oracle artifact {ex_path}")
    ex = np.load(ex_path)
    for k, v in {**REG_PANEL, "n_rows": 20}.items():
        if k not in ex.files or int(ex[k]) != v:
            raise SystemExit(f"{ex_path}: {k} != registered {v}")
    if "confirm" not in ex.files or not bool(ex["confirm"]):
        raise SystemExit(f"{ex_path}: not built under MECH_CONFIRM=1 (key absent or false)")
    want = None
    if fleet_cell is not None:
        fz = _load_raw(fleet_cell, eps, step, "orig", scale_tag("base", REG_LR))
        want = json.loads(str(fz["ckpt_sha256"]))
    seed_list = seeds if isinstance(seeds, list) else [seeds]
    stats, oks = [], []
    for s in seed_list:
        pp = readout / f"proj_eps{eps_tag(eps)}_s{s}_ck{step}.npz"
        if not pp.is_file():
            raise SystemExit(f"MISSING G4 projection {pp}")
        pr = np.load(pp)
        for k, v in {**REG_PANEL, "n_rows": 20}.items():
            if k not in pr.files or int(pr[k]) != v:
                raise SystemExit(f"{pp}: {k} != registered {v}")
        if "confirm" not in pr.files or not bool(pr["confirm"]):
            raise SystemExit(f"{pp}: not projected under MECH_CONFIRM=1 (key absent or false)")
        got = str(pr["ckpt_sha256"]) if "ckpt_sha256" in pr.files else "ABSENT"
        if want is not None and got != want.get(str(s)):
            raise SystemExit(f"{pp}: checkpoint sha {got[:12]} != the registered fleet's "
                             f"{str(want.get(str(s)))[:12]} for seed {s}")
        w = pr["w_proj"]; ok = np.isfinite(w).all(1)
        st = np.full(len(w), np.nan)
        st[ok] = _js(w[ok], ex["w_uniform"][ok]) - _js(w[ok], ex["w_exact"][ok])
        stats.append(st); oks.append(ok)
    ok = np.all(oks, axis=0)
    stat = np.mean(stats, axis=0)[ok]
    bs = np.array([stat[_boot_idx(len(stat), rng)].mean() for _ in range(N_BOOT)])
    se = float(bs.std(ddof=1)); mean = float(stat.mean()); z = mean / se
    evaluable = bool(ok.sum() >= min_frac * len(ok))
    return {"eps": eps, "step": step, "seed": seeds if isinstance(seeds, int) else "mean",
            "n_ctx": int(ok.sum()), "n_lp_failed": int((~ok).sum()),
            "diff": mean, "se": se, "z": z, "evaluable": evaluable,
            "passes": bool(z < -Z[alpha]) if evaluable else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=Path, required=True,
                   help="root of the registered n_rows-20 base cells ({arch}_lr{lr}_d{dose}/)")
    p.add_argument("--n40", type=Path, required=True, help="the n_rows-40 cell directory")
    p.add_argument("--readout", type=Path, required=True, help="mech_phase1 outputs on the fresh panel (G4)")
    p.add_argument("--grid", type=Path, default=None,
                   help="root of the gate-panel LR-grid cells (G5a/G5b/G6). Omit -> those gates NOT EVALUABLE")
    p.add_argument("--select", type=Path, default=None,
                   help="root of the SELECTION-panel cells. Omit -> G5a/G5b/G6 NOT EVALUABLE")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--eps", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    p.add_argument("--steps", type=int, nargs="+", default=[100_000, 500_000])
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--readout-step", type=int, default=500_000,
                   help="checkpoint step of the G4 projections (registered: 500000)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = None                                   # every test seeds its own RNG (_rng)
    _SEEDS_EXPECTED[:] = list(a.seeds)

    lo, hi = min(a.steps), max(a.steps)
    units = ["mean"] + list(a.seeds)               # primary first, then per-seed consistency
    BASE = scale_tag("base", REG_LR)

    def base_dir(dose):
        return cell_dir(a.fleet, "base", REG_LR, dose)

    cells = {(e, st, u): cell(base_dir(st), e, st, u, rng, scale_expected=BASE,
                              label=f"cell:base:{REG_LR}:{e}:{st}:{u}")
             for e in a.eps for st in a.steps for u in units}
    n40 = {u: cell(a.n40, 0.75, hi, u, rng, kind="n40ext", scale_expected="base_n40",
                   label=f"cell:base_n40:{REG_LR}:0.75:{hi}:{u}") for u in units}

    def g1(u, st):
        return {"step": st, "seed": u, **contrast_unpaired(cells[(0.5, st, u)], cells[(1.0, st, u)], a.alpha)}

    def g2(u, e):
        return {"eps": e, "seed": u,
                **contrast_paired((base_dir(lo), e, lo, "orig", BASE),
                                  (base_dir(hi), e, hi, "orig", BASE), u, rng, a.alpha,
                                  cells[(e, lo, u)], cells[(e, hi, u)],
                                  label=f"G2:{e}:{lo}:{hi}:{u}")}

    def g3(u):
        return {"seed": u, **contrast_paired((base_dir(hi), 0.75, hi, "orig", BASE),
                                             (a.n40, 0.75, hi, "n40ext", "base_n40"), u, rng, a.alpha,
                                             cells[(0.75, hi, u)], n40[u], label=f"G3:0.75:{hi}:{u}")}

    def g4(u, e):
        return g4_cell(a.readout, e, a.readout_step, list(a.seeds) if u == "mean" else u,
                       rng, a.alpha, fleet_cell=base_dir(a.readout_step))

    gates = {}
    G1 = [gate(g1("mean", st), [g1(s, st) for s in a.seeds]) for st in a.steps]
    G2 = [gate(g2("mean", e), [g2(s, e) for s in a.seeds]) for e in a.eps]
    G3 = [gate(g3("mean"), [g3(s) for s in a.seeds])]
    G4 = [gate(g4("mean", e), [g4(s, e) for s in a.seeds]) for e in (0.5, 0.75)]

    # ---- capacity gates: select each architecture's LR, then contrast ----
    selection: list = []

    def capacity_gate(eps_val: float, base_dose: int, label: str):
        """b(large @ LR*_large, DOSE_MAIN) - b(base @ LR*_base, base_dose) < 0, paired."""
        if a.grid is None or a.select is None:
            return not_evaluable("--grid / --select not supplied: the LR-grid arms were not scored")
        sel_b = select_lr(a.select, "base", base_dose, eps_val, a.seeds)
        sel_l = select_lr(a.select, "large", DOSE_MAIN, eps_val, a.seeds)
        selection.extend([{"gate": label, **sel_b}, {"gate": label, **sel_l}])
        if sel_b["lr"] is None or sel_l["lr"] is None:
            return not_evaluable(f"{label}: every LR DIVERGED for "
                                 f"{'base' if sel_b['lr'] is None else 'large'}")
        db = cell_dir(a.grid, "base", sel_b["lr"], base_dose)
        dl = cell_dir(a.grid, "large", sel_l["lr"], DOSE_MAIN)
        sb, sl = scale_tag("base", sel_b["lr"]), scale_tag("large", sel_l["lr"])
        cb = {u: cell(db, eps_val, base_dose, u, rng, scale_expected=sb,
                      label=f"cell:base:{sel_b['lr']}:{eps_val}:{base_dose}:{u}") for u in units}
        cl = {u: cell(dl, eps_val, DOSE_MAIN, u, rng, scale_expected=sl,
                      label=f"cell:large:{sel_l['lr']}:{eps_val}:{DOSE_MAIN}:{u}") for u in units}

        def t(u):
            return {"seed": u, **contrast_paired((db, eps_val, base_dose, "orig", sb),
                                                 (dl, eps_val, DOSE_MAIN, "orig", sl), u, rng, a.alpha,
                                                 cb[u], cl[u],
                                                 label=f"{label}:{eps_val}:{base_dose}:{DOSE_MAIN}:{u}")}
        g = gate(t("mean"), [t(s) for s in a.seeds])
        g["selected_lr"] = {"base": sel_b["lr"], "large": sel_l["lr"]}
        # G.1 promises b, a, SE and r for every cell a gate uses.
        g["cells"] = {"base": list(cb.values()), "large": list(cl.values())}
        return g

    G5a = [capacity_gate(0.75, DOSE_MAIN, "G5a")]
    G5b = [capacity_gate(1.0, DOSE_MAIN, "G5b")]
    G6 = [capacity_gate(0.75, DOSE_LONG, "G6")]

    def conj(parts, labels):
        res = [p["result"] for p in parts]
        # Kleene strong conjunction (G.3): a definitely false conjunct forecloses
        # PASS whatever the unmeasurable ones would have said (F & U = F, T & U = U).
        overall = ("FAIL" if "FAIL" in res
                   else "NOT_EVALUABLE" if "NOT_EVALUABLE" in res
                   else "PASS")
        return {"result": overall, "parts": [{"label": l, **p} for l, p in zip(labels, parts)]}

    gates["G1_eps_direction"] = conj(G1, [f"dose={st}" for st in a.steps])
    gates["G2_dose_direction"] = conj(G2, [f"eps={e}" for e in a.eps])
    gates["G3_signal_share"] = conj(G3, ["eps=0.75"])
    gates["G4_instrument"] = conj(G4, ["eps=0.5", "eps=0.75"])
    gates["G5a_capacity_eps075"] = conj(G5a, ["eps=0.75"])
    gates["G5b_capacity_eps1"] = conj(G5b, ["eps=1.0"])
    gates["G6_capacity_vs_dose"] = conj(G6, ["eps=0.75"])

    verdict = {
        "amendment": "G", "argv": sys.argv, "registered_panel": REG_PANEL,
        "selection_panel": REG_SELECT_PANEL, "lr_grid": list(LR_GRID), "registered_lr": REG_LR,
        "alpha": a.alpha, "se_max": SE_MAX, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "seeds": a.seeds, "eps": a.eps, "steps": a.steps, "readout_step": a.readout_step,
        "primary_unit": "per-context mean over seeds (decision-B convention); per-seed sign agreement required",
        "conjunction_rule": "Kleene strong: FAIL > NOT_EVALUABLE > PASS",
        "selection_table": selection,
        "cells": list(cells.values()), "n40_cells": list(n40.values()), "gates": gates,
    }
    verdict["headline"] = {k: v["result"] for k, v in gates.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(verdict, indent=2, default=float))
    print(json.dumps(verdict["headline"]))
    for k, v in gates.items():
        for part in v["parts"]:
            pr = part["primary"]
            signs = [t.get("diff") for t in part["per_seed"]]
            extra = f"  ({part['reason']})" if part.get("reason") else ""
            print(f"  {k} {part['label']}: {part['result']}  primary diff={pr.get('diff', float('nan')):+.3f} "
                  f"z={pr.get('z', float('nan')):+.2f}  per-seed diffs="
                  + ",".join(f"{d:+.3f}" if d is not None else "NA" for d in signs) + extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
