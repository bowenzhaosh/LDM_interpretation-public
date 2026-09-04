"""EXT campaign report (REPORTED, never gated): (a, b) per scored extension cell
through the LOCKED estimator (scripts/mech_gates.py, imported, never edited).

For every predgain_eps{tag}_ck{step}[_n40ext].npz under <ext-root>/predgain/<cell>/:
  * the cell name is parsed back into (arch, lr, n_rows, dose, K, d);
  * the file's own seeds set mech_gates' expected-seeds list (the locked loader
    asserts panel identity, MECH_CONFIRM, m_q and the net prefix as for a gate);
  * MG.cell() gives b, a, se_b (N_BOOT / BOOT_SEED as registered) for the
    seed-mean unit and for each seed, bootstrap label ext:{cell}:{eps}:{step}:{unit}:{kind}.

Then, purely descriptive:
  * families = cells that differ in exactly one factor (dose, K, d, n_rows, eps, arch);
  * seed-level summaries: mean, sd, n, t over seeds of b and of within-seed dose
    differences (random-effects reading; the registered inference is fixed-effects);
  * --pool-registered folds the registered seeds 3-5 of the same cell name from
    campaigns/mech_20260827/predgain_confirm in, labelled source="registered".

--validate re-fits the registered cells with the verdict's own labels and refuses
to write anything unless every (b, se_b) matches reported/refits.json bit-for-bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import mech_gates as MG  # noqa: E402  (locked)

CELL_RE = re.compile(r"^(small|base|large)(?:_n(\d+))?_lr([0-9.e-]+)_d(\d+)(?:_K(\d+))?(?:_dim(\d+))?(?:_ws(\d+))?$")
FILE_RE = re.compile(r"^predgain_eps(\d+p\d+)_ck(\d+)(?:_(n40ext))?\.npz$")
FACTORS = ("arch", "lr", "n_rows", "dose", "K", "d", "ws", "eps", "kind")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fnum(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def parse_cell(name: str) -> dict:
    m = CELL_RE.match(name)
    if not m:
        raise SystemExit(f"unparseable cell name {name}")
    arch, n, lr, dose, K, d, ws = m.groups()
    return {"arch": arch, "n_rows": int(n or 20), "lr": float(lr), "dose": int(dose),
            "K": int(K or 8), "d": int(d or 3), "ws": int(ws) if ws else None}


def prefix_of(c: dict) -> str:
    """The net prefix mech_train.py writes (and MECH_SCALE the scorer records)."""
    p = c["arch"]
    lt = "" if c["lr"] == MG.REG_LR else f"_lr{c['lr']:g}"
    if c["n_rows"] != 20:
        lt = f"_n{c['n_rows']}" + lt
    return p + lt


def fit_cell(cell_dir: Path, cellname: str, c: dict, f: Path, source: str) -> list[dict]:
    m = FILE_RE.match(f.name)
    if not m:
        return []
    tag, step, kind = m.group(1), int(m.group(2)), (m.group(3) or "orig")
    eps = float(tag.replace("p", "."))
    z = np.load(f)
    seeds = [int(s) for s in z["seeds"]]
    MG._SEEDS_EXPECTED[:] = seeds
    scale = prefix_of(c)
    if kind == "n40ext":                       # the n20 cell's paired extension (G3 convention)
        scale = prefix_of({**c, "n_rows": 20})
    rows = []
    for unit in ["mean"] + seeds:
        label = f"ext:{cellname}:{eps}:{step}:{unit}:{kind}"
        r = MG.cell(cell_dir, eps, step, unit, None, kind=kind, scale_expected=scale, label=label)
        rows.append({**{k: c[k] for k in ("arch", "lr", "n_rows", "dose", "K", "d", "ws")},
                     "cell": cellname, "eps": eps, "step": step, "kind": kind,
                     "n_rows_scored": int(z["n_rows"]), "unit": unit, "seeds_in_file": seeds,
                     "source": source, "file": str(f.relative_to(ROOT)), "sha256": _sha(f),
                     "b": r["b"], "a": r["a"], "se_b": r["se_b"], "evaluable": r["evaluable"],
                     "G_mean": r["G_order_mean"], "regret_mean": r["regret_mean"],
                     "eta_order_summary": r["eta_order_summary"], "r_regret_G": r["r_regret_G"],
                     "n_ctx": r["n_ctx"], "boot_label": label})
    return rows


def scan(root: Path, source: str) -> list[dict]:
    out = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            c = parse_cell(cell_dir.name)
        except SystemExit:
            print(f"  skip {cell_dir.name}: not a cell name", flush=True)
            continue
        for f in sorted(cell_dir.glob("**/predgain_eps*_ck*.npz")):
            idx_p = f.parent / "world_index.json"
            if idx_p.is_file():
                w = json.loads(idx_p.read_text()).get(f.name)
                if w and (w["d"] != c["d"] or w["K"] != c["K"] or w.get("world_seed") != c["ws"]):
                    raise SystemExit(f"{f}: world index (d={w['d']}, K={w['K']}, ws={w.get('world_seed')}) != cell name {cell_dir.name}")
            out.extend(fit_cell(f.parent, cell_dir.name, c, f, source))
            print(f"  {source} {cell_dir.name} {f.name}: {len(out)} fits so far", flush=True)
    return out


def families(rows: list[dict]) -> list[dict]:
    """Group unit=='mean' fits that differ in exactly one factor."""
    fams = []
    mean_rows = [r for r in rows if r["unit"] == "mean"]
    for vary in FACTORS:
        groups = defaultdict(list)
        for r in mean_rows:
            key = tuple((k, r[k]) for k in FACTORS if k != vary)
            groups[key].append(r)
        for key, members in groups.items():
            vals = sorted({r[vary] for r in members}, key=lambda v: (str(type(v)), v))
            if len(vals) < 2:
                continue
            members = sorted(members, key=lambda r: (str(type(r[vary])), r[vary]))
            fams.append({"varies": vary, "fixed": dict(key),
                         "levels": [{vary: r[vary], "b": r["b"], "se_b": r["se_b"], "a": r["a"],
                                     "G_mean": r["G_mean"], "source": r["source"], "cell": r["cell"]}
                                    for r in members]})
    return fams


def seed_summaries(rows: list[dict]) -> list[dict]:
    """Random-effects reading over seeds: b per (cell, eps, step, kind), and the
    within-seed dose differences for cells that differ only in dose."""
    out = []
    per_seed = [r for r in rows if r["unit"] != "mean"]
    by_cell = defaultdict(dict)
    for r in per_seed:
        by_cell[(r["cell"], r["eps"], r["step"], r["kind"])][r["unit"]] = r
    for key, seeds in by_cell.items():
        b = np.array([seeds[s]["b"] for s in sorted(seeds)])
        n = len(b)
        out.append({"kind": "b_over_seeds", "cell": key[0], "eps": key[1], "step": key[2],
                    "file_kind": key[3], "seeds": sorted(seeds), "n": n,
                    "b_mean": float(b.mean()), "b_sd": float(b.std(ddof=1)) if n > 1 else None,
                    "t": float(b.mean() / (b.std(ddof=1) / np.sqrt(n))) if n > 1 and b.std(ddof=1) > 0 else None,
                    "sources": sorted({seeds[s]["source"] for s in seeds})})
    # within-seed dose differences
    by_fam = defaultdict(dict)
    for r in per_seed:
        fam = tuple((k, r[k]) for k in FACTORS if k != "dose")
        by_fam[fam].setdefault(r["unit"], {})[r["dose"]] = r["b"]
    for fam, seeds in by_fam.items():
        doses = sorted({d for s in seeds.values() for d in s})
        for lo, hi in zip(doses, doses[1:]):
            diffs = {s: v[hi] - v[lo] for s, v in seeds.items() if lo in v and hi in v}
            if len(diffs) < 2:
                continue
            d = np.array(list(diffs.values()))
            out.append({"kind": "dose_diff_over_seeds", "fixed": dict(fam), "dose_lo": lo, "dose_hi": hi,
                        "seeds": sorted(diffs), "n": len(d), "diff_mean": float(d.mean()),
                        "diff_sd": float(d.std(ddof=1)), "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else None,
                        "n_negative": int((d < 0).sum()), "per_seed": {str(s): float(v) for s, v in diffs.items()}})
    return out


def validate(confirm_root: Path, refits_path: Path) -> dict:
    """Re-fit the registered cells with the verdict's own labels; every (b, se_b)
    must equal reported/refits.json to the last bit."""
    ref = json.loads(refits_path.read_text())
    entries = None
    for k, v in ref.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "boot_label" in v[0]:
            entries = v
            break
        if isinstance(v, dict):
            vals = list(v.values())
            if vals and isinstance(vals[0], dict) and "boot_label" in vals[0]:
                entries = vals
                break
    if not entries:
        raise SystemExit("refits.json: no fit entries with boot_label found")
    ref_by = {(e["file"], str(e["unit"])): e for e in entries}
    MG._SEEDS_EXPECTED[:] = [3, 4, 5]
    n_ok = n_bad = 0
    bad = []
    for e in entries:
        if e["unit"] != "mean":
            continue
        f = ROOT / e["file"]
        d = f.parent
        cellname = d.name
        c = parse_cell(cellname)
        if c["arch"] == "base" and "n40" in cellname:
            continue
        kind = e.get("kind", "orig")
        r = MG.cell(d, e["eps"], e["step"], "mean", None, kind=kind, scale_expected=e["scale"], label=e["boot_label"])
        same = (r["b"] == e["b"]) and (r["se_b"] == e["se_b"])
        n_ok += int(same); n_bad += int(not same)
        if not same:
            bad.append({"file": e["file"], "b_ref": e["b"], "b_new": r["b"], "se_ref": e["se_b"], "se_new": r["se_b"]})
    return {"n_ok": n_ok, "n_bad": n_bad, "mismatches": bad, "refits": str(refits_path.relative_to(ROOT)),
            "refits_sha256": _sha(refits_path)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ext-root", type=Path, default=ROOT / "campaigns/mech_ext_20260902")
    p.add_argument("--confirm-root", type=Path, default=ROOT / "campaigns/mech_20260827/predgain_confirm")
    p.add_argument("--refits", type=Path, default=ROOT / "campaigns/mech_20260827/reported/refits.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pool-registered", action="store_true",
                   help="also fit the registered seeds 3-5 of any cell name present in the ext tree")
    p.add_argument("--validate", action="store_true", help="only re-fit the registered cells and compare to refits.json")
    a = p.parse_args()
    out = a.out or (a.ext_root / "reported")
    if "campaigns/mech_20260827" in str(out.resolve()):
        raise SystemExit("REFUSING: --out inside the registered campaign")
    t0 = time.time()
    val = validate(a.confirm_root, a.refits)
    print(f"[validate] registered re-fit: {val['n_ok']} match, {val['n_bad']} mismatch", flush=True)
    if val["n_bad"]:
        print(json.dumps(val["mismatches"][:5], indent=1))
        raise SystemExit("REFUSING: the locked estimator did not reproduce refits.json through this script")
    if a.validate:
        return 0
    pg = a.ext_root / "predgain"
    rows = scan(pg, "ext") if pg.is_dir() else []
    if a.pool_registered and rows:
        names = sorted({r["cell"] for r in rows})
        for name in names:
            d = a.confirm_root / name
            if d.is_dir():
                c = parse_cell(name)
                for f in sorted(d.glob("predgain_eps*_ck*.npz")):
                    rows.extend(fit_cell(d, name, c, f, "registered"))
        n40 = a.confirm_root.parent / "predgain_confirm_n40"
        for name in names:
            d = n40 / name
            if d.is_dir():
                c = parse_cell(name)
                for f in sorted(d.glob("predgain_eps*_ck*.npz")):
                    rows.extend(fit_cell(d, name, c, f, "registered"))
    out.mkdir(parents=True, exist_ok=True)
    doc = {"status": "REPORTED — never gated; Amendment G verdict untouched", "campaign": "mech_ext_20260902",
           "estimator": "scripts/mech_gates.py (locked; imported)", "n_boot": MG.N_BOOT, "boot_seed": MG.BOOT_SEED,
           "se_max": MG.SE_MAX, "panel": MG.REG_PANEL, "validation": val,
           "n_fits": len(rows), "cells": rows}
    (out / "ext_cells.json").write_text(json.dumps(doc, indent=1, allow_nan=False, default=_fnum))
    (out / "ext_families.json").write_text(json.dumps({"status": doc["status"], "families": families(rows)},
                                                       indent=1, allow_nan=False, default=_fnum))
    (out / "ext_seed_summaries.json").write_text(json.dumps({"status": doc["status"], "summaries": seed_summaries(rows)},
                                                             indent=1, allow_nan=False, default=_fnum))
    man = {"script": "scripts/mech_ext_report.py", "script_sha256": _sha(Path(__file__).resolve()),
           "mech_gates_sha256": _sha(ROOT / "scripts/mech_gates.py"),
           "inputs_sha256": {r["file"]: r["sha256"] for r in rows},
           "outputs_sha256": {n: _sha(out / n) for n in ("ext_cells.json", "ext_families.json", "ext_seed_summaries.json")},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": time.time() - t0}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"[ext] {len(rows)} fits -> {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
