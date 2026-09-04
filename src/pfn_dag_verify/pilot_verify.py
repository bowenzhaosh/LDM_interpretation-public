"""Independent raw-array verifier for the oracle-precision pilot.

Reads only the joined raw arrays produced by the pilot scorer and reconstructs
the preregistered gates and decisions. It imports only the Python standard
library and NumPy: it does NOT import any production estimator, likelihood,
proposal, aggregation, or decision module, and it never touches PFN outputs or
the archived oracle/join result files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


N_BINS = 100
N_ORDERINGS = 24


def load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        got = set(z.files)
        if not set(names) <= got:
            raise RuntimeError(f"verifier: raw array missing keys: {set(names) - got}")
        return {n: z[n].copy() for n in names}


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    m = np.maximum(m, 1e-300)
    p = np.maximum(p, 1e-300)
    q = np.maximum(q, 1e-300)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def mean_of_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 886910000) -> tuple[float, float, float]:
    """Bootstrap 95% CI of the row mean (percentile, linear quantile)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def run_gates(raw: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the preregistered gates from the joined raw arrays."""
    g = {}
    n = len(raw["row_id"])
    g["n_rows"] = n
    prior_code = raw["prior_code"]
    # --- SMC vs MCMC predictive agreement (per prior) ---
    g["smc_mcmc_nll_full_median_abs"] = {}
    g["smc_mcmc_nll_ablated_median_abs"] = {}
    g["smc_mcmc_nll_full_max_abs"] = {}
    g["smc_mcmc_nll_ablated_max_abs"] = {}
    g["order_js_median"] = {}
    g["order_js_p95"] = {}
    g["row_catastrophe_count"] = {}
    for prior_code_i, pname in ((0, "C"), (1, "N")):
        m = prior_code == prior_code_i
        d_full = np.abs(raw["smc_full_nll"][m] - raw["mcmc_full_nll"][m])
        d_abl = np.abs(raw["smc_ablated_nll"][m] - raw["mcmc_ablated_nll"][m])
        g["smc_mcmc_nll_full_median_abs"][pname] = float(np.median(d_full))
        g["smc_mcmc_nll_ablated_median_abs"][pname] = float(np.median(d_abl))
        g["smc_mcmc_nll_full_max_abs"][pname] = float(np.max(d_full))
        g["smc_mcmc_nll_ablated_max_abs"][pname] = float(np.max(d_abl))
        op_s = raw["smc_order_posterior"][m]
        op_m = raw["mcmc_order_posterior"][m]
        jsv = np.array([js_div(op_s[i], op_m[i]) for i in range(len(op_s))])
        g["order_js_median"][pname] = float(np.median(jsv))
        g["order_js_p95"][pname] = float(np.quantile(jsv, 0.95))
        g["row_catastrophe_count"][pname] = int(np.sum((d_full > 0.5) | (d_abl > 0.5)))
    # --- ladder convergence (coarse vs fine SMC rung) — placeholders for the
    # coarse rung arrays if provided; the pilot run provides them. ---
    g["ladder"] = {}
    if "smc_full_nll_coarse" in raw:
        for prior_code_i, pname in ((0, "C"), (1, "N")):
            m = prior_code == prior_code_i
            d_full = raw["smc_full_nll"][m] - raw["smc_full_nll_coarse"][m]
            d_abl = raw["smc_ablated_nll"][m] - raw["smc_ablated_nll_coarse"][m]
            for label, d in (("full", d_full), ("ablated", d_abl)):
                mean, lo, hi = mean_of_ci(d)
                g["ladder"][f"{pname}_{label}"] = {
                    "mean": mean, "ci": [lo, hi],
                    "pass_0005": lo > -0.0005 and hi < 0.0005,
                }
    # --- gate summary ---
    g["gates"] = {
        "smc_mcmc_nll_median_under_0.002": all(
            v < 0.002 for v in g["smc_mcmc_nll_ablated_median_abs"].values()),
        "smc_mcmc_nll_max_under_0.02": all(
            v < 0.02 for v in g["smc_mcmc_nll_ablated_max_abs"].values()),
        "order_js_median_under_1e-4": all(
            v < 1e-4 for v in g["order_js_median"].values()),
        "order_js_p95_under_1e-3": all(
            v < 1e-3 for v in g["order_js_p95"].values()),
        "no_row_catastrophe": all(
            v == 0 for v in g["row_catastrophe_count"].values()),
    }
    g["all_pass"] = bool(all(g["gates"].values()))
    return g


def join_shards(run_root: Path, config_path: Path, out_path: Path) -> dict[str, Any]:
    """Join the per-shard smc_raw.npz outputs into one sorted raw array."""
    config = json.loads(config_path.read_text())
    shard_dirs = sorted(run_root.glob("rows_*_*"), key=lambda p: int(p.name.split("_")[1]))
    arrays: dict[str, list[np.ndarray]] = {}
    meta: list[dict[str, Any]] = []
    for d in shard_dirs:
        raw_path = d / "smc_raw.npz"
        if not raw_path.is_file():
            raise RuntimeError(f"verifier: missing shard output {raw_path}")
        with np.load(raw_path, allow_pickle=False) as z:
            for k in z.files:
                arrays.setdefault(k, []).append(z[k].copy())
            meta.append({
                "shard": d.name,
                "n_rows": int(len(z["row_id"])),
                "row_id_min": int(z["row_id"].min()) if len(z["row_id"]) else None,
                "row_id_max": int(z["row_id"].max()) if len(z["row_id"]) else None,
            })
    joined = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.lexsort((joined["row_id"], joined["prior_code"]))
    joined = {k: v[order] for k, v in joined.items()}
    if len(joined["row_id"]) != 400:
        raise RuntimeError(f"verifier: joined rows != 400 ({len(joined['row_id'])})")
    expected = np.concatenate([np.arange(0, 200), np.arange(3201, 3401)])
    if not np.array_equal(np.sort(joined["row_id"]), expected):
        raise RuntimeError("verifier: joined row identity is not the frozen 400-row set")
    np.savez(out_path, **joined)
    return {"n_shards": len(shard_dirs), "n_rows": len(joined["row_id"]), "shards": meta}


def verify(joined_path: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    names = (
        "row_id", "prior_code",
        "smc_full_nll", "smc_ablated_nll", "smc_order_posterior",
        "mcmc_full_nll", "mcmc_ablated_nll", "mcmc_order_posterior",
    )
    raw = load_npz(joined_path, names)
    gates = run_gates(raw, config)
    # normalization checks on a sample of the predictive vectors
    n = len(raw["row_id"])
    checks = {
        "smc_full_probability": None, "smc_ablated_probability": None,
        "mcmc_full_probability": None, "mcmc_ablated_probability": None,
    }
    for key in ("smc_full_probability", "smc_ablated_probability",
                "mcmc_full_probability", "mcmc_ablated_probability"):
        if key in np.load(joined_path, allow_pickle=False).files:
            with np.load(joined_path, allow_pickle=False) as z:
                arr = z[key]
            # arr is (n_rows, 100): check each row's normalization
            row_sums = np.abs(arr.sum(axis=1) - 1.0)
            bad = int(np.sum(~np.isfinite(arr).any(axis=1) | (arr.min(axis=1) < 0) | (row_sums > 1e-8)))
            checks[key] = bad
    gates["predictive_normalization_bad"] = checks
    gates["normalization_ok"] = bool(
        all(v is not None and v == 0 for v in checks.values()))
    return {"n_rows": n, "gates": gates}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    result = verify(args.joined, args.config)
    print(json.dumps(result, indent=2))
    return 0 if result["gates"]["all_pass"] and result["gates"]["normalization_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
