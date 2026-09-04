"""Outcome-blind scorer for the oracle-precision pilot on the frozen 400-row panel.

Reads only the frozen panel inputs and labels (contexts, queries, outcome bins)
for the nested-half rows (draw 0, stream index < 200, priors C and N). Runs the
frozen SMC estimator (primary) and the MCMC+TI estimator (independent) for all
24 orderings per row, computes the full and ordering-ablated predictives, the
order posterior, and the held-out NLLs. Writes raw arrays and diagnostics.

The guard in `pilot_shared.assert_path_allowed` prevents any access to PFN
checkpoints, PFN predictions, or the archived oracle/join result files. The
pilot never computes deficit, gap, or Delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .pilot_shared import (
    fleet,
    production_quadrature,
    assert_path_allowed,
    assert_no_forbidden_imports,
    N_ORDERINGS,
    N_BINS,
)
from .pilot_smc import (
    run_smc_posterior,
    order_predictive as smc_order_predictive,
    full_and_ablated_predictives as smc_full_ablated,
)
from .pilot_mcmc import (
    run_mcmc_predictive,
    mcmc_evidence_ti,
    order_predictive as mcmc_order_predictive,
)


PILOT_CONFIG = "config/oracle_precision_pilot_v1.json"


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


ROW_KEY_FIELDS = ("row_id", "prior_code", "draw_index", "stream_index", "row_key_sha256")


def _panel_rows(panel_dir: Path, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Collect the 400 nested-half rows (draw 0, stream<200) from the frozen panel.

    Verifies input-label alignment per shard (the production pipeline hard-fails
    on misalignment), verifies the shard hashes against the panel manifest, and
    asserts the exact frozen 400-row identity.
    """
    from .pilot_shared import sha256_file
    rows: dict[str, list[np.ndarray]] = {
        "row_id": [], "prior_code": [], "contexts": [], "queries": [], "outcome_bins": []
    }
    half = config["nested_half_subset"]
    manifest = None
    manifest_path = panel_dir / "panel_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    for prior in ("C", "N"):
        for bank in range(3):
            input_path = panel_dir / "inputs" / f"{prior}_d0_b{bank}.npz"
            label_path = panel_dir / "labels" / f"{prior}_d0_b{bank}.npz"
            assert_path_allowed(input_path)
            assert_path_allowed(label_path)
            # provenance: hash against the frozen panel manifest
            if manifest is not None:
                expected = manifest.get("artifacts", {}).get(f"inputs/{input_path.name}")
                if expected is not None and sha256_file(input_path) != expected["sha256"]:
                    raise RuntimeError(f"panel input hash mismatch: {input_path.name}")
                expected_l = manifest.get("artifacts", {}).get(f"labels/{label_path.name}")
                if expected_l is not None and sha256_file(label_path) != expected_l["sha256"]:
                    raise RuntimeError(f"panel label hash mismatch: {label_path.name}")
            with np.load(input_path, allow_pickle=False) as a_in, np.load(label_path, allow_pickle=False) as a_lab:
                # hard-fail on input/label misalignment, matching phase1_join
                for k in ROW_KEY_FIELDS:
                    if not np.array_equal(a_in[k], a_lab[k]):
                        raise RuntimeError(f"panel input/label misalignment: {k} {input_path.name}")
                mask = (a_in["draw_index"] == half["draw_index"]) & (
                    a_in["stream_index"] < half["stream_index_stop_exclusive"]
                )
                # the mask must equal the panel's stored nested_half_mask
                if not np.array_equal(mask, a_in["nested_half_mask"] == 1):
                    raise RuntimeError(f"nested-half mask mismatch: {input_path.name}")
                rows["row_id"].append(a_in["row_id"][mask])
                rows["prior_code"].append(a_in["prior_code"][mask])
                rows["contexts"].append(a_in["contexts"][mask])
                rows["queries"].append(a_in["queries"][mask])
                rows["outcome_bins"].append(a_lab["outcome_bins"][mask])
    out = {k: np.concatenate(v) for k, v in rows.items()}
    # order deterministically by (prior_code, row_id)
    order = np.lexsort((out["row_id"], out["prior_code"]))
    out = {k: v[order] for k, v in out.items()}
    # exact frozen 400-row identity: C rows 0..199, N rows 3201..3400
    if len(out["row_id"]) != 400:
        raise RuntimeError(f"pilot panel must be exactly 400 rows, got {len(out['row_id'])}")
    expected_ids = np.concatenate([np.arange(0, 200), np.arange(3201, 3401)])
    if not np.array_equal(np.sort(out["row_id"]), expected_ids):
        raise RuntimeError("pilot panel row identity is not the frozen nested-half set")
    return out


def _smc_row(
    context: np.ndarray,
    query: np.ndarray,
    outcome_bin: int,
    prior: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    n_particles = int(config["smc"]["n_particles"])
    mh_steps = int(config["smc"]["mh_steps"])
    smc = {
        o: run_smc_posterior(
            context, o, prior,
            n_particles=n_particles, mh_steps=mh_steps, seed=seed + o, device=device,
        )
        for o in range(N_ORDERINGS)
    }
    full, ablated, w_o = smc_full_ablated(context, query, prior, smc, device)
    for p in (full, ablated):
        if not (np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1.0) < 1e-8):
            raise RuntimeError("smc predictive is not a valid probability vector")
    if full[outcome_bin] <= 0.0 or not np.isfinite(full[outcome_bin]):
        raise RuntimeError(f"smc full predictive has zero observed-bin mass: bin {outcome_bin}")
    if ablated[outcome_bin] <= 0.0 or not np.isfinite(ablated[outcome_bin]):
        raise RuntimeError(f"smc ablated predictive has zero observed-bin mass: bin {outcome_bin}")
    nll_full = -np.log(full[outcome_bin])
    nll_ablated = -np.log(ablated[outcome_bin])
    return {
        "full_probability": full,
        "ablated_probability": ablated,
        "ordering_posterior": w_o,
        "nll_full": float(nll_full),
        "nll_ablated": float(nll_ablated),
        "logZ": {o: smc[o]["logZ"] for o in range(N_ORDERINGS)},
        "ess": {o: smc[o]["ess"] for o in range(N_ORDERINGS)},
        "n_temps": {o: len(smc[o]["temperatures"]) for o in range(N_ORDERINGS)},
        "accept": {o: float(np.mean(smc[o]["accept_history"])) for o in range(N_ORDERINGS)},
    }


def _mcmc_row_ordering(
    context: np.ndarray,
    ordering: int,
    prior: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    from .pilot_mcmc import _mcmc_chain, find_mode
    from .pilot_shared import sample_prior_z
    nc = int(config["mcmc"]["n_chains"])
    betas = np.asarray(config["mcmc"]["betas"], dtype=np.float64)
    n_per_beta = int(config["mcmc"]["n_iter_per_beta"])
    # evidence via TI, beta=0 chain from the prior
    zp, sgp = sample_prior_z((), nc, np.random.default_rng(seed + 1000))
    z_cur = torch.as_tensor(zp, dtype=torch.float64, device=device)
    sg_cur = torch.as_tensor(sgp, dtype=torch.float64, device=device)
    means: list[float] = []
    for b in betas:
        z, sg, ll, _ = _mcmc_chain(
            context, ordering, prior, z_cur, sg_cur, float(b),
            n_per_beta, nc, seed + int(b * 1000), device,
        )
        fin = torch.isfinite(ll)
        means.append(float(torch.mean(ll[fin]).item()))
        finidx = torch.nonzero(fin).flatten()
        z_cur = z[finidx[-1]].unsqueeze(0)
        sg_cur = sg[finidx[-1]].unsqueeze(0)
    m = np.asarray(means)
    logz = float(np.sum(np.diff(betas) * (m[:-1] + m[1:]) / 2.0))
    return {"logZ": logz, "ti_means": m}


def score_pilot(
    config_path: Path,
    panel_dir: Path,
    out_dir: Path,
    device_name: str = "cuda",
    row_start: int = 0,
    row_count: int | None = None,
) -> dict[str, Any]:
    assert_no_forbidden_imports()
    config = _load_config(config_path)
    from .pilot_shared import FORBIDDEN_SEED_NAMESPACES
    seed_root = int(config["seed_root"])
    for ns in FORBIDDEN_SEED_NAMESPACES:
        if abs(seed_root - int(ns)) < 1_000_000:
            raise RuntimeError(f"pilot seed root {seed_root} is within 1e6 of a forbidden seed {ns}")
    device = torch.device(device_name)
    rows = _panel_rows(panel_dir, config)
    n = len(rows["row_id"])
    if row_count is None:
        row_count = n - row_start
    selected = slice(row_start, min(row_start + row_count, n))
    results = {
        "smc": [], "mcmc": [], "row_keys": rows["row_id"][selected],
        "prior_code": rows["prior_code"][selected],
    }
    seed_root = int(config["seed_root"]) + row_start
    for i in range(row_start, row_start + row_count):
        if i >= n:
            break
        li = i - row_start
        prior = "C" if rows["prior_code"][i] == 0 else "N"
        context = rows["contexts"][i]
        query = rows["queries"][i]
        ob = int(rows["outcome_bins"][i])
        smc_row = _smc_row(context, query, ob, prior, config, device, seed_root + i)
        mcmc_logz = [
            _mcmc_row_ordering(context, o, prior, config, device, seed_root + i)[
                "logZ"
            ]
            for o in range(N_ORDERINGS)
        ]
        mcmc_predictives = {
            o: mcmc_order_predictive(
                context, query, o, prior,
                run_mcmc_predictive(
                    context, o, prior,
                    n_chains=int(config["mcmc"]["n_chains"]),
                    n_iter=int(config["mcmc"]["n_iter"]),
                    seed=seed_root + i + o, device=device,
                ),
                device,
            )
            for o in range(N_ORDERINGS)
        }
        mcmc_logz = np.asarray(mcmc_logz, dtype=np.float64)
        full = sum(mcmc_predictives[o] * np.exp(mcmc_logz[o] - mcmc_logz.max()) for o in range(N_ORDERINGS))
        full /= np.sum(np.exp(mcmc_logz - mcmc_logz.max()))
        ablated = sum(mcmc_predictives[o] for o in range(N_ORDERINGS)) / N_ORDERINGS
        w_o = np.exp(mcmc_logz - mcmc_logz.max()); w_o /= w_o.sum()
        for p in (full, ablated, w_o):
            if not (np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1.0) < 1e-6):
                raise RuntimeError("mcmc predictive/order posterior is not a valid probability vector")
        if full[ob] <= 0.0 or ablated[ob] <= 0.0:
            raise RuntimeError(f"mcmc predictive has zero observed-bin mass: bin {ob}")
        results["smc"].append(smc_row)
        results["mcmc"].append({
            "full_probability": full,
            "ablated_probability": ablated,
            "ordering_posterior": w_o,
            "nll_full": float(-np.log(full[ob])),
            "nll_ablated": float(-np.log(ablated[ob])),
            "logZ": mcmc_logz,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_path": str(config_path), "n_rows": n, "seed_root": seed_root,
        "row_start": row_start, "row_count": row_count,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "smc_raw.npz",
             row_id=results["row_keys"], prior_code=results["prior_code"],
             smc_full_nll=np.array([r["nll_full"] for r in results["smc"]]),
             smc_ablated_nll=np.array([r["nll_ablated"] for r in results["smc"]]),
             smc_order_posterior=np.array([r["ordering_posterior"] for r in results["smc"]]),
             smc_full_probability=np.array([r["full_probability"] for r in results["smc"]]),
             smc_ablated_probability=np.array([r["ablated_probability"] for r in results["smc"]]),
             smc_logZ=np.array([[r["logZ"][o] for o in range(N_ORDERINGS)] for r in results["smc"]]),
             smc_ess=np.array([[r["ess"][o] for o in range(N_ORDERINGS)] for r in results["smc"]]),
             smc_accept=np.array([[r["accept"][o] for o in range(N_ORDERINGS)] for r in results["smc"]]),
             mcmc_full_nll=np.array([r["nll_full"] for r in results["mcmc"]]),
             mcmc_ablated_nll=np.array([r["nll_ablated"] for r in results["mcmc"]]),
             mcmc_order_posterior=np.array([r["ordering_posterior"] for r in results["mcmc"]]),
             mcmc_full_probability=np.array([r["full_probability"] for r in results["mcmc"]]),
             mcmc_ablated_probability=np.array([r["ablated_probability"] for r in results["mcmc"]]),
             mcmc_logZ=np.array([r["logZ"] for r in results["mcmc"]]),
             )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(PILOT_CONFIG))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-count", type=int, default=None)
    args = parser.parse_args(argv)
    result = score_pilot(args.config, args.panel, args.out, args.device,
                         args.row_start, args.row_count)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
