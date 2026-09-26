import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .constants import CALIBRATION_SEED
from .generative import generate_group
from .oracle import GridOracle
from .query_bank import CANDIDATE_QUERIES, select_symmetric_query_bank
from .storage import write_numeric_npz_atomic


def build_calibration_panel(n_contexts: int = 512):
    rng = np.random.default_rng(CALIBRATION_SEED)
    contexts = np.stack(
        [
            np.concatenate(
                [
                    (group := generate_group(rng, n_continuations=2)).core,
                    group.reference,
                ],
                axis=0,
            )
            for _ in range(n_contexts)
        ],
        axis=0,
    )
    return contexts


def calibrate(n_contexts: int = 512, quadrature: int = 15):
    started = time.perf_counter()
    contexts = build_calibration_panel(n_contexts)
    oracle = GridOracle(queries=CANDIDATE_QUERIES, quadrature=quadrature)
    f0 = np.empty((n_contexts, len(CANDIDATE_QUERIES), 100), dtype=np.float64)
    f1 = np.empty_like(f0)
    ell = np.empty(n_contexts, dtype=np.float64)
    for index, context in enumerate(contexts):
        bundle = oracle.evaluate(context)
        f0[index] = bundle.f0
        f1[index] = bundle.f1
        ell[index] = bundle.ell
    selection = select_symmetric_query_bank(f0, f1)
    context_hash = hashlib.sha256(contexts.tobytes(order="C")).hexdigest()
    result = {
        "schema_version": 1,
        "seed": CALIBRATION_SEED,
        "n_contexts": n_contexts,
        "quadrature": quadrature,
        "candidate_queries": CANDIDATE_QUERIES.tolist(),
        "selected_queries": selection.queries.tolist(),
        "objective_trace": selection.objective_trace.tolist(),
        "identifiable_fraction": selection.identifiable_fraction,
        "context_sha256": context_hash,
        "oracle_grid_size": oracle.grid_size,
        "wall_seconds": time.perf_counter() - started,
    }
    return contexts, f0, f1, ell, result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-contexts", type=int, default=512)
    parser.add_argument("--quadrature", type=int, default=15)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    contexts, f0, f1, ell, result = calibrate(args.n_contexts, args.quadrature)
    write_numeric_npz_atomic(
        args.out / "calibration_panel.npz",
        contexts=contexts,
        candidate_queries=CANDIDATE_QUERIES,
        f0=f0,
        f1=f1,
        ell=ell,
    )
    result_path = args.out / "query_bank.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["identifiable_fraction"] < 0.5:
        raise SystemExit("query-bank calibration failed the 50 percent JS gate")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
