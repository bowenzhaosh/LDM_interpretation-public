import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .decision import DecisionInputs, decide_primary
from .integrity import EXPECTED_IDENTITIES, verify_prediction_ledger
from .instrument import (
    project_coordinate_batch,
    project_kl_batch,
    reconstruct_updated_batch,
)
from .model import configure_determinism
from .provenance import (
    derive_seed,
    require_scientific_run_path,
    repository_root,
    validate_locked_validations,
    verify_panel_lock,
)
from .registry import sha256_file
from .statistics import (
    crossed_bootstrap_slope,
    crossed_resample_weights,
    permutation_null_slopes,
    within_group_slope,
)
from .storage import load_numeric_npz, write_json_atomic, write_numeric_npz_atomic


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _in_chunks(p, f0, f1, function, chunk_size=64):
    values = []
    for start in range(0, len(p), chunk_size):
        stop = min(len(p), start + chunk_size)
        values.append(function(p[start:stop], f0[start:stop], f1[start:stop]))
    fields = type(values[0]).__dataclass_fields__
    return {
        field: np.concatenate([np.asarray(getattr(value, field)) for value in values])
        for field in fields
    }


def _derive_one(panel, prediction, bank_index):
    groups, continuations = panel["ell_target"].shape
    target_count = groups * continuations
    f0_core = panel["f0_core"][bank_index]
    f1_core = panel["f1_core"][bank_index]
    f0_base = panel["f0_base"][bank_index]
    f1_base = panel["f1_base"][bank_index]
    f0_target = panel["f0_target"][bank_index].reshape(target_count, 8, 100)
    f1_target = panel["f1_target"][bank_index].reshape(target_count, 8, 100)
    p_target = prediction["p_target"].reshape(target_count, 8, 100)

    coord_core = _in_chunks(
        prediction["p_core"], f0_core, f1_core, project_coordinate_batch
    )
    coord_base = _in_chunks(
        prediction["p_base"], f0_base, f1_base, project_coordinate_batch
    )
    coord_target = _in_chunks(p_target, f0_target, f1_target, project_coordinate_batch)
    kl_core = _in_chunks(prediction["p_core"], f0_core, f1_core, project_kl_batch)
    kl_base = _in_chunks(prediction["p_base"], f0_base, f1_base, project_kl_batch)
    kl_target = _in_chunks(p_target, f0_target, f1_target, project_kl_batch)

    delta_replace = (panel["ell_target"] - panel["ell_base"][:, None]).reshape(-1)
    delta_append = (panel["ell_target"] - panel["ell_core"][:, None]).reshape(-1)
    base_weight_replace = np.repeat(np.clip(coord_base["w"], 1e-12, 1 - 1e-12), continuations)
    base_weight_append = np.repeat(np.clip(coord_core["w"], 1e-12, 1 - 1e-12), continuations)
    _, reconstruction_replace = reconstruct_updated_batch(
        p_target,
        f0_target,
        f1_target,
        w_base=base_weight_replace,
        delta_ell=delta_replace,
    )
    _, reconstruction_append = reconstruct_updated_batch(
        p_target,
        f0_target,
        f1_target,
        w_base=base_weight_append,
        delta_ell=delta_append,
    )

    return {
        "g_core": coord_core["g"],
        "g_base": coord_base["g"],
        "g_target": coord_target["g"].reshape(groups, continuations),
        "w_core": coord_core["w"],
        "w_base": coord_base["w"],
        "w_target": coord_target["w"].reshape(groups, continuations),
        "raw_w_core": coord_core["raw_w"],
        "raw_w_base": coord_base["raw_w"],
        "raw_w_target": coord_target["raw_w"].reshape(groups, continuations),
        "kl_g_core": kl_core["g"],
        "kl_g_base": kl_base["g"],
        "kl_g_target": kl_target["g"].reshape(groups, continuations),
        "mix_residual_core": coord_core["normalized_residual"],
        "mix_residual_base": coord_base["normalized_residual"],
        "mix_residual_target": coord_target["normalized_residual"].reshape(
            groups, continuations
        ),
        "boundary_core": coord_core["boundary"].astype(np.uint8),
        "boundary_base": coord_base["boundary"].astype(np.uint8),
        "boundary_target": coord_target["boundary"].reshape(groups, continuations).astype(
            np.uint8
        ),
        "reconstruction_replace": reconstruction_replace.reshape(groups, continuations),
        "reconstruction_append": reconstruction_append.reshape(groups, continuations),
    }


DERIVED_VECTOR_FIELDS = {
    "g_core",
    "g_base",
    "w_core",
    "w_base",
    "raw_w_core",
    "raw_w_base",
    "kl_g_core",
    "kl_g_base",
    "mix_residual_core",
    "mix_residual_base",
    "boundary_core",
    "boundary_base",
}
DERIVED_MATRIX_FIELDS = {
    "g_target",
    "w_target",
    "raw_w_target",
    "kl_g_target",
    "mix_residual_target",
    "boundary_target",
    "reconstruction_replace",
    "reconstruction_append",
}
DERIVED_METADATA_FIELDS = {
    "seed",
    "step",
    "bank_index",
    "prediction_sha256",
    "panel_sha256",
    "scientific",
}


def _validate_derived_shard(
    shard: dict,
    *,
    groups: int,
    continuations: int,
    seed: int,
    step: int,
    bank_index: int,
    prediction_sha256: str,
    panel_sha256: str,
) -> None:
    expected = DERIVED_VECTOR_FIELDS | DERIVED_MATRIX_FIELDS | DERIVED_METADATA_FIELDS
    if set(shard) != expected:
        raise ValueError(f"derived shard schema mismatch: {sorted(set(shard) ^ expected)}")
    for name in DERIVED_VECTOR_FIELDS:
        if shard[name].shape != (groups,) or not np.isfinite(shard[name]).all():
            raise ValueError(f"derived vector is malformed: {name}")
    for name in DERIVED_MATRIX_FIELDS:
        if shard[name].shape != (groups, continuations) or not np.isfinite(shard[name]).all():
            raise ValueError(f"derived matrix is malformed: {name}")
    for name in ("boundary_core", "boundary_base", "boundary_target"):
        if not np.isin(shard[name], [0, 1]).all():
            raise ValueError(f"derived boundary flag is malformed: {name}")
    for name in ("w_core", "w_base", "w_target"):
        if np.any(shard[name] < 0) or np.any(shard[name] > 1):
            raise ValueError(f"derived fitted weight is outside [0, 1]: {name}")
    if (
        int(shard["seed"]) != seed
        or int(shard["step"]) != step
        or int(shard["bank_index"]) != bank_index
        or int(shard["scientific"]) != 1
    ):
        raise ValueError("derived shard identity mismatch")
    if bytes(shard["prediction_sha256"].tolist()).hex() != prediction_sha256:
        raise ValueError("derived prediction hash mismatch")
    if bytes(shard["panel_sha256"].tolist()).hex() != panel_sha256:
        raise ValueError("derived panel hash mismatch")


def _validate_derive_progress(
    progress: dict,
    *,
    commit_sha: str,
    panel_sha256: str,
    prediction_ledger_sha256: str,
    require_complete: bool,
) -> None:
    identity = {
        "schema_version": 1,
        "scientific": True,
        "commit_sha": commit_sha,
        "panel_sha256": panel_sha256,
        "prediction_ledger_sha256": prediction_ledger_sha256,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    expected_top = set(identity) | {
        "attempts",
        "cumulative_wall_seconds",
        "peak_rss_bytes",
        "validated_shard_identities",
    }
    if set(progress) != expected_top or any(
        progress.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("derive progress identity mismatch")
    attempts = progress.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("derive progress has no attempts")
    expected_attempt_keys = {
        "attempt_index",
        "status",
        "wall_seconds",
        "peak_rss_bytes",
        "completed_shards",
    }
    attempts_valid = all(
        set(attempt) == expected_attempt_keys
        and attempt.get("attempt_index") == index
        and attempt.get("status") in {"RUNNING", "INTERRUPTED", "COMPLETE"}
        and (index == len(attempts) - 1 or attempt.get("status") != "RUNNING")
        and np.isfinite(float(attempt.get("wall_seconds", np.nan)))
        and float(attempt.get("wall_seconds", -1.0)) >= 0.0
        and int(attempt.get("peak_rss_bytes", -1)) >= 0
        and 0 <= int(attempt.get("completed_shards", -1)) <= 64
        for index, attempt in enumerate(attempts)
    )
    identities = progress.get("validated_shard_identities")
    identities_valid = isinstance(identities, list) and all(
        isinstance(identity_value, list)
        and len(identity_value) == 3
        and tuple(int(value) for value in identity_value) in EXPECTED_IDENTITIES
        for identity_value in identities
    )
    if (
        not attempts_valid
        or not identities_valid
        or identities != sorted(identities)
        or len(identities) != len({tuple(value) for value in identities})
        or not np.isclose(
            float(progress.get("cumulative_wall_seconds", np.nan)),
            sum(float(attempt["wall_seconds"]) for attempt in attempts),
            rtol=1e-12,
            atol=1e-9,
        )
        or int(progress.get("peak_rss_bytes", -1))
        != max(int(attempt["peak_rss_bytes"]) for attempt in attempts)
    ):
        raise ValueError("derive progress resource history mismatch")
    if require_complete and (
        attempts[-1]["status"] != "COMPLETE"
        or {tuple(value) for value in identities} != EXPECTED_IDENTITIES
    ):
        raise ValueError("derive progress is incomplete")


def _start_derive_progress(
    path: Path,
    *,
    commit_sha: str,
    panel_sha256: str,
    prediction_ledger_sha256: str,
) -> dict:
    if path.exists():
        progress = json.loads(path.read_text())
        _validate_derive_progress(
            progress,
            commit_sha=commit_sha,
            panel_sha256=panel_sha256,
            prediction_ledger_sha256=prediction_ledger_sha256,
            require_complete=False,
        )
        if progress["attempts"][-1]["status"] == "RUNNING":
            progress["attempts"][-1]["status"] = "INTERRUPTED"
        elif progress["attempts"][-1]["status"] != "COMPLETE":
            raise ValueError("derive progress is not safely resumable")
    else:
        progress = {
            "schema_version": 1,
            "scientific": True,
            "commit_sha": commit_sha,
            "panel_sha256": panel_sha256,
            "prediction_ledger_sha256": prediction_ledger_sha256,
            "implementation_sha256": sha256_file(Path(__file__)),
            "attempts": [],
            "cumulative_wall_seconds": 0.0,
            "peak_rss_bytes": 0,
            "validated_shard_identities": [],
        }
    progress["attempts"].append(
        {
            "attempt_index": len(progress["attempts"]),
            "status": "RUNNING",
            "wall_seconds": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
            "completed_shards": 0,
        }
    )
    write_json_atomic(path, progress)
    return progress


def _update_derive_progress(
    path: Path,
    progress: dict,
    *,
    attempt_started: float,
    records: list[dict],
    status: str,
) -> dict:
    if status not in {"RUNNING", "COMPLETE"}:
        raise ValueError("unsupported derive progress status")
    progress["attempts"][-1].update(
        {
            "status": status,
            "wall_seconds": time.perf_counter() - attempt_started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "completed_shards": len(records),
        }
    )
    progress["cumulative_wall_seconds"] = float(
        sum(float(attempt["wall_seconds"]) for attempt in progress["attempts"])
    )
    progress["peak_rss_bytes"] = int(
        max(int(attempt["peak_rss_bytes"]) for attempt in progress["attempts"])
    )
    progress["validated_shard_identities"] = sorted(
        [
            [int(record["seed"]), int(record["step"]), int(record["bank_index"])]
            for record in records
        ]
    )
    write_json_atomic(path, progress)
    return progress


def derive_all(*, panel_path: Path, prediction_dir: Path, out_dir: Path):
    configure_determinism(0)
    started = time.perf_counter()
    panel, prediction_ledger, prediction_paths = verify_prediction_ledger(
        panel_path=panel_path, prediction_dir=prediction_dir
    )
    commit, _ = verify_panel_lock(panel)
    require_scientific_run_path(
        panel_path, commit_sha=commit, relative="panel.npz"
    )
    require_scientific_run_path(
        prediction_dir, commit_sha=commit, relative="predictions"
    )
    require_scientific_run_path(
        out_dir, commit_sha=commit, relative="derived"
    )
    panel_hash = sha256_file(panel_path)
    prediction_ledger_hash = sha256_file(
        prediction_dir / "prediction_ledger.json"
    )
    prediction_records = {
        record["path"]: record for record in prediction_ledger["records"]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir.parent / "derive_progress.json"
    progress = _start_derive_progress(
        progress_path,
        commit_sha=commit,
        panel_sha256=panel_hash,
        prediction_ledger_sha256=prediction_ledger_hash,
    )
    records = []
    try:
        for prediction_path in prediction_paths:
            prediction_record = prediction_records[prediction_path.name]
            prediction = load_numeric_npz(prediction_path)
            bank_index = int(prediction["bank_index"])
            seed = int(prediction["seed"])
            step = int(prediction["step"])
            identity = (seed, step, bank_index)
            if identity not in EXPECTED_IDENTITIES:
                raise ValueError(f"unexpected prediction identity: {identity}")
            path = out_dir / f"derived_s{seed:02d}_step{step:05d}_bank{bank_index}.npz"
            if path.exists():
                existing = load_numeric_npz(path)
                _validate_derived_shard(
                    existing,
                    groups=panel["core"].shape[0],
                    continuations=panel["continuations"].shape[1],
                    seed=seed,
                    step=step,
                    bank_index=bank_index,
                    prediction_sha256=prediction_record["sha256"],
                    panel_sha256=panel_hash,
                )
            else:
                derived = _derive_one(panel, prediction, bank_index)
                write_numeric_npz_atomic(
                    path,
                    **derived,
                    seed=np.asarray(seed, dtype=np.int16),
                    step=np.asarray(step, dtype=np.int32),
                    bank_index=np.asarray(bank_index, dtype=np.int8),
                    prediction_sha256=np.frombuffer(
                        bytes.fromhex(prediction_record["sha256"]), dtype=np.uint8
                    ),
                    panel_sha256=np.frombuffer(bytes.fromhex(panel_hash), dtype=np.uint8),
                    scientific=np.asarray(1, dtype=np.int8),
                )
                _validate_derived_shard(
                    load_numeric_npz(path),
                    groups=panel["core"].shape[0],
                    continuations=panel["continuations"].shape[1],
                    seed=seed,
                    step=step,
                    bank_index=bank_index,
                    prediction_sha256=prediction_record["sha256"],
                    panel_sha256=panel_hash,
                )
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "bank_index": bank_index,
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "prediction": prediction_path.name,
                    "prediction_sha256": prediction_record["sha256"],
                }
            )
            progress = _update_derive_progress(
                progress_path,
                progress,
                attempt_started=started,
                records=records,
                status="RUNNING",
            )
    except BaseException:
        _update_derive_progress(
            progress_path,
            progress,
            attempt_started=started,
            records=records,
            status="RUNNING",
        )
        raise
    identities = {
        (record["seed"], record["step"], record["bank_index"]) for record in records
    }
    if len(records) != 64 or identities != EXPECTED_IDENTITIES:
        raise AssertionError("derived fleet is incomplete or duplicated")
    actual_names = {path.name for path in out_dir.glob("derived_*.npz")}
    expected_names = {record["path"] for record in records}
    if actual_names != expected_names:
        raise ValueError("derived directory contains a stale or missing shard")
    progress = _update_derive_progress(
        progress_path,
        progress,
        attempt_started=started,
        records=records,
        status="COMPLETE",
    )
    ledger = {
        "schema_version": 1,
        "scientific": True,
        "commit_sha": commit,
        "panel_sha256": panel_hash,
        "prediction_ledger_sha256": prediction_ledger_hash,
        "derive_progress_sha256": sha256_file(progress_path),
        "records": records,
        "wall_seconds": progress["cumulative_wall_seconds"],
        "peak_rss_bytes": progress["peak_rss_bytes"],
    }
    write_json_atomic(out_dir / "derived_ledger.json", ledger)
    return ledger


def verify_derived_ledger(
    *, panel_path: Path, prediction_dir: Path, derived_dir: Path
) -> tuple[dict, dict, dict[tuple[int, int, int], dict]]:
    panel, _, _ = verify_prediction_ledger(
        panel_path=panel_path, prediction_dir=prediction_dir
    )
    commit, _ = verify_panel_lock(panel)
    panel_hash = sha256_file(panel_path)
    ledger = json.loads((derived_dir / "derived_ledger.json").read_text())
    prediction_ledger_hash = sha256_file(prediction_dir / "prediction_ledger.json")
    progress_path = derived_dir.parent / "derive_progress.json"
    if (
        ledger.get("schema_version") != 1
        or ledger.get("scientific") is not True
        or ledger.get("commit_sha") != commit
        or ledger.get("panel_sha256") != panel_hash
        or ledger.get("prediction_ledger_sha256") != prediction_ledger_hash
        or ledger.get("derive_progress_sha256") != sha256_file(progress_path)
    ):
        raise ValueError("derived ledger identity mismatch")
    progress = json.loads(progress_path.read_text())
    _validate_derive_progress(
        progress,
        commit_sha=commit,
        panel_sha256=panel_hash,
        prediction_ledger_sha256=prediction_ledger_hash,
        require_complete=True,
    )
    try:
        wall_seconds = float(ledger["wall_seconds"])
        peak_rss_bytes = int(ledger["peak_rss_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("derived ledger resource fields are invalid") from error
    if (
        not np.isfinite(wall_seconds)
        or wall_seconds < 0.0
        or peak_rss_bytes < 0
        or not np.isclose(
            wall_seconds,
            float(progress["cumulative_wall_seconds"]),
            rtol=1e-12,
            atol=1e-9,
        )
        or peak_rss_bytes != int(progress["peak_rss_bytes"])
    ):
        raise ValueError("derived ledger resources do not match cumulative progress")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError("derived ledger must contain exactly 64 records")
    shards = {}
    expected_names = set()
    for record in records:
        identity = (int(record["seed"]), int(record["step"]), int(record["bank_index"]))
        if identity in shards or identity not in EXPECTED_IDENTITIES:
            raise ValueError(f"duplicate or unexpected derived identity: {identity}")
        expected_name = (
            f"derived_s{identity[0]:02d}_step{identity[1]:05d}_bank{identity[2]}.npz"
        )
        if record.get("path") != expected_name:
            raise ValueError("derived record path does not match its identity")
        path = derived_dir / expected_name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"derived record content mismatch: {expected_name}")
        shard = load_numeric_npz(path)
        _validate_derived_shard(
            shard,
            groups=panel["core"].shape[0],
            continuations=panel["continuations"].shape[1],
            seed=identity[0],
            step=identity[1],
            bank_index=identity[2],
            prediction_sha256=record["prediction_sha256"],
            panel_sha256=panel_hash,
        )
        shards[identity] = shard
        expected_names.add(expected_name)
    if set(shards) != EXPECTED_IDENTITIES:
        raise ValueError("derived identities are incomplete")
    if {path.name for path in derived_dir.glob("derived_*.npz")} != expected_names:
        raise ValueError("derived directory contains a stale or missing shard")
    return panel, ledger, shards


def _fleet_arrays(
    derived_dir: Path,
    step: int,
    bank_index: int,
    *,
    verified: dict[tuple[int, int, int], dict] | None = None,
):
    if verified is not None:
        return [verified[(seed, step, bank_index)] for seed in range(16)]
    shards = []
    for seed in range(16):
        path = derived_dir / f"derived_s{seed:02d}_step{step:05d}_bank{bank_index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(load_numeric_npz(path))
    return shards


def _valid_groups(mask):
    retained = np.sum(mask, axis=1) >= 4
    return retained, mask[retained]


def _nrmse(x, y, mask):
    error = y - x[None, :, :]
    selected_error = error[:, mask]
    scale = float(np.std(x[mask]))
    if scale <= 1e-15:
        raise ValueError("exact-response scale is zero")
    return float(np.sqrt(np.mean(selected_error**2)) / scale)


def _nmae(x, y, mask):
    selected_x = x[mask]
    quartiles = np.quantile(selected_x, [0.25, 0.75], method="inverted_cdf")
    scale = float(quartiles[1] - quartiles[0])
    if scale <= 1e-15:
        raise ValueError("exact-response IQR is zero")
    return float(np.mean(np.abs(y - x[None, :, :])[:, mask]) / scale)


def _bootstrap_nrmse(
    x,
    y,
    mask,
    rng,
    n_boot=10_000,
    resample_weights: tuple[np.ndarray, np.ndarray] | None = None,
):
    seeds, groups, _ = y.shape
    error = y - x[None, :, :]
    squared = np.zeros((seeds, groups), dtype=np.float64)
    counts = np.sum(mask, axis=1).astype(np.float64)
    x_sum = np.zeros(groups, dtype=np.float64)
    x_sum2 = np.zeros(groups, dtype=np.float64)
    for group in range(groups):
        keep = mask[group]
        squared[:, group] = np.sum(error[:, group, keep] ** 2, axis=1)
        x_sum[group] = np.sum(x[group, keep])
        x_sum2[group] = np.sum(x[group, keep] ** 2)
    if resample_weights is None:
        seed_weights, group_weights = crossed_resample_weights(
            n_boot=n_boot, seeds=seeds, groups=groups, rng=rng
        )
    else:
        seed_weights, group_weights = resample_weights
    error_sum = np.einsum("bs,sg,bg->b", seed_weights, squared, group_weights)
    count = seeds * (group_weights @ counts)
    rmse = np.sqrt(error_sum / count)
    x_count = group_weights @ counts
    mean = (group_weights @ x_sum) / x_count
    variance = (group_weights @ x_sum2) / x_count - mean * mean
    return rmse / np.sqrt(np.maximum(variance, 1e-30))


def _bootstrap_nmae(
    x,
    y,
    mask,
    rng,
    n_boot=10_000,
    resample_weights: tuple[np.ndarray, np.ndarray] | None = None,
):
    seeds, groups, _ = y.shape
    absolute = np.zeros((seeds, groups), dtype=np.float64)
    counts = np.sum(mask, axis=1).astype(np.float64)
    for group in range(groups):
        keep = mask[group]
        absolute[:, group] = np.sum(
            np.abs(y[:, group, keep] - x[group, keep][None, :]), axis=1
        )
    if resample_weights is None:
        seed_weights, group_weights = crossed_resample_weights(
            n_boot=n_boot, seeds=seeds, groups=groups, rng=rng
        )
    else:
        seed_weights, group_weights = resample_weights
    error_sum = np.einsum("bs,sg,bg->b", seed_weights, absolute, group_weights)
    count = seeds * (group_weights @ counts)

    flat_mask = mask.reshape(-1)
    flat_group = np.repeat(np.arange(groups), x.shape[1])[flat_mask]
    flat_x = x.reshape(-1)[flat_mask]
    order = np.argsort(flat_x)
    ordered_x = flat_x[order]
    ordered_group = flat_group[order]
    iqr = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        weights = group_weights[index, ordered_group]
        cumulative = np.cumsum(weights)
        total = cumulative[-1]
        if total <= 0:
            raise FloatingPointError("bootstrap produced an empty exact-response sample")
        low = min(
            np.searchsorted(cumulative, 0.25 * total, side="left"),
            len(ordered_x) - 1,
        )
        high = min(
            np.searchsorted(cumulative, 0.75 * total, side="left"),
            len(ordered_x) - 1,
        )
        iqr[index] = ordered_x[high] - ordered_x[low]
    if np.any(iqr <= 1e-15):
        raise FloatingPointError("bootstrap produced zero exact-response IQR")
    return (error_sum / count) / iqr


def _bootstrap_nrmse_difference(x, y0, y1, mask, rng, n_boot=10_000):
    weights = crossed_resample_weights(
        n_boot=n_boot, seeds=y0.shape[0], groups=y0.shape[1], rng=rng
    )
    first = _bootstrap_nrmse(x, y0, mask, rng, n_boot, weights)
    second = _bootstrap_nrmse(x, y1, mask, rng, n_boot, weights)
    return second - first


def _interval(draws):
    return [float(value) for value in np.quantile(draws, [0.02, 0.98])]


def _metric_block(panel, shards, bank_index, contrast, seeds):
    if contrast == "replace":
        mask = panel["eligible_replace"].astype(bool)
        x_all = panel["ell_target"] - panel["ell_base"][:, None]
        y_all = np.stack(
            [shard["g_target"] - shard["g_base"][:, None] for shard in shards]
        )
    elif contrast == "append":
        mask = panel["eligible_append"].astype(bool)
        x_all = panel["ell_target"] - panel["ell_core"][:, None]
        y_all = np.stack(
            [shard["g_target"] - shard["g_core"][:, None] for shard in shards]
        )
    else:
        raise ValueError(contrast)
    retained, retained_mask = _valid_groups(mask)
    if contrast == "replace" and (
        mask.shape != (256, 8) or not mask.all() or not retained.all()
    ):
        raise ValueError("primary replace metric requires all 256 by 8 locked rows")
    x = x_all[retained]
    y = y_all[:, retained]
    if len(x) == 0:
        return None
    slope_weights = crossed_resample_weights(
        n_boot=10_000,
        seeds=y.shape[0],
        groups=y.shape[1],
        rng=np.random.default_rng(seeds["slope"]),
    )
    slope_draws = crossed_bootstrap_slope(
        x,
        y,
        mask=retained_mask,
        n_boot=10_000,
        rng=np.random.default_rng(seeds["slope"]),
        resample_weights=slope_weights,
    )
    nrmse_weights = crossed_resample_weights(
        n_boot=10_000,
        seeds=y.shape[0],
        groups=y.shape[1],
        rng=np.random.default_rng(seeds["nrmse"]),
    )
    nrmse_draws = _bootstrap_nrmse(
        x,
        y,
        retained_mask,
        np.random.default_rng(seeds["nrmse"]),
        resample_weights=nrmse_weights,
    )
    nmae_weights = crossed_resample_weights(
        n_boot=10_000,
        seeds=y.shape[0],
        groups=y.shape[1],
        rng=np.random.default_rng(seeds["nmae"]),
    )
    nmae_draws = _bootstrap_nmae(
        x,
        y,
        retained_mask,
        np.random.default_rng(seeds["nmae"]),
        resample_weights=nmae_weights,
    )
    canary = permutation_null_slopes(
        x,
        y,
        mask=retained_mask,
        n_permutations=2_000,
        rng=np.random.default_rng(seeds["permutation"]),
    )
    individual = [
        within_group_slope(x, y[index : index + 1], retained_mask)
        for index in range(y.shape[0])
    ]
    return {
        "bank_index": bank_index,
        "contrast": contrast,
        "eligible_groups": int(len(x)),
        "eligible_rows": int(np.sum(retained_mask)),
        "slope": within_group_slope(x, y, retained_mask),
        "slope_interval": _interval(slope_draws),
        "individual_seed_slopes": individual,
        "nrmse": _nrmse(x, y, retained_mask),
        "nrmse_interval": _interval(nrmse_draws),
        "nmae": _nmae(x, y, retained_mask),
        "nmae_interval": _interval(nmae_draws),
        "permutation_null_interval": [
            float(value) for value in np.quantile(canary, [0.025, 0.975])
        ],
    }


def _mapping_block(panel, shards, bank_index, contrast="replace"):
    if contrast == "replace":
        mask = panel["eligible_replace"].astype(bool)
        source_name = "base"
        boundary_source_field = "boundary_base"
        g_source_field = "g_base"
        kl_source_field = "kl_g_base"
        mix_source_field = "mix_residual_base"
        reconstruction_field = "reconstruction_replace"
    elif contrast == "append":
        mask = panel["eligible_append"].astype(bool)
        source_name = "core"
        boundary_source_field = "boundary_core"
        g_source_field = "g_core"
        kl_source_field = "kl_g_core"
        mix_source_field = "mix_residual_core"
        reconstruction_field = "reconstruction_append"
    else:
        raise ValueError(contrast)
    retained, retained_mask = _valid_groups(mask)
    if int(np.sum(retained)) < 1:
        return {"pass": False, "reason": "no eligible groups"}
    source_group = retained
    target_mask = mask & retained[:, None]
    boundary_source = np.concatenate(
        [shard[boundary_source_field][source_group] for shard in shards]
    )
    boundary_target = np.concatenate([shard["boundary_target"][target_mask] for shard in shards])
    gdiff_source = np.concatenate(
        [
            np.abs(shard[g_source_field] - shard[kl_source_field])[source_group]
            for shard in shards
        ]
    )
    gdiff_target = np.concatenate(
        [np.abs(shard["g_target"] - shard["kl_g_target"])[target_mask] for shard in shards]
    )
    mix_source = np.concatenate(
        [shard[mix_source_field][source_group] for shard in shards]
    )
    mix_target = np.concatenate(
        [shard["mix_residual_target"][target_mask] for shard in shards]
    )
    reconstruction = np.concatenate(
        [shard[reconstruction_field][target_mask] for shard in shards]
    )

    def pair(values):
        return {"median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}

    gdiff = np.concatenate([gdiff_source, gdiff_target])
    result = {
        "bank_index": bank_index,
        "contrast": contrast,
        "eligible_groups": int(np.sum(retained)),
        "eligible_rows": int(np.sum(target_mask)),
        "boundary_rate": float(
            np.mean(np.concatenate([boundary_source, boundary_target]).astype(bool))
        ),
        "coordinate_kl_abs_g": pair(gdiff),
        f"mixture_residual_{source_name}": pair(mix_source),
        "mixture_residual_target": pair(mix_target),
        "identity_reconstruction_residual": pair(reconstruction),
    }
    result["pass"] = bool(
        result["boundary_rate"] == 0.0
        and result["coordinate_kl_abs_g"]["median"] <= 0.10
        and result["coordinate_kl_abs_g"]["p95"] <= 0.30
        and result[f"mixture_residual_{source_name}"]["median"] <= 0.10
        and result[f"mixture_residual_{source_name}"]["p95"] <= 0.30
        and result["mixture_residual_target"]["median"] <= 0.10
        and result["mixture_residual_target"]["p95"] <= 0.30
    )
    return result


def _summarize(*, run_dir: Path, replay_mode: bool):
    configure_determinism(0)
    from .seal import verify_sealed_manifest

    run_dir = run_dir.resolve()
    panel_path = run_dir / "panel.npz"
    panel_preview = load_numeric_npz(panel_path)
    preview_commit, _ = verify_panel_lock(panel_preview)
    require_scientific_run_path(
        run_dir, commit_sha=preview_commit, relative="."
    )
    prediction_dir = run_dir / "predictions"
    derived_dir = run_dir / "derived"
    manifest_path = run_dir / "sealed_manifest.json"
    manifest = verify_sealed_manifest(
        manifest_path, require_archive=not replay_mode
    )
    archive_path = (
        repository_root()
        / "bundles"
        / f"{manifest['content_tree_sha256']}.tar"
    )
    panel, _, verified_shards = verify_derived_ledger(
        panel_path=panel_path,
        prediction_dir=prediction_dir,
        derived_dir=derived_dir,
    )
    commit, _ = verify_panel_lock(panel)
    validations = validate_locked_validations(
        repository_root(), query_banks=panel["query_banks"]
    )
    instrument_pass = True
    seed_record = {
        "replace": {
            "slope": derive_seed(commit, "bootstrap:replace:slope"),
            "nrmse": derive_seed(commit, "bootstrap:replace:nrmse"),
            "nmae": derive_seed(commit, "bootstrap:replace:nmae"),
            "permutation": derive_seed(commit, "permutation:replace"),
        },
        "append": {
            "slope": derive_seed(commit, "bootstrap:append:slope"),
            "nrmse": derive_seed(commit, "bootstrap:append:nrmse"),
            "nmae": derive_seed(commit, "bootstrap:append:nmae"),
            "permutation": derive_seed(commit, "permutation:append"),
        },
        "training_nrmse_difference": derive_seed(
            commit, "bootstrap:training:nrmse-difference"
        ),
    }
    replace = []
    append = []
    mapping = []
    training = []
    for bank_index in (0, 1):
        trained = _fleet_arrays(
            derived_dir, 12_000, bank_index, verified=verified_shards
        )
        initial = _fleet_arrays(derived_dir, 0, bank_index, verified=verified_shards)
        replace_block = _metric_block(
            panel, trained, bank_index, "replace", seed_record["replace"]
        )
        append_block = _metric_block(
            panel, trained, bank_index, "append", seed_record["append"]
        )
        replace_mapping = _mapping_block(panel, trained, bank_index, "replace")
        append_mapping = _mapping_block(panel, trained, bank_index, "append")
        if append_block is not None:
            append_block.update(
                {
                    key: value
                    for key, value in append_mapping.items()
                    if key
                    in {
                        "boundary_rate",
                        "coordinate_kl_abs_g",
                        "mixture_residual_core",
                        "mixture_residual_target",
                        "identity_reconstruction_residual",
                    }
                }
            )
            append_block["mapping_pass"] = append_mapping.get("pass", False)
        replace.append(replace_block)
        append.append(append_block)
        mapping.append(replace_mapping)
        if replace_block is None:
            training.append(None)
            continue
        mask_all = panel["eligible_replace"].astype(bool)
        retained, mask = _valid_groups(mask_all)
        x = (panel["ell_target"] - panel["ell_base"][:, None])[retained]
        y0 = np.stack(
            [shard["g_target"] - shard["g_base"][:, None] for shard in initial]
        )[:, retained]
        y1 = np.stack(
            [shard["g_target"] - shard["g_base"][:, None] for shard in trained]
        )[:, retained]
        difference = _bootstrap_nrmse_difference(
            x,
            y0,
            y1,
            mask,
            np.random.default_rng(seed_record["training_nrmse_difference"]),
        )
        training.append(
            {
                "bank_index": bank_index,
                "step0_nrmse": _nrmse(x, y0, mask),
                "step12000_nrmse": _nrmse(x, y1, mask),
                "difference_step12000_minus_step0": float(
                    _nrmse(x, y1, mask) - _nrmse(x, y0, mask)
                ),
                "difference_interval": _interval(difference),
            }
        )

    identifiable = bool(
        panel["eligible_replace"].shape == (256, 8)
        and panel["eligible_replace"].astype(bool).all()
        and all(
            block is not None
            and block["eligible_groups"] == 256
            and block["eligible_rows"] == 2048
            for block in replace
        )
    )
    mapping_pass = all(block.get("pass") is True for block in mapping)
    reconstruction_confirmation_pass = all(
        block["identity_reconstruction_residual"]["median"] <= 0.10
        and block["identity_reconstruction_residual"]["p95"] <= 0.30
        for block in mapping
    )
    if not all(block is not None for block in replace):
        raise AssertionError("locked primary metric unexpectedly produced no rows")
    decision = decide_primary(
        DecisionInputs(
            instrument_pass=instrument_pass,
            identifiable=identifiable,
            mapping_pass=mapping_pass,
            reconstruction_confirmation_pass=reconstruction_confirmation_pass,
            canary_intervals=tuple(
                tuple(block["permutation_null_interval"]) for block in replace
            ),
            slope_intervals=tuple(tuple(block["slope_interval"]) for block in replace),
            nrmse_upper=tuple(block["nrmse_interval"][1] for block in replace),
        )
    )
    replay_archive = (
        {
            "path": archive_path.relative_to(repository_root()).as_posix(),
            "sha256": sha256_file(archive_path),
            "size": archive_path.stat().st_size,
            "verification": "content-addressed-archive",
        }
        if not replay_mode
        else {
            "path": None,
            "sha256": None,
            "size": None,
            "verification": "extracted-manifest-tree",
        }
    )
    summary = {
        "schema_version": 2,
        "status": (
            "REPLAY_VERIFIED_NONCANONICAL" if replay_mode else "LOCALLY_VERIFIED"
        ),
        "decision": {"code": decision.code, "reason": decision.reason},
        "licensed_scope": (
            "Frozen induced-coordinate response on the tested identifiable-interior AL40 "
            "replace-10 regime; no mechanism or architecture claim."
        ),
        "panel_sha256": sha256_file(panel_path),
        "sealed_manifest_sha256": sha256_file(manifest_path),
        "content_tree_sha256": manifest["content_tree_sha256"],
        "replay_archive": replay_archive,
        "commit_sha": commit,
        "seed_record": seed_record,
        "validations": validations,
        "instrument_pass": instrument_pass,
        "identifiable": identifiable,
        "mapping": mapping,
        "reconstruction_confirmation_pass": reconstruction_confirmation_pass,
        "replace_primary": replace,
        "append_secondary": append,
        "training_secondary": training,
        "resource_totals": manifest["resource_totals"],
    }
    out_path = run_dir / ("replay_summary.json" if replay_mode else "summary.json")
    write_json_atomic(out_path, summary)
    return summary


def summarize(*, run_dir: Path):
    """Write the canonical verdict, requiring the content-addressed archive."""
    return _summarize(run_dir=run_dir, replay_mode=False)


def replay_summarize(*, run_dir: Path):
    """Recompute from an extracted archive without issuing a canonical verdict."""
    return _summarize(run_dir=run_dir, replay_mode=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive")
    derive.add_argument("--panel", type=Path, required=True)
    derive.add_argument("--predictions", type=Path, required=True)
    derive.add_argument("--out", type=Path, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "derive":
        ledger = derive_all(panel_path=args.panel, prediction_dir=args.predictions, out_dir=args.out)
        print(
            json.dumps(
                {"records": len(ledger["records"]), "wall_seconds": ledger["wall_seconds"]},
                sort_keys=True,
            )
        )
    elif args.command == "summarize":
        result = summarize(run_dir=args.run_dir)
        print(json.dumps({"decision": result["decision"], "status": result["status"]}))
    else:
        result = replay_summarize(run_dir=args.run_dir)
        print(json.dumps({"decision": result["decision"], "status": result["status"]}))


if __name__ == "__main__":
    main()
