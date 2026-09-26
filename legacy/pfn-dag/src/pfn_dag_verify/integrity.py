import json
from pathlib import Path

import numpy as np

from .evaluation import _validate_prediction_shard
from .provenance import derive_seed, repository_root, verify_panel_lock
from .registry import expanded_checkpoint_record, load_checkpoint_registry, sha256_file
from .storage import load_numeric_npz


EXPECTED_IDENTITIES = {
    (seed, step, bank)
    for seed in range(16)
    for step in (0, 12_000)
    for bank in (0, 1)
}


def verify_prediction_ledger(
    *, panel_path: Path, prediction_dir: Path
) -> tuple[dict, dict, list[Path]]:
    panel_path = panel_path.resolve()
    prediction_dir = prediction_dir.resolve()
    panel = load_numeric_npz(panel_path)
    commit, _ = verify_panel_lock(panel)
    panel_hash = sha256_file(panel_path)
    ledger_path = prediction_dir / "prediction_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if (
        ledger.get("schema_version") != 1
        or ledger.get("scientific") is not True
        or ledger.get("commit_sha") != commit
        or ledger.get("panel_sha256") != panel_hash
        or ledger.get("run_lock_sha256")
        != bytes(panel["run_lock_sha256"].tolist()).hex()
    ):
        raise ValueError("prediction ledger identity mismatch")
    registry_path = repository_root() / "config" / "checkpoint_registry.json"
    if ledger.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("prediction ledger registry hash mismatch")
    guard_path = prediction_dir.parent / "pre_score_guard.json"
    if ledger.get("pre_score_guard_sha256") != sha256_file(guard_path):
        raise ValueError("prediction ledger pre-score guard hash mismatch")
    guard = json.loads(guard_path.read_text())
    progress_path = prediction_dir.parent / "score_progress.json"
    if ledger.get("score_progress_sha256") != sha256_file(progress_path):
        raise ValueError("prediction ledger score-progress hash mismatch")
    progress = json.loads(progress_path.read_text())
    attempts = progress.get("attempts", [])
    attempt_values_pass = bool(attempts) and all(
        attempt.get("attempt_index") == index
        and attempt.get("status")
        in ({"COMPLETE"} if index == len(attempts) - 1 else {"INTERRUPTED", "COMPLETE"})
        and np.isfinite(float(attempt.get("wall_seconds", np.nan)))
        and float(attempt.get("wall_seconds", -1.0)) >= 0.0
        and int(attempt.get("peak_rss_bytes", -1)) >= 0
        and int(attempt.get("completed_shards", -1)) >= 0
        for index, attempt in enumerate(attempts)
    )
    progress_identities = sorted(
        [[seed, step, bank] for seed, step, bank in EXPECTED_IDENTITIES]
    )
    if (
        set(progress)
        != {
            "schema_version",
            "scientific",
            "commit_sha",
            "panel_sha256",
            "pre_score_guard_sha256",
            "pre_score_guard_wall_seconds",
            "pre_score_guard_peak_rss_bytes",
            "implementation_sha256",
            "attempts",
            "cumulative_wall_seconds",
            "peak_rss_bytes",
            "validated_shard_identities",
        }
        or progress.get("schema_version") != 1
        or progress.get("scientific") is not True
        or progress.get("commit_sha") != commit
        or progress.get("panel_sha256") != panel_hash
        or progress.get("pre_score_guard_sha256") != sha256_file(guard_path)
        or progress.get("pre_score_guard_wall_seconds")
        != guard.get("guard_wall_seconds")
        or progress.get("pre_score_guard_peak_rss_bytes")
        != guard.get("guard_peak_rss_bytes")
        or not np.isfinite(
            float(progress.get("pre_score_guard_wall_seconds", np.nan))
        )
        or float(progress.get("pre_score_guard_wall_seconds", -1.0)) < 0.0
        or int(progress.get("pre_score_guard_peak_rss_bytes", -1)) < 0
        or progress.get("implementation_sha256")
        != sha256_file(repository_root() / "src" / "pfn_dag_verify" / "evaluation.py")
        or not attempt_values_pass
        or not np.isclose(
            float(progress.get("cumulative_wall_seconds", np.nan)),
            float(progress["pre_score_guard_wall_seconds"])
            + sum(float(attempt["wall_seconds"]) for attempt in attempts),
            rtol=1e-12,
            atol=1e-9,
        )
        or int(progress.get("peak_rss_bytes", -1))
        != max(
            int(progress["pre_score_guard_peak_rss_bytes"]),
            *(int(attempt["peak_rss_bytes"]) for attempt in attempts),
        )
        or progress.get("validated_shard_identities") != progress_identities
        or ledger.get("wall_seconds") != progress.get("cumulative_wall_seconds")
        or ledger.get("peak_rss_bytes") != progress.get("peak_rss_bytes")
    ):
        raise ValueError("score progress identity or resource accounting mismatch")
    expected_implementations = {
        "evaluation.py": sha256_file(
            repository_root() / "src" / "pfn_dag_verify" / "evaluation.py"
        ),
        "model.py": sha256_file(
            repository_root() / "src" / "pfn_dag_verify" / "model.py"
        ),
    }
    diagnostics = guard.get("diagnostics", {})
    guard_records = diagnostics.get("records", [])
    guard_identities = {
        (
            int(record.get("seed", -1)),
            int(record.get("step", -1)),
            int(record.get("bank_index", -1)),
            record.get("context_kind"),
        )
        for record in guard_records
    }
    expected_guard_identities = {
        (seed, step, bank, context_kind)
        for seed in range(16)
        for step in (0, 12_000)
        for bank in (0, 1)
        for context_kind in ("core20", "length30")
    }
    expected_sample_indices = {
        "core20": np.random.default_rng(
            derive_seed(commit, "guard:core-context-sample")
        ).choice(int(panel["core"].shape[0]), size=64, replace=False).tolist(),
        "length30": np.random.default_rng(
            derive_seed(commit, "guard:target-context-sample")
        ).choice(
            int(panel["core"].shape[0] * panel["continuations"].shape[1]),
            size=64,
            replace=False,
        ).tolist(),
    }
    numeric_records_pass = all(
        record.get("pass") is True
        and record.get("production_replay_byte_identical") is True
        and record.get("batch_axis_permutation_byte_identical") is True
        and record.get("companion_replacement_byte_identical") is True
        and float(record.get("max_batch_axis_permutation_error", np.inf)) == 0.0
        and float(record.get("max_companion_replacement_error", np.inf)) == 0.0
        and 0.0 <= float(record.get("max_row_permutation_error", np.inf)) <= 1e-6
        and 0.0
        <= float(record.get("descriptive_max_batch_1_vs_64_error", np.inf))
        < np.inf
        and record.get("focal_contexts_checked") == 64
        and record.get("companion_variants") == 16
        and record.get("companion_group_max_errors") == [0.0] * 16
        and record.get("context_kind") in ("core20", "length30")
        for record in guard_records
    )
    aggregate_pass = bool(
        diagnostics.get("production_replay_byte_identical") is True
        and diagnostics.get("batch_axis_permutation_byte_identical") is True
        and diagnostics.get("companion_replacement_byte_identical") is True
        and float(diagnostics.get("max_batch_axis_permutation_error", np.inf)) == 0.0
        and float(diagnostics.get("max_companion_replacement_error", np.inf)) == 0.0
        and 0.0 <= float(diagnostics.get("max_row_permutation_error", np.inf)) <= 1e-6
        and 0.0
        <= float(diagnostics.get("descriptive_max_batch_1_vs_64_error", np.inf))
        < np.inf
    )
    if (
        guard.get("schema_version") != 1
        or guard.get("scientific") is not True
        or guard.get("status") != "COMPLETE"
        or guard.get("pass") is not True
        or guard.get("commit_sha") != commit
        or guard.get("panel_sha256") != panel_hash
        or guard.get("run_lock_sha256")
        != bytes(panel["run_lock_sha256"].tolist()).hex()
        or guard.get("registry_sha256") != sha256_file(registry_path)
        or guard.get("implementation_sha256s") != expected_implementations
        or not np.isfinite(float(guard.get("guard_wall_seconds", np.nan)))
        or float(guard.get("guard_wall_seconds", -1.0)) < 0.0
        or int(guard.get("guard_peak_rss_bytes", -1)) < 0
        or not np.isfinite(float(guard.get("golden_wall_seconds", np.nan)))
        or float(guard.get("golden_wall_seconds", -1.0)) < 0.0
        or ledger.get("golden") != diagnostics
        or ledger.get("golden_wall_seconds") != guard.get("golden_wall_seconds")
        or diagnostics.get("pass") is not True
        or diagnostics.get("identities_checked") != 32
        or diagnostics.get("banks_checked") != 2
        or diagnostics.get("context_kinds") != ["core20", "length30"]
        or diagnostics.get("contexts_checked_per_kind") != 64
        or diagnostics.get("sample_source") != "scientific-panel"
        or diagnostics.get("sample_flat_indices") != expected_sample_indices
        or len(guard_records) != 128
        or guard_identities != expected_guard_identities
        or not numeric_records_pass
        or not aggregate_pass
    ):
        raise ValueError("pre-score guard identity or result mismatch")
    registry = load_checkpoint_registry(registry_path)
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError("prediction ledger must contain exactly 64 records")
    observed = set()
    paths = []
    groups = panel["core"].shape[0]
    continuations = panel["continuations"].shape[1]
    for record in records:
        identity = (
            int(record.get("seed", -1)),
            int(record.get("step", -1)),
            int(record.get("bank_index", -1)),
        )
        if identity in observed:
            raise ValueError(f"duplicate prediction identity: {identity}")
        observed.add(identity)
        seed, step, bank_index = identity
        expected_name = f"pred_s{seed:02d}_step{step:05d}_bank{bank_index}.npz"
        if record.get("path") != expected_name:
            raise ValueError("prediction record path does not match its identity")
        path = prediction_dir / expected_name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"prediction record content mismatch: {expected_name}")
        checkpoint = expanded_checkpoint_record(registry, seed, step)
        shard = load_numeric_npz(path)
        _validate_prediction_shard(
            shard,
            core_count=groups,
            continuation_count=continuations,
            queries=panel["query_banks"][bank_index],
            seed=seed,
            step=step,
            bank_index=bank_index,
            checkpoint_sha256=checkpoint["sha256"],
            panel_sha256=panel_hash,
            scientific=True,
        )
        paths.append(path)
    if observed != EXPECTED_IDENTITIES:
        raise ValueError("prediction ledger identities are incomplete")
    actual_names = {path.name for path in prediction_dir.glob("pred_*.npz")}
    expected_names = {path.name for path in paths}
    if actual_names != expected_names:
        raise ValueError("prediction directory contains a stale or missing shard")
    return panel, ledger, paths
