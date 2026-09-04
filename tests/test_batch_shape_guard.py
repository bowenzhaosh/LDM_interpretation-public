import json
import base64
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

import pfn_dag_verify.evaluation as evaluation
from pfn_dag_verify.evaluation import production_shape_diagnostics
from pfn_dag_verify.model import configure_determinism
from pfn_dag_verify.provenance import _verify_distribution_payloads, verify_runtime


def _normalized_predictions(contexts, queries):
    contexts = np.asarray(contexts)
    values = np.full((len(contexts), len(queries), 100), 0.01, dtype=np.float32)
    signal = np.tanh(contexts.sum(axis=(1, 2))).astype(np.float32) * 1e-4
    values[:, :, 0] += signal[:, None]
    values[:, :, 1] -= signal[:, None]
    return values


def _inputs():
    rng = np.random.default_rng(91)
    sample = rng.normal(size=(64, 30, 2))
    companions = rng.normal(size=(64, 30, 2))
    queries = np.linspace(-3.5, 3.5, 8)
    batch_permutation = np.random.default_rng(92).permutation(64)
    row_permutation = np.random.default_rng(93).permutation(30)
    return sample, companions, queries, batch_permutation, row_permutation


def test_singleton_roundoff_is_descriptive_not_a_production_failure(monkeypatch):
    sample, companions, queries, batch_permutation, row_permutation = _inputs()

    def fake_predict(_model, contexts, queries, *, batch_size):
        values = _normalized_predictions(contexts, queries)
        if batch_size == 1:
            values[:, :, 0] += 2e-6
            values[:, :, 1] -= 2e-6
        return values

    monkeypatch.setattr("pfn_dag_verify.evaluation.predict_probabilities", fake_predict)
    result = production_shape_diagnostics(
        object(),
        sample,
        companions,
        queries,
        batch_permutation=batch_permutation,
        row_permutation=row_permutation,
    )
    assert result["pass"]
    assert result["descriptive_max_batch_1_vs_64_error"] > 1e-6
    assert result["production_replay_byte_identical"]
    assert result["batch_axis_permutation_byte_identical"]
    assert result["companion_replacement_byte_identical"]


def test_fixed_shape_companion_dependence_fails_closed(monkeypatch):
    sample, companions, queries, batch_permutation, row_permutation = _inputs()

    def contaminated_predict(_model, contexts, queries, *, batch_size):
        values = _normalized_predictions(contexts, queries)
        contamination = np.float32(np.asarray(contexts).mean() * 1e-4)
        values[:, :, 0] += contamination
        values[:, :, 1] -= contamination
        return values

    monkeypatch.setattr(
        "pfn_dag_verify.evaluation.predict_probabilities", contaminated_predict
    )
    result = production_shape_diagnostics(
        object(),
        sample,
        companions,
        queries,
        batch_permutation=batch_permutation,
        row_permutation=row_permutation,
    )
    assert not result["pass"]
    assert not result["companion_replacement_byte_identical"]


def test_tail_conditioned_nonsentinel_batch_coupling_fails_closed(monkeypatch):
    sample, companions, queries, batch_permutation, row_permutation = _inputs()
    sample[4] = 1.0
    companions[:] = 0.0

    def tail_contaminated_predict(_model, contexts, queries, *, batch_size):
        contexts = np.asarray(contexts)
        values = _normalized_predictions(contexts, queries)
        affected = contexts.sum(axis=(1, 2)) > 12
        contamination = np.float32(contexts.mean() * 1e-4)
        values[affected, :, 0] += contamination
        values[affected, :, 1] -= contamination
        return values

    monkeypatch.setattr(
        "pfn_dag_verify.evaluation.predict_probabilities", tail_contaminated_predict
    )
    result = production_shape_diagnostics(
        object(),
        sample,
        companions,
        queries,
        batch_permutation=batch_permutation,
        row_permutation=row_permutation,
    )
    assert not result["pass"]
    assert not result["companion_replacement_byte_identical"]
    assert result["focal_contexts_checked"] == 64


def test_configure_determinism_locks_every_live_inference_setting():
    configure_determinism(17)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.mha.get_fastpath_enabled()


def test_runtime_verification_fails_when_live_matmul_setting_drifts():
    configure_determinism(19)
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    verify_runtime(root)
    torch.set_float32_matmul_precision("medium")
    try:
        with pytest.raises(RuntimeError, match="torch_float32_matmul_precision"):
            verify_runtime(root)
    finally:
        configure_determinism(19)


def test_distribution_payload_verifier_rejects_bytes_changed_behind_record(tmp_path):
    package = tmp_path / "site" / "demo"
    info = tmp_path / "site" / "demo-1.0.dist-info"
    package.mkdir(parents=True)
    info.mkdir()
    payload = package / "__init__.py"
    payload.write_bytes(b"locked\n")
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).decode().rstrip("=")
    record = info / "RECORD"
    record.write_text(
        f"demo/__init__.py,sha256={encoded},{payload.stat().st_size}\n"
        "demo-1.0.dist-info/RECORD,,\n"
    )

    class FakeDistribution:
        def locate_file(self, relative):
            return tmp_path / "site" / relative

    entries = []
    for relative in ("demo/__init__.py", "demo-1.0.dist-info/RECORD"):
        path = tmp_path / "site" / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    tree = hashlib.sha256()
    import json as _json
    for entry in sorted(entries, key=lambda value: value["path"]):
        tree.update((_json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode())
    locked = {
        "payload_files": len(entries),
        "payload_bytes": sum(entry["size"] for entry in entries),
        "payload_tree_sha256": tree.hexdigest(),
    }
    verified = _verify_distribution_payloads(FakeDistribution(), record, "demo", locked)
    assert payload.resolve() in verified
    payload.write_bytes(b"tamper\n")
    with pytest.raises(RuntimeError, match="payload mismatch"):
        _verify_distribution_payloads(FakeDistribution(), record, "demo", locked)


def test_nonfleet_golden_guard_uses_a_full_dedicated_context_bank(monkeypatch):
    panel = {
        "core": np.zeros((8, 20, 2), dtype=np.float64),
        "reference": np.zeros((8, 10, 2), dtype=np.float64),
        "continuations": np.zeros((8, 8, 10, 2), dtype=np.float64),
        "query_banks": np.stack(
            [np.linspace(-4.0, 4.0, 8), np.linspace(-3.5, 3.5, 8)]
        ),
        "commit_sha": np.frombuffer(b"2" * 40, dtype=np.uint8),
    }
    observed_shapes = []
    monkeypatch.setattr(evaluation, "expanded_checkpoint_record", lambda *args: {})
    monkeypatch.setattr(evaluation, "load_registered_checkpoint", lambda record: object())

    def fake_diagnostics(model, sample, companion, queries, **kwargs):
        observed_shapes.append(sample.shape)
        return {
            "pass": True,
            "production_replay_byte_identical": True,
            "batch_axis_permutation_byte_identical": True,
            "companion_replacement_byte_identical": True,
            "max_batch_axis_permutation_error": 0.0,
            "max_companion_replacement_error": 0.0,
            "max_row_permutation_error": 0.0,
            "descriptive_max_batch_1_vs_64_error": 0.0,
        }

    monkeypatch.setattr(evaluation, "production_shape_diagnostics", fake_diagnostics)
    result = evaluation.golden_replay(panel, {}, fleet_guard=False)
    assert result["pass"]
    assert result["sample_source"] == "dedicated-deterministic-smoke"
    assert observed_shapes == [(64, 20, 2), (64, 30, 2)] * 2


def test_historical_panel_commit_is_rejected_before_scientific_writes(monkeypatch):
    monkeypatch.setattr(evaluation, "current_head", lambda root: "b" * 40)
    with pytest.raises(ValueError, match="current repository HEAD"):
        evaluation._require_current_panel_commit("a" * 40, __import__("pathlib").Path("."))


def test_known_failed_panel_is_rejected_even_while_it_is_current_head(monkeypatch):
    failed = "d0b049d6241845e55443f4950e52b70644b2b1ab"
    monkeypatch.setattr(evaluation, "current_head", lambda root: failed)
    with pytest.raises(ValueError, match="immutable failed stream"):
        evaluation._require_current_panel_commit(
            failed, __import__("pathlib").Path(".")
        )


def test_resumed_score_progress_accumulates_interrupted_attempt_wall_time(
    tmp_path, monkeypatch
):
    path = tmp_path / "score_progress.json"
    identity = {
        "schema_version": 1,
        "scientific": True,
        "commit_sha": "3" * 40,
        "panel_sha256": "4" * 64,
        "pre_score_guard_sha256": "5" * 64,
        "pre_score_guard_wall_seconds": 0.0,
        "pre_score_guard_peak_rss_bytes": 0,
        "implementation_sha256": evaluation.sha256_file(Path(evaluation.__file__)),
    }
    prior = {
        **identity,
        "attempts": [
            {
                "attempt_index": 0,
                "status": "RUNNING",
                "wall_seconds": 2600.0,
                "peak_rss_bytes": 100,
                "completed_shards": 63,
            }
        ],
        "cumulative_wall_seconds": 2600.0,
        "peak_rss_bytes": 100,
        "validated_shard_identities": [],
    }
    path.write_text(json.dumps(prior))
    progress = evaluation._start_score_progress(
        path,
        scientific=True,
        commit_sha=identity["commit_sha"],
        panel_sha256=identity["panel_sha256"],
        pre_score_guard_sha256=identity["pre_score_guard_sha256"],
        pre_score_guard_wall_seconds=identity["pre_score_guard_wall_seconds"],
        pre_score_guard_peak_rss_bytes=identity[
            "pre_score_guard_peak_rss_bytes"
        ],
    )
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: 3000.0)
    monkeypatch.setattr(evaluation, "_peak_rss_bytes", lambda: 200)
    progress = evaluation._update_score_progress(
        path,
        progress,
        attempt_started=2800.0,
        records=[],
        status="COMPLETE",
    )
    assert progress["attempts"][0]["status"] == "INTERRUPTED"
    assert progress["cumulative_wall_seconds"] == 2800.0
    assert progress["peak_rss_bytes"] == 200


def test_completed_guard_cost_seeds_first_score_progress_attempt(tmp_path, monkeypatch):
    path = tmp_path / "score_progress.json"
    monkeypatch.setattr(evaluation, "_peak_rss_bytes", lambda: 200)
    progress = evaluation._start_score_progress(
        path,
        scientific=True,
        commit_sha="3" * 40,
        panel_sha256="4" * 64,
        pre_score_guard_sha256="5" * 64,
        pre_score_guard_wall_seconds=2600.0,
        pre_score_guard_peak_rss_bytes=100,
    )
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: 3000.0)
    progress = evaluation._update_score_progress(
        path,
        progress,
        attempt_started=2800.0,
        records=[],
        status="COMPLETE",
    )
    assert progress["cumulative_wall_seconds"] == 2800.0
    assert progress["peak_rss_bytes"] == 200


def test_pre_score_guard_persists_python_exceptions(tmp_path, monkeypatch):
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(
        panel_path,
        commit_sha=np.frombuffer(b"1" * 40, dtype=np.uint8),
        run_lock_sha256=np.zeros(32, dtype=np.uint8),
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(evaluation, "load_checkpoint_registry", lambda path: {})

    def fail_guard(*args, **kwargs):
        raise RuntimeError("injected guard failure")

    monkeypatch.setattr(evaluation, "golden_replay", fail_guard)
    out_dir = tmp_path / "run" / "predictions"
    with pytest.raises(RuntimeError, match="injected guard failure"):
        evaluation.score_checkpoints(
            panel_path=panel_path,
            registry_path=registry_path,
            out_dir=out_dir,
            seeds=[0],
            steps=[0],
            scientific=False,
        )
    guard = json.loads((out_dir.parent / "pre_score_guard.json").read_text())
    assert guard["status"] == "ERROR"
    assert guard["pass"] is False
    assert guard["error_type"] == "RuntimeError"
    assert guard["error_message"] == "injected guard failure"
    assert guard["guard_peak_rss_bytes"] >= 0
    assert not out_dir.exists()


def test_pre_score_guard_persists_registry_loader_exceptions(tmp_path, monkeypatch):
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(
        panel_path,
        commit_sha=np.frombuffer(b"1" * 40, dtype=np.uint8),
        run_lock_sha256=np.zeros(32, dtype=np.uint8),
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")

    def fail_registry(path):
        raise ValueError("injected registry hash failure")

    monkeypatch.setattr(evaluation, "load_checkpoint_registry", fail_registry)
    out_dir = tmp_path / "run" / "predictions"
    with pytest.raises(ValueError, match="injected registry hash failure"):
        evaluation.score_checkpoints(
            panel_path=panel_path,
            registry_path=registry_path,
            out_dir=out_dir,
            seeds=[0],
            steps=[0],
            scientific=False,
        )
    guard = json.loads((out_dir.parent / "pre_score_guard.json").read_text())
    assert guard["status"] == "ERROR"
    assert guard["pass"] is False
    assert guard["error_type"] == "ValueError"
    assert guard["error_message"] == "injected registry hash failure"
    assert guard["guard_peak_rss_bytes"] >= 0
    assert not out_dir.exists()
