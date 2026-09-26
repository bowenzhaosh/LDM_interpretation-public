import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pfn_dag_verify.evaluation as evaluation
import pfn_dag_verify.analysis as analysis
import pfn_dag_verify.seal as seal
from pfn_dag_verify.legacy_compare import compare as compare_legacy
from pfn_dag_verify.analysis import (
    _bootstrap_nmae,
    _nmae,
    _start_derive_progress,
    _update_derive_progress,
    _validate_derive_progress,
)
from pfn_dag_verify.evaluation import _validate_prediction_shard
from pfn_dag_verify.integrity import EXPECTED_IDENTITIES
from pfn_dag_verify.provenance import (
    REQUIRED_VALIDATIONS,
    RUN_LOCK_CLAIM_SCOPE,
    RUN_LOCK_SETTINGS,
    derive_seed,
    enforce_cost_gate,
    evaluation_root,
    require_scientific_run_path,
    scientific_run_directory,
    recompute_smoke_projections,
    validate_run_lock_metadata,
)
from pfn_dag_verify.registry import sha256_file
from pfn_dag_verify.seal import (
    _canonical_tree_hash,
    _create_archive,
    _create_replay_material,
    _validated_panel_resources,
    _verify_archive,
    _write_manifest_and_archive,
)


TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_legacy_comparison_rejects_any_unlocked_source_path(tmp_path):
    candidate = tmp_path / "stage1_functional_law.py"
    candidate.write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="requires locked source"):
        compare_legacy(candidate, np.linspace(-4.0, 4.0, 8), n_contexts=1)


def test_run_lock_claim_scope_tamper_fails_closed():
    lock = {
        "schema_version": 1,
        "files": {"placeholder": "00" * 32},
        "required_validations": sorted(REQUIRED_VALIDATIONS),
        "settings": RUN_LOCK_SETTINGS,
        "claim_scope": RUN_LOCK_CLAIM_SCOPE,
    }
    validate_run_lock_metadata(lock)
    lock["claim_scope"] = "all PFN behavior"
    with pytest.raises(ValueError, match="claim scope"):
        validate_run_lock_metadata(lock)


def test_cost_gate_recomputes_reported_projections_from_measured_components():
    smoke = {
        "projection_multipliers": {
            "group_ratio": 32.0,
            "shard_ratio": 16.0,
            "guard_ratio": 32.0,
            "safety_factor": 1.25,
        },
        "components": {
            "panel_wall_seconds": 1.0,
            "score_wall_seconds": 2.0,
            "guard_wall_seconds": 1.0,
            "non_guard_score_wall_seconds": 1.0,
            "derive_wall_seconds": 1.0,
            "panel_bytes": 100,
            "prediction_shard_bytes": 100,
            "prediction_metadata_bytes": 10,
            "prediction_bytes": 110,
            "derived_bytes": 100,
            "replay_bundle_bytes": 1_000,
            "tracked_repository_bytes": 2_000,
            "measured_peak_rss_bytes": 100,
        },
    }
    wall, peak, raw = recompute_smoke_projections(smoke)
    smoke.update(
        projected_wall_seconds=wall,
        projected_peak_rss_bytes=peak,
        projected_raw_bytes=raw,
    )
    enforce_cost_gate(smoke)
    smoke["projected_wall_seconds"] = 0.0
    with pytest.raises(RuntimeError, match="does not match components"):
        enforce_cost_gate(smoke)


def test_derive_progress_preserves_interrupted_attempt_cost(monkeypatch, tmp_path):
    path = tmp_path / "derive_progress.json"
    prediction_hash = "12" * 32
    panel_hash = "34" * 32
    commit = TEST_COMMIT
    clock = iter([10.0, 15.0])
    monkeypatch.setattr(analysis.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(analysis, "_peak_rss_bytes", lambda: 123)
    partial = [{"seed": 0, "step": 0, "bank_index": 0}]
    complete = [
        {"seed": seed, "step": step, "bank_index": bank}
        for seed, step, bank in sorted(EXPECTED_IDENTITIES)
    ]

    first = _start_derive_progress(
        path,
        commit_sha=commit,
        panel_sha256=panel_hash,
        prediction_ledger_sha256=prediction_hash,
    )
    _update_derive_progress(
        path,
        first,
        attempt_started=0.0,
        records=partial,
        status="RUNNING",
    )
    second = _start_derive_progress(
        path,
        commit_sha=commit,
        panel_sha256=panel_hash,
        prediction_ledger_sha256=prediction_hash,
    )
    final = _update_derive_progress(
        path,
        second,
        attempt_started=10.0,
        records=complete,
        status="COMPLETE",
    )
    assert [attempt["status"] for attempt in final["attempts"]] == [
        "INTERRUPTED",
        "COMPLETE",
    ]
    assert final["cumulative_wall_seconds"] == 15.0
    _validate_derive_progress(
        final,
        commit_sha=commit,
        panel_sha256=panel_hash,
        prediction_ledger_sha256=prediction_hash,
        require_complete=True,
    )
    final["attempts"][0]["wall_seconds"] = -1.0
    with pytest.raises(ValueError, match="resource history"):
        _validate_derive_progress(
            final,
            commit_sha=commit,
            panel_sha256=panel_hash,
            prediction_ledger_sha256=prediction_hash,
            require_complete=True,
        )


def test_panel_resource_metadata_rejects_negative_wall(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    panel_path = run_dir / "panel.npz"
    panel_path.write_bytes(b"panel")
    run_lock_hash = "56" * 32
    panel = {
        "core": np.zeros((2, 20, 2)),
        "continuations": np.zeros((2, 8, 10, 2)),
        "eligible_replace": np.ones((2, 8), dtype=np.uint8),
        "eligible_append": np.ones((2, 8), dtype=np.uint8),
        "selection_mode": np.asarray(1),
        "scientific": np.asarray(1),
        "evaluation_root": np.asarray(123, dtype=np.uint64),
        "candidate_core_seed": np.zeros(3, dtype=np.uint64),
        "candidate_block_seed": np.zeros(4, dtype=np.uint64),
    }
    metadata = {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(evaluation.__file__)),
        "commit_sha": TEST_COMMIT,
        "n_groups": 2,
        "n_continuations": 8,
        "eligible_replace_groups": 2,
        "eligible_append_groups": 2,
        "interior_selected": True,
        "scientific": True,
        "run_lock_sha256": run_lock_hash,
        "evaluation_root": 123,
        "candidate_core_count": 3,
        "candidate_block_count": 4,
        "panel_sha256": sha256_file(panel_path),
        "panel_bytes": panel_path.stat().st_size,
        "wall_seconds": 1.0,
        "peak_rss_bytes": 100,
    }
    (run_dir / "panel.json").write_text(json.dumps(metadata))
    assert _validated_panel_resources(
        run_dir, panel, commit=TEST_COMMIT, run_lock_hash=run_lock_hash
    ) == (1.0, 100)
    metadata["wall_seconds"] = -1e9
    (run_dir / "panel.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _validated_panel_resources(
            run_dir, panel, commit=TEST_COMMIT, run_lock_hash=run_lock_hash
        )


def test_scientific_paths_are_bound_to_the_first_seven_commit_characters():
    expected = scientific_run_directory(TEST_COMMIT)
    assert expected.name == "scientific-0123456"
    assert (
        require_scientific_run_path(
            expected / "panel.npz",
            commit_sha=TEST_COMMIT,
            relative="panel.npz",
        )
        == expected / "panel.npz"
    )
    with pytest.raises(ValueError, match="commit-named"):
        require_scientific_run_path(
            expected.parent / "scientific-d0b049d" / "panel.npz",
            commit_sha=TEST_COMMIT,
            relative="panel.npz",
        )


def test_scientific_path_rejects_a_commit_directory_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    runs = tmp_path / "repo" / "runs"
    runs.mkdir(parents=True)
    linked = runs / "scientific-0123456"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        require_scientific_run_path(
            linked / "panel.npz",
            commit_sha=TEST_COMMIT,
            relative="panel.npz",
            repo=tmp_path / "repo",
        )


def test_seed_chain_matches_locked_two_stage_vectors_and_keeps_high_bit():
    root = int.from_bytes(
        hashlib.sha256(
            (TEST_COMMIT + "pfn-dag-essential-evaluation-v2").encode("ascii")
        ).digest()[:8],
        "big",
    )
    assert evaluation_root(TEST_COMMIT) == root == 4_849_711_420_112_403_902
    assert derive_seed(TEST_COMMIT, "groups") == 2_118_269_529_734_854_837
    assert (
        derive_seed(TEST_COMMIT, "bootstrap:replace:slope")
        == 13_438_655_072_475_084_798
    )
    assert derive_seed(TEST_COMMIT, "permutation:replace") > 2**63


def _valid_prediction_shard():
    probabilities = np.full((2, 8, 100), 0.01, dtype=np.float32)
    return {
        "p_core": probabilities.copy(),
        "p_base": probabilities.copy(),
        "p_target": np.broadcast_to(probabilities[:, None], (2, 3, 8, 100)).copy(),
        "query_bank": np.arange(8, dtype=np.float64),
        "checkpoint_sha256": np.frombuffer(bytes.fromhex("11" * 32), dtype=np.uint8),
        "panel_sha256": np.frombuffer(bytes.fromhex("22" * 32), dtype=np.uint8),
        "seed": np.asarray(4, dtype=np.int16),
        "step": np.asarray(12_000, dtype=np.int32),
        "bank_index": np.asarray(1, dtype=np.int8),
        "scientific": np.asarray(1, dtype=np.int8),
    }


def _check_prediction(shard):
    _validate_prediction_shard(
        shard,
        core_count=2,
        continuation_count=3,
        queries=np.arange(8, dtype=np.float64),
        seed=4,
        step=12_000,
        bank_index=1,
        checkpoint_sha256="11" * 32,
        panel_sha256="22" * 32,
        scientific=True,
    )


def test_prediction_resume_schema_fails_closed_on_every_load_bearing_identity():
    shard = _valid_prediction_shard()
    _check_prediction(shard)
    mutations = []
    wrong_target = _valid_prediction_shard()
    wrong_target["p_target"][:] = 0
    mutations.append(wrong_target)
    wrong_query = _valid_prediction_shard()
    wrong_query["query_bank"] = wrong_query["query_bank"][::-1]
    mutations.append(wrong_query)
    wrong_panel = _valid_prediction_shard()
    wrong_panel["panel_sha256"] = np.zeros(32, dtype=np.uint8)
    mutations.append(wrong_panel)
    wrong_checkpoint = _valid_prediction_shard()
    wrong_checkpoint["checkpoint_sha256"] = np.zeros(32, dtype=np.uint8)
    mutations.append(wrong_checkpoint)
    wrong_shape = _valid_prediction_shard()
    wrong_shape["p_base"] = wrong_shape["p_base"][:1]
    mutations.append(wrong_shape)
    wrong_identity = _valid_prediction_shard()
    wrong_identity["step"] = np.asarray(0, dtype=np.int32)
    mutations.append(wrong_identity)
    for mutation in mutations:
        with pytest.raises(ValueError):
            _check_prediction(mutation)


class _FakeOracle:
    def __init__(self, queries, quadrature):
        pass

    @staticmethod
    def log_evidence(context):
        return float(context[-1, 0])

    def evaluate(self, context):
        return SimpleNamespace(ell=self.log_evidence(context), js=0.2)


def test_selected_cohort_uses_exactly_first_nine_passing_blocks(monkeypatch):
    monkeypatch.setattr(evaluation, "GridOracle", _FakeOracle)
    monkeypatch.setattr(evaluation, "sample_valid_sigma", lambda rng: np.eye(2))
    monkeypatch.setattr(
        evaluation,
        "sigma_to_params",
        lambda covariance, graph: SimpleNamespace(beta=0.1, b_root=1.0, b_effect=1.0),
    )

    def fake_context(rng, graph, parameters, rows):
        value = rng.uniform(-0.9, 0.9)
        result = np.zeros((rows, 2), dtype=np.float64)
        result[:, 0] = value
        return result

    monkeypatch.setattr(evaluation, "sample_context", fake_context)
    groups, selection = evaluation._select_interior_groups(
        commit_sha=TEST_COMMIT,
        query_banks=np.stack([np.arange(8), np.arange(8) + 0.5]),
        n_groups=2,
        n_continuations=8,
        max_core_candidates=20,
        max_blocks_per_core=20,
        min_within_group_sd=0.0,
    )
    accepted = np.flatnonzero(selection["candidate_core_reason"] == 0)
    np.testing.assert_array_equal(
        selection["candidate_core_acceptance_rank"][accepted], np.arange(2)
    )
    for index, core_id in enumerate(accepted):
        rows = np.flatnonzero(
            (selection["candidate_block_core_index"] == core_id)
            & (selection["candidate_block_reason"] == 0)
        )
        np.testing.assert_array_equal(
            selection["candidate_block_eligible_rank"][rows], np.arange(9)
        )
        np.testing.assert_array_equal(groups[index].reference, selection["candidate_block_context"][rows[0]])
        np.testing.assert_array_equal(
            groups[index].continuations, selection["candidate_block_context"][rows[1:]]
        )


def test_selected_cohort_cap_failure_is_nonzero(monkeypatch):
    monkeypatch.setattr(evaluation, "GridOracle", _FakeOracle)
    monkeypatch.setattr(evaluation, "sample_valid_sigma", lambda rng: np.eye(2))
    monkeypatch.setattr(
        evaluation,
        "sigma_to_params",
        lambda covariance, graph: SimpleNamespace(beta=0.1, b_root=1.0, b_effect=1.0),
    )
    monkeypatch.setattr(
        evaluation,
        "sample_context",
        lambda rng, graph, parameters, rows: np.zeros((rows, 2)),
    )
    with pytest.raises(RuntimeError, match="INCONCLUSIVE_IDENTIFIABILITY"):
        evaluation._select_interior_groups(
            commit_sha=TEST_COMMIT,
            query_banks=np.stack([np.arange(8), np.arange(8) + 0.5]),
            n_groups=1,
            n_continuations=8,
            max_core_candidates=1,
            max_blocks_per_core=8,
            min_within_group_sd=0.0,
        )


def test_selected_cohort_uses_caller_locked_js_threshold(monkeypatch):
    monkeypatch.setattr(evaluation, "GridOracle", _FakeOracle)
    monkeypatch.setattr(evaluation, "sample_valid_sigma", lambda rng: np.eye(2))
    monkeypatch.setattr(
        evaluation,
        "sigma_to_params",
        lambda covariance, graph: SimpleNamespace(
            beta=0.1, b_root=1.0, b_effect=1.0
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "sample_context",
        lambda rng, graph, parameters, rows: np.zeros((rows, 2)),
    )
    with pytest.raises(RuntimeError, match="INCONCLUSIVE_IDENTIFIABILITY"):
        evaluation._select_interior_groups(
            commit_sha=TEST_COMMIT,
            query_banks=np.stack([np.arange(8), np.arange(8) + 0.5]),
            n_groups=1,
            n_continuations=2,
            max_core_candidates=1,
            max_blocks_per_core=8,
            min_within_group_sd=0.0,
            minimum_endpoint_js=0.21,
        )


def test_content_tree_hash_is_order_sensitive_only_after_canonicalization():
    entries = [
        {"scope": "run", "path": "b", "size": 1, "sha256": "11" * 32},
        {"scope": "run", "path": "a", "size": 2, "sha256": "22" * 32},
    ]
    ordered = sorted(entries, key=lambda value: (value["scope"], value["path"]))
    first = _canonical_tree_hash(ordered)
    changed = [dict(value) for value in ordered]
    changed[0]["size"] += 1
    assert first != _canonical_tree_hash(changed)


def test_point_and_weighted_bootstrap_nmae_use_the_same_quantile_convention():
    x = np.array([[0.0, 1.0], [2.0, 100.0]])
    y = x[None, :, :] + 1.0
    mask = np.ones_like(x, dtype=bool)
    point = _nmae(x, y, mask)
    draws = _bootstrap_nmae(
        x,
        y,
        mask,
        np.random.default_rng(1),
        n_boot=1,
        resample_weights=(np.ones((1, 1), dtype=int), np.ones((1, 2), dtype=int)),
    )
    assert point == pytest.approx(0.5)
    assert draws[0] == pytest.approx(point, abs=1e-12)


def test_content_addressed_archive_contains_and_verifies_every_manifest_file(tmp_path):
    root = tmp_path / "repo"
    run_dir = root / "runs" / "one"
    run_dir.mkdir(parents=True)
    repository_file = root / "source.py"
    run_file = run_dir / "panel.npz"
    repository_file.write_bytes(b"source")
    run_file.write_bytes(b"raw")
    entries = [
        {
            "scope": "repository",
            "path": "source.py",
            "size": repository_file.stat().st_size,
            "sha256": sha256_file(repository_file),
        },
        {
            "scope": "run",
            "path": "panel.npz",
            "size": run_file.stat().st_size,
            "sha256": sha256_file(run_file),
        },
    ]
    entries.sort(key=lambda value: (value["scope"], value["path"]))
    manifest = {"content_tree_sha256": _canonical_tree_hash(entries), "files": entries}
    manifest_path = run_dir / "sealed_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    archive = _create_archive(
        manifest_path=manifest_path,
        entries=entries,
        root=root,
        run_dir=run_dir,
    )
    assert archive.name == f"{manifest['content_tree_sha256']}.tar"
    _verify_archive(archive_path=archive, manifest_path=manifest_path, entries=entries)
    repository_file.write_bytes(b"changed")
    _verify_archive(archive_path=archive, manifest_path=manifest_path, entries=entries)


def test_sealed_raw_total_is_exact_archive_size_including_tar_overhead(tmp_path):
    root = tmp_path / "repo"
    run_dir = root / "runs" / "one"
    run_dir.mkdir(parents=True)
    run_file = run_dir / "panel.npz"
    run_file.write_bytes(b"x")
    entries = [
        {
            "scope": "run",
            "path": "panel.npz",
            "size": 1,
            "sha256": sha256_file(run_file),
        }
    ]
    manifest = {
        "content_tree_sha256": _canonical_tree_hash(entries),
        "files": entries,
        "resource_totals": {"raw_bytes": 1},
    }
    manifest_path = run_dir / "sealed_manifest.json"
    archive = _write_manifest_and_archive(
        manifest_path=manifest_path,
        manifest=manifest,
        entries=entries,
        root=root,
        run_dir=run_dir,
    )
    assert archive.stat().st_size > 1
    assert manifest["resource_totals"]["raw_bytes"] == archive.stat().st_size
    assert json.loads(manifest_path.read_text()) == manifest


def test_replay_bundle_restores_exact_commit_without_original_git_directory(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Audit"], cwd=root, check=True)
    source = root / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "locked"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_dir = root / "runs" / f"scientific-{commit[:7]}"
    run_dir.mkdir(parents=True)
    replay_files = _create_replay_material(root, run_dir, commit)
    entries = []
    for relative in replay_files:
        path = run_dir / "replay" / relative
        entries.append(
            {
                "scope": "replay",
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    entries.sort(key=lambda value: (value["scope"], value["path"]))
    manifest = {"content_tree_sha256": _canonical_tree_hash(entries), "files": entries}
    manifest_path = run_dir / "sealed_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    archive = _create_archive(
        manifest_path=manifest_path,
        entries=entries,
        root=root,
        run_dir=run_dir,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(extracted, filter="data")
    restored = tmp_path / "restored"
    subprocess.run(
        ["git", "clone", "-q", str(extracted / "replay" / "source.bundle"), str(restored)],
        check=True,
    )
    restored_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=restored,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert restored_commit == commit
    assert (restored / "source.py").read_text() == "value = 1\n"
    replay_readme = (extracted / "replay" / "README.txt").read_text()
    assert "analysis replay --run-dir" in replay_readme
    assert "REPLAY_VERIFIED_NONCANONICAL" in replay_readme


def _stop_summary_at_archive_gate(monkeypatch, tmp_path):
    commit = TEST_COMMIT
    run_dir = tmp_path / "repo" / "runs" / f"scientific-{commit[:7]}"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(analysis, "load_numeric_npz", lambda path: {"commit_sha": commit})
    monkeypatch.setattr(analysis, "verify_panel_lock", lambda panel: (commit, {}))
    monkeypatch.setattr(analysis, "require_scientific_run_path", lambda *args, **kwargs: None)
    seen = []

    def missing_archive(path, *, require_archive=True):
        seen.append(require_archive)
        raise FileNotFoundError("content-addressed archive is missing")

    monkeypatch.setattr(seal, "verify_sealed_manifest", missing_archive)
    return run_dir, seen


def test_every_canonical_summary_entrypoint_requires_archive(monkeypatch, tmp_path):
    run_dir, seen = _stop_summary_at_archive_gate(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="archive is missing"):
        analysis.summarize(run_dir=run_dir)
    with pytest.raises(FileNotFoundError, match="archive is missing"):
        analysis.main(["summarize", "--run-dir", str(run_dir)])
    assert seen == [True, True]
    assert not (run_dir / "summary.json").exists()


def test_removed_offline_flag_cannot_bypass_canonical_summary(tmp_path):
    with pytest.raises(SystemExit):
        analysis.main(
            ["summarize", "--run-dir", str(tmp_path), "--offline-archive"]
        )


def test_replay_entrypoint_is_the_only_manifest_tree_mode(monkeypatch, tmp_path):
    run_dir, seen = _stop_summary_at_archive_gate(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="archive is missing"):
        analysis.replay_summarize(run_dir=run_dir)
    assert seen == [False]
    assert not (run_dir / "summary.json").exists()
