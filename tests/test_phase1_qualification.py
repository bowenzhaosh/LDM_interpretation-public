import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pfn_dag_verify.phase1_qualification as qualification
import pfn_dag_verify.phase1_qualification_verify as independent_verifier
import pfn_dag_verify.phase1_ordering as ordering_runner
from pfn_dag_verify.phase1_ordering import (
    _sha256_file,
    _sha256_json,
    production_qualification_protocol,
    production_qualification_protocol_v2,
    production_qualification_protocol_v3,
)
from pfn_dag_verify.storage import write_json_atomic, write_numeric_npz_atomic


ROOT = Path(__file__).resolve().parents[1]


def test_independent_verifier_uses_exact_tag_namespace_and_commit_peel(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        if arguments[1:3] == ["cat-file", "-t"]:
            return SimpleNamespace(stdout="tag\n")
        return SimpleNamespace(stdout="abc123\n")

    monkeypatch.setattr(independent_verifier.subprocess, "run", fake_run)
    independent_verifier._verify_annotated_tag(
        tmp_path, "phase1-ordering-qualification-v3", "abc123"
    )
    assert calls == [
        [
            "git",
            "cat-file",
            "-t",
            "refs/tags/phase1-ordering-qualification-v3",
        ],
        [
            "git",
            "rev-parse",
            "--verify",
            "refs/tags/phase1-ordering-qualification-v3^{commit}",
        ],
    ]


def test_independent_verifier_maps_source_inventory_from_arbitrary_checkout_name(
    tmp_path, monkeypatch
):
    relative = "PHASE1_ORDERING_PREREG.md"
    payload = b"frozen preregistration"
    inventory = {
        "/engrfs/project/class/zhao.b/"
        "LDM_interpretation_phase1_qv3_fc0b8eb/PHASE1_ORDERING_PREREG.md":
        independent_verifier._sha256_bytes(payload)
    }
    monkeypatch.setattr(
        independent_verifier,
        "_git_blob",
        lambda repo, commit, source_relative: payload
        if source_relative == relative
        else b"wrong",
    )
    independent_verifier._verify_source_inventory(
        inventory, tmp_path, "artifact-source-commit", {relative}
    )


def test_verifier_fix_pins_exact_frozen_joined_artifacts(tmp_path):
    joined = tmp_path / "joined"
    joined.mkdir()
    payloads = {
        "COMPLETE.json": b"complete",
        "qualification_raw.npz": b"raw",
        "qualification_summary.json": b"summary",
    }
    for name, payload in payloads.items():
        (joined / name).write_bytes(payload)
    expected = {
        name: independent_verifier._sha256_bytes(payload)
        for name, payload in payloads.items()
    }
    independent_verifier._verify_frozen_joined_artifacts(tmp_path, expected)
    (joined / "qualification_raw.npz").write_bytes(b"raw-mutated")
    with pytest.raises(RuntimeError, match="frozen joined artifact hash mismatch"):
        independent_verifier._verify_frozen_joined_artifacts(tmp_path, expected)
    with pytest.raises(RuntimeError, match="missing frozen joined artifact"):
        independent_verifier._verify_frozen_joined_artifacts(
            tmp_path / "wrong-root", expected
        )


def _write_shard(
    run_dir: Path,
    bank_index: int,
    fail_first_candidate: bool,
    protocol_version: int = 1,
) -> None:
    run_dir.mkdir(parents=True)
    suffix = "" if protocol_version == 1 else f"_v{protocol_version}"
    config_path = (
        ROOT / "config" / f"phase1_ordering_qualification{suffix}_bank{bank_index}.json"
    )
    config = {
        1: production_qualification_protocol,
        2: production_qualification_protocol_v2,
        3: production_qualification_protocol_v3,
    }[protocol_version](bank_index)
    identity_object = {
        "config_sha256": _sha256_file(config_path),
        "fleet_sha256": config["fleet_sha256"],
        "source_inventory": {"test": f"bank-{bank_index}"},
        "git_commit": "test-qualification-commit",
        "execution_contract": {"test": True},
    }
    identity = _sha256_json(identity_object)
    identity_bytes = np.frombuffer(bytes.fromhex(identity), dtype=np.uint8)
    partial = {
        "attempt_identity_sha256": np.tile(
            identity_bytes, (2, 1)
        ),
        "completed": np.ones((2, 160), dtype=np.int8),
        "keep_full": np.ones((2, 3, 160), dtype=np.float64),
        "keep_ablated": np.ones((2, 3, 160), dtype=np.float64),
        "ess_full_atoms": np.ones((2, 160), dtype=np.float64),
        "ess_ablated_atoms": np.ones((2, 160), dtype=np.float64),
        "full_probability_sum_error": np.zeros((2, 3, 160), dtype=np.float64),
        "ablated_probability_sum_error": np.zeros((2, 3, 160), dtype=np.float64),
        "full_probability_minimum": np.full((2, 3, 160), 1e-4, dtype=np.float64),
        "ablated_probability_minimum": np.full((2, 3, 160), 1e-4, dtype=np.float64),
        "js_full": np.zeros((2, 2, 160), dtype=np.float64),
        "js_ablated": np.zeros((2, 2, 160), dtype=np.float64),
        "abs_logp_change_full": np.zeros((2, 2, 160), dtype=np.float64),
        "abs_logp_change_ablated": np.zeros((2, 2, 160), dtype=np.float64),
    }
    if fail_first_candidate and bank_index == 0:
        partial["js_full"][0, 0, 0] = 0.001
    raw_metadata: dict[str, np.ndarray] = {}
    if protocol_version >= 2:
        probabilities = np.full((2, 3, 160, 100), 0.01, dtype=np.float64)
        partial.update(
            {
                "outcome_bins": np.zeros((2, 160), dtype=np.int64),
                "full_probability": probabilities.copy(),
                "ablated_probability": probabilities.copy(),
            }
        )
    if protocol_version == 3:
        grid_probability = np.full((2, 2, 3, 160, 100), 0.01, dtype=np.float64)
        partial.update(
            {
                "quadrature_grid_full_probability": grid_probability.copy(),
                "quadrature_grid_ablated_probability": grid_probability.copy(),
                "quadrature_js_full": np.zeros((2, 3, 160), dtype=np.float64),
                "quadrature_js_ablated": np.zeros((2, 3, 160), dtype=np.float64),
                "quadrature_max_bin_abs_logp_change_full": np.zeros(
                    (2, 3, 160), dtype=np.float64
                ),
                "quadrature_max_bin_abs_logp_change_ablated": np.zeros(
                    (2, 3, 160), dtype=np.float64
                ),
                "quadrature_reference_weighted_abs_logp_change_full": np.zeros(
                    (2, 3, 160), dtype=np.float64
                ),
                "quadrature_reference_weighted_abs_logp_change_ablated": np.zeros(
                    (2, 3, 160), dtype=np.float64
                ),
                "quadrature_max_bin_abs_ordering_value_change": np.zeros(
                    (2, 3, 160), dtype=np.float64
                ),
            }
        )
        raw_metadata = {
            "quadrature_grid_interior_nodes": np.array([32, 64], dtype=np.int64),
            "quadrature_grid_tail_nodes": np.array([128, 256], dtype=np.int64),
            "quadrature_truncation_levels": np.array(
                [8192, 16384, 32768], dtype=np.int64
            ),
        }
    raw_path = run_dir / "calibration_raw.npz"
    write_numeric_npz_atomic(
        raw_path,
        prior_codes=np.array([0, 1], dtype=np.int64),
        candidates=np.array([8192, 16384], dtype=np.int64),
        reference_truncation=np.array([32768], dtype=np.int64),
        **raw_metadata,
        **partial,
    )
    for prior_index, prior in enumerate(("C", "N")):
        write_numeric_npz_atomic(
            run_dir / f"partial_{prior}.npz",
            **{name: value[prior_index] for name, value in partial.items()},
        )
    running_path = run_dir / "RUNNING.json"
    write_json_atomic(
        running_path, {"identity": identity_object, "identity_sha256": identity}
    )
    atom_bank = {
        "schema_version": 1,
        "seed": config["atom_seed"],
        "count": config["atom_count"],
        "shape": [config["atom_count"], 4, 4],
        "dtype": "<f8",
        "sha256": f"{bank_index + 11:064x}",
        "determinism_canary": {
            "seed": config["atom_determinism_canary_seed"],
            "count": config["atom_determinism_canary_count"],
            "shape": [config["atom_determinism_canary_count"], 4, 4],
            "dtype": "<f8",
            "sha256": "a" * 64,
        },
    }
    atom_bank_path = run_dir / "ATOM_BANK.json"
    write_json_atomic(atom_bank_path, atom_bank)
    summary_path = run_dir / "calibration_summary.json"
    summary = {
        "stage": (
            "phase1_ordering_cross_bank_qualification"
            if protocol_version == 1
            else f"phase1_ordering_cross_bank_qualification_v{protocol_version}"
        ),
        "scientific_endpoints_computed": False,
        "oracle_internal_dtype": (
            "float64" if protocol_version == 3 else "float32"
        ),
        "completion_state": "INCOMPLETE_WITHOUT_VERIFIED_COMPLETE_MARKER",
        "attempt_identity_sha256": identity,
        "git": {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
        "config_sha256": _sha256_file(config_path),
        "raw_sha256": _sha256_file(raw_path),
        "atom_seed": config["atom_seed"],
        "atom_bank": atom_bank,
        "calibration_stream_seeds": {
            "C": config["calibration_seed_root"],
            "N": config["calibration_seed_root"] + 10_000,
        },
        "candidate_results": [
            {"truncation": 8192, "compared_with": 32768},
            {"truncation": 16384, "compared_with": 32768},
        ],
        "quadrature_qualification": (
            qualification._quadrature_report(partial, config)
            if protocol_version == 3
            else None
        ),
        "decision": "CALIBRATION_PASS",
    }
    write_json_atomic(summary_path, summary)
    payloads = [
        running_path,
        atom_bank_path,
        raw_path,
        summary_path,
        run_dir / "partial_C.npz",
        run_dir / "partial_N.npz",
    ]
    write_json_atomic(
        run_dir / "COMPLETE.json",
        {
            "identity": identity_object,
            "identity_sha256": identity,
            "decision": summary["decision"],
            "artifacts": {
                path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
                for path in payloads
            },
        },
    )


def _refresh_artifact_hashes(run_dir: Path) -> None:
    complete_path = run_dir / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    for name in complete["artifacts"]:
        path = run_dir / name
        complete["artifacts"][name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    write_json_atomic(complete_path, complete)


def test_qualification_join_selects_first_globally_passing_candidate(
    tmp_path, monkeypatch
):
    run_dirs = [tmp_path / f"bank{index}" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(run_dir, index, fail_first_candidate=False)
    monkeypatch.setattr(
        qualification,
        "_git_provenance",
        lambda require_clean: {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
    )
    summary = qualification.join_qualification(run_dirs, tmp_path / "joined")
    assert summary["decision"] == "QUALIFICATION_PASS"
    assert summary["selected_truncation"] == 8192
    assert (tmp_path / "joined" / "COMPLETE.json").is_file()
    assert not (tmp_path / "joined" / "ATTEMPT.lock").exists()


def test_v2_protocol_is_registered_with_fresh_contexts_and_raw_replay() -> None:
    for bank_index in range(3):
        config = production_qualification_protocol_v2(bank_index)
        registered = json.loads(
            (
                ROOT
                / "config"
                / f"phase1_ordering_qualification_v2_bank{bank_index}.json"
            ).read_text()
        )
        assert registered == config
        assert config["calibration_seed_root"] == 880_923_000 + bank_index
        assert config["archive_predictive_arrays"] is True
        assert config["atom_seed"] == 881_003_101 + bank_index
        v3 = production_qualification_protocol_v3(bank_index)
        registered_v3 = json.loads(
            (
                ROOT
                / "config"
                / f"phase1_ordering_qualification_v3_bank{bank_index}.json"
            ).read_text()
        )
        assert registered_v3 == v3
        assert v3["calibration_seed_root"] == 880_943_000 + bank_index
        assert v3["quadrature_interior_nodes"] == 32
        assert v3["quadrature_reference_interior_nodes"] == 64


def test_blocked_v2_cannot_write_shard_join_or_verification_outputs(tmp_path):
    config_path = ROOT / "config" / "phase1_ordering_qualification_v2_bank0.json"
    shard_out = tmp_path / "shard"
    with pytest.raises(RuntimeError, match="blocked-before-execution"):
        ordering_runner.run_calibration(
            config_path,
            shard_out,
            "cpu",
            production_protocol=False,
            strict_runtime=False,
        )
    assert not shard_out.exists()
    with pytest.raises(RuntimeError, match="blocked-before-execution"):
        ordering_runner.main(
            [
                "qualify",
                "--config",
                str(config_path),
                "--out",
                str(shard_out),
                "--device",
                "cpu",
            ]
        )
    assert not shard_out.exists()
    with pytest.raises(RuntimeError, match="blocked-before-execution"):
        qualification.join_qualification([], tmp_path / "joined", protocol_version=2)
    assert not (tmp_path / "joined").exists()
    with pytest.raises(RuntimeError, match="blocked-before-execution"):
        independent_verifier.verify_qualification(
            tmp_path / "missing", ROOT, commit="test", protocol_version=2
        )


def test_registered_v3_refuses_non_strict_direct_execution(tmp_path):
    config_path = ROOT / "config" / "phase1_ordering_qualification_v3_bank0.json"
    shard_out = tmp_path / "shard"
    with pytest.raises(RuntimeError, match="frozen CLI execution path"):
        ordering_runner.run_calibration(
            config_path,
            shard_out,
            "cpu",
            production_protocol=False,
            frozen_protocol=production_qualification_protocol_v3(0),
            strict_runtime=False,
            stage_name="phase1_ordering_cross_bank_qualification_v3",
        )
    assert not shard_out.exists()


def test_v3_join_recomputes_and_archives_native_predictive_arrays(
    tmp_path, monkeypatch
):
    run_dirs = [tmp_path / f"bank{index}" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(
            run_dir,
            index,
            fail_first_candidate=False,
            protocol_version=3,
        )
    monkeypatch.setattr(
        qualification,
        "_git_provenance",
        lambda require_clean: {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
    )
    monkeypatch.setattr(
        qualification, "_verify_annotated_tag", lambda tag, commit: None
    )
    summary = qualification.join_qualification(
        run_dirs, tmp_path / "joined", protocol_version=3
    )
    assert summary["stage"] == "phase1_ordering_cross_bank_qualification_v3_join"
    assert summary["selected_truncation"] == 8192
    with np.load(tmp_path / "joined" / "qualification_raw.npz") as raw:
        assert raw["outcome_bins"].shape == (3, 2, 160)
        assert raw["full_probability"].shape == (3, 2, 3, 160, 100)
        assert raw["ablated_probability"].shape == (3, 2, 3, 160, 100)
        assert raw["quadrature_grid_full_probability"].shape == (
            3,
            2,
            2,
            3,
            160,
            100,
        )


def test_v3_shard_rejects_derived_metric_that_disagrees_with_predictions(tmp_path):
    run_dir = tmp_path / "bank0"
    _write_shard(
        run_dir,
        0,
        fail_first_candidate=False,
        protocol_version=3,
    )
    raw_path = run_dir / "calibration_raw.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    raw["js_full"][0, 0, 0] = 1e-6
    write_numeric_npz_atomic(raw_path, **raw)
    summary_path = run_dir / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["raw_sha256"] = _sha256_file(raw_path)
    write_json_atomic(summary_path, summary)
    _refresh_artifact_hashes(run_dir)
    with pytest.raises(RuntimeError, match="archived full JS mismatch"):
        qualification._verify_shard(run_dir, 0, protocol_version=3)


def test_independent_v3_verifier_recomputes_predictions_without_runner_imports(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "qualification"
    run_dirs = [artifact_root / f"bank{index}" / "run" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(
            run_dir,
            index,
            fail_first_candidate=False,
            protocol_version=3,
        )
    def fake_git(require_clean):
        return {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        }
    monkeypatch.setattr(qualification, "_git_provenance", fake_git)
    monkeypatch.setattr(
        qualification, "_verify_annotated_tag", lambda tag, commit: None
    )
    qualification.join_qualification(
        run_dirs, artifact_root / "joined", protocol_version=3
    )
    monkeypatch.setattr(
        independent_verifier,
        "_verify_source_inventory",
        lambda inventory, repo, commit, expected_relative_paths: None,
    )
    monkeypatch.setattr(
        independent_verifier,
        "_verify_annotated_tag",
        lambda repo, tag, commit: None,
    )
    monkeypatch.setattr(
        independent_verifier,
        "_git_blob",
        lambda repo, commit, relative: (ROOT / relative).read_bytes(),
    )
    monkeypatch.setattr(
        independent_verifier,
        "V3_VERIFIER_FIX1_JOINED_SHA256",
        {
            "COMPLETE.json": _sha256_file(artifact_root / "joined" / "COMPLETE.json"),
            "qualification_raw.npz": _sha256_file(
                artifact_root / "joined" / "qualification_raw.npz"
            ),
            "qualification_summary.json": _sha256_file(
                artifact_root / "joined" / "qualification_summary.json"
            ),
        },
    )
    result = independent_verifier.verify_qualification(
        artifact_root,
        ROOT,
        commit="test-qualification-commit",
        protocol_version=3,
        verifier_commit="test-verifier-fix-commit",
        verifier_tag=independent_verifier.V3_VERIFIER_FIX_TAG,
    )
    assert result["verification"] == "INDEPENDENT_RAW_RECOMPUTATION_PASS"
    assert result["selected_truncation"] == 8192
    assert result["qualification_protocol_version"] == 3
    assert result["source_commit"] == "test-qualification-commit"
    assert result["source_tag"] == independent_verifier.V3_SOURCE_TAG
    assert result["verifier_commit"] == "test-verifier-fix-commit"
    assert result["verifier_source_tag"] == independent_verifier.V3_VERIFIER_FIX_TAG


def test_one_tail_exceedance_disqualifies_only_the_affected_candidate(
    tmp_path, monkeypatch
):
    run_dirs = [tmp_path / f"bank{index}" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(run_dir, index, fail_first_candidate=True)
    monkeypatch.setattr(
        qualification,
        "_git_provenance",
        lambda require_clean: {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
    )
    summary = qualification.join_qualification(run_dirs, tmp_path / "joined")
    assert summary["candidate_reports"][0]["pass_all_banks"] is False
    assert summary["candidate_reports"][1]["pass_all_banks"] is True
    assert summary["selected_truncation"] == 16384


def test_shard_without_required_identity_objects_is_rejected(tmp_path):
    run_dir = tmp_path / "bank0"
    _write_shard(run_dir, 0, fail_first_candidate=False)
    complete_path = run_dir / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    del complete["identity"]
    write_json_atomic(complete_path, complete)
    running_path = run_dir / "RUNNING.json"
    running = json.loads(running_path.read_text())
    del running["identity"]
    write_json_atomic(running_path, running)
    _refresh_artifact_hashes(run_dir)
    with pytest.raises(RuntimeError, match="identity object missing"):
        qualification._verify_shard(run_dir, 0)


def test_shard_raw_payload_identity_mismatch_is_rejected(tmp_path):
    run_dir = tmp_path / "bank0"
    _write_shard(run_dir, 0, fail_first_candidate=False)
    raw_path = run_dir / "calibration_raw.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    raw["attempt_identity_sha256"][0, 0] ^= np.uint8(1)
    write_numeric_npz_atomic(raw_path, **raw)
    _refresh_artifact_hashes(run_dir)
    summary_path = run_dir / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["raw_sha256"] = _sha256_file(raw_path)
    write_json_atomic(summary_path, summary)
    _refresh_artifact_hashes(run_dir)
    with pytest.raises(RuntimeError, match="raw identity mismatch"):
        qualification._verify_shard(run_dir, 0)


def test_cross_node_atom_canary_disagreement_is_rejected(tmp_path, monkeypatch):
    run_dirs = [tmp_path / f"bank{index}" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(run_dir, index, fail_first_candidate=False)
    run_dir = run_dirs[1]
    atom_path = run_dir / "ATOM_BANK.json"
    atom_bank = json.loads(atom_path.read_text())
    atom_bank["determinism_canary"]["sha256"] = "b" * 64
    write_json_atomic(atom_path, atom_bank)
    summary_path = run_dir / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["atom_bank"] = atom_bank
    write_json_atomic(summary_path, summary)
    _refresh_artifact_hashes(run_dir)
    monkeypatch.setattr(
        qualification,
        "_git_provenance",
        lambda require_clean: {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
    )
    with pytest.raises(RuntimeError, match="atom determinism canary"):
        qualification.join_qualification(run_dirs, tmp_path / "joined")


def test_duplicate_full_atom_banks_are_rejected(tmp_path, monkeypatch):
    run_dirs = [tmp_path / f"bank{index}" for index in range(3)]
    for index, run_dir in enumerate(run_dirs):
        _write_shard(run_dir, index, fail_first_candidate=False)
    first_bank = json.loads((run_dirs[0] / "ATOM_BANK.json").read_text())
    run_dir = run_dirs[1]
    atom_path = run_dir / "ATOM_BANK.json"
    atom_bank = json.loads(atom_path.read_text())
    atom_bank["sha256"] = first_bank["sha256"]
    write_json_atomic(atom_path, atom_bank)
    summary_path = run_dir / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["atom_bank"] = atom_bank
    write_json_atomic(summary_path, summary)
    _refresh_artifact_hashes(run_dir)
    monkeypatch.setattr(
        qualification,
        "_git_provenance",
        lambda require_clean: {
            "commit": "test-qualification-commit",
            "dirty": False,
            "status": [],
        },
    )
    with pytest.raises(RuntimeError, match="three distinct full atom banks"):
        qualification.join_qualification(run_dirs, tmp_path / "joined")
