import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pfn_dag_verify.phase1_confirm_common import (
    CANONICAL_CONFIRMATION_SOURCES,
    _qualification_artifact_hashes,
    _verify_qualification_artifacts,
    attempt_identity,
    sha256_file,
    validate_confirmation_config,
)
from pfn_dag_verify.phase1_confirmation_verify import CANONICAL_SOURCES


ROOT = Path(__file__).resolve().parents[1]


def _qualified_v3_binding() -> tuple[dict, dict]:
    hashes = {
        "complete": "1" * 64,
        "raw": "2" * 64,
        "summary": "3" * 64,
    }
    config = {
        "qualification_protocol_version": 3,
        "qualification_source_tag": "phase1-ordering-qualification-v3",
        "qualification_source_commit": "a" * 40,
        "qualification_verifier_commit": "e" * 40,
        "qualification_verifier_tag": (
            "phase1-ordering-qualification-v3-verifier-fix1"
        ),
        "qualification_verifier_source_sha256": "f" * 64,
        "selected_truncation": 16_384,
        "qualification_reference_truncation": 32_768,
        "atom_banks": [{"sha256": "b" * 64}, {"sha256": "c" * 64}],
        "atom_determinism_canary_seed": 881_103_999,
        "atom_determinism_canary_count": 4096,
        "atom_determinism_canary_sha256": "d" * 64,
        "qualification_complete_sha256": hashes["complete"],
        "qualification_raw_sha256": hashes["raw"],
        "qualification_summary_sha256": hashes["summary"],
    }
    verification = {
        "schema_version": 3,
        "qualification_protocol_version": 3,
        "verification": "INDEPENDENT_RAW_RECOMPUTATION_PASS",
        "decision": "QUALIFICATION_PASS",
        "source_commit": config["qualification_source_commit"],
        "source_tag": config["qualification_source_tag"],
        "verifier_commit": config["qualification_verifier_commit"],
        "verifier_source_tag": config["qualification_verifier_tag"],
        "verifier_source_sha256": config["qualification_verifier_source_sha256"],
        "selected_truncation": config["selected_truncation"],
        "reference_truncation": config["qualification_reference_truncation"],
        "bank_atom_sha256": [row["sha256"] for row in config["atom_banks"]],
        "determinism_canary_sha256": config["atom_determinism_canary_sha256"],
        "joined_complete_sha256": hashes["complete"],
        "joined_raw_sha256": hashes["raw"],
        "joined_summary_sha256": hashes["summary"],
    }
    return config, verification


def test_confirmation_rejects_a_noncanonical_config_path(tmp_path):
    copied_config = tmp_path / "phase1_ordering_confirmation.json"
    copied_config.write_text(
        (ROOT / "config/phase1_ordering_confirmation.json").read_text()
    )
    with pytest.raises(RuntimeError, match="canonical repository config path"):
        validate_confirmation_config(copied_config)


def test_confirmation_accepts_the_frozen_v3_qualification():
    config = validate_confirmation_config(
        ROOT / "config/phase1_ordering_confirmation.json"
    )
    assert config["qualification_protocol_version"] == 3
    assert config["qualification_source_tag"] == "phase1-ordering-qualification-v3"
    assert config["qualification_verifier_tag"] == (
        "phase1-ordering-qualification-v3-verifier-fix1"
    )


def test_qualification_unlock_crosslinks_v3_identity_and_all_joined_hashes():
    config, verification = _qualified_v3_binding()
    assert _qualification_artifact_hashes(config, verification) == {
        "COMPLETE.json": config["qualification_complete_sha256"],
        "qualification_raw.npz": config["qualification_raw_sha256"],
        "qualification_summary.json": config["qualification_summary_sha256"],
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("qualification_protocol_version", 2),
        ("qualification_source_tag", "phase1-ordering-qualification-v1"),
        ("qualification_verifier_tag", "phase1-ordering-qualification-v3"),
    ],
)
def test_qualification_unlock_requires_the_v3_config_binding(field, bad_value):
    config, verification = _qualified_v3_binding()
    config[field] = bad_value
    with pytest.raises(RuntimeError):
        _qualification_artifact_hashes(config, verification)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("qualification_protocol_version", 2),
        ("decision", "QUALIFICATION_FAIL"),
        ("source_commit", "e" * 40),
        ("source_tag", "phase1-ordering-qualification-v2"),
        ("verifier_commit", "0" * 40),
        ("verifier_source_tag", "phase1-ordering-qualification-v3"),
        ("verifier_source_sha256", "0" * 64),
        ("selected_truncation", 8192),
        ("reference_truncation", 16_384),
        ("bank_atom_sha256", ["f" * 64]),
        ("determinism_canary_sha256", "f" * 64),
        ("joined_complete_sha256", "0" * 64),
        ("joined_raw_sha256", "0" * 64),
        ("joined_summary_sha256", "0" * 64),
    ],
)
def test_qualification_unlock_rejects_unbound_verifier_fields(field, bad_value):
    config, verification = _qualified_v3_binding()
    verification[field] = bad_value
    with pytest.raises(RuntimeError):
        _qualification_artifact_hashes(config, verification)


def test_qualification_unlock_checks_optional_canary_coordinates():
    config, verification = _qualified_v3_binding()
    verification["determinism_canary_seed"] = config["atom_determinism_canary_seed"]
    verification["determinism_canary_count"] = 4095
    with pytest.raises(RuntimeError, match="canary identity"):
        _qualification_artifact_hashes(config, verification)


def test_qualification_unlock_hashes_the_actual_joined_artifacts(tmp_path):
    joined = tmp_path / "joined"
    joined.mkdir()
    paths = {
        "COMPLETE.json": joined / "COMPLETE.json",
        "qualification_raw.npz": joined / "qualification_raw.npz",
        "qualification_summary.json": joined / "qualification_summary.json",
    }
    for filename, path in paths.items():
        path.write_bytes(filename.encode())
    expected = {filename: sha256_file(path) for filename, path in paths.items()}
    _verify_qualification_artifacts(tmp_path, expected, label="local")
    paths["qualification_raw.npz"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="local qualification artifact mismatch"):
        _verify_qualification_artifacts(tmp_path, expected, label="local")


def test_confirmation_config_freezes_qualified_substrate():
    config = json.loads(
        (ROOT / "config" / "phase1_ordering_confirmation.json").read_text()
    )
    assert config["status"] == "confirmatory_locked"
    assert config["selected_truncation"] == 16384
    assert config["qualification_reference_truncation"] == 32768
    assert config["qualification_protocol_version"] == 3
    assert config["qualification_source_commit"] == (
        "fc0b8eb48c75f7c0d1dc208b1d344d663b16baf3"
    )
    assert config["qualification_verifier_commit"] == (
        "04429ae5713dff0f101d842b8e6b4890c7f4e668"
    )
    assert (config["quadrature_interior_nodes"], config["quadrature_tail_nodes"]) == (
        32,
        128,
    )
    assert config["oracle_compute_dtype"] == "float64"
    assert config["contexts_per_prior_draw"] == 1067
    assert config["evaluation_seeds"] == {
        "C": [881003000, 881003001, 881003002],
        "N": [881013000, 881013001, 881013002],
    }
    assert [row["seed"] for row in config["atom_banks"]] == [
        881003101,
        881003102,
        881003103,
    ]
    assert len({row["sha256"] for row in config["atom_banks"]}) == 3
    assert config["nested_half_subset"] == {
        "draw_index": 0,
        "stream_index_stop_exclusive": 200,
    }
    assert config["bootstrap"]["replicates"] == 50000
    assert config["bootstrap"]["chunk_size"] == 256
    assert config["pfn_batch_size"] == 64
    assert config["pfn_batch_logp_atol"] == 5e-5
    assert config["pfn_combined_context_batch_logp_atol"] == 8e-5
    assert config["pfn_context_permutation_logp_atol"] == 3e-5
    assert config["pfn_replay_probability_atol"] == 1e-6
    assert config["pfn_replay_total_variation_atol"] == 3e-6


def test_confirmation_seed_namespaces_are_disjoint_except_qualified_atom_reuse():
    config = json.loads(
        (ROOT / "config" / "phase1_ordering_confirmation.json").read_text()
    )
    evaluation = {
        seed for values in config["evaluation_seeds"].values() for seed in values
    }
    atoms = {row["seed"] for row in config["atom_banks"]}
    calibration_and_qualification_contexts = {
        880803000,
        880813000,
        880903000,
        880903001,
        880903002,
        880913000,
        880913001,
        880913002,
        880943000,
        880943001,
        880943002,
        880953000,
        880953001,
        880953002,
    }
    assert evaluation.isdisjoint(atoms)
    assert evaluation.isdisjoint(calibration_and_qualification_contexts)
    assert atoms.isdisjoint(calibration_and_qualification_contexts)


def test_all_producers_derive_one_portable_canonical_attempt_identity():
    config = ROOT / "config/phase1_ordering_confirmation.json"
    frozen_config = json.loads(config.read_text())
    runtime = {
        "runtime_binary_fingerprint": (
            "44394126ccf9b4d75eec6d1c3d691f0ab34fbbd50c579deed3abdcb06296636a"
        )
    }
    git = {"commit": "f" * 40, "dirty": False, "status": []}
    source_sets = [
        [ROOT / "src/pfn_dag_verify/phase1_panel.py"],
        [ROOT / "src/pfn_dag_verify/phase1_pfn.py"],
        [ROOT / "src/pfn_dag_verify/phase1_oracle_confirm.py"],
    ]
    with (
        patch(
            "pfn_dag_verify.phase1_confirm_common.validate_confirmation_config",
            return_value=frozen_config,
        ),
        patch("pfn_dag_verify.phase1_confirm_common.git_provenance", return_value=git),
        patch(
            "pfn_dag_verify.phase1_confirm_common.verify_locked_runtime",
            return_value=runtime,
        ),
        patch(
            "pfn_dag_verify.phase1_confirm_common.validate_checkpoint_registry",
            return_value={},
        ) as registry_validator,
    ):
        identities = [
            attempt_identity(config, paths, device=torch.device("cpu"))[:2]
            for paths in source_sets
        ]
    assert identities[0] == identities[1] == identities[2]
    assert registry_validator.call_count == 3
    assert all(
        call.kwargs == {"verify_remote_files": True}
        for call in registry_validator.call_args_list
    )
    source_inventory = identities[0][0]["source_inventory"]
    assert "PHASE1_ORDERING_CONFIRMATION_AMENDMENT.md" in source_inventory
    assert "PHASE1_ORDERING_CONFIRMATION_FIX1.md" in source_inventory
    assert "PHASE1_ORDERING_CONFIRMATION_FIX2.md" in source_inventory
    assert "cluster/submit_phase1_confirmation.py" in source_inventory
    assert any(name.endswith("p1-replay-sweep-160041.out") for name in source_inventory)
    assert all(not Path(name).is_absolute() for name in source_inventory)


def test_common_and_independent_verifier_use_the_same_canonical_sources():
    assert CANONICAL_SOURCES == CANONICAL_CONFIRMATION_SOURCES


def test_production_identity_fails_before_runtime_when_checkpoint_mount_is_missing():
    config_path = ROOT / "config/phase1_ordering_confirmation.json"
    frozen_config = json.loads(config_path.read_text())
    frozen_identity = {
        "runtime_binary_fingerprint": frozen_config["runtime_binary_fingerprint"]
    }
    with (
        patch(
            "pfn_dag_verify.phase1_confirm_common.expected_attempt_identity",
            return_value=(frozen_identity, "ab" * 32, frozen_config, {}),
        ),
        patch(
            "pfn_dag_verify.phase1_confirm_common.validate_checkpoint_registry",
            side_effect=FileNotFoundError("checkpoint mount missing"),
        ),
        patch(
            "pfn_dag_verify.phase1_confirm_common.verify_locked_runtime"
        ) as runtime_validator,
        pytest.raises(FileNotFoundError, match="checkpoint mount missing"),
    ):
        attempt_identity(config_path, [], device=torch.device("cpu"))
    runtime_validator.assert_not_called()
