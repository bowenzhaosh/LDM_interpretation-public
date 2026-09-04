import numpy as np
import pytest
from scipy.special import expit, logit
import subprocess

from pfn_dag_verify.constants import FIXED_PRIOR_SEEDS
from pfn_dag_verify.mapping_qualification import (
    ATTEMPT_TAG,
    QUALIFICATION_SETTINGS,
    STREAM_NAMESPACE,
    _register_attempt,
    _verify_attempt_tag,
    cross_bank_block,
    inference_guard_from_predictions,
    mapping_block,
    qualification_decision,
    seed_label,
)
import pfn_dag_verify.mapping_qualification as qualification_module
from pfn_dag_verify.provenance import (
    audit_readiness_subject_files,
    audit_review_subject_files,
)


def _endpoints(n_contexts=6, n_queries=2, n_bins=4):
    f0 = np.full((n_contexts, n_queries, n_bins), 0.1, dtype=np.float64)
    f1 = np.full_like(f0, 0.1)
    f0[..., 0] = 0.7
    f1[..., 1] = 0.7
    return f0, f1


def _mixture(f0, f1, weight):
    weight = np.asarray(weight, dtype=np.float64)
    return f0 + weight[:, None, None] * (f1 - f0)


def _exact_fixture(n_groups=6, n_targets=2):
    f0_base, f1_base = _endpoints(n_groups)
    f0_target_flat, f1_target_flat = _endpoints(n_groups * n_targets)
    f0_target = f0_target_flat.reshape(n_groups, n_targets, 2, 4)
    f1_target = f1_target_flat.reshape(n_groups, n_targets, 2, 4)
    w_base = np.linspace(0.2, 0.7, n_groups)
    delta_ell = np.linspace(-0.4, 0.5, n_groups * n_targets).reshape(
        n_groups, n_targets
    )
    w_target = expit(logit(w_base)[:, None] + delta_ell)
    p_base = _mixture(f0_base, f1_base, w_base)
    p_target = _mixture(
        f0_target_flat,
        f1_target_flat,
        w_target.reshape(-1),
    ).reshape(n_groups, n_targets, 2, 4)
    ell_base = logit(w_base)
    ell_target = ell_base[:, None] + delta_ell
    return p_base, p_target, f0_base, f1_base, f0_target, f1_target, ell_base, ell_target


def test_mapping_block_passes_exact_native_mixture_fixture():
    block, arrays = mapping_block(*_exact_fixture()[:6])
    assert block["pass"] is True
    assert block["boundary_rate"] == 0.0
    assert block["coordinate_kl_abs_g"]["p95"] < 1e-7
    assert arrays["g_base"].shape == (6,)
    assert arrays["g_target"].shape == (6, 2)
    assert "reconstruction_residual" not in block
    assert "reconstruction_residual" not in arrays


def test_tempered_on_segment_readout_passes_mapping_without_testing_gain():
    fixture = list(_exact_fixture())
    ell_base, ell_target = fixture[6], fixture[7]
    delta_ell = ell_target - ell_base[:, None]
    tempered_target = expit(ell_base[:, None] + 0.5 * delta_ell)
    n_groups, n_targets = tempered_target.shape
    fixture[1] = _mixture(
        fixture[4].reshape(n_groups * n_targets, 2, 4),
        fixture[5].reshape(n_groups * n_targets, 2, 4),
        tempered_target.reshape(-1),
    ).reshape(n_groups, n_targets, 2, 4)
    block, arrays = mapping_block(*fixture[:6])
    assert block["pass"] is True
    assert cross_bank_block(arrays, arrays)["pass"] is True


def test_mapping_block_fails_off_segment_predictions_even_when_finite():
    fixture = list(_exact_fixture())
    p_target = fixture[1].copy()
    p_target[..., 2] += 0.08
    p_target[..., 3] -= 0.08
    fixture[1] = p_target
    block, _arrays = mapping_block(*fixture[:6])
    assert block["pass"] is False
    assert block["mixture_residual_target"]["median"] > 0.10


def test_mapping_block_fails_closed_on_one_boundary_prediction():
    fixture = list(_exact_fixture())
    fixture[0][0] = fixture[2][0]
    block, _arrays = mapping_block(*fixture[:6])
    assert block["boundary_rate"] > 0.0
    assert block["pass"] is False


def test_mapping_block_counts_sparse_kl_only_boundary():
    fixture = list(_exact_fixture(n_groups=64))
    fixture[2][0] = np.asarray(
        [
            [0.16756010335740776, 0.12371370079251415, 0.5296060654162104, 0.17912013043386762],
            [0.1497410989833778, 0.1970213057914621, 0.10763759236377958, 0.5456000028613804],
        ]
    )
    fixture[3][0] = np.asarray(
        [
            [0.21686681910184408, 0.4148823796495385, 0.2269157319222367, 0.1413350693263806],
            [0.4852575805687991, 0.31686342611508106, 0.09011058596009425, 0.10776840735602548],
        ]
    )
    fixture[0][0] = np.asarray(
        [
            [0.09585309589604925, 0.33913368315314396, 0.10957121948101736, 0.4554420014697896],
            [0.34245032916778273, 0.275021443190343, 0.36952809974717427, 0.013000127894699919],
        ]
    )
    block, arrays = mapping_block(*fixture[:6])
    assert arrays["boundary_base"][0] == 0
    assert arrays["kl_boundary_base"][0] == 1
    assert block["boundary_rate"] > 0.0
    assert block["pass"] is False


def test_qualification_decision_requires_every_bank_and_cross_bank_block():
    banks = [
        {"pass": True, "seed": seed, "bank_index": bank, "step": 12_000}
        for seed in range(16)
        for bank in range(2)
    ]
    cross = [{"pass": True, "seed": seed, "step": 12_000} for seed in range(16)]
    assert qualification_decision(banks, cross) == "QUALIFIED"
    banks[-1] = {**banks[-1], "pass": False}
    assert qualification_decision(banks, cross) == "FAILED_NATIVE_MAPPING"
    banks[-1] = {**banks[-1], "pass": True}
    cross[-1] = {**cross[-1], "pass": False}
    assert qualification_decision(banks, cross) == "FAILED_NATIVE_MAPPING"


def test_cross_bank_gate_detects_latent_coordinate_disagreement():
    first = {"g_base": np.zeros(4), "g_target": np.zeros((4, 2))}
    second = {"g_base": np.zeros(4), "g_target": np.zeros((4, 2))}
    assert cross_bank_block(first, second)["pass"] is True
    second["g_target"][:] = 0.5
    failed = cross_bank_block(first, second)
    assert failed["pass"] is False
    assert failed["absolute_g_disagreement"]["p95"] == 0.5


def test_qualification_decision_rejects_incomplete_or_malformed_blocks():
    banks = [
        {"pass": True, "seed": seed, "bank_index": bank, "step": 12_000}
        for seed in range(16)
        for bank in range(2)
    ]
    cross = [{"pass": True, "seed": seed, "step": 12_000} for seed in range(16)]
    for bank_blocks, cross_blocks in (
        (banks[:-1], cross),
        (banks, cross[:-1]),
        (banks[:-1] + [{}], cross),
        (banks[:-1] + [banks[0]], cross),
    ):
        try:
            qualification_decision(bank_blocks, cross_blocks)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete or malformed qualification was accepted")


def test_qualification_decision_rejects_forbidden_output_fields():
    banks = [
        {"pass": True, "seed": seed, "bank_index": bank, "step": 12_000}
        for seed in range(16)
        for bank in range(2)
    ]
    cross = [{"pass": True, "seed": seed, "step": 12_000} for seed in range(16)]
    banks[0]["composition_slope"] = 1.0
    try:
        qualification_decision(banks, cross)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden composition output survived the information barrier")


def test_mapping_seed_namespace_is_explicit_and_nonempty():
    assert seed_label(STREAM_NAMESPACE, "core-candidate:7") == "mapping-calibration-v1:core-candidate:7"
    for namespace in ("", "science", "mapping-calibration-v1:"):
        try:
            seed_label(namespace, "x")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid namespace accepted: {namespace!r}")


def test_failed_qualification_cli_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        qualification_module,
        "score_qualification",
        lambda: {"status": "COMPLETE", "decision": "FAILED_NATIVE_MAPPING"},
    )
    with pytest.raises(SystemExit) as error:
        qualification_module.main(["score"])
    assert error.value.code == 2
    assert "FAILED_NATIVE_MAPPING" in capsys.readouterr().out


def test_raw_inference_guard_predictions_determine_the_guard():
    reference = np.linspace(0.1, 0.9, 24, dtype=np.float32).reshape(3, 2, 4)
    exact = inference_guard_from_predictions(reference, reference, reference, reference)
    assert exact["pass"] is True
    changed = reference.copy()
    changed[0, 0, 0] += 2e-6
    failed = inference_guard_from_predictions(reference, reference, reference, changed)
    assert failed["max_row_permutation_error"] > 1e-6
    assert failed["pass"] is False


def test_qualification_settings_and_fixed_seed_inventory_are_complete():
    assert QUALIFICATION_SETTINGS["panel"] == {
        "groups": 64,
        "core_rows": 20,
        "reference_rows": 10,
        "targets_per_group": 2,
        "target_rows": 10,
    }
    assert QUALIFICATION_SETTINGS["instrument"]["banks"] == 2
    assert QUALIFICATION_SETTINGS["instrument"]["queries_per_bank"] == 8
    assert QUALIFICATION_SETTINGS["instrument"]["bins"] == 100
    assert FIXED_PRIOR_SEEDS == {
        810000,
        810099,
        810101,
        810777,
        820001,
        850002,
        *range(860003, 860008),
    }
    assert "MAPPING_QUALIFICATION_PREREG.md" in audit_review_subject_files()
    assert "MAPPING_QUALIFICATION_PREREG.md" in audit_readiness_subject_files()


def test_attempt_tag_is_a_durable_one_shot_guard(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "PFN DAG test"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "pfn-dag-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("fixture\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    attestation = {
        "attempt_id": "native-mapping-v1",
    }
    tag_object = _register_attempt(tmp_path, commit, attestation)
    assert _verify_attempt_tag(tmp_path, commit, attestation) == tag_object
    with pytest.raises(RuntimeError, match="one-shot"):
        _register_attempt(tmp_path, commit, attestation)
    assert subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{ATTEMPT_TAG}"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
    ).returncode == 0
