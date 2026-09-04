import json
import platform
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
panel_dir = Path(sys.argv[2]).resolve()
site_packages = Path(
    "/engrfs/project/class/zhao.b/conda_envs/tidpo/lib/python3.10/site-packages"
)
sys.path[:0] = [str(root / "src"), str(site_packages)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pfn_dag_verify.phase1_confirm_common import (  # noqa: E402
    attempt_identity,
    validate_checkpoint_registry,
)
from pfn_dag_verify.phase1_pfn import (  # noqa: E402
    _infer,
    _load_input_shard,
    _verify_inputs_marker,
    load_checkpoint,
)


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    log_error = np.abs(left - right)
    probability_error = np.abs(np.exp(left) - np.exp(right))
    flat_index = int(np.argmax(log_error))
    row, output_bin = np.unravel_index(flat_index, log_error.shape)
    return {
        "max_abs_logp_error": float(log_error[row, output_bin]),
        "mean_abs_logp_error": float(np.mean(log_error)),
        "p99_abs_logp_error": float(np.quantile(log_error, 0.99)),
        "max_abs_probability_error": float(np.max(probability_error)),
        "max_total_variation": float(0.5 * np.max(np.sum(probability_error, axis=1))),
        "max_row": int(row),
        "max_bin": int(output_bin),
        "left_logp": float(left[row, output_bin]),
        "right_logp": float(right[row, output_bin]),
    }


config_path = root / "config/phase1_ordering_confirmation.json"
device = torch.device("cuda")
identity, identity_sha256, config, git = attempt_identity(
    config_path,
    [
        root / "src/pfn_dag_verify/phase1_pfn.py",
        root / "src/pfn_dag_verify/phase1_panel.py",
    ],
    device=device,
)
_verify_inputs_marker(panel_dir, identity, identity_sha256)
registry = validate_checkpoint_registry(config, verify_remote_files=False)
take = int(config["pfn_replay_rows_per_stratum"])
permutation = np.roll(
    np.arange(30), int(config["pfn_context_permutation_roll"])
)
records = []
for prior in ("C", "N"):
    shards = []
    for draw in range(3):
        for bank in range(3):
            shards.append(
                _load_input_shard(
                    panel_dir / "inputs" / f"{prior}_d{draw}_b{bank}.npz",
                    identity_sha256,
                    prior,
                    draw,
                    bank,
                    config,
                )
            )
    contexts = np.concatenate([shard["contexts"][:take] for shard in shards])
    queries = np.concatenate([shard["queries"][:take] for shard in shards])
    reverse = np.arange(len(contexts) - 1, -1, -1)
    replacement_contexts = contexts[:64].copy()
    replacement_queries = queries[:64].copy()
    replacement_contexts[take:] = np.tile(contexts[64:72], (7, 1, 1))
    replacement_queries[take:] = np.tile(queries[64:72], (7, 1))
    relocated_contexts = np.concatenate(
        [replacement_contexts[take:], contexts[:take]], axis=0
    )
    relocated_queries = np.concatenate(
        [replacement_queries[take:], queries[:take]], axis=0
    )
    for seed in range(3):
        for step in (20_000, 60_000, 120_000):
            model, checkpoint = load_checkpoint(
                config, prior, seed, step, device, registry=registry
            )
            baseline = _infer(model, contexts, queries, 64, device)
            singleton = _infer(model, contexts, queries, 1, device)
            batch_8 = _infer(model, contexts, queries, 8, device)
            reversed_output = _infer(
                model, contexts[reverse], queries[reverse], 64, device
            )[reverse]
            context_roll = _infer(
                model, contexts[:, permutation], queries, 64, device
            )
            repeat = _infer(model, contexts, queries, 64, device)
            block_permutation = np.concatenate(
                [np.arange(63, -1, -1), np.arange(71, 63, -1)]
            )
            inverse_block_permutation = np.argsort(block_permutation)
            same_shape_permutation = _infer(
                model,
                contexts[block_permutation],
                queries[block_permutation],
                64,
                device,
            )[inverse_block_permutation]
            companion_replacement = _infer(
                model, replacement_contexts, replacement_queries, 64, device
            )
            focal_relocation = _infer(
                model, relocated_contexts, relocated_queries, 64, device
            )
            remainder_comparisons = {}
            for remainder in (35, 36, 43):
                extended_contexts = np.concatenate(
                    [contexts[:64], contexts[:remainder]], axis=0
                )
                extended_queries = np.concatenate(
                    [queries[:64], queries[:remainder]], axis=0
                )
                remainder_output = _infer(
                    model, extended_contexts, extended_queries, 64, device
                )[64:]
                remainder_comparisons[f"remainder_{remainder}"] = comparison(
                    baseline[:remainder], remainder_output
                )
            records.append(
                {
                    "prior": prior,
                    "seed": seed,
                    "step": step,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "rows": len(contexts),
                    "singleton": comparison(baseline, singleton),
                    "batch_8": comparison(baseline, batch_8),
                    "reverse_batch": comparison(baseline, reversed_output),
                    "context_roll": comparison(baseline, context_roll),
                    "repeat": comparison(baseline, repeat),
                    "same_shape_block_permutation": comparison(
                        baseline, same_shape_permutation
                    ),
                    "fixed_shape_companion_replacement": comparison(
                        baseline[:take], companion_replacement[:take]
                    ),
                    "fixed_shape_focal_relocation": comparison(
                        baseline[:take], focal_relocation[-take:]
                    ),
                    **remainder_comparisons,
                }
            )
print(
    json.dumps(
        {
            "schema_version": 1,
            "source_commit": git["commit"],
            "source_tag": config["required_attempt_tag"],
            "attempt_identity_sha256": identity_sha256,
            "registered_batch_atol": float(config["pfn_batch_logp_atol"]),
            "registered_context_roll_atol": float(
                config["pfn_context_permutation_logp_atol"]
            ),
            "runtime_contract_verified": True,
            "runtime_binary_fingerprint": identity["runtime_binary_fingerprint"],
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            },
            "records": records,
        },
        indent=2,
        sort_keys=True,
    )
)
