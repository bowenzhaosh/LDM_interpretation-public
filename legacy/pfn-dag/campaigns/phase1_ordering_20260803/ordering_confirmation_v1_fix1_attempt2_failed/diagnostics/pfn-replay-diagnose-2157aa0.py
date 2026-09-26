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
    expected_attempt_identity,
    validate_checkpoint_registry,
)
from pfn_dag_verify.phase1_pfn import (  # noqa: E402
    _infer,
    _load_input_shard,
    _verify_inputs_marker,
    load_checkpoint,
)


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    error = np.abs(left - right)
    flat_index = int(np.argmax(error))
    row, output_bin = np.unravel_index(flat_index, error.shape)
    return {
        "max_abs_logp_error": float(error[row, output_bin]),
        "mean_abs_logp_error": float(np.mean(error)),
        "p99_abs_logp_error": float(np.quantile(error, 0.99)),
        "max_row": int(row),
        "max_bin": int(output_bin),
        "left_logp": float(left[row, output_bin]),
        "right_logp": float(right[row, output_bin]),
    }


config_path = root / "config/phase1_ordering_confirmation.json"
identity, identity_sha256, config, git = expected_attempt_identity(config_path)
_verify_inputs_marker(panel_dir, identity, identity_sha256)
registry = validate_checkpoint_registry(config, verify_remote_files=True)
device = torch.device("cuda")
model, checkpoint = load_checkpoint(config, "C", 0, 20_000, device, registry=registry)
shards = []
for draw in range(3):
    for bank in range(3):
        shards.append(
            _load_input_shard(
                panel_dir / "inputs" / f"C_d{draw}_b{bank}.npz",
                identity_sha256,
                "C",
                draw,
                bank,
                config,
            )
        )
take = int(config["pfn_replay_rows_per_stratum"])
contexts = np.concatenate([shard["contexts"][:take] for shard in shards])
queries = np.concatenate([shard["queries"][:take] for shard in shards])
outputs = {}
for batch_size in (1, 8, 16, 32, 64):
    outputs[f"batch_{batch_size}"] = _infer(
        model, contexts, queries, batch_size, device
    )
outputs["batch_64_repeat"] = _infer(model, contexts, queries, 64, device)
reverse = np.arange(len(contexts) - 1, -1, -1)
outputs["batch_64_reverse"] = _infer(
    model, contexts[reverse], queries[reverse], 64, device
)[reverse]
permutation = np.roll(
    np.arange(30), int(config["pfn_context_permutation_roll"])
)
outputs["batch_64_context_roll"] = _infer(
    model, contexts[:, permutation], queries, 64, device
)
baseline = outputs["batch_64"]
comparisons = {
    name: comparison(baseline, output)
    for name, output in outputs.items()
    if name != "batch_64"
}
print(
    json.dumps(
        {
            "schema_version": 1,
            "source_commit": git["commit"],
            "source_tag": config["required_attempt_tag"],
            "attempt_identity_sha256": identity_sha256,
            "checkpoint": {
                "prior": "C",
                "seed": 0,
                "step": 20_000,
                "sha256": checkpoint["sha256"],
            },
            "rows": len(contexts),
            "registered_batch_atol": float(config["pfn_batch_logp_atol"]),
            "registered_context_roll_atol": float(
                config["pfn_context_permutation_logp_atol"]
            ),
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
            "comparisons": comparisons,
        },
        indent=2,
        sort_keys=True,
    )
)
