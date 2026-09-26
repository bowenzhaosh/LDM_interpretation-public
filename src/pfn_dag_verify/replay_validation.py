"""Compare trusted decoded checkpoints using dtype, shape, and value bytes.

This is a new portable implementation of the criterion documented in the
independent replay audit, not an extraction of the production training code.
The companion fixture records previous checks; it cannot replace checkpoints.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


class StateMismatch(ValueError):
    pass


def exact_state(left, right, path="state"):
    """Require identical decoded values, including array dtype and shape."""
    def require(condition, detail):
        if not condition:
            raise StateMismatch(f"{path}: {detail}")

    if torch.is_tensor(left):
        require(torch.is_tensor(right), "tensor type differs")
        require(left.dtype == right.dtype, "tensor dtype differs")
        require(left.shape == right.shape, "tensor shape differs")
        require(left.layout == right.layout == torch.strided, "unsupported tensor layout")
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(a, b), "tensor value bytes differ")
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray), "array type differs")
        require(left.dtype == right.dtype, "array dtype differs")
        require(left.shape == right.shape, "array shape differs")
        require(not left.dtype.hasobject, "object arrays are unsupported")
        require(left.tobytes(order="C") == right.tobytes(order="C"), "array value bytes differ")
    elif isinstance(left, dict):
        require(isinstance(right, dict), "mapping type differs")
        require(left.keys() == right.keys(), "mapping keys differ")
        for key in left:
            exact_state(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right), "sequence type differs")
        require(len(left) == len(right), "sequence length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            exact_state(a, b, f"{path}/{index}")
    elif isinstance(left, np.generic):
        require(isinstance(right, np.generic), "NumPy scalar type differs")
        require(left.dtype == right.dtype, "NumPy scalar dtype differs")
        require(not left.dtype.hasobject, "object scalars are unsupported")
        require(left.tobytes() == right.tobytes(), "NumPy scalar value bytes differ")
    elif isinstance(left, float):
        require(type(left) is type(right), "scalar type differs")
        require(struct.pack("!d", left) == struct.pack("!d", right), "float value bytes differ")
    else:
        require(type(left) is type(right), "scalar type differs")
        require(isinstance(left, (str, int, bool, bytes, type(None))), "unsupported state value")
        require(left == right, "scalar value differs")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_checkpoints(original, replay):
    """Load only trusted files: full checkpoints use Python pickle objects."""
    left = torch.load(original, map_location="cpu", weights_only=False)
    right = torch.load(replay, map_location="cpu", weights_only=False)
    exact_state(left, right)
    original_hash, replay_hash = sha256(original), sha256(replay)
    return {"exact_full_state": True, "original_sha256": original_hash,
            "replay_sha256": replay_hash, "serialized_bytes_equal": original_hash == replay_hash,
            "compared_state_fields": sorted(left),
            "scope": "Decoded checkpoint equality only; no training replay or intermediate-state check."}


def self_test():
    baseline = {"model": torch.tensor([1.0, -0.0]),
                "perm": np.arange(3, dtype=np.int64), "rng": [torch.tensor([2], dtype=torch.uint8)],
                "step": 5, "identity": {"arm": "qz"}}
    exact_state(baseline, copy.deepcopy(baseline))
    changes = [
        ("model", torch.tensor([2.0, -0.0])),
        ("model", baseline["model"].double()),
        ("model", baseline["model"].reshape(1, 2)),
        ("model", torch.tensor([1.0, 0.0])),
        ("perm", np.arange(3, dtype=np.int32)),
        ("perm", np.arange(3, dtype=np.int64).reshape(1, 3)),
        ("perm", np.array([0, 1, 3], dtype=np.int64)),
        ("step", 6), ("rng", tuple(baseline["rng"])), ("identity", {"other": "qz"}),
    ]
    for key, value in changes:
        altered = copy.deepcopy(baseline)
        altered[key] = value
        try:
            exact_state(baseline, altered)
        except StateMismatch:
            continue
        raise AssertionError(f"Failed to detect changed {key}")
    return {"self_test": "PASS", "equal_state_checks": 1, "mismatch_checks": len(changes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, nargs="?")
    parser.add_argument("replay", type=Path, nargs="?")
    parser.add_argument("--trusted-checkpoints", action="store_true",
                        help="Confirm both input files are trusted; loading uses Python pickle.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.original or args.replay:
            parser.error("--self-test does not take checkpoints")
        result = self_test()
    else:
        if not args.original or not args.replay or not args.trusted_checkpoints:
            parser.error("Provide original and replay paths and --trusted-checkpoints")
        try:
            result = compare_checkpoints(args.original, args.replay)
        except StateMismatch as error:
            parser.exit(1, f"Checkpoint comparison failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
