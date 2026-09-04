import hashlib

import numpy as np
import pytest

from pfn_dag_verify.registry import verify_file_record
from pfn_dag_verify.storage import (
    load_numeric_npz,
    write_json_atomic,
    write_numeric_npz_atomic,
)


def test_json_writer_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "bad.json", {"value": float("nan")})


def test_numeric_shard_round_trip_and_no_pickle(tmp_path):
    path = tmp_path / "shard.npz"
    write_numeric_npz_atomic(path, x=np.arange(12).reshape(3, 4), y=np.eye(3))
    out = load_numeric_npz(path)
    np.testing.assert_array_equal(out["x"], np.arange(12).reshape(3, 4))
    with pytest.raises(TypeError):
        write_numeric_npz_atomic(tmp_path / "bad.npz", x=np.array([object()], dtype=object))


def test_file_registry_hash_is_strict(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint-v1")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"path": str(path), "size": path.stat().st_size, "sha256": sha}
    verify_file_record(record)
    path.write_bytes(b"checkpoint-v2")
    with pytest.raises(ValueError):
        verify_file_record(record)
