import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from .constants import LEGACY_COMPARISON_SEED
from .generative import generate_group
from .model import configure_determinism
from .oracle import GridOracle
from .provenance import verify_runtime
from .query_bank import FIXED_SENSITIVITY_BANK
from .registry import package_source_hashes, sha256_file, validation_input_hashes

LEGACY_RELATIVE_PATH = Path("artifacts/legacy/stage1_functional_law.py")
LEGACY_MODULES = {
    "EF": "e21_fleet.py",
    "A": "d5c_analyze.py",
    "E": "experiment_v3bump.py",
    "G": "d5c_gate0.py",
}
SNAPSHOT_IMPORT_ORDER = (
    "experiment_v3bump.py",
    "d5c_gate0.py",
    "e21_fleet.py",
    "d5c_analyze.py",
)


def _locked_legacy_paths() -> tuple[Path, Path, dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    legacy_path = (root / LEGACY_RELATIVE_PATH).resolve()
    registry = json.loads((root / "config" / "checkpoint_registry.json").read_text())
    sources = {
        str(record["filename"]): record for record in registry.get("original_sources", [])
    }
    if set(sources) != set(LEGACY_MODULES.values()):
        raise ValueError("legacy source registry does not contain the exact dependency set")
    expected = {}
    for filename, record in sources.items():
        source_path = (root / str(record["path"])).resolve()
        if not source_path.is_file() or sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"legacy dependency snapshot mismatch: {filename}")
        expected[filename] = str(source_path)
    return legacy_path, (root / "artifacts" / "source_snapshots").resolve(), expected


def _assert_legacy_module_origins(module, expected: dict[str, str]) -> None:
    for attribute, filename in LEGACY_MODULES.items():
        imported = getattr(module, attribute, None)
        imported_path = Path(str(getattr(imported, "__file__", ""))).resolve()
        if imported_path != Path(expected[filename]):
            raise ValueError(
                f"legacy dependency import mismatch for {filename}: {imported_path}"
            )


def _import_snapshot(filename: str, expected: dict[str, str]):
    name = Path(filename).stem
    path = Path(expected[filename])
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_legacy(path: Path, source_root: Path, expected: dict[str, str]):
    spec = importlib.util.spec_from_file_location("pfn_dag_legacy_stage1_for_compare", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    original_stage1_src = os.environ.get("STAGE1_SRC")
    original_nthread = os.environ.get("NTHREAD")
    original_sys_path = list(sys.path)
    module_names = {Path(value).stem for value in LEGACY_MODULES.values()}
    original_modules = {name: sys.modules.get(name) for name in module_names}
    sys.argv = [str(path)]
    os.environ["STAGE1_SRC"] = str(source_root)
    os.environ["NTHREAD"] = "1"
    sys.path.insert(0, str(source_root))
    for name in module_names:
        sys.modules.pop(name, None)
    try:
        for filename in SNAPSHOT_IMPORT_ORDER:
            _import_snapshot(filename, expected)
        spec.loader.exec_module(module)
        _assert_legacy_module_origins(module, expected)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path
        for name in module_names:
            sys.modules.pop(name, None)
            if original_modules[name] is not None:
                sys.modules[name] = original_modules[name]
        if original_stage1_src is None:
            os.environ.pop("STAGE1_SRC", None)
        else:
            os.environ["STAGE1_SRC"] = original_stage1_src
        if original_nthread is None:
            os.environ.pop("NTHREAD", None)
        else:
            os.environ["NTHREAD"] = original_nthread
    return module


def compare(
    legacy_file: Path,
    queries: np.ndarray,
    n_contexts: int = 8,
    bank_role: str = "primary",
):
    configure_determinism(0)
    verify_runtime()
    if n_contexts < 1:
        raise ValueError("legacy comparison requires at least one context")
    expected_legacy, source_root, expected_modules = _locked_legacy_paths()
    if legacy_file.resolve() != expected_legacy:
        raise ValueError(f"legacy comparison requires locked source: {expected_legacy}")
    legacy = _load_legacy(expected_legacy, source_root, expected_modules)
    configure_determinism(0)
    verify_runtime()
    current = GridOracle(queries=queries, quadrature=15)
    old = legacy.GridOracle("AL40", quad=15, queries=queries)
    rng = np.random.default_rng(LEGACY_COMPARISON_SEED)
    ell_errors = []
    f0_errors = []
    f1_errors = []
    for _ in range(n_contexts):
        group = generate_group(rng, n_continuations=2)
        context = np.concatenate([group.core, group.reference], axis=0)
        new_bundle = current.evaluate(context)
        old_bundle = old.eval_context(context)
        ell_errors.append(abs(new_bundle.ell - float(old_bundle["ell"])))
        f0_errors.append(float(np.max(np.abs(new_bundle.f0 - old_bundle["q2"]))))
        f1_errors.append(float(np.max(np.abs(new_bundle.f1 - old_bundle["q1"]))))
    errors = np.asarray([ell_errors, f0_errors, f1_errors], dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise FloatingPointError("legacy comparison produced empty or non-finite errors")
    max_ell, max_f0, max_f1 = [float(np.max(row)) for row in errors]
    result = {
        "schema_version": 1,
        "producer_sha256": sha256_file(Path(__file__)),
        "implementation_sha256s": package_source_hashes(),
        "input_sha256s": validation_input_hashes(),
        "legacy_file": LEGACY_RELATIVE_PATH.as_posix(),
        "legacy_sha256": sha256_file(expected_legacy),
        "n_contexts": n_contexts,
        "queries": queries.tolist(),
        "bank_role": bank_role,
        "max_ell_error": max_ell,
        "max_f0_error": max_f0,
        "max_f1_error": max_f1,
        "pass": bool(max(max_ell, max_f0, max_f1) <= 1e-10),
    }
    verify_runtime()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-file", type=Path, required=True)
    parser.add_argument("--query-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", choices=("primary", "sensitivity"), default="primary")
    args = parser.parse_args(argv)
    if args.bank == "primary":
        bank = np.asarray(json.loads(args.query_bank.read_text())["selected_queries"], dtype=float)
    else:
        bank = FIXED_SENSITIVITY_BANK
    result = compare(
        args.legacy_file,
        bank,
        bank_role=args.bank,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
