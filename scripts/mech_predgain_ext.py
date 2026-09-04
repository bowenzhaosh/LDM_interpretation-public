"""EXT-campaign scorer (REPORTED, never gated).

Delegates to the locked scripts/mech_predgain.py unchanged, after patching the
world constants the locked scripts/mech_phase1.py hard-codes (D=3, K=8) from the
environment (MECH_D, MECH_K). mech_predgain and mech_interventions both read the
same module object at call time, so the patch reaches every make_world call.

Writes one sidecar per invocation, ext_provenance.json in MECH_OUT, recording the
world, the net prefix and the panel so a scored EXT cell can never be mistaken for
a registered one. Everything digested by AMENDMENT_G.md G.0 is left untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech_phase1 as M  # noqa: E402  (locked; patched below, never edited)

M.D = int(os.environ.get("MECH_D", M.D))
M.K = int(os.environ.get("MECH_K", M.K))

import mech_predgain  # noqa: E402  (locked)
import mech_interventions  # noqa: E402  (locked)

WORLD_SEED = os.environ.get("MECH_WORLD_SEED")
if WORLD_SEED:
    # world-seed replication: every make_world call in the three locked modules goes
    # through the same partial (they bind the function name at import, so the module
    # attribute is what their call sites resolve).
    import functools
    from pfn_dag_verify.corrected_world import make_world as _make_world
    _mw = functools.partial(_make_world, seed=int(WORLD_SEED))
    M.make_world = _mw
    mech_predgain.make_world = _mw
    mech_interventions.make_world = _mw


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    out = Path(os.environ["MECH_OUT"])
    # Fail-closed BEFORE the directory is created, and over the whole registered
    # campaign (not just predgain_confirm/): resolve() also normalises a relative
    # path and a '..' traversal, both of which the earlier substring test let past.
    if "/campaigns/mech_20260827/" in str(out.resolve()) + "/":
        raise SystemExit(f"REFUSING: MECH_OUT is inside the registered campaign ({out})")
    out.mkdir(parents=True, exist_ok=True)
    prov = {
        "campaign": "mech_ext_20260902", "reported_only": True,
        "world": {"d": M.D, "K": M.K, "world_seed": int(WORLD_SEED) if WORLD_SEED else None},
        "scale_prefix": M.SCALE, "n_rows": M.N_ROWS, "nets": str(M.NETS),
        "panel": {"panel_seed": M.PANEL_SEED_EFF, "split_seed": M.SPLIT_SEED_EFF,
                  "n_per_half": M.N_PER_HALF_EFF},
        "confirm_env": os.environ.get("MECH_CONFIRM", "0"),
        "argv": sys.argv[1:], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {"mech_predgain.py": _sha(HERE / "mech_predgain.py"),
                        "mech_phase1.py": _sha(HERE / "mech_phase1.py"),
                        "mech_interventions.py": _sha(HERE / "mech_interventions.py"),
                        "mech_predgain_ext.py": _sha(Path(__file__).resolve())},
    }
    side = out / "ext_provenance.json"
    hist = json.loads(side.read_text()) if side.is_file() else []
    hist.append(prov)
    side.write_text(json.dumps(hist, indent=1))
    mech_predgain.main()
    # per-file world index: the locked npz records neither K nor d nor the world seed
    idx_p = out / "world_index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.is_file() else {}
    for f in sorted(out.glob("predgain_eps*_ck*.npz")):
        idx[f.name] = {"d": M.D, "K": M.K, "world_seed": int(WORLD_SEED) if WORLD_SEED else None,
                       "scale_prefix": M.SCALE, "n_rows": M.N_ROWS,
                       "panel": {"panel_seed": M.PANEL_SEED_EFF, "split_seed": M.SPLIT_SEED_EFF,
                                 "n_per_half": M.N_PER_HALF_EFF},
                       "sha256": _sha(f)}
    idx_p.write_text(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
