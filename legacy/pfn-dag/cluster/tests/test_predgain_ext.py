"""Unit test for scripts/mech_predgain_ext.py — the EXT scorer wrapper.

No GPU, no cluster, no torch: mech_phase1 / mech_predgain / mech_interventions and
pfn_dag_verify.corrected_world are replaced by stubs in sys.modules before the
wrapper is imported, so this exercises the wrapper's OWN logic only:

  1. MECH_D / MECH_K patch the world constants the locked mech_phase1 hard-codes
  2. MECH_WORLD_SEED rebinds make_world in all three locked modules
  3. the registered-campaign guard on MECH_OUT refuses the whole tree
  4. ext_provenance.json appends (never truncates) one record per invocation
  5. world_index.json records d, K, world seed, n_rows and a sha256 PER npz file
     (the locked npz records none of them — reviewer finding R9)

Run:  .venv/bin/python cluster/tests/test_predgain_ext.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The child process: stub the locked modules, then drive the wrapper.
CHILD = textwrap.dedent(
    '''
    import json, sys, types
    from pathlib import Path
    ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); NPZ = sys.argv[3:]

    M = types.ModuleType("mech_phase1")
    M.D, M.K = 3, 8
    M.SCALE, M.N_ROWS = "base", 20
    M.NETS = Path("/nets")
    M.PANEL_SEED_EFF, M.SPLIT_SEED_EFF, M.N_PER_HALF_EFF = 770000101, 880000101, 1000
    M.make_world = lambda *a, **k: "REGISTERED"
    sys.modules["mech_phase1"] = M

    def _mk(name):
        m = types.ModuleType(name)
        m.make_world = lambda *a, **k: "REGISTERED"
        m.main = lambda: [ (OUT / f).write_bytes(f.encode()) for f in NPZ ]
        sys.modules[name] = m
        return m
    PG = _mk("mech_predgain"); IV = _mk("mech_interventions")

    cw = types.ModuleType("pfn_dag_verify.corrected_world")
    cw.make_world = lambda *a, **k: ("SEEDED", k.get("seed"))
    pkg = types.ModuleType("pfn_dag_verify"); pkg.__path__ = []
    sys.modules.setdefault("pfn_dag_verify", pkg)
    sys.modules["pfn_dag_verify.corrected_world"] = cw

    sys.path.insert(0, str(ROOT / "scripts"))
    import mech_predgain_ext as W
    W.main()
    print(json.dumps({"D": M.D, "K": M.K,
                      "mw_M": M.make_world(), "mw_PG": PG.make_world(), "mw_IV": IV.make_world()}))
    '''
)

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + ((" — " + detail) if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def run(out: Path, env: dict, npz=("predgain_eps1p0_ck500000.npz",)) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.pop("MECH_WORLD_SEED", None)
    e.update(MECH_OUT=str(out), **env)
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run([sys.executable, "-c", CHILD, str(ROOT), str(out), *npz],
                              capture_output=True, text=True, env=e, cwd=cwd)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 1+5: d=4, K=32, world seed 11, two npz (orig + an n40ext kind)
        out = td / "predgain" / "cell" / "s3-8"
        r = run(out, {"MECH_D": "4", "MECH_K": "32", "MECH_WORLD_SEED": "11"},
                npz=("predgain_eps0p75_ck500000.npz", "predgain_eps0p75_ck500000_n40ext.npz"))
        check("wrapper runs with a patched world", r.returncode == 0, r.stderr[-600:])
        if r.returncode == 0:
            seen = json.loads(r.stdout.strip().splitlines()[-1])
            check("MECH_D/MECH_K patch mech_phase1", (seen["D"], seen["K"]) == (4, 32), str(seen))
            check("MECH_WORLD_SEED rebinds make_world in all 3 locked modules",
                  seen["mw_M"] == seen["mw_PG"] == seen["mw_IV"] == ["SEEDED", 11], str(seen))
            idx = json.loads((out / "world_index.json").read_text())
            check("world_index.json indexes EVERY npz", len(idx) == 2, str(sorted(idx)))
            row = idx.get("predgain_eps0p75_ck500000.npz", {})
            check("world index records d, K, world_seed, n_rows",
                  (row.get("d"), row.get("K"), row.get("world_seed"), row.get("n_rows")) == (4, 32, 11, 20),
                  str(row))
            check("world index records a sha256 per file",
                  len(row.get("sha256", "")) == 64
                  and row["sha256"] != idx["predgain_eps0p75_ck500000_n40ext.npz"]["sha256"])
            prov = json.loads((out / "ext_provenance.json").read_text())
            check("ext_provenance.json is a one-record list", len(prov) == 1 and prov[0]["world"]["d"] == 4)

        # 4: a second invocation appends and re-indexes, never truncates
        r2 = run(out, {"MECH_D": "4", "MECH_K": "32", "MECH_WORLD_SEED": "11"},
                 npz=("predgain_eps0p5_ck500000.npz",))
        check("second invocation succeeds", r2.returncode == 0, r2.stderr[-400:])
        if r2.returncode == 0:
            check("ext_provenance appends", len(json.loads((out / "ext_provenance.json").read_text())) == 2)
            check("world index keeps earlier files",
                  len(json.loads((out / "world_index.json").read_text())) == 3)

        # 3: the registered-campaign guard, on paths that never existed on disk
        for sub, expect_refuse in [("campaigns/mech_20260827/predgain_confirm/c", True),
                                   ("campaigns/mech_20260827/phase1", True),
                                   ("campaigns/mech_20260827", True),
                                   ("campaigns/mech_ext_20260902/predgain/c", False)]:
            p = td / "repo" / sub
            rg = run(p, {})
            refused = rg.returncode != 0 and "REFUSING" in (rg.stdout + rg.stderr)
            check(f"MECH_OUT guard {'refuses' if expect_refuse else 'permits'} {sub}",
                  refused == expect_refuse, (rg.stdout + rg.stderr)[-300:])

        # 3b: a '..' traversal into the registered campaign
        p = td / "repo" / "campaigns/mech_ext_20260902/../mech_20260827/x"
        rg = run(p, {})
        check("MECH_OUT guard refuses a '..' traversal into the registered campaign",
              rg.returncode != 0 and "REFUSING" in (rg.stdout + rg.stderr), (rg.stdout + rg.stderr)[-300:])

    print(("FAILED: " + ", ".join(FAILED)) if FAILED else "all predgain_ext checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
