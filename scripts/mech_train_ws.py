"""Mechanism-phase training entry point WITH AN EXPLICIT WORLD SEED (EXT campaign, REPORTED).

Copy of the locked scripts/mech_train.py (never edited) plus --world-seed, passed to
make_world(seed=...), so the atom library can be replicated. Default None = the
registered library (make_world's own seed rule). Everything else is identical.

Original docstring:

The registered trainer (`pfn_dag_verify.w2_run`) hard-codes SCALE="base" and
lives under the write-protected verifier tree, so contrasts that vary the
architecture or the context length go through this file instead. It makes the
SAME `train_pfn` call w2_run makes (same seeding, schedule, batch, n_query,
checkpoint capture) and differs only in what it exposes: --scale, --n-rows.
No in-run evaluation: scoring is done separately by scripts/mech_predgain.py
with MECH_SCALE / MECH_N_ROWS set to match.

Outputs (netdir = OUTROOT/nets/eps{tag}/):
  {scale}_s{seed}_ck{step}.pt          per requested checkpoint (train_pfn)
  {scale}_s{seed}.pt / .provenance.json final state + provenance (train_pfn)
  OUTROOT/train_{scale}_n{n_rows}_eps{tag}_s{seed}.json   this script's record
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pfn_dag_verify.corrected_models import DEV, ModelConfig, train_pfn  # noqa: E402
from pfn_dag_verify.corrected_trackb import SCALES  # noqa: E402
from pfn_dag_verify.corrected_verdict import eps_tag  # noqa: E402
from pfn_dag_verify.corrected_world import make_world  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--scale", choices=sorted(SCALES), default="base")
    p.add_argument("--n-rows", type=int, default=20)
    p.add_argument("--ckpt-steps", type=int, nargs="*", default=[0])
    p.add_argument("--peak-lr", type=float, default=1e-3,
                   help="train_pfn's peak_lr; the registered recipe is 1e-3. A value other "
                        "than 1e-3 is written into the net filenames as _lr{value}.")
    p.add_argument("--outroot", type=Path, required=True)
    p.add_argument("--world-seed", type=int, default=None,
                   help="make_world(seed=); None = the registered atom library")
    a = p.parse_args()

    tag = eps_tag(a.eps)
    world = make_world(k=a.K, d=a.d, eps=a.eps, seed=a.world_seed)
    cfg = ModelConfig(name=a.scale, d=a.d, **SCALES[a.scale])
    # The net prefix must be unique per (scale, n_rows, lr) within one netdir, or
    # an n_rows-40 fleet overwrites the n_rows-20 checkpoints of the same eps.
    lr_tag = "" if a.peak_lr == 1e-3 else f"_lr{a.peak_lr:g}"
    if a.n_rows != 20:
        lr_tag = f"_n{a.n_rows}" + lr_tag
    netdir = a.outroot / "nets" / f"eps{tag}"
    ckpts = tuple(sorted(set(int(s) for s in a.ckpt_steps) | {a.steps}))
    t0 = time.time()
    res = train_pfn(world, steps=a.steps, seed=a.seed, cfg=cfg, n_ctx=a.n_rows,
                    n_query=7, peak_lr=a.peak_lr, ckpt_steps=ckpts, outdir=netdir,
                    tag_prefix=f"{a.scale}{lr_tag}_s{a.seed}")
    doc = {
        "diagnostic": True, "purpose": "EXT campaign world-seed replication (mech_train_ws.py)",
        "world": {"d": a.d, "K": a.K, "eps": a.eps, "spec_r": world.spec.r, "world_seed": a.world_seed,
                  "sigmas_sha256": __import__("hashlib").sha256(np.ascontiguousarray(world.sigmas, dtype=np.float64).tobytes()).hexdigest()},
        "scale": a.scale, "cfg": cfg.__dict__, "n_rows": a.n_rows, "seed": a.seed,
        "peak_lr": a.peak_lr, "net_prefix": f"{a.scale}{lr_tag}_s{a.seed}",
        "steps": a.steps, "ckpt_steps": list(ckpts), "n_params": res["n_params"],
        "final_loss": res["final_loss"], "train_wallclock_s": float(time.time() - t0),
        "device": str(DEV), "torch": torch.__version__, "numpy": np.__version__,
        "trainer": "pfn_dag_verify.corrected_models.train_pfn (same call as w2_run.run)",
    }
    out = a.outroot / f"train_{a.scale}{lr_tag}{'' if a.n_rows != 20 else '_n20'}_eps{tag}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str))
    print(f"[mech_train] {a.scale} n{a.n_rows} eps={a.eps} s{a.seed}: {a.steps} steps, "
          f"final_loss {res['final_loss']:.4f}, {doc['train_wallclock_s'] / 3600:.2f} h -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
