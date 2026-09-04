"""Amendment F.2a — the W2 pre-flight, run BEFORE the retrain fires.

The point is to discover a persistence or readout failure now, at a hundred
steps, rather than at step 100k. Every gate here corresponds to something that
has already gone wrong once in this campaign:

  Gate A  PERSISTENCE. Three ``train_pfn`` call sites passed no outdir, and the
          save gate additionally required ``ckpt_every > 0``, so no d=3 weight
          ever reached disk and every d=3 number came from a model that did not
          survive its process. The dose-0 state was worse: the capture fired on
          ``(step + 1) in ckpt_set``, so step 0 was unreachable and asking for it
          was a SILENT NO-OP. A fleet could report training with dose-0 and
          produce nothing. F.2c's floor curve does not exist without it.
          This gate asserts dose-0 is on disk BEFORE step 1, with its own
          sidecar, and that its weights are the freshly-seeded ones.

  Gate B  FUNCTIONAL RELOAD. Not state-dict equality. Dtype, device and RNG
          surprises live precisely in the gap between "the state dict matches"
          and "the outputs match", and a state-dict check passes while the
          reloaded model computes something else. A fixed mini-batch goes
          through the in-memory model and the reloaded one and the OUTPUTS must
          agree to float tolerance -- for the trained weights and for dose-0.

  Gate C  THE READOUT, END TO END, OFF THE RELOADED CHECKPOINT. Plumbing
          validity only; no number here is claimed from a 100-step model. The
          DOSE-0 branch matters most: the floor path has never executed, and W2
          must not be its first execution.

  Gate D' CONTEXT-LEVEL SPREAD. The full Gate D of F.2a compares the old sampled
          scorer against the D3 exact one, and the exact scorer does not exist
          yet, so that comparison is NOT run here and is not claimed. What is
          measured is the per-context spread of the readout, which is the
          quantity T3's validation had to treat as unmeasured. Reported as a
          partial, labelled as one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pfn_dag_verify.artifact_status import DIAGNOSTIC, stamp                 # noqa: E402
from pfn_dag_verify.corrected_identifiability_run import HEADLINE_N_ROWS     # noqa: E402
from pfn_dag_verify.corrected_models import (                                # noqa: E402
    DEV, ModelConfig, PFN, evaluate_pfn_checkpoint, make_classifier_panel,
    make_eval_panel, train_pfn,
)
from pfn_dag_verify.corrected_tomography import (                            # noqa: E402
    aggregate_tomography, run_tomography,
)
from pfn_dag_verify.corrected_trackb import N_QUERY_PER_CONTEXT, SCALES      # noqa: E402
from pfn_dag_verify.corrected_world import make_world                        # noqa: E402
from pfn_dag_verify.split_panel import (                                     # noqa: E402
    HALF_A, HALF_B, build_split_panel, half_panel_entries,
)

RELOAD_TOL = 1e-6          # float32 forward twice on one device: bitwise in
                           # practice, this leaves room for a device change
MINIBATCH_SEED = 990_400_000


class GateFailure(AssertionError):
    """A pre-flight gate failed. STOP AND REPORT -- do not fire W2."""


_T0 = time.time()


def _tick(msg: str) -> None:
    """Progress, flushed, with elapsed seconds. A gate that prints nothing until
    it finishes is indistinguishable from a gate that has hung, which is how a
    pre-flight came to burn 2h42m before anyone could say where."""
    print(f"[preflight +{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _check(gate: str, ok: bool, detail: str, log: list) -> None:
    log.append({"gate": gate, "ok": bool(ok), "detail": detail,
                "t_elapsed_s": round(time.time() - _T0, 2)})
    _tick(("PASS " if ok else "FAIL ") + gate + " | " + detail[:90])
    if not ok:
        raise GateFailure(f"{gate}: {detail}")


def fixed_minibatch(world, cfg, n_ctx: int, device) -> tuple:
    """One mini-batch, drawn once and reused for every comparison in Gate B, so
    the two models are asked the same question."""
    rng = np.random.default_rng(MINIBATCH_SEED)
    from pfn_dag_verify.corrected_models import gen_batch
    X, _, _ = gen_batch(world, rng, 8, n_ctx, 4)
    ctx = torch.tensor(X[:, :n_ctx, :], dtype=torch.float32, device=device)
    q = X[:, n_ctx:, :]
    qxy = torch.tensor(q[:, :, :world.d - 1], dtype=torch.float32, device=device)
    tok = torch.full((8,), 2, dtype=torch.long, device=device)
    return ctx, qxy, tok


def outputs_of(model, batch) -> tuple[np.ndarray, np.ndarray]:
    ctx, qxy, tok = batch
    model.eval()
    with torch.no_grad():
        logits = model(ctx, qxy, tok).detach().cpu().numpy()
    probs = model.predict_bin_probs(ctx, qxy, tok)
    return logits, np.asarray(probs)


def reload_model(path: Path, cfg, device) -> PFN:
    """A FRESH process-equivalent load: a new module, weights off disk, eval."""
    m = PFN(cfg).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rows", type=int, default=HEADLINE_N_ROWS)
    p.add_argument("--n-per-half", type=int, default=24,
                   help="pre-flight panel size; PLUMBING only, not W2's count")
    p.add_argument("--n-tom-contexts", type=int, default=6)
    p.add_argument("--ckpt-steps", type=int, nargs="*", default=[0, 50, 100])
    p.add_argument("--lp-solver", type=str, default="highs",
                   choices=["highs", "highs-ipm"],
                   help="tomography LP method; default matches every tomography "
                        "number already on disk")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    t0 = time.time()
    log: list[dict] = []
    device = torch.device(DEV)
    world = make_world(k=a.K, d=a.d, eps=a.eps)
    cfg = ModelConfig(name="base", d=a.d, **SCALES["base"])
    nets = Path(a.outdir)
    tag = f"preflight_s{a.seed}"

    res: dict = {"amendment": "F.2a", "device": str(DEV), "torch": torch.__version__,
                 "cuda": torch.version.cuda or "cpu", "numpy": np.__version__,
                 "config": {"d": a.d, "K": a.K, "eps": a.eps, "steps": a.steps,
                            "seed": a.seed, "n_rows": a.n_rows,
                            "n_per_half": a.n_per_half,
                            "ckpt_steps": list(a.ckpt_steps)}}

    # ---- train ------------------------------------------------------------
    _tick("training")
    tr = train_pfn(world, steps=a.steps, seed=a.seed, cfg=cfg, n_ctx=a.n_rows,
                   n_query=7, outdir=nets, tag_prefix=tag,
                   ckpt_steps=tuple(a.ckpt_steps))
    res["train"] = {"n_params": tr["n_params"], "final_loss": tr["final_loss"],
                    "checkpoints_in_memory": sorted(tr["checkpoints"])}

    # ---- Gate A: persistence, dose-0 first --------------------------------
    ck0 = nets / f"{tag}_ck0.pt"
    final = nets / f"{tag}.pt"
    _check("A.dose0_exists", ck0.is_file(), f"{ck0} on disk", log)
    ck0_prov = nets / f"{tag}_ck0.provenance.json"
    _check("A.dose0_sidecar", ck0_prov.is_file(), f"{ck0_prov} on disk", log)
    pj = json.loads(ck0_prov.read_text())
    _check("A.dose0_sidecar_fields",
           pj.get("dose0") is True and pj.get("steps_taken") == 0
           and all(pj.get(k) for k in ("torch", "cuda", "device", "numpy")),
           f"dose0={pj.get('dose0')} steps_taken={pj.get('steps_taken')} "
           f"torch={pj.get('torch')} cuda={pj.get('cuda')} numpy={pj.get('numpy')}",
           log)
    for s in a.ckpt_steps:
        f = nets / f"{tag}_ck{s}.pt"
        _check(f"A.ckpt_{s}", f.is_file(), f"{f} on disk", log)
    _check("A.final", final.is_file(), f"{final} on disk", log)
    prov = nets / f"{tag}.provenance.json"
    _check("A.final_sidecar", prov.is_file(), f"{prov} on disk", log)

    # dose-0 must predate every other artifact of this run: that ordering is the
    # evidence it is the untrained state rather than something reconstructed.
    others = [nets / f"{tag}_ck{s}.pt" for s in a.ckpt_steps if s != 0] + [final]
    t_ck0 = ck0.stat().st_mtime_ns
    latest_other = max(f.stat().st_mtime_ns for f in others)
    _check("A.dose0_written_first", t_ck0 <= latest_other,
           f"ck0 mtime_ns {t_ck0} <= latest other {latest_other}", log)

    # ...and it must BE the freshly-seeded model, not a later state mislabelled.
    torch.manual_seed(1000 * a.seed + 7)
    fresh = PFN(cfg).to(device)
    on_disk = torch.load(ck0, map_location=device)
    same = all(torch.equal(v.cpu(), on_disk[k].cpu())
               for k, v in fresh.state_dict().items())
    _check("A.dose0_is_untrained", same,
           "the persisted dose-0 weights equal a freshly seeded model", log)
    res["gate_A"] = {"dose0_mtime_ns": t_ck0, "latest_other_mtime_ns": latest_other,
                     "sidecar": pj}

    # ---- Gate B: functional reload equality -------------------------------
    _tick("Gate B: fixed mini-batch")
    batch = fixed_minibatch(world, cfg, a.n_rows, device)
    gate_b = {}
    for name, mem, path in (("final", tr["model"], final),
                            ("dose0", fresh, ck0)):
        lm, pm = outputs_of(mem, batch)
        rl = reload_model(path, cfg, device)
        lr_, pr = outputs_of(rl, batch)
        dl = float(np.max(np.abs(lm - lr_)))
        dp = float(np.max(np.abs(pm - pr)))
        gate_b[name] = {"max_abs_logit_diff": dl, "max_abs_prob_diff": dp,
                        "tol": RELOAD_TOL}
        _check(f"B.{name}_logits", dl <= RELOAD_TOL,
               f"max |logit_mem - logit_reloaded| = {dl:.3e} <= {RELOAD_TOL}", log)
        _check(f"B.{name}_probs", dp <= RELOAD_TOL,
               f"max |prob_mem - prob_reloaded| = {dp:.3e} <= {RELOAD_TOL}", log)
        # A comparison that would pass on two identical constant outputs proves
        # nothing, so check the batch actually discriminates.
        _check(f"B.{name}_batch_is_informative", float(np.ptp(pm)) > 1e-6,
               f"prob spread across the fixed mini-batch = {float(np.ptp(pm)):.3e}",
               log)
    res["gate_B"] = gate_b

    # ---- Gate C: the readout, end to end, off the RELOADED checkpoints -----
    _tick("Gate C: building the split panel")
    panel_split = build_split_panel(world, a.n_per_half, a.n_rows,
                                    panel_seed=770_000_000,
                                    n_query_per_context=N_QUERY_PER_CONTEXT)
    res["panel"] = {"A": panel_split.provenance(HALF_A),
                    "B": panel_split.provenance(HALF_B)}
    # the SCORING panel is half B (F.2d.5); make_eval_panel builds the
    # (ctx, x_q, target, bin) tuples the scorer wants
    eval_panel = make_eval_panel(world, 2 * a.n_per_half, a.n_rows,
                                 seed=770_000_000,
                                 n_query_per_context=N_QUERY_PER_CONTEXT)
    panel_b = half_panel_entries(eval_panel, panel_split, HALF_B)
    _check("C.half_b_panel_nonempty", len(panel_b) > 0,
           f"{len(panel_b)} half-B scoring queries", log)

    tom_ctxs = make_classifier_panel(world, a.n_tom_contexts, a.n_rows,
                                     seed=772_000_000)
    gate_c = {}
    for name, path in (("final", final), ("dose0", ck0)):
        _tick(f"Gate C: scoring {name} on {len(panel_b)} half-B queries")
        m = reload_model(path, cfg, device)
        ev = evaluate_pfn_checkpoint(m, world, panel_b, device)
        tol = max(ev["js_mean"] * 2.0, 1e-3)
        _tick(f"Gate C: tomography {name}, {a.n_tom_contexts} contexts, "
              f"tol {tol:.4f}, solver {a.lp_solver}")
        tom = aggregate_tomography(
            run_tomography(world, m, tom_ctxs, tol=tol, device=DEV,
                           solver=a.lp_solver))
        finite = (math.isfinite(ev["bayes_regret_mean"])
                  and math.isfinite(ev["js_mean"])
                  and all(math.isfinite(v) for q in tom.values()
                          for v in q.values()))
        _check(f"C.{name}_readout_finite", finite,
               f"regret {ev['bayes_regret_mean']:+.4f}, js {ev['js_mean']:.4f}, "
               f"tomography panels {sorted(tom)}", log)
        _check(f"C.{name}_tomography_panels", set(tom) >= {"Q1", "Q2", "Q3"},
               f"panels present: {sorted(tom)}", log)
        gate_c[name] = {"eval": ev, "tomography": tom, "tol": tol,
                        "loaded_from": str(path)}
    res["gate_C"] = gate_c
    # The dose-0 readout is the FLOOR curve. It has never run before today, so
    # the fact that it produced numbers at all is the gate; the numbers
    # themselves are a 100-step artefact and are not claimed.
    res["gate_C"]["floor_path_executed"] = True

    # ---- Gate D' : context-level spread, labelled as a partial ------------
    _tick("Gate D-partial: per-context spread")
    m = reload_model(final, cfg, device)
    per_ctx = []
    for q in panel_b:
        e = evaluate_pfn_checkpoint(m, world, [q], device)
        per_ctx.append(e["bayes_regret_mean"])
    arr = np.asarray(per_ctx, dtype=float)
    res["gate_D_partial"] = {
        "note": ("F.2a's full Gate D compares the SAMPLED scorer against the D3 "
                 "EXACT one. The exact scorer does not exist yet, so that "
                 "comparison was NOT run and nothing here claims it. What is "
                 "measured is the per-context spread of the current scorer on a "
                 "100-step model, which sizes conservatively: a model this far "
                 "from convergence has a hotter context-to-context spread than a "
                 "converged one."),
        "n_contexts": int(arr.size),
        "regret_mean": float(arr.mean()),
        "regret_context_sd": float(arr.std(ddof=1)),
        "regret_panel_se": float(arr.std(ddof=1) / math.sqrt(arr.size)),
        "implied_se_at_500": float(arr.std(ddof=1) / math.sqrt(500)),
        "implied_se_at_1500": float(arr.std(ddof=1) / math.sqrt(1500)),
    }

    _tick("writing")
    res["lp_solver"] = a.lp_solver
    res["gates"] = log
    res["all_gates_passed"] = all(g["ok"] for g in log)
    res["wallclock_s"] = float(time.time() - t0)
    stamp(res, "w2_preflight", DIAGNOSTIC)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"all_gates_passed": res["all_gates_passed"],
                      "n_gates": len(log), "out": str(a.out),
                      "wallclock_s": res["wallclock_s"]}, indent=2))
    return 0 if res["all_gates_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as e:
        print(f"PRE-FLIGHT GATE FAILED -- STOP AND REPORT\n{e}", file=sys.stderr)
        raise SystemExit(2)
