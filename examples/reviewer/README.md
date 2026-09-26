# Reviewer CPU example

Install the repository following the root README, then run from its root:

```sh
.venv/bin/python scripts/train.py --config configs/reviewer/tiny.json --out outputs/tiny.json
.venv/bin/python scripts/reproduce_contrast.py
```

The example generates the original finite support and coupled targets, uses four training rows and two held-out source rows, and runs two updates for each of Q, qz, signed and FreshY1. The runner checks shared initialization and traversal, finite gradients and distinct final arm states. Its JSON records parameters, package versions, input/support hashes, per-arm losses and source KL. It writes no checkpoint.

The covariance support has eight atoms and six orderings. FreshY1 independently resamples a posterior component for each query and visit, with its label coupled to that component. The signed objective is CE(Q) + CE(Y) - CE(qz); targets are not clipped or renormalized. See [method](../../docs/method.md).

`fixtures/withheld_seed_risks.json` reconstructs the saved five-seed point contrasts. It does not contain query-level arrays or bootstrap draws. `fixtures/replay_validation.json` records historical checkpoint comparisons; it is not a replacement for the checkpoint bytes. Both files are unchanged copies from the selected submission ZIP.

The original source and portable scientific-file hashes are recorded in `manifests/scientific-provenance.json`. The commands exercise the installed package in `src/` and use explicit repository inputs.
