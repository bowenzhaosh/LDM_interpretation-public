# Environments

`reviewer-requirements.txt` preserves the submission's NumPy 1.26.4 and PyTorch 2.9.1 pins. `pyproject.toml` uses the same runtime pins and fixes the build tools. Python 3.11 is the supported portable interpreter; validation uses 3.11.7 on macOS arm64.

`reviewer-macos-arm64-py311.lock.txt` records the complete runtime dependency resolution from the clean validation environment. Install it before the local package when reproducing that environment:

```sh
.venv/bin/python -m pip install -r locks/reviewer-macos-arm64-py311.lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

This is a platform-specific version lock, not a wheel-hash lock or a claim of cross-platform numerical identity. For another platform, resolve the two direct reviewer pins and record/test the resulting environment separately.

`production-environment.json` preserves documented production versions separately. It is an incomplete historical environment record, not an installable production lock. It lacks the full transitive package set, CUDA driver/system libraries and complete machine state. A production restart requires its original runtime receipts and checkpoint/input identities.
