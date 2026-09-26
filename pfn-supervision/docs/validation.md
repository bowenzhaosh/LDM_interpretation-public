# Public export validation, 2026-09-26

The public package was installed in a fresh Python 3.11.7 virtual environment on macOS arm64 using the included runtime lock. NumPy was 1.26.4 and PyTorch was 2.9.1. The environment is separate from the historical export at the repository root.

| Check | Result |
| --- | --- |
| Fresh editable installation and dependency check | PASS |
| Scientific invariant and provenance tests | PASS; all six tests |
| Saved five-seed contrast reconstruction | PASS |
| Checkpoint comparator self-test | PASS; equality case and ten mismatch cases |
| Four-arm CPU training and source evaluation | PASS; two updates per arm |
| CPU result versus the verified source snapshot | PASS; parsed JSON exactly equal |
| Public export checksums and seven pinned scientific files | PASS |

Scientific modules, configurations, dependency pins and fixtures retain their source bytes. The public adaptation changes documentation, the provenance verifier and the test's provenance-file path. The new file selection was checked for private filesystem roots, archive references and credential patterns. No private Git objects were imported into this repository.

The earlier public tree is preserved; only its README gains a link to this package. Its historical test suite was not rerun. Full production training, raw-array uncertainty reconstruction, actual checkpoint comparison and continuation were not run and require artifacts outside this export. The CPU example and comparator self-test do not establish those capabilities.
