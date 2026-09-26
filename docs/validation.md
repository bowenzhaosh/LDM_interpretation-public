# Publication companion validation, 2026-09-26

The repository-root package was installed in a fresh Python 3.11.7 environment on macOS arm64 using the included runtime lock. NumPy was 1.26.4 and PyTorch was 2.9.1.

| Check | Result |
| --- | --- |
| Fresh editable installation and dependency check | PASS |
| Scientific identities, provenance and saved-contrast tests | PASS; all six checks |
| Saved five-seed contrast reconstruction | PASS |
| Checkpoint comparator self-test | PASS; equality case and ten mismatch cases |
| Four-arm CPU training and source evaluation | PASS; two updates per arm |
| CPU result versus the preceding public export | PASS; parsed JSON exactly equal |
| Public checksums and seven original scientific-file pins | PASS |
| Historical snapshot preservation | PASS; relocated Git tree equals the original 2,592-file public tree |

The numerical implementation, configs and fixtures retain their source bytes. One contrast-output description now says “saved means” to match the manuscript's submission status. Documentation, package metadata, integrity tooling and repository layout were updated. The MIT license was selected by the repository owner.

[GitHub Actions](https://github.com/bowenzhaosh/pfn-supervision-public/actions/workflows/ci.yml) performs a regular package installation on Linux with CPU PyTorch, then runs provenance verification, the six checks, saved-contrast reconstruction, comparator self-test and bounded training. Its live status is shown on the README badge.

The earlier public studies are preserved byte-for-byte under `legacy/pfn-dag/` with their own environment. Their historical test suite was not rerun. Full production training, raw-array uncertainty reconstruction, actual checkpoint comparison and continuation were not run and require artifacts outside the current companion package. The CPU example and comparator self-test do not establish those capabilities.
