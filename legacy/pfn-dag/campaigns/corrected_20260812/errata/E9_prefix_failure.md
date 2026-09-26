# E.9 pre-fix failure capture — sampler<->density invariance

Repo state: HEAD 1bea6ae, corrected_sem.py sha256 7277d76bc201801a..., corrected_models.py sha256 521509caaef47d56...
Captured 2026-08-21T18:59:14Z BEFORE the E.1 mask unification.

Amendment E.9 requires this test to FAIL on pre-fix code at eps in {0.1,0.25,0.5}
for sample_residuals and _vectorized_observational, and to PASS post-fix.

```
AssertionError: sample_residuals at eps=0.1: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.00848 +/- 0.00079, z=-10.8). The generation path does not sample from the law residual_logpdf scores.
AssertionError: sample_residuals at eps=0.25: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.02662 +/- 0.00137, z=-19.4). The generation path does not sample from the law residual_logpdf scores.
AssertionError: sample_residuals at eps=0.5: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.04272 +/- 0.00174, z=-24.6). The generation path does not sample from the law residual_logpdf scores.
AssertionError: generate_observational at eps=0.1: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.00787 +/- 0.00077, z=-10.2). The generation path does not sample from the law residual_logpdf scores.
AssertionError: generate_observational at eps=0.25: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.02676 +/- 0.00138, z=-19.4). The generation path does not sample from the law residual_logpdf scores.
AssertionError: generate_observational at eps=0.5: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.04566 +/- 0.00176, z=-25.9). The generation path does not sample from the law residual_logpdf scores.
AssertionError: _vectorized_observational at eps=0.1: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.00687 +/- 0.00072, z=-9.5). The generation path does not sample from the law residual_logpdf scores.
AssertionError: _vectorized_observational at eps=0.25: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.02248 +/- 0.00123, z=-18.3). The generation path does not sample from the law residual_logpdf scores.
AssertionError: _vectorized_observational at eps=0.5: samples favour the 'row' mixing law over the declared per-coordinate law (mean log-density difference -0.03703 +/- 0.00153, z=-24.1). The generation path does not sample from the law residual_logpdf scores.
FAILED tests/test_sampler_density_invariance.py::test_sample_residuals_uses_declared_law[0.1]
FAILED tests/test_sampler_density_invariance.py::test_sample_residuals_uses_declared_law[0.25]
FAILED tests/test_sampler_density_invariance.py::test_sample_residuals_uses_declared_law[0.5]
FAILED tests/test_sampler_density_invariance.py::test_generate_observational_uses_declared_law[0.1]
FAILED tests/test_sampler_density_invariance.py::test_generate_observational_uses_declared_law[0.25]
FAILED tests/test_sampler_density_invariance.py::test_generate_observational_uses_declared_law[0.5]
FAILED tests/test_sampler_density_invariance.py::test_vectorized_observational_uses_declared_law[0.1]
FAILED tests/test_sampler_density_invariance.py::test_vectorized_observational_uses_declared_law[0.25]
FAILED tests/test_sampler_density_invariance.py::test_vectorized_observational_uses_declared_law[0.5]
9 failed, 10 passed in 5.10s
```

## Negative control, 2026-08-22

Re-run as a standing check that the test still has power rather than merely
passing. `sample_residuals` was monkeypatched back to the 9eca9ac9 per-ROW mask
at runtime (the shipped module untouched) and the suite re-run:

    6 failed, 13 passed
    sample_residuals        eps 0.1 / 0.25 / 0.5   z = -10.8 / -19.4 / -24.6
    generate_observational  eps 0.1 / 0.25 / 0.5   z = -10.2 / -19.4 / -25.9

Six, not the nine recorded above, and the difference is the point rather than a
discrepancy: this control reverts ONE of the two defects. Patching
`sample_residuals` breaks the two paths that call it; `_vectorized_observational`
kept its own per-DATASET mask in the original pre-fix state, which is where the
other three failures came from. Reverting one defect failing exactly six tests
is what the fault model predicts.

On the shipped law the same suite is 19/19 in 5.0 s, so the test is cheap enough
to keep in the default suite and discriminating enough to be worth it.
