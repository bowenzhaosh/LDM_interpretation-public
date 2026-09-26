# Reproduction scope

This repository accompanies the manuscript *Separating Supervision Randomness in Prior-Data Fitted Networks*, currently under submission. It includes the portable finite prior, predictive oracle, original PFN architecture, coupled targets, six checks covering scientific identities, provenance, and saved contrasts, and a checkpoint-state comparator. Numerical computations retain the verified portable implementation. The public adaptation updates documentation, integrity checks and the test's provenance-file path; the contrast command describes its output as saved means.

| Capability | Supported scope |
| --- | --- |
| Scientific identities | Common expected loss, covariance decomposition, signed-loss gradients, query-label isolation, deterministic inputs and quadrature |
| CPU training and source evaluation | Four arms, two updates by default, four training rows and two held-out rows |
| Saved contrast reconstruction | Reported point estimates from the included five-seed score fixture |
| Checkpoint comparison | Comparator and self-test; actual comparison requires trusted external checkpoint pairs |
| Full training, shifted scoring and confidence intervals | Production banks, model checkpoints, query-level predictions and resampling arrays are not included |

The tiny output is an execution check, not fresh evidence for the reported source/shift effect. The score fixture contains seed means rather than the raw query-level arrays needed to reproduce confidence intervals. The replay fixture records historical comparisons; its presence does not reproduce those comparisons or demonstrate continuation.

The package preserves the documented fixed-bin support adapter and trimmed original modules. Their portable identities differ from the original production files; both sets of hashes are recorded in `manifests/scientific-provenance.json`. The production recipe and environment versions are documentary and do not form a complete runnable production setup. Historical execution identities, complete states and required input streams are necessary for continuation.

The current companion package follows the existing public selection policy: manuscript prose, figures, internal reports, private retrieval paths, operational records and private repository history are excluded. No public artifact endpoint for the missing production inputs is supplied here. The preserved earlier public tree under `legacy/pfn-dag/` has its own historical export policy and may contain old cluster paths in result records. Its release assets belong to the earlier studies and must not be treated as inputs for this four-arm study.
