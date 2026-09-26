# Reproduction scope

The public package includes the portable finite prior, predictive oracle, original PFN architecture, coupled targets, six scientific tests, saved five-seed point estimates and a checkpoint-state comparator. Scientific modules are unchanged from the verified portable source snapshot. The public export adapts documentation, integrity checks and the test's provenance-file path.

| Capability | Supported scope |
| --- | --- |
| Scientific identities | Common expected loss, covariance decomposition, signed-loss gradients, query-label isolation, deterministic inputs and quadrature |
| CPU training and source evaluation | Four arms, two updates by default, four training rows and two held-out rows |
| Saved contrast reconstruction | Published point estimates from the included five-seed score fixture |
| Checkpoint comparison | Comparator and self-test; actual comparison requires trusted external checkpoint pairs |
| Full training, shifted scoring and confidence intervals | Production banks, model checkpoints, query-level predictions and resampling arrays are not included |

The tiny output is an execution check, not fresh evidence for the reported source/shift effect. The score fixture contains seed means rather than the raw query-level arrays needed to reproduce confidence intervals. The replay fixture records historical comparisons; its presence does not reproduce those comparisons or demonstrate continuation.

The package preserves the documented fixed-bin support adapter and trimmed original modules. Their portable identities differ from the original production files; both sets of hashes are recorded in `manifests/scientific-provenance.json`. The production recipe and environment versions are documentary and do not form a complete runnable production setup. Historical execution identities, complete states and required input streams are necessary for continuation.

This export follows the repository's existing public selection policy. Manuscript prose, figures, internal reports, private retrieval paths, operational records and private repository history are excluded. No public artifact endpoint for the missing production inputs is supplied here. The earlier public release assets described in the root README belong to their original studies and must not be treated as inputs for this later four-arm study.
