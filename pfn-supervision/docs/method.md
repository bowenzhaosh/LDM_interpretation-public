# Scientific code map

`corrected_sem.py`, `corrected_world.py` and `corrected_oracle.py` implement the finite structural prior and posterior predictive calculations. `corrected_models.py` contains the PFN architecture, input generation and learning-rate schedule. `pilot_shared.py` supplies the portable fixed bin edges and quadrature. These are the authenticated portable extractions in the selected submission package, with original production hashes retained in its provenance record.

`target_construction.py` builds the coupled Q, component qz, signed and posterior-resampled FreshY1 targets. The signed loss is the linear expression CE(Q) + CE(Y) - CE(qz), including negative target entries. `training.py` uses the same numerical functions as the submission example. Repository changes only relocate imports and make the configuration path explicit.

`evaluation.py` reconstructs saved point contrasts from an explicit fixture. The small training runner separately computes held-out source KL. Neither operation recovers the full paper's query-level inference or generates shifted training results.

The existing scientific tests check the common expected loss, covariance decomposition and its common-Jacobian projection, signed-loss/autograd agreement, query-label isolation, deterministic input generation, quadrature equivalence and reported saved contrasts. They do not qualify a migrated production campaign or its continuation states.
