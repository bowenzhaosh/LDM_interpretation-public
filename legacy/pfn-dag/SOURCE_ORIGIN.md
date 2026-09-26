# Source and artifact origin

The verification harness in `src/pfn_dag_verify/` was written independently for this audit. It does not import experiment, oracle, metric, or report code from the generated ten-experiment pipeline during scientific evaluation.

The following artifacts were copied into the repository before the run lock was created:

- `artifacts/checkpoints/M_base_AL40_s{0..15}_dose{0,12000}.pt` from `~/pfn-dag/G-experiments/e18b-committed/dose_nets/`;
- `artifacts/source_snapshots/` from `~/pfn-dag/G-experiments/e18b-committed/`;
- `artifacts/legacy/stage1_functional_law.py` from `~/pfn-dag/evidence-integration/`.

`config/checkpoint_registry.json` records the size and SHA-256 of every copied checkpoint and source snapshot. The loader resolves only repository-relative paths, verifies every hash and tensor schema, and rejects a step-0/step-12000 pair with identical content.

The legacy Stage-1 snapshot is used only for a locked pre-run equivalence check against the independent oracle. Its result records the snapshot hash. Scientific panel generation and checkpoint scoring use only the independent implementation.

The original training stream was online and no complete sample ledger or immutable upstream release was available. This is an explicit provenance limitation. The result can verify behavior of the bundled checkpoint bytes, not reconstruct their full training history.
