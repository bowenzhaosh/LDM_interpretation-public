# Provenance — ext3 binary-duel line, vendored 2026-08-21

**Source:** `~/pfn-dag/experiments/binary-duel/` — an unversioned working directory on one local disk,
outside any git repository (`~/pfn-dag` is not a git work tree). Vendored here verbatim under R2 of the
signed plan v2 ("nothing decision-relevant stays untracked"), excluding `__pycache__` and `*.pyc`.

**Relationship to `campaigns/binary_duel_20260813/`.** That earlier campaign is a partial snapshot of the
same line: `exp1/exp2/exp3` at prior=AL40, scale=base, dose=12000, seeds s0-s3 only (12 result JSONs). This
directory is the live superset and supersedes it for all C1-C4 and C6 numbers:

| stream | files here | in the 08-13 snapshot |
|---|---|---|
| `results/exp1` (tracking, C1/C2/C3) | 55 | 4 |
| `results/exp2` (mislead, C4)        | 54 | 4 |
| `results/exp3` (Phase-A screen, C6) |  4 | 4 |
| `results/exp3b` (Phase-B + Amendment-D confirmation, C6b) | 8 | absent |
| `results/exp4` (d=3 readout port)   | 11 | absent |

It also carries `PREREG-EXT3.md` and `aggregate_ext3.py` / `aggregate_ext3.json`, none of which existed in
version control before today.

**Status of the numbers in this directory.** Pre-Amendment-E. Specifically:
- the d=2 estimands are single-sampled-outcome, superseded by Amendment E.3 (exact expectations);
- `exp4` (d=3) inherits the E.1 mixture defect wherever `0 < eps < 1` is involved;
- `aggregate_ext3.py` applies gates by pooling across scales; per Amendment E it must filter to a declared
  cell before any gate is evaluated.

Nothing here is citable until re-derived under Amendment E. It is committed so that the pre-fix values exist
in the record for the E.10 errata appendix, and so that the line stops depending on one undamaged local disk.
