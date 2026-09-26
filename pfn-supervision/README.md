# PFN: separating supervision randomness

This public package provides the finite structural prior, posterior oracle, PFN architecture and coupled supervision targets, together with a small CPU example and frozen score fixtures. The scientific code is the portable extraction used in the September 25 artifact. Original source hashes and portable-file hashes are retained in [scientific provenance](manifests/scientific-provenance.json).

## Install and check

Run from this directory, `pfn-supervision/`, with Python 3.11. Use a separate virtual environment from the earlier export at the repository root.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r locks/reviewer-requirements.txt -e .
.venv/bin/python scripts/verify_provenance.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce_contrast.py
.venv/bin/python scripts/compare_checkpoints.py --self-test
```

The [environment notes](locks/README.md) distinguish the portable environment from the documentary production environment. The [validation record](docs/validation.md) reports the checks run on this export.

## Run the CPU example

```sh
.venv/bin/python scripts/train.py --config configs/reviewer/tiny.json --out outputs/tiny.json
```

This runs Q, qz, signed and FreshY1 for two updates using paired initialization and input traversal, then scores held-out source inputs. It checks software execution; it does not reproduce the full study's effect, uncertainty intervals or production training. The saved-contrast command separately reconstructs point estimates from a five-seed fixture.

| Location | Contents |
| --- | --- |
| `src/pfn_dag_verify/` | Finite prior, oracle, model, targets, CPU training/evaluation and comparator |
| `configs/` | Tiny execution config and documentary production recipe |
| `scripts/` | Training, saved-contrast reconstruction, checkpoint comparison and integrity check |
| `tests/` | Six scientific invariant and provenance tests |
| `examples/reviewer/` | Example guide and frozen score/replay fixtures |
| `locks/` | Portable dependency pins, tested platform lock and historical runtime versions |
| `manifests/` | Original scientific provenance and public export checksums |
| `docs/` | Method, reproduction limits and validation |

Read the [method](docs/method.md) and [reproduction scope](docs/reproduction-scope.md) before interpreting outputs. This export includes no manuscript, figures, private archive index, operational records or private Git history. Earlier public research remains at the repository's original paths and has its own documentation and environment.
