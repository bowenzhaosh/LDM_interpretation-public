# Contributing

Use Python 3.11 and follow the installation steps in the README. Keep generated results under `outputs/` and preserve the frozen fixtures and source provenance.

Before proposing a change, run:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce_contrast.py
.venv/bin/python scripts/compare_checkpoints.py --self-test
.venv/bin/python scripts/train.py --config configs/reviewer/tiny.json --out outputs/tiny.json
```

For a scientific change, explain which assumption or computation changed and add a test for the affected property. The signed objective must retain negative target entries, and query labels must remain isolated from model inputs. Record deliberate departures from the original scientific source; do not update a source pin merely to silence a failing check.

After reviewing intentional changes, stage the current package files and refresh its checksum manifest:

```sh
git add <changed-files>
.venv/bin/python scripts/update_manifest.py
.venv/bin/python scripts/verify_provenance.py
git add manifests/public-export.json
```

The update command hashes tracked current-package files and excludes `legacy/`, generated outputs, and the checksum manifest itself. Review its diff. The original scientific-file pins remain a separate check.

Keep `legacy/pfn-dag/` unchanged so its snapshot stays verifiable. New work should use the current package and a distinct output location. Report reproduction limits and skipped checks with the change.
