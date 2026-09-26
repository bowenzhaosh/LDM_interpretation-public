# Separating Supervision Randomness in Prior-Data Fitted Networks

[![Checks](https://github.com/bowenzhaosh/pfn-supervision-public/actions/workflows/ci.yml/badge.svg)](https://github.com/bowenzhaosh/pfn-supervision-public/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code companion for the manuscript **Separating Supervision Randomness in Prior-Data Fitted Networks**, currently under submission.

The package implements a finite structural prior, posterior predictive oracle, PFN model, and four coupled supervision targets: `Q`, `qz`, `signed`, and `FreshY1`. It includes a small CPU training example, checks of the scientific identities, and reconstruction of saved five-seed point estimates. Full production training and uncertainty reconstruction require artifacts outside this release; see [reproduction scope](docs/reproduction-scope.md).

## Quick start

Clone the repository, then create a Python 3.11 environment:

```sh
git clone https://github.com/bowenzhaosh/pfn-supervision-public.git
cd pfn-supervision-public
python3.11 -m venv .venv
```

On Linux, first install the CPU-only PyTorch wheel:

```sh
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

Install the package and check it on either platform:

```sh
.venv/bin/python -m pip install -r locks/reviewer-requirements.txt -e .
.venv/bin/python scripts/verify_provenance.py
.venv/bin/python -m unittest discover -s tests -v
```

The suite contains six checks covering scientific identities, provenance, and saved contrasts. [Environment notes](locks/README.md) document the tested macOS lock, Linux CPU installation, and the separate historical production runtime.

## Run the CPU example

```sh
.venv/bin/python scripts/train.py --config configs/reviewer/tiny.json --out outputs/tiny.json
```

The example runs all four arms for two updates on four training rows and scores two held-out source rows. It checks shared initialization and input traversal, finite gradients, and distinct final arm states. `outputs/tiny.json` records the configuration, versions, input/support hashes, training losses, and source KL for each arm. It writes no checkpoint. Its risk values are an execution check and do not reproduce the full study's effect.

## Reconstruct saved contrasts

```sh
.venv/bin/python scripts/reproduce_contrast.py
```

Expected mean KL differences, `FreshY1` minus `qz`, from the frozen five-seed fixture:

| Fixture regime | Difference |
| --- | ---: |
| `source` | +0.009754286650 |
| `coef` | -0.004130631977 |
| `cov` | +0.003394102245 |

Positive values mean `FreshY1` has higher KL. These are saved point estimates, not a new training result or reconstructed confidence intervals.

The checkpoint comparator can be checked separately:

```sh
.venv/bin/python scripts/compare_checkpoints.py --self-test
```

Comparing actual states requires the original trusted checkpoint files, which are not included. The self-test does not replay training.

## Code and evidence

| Location | Contents |
| --- | --- |
| `src/pfn_dag_verify/` | Finite prior, oracle, PFN, coupled targets, CPU training/evaluation, state comparator |
| `configs/` | Runnable tiny config and documentary production recipe |
| `scripts/`, `tests/` | Execution commands, integrity verification, scientific checks |
| `examples/reviewer/fixtures/` | Frozen score and historical replay fixtures |
| `locks/`, `manifests/` | Environment records, source provenance, release checksums |
| `docs/` | [Method](docs/method.md), [reproduction scope](docs/reproduction-scope.md), [validation](docs/validation.md) |
| `legacy/` | Preserved earlier PFN-DAG studies with their own environment and evidence |

The production recipe is a historical record, not an input to the tiny runner. Earlier [PFN-DAG exports](legacy/README.md) are separate studies; their arrays and GitHub release assets do not fill the missing inputs of this supervision study.

## Paper and citation

The manuscript is under submission. A public paper link and author citation will be added when available. For now, identify the software by this repository URL and the commit used. Manuscript files and private operational records are not included in this code release.

## License and contributions

The current companion code is available under the [MIT License](LICENSE). Historical and third-party files retain their original notices; see [legacy notes](legacy/README.md). For changes, follow [CONTRIBUTING.md](CONTRIBUTING.md), including the scientific checks and provenance update procedure.
