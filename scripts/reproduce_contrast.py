"""Reconstruct saved five-seed point estimates; does not score new models."""
import argparse
import json
from pathlib import Path

from pfn_dag_verify.evaluation import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1]
                        / "examples/reviewer/fixtures/withheld_seed_risks.json")
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.fixture), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
