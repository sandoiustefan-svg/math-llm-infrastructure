#!/usr/bin/env python3
"""
Inspect the raw GSM8K and MATH datasets to see their structure and fields.

Usage:
    python scripts/python/inspect_datasets.py
    python scripts/python/inspect_datasets.py --limit 3
"""
import argparse
import json
from datasets import load_dataset


def inspect_gsm8k(limit: int) -> None:
    print("\n" + "=" * 60)
    print("GSM8K — test split")
    print("=" * 60)
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"Total problems: {len(ds)}")
    print(f"Fields: {list(ds.features.keys())}\n")

    for i, ex in enumerate(ds):
        if i >= limit:
            break
        print(f"--- Problem {i + 1} ---")
        for k, v in ex.items():
            print(f"  [{k}]\n    {v}\n")


def inspect_math(limit: int) -> None:
    print("\n" + "=" * 60)
    print("MATH — test split")
    print("=" * 60)
    ds = load_dataset("hendrycks/competition_math", split="test")
    print(f"Total problems: {len(ds)}")
    print(f"Fields: {list(ds.features.keys())}\n")

    for i, ex in enumerate(ds):
        if i >= limit:
            break
        print(f"--- Problem {i + 1} ---")
        for k, v in ex.items():
            print(f"  [{k}]\n    {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2,
                    help="Number of examples to print per dataset")
    args = ap.parse_args()

    inspect_gsm8k(args.limit)
    inspect_math(args.limit)


if __name__ == "__main__":
    main()
