from __future__ import annotations

import json
import re
from pathlib import Path


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} problems → {out_path}")


def build_openmath_tail(out_path: Path, limit: int = 500) -> list[dict]:
    """
    Sample `limit` problems from OpenMathInstruct-2 train split.

    Uses streaming + a fixed shuffle so the sample is reproducible but spread
    across the full dataset. Note: the full train split was used during LoRA
    fine-tuning, so this set may overlap with training data — treat it as an
    approximate in-distribution calibration check, not a clean held-out test.
    """
    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=2026, buffer_size=50_000)
    records = []
    for ex in ds:
        if len(records) >= limit:
            break
        problem = (ex.get("problem") or "").strip()
        answer = (ex.get("expected_answer") or "").strip()
        if problem and answer:
            records.append({"problem": problem, "expected_answer": answer})

    _write_jsonl(records, out_path)
    return records


def build_gsm8k(out_path: Path, limit: int = 0) -> list[dict]:
    """GSM8K test split — grade-school math, ~1319 problems."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    records = []
    for ex in ds:
        m = re.search(r"####\s*(.+?)\s*$", ex["answer"], re.MULTILINE)
        if not m:
            continue
        problem = ex["question"].strip()
        answer = m.group(1).replace(",", "").strip()
        # full reasoning used for embedding similarity
        reference_solution = ex["answer"].strip()
        records.append({
            "problem": problem,
            "expected_answer": answer,
            "reference_solution": reference_solution,
        })
        if limit and len(records) >= limit:
            break

    _write_jsonl(records, out_path)
    return records


def build_math(out_path: Path, limit: int = 0) -> list[dict]:
    """MATH-Hard benchmark test split — competition problems levels 3-5, ~1324 items."""
    from datasets import load_dataset

    ds = load_dataset("lighteval/MATH-Hard", split="test")
    records = []
    from src.uq.metrics import _extract_boxed
    for ex in ds:
        sol = (ex.get("solution") or "").strip()
        answer = _extract_boxed(sol)
        if answer is None:
            continue
        answer = answer.strip()
        problem = (ex.get("problem") or "").strip()
        records.append({
            "problem": problem,
            "expected_answer": answer,
            "reference_solution": sol,
        })
        if limit and len(records) >= limit:
            break

    _write_jsonl(records, out_path)
    return records


def load_or_build(source: str, out_path: Path, limit: int) -> list[dict]:
    """Return cached JSONL if it exists, otherwise download and build it."""
    if out_path.exists():
        print(f"Using cached test set: {out_path}")
        with open(out_path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        return records[:limit] if limit else records

    if source == "openmath_tail":
        return build_openmath_tail(out_path, limit=limit or 500)
    elif source == "gsm8k":
        return build_gsm8k(out_path, limit=limit)
    elif source == "math":
        return build_math(out_path, limit=limit)
    else:
        raise ValueError(f"Unknown test source: {source!r}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and cache UQ test sets.")
    ap.add_argument("--source", default="all",
                    choices=["openmath_tail", "gsm8k", "math", "all"])
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write <source>.jsonl files")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max problems per set (0 = all; openmath_tail default is 500)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sources = ["openmath_tail", "gsm8k", "math"] if args.source == "all" else [args.source]
    for src in sources:
        load_or_build(src, out_dir / f"{src}.jsonl", limit=args.limit)
