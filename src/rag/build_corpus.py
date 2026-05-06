"""
Build the RAG retrieval corpus from SVAMP, ASDiv, and MAWPS.

Usage
-----
    # full run
    python -m src.rag.build_corpus --out-dir data/rag_corpus

    # debug: 5 examples per dataset, prints samples to stdout
    python -m src.rag.build_corpus --out-dir data/rag_corpus --debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

from datasets import load_dataset


def _load_svamp(limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """ChilleD/SVAMP  ~1 000 examples.  Columns: Body, Question, Equation, Answer"""
    ds = load_dataset("ChilleD/SVAMP", split="train", trust_remote_code=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        body     = (ex.get("Body") or "").strip()
        question = (ex.get("Question") or "").strip()
        equation = (ex.get("Equation") or "").strip()
        answer   = str(ex.get("Answer") or "").strip()
        yield {
            "problem":  f"{body} {question}".strip(),
            "solution": (
                f"Let me solve this step by step.\n"
                f"Equation: {equation}\n"
                f"Evaluating: {equation} = {answer}"
            ),
            "answer": answer,
            "source": "svamp",
        }


def _load_asdiv(limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """EleutherAI/asdiv  ~2 300 examples.  Columns: body, question, formula, answer"""
    # ASDiv ships only a 'validation' split
    ds = load_dataset("EleutherAI/asdiv", split="validation", trust_remote_code=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        body     = (ex.get("body") or "").strip()
        question = (ex.get("question") or "").strip()
        formula  = (ex.get("formula") or "").strip()
        answer   = str(ex.get("answer") or "").strip()
        yield {
            "problem":  f"{body} {question}".strip(),
            "solution": (
                f"Let me solve this step by step.\n"
                f"Formula: {formula}\n"
                f"Result: {answer}"
            ),
            "answer": answer,
            "source": "asdiv",
        }


def _load_mawps(limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """MU-NLPC/Calc-mawps  ~2 600 examples.  Columns: question, equation, result"""
    ds = load_dataset("MU-NLPC/Calc-mawps", split="train", trust_remote_code=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        question = (ex.get("question") or "").strip()
        equation = (ex.get("equation") or ex.get("expression") or "").strip()
        answer   = str(ex.get("result") or ex.get("result_float") or "").strip()
        yield {
            "problem":  question,
            "solution": (
                f"Let me solve this step by step.\n"
                f"Equation: {equation}\n"
                f"Evaluating: {equation} = {answer}"
            ),
            "answer": answer,
            "source": "mawps",
        }

def _format_demonstration(ex: Dict[str, Any]) -> str:                                                                                                                                                                                                      
    return (                                                                                                                                                                                                                                               
        f"{ex['problem']}\n"
        f"{ex['solution']}\n\n"                                                                                                                                                                                                                            
        f"Final Answer: {ex['answer']}"                                                                                                                                                                                                                  
    )    


def fetch_all(limit: int | None = None, debug: bool = False) -> List[Dict[str, Any]]:
    loaders = [
        ("svamp", _load_svamp),
        ("asdiv", _load_asdiv),
        ("mawps", _load_mawps),
    ]

    all_examples: List[Dict[str, Any]] = []

    for name, loader in loaders:
        print(f"Loading {name} ...", flush=True)
        before = len(all_examples)
        try:
            for ex in loader(limit=limit):
                ex["text"] = _format_demonstration(ex)
                all_examples.append(ex)
        except Exception as exc:
            print(f"  WARNING: failed to load {name}: {exc}", file=sys.stderr)
            continue
        count = len(all_examples) - before
        print(f"  {name}: {count} examples loaded")

        if debug and count > 0:
            sample = all_examples[before]
            print(f"\n  [DEBUG] First {name} example:")
            print(json.dumps(sample, indent=4, ensure_ascii=False))
            print()

    print(f"Total examples: {len(all_examples)}")
    return all_examples


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fetch SVAMP, ASDiv, MAWPS and save to raw_corpus.json"
    )
    ap.add_argument(
        "--out-dir", default="data/rag_corpus",
        help="Output directory (default: data/rag_corpus)",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Limit to 5 examples per dataset and print a sample from each to stdout",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = 5 if args.debug else None
    examples = fetch_all(limit=limit, debug=args.debug)

    out_path = out_dir / "raw_corpus.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(examples)} examples → {out_path}")


if __name__ == "__main__":
    main()
