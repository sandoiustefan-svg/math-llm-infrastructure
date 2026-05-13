#!/usr/bin/env python3
"""
Show and save rendered inference prompts for all three prompt variants.

Usage:
    # Print example with built-in Janet's ducks problem
    python scripts/python/show_prompts.py

    # Read 5 problems from GSM8K, save to data/prompts/gsm8k/
    python scripts/python/show_prompts.py --source gsm8k --n 5

    # Read 10 problems from MATH
    python scripts/python/show_prompts.py --source math --n 10

    # Use the real Llama tokenizer (requires model access)
    python scripts/python/show_prompts.py --source gsm8k --n 5 \
        --tokenizer meta-llama/Llama-3.2-1B-Instruct

Output is saved to data/prompts/<source>/<variant>.jsonl
Each line: {problem, expected_answer, rendered_prompt, messages}
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import PROMPT_BUILDERS

_DEFAULT_PROBLEM = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. She sells "
    "the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?"
)
_DEFAULT_ANSWER = "18"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_manual(messages: list[dict]) -> str:
    """Manually apply the Llama 3 chat template without loading a tokenizer."""
    out = "<|begin_of_text|>"
    for msg in messages:
        out += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        out += msg["content"]
        out += "<|eot_id|>"
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out


def _make_render_fn(tokenizer_name: str | None):
    if tokenizer_name is None:
        return _render_manual
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {tokenizer_name} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda msgs: tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_section(title: str, content: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(content)


def _format_messages_preview(messages: list[dict]) -> str:
    lines = ["["]
    for i, m in enumerate(messages):
        comma = "," if i < len(messages) - 1 else ""
        preview = repr(m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"])
        lines.append(f'    {{"role": {repr(m["role"])}, "content": {preview}}}{comma}')
    lines.append("]")
    return "\n".join(lines)


def _display_problem(problem: str, answer: str, render_fn) -> None:
    print(f"\nProblem:\n{textwrap.fill(problem, width=72)}")
    print(f"Expected answer: {answer}")
    for name, builder in PROMPT_BUILDERS.items():
        messages = builder(problem)
        _print_section(f"Variant: {name}  —  messages list", _format_messages_preview(messages))
        _print_section(f"Variant: {name}  —  rendered prompt (sent to model)", render_fn(messages))
    print("\n" + "=" * 80)
    print("  Model generates from the trailing assistant header onwards.")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_problems(problems: list[dict], source: str, render_fn) -> None:
    out_base = PROJECT_ROOT / "data" / "prompts" / source
    out_base.mkdir(parents=True, exist_ok=True)

    for variant_name, builder in PROMPT_BUILDERS.items():
        out_path = out_base / f"{variant_name}.jsonl"
        records = []
        for item in problems:
            messages = builder(item["problem"])
            records.append({
                "problem":         item["problem"],
                "expected_answer": item.get("expected_answer", ""),
                "messages":        messages,
                "rendered_prompt": render_fn(messages),
            })
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records):>4} problems → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Render and save inference prompts for all variants.")
    ap.add_argument("--source", default=None, choices=["gsm8k", "math"],
                    help="Test set to sample from. If omitted, uses the built-in example.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of problems to sample (default: 5)")
    ap.add_argument("--test-sets-dir", default="",
                    help="Directory for cached test-set JSONLs. Defaults to data/test_sets/")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name for exact rendering. "
                         "If omitted, uses a manual Llama-3 template approximation.")
    ap.add_argument("--print", dest="do_print", action="store_true", default=False,
                    help="Print rendered prompts to stdout (always on when --source is omitted)")
    args = ap.parse_args()

    render_fn = _make_render_fn(args.tokenizer)

    if args.source is None:
        # Built-in example — just display, nothing to save
        _display_problem(_DEFAULT_PROBLEM, _DEFAULT_ANSWER, render_fn)
        return

    # Load problems from test set
    test_sets_dir = (
        Path(args.test_sets_dir) if args.test_sets_dir
        else PROJECT_ROOT / "data" / "test_sets"
    )
    test_set_path = test_sets_dir / f"{args.source}.jsonl"

    from src.uq.test_sets import load_or_build
    print(f"Loading {args.source} test set ...")
    all_problems = load_or_build(args.source, test_set_path, limit=0)
    problems = all_problems[: args.n]
    print(f"Sampled {len(problems)} problems from {args.source}")

    # Optionally print first problem to stdout
    if args.do_print and problems:
        _display_problem(
            problems[0]["problem"],
            problems[0].get("expected_answer", "?"),
            render_fn,
        )

    # Save all variants to data/prompts/<source>/
    _save_problems(problems, args.source, render_fn)


if __name__ == "__main__":
    main()
