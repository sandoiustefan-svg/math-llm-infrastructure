"""
LLM-as-judge post-processing for results.json.

Reads results.json produced by run_uq_eval.py, calls an LLM judge to assign
good / medium / bad correctness labels, writes results_judged.json, and produces
judge reliability diagrams and ROC curves in a judge_correctness/ subdirectory.

Labels:
  good   — correct final answer, sound reasoning
  medium — wrong final answer but approach/reasoning mostly correct
  bad    — wrong answer, flawed reasoning

Usage:
    python -m src.uq.llm_judge \\
        --results results/mc_dropout/seed42/gsm8k/cot/results.json \\
        --model gpt-4o-mini

    # Anthropic judge
    python -m src.uq.llm_judge \\
        --results results/mc_dropout/seed42/gsm8k/cot/results.json \\
        --provider anthropic --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root if present (OPENAI_API_KEY / ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed — fall back to shell environment

_VALID_LABELS = {"good", "medium", "bad"}

_JUDGE_SYSTEM = (
    "You are a math answer evaluator. "
    "Given a problem and a model's full response, classify it as exactly one of:\n"
    "  good   — correct final answer and sound reasoning\n"
    "  medium — wrong final answer but the approach and intermediate steps are "
    "mostly correct (error only in the last computation or answer extraction)\n"
    "  bad    — wrong answer with flawed or irrelevant reasoning\n"
    "Respond with exactly one word: good, medium, or bad."
)

_JUDGE_USER = "Problem:\n{problem}\n\nModel response:\n{response}"


# ---------------------------------------------------------------------------
# Async judge calls
# ---------------------------------------------------------------------------

async def _call_openai(client, model: str, problem: str, response: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            result = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": _JUDGE_USER.format(problem=problem, response=response)},
                ],
                max_tokens=5,
                temperature=0,
            )
            label = result.choices[0].message.content.strip().lower()
            return label if label in _VALID_LABELS else "bad"
        except Exception as e:
            print(f"  [judge error] {e}")
            return "bad"


async def _call_anthropic(client, model: str, problem: str, response: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            result = await client.messages.create(
                model=model,
                max_tokens=5,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": _JUDGE_USER.format(problem=problem, response=response)}],
            )
            label = result.content[0].text.strip().lower()
            return label if label in _VALID_LABELS else "bad"
        except Exception as e:
            print(f"  [judge error] {e}")
            return "bad"


async def _run_all(
    results: list[dict],
    provider: str,
    model: str,
    max_concurrent: int,
) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent)

    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError:
            sys.exit("openai package not installed. Run: pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set. Add it to .env or export it in your shell.")
        client = AsyncOpenAI()
        call_fn = _call_openai
    else:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            sys.exit("anthropic package not installed. Run: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell.")
        client = AsyncAnthropic()
        call_fn = _call_anthropic

    tasks = [
        call_fn(client, model, r.get("problem", ""), (r.get("raws") or [""])[0], sem)
        for r in results
    ]
    # gather preserves input order
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------

def enrich(results: list[dict], provider: str, model: str, max_concurrent: int) -> list[dict]:
    """Run judge on every result and add judge_rank + judge_score in-place."""
    print(f"Running LLM judge ({provider}/{model}) on {len(results)} problems ...")
    labels = asyncio.run(_run_all(results, provider, model, max_concurrent))
    from src.uq.metrics import _JUDGE_SCORE
    for r, label in zip(results, labels):
        r["judge_rank"]  = label
        r["judge_score"] = _JUDGE_SCORE[label]
    counts = {k: sum(1 for r in results if r["judge_rank"] == k) for k in ("good", "medium", "bad")}
    print(f"  good={counts['good']}  medium={counts['medium']}  bad={counts['bad']}")
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_CONF_KEYS = [
    ("confidence",       "Majority Vote Confidence"),
    ("consistency_rate", "Consistency Rate"),
]


def _make_plots(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import (
        plot_reliability_diagram_judge,
        plot_roc_curve,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, title in _CONF_KEYS:
        p = str(out_dir / f"reliability_judge_{key}.png")
        plot_reliability_diagram_judge(results, p, confidence_key=key, title=title)
        print(f"Reliability-judge → {p}")

    p = str(out_dir / "roc_judge.png")
    plot_roc_curve(
        results, p,
        label_fn=lambda r: (True  if r.get("judge_rank") == "good"
                            else False if r.get("judge_rank") in ("medium", "bad")
                            else None),
        title="ROC Curve — LLM Judge (good vs medium+bad)",
    )
    print(f"ROC (judge)       → {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge post-processing for results.json.")
    ap.add_argument("--results", required=True,
                    help="Path to results.json produced by run_uq_eval.py")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                    help="API provider (default: openai)")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="Judge model (default: gpt-4o-mini)")
    ap.add_argument("--max-concurrent", type=int, default=20,
                    help="Max simultaneous API requests (default: 20)")
    ap.add_argument("--output", default=None,
                    help="Output directory for all judge outputs. "
                         "Defaults to <results_dir>/judge_correctness/")
    ap.add_argument("--no-plots", action="store_true", default=False,
                    help="Skip plot generation")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"results.json not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {results_path}")

    results = enrich(results, args.provider, args.model, args.max_concurrent)

    judge_dir = Path(args.output) if args.output else results_path.parent / "judge_correctness"
    judge_dir.mkdir(parents=True, exist_ok=True)

    out_path = judge_dir / "results_judged.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved  → {out_path}")

    from src.uq.metrics import summarise
    summary = summarise(results)
    summary_path = judge_dir / "summary_judged.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    if not args.no_plots:
        _make_plots(results, judge_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
