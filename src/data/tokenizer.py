from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into memory.

    Each line must contain one JSON object.
    """
    examples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    return examples


def save_jsonl(path: Path, examples: List[Dict[str, Any]]) -> None:
    """
    Save examples to JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def render_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, str]:
    """
    Render readable chat-template text.
    """
    messages = example["messages"]

    prompt_messages = messages[:-1]
    completion_text = messages[-1]["content"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "full_text": full_text,
        "prompt_text": prompt_text,
        "completion_text": completion_text,
    }


def _extract_input_ids(output: Any) -> List[int]:
    """
    Extract input_ids from either a plain list[int] or a tokenizer BatchEncoding/dict.
    """
    if isinstance(output, dict):
        return list(output["input_ids"])

    return list(output)


def tokenize_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]]:
    """
    Tokenize one example into input_ids, attention_mask, and loss_mask.

    loss_mask is 1 for completion tokens (assistant response) and 0 for
    prompt tokens (system + user), so only the response contributes to loss.
    """
    messages = example["messages"]

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected the last message to be the assistant response.")

    prompt_messages = messages[:-1]

    full_output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_dict=False,
    )

    prompt_output = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_length,
        return_dict=False,
    )

    input_ids = _extract_input_ids(full_output)
    prompt_ids = _extract_input_ids(prompt_output)

    prompt_len = min(len(prompt_ids), len(input_ids))

    loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "loss_mask": loss_mask,
    }


def process_dataset(
    input_path: Path,
    output_path: Path,
    model_name_or_path: str,
    mode: str,
    max_length: int,
) -> None:
    """
    Process dataset in text or tokens mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_examples = load_jsonl(input_path)

    outputs = []
    skipped = 0

    for i, example in enumerate(raw_examples, start=1):
        try:
            if mode == "text":
                out = render_example(example, tokenizer)
            else:
                out = tokenize_example(example, tokenizer, max_length)

            outputs.append(out)

        except Exception as exc:
            print(f"Skipping example {i}: {exc}")
            skipped += 1

    save_jsonl(output_path, outputs)

    print("Done.")
    print(f"Mode: {mode}")
    print(f"Written: {len(outputs)}")
    print(f"Skipped: {skipped}")