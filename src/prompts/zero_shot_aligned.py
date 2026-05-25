"""
Aligned zero-shot prompt builder.

System prompt matches the format used during LoRA fine-tuning on
OpenMathInstruct-2 (Strategy 3): no annotation format instruction,
no few-shot examples, single user/assistant turn.

The only addition over the raw training system prompt is an explicit
"Final Answer: {answer}" reminder — the model already learned this
termination pattern during fine-tuning, so this reinforces rather
than shifts the distribution.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    'End your response with "Final Answer: {answer}".'
)


def build_zero_shot_aligned_messages(
    problem: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": problem},
    ]
