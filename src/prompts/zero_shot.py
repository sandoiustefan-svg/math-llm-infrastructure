"""
Zero-shot prompt builder.

No examples are provided — the model sees only the system prompt and the problem.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step."
)


def build_zero_shot_messages(
    problem: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": problem},
    ]
