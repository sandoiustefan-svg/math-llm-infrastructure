"""
Zero-shot prompt builder.

No examples are provided — the model must follow the <<expr=result>> annotation
format from the system prompt instruction alone.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    "For every arithmetic operation write the expression and its result as "
    "<<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. "
    'End your response with "Final Answer: {answer}".'
)


def build_zero_shot_messages(
    problem: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": problem},
    ]
