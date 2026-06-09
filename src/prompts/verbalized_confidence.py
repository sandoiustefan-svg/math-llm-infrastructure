"""
Verbalized-confidence prompt: the model is asked to state its own confidence
after the Final Answer line.

Used for raw (un-fine-tuned) baseline evaluation so the confidence score can
be calibration-tested against actual correctness and compared with the MC
Dropout confidence of the fine-tuned models.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    'End your response with "Final Answer: {answer}" on one line, '
    'then on the next line write "Confidence: {score}" where {score} is '
    "a number between 0.0 and 1.0 representing how confident you are that "
    "your final answer is correct."
)


def build_verbalized_confidence_messages(
    problem: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": problem},
    ]
