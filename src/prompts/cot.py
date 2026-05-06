"""
Few-shot Chain-of-Thought prompt builder.

Fixed hand-written examples are prepended as user/assistant turns before the
test problem. The same examples are used for every query.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step."
)

# Fixed few-shot examples — GSM8K-style, matching the training format:
# assistant turn = step-by-step reasoning + "\n\nFinal Answer: {answer}"
COT_EXAMPLES: list[dict] = [
    {
        "problem": (
            "A store has 50 apples. They sell 23 in the morning and receive "
            "a delivery of 15 more in the afternoon. How many apples does the "
            "store have at the end of the day?"
        ),
        "solution": (
            "Let me solve this step by step.\n"
            "The store starts with 50 apples.\n"
            "After selling 23 in the morning: 50 - 23 = 27 apples.\n"
            "After receiving a delivery of 15: 27 + 15 = 42 apples."
        ),
        "answer": "42",
    },
    {
        "problem": (
            "Sarah earns $12 per hour and works 8 hours a day, 5 days a week. "
            "How much does she earn in a week?"
        ),
        "solution": (
            "Let me solve this step by step.\n"
            "Sarah earns $12 per hour.\n"
            "She works 8 hours per day: 12 × 8 = $96 per day.\n"
            "She works 5 days a week: 96 × 5 = $480 per week."
        ),
        "answer": "480",
    },
    {
        "problem": (
            "A train travels at 60 miles per hour. How far does it travel in "
            "2 hours and 30 minutes?"
        ),
        "solution": (
            "Let me solve this step by step.\n"
            "Convert time to hours: 2 hours 30 minutes = 2.5 hours.\n"
            "Distance = speed × time: 60 × 2.5 = 150 miles."
        ),
        "answer": "150",
    },
]


def build_cot_messages(
    problem: str,
    examples: list[dict] = COT_EXAMPLES,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    for ex in examples:
        messages.append({"role": "user", "content": ex["problem"]})
        messages.append({
            "role":    "assistant",
            "content": f"{ex['solution']}\n\nFinal Answer: {ex['answer']}",
        })

    messages.append({"role": "user", "content": problem})
    return messages
