"""
Few-shot Chain-of-Thought + Step-by-step prompt builder.

Same three examples as cot.py but each assistant turn uses explicit Step N:
labels, enforcing a more structured decomposition. The system prompt also
reinforces numbered steps.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Break your solution into explicit numbered steps, one per line. "
    "For every arithmetic operation write the expression and its result as "
    "<<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. "
    'End your response with "Final Answer: {answer}".'
)

COT_STEP_EXAMPLES: list[dict] = [
    {
        "problem": (
            "A store has 50 apples. They sell 23 in the morning and receive "
            "a delivery of 15 more in the afternoon. How many apples does the "
            "store have at the end of the day?"
        ),
        "solution": (
            "Step 1: The store starts with 50 apples.\n"
            "Step 2: After the morning sale: 50 - 23 = <<50-23=27>>27 apples.\n"
            "Step 3: After the afternoon delivery: 27 + 15 = <<27+15=42>>42 apples."
        ),
        "answer": "42",
    },
    {
        "problem": (
            "Sarah earns $12 per hour and works 8 hours a day, 5 days a week. "
            "How much does she earn in a week?"
        ),
        "solution": (
            "Step 1: Sarah earns $12 per hour.\n"
            "Step 2: Daily earnings: 12 × 8 = <<12*8=96>>96 per day.\n"
            "Step 3: Weekly earnings: 96 × 5 = <<96*5=480>>480 per week."
        ),
        "answer": "480",
    },
    {
        "problem": (
            "A train travels at 60 miles per hour. How far does it travel in "
            "2 hours and 30 minutes?"
        ),
        "solution": (
            "Step 1: Convert time to hours: 2 hours 30 minutes = 2.5 hours.\n"
            "Step 2: Distance = speed × time: 60 × 2.5 = <<60*2.5=150>>150 miles."
        ),
        "answer": "150",
    },
]


def build_cot_step_by_step_messages(
    problem: str,
    examples: list[dict] = COT_STEP_EXAMPLES,
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
