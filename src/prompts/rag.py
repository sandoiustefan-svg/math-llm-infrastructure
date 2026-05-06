"""
RAG prompt builder.

Retrieved examples are dynamically prepended as user/assistant turns based on
semantic similarity to the test problem.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step."
)


def build_rag_messages(
    problem: str,
    retrieved: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """
    retrieved: list of dicts from Retriever.retrieve()
               each has {problem, solution, answer, source, score}
    """
    messages = [{"role": "system", "content": system_prompt}]

    for ex in retrieved:
        messages.append({"role": "user", "content": ex["problem"]})
        messages.append({
            "role":    "assistant",
            "content": f"{ex['solution']}\n\nFinal Answer: {ex['answer']}",
        })

    messages.append({"role": "user", "content": problem})
    return messages
