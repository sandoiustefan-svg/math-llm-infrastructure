"""
Build a RAG prompt by prepending retrieved examples as chat turns.

Usage
-----
    from src.rag.retriever import Retriever
    from src.rag.prompt import build_rag_messages

    retriever = Retriever("data/rag_corpus")
    retrieved = retriever.retrieve(test_problem, top_k=3)
    messages  = build_rag_messages(test_problem, retrieved)

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step."
)


def build_rag_messages(
    test_problem: str,
    retrieved: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """
    Assemble chat messages with retrieved examples as user/assistant turns
    followed by the actual test problem as the final user turn.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for ex in retrieved:
        messages.append({"role": "user", "content": ex["problem"]})
        messages.append({
            "role":    "assistant",
            "content": f"{ex['solution']}\n\nFinal Answer: {ex['answer']}",
        })

    messages.append({"role": "user", "content": test_problem})
    return messages
