from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass(frozen=True)
class FormatConfig:
    """
    Configuration for formatting OpenMathInstruct-2 examples.

    Attributes:
        include_system_prompt: Whether to prepend a system message.
        include_final_answer: Whether to append final answer separately.
        system_prompt: Content of system prompt.
    """
    include_system_prompt: bool = True
    include_final_answer: bool = True
    system_prompt: str = (
        "You are a careful mathematical reasoning assistant. "
        "Solve the problem step by step."
    )


def format_openmathinstruct2_example(
        ex: Dict[str, Any],
        cfg: FormatConfig
) -> Dict[str, Any]:
    """
    Convert a raw OpenMathInstruct-2 example into Llama chat-style messages.

    Returns:
        {
            "messages": [...],
            "prompt_messages": [...],
            "completion_text": ...
        }

    prompt_messages:
        messages before assistant answer (useful for inference)

    completion_text:
        assistant target text (useful for masking/loss)
    """
    problem = (ex.get("problem") or "").strip()
    solution = (ex.get("generated_solution") or "").strip()
    answer = (ex.get("expected_answer") or "").strip()

    messages: List[Dict[str, str]] = []

    if cfg.include_system_prompt:
        messages.append({
            "role": "system",
            "content": cfg.system_prompt
        })

    messages.append({
        "role": "user",
        "content": problem
    })

    if cfg.include_final_answer and answer:
        assistant_text = f"{solution}\n\nFinal Answer: {answer}"
    else:
        assistant_text = solution

    messages.append({
        "role": "assistant",
        "content": assistant_text
    })

    prompt_messages = messages[:-1]

    return {
        "messages": messages,
        "prompt_messages": prompt_messages,
        "completion_text": assistant_text,
    }

