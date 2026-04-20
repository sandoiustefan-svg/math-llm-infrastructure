from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class FormatConfig:
    """
    Configuration for formatting OpenMathInstruct-2 examples.

    Attributes:
        include_final_answer: Whether to append the final answer section.
        instruct_format: If True, use LLaMA 3.1 Instruct chat template
                         (<|start_header_id|> etc.). If False, use the
                         legacy ### Problem / ### Solution format.
    """
    include_final_answer: bool = True
    instruct_format: bool = True


def format_openmathinstruct2_exmaple(ex: Dict[str, Any], cfg: FormatConfig) -> Dict[str, str]:
    """
    Format a raw OpenMathInstruct-2 example into prompt and completion text.

    Returns a dictionary containing:
        - prompt_text     (loss_mask = 0, not trained on)
        - completion_text (loss_mask = 1, trained on)
        - full_text       (prompt + completion)
    """
    problem = (ex.get("problem") or "").strip()
    sol = (ex.get("generated_solution") or "").strip()
    ans = (ex.get("expected_answer") or "").strip()

    if cfg.instruct_format:
        # LLaMA 3.1 Instruct native chat template
        # <|begin_of_text|> acts as a per-example separator in packed sequences
        prompt_text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful math assistant. Solve the following problem step by step.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if cfg.include_final_answer and ans:
            completion_text = f"{sol}\n\nFinal Answer: {ans}<|eot_id|>"
        else:
            completion_text = f"{sol}<|eot_id|>"
    else:
        # Legacy format (for base models or from-scratch training)
        prompt_text = f"### Problem:\n{problem}\n\n"
        if cfg.include_final_answer and ans:
            completion_text = f"### Solution:\n{sol}\n\n### Final Answer:\n{ans}\n"
        else:
            completion_text = f"### Solution:\n{sol}\n"

    full_text = prompt_text + completion_text

    return {
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "full_text": full_text,
    }
