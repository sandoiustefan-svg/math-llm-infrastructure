from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class FormatConfig:
    include_final_answer: bool = True
    add_eos: bool = True

def format_openmathinstruct2_exmaple(ex: Dict[str, Any], cfg: FormatConfig) ->  Dict[str, str]:
    problem = (ex.get("problem") or "").strip()
    sol = (ex.get("generated_solution") or "").strip()
    ans = (ex.get("expected_answer") or "").strip()

    prompt_text = f"### Problem:\n{problem}\n\n"

    if cfg.include_final_answer and ans:
        completion_text = f"### Solution:\n{sol}\n\n### Final Answer:\n{ans}\n"
    else:
        completion_text = f"### Solution:\n{sol}\n"

    full_text = prompt_text + completion_text

    return {
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "full_text": full_text
    }