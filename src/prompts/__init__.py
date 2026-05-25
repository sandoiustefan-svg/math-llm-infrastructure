from .zero_shot import build_zero_shot_messages
from .zero_shot_aligned import build_zero_shot_aligned_messages
from .cot import build_cot_messages
from .cot_step_by_step import build_cot_step_by_step_messages

PROMPT_BUILDERS: dict = {
    "zero_shot":         build_zero_shot_messages,
    "zero_shot_aligned": build_zero_shot_aligned_messages,
    "cot":               build_cot_messages,
    "cot_step_by_step":  build_cot_step_by_step_messages,
}
