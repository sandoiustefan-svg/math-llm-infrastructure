from __future__ import annotations

from typing import Tuple

import torch


def build_doc_mask_and_positions(
    input_ids: torch.Tensor, bos_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build block-diagonal causal attention mask and per-document position ids.

    The formatter emits `<|begin_of_text|>` (BOS) at the start of every example
    and the tokenizer runs with `add_special_tokens=False`, so BOS tokens in a
    packed sequence mark document boundaries. Tokens are allowed to attend only
    to earlier tokens within the same document; position ids reset to 0 at each
    BOS so RoPE sees each document as starting from position 0.

    Args:
        input_ids: (B, T) long tensor of token ids.
        bos_id: token id of <|begin_of_text|> (128000 for LLaMA 3.1/3.2).

    Returns:
        attention_mask: (B, 1, T, T) additive float mask, 0.0 where attention
            is allowed and -inf where it is blocked. Dtype is float32; cast to
            the model dtype at the call site if needed.
        position_ids: (B, T) long tensor, reset to 0 at each BOS.
    """
    B, T = input_ids.shape
    device = input_ids.device

    is_bos = input_ids == bos_id
    doc_id = torch.cumsum(is_bos.long(), dim=1)

    arange = torch.arange(T, device=device).expand(B, T)
    bos_positions = torch.where(is_bos, arange, torch.full_like(arange, -1))
    start_idx, _ = bos_positions.cummax(dim=1)
    start_idx = start_idx.clamp(min=0)
    position_ids = (arange - start_idx).long()

    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)
    causal = arange.unsqueeze(1) >= arange.unsqueeze(2)
    allow = same_doc & causal

    neg_inf = torch.finfo(torch.float32).min
    attention_mask = torch.where(
        allow,
        torch.tensor(0.0, device=device),
        torch.tensor(neg_inf, device=device),
    ).unsqueeze(1)

    return attention_mask, position_ids
