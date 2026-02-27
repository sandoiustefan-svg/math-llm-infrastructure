from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import LlamaConfig, LlamaForCausalLM


@dataclass(frozen=True)
class LlamaModelConfig:
    """
    Configuration container for building a LLaMA-style causal language model
    from scratch.

    This configuration defines the architectural hyperparameters of the model.
    It does NOT load pretrained weights. The resulting model will be randomly
    initialized and trained from scratch.

    Attributes:
        vocab_size (int):
            Size of the tokenizer vocabulary. Must match the tokenizer used
            during preprocessing.

        max_position_embeddings (int):
            Maximum sequence length supported by the model. Should be
            >= the packed `seq_len` used in preprocessing.

        num_hidden_layers (int):
            Number of transformer decoder layers.

        hidden_size (int):
            Dimensionality of token embeddings and hidden states.

        num_attention_heads (int):
            Number of attention heads in each transformer layer.
            Must divide `hidden_size`.

        intermediate_size (Optional[int]):
            Size of the feed-forward (MLP) hidden layer.
            If None, defaults to 4 × hidden_size.

        rms_norm_eps (float):
            Epsilon value used in RMSNorm layers.

        rope_theta (float):
            Base frequency for Rotary Positional Embeddings (RoPE).

        tie_word_embeddings (bool):
            Whether to tie input embeddings and output projection weights.

        use_cache (bool):
            Whether to enable KV caching. Typically False during training.
    """
    vocab_size: int
    max_position_embeddings: int

    num_hidden_layers: int = 8
    hidden_size: int = 512
    num_attention_heads: int = 8
    intermediate_size: Optional[int] = None

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    use_cache: bool = False


def build_llama(cfg: LlamaModelConfig) -> LlamaForCausalLM:
    """
    Build a randomly initialized LLaMA-style causal language model.

    This function validates architectural constraints and constructs a
    `LlamaForCausalLM` instance using HuggingFace Transformers.

    Args:
        cfg (LlamaModelConfig):
            Model architecture configuration.

    Returns:
        LlamaForCausalLM:
            A randomly initialized causal language model.

    Raises:
        ValueError:
            If:
                - vocab_size <= 0
                - max_position_embeddings <= 0
                - hidden_size is not divisible by num_attention_heads
    """
    if cfg.vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {cfg.vocab_size}")

    if cfg.max_position_embeddings <= 0:
        raise ValueError(
            f"max_position_embeddings must be positive, got {cfg.max_position_embeddings}"
        )

    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) must be divisible by "
            f"num_attention_heads ({cfg.num_attention_heads})"
        )

    intermediate_size = cfg.intermediate_size or (4 * cfg.hidden_size)

    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        use_cache=cfg.use_cache,
    )

    return LlamaForCausalLM(llama_cfg)