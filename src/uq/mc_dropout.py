from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import math
from src.uq.metrics import answer_entropy, answers_are_equal, normalize_math_answer, token_probability_confidence
from src.prompts.zero_shot import build_zero_shot_messages



@dataclass
class MCDropoutConfig:
    model_path: str
    base_model: str
    tokenizer_name: str
    num_passes: int = 20
    max_new_tokens: int = 512
    mc_dropout_rate: float = 0.1
    device: str = "cuda"


class MCDropoutEvaluator:
    """
    Uncertainty quantification via Monte Carlo Dropout.

    The LoRA fine-tune was trained with lora_dropout=0.1 plus an extra
    nn.Dropout after the final RMSNorm (_add_mc_dropout_hook in trainer.py).
    Both are re-activated by keeping the model in train() mode at inference,
    so each greedy forward pass produces a different stochastic prediction.
    Running num_passes passes and measuring answer disagreement gives an
    estimate of the model's epistemic uncertainty.
    """

    def __init__(self, cfg: MCDropoutConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # this loads the 1B/8B Llama model (base model)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
        if (Path(cfg.model_path) / "adapter_config.json").exists():
            # this loads the Lora metrices on top of the base model
            self.model = PeftModel.from_pretrained(base, cfg.model_path)
        else:
            self.model = base
        self.model.to(cfg.device)

        # Remove temperature/top_p from generation_config — irrelevant with do_sample=False
        # but transformers warns about them on every generate() call otherwise.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

        # Keep dropout active — this is the key difference from standard inference.
        self.model.train()

        self._eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    @torch.no_grad()
    def _generate_once(self, input_ids: torch.Tensor) -> str:
        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self._eot_id],
            repetition_penalty=1.3,
        )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        import re
        marker = "Final Answer:"
        idx = text.find(marker)
        if idx != -1:
            lines = text[idx + len(marker):].strip().splitlines()
            return lines[0].strip() if lines else ""
        # fallback: last \boxed{...} in the response
        boxes = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxes:
            return boxes[-1].strip()
        return ""

    def evaluate(self, problems: list[dict], prompt_fn=None) -> list[dict]:
        """
        Args:
            problems:  list of dicts with keys:
                - "problem" (str, required)
                - "expected_answer" (str, optional — for correctness scoring)
            prompt_fn: callable(problem: str) -> list[dict]  (chat messages)
                       Defaults to zero-shot if not provided.

        Returns:
            list of dicts, one per problem.
        """
        if prompt_fn is None:
            prompt_fn = build_zero_shot_messages

        results = []
        for item in problems:
            messages = prompt_fn(item["problem"])
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # add_special_tokens=False: apply_chat_template already includes <|begin_of_text|>
            input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.cfg.device)

            answers = []
            raws = []
            for _ in range(self.cfg.num_passes):
                raw = self._generate_once(input_ids)
                raws.append(raw)
                # Normalize before counting so "2" and "\boxed{2}" vote together.
                answers.append(normalize_math_answer(self._extract_final_answer(raw)))

            counts: dict[str, int] = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            majority = max(counts, key=counts.__getitem__)
            confidence = counts[majority] / len(answers)
            entropy = answer_entropy(answers)

            expected = item.get("expected_answer")
            correct: Optional[bool] = None
            if expected is not None:
                correct = answers_are_equal(majority, str(expected))

            # Token-level confidence averaged over all N passes.
            pass_confs = [
                token_probability_confidence(self.model, self.tokenizer, prompt, raw, self.cfg.device)
                for raw in raws
            ]

            def _avg(key):
                vals = [c[key] for c in pass_confs if not math.isnan(c.get(key, float("nan")))]
                return round(sum(vals) / len(vals), 6) if vals else float("nan")

            results.append({
                "problem": item["problem"],
                "raws": raws,
                "answers": answers,
                "majority_answer": majority,
                "confidence": confidence,
                "entropy": entropy,
                "correct": correct,
                # Full sequence (baseline — inflated by glue words)
                "full_sequence_mean_confidence": _avg("full_sequence_mean_confidence"),
                "full_sequence_perplexity":      _avg("full_sequence_perplexity"),
                "full_sequence_min_token_prob":  _avg("full_sequence_min_token_prob"),
                "full_sequence_std_token_prob":  _avg("full_sequence_std_token_prob"),
                # Answer span only (tokens after Final Answer:)
                "answer_span_mean_confidence":   _avg("answer_span_mean_confidence"),
                "answer_span_perplexity":        _avg("answer_span_perplexity"),
                "answer_span_min_token_prob":    _avg("answer_span_min_token_prob"),
                "answer_span_std_token_prob":    _avg("answer_span_std_token_prob"),
                # Numeric tokens — full sequence (all digit appearances)
                "numeric_mean_confidence":       _avg("numeric_mean_confidence"),
                "numeric_perplexity":            _avg("numeric_perplexity"),
                "numeric_min_token_prob":        _avg("numeric_min_token_prob"),
                "numeric_std_token_prob":        _avg("numeric_std_token_prob"),
                # Numeric tokens — answer span only
                "numeric_span_mean_confidence":  _avg("numeric_span_mean_confidence"),
                "numeric_span_perplexity":       _avg("numeric_span_perplexity"),
                "numeric_span_min_token_prob":   _avg("numeric_span_min_token_prob"),
                "numeric_span_std_token_prob":   _avg("numeric_span_std_token_prob"),
                # Position-weighted token probability (Metric 6)
                "weighted_mean_confidence":      _avg("weighted_mean_confidence"),
                "weighted_perplexity":           _avg("weighted_perplexity"),
            })

        return results
