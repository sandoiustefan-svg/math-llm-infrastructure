from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import math
from src.uq.metrics import answer_entropy, answers_are_equal, token_probability_confidence


@dataclass
class MCDropoutConfig:
    model_path: str
    tokenizer_name: str
    num_passes: int = 20
    max_new_tokens: int = 512
    temperature: float = 1.0
    device: str = "cuda"


class MCDropoutEvaluator:
    """
    Uncertainty quantification via Monte Carlo Dropout.

    The scratch-trained model was trained with attention_dropout=0.1 and
    hidden_dropout=0.1. Calling model.train() at inference keeps those dropout
    masks active, so each forward pass produces a different stochastic prediction.
    Running num_passes passes and measuring answer disagreement gives an estimate
    of the model's epistemic uncertainty.
    """

    def __init__(self, cfg: MCDropoutConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
        self.model.to(cfg.device)
        # Keep dropout active — this is the key difference from standard inference.
        self.model.train()

    @torch.no_grad()
    def _generate_once(self, input_ids: torch.Tensor) -> str:
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=(self.cfg.temperature > 0),
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.3,
        )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        marker = "### Final Answer:"
        idx = text.find(marker)
        if idx == -1:
            return text.strip()
        return text[idx + len(marker):].strip().splitlines()[0].strip()

    def evaluate(self, problems: list[dict]) -> list[dict]:
        """
        Args:
            problems: list of dicts with keys:
                - "problem" (str, required)
                - "expected_answer" (str, optional — for correctness scoring)

        Returns:
            list of dicts, one per problem:
                - "problem": str
                - "answers": list of N extracted answers (one per pass)
                - "majority_answer": str
                - "confidence": float  (fraction of passes matching majority)
                - "entropy": float     (predictive entropy over answer distribution)
                - "correct": bool | None
        """
        results = []
        for item in problems:
            prompt = f"### Problem:\n{item['problem'].strip()}\n\n### Solution:\n"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.cfg.device)

            answers = []
            raws = []
            for _ in range(self.cfg.num_passes):
                raw = self._generate_once(input_ids)
                raws.append(raw)
                answers.append(self._extract_final_answer(raw))

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
            # Computes all four confidence measures per pass, then averages.
            pass_confs = [
                token_probability_confidence(self.model, self.tokenizer, prompt, raw, self.cfg.device)
                for raw in raws
            ]

            def _avg(key):
                vals = [c[key] for c in pass_confs if not math.isnan(c.get(key, float("nan")))]
                return round(sum(vals) / len(vals), 6) if vals else float("nan")

            results.append({
                "problem": item["problem"],
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
                # Answer span only (tokens after ### Final Answer:)
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
                # Legacy key kept for backwards compatibility
                "token_mean_confidence":         _avg("full_sequence_mean_confidence"),
                "token_perplexity":              _avg("full_sequence_perplexity"),
                "token_min_prob":                _avg("full_sequence_min_token_prob"),
            })

        return results
