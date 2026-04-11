from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import math
from src.uq.metrics import answer_entropy, answers_are_equal, token_probability_confidence


@dataclass
class EnsembleConfig:
    model_paths: list[str]   # one path per ensemble member (different seeds)
    tokenizer_name: str
    max_new_tokens: int = 512
    device: str = "cuda"


class EnsembleEvaluator:
    """
    Uncertainty quantification via Deep Ensembles.

    Each ensemble member is a fine-tuned model trained with a different random
    seed. Models are loaded one at a time to avoid multiplying GPU memory by K.
    Uncertainty comes from disagreement across members rather than stochastic
    dropout, making it complementary to MC Dropout.
    """

    def __init__(self, cfg: EnsembleConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def _generate_with_model(self, model: AutoModelForCausalLM, input_ids: torch.Tensor) -> str:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,  # greedy — each member is deterministic
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
                - "answers": list of K answers (one per ensemble member)
                - "majority_answer": str
                - "confidence": float  (fraction of members agreeing with majority)
                - "entropy": float     (predictive entropy over answer distribution)
                - "correct": bool | None
        """
        # Collect one answer + token confidence per model per problem.
        # Load each model once and run all problems through it before the next.
        per_model_answers: list[list[str]] = []
        per_model_raws: list[list[str]] = []
        per_model_token_confs: list[list[dict]] = []

        for path in self.cfg.model_paths:
            model = AutoModelForCausalLM.from_pretrained(path)
            model.to(self.cfg.device)
            model.eval()

            model_answers = []
            model_raws = []
            model_token_confs = []
            for item in problems:
                prompt = f"### Problem:\n{item['problem'].strip()}\n\n### Solution:\n"
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.cfg.device)
                raw = self._generate_with_model(model, input_ids)
                answer = self._extract_final_answer(raw)
                model_answers.append(answer)
                model_raws.append(raw)
                # Token-level confidence over the full generation (reasoning + answer),
                # not just the extracted final answer string.
                model_token_confs.append(
                    token_probability_confidence(model, self.tokenizer, prompt, raw, self.cfg.device)
                )

            per_model_answers.append(model_answers)
            per_model_raws.append(model_raws)
            per_model_token_confs.append(model_token_confs)

            # Free GPU memory before loading the next model
            del model
            torch.cuda.empty_cache()

        # Transpose: per_model_answers[model][problem] → answers_per_problem[problem][model]
        results = []
        for i, item in enumerate(problems):
            answers = [per_model_answers[m][i] for m in range(len(self.cfg.model_paths))]

            counts: dict[str, int] = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            majority = max(counts, key=counts.__getitem__)
            confidence = counts[majority] / len(answers)
            entropy = answer_entropy(answers)

            member_confs = [per_model_token_confs[m][i] for m in range(len(self.cfg.model_paths))]

            def _avg(key):
                vals = [c[key] for c in member_confs if not math.isnan(c.get(key, float("nan")))]
                return round(sum(vals) / len(vals), 6) if vals else float("nan")

            expected = item.get("expected_answer")
            correct: Optional[bool] = None
            if expected is not None:
                correct = answers_are_equal(majority, str(expected))

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
