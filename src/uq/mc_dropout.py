from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.uq.metrics import answer_entropy, token_probability_confidence


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
                correct = majority.strip() == str(expected).strip()

            # Token-level confidence averaged over all N passes (full generation).
            # Using the full raw text (reasoning + answer) captures uncertainty in
            # the chain-of-thought, not just the final answer token(s).
            pass_confs = [
                token_probability_confidence(self.model, self.tokenizer, prompt, raw, self.cfg.device)
                for raw in raws
            ]
            mean_token_conf = sum(c["mean_confidence"] for c in pass_confs) / len(pass_confs)
            mean_perplexity = sum(c["perplexity"] for c in pass_confs) / len(pass_confs)
            mean_min_prob = sum(c["min_token_prob"] for c in pass_confs) / len(pass_confs)

            results.append({
                "problem": item["problem"],
                "answers": answers,
                "majority_answer": majority,
                "confidence": confidence,
                "entropy": entropy,
                "token_mean_confidence": round(mean_token_conf, 6),
                "token_perplexity": round(mean_perplexity, 4),
                "token_min_prob": round(mean_min_prob, 6),
                "correct": correct,
            })

        return results
