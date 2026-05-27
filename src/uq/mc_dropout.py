from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import math
import statistics
from src.uq.metrics import (
    answer_entropy, answers_are_equal, normalize_math_answer,
    token_probability_confidence, compute_nlg_scores,
)
from src.prompts.zero_shot import build_zero_shot_messages



@dataclass
class MCDropoutConfig:
    model_path: str
    base_model: str
    tokenizer_name: str
    num_passes: int = 20
    max_new_tokens: int = 512
    device: str = "cuda"


class MCDropoutEvaluator:
    """
    Uncertainty quantification via Monte Carlo Dropout.

    The LoRA fine-tune was trained with lora_dropout=0.1. Keeping the model
    in train() mode at inference re-activates those same dropout layers, so
    each greedy forward pass produces a different stochastic prediction.
    Running num_passes passes and measuring answer disagreement gives an
    estimate of epistemic uncertainty over the LoRA adapter weights.
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
        marker = "Final Answer:"
        idx = text.find(marker)
        if idx != -1:
            lines = text[idx + len(marker):].strip().splitlines()
            return lines[0].strip() if lines else ""
        # fallback: last \boxed{...} in the response (depth-aware)
        from src.uq.metrics import _extract_boxed
        last, search = None, text
        while True:
            inner = _extract_boxed(search)
            if inner is None:
                break
            last = inner
            search = search[search.find(r"\boxed{") + 1:]
        return last.strip() if last else ""

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
            n_answers = len(answers)
            confidence = counts[majority] / n_answers
            entropy = answer_entropy(answers)
            consistency_rate = sum((c / n_answers) ** 2 for c in counts.values())
            n_unique_answers = len(counts)

            expected = item.get("expected_answer")
            correct: Optional[bool] = None
            if expected is not None:
                correct = answers_are_equal(majority, str(expected))

            # Token-level confidence averaged over all N passes.
            pass_confs = [
                token_probability_confidence(self.model, self.tokenizer, prompt, raw, self.cfg.device)
                for raw in raws
            ]

            def _std_log(key):
                """Std of per-pass avg log-prob across passes — epistemic variance signal."""
                vals = [math.log(c[key]) for c in pass_confs
                        if not math.isnan(c.get(key, float("nan"))) and c.get(key, 0) > 0]
                return round(statistics.stdev(vals), 6) if len(vals) > 1 else float("nan")

            # NLG scores per pass against the reference solution.
            reference = item.get("reference_solution") or ""
            nlg_per_pass = [compute_nlg_scores(raw, reference) for raw in raws]

            def _nlg_mean(metric: str) -> float:
                vals = [s[metric] for s in nlg_per_pass if not math.isnan(s[metric])]
                return round(sum(vals) / len(vals), 6) if vals else float("nan")

            results.append({
                "problem":            item["problem"],
                "expected_answer":    item.get("expected_answer"),
                "reference_solution": item.get("reference_solution"),
                "raws":               raws,
                "answers":            answers,
                "majority_answer":    majority,
                "correct":            correct,
                # Confidence scores
                "confidence":         confidence,
                "consistency_rate":   round(consistency_rate, 6),
                # Uncertainty measures
                "entropy":            entropy,
                "n_unique_answers":   n_unique_answers,
                "std_log_prob":       _std_log("full_sequence_mean_confidence"),
                "std_numeric_span_log_prob": _std_log("numeric_span_mean_confidence"),
                # NLG correctness baselines
                "mean_rougeL":        _nlg_mean("rougeL"),
                "mean_meteor":        _nlg_mean("meteor"),
            })

        return results
