"""Reward model implementations (split from legacy monolith)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch


class QualityRewardModel:
    """LLM-based Generative Reward Model.

    Evaluates thinking content and final response for:
      - Redundancy
      - Consistency between thinking and final answer
      - Self-contradictions
      - Reward hacking behaviors

    Falls back to heuristics when no judge model is available.
    Output: score in {0.0, 0.5, 1.0}
    """

    def __init__(
        self,
        judge_model: Optional[Any] = None,
        judge_tokenizer: Optional[Any] = None,
    ):
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        if self.judge_model is None:
            return self._heuristic_score(text)

        prompt = self._build_prompt(text)
        return self._judge_inference(prompt)

    def _judge_inference(self, prompt: str) -> float:
        """Run LLM judge inference and parse the score."""
        try:
            inputs = self.judge_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            device = next(self.judge_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.judge_tokenizer.eos_token_id,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = self.judge_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            return self._parse_judge_score(decoded)
        except Exception:
            return self._heuristic_score(prompt)

    def _heuristic_score(self, text: str) -> float:
        """Fallback heuristic when no LLM judge is available."""
        score = 1.0
        # Penalize extreme length (redundancy)
        if len(text) > 3000:
            score -= 0.3
        # Penalize repetition
        lines = text.split("\n")
        unique_lines = set(lines)
        if len(unique_lines) < len(lines) * 0.7:
            score -= 0.2
        return max(0.0, score)

    @staticmethod
    def _parse_judge_score(decoded: str) -> float:
        """Parse judge model output into a score in {0.0, 0.5, 1.0}."""
        decoded = decoded.strip()
        for candidate in ["1.0", "0.5", "0.0"]:
            if candidate in decoded:
                return float(candidate)
        # Fallback: look for integer tokens
        nums = re.findall(r"[01]", decoded)
        if nums:
            return float(nums[0])
        return 0.5  # neutral fallback

    def _build_prompt(self, text: str) -> str:
        return (
            "Evaluate the following response for:\n"
            "1. Redundancy\n"
            "2. Consistency between thinking and final answer\n"
            "3. Self-contradictions\n"
            "4. Reward hacking behaviors\n\n"
            f"Response:\n{text}\n\n"
            "Score: 0.0 (poor), 0.5 (fair), or 1.0 (good). Output only the score."
        )


