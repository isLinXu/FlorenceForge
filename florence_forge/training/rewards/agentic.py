"""Agentic meta-cognitive reward models."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import torch

from .accuracy import MixedAccuracyRewardModel
from .quality import QualityRewardModel


class AgenticFormatRewardModel:
    """Reward model for agentic meta-cognitive token structure."""

    def __init__(self):
        from ...core.agentic_tokens import (
            AGENTIC_PHASE_TOKENS,
            AGENTIC_PHASE_ORDER,
            extract_all_phases,
            get_phase_order,
            has_required_phases,
        )
        self._phase_tokens = AGENTIC_PHASE_TOKENS
        self._phase_order = AGENTIC_PHASE_ORDER
        self._has_required_phases = has_required_phases
        self._extract_all_phases = extract_all_phases
        self._get_phase_order = get_phase_order

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        score = 1.0
        if not self._has_required_phases(text):
            score -= 0.5
        for _phase, (open_tok, close_tok) in self._phase_tokens.items():
            open_count = text.count(open_tok)
            close_count = text.count(close_tok)
            if open_count != close_count:
                score -= 0.15 * abs(open_count - close_count)
        phase_seq = self._get_phase_order(text)
        canonical = [p for p in self._phase_order if p in phase_seq]
        if phase_seq != canonical:
            score -= 0.2
        all_phases = self._extract_all_phases(text)
        for _phase, contents in all_phases.items():
            for content in contents:
                if not content.strip():
                    score -= 0.1
        if all_phases.get("reflect"):
            score += 0.1
        return max(0.0, min(1.0, score))


class AgenticQualityRewardModel:
    """Evaluate agentic reasoning quality (heuristic or LLM judge)."""

    def __init__(self, judge_model: Optional[Any] = None, judge_tokenizer: Optional[Any] = None):
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        from ...core.agentic_tokens import extract_all_phases
        self._extract_all_phases = extract_all_phases

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        if self.judge_model is not None:
            return self._judge_inference(text)
        return self._heuristic_score(text)

    def _heuristic_score(self, text: str) -> float:
        phases = self._extract_all_phases(text)
        score = 1.0
        act_contents = phases.get("act", [])
        verify_contents = phases.get("verify", [])
        if act_contents and verify_contents:
            act_numbers = set(re.findall(r"\d+", " ".join(act_contents)))
            verify_numbers = set(re.findall(r"\d+", " ".join(verify_contents)))
            if act_numbers and not act_numbers.intersection(verify_numbers):
                score -= 0.2
        reflect_contents = phases.get("reflect", [])
        if reflect_contents:
            reflect_text = " ".join(reflect_contents).lower()
            error_keywords = [
                "miss", "wrong", "error", "mistake", "confus",
                "incorrect", "forgot", "skip", "overlook",
            ]
            if not any(kw in reflect_text for kw in error_keywords):
                score -= 0.15
        decide_contents = phases.get("decide", [])
        if decide_contents and len(" ".join(decide_contents)) > 300:
            score -= 0.15
        if len(verify_contents) > 1 and len(set(verify_contents)) < len(verify_contents):
            score -= 0.2
        if len(text) > 4000:
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _judge_inference(self, text: str) -> float:
        prompt = (
            "Evaluate the following agentic visual reasoning response.\n"
            "Score: 0.0 (poor), 0.5 (fair), or 1.0 (good). Output only the score.\n\n"
            f"Response:\n{text}\n"
        )
        judge_model = self.judge_model
        judge_tokenizer = self.judge_tokenizer
        if judge_model is None or judge_tokenizer is None:
            return self._heuristic_score(text)
        try:
            inputs = judge_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024,
            )
            device = next(judge_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = judge_model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=judge_tokenizer.eos_token_id,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = judge_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            return QualityRewardModel._parse_judge_score(decoded)
        except Exception:
            return self._heuristic_score(text)


class AgenticSelfCorrectionRewardModel:
    """Reward self-correction trajectories (VERIFY → REFLECT → corrected DECIDE)."""

    def __init__(self):
        from ...core.agentic_tokens import extract_all_phases
        self._extract_all_phases = extract_all_phases

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        phases = self._extract_all_phases(text)
        score = 0.0
        verify_text = " ".join(phases.get("verify", "")).lower()
        error_detected = any(
            kw in verify_text
            for kw in ["wrong", "error", "mistake", "incorrect", "not", "but", "wait"]
        )
        reflect_text = " ".join(phases.get("reflect", "")).lower()
        error_acknowledged = bool(phases.get("reflect")) and any(
            kw in reflect_text
            for kw in ["miss", "wrong", "error", "mistake", "confus", "incorrect", "forgot", "skip"]
        )
        act_text = " ".join(phases.get("act", ""))
        decide_text = " ".join(phases.get("decide", ""))
        if error_detected:
            score += 0.3
        if error_acknowledged:
            score += 0.3
        if act_text and decide_text and act_text.strip() != decide_text.strip():
            score += 0.2
        if metadata and metadata.get("error_injected"):
            if error_detected and error_acknowledged:
                score += 0.2
            elif not error_detected:
                score -= 0.1
        return max(0.0, min(1.0, score))


def build_agentic_reward_models(
    judge_model: Optional[Any] = None,
    judge_tokenizer: Optional[Any] = None,
) -> List[Callable[[str, Dict], float]]:
    """Build reward function list for agentic meta-cognitive training."""
    return [
        AgenticFormatRewardModel(),
        AgenticQualityRewardModel(judge_model, judge_tokenizer),
        AgenticSelfCorrectionRewardModel(),
        MixedAccuracyRewardModel(),
    ]
