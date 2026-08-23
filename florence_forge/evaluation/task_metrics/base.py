"""Base metric calculator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MetricCalculator:
    """指标计算器基类"""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.predictions: List[str] = []
        self.references: List[str] = []

    def add_batch(self, predictions: List[str], references: List[str]) -> None:
        self.predictions.extend(predictions)
        self.references.extend(references)

    def compute(self) -> Dict[str, float]:
        if not self.predictions or not self.references:
            logger.warning(f"任务 {self.task_type} 没有预测或参考数据，返回空指标")
            return {}

        if len(self.predictions) != len(self.references):
            logger.error(
                f"预测数量({len(self.predictions)})与参考数量({len(self.references)})不匹配"
            )
            return {}

        metrics: Dict[str, Any] = {"num_samples": len(self.predictions)}

        pred_lengths = [len(str(pred).split()) for pred in self.predictions]
        ref_lengths = [len(str(ref).split()) for ref in self.references]

        metrics.update({
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "pred_length_std": np.std(pred_lengths),
            "ref_length_std": np.std(ref_lengths),
        })

        exact_matches = sum(
            1
            for p, r in zip(self.predictions, self.references)
            if str(p).strip() == str(r).strip()
        )
        metrics["exact_match_ratio"] = exact_matches / len(self.predictions)
        return metrics

    def reset(self) -> None:
        self.predictions = []
        self.references = []
