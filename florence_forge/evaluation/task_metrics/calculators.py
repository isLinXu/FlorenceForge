"""Task-specific metric calculators (caption, detection, OCR, segmentation)."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.optional_dependencies import missing_dependency_message
from .base import MetricCalculator
from ._deps import CV2_AVAILABLE, ROUGE_AVAILABLE, rouge_scorer

logger = logging.getLogger(__name__)


class CaptionMetrics(MetricCalculator):
    """图像描述任务指标"""

    def __init__(self):
        super().__init__("caption")

    def compute(self) -> Dict[str, float]:
        if not self.predictions or not self.references:
            return {}

        metrics: Dict[str, float] = {}
        metrics.update(self._compute_bleu())
        metrics.update(self._compute_rouge())
        metrics["word_overlap"] = self._compute_word_overlap()
        metrics.update(self._compute_length_stats())
        return metrics

    def _compute_bleu(self) -> Dict[str, float]:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            warned_tokenizer_fallback = False

            def tokenize(text: str) -> List[str]:
                nonlocal warned_tokenizer_fallback
                try:
                    return word_tokenize(text.lower())
                except LookupError:
                    if not warned_tokenizer_fallback:
                        logger.warning(
                            "NLTK punkt 数据未安装，BLEU 计算降级为正则分词；"
                            "如需标准分词，请安装 evaluation 额外依赖并预先下载 punkt。"
                        )
                        warned_tokenizer_fallback = True
                    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

            smoothing = SmoothingFunction().method1
            bleu_scores = []
            for pred, ref in zip(self.predictions, self.references):
                pred_tokens = tokenize(pred)
                ref_tokens = [tokenize(ref)]
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)

            return {"bleu": np.mean(bleu_scores), "bleu_std": np.std(bleu_scores)}

        except ImportError:
            logger.warning(missing_dependency_message("BLEU计算", "nltk", "evaluation"))
            return {}

    def _compute_rouge(self) -> Dict[str, float]:
        if not ROUGE_AVAILABLE:
            return {}
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_scores: Dict[str, List[float]] = defaultdict(list)
            for pred, ref in zip(self.predictions, self.references):
                scores = scorer.score(ref, pred)
                for key, score in scores.items():
                    rouge_scores[f"{key}_f1"].append(score.fmeasure)
                    rouge_scores[f"{key}_precision"].append(score.precision)
                    rouge_scores[f"{key}_recall"].append(score.recall)
            return {key: np.mean(values) for key, values in rouge_scores.items()}
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            return {}

    def _compute_word_overlap(self) -> float:
        overlaps = []
        for pred, ref in zip(self.predictions, self.references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words & ref_words) / len(ref_words) if ref_words else 0.0
            overlaps.append(overlap)
        return float(np.mean(overlaps))

    def _compute_length_stats(self) -> Dict[str, float]:
        pred_lengths = [len(pred.split()) for pred in self.predictions]
        ref_lengths = [len(ref.split()) for ref in self.references]
        return {
            "pred_avg_length": np.mean(pred_lengths),
            "ref_avg_length": np.mean(ref_lengths),
            "length_ratio": (
                np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0
            ),
        }


class DetectionMetrics(MetricCalculator):
    """目标检测任务指标"""

    def __init__(self):
        super().__init__("detection")
        self.iou_threshold = 0.5

    def compute(self) -> Dict[str, float]:
        if not self.predictions or not self.references:
            return {}

        parsed_predictions = [self._parse_detection_result(pred) for pred in self.predictions]
        parsed_references = [self._parse_detection_result(ref) for ref in self.references]
        metrics = self._compute_basic_metrics(parsed_predictions, parsed_references)
        try:
            metrics["mAP"] = self._compute_map(parsed_predictions, parsed_references)
        except Exception as e:
            logger.warning(f"mAP计算失败: {e}")
        return metrics

    def _parse_detection_result(self, result: str) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        try:
            from ..structured_vp_decoder import (
                FlorenceNativeDetectionParser,
                StructuredVisualPrimitiveDecoder,
            )

            text = str(result or "").strip()
            if not text:
                return detections

            native_detections = FlorenceNativeDetectionParser().parse(text)
            if native_detections:
                return native_detections

            structured = StructuredVisualPrimitiveDecoder().decode(text)
            if structured.detections:
                return structured.detections

            if text.startswith("[") or text.startswith("{"):
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "objects" in data:
                    return data["objects"]

            vp_pattern = (
                r"<\|?ref\|?>([^<]+)<\|?/ref\|?>\s*<\|?box\|?>\s*\[\[([^\]]*)\]\]\s*<\|?/box\|?>"
            )
            vp_matches = re.findall(vp_pattern, text)
            if vp_matches:
                for label, coords_str in vp_matches:
                    coords = [float(x.strip()) for x in coords_str.split(",")]
                    detections.append({"label": label.strip(), "bbox": coords, "confidence": 1.0})
                return detections

            pattern = r"([^<]+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
            for label, x1, y1, x2, y2 in re.findall(pattern, text):
                detections.append({
                    "label": label.strip(),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": 1.0,
                })
        except Exception as e:
            logger.warning(f"解析检测结果失败: {e}")
        return detections

    def _compute_basic_metrics(
        self,
        predictions: List[List[Dict]],
        references: List[List[Dict]],
    ) -> Dict[str, float]:
        total_tp = total_fp = total_fn = 0
        for pred_boxes, ref_boxes in zip(predictions, references):
            tp, fp, fn = self._compute_matches(pred_boxes, ref_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }

    def _compute_matches(
        self,
        pred_boxes: List[Dict],
        ref_boxes: List[Dict],
    ) -> Tuple[int, int, int]:
        if not pred_boxes and not ref_boxes:
            return 0, 0, 0
        if not pred_boxes:
            return 0, 0, len(ref_boxes)
        if not ref_boxes:
            return 0, len(pred_boxes), 0

        iou_matrix = np.zeros((len(pred_boxes), len(ref_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, ref_box in enumerate(ref_boxes):
                if pred_box.get("label") == ref_box.get("label"):
                    iou_matrix[i, j] = self._compute_iou(pred_box["bbox"], ref_box["bbox"])

        matched_pred: set[int] = set()
        matched_ref: set[int] = set()
        matches = []
        for i in range(len(pred_boxes)):
            for j in range(len(ref_boxes)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        matches.sort(reverse=True)

        tp = 0
        for _iou, i, j in matches:
            if i not in matched_pred and j not in matched_ref:
                matched_pred.add(i)
                matched_ref.add(j)
                tp += 1
        return tp, len(pred_boxes) - tp, len(ref_boxes) - tp

    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _compute_map(
        self,
        predictions: List[List[Dict]],
        references: List[List[Dict]],
    ) -> float:
        """Compute mAP using torchvision box_iou (no pycocotools required)."""
        import torch
        from collections import defaultdict as dd

        if not predictions or not references:
            return 0.0

        try:
            from torchvision.ops import box_iou
        except ImportError:
            logger.warning(missing_dependency_message("mAP计算", "torchvision"))
            return 0.0

        iou_threshold = 0.5
        all_aps: List[float] = []
        pred_by_class: Dict[str, List[Tuple[int, torch.Tensor, float]]] = dd(list)
        gt_by_class: Dict[str, Dict[int, List[torch.Tensor]]] = dd(lambda: dd(list))

        for image_idx, (pred_boxes, ref_boxes) in enumerate(zip(predictions, references)):
            for pred in pred_boxes:
                cat = pred.get("category") or pred.get("label") or "default"
                bbox = pred.get("bbox", [0, 0, 0, 0])
                score = float(pred.get("score", pred.get("confidence", 1.0)))
                pred_by_class[cat].append(
                    (image_idx, torch.tensor(bbox, dtype=torch.float32).unsqueeze(0), score)
                )
            for ref in ref_boxes:
                cat = ref.get("category") or ref.get("label") or "default"
                bbox = ref.get("bbox", [0, 0, 0, 0])
                gt_by_class[cat][image_idx].append(
                    torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)
                )

        all_categories = set(pred_by_class.keys()) | set(gt_by_class.keys())
        if not all_categories:
            all_categories = {"default"}

        for cat in all_categories:
            preds = pred_by_class.get(cat, [])
            gts = gt_by_class.get(cat, {})
            total_gts = sum(len(image_gts) for image_gts in gts.values())
            if total_gts == 0:
                continue
            if not preds:
                all_aps.append(0.0)
                continue

            preds.sort(key=lambda x: x[2], reverse=True)
            tp = torch.zeros(len(preds))
            fp = torch.zeros(len(preds))
            matched_gt_by_image: Dict[int, set[int]] = dd(set)

            for i, (image_idx, pred_box, _score) in enumerate(preds):
                image_gt_list = gts.get(image_idx, [])
                if not image_gt_list:
                    fp[i] = 1
                    continue
                image_gt_boxes = torch.cat(image_gt_list)
                ious = box_iou(pred_box, image_gt_boxes).squeeze(0)
                if ious.numel() == 0:
                    fp[i] = 1
                    continue
                best_iou, best_j = ious.max(dim=0)
                best_j_int = int(best_j.item())
                if best_iou >= iou_threshold and best_j_int not in matched_gt_by_image[image_idx]:
                    tp[i] = 1
                    matched_gt_by_image[image_idx].add(best_j_int)
                else:
                    fp[i] = 1

            tp_cumsum = tp.cumsum(dim=0)
            fp_cumsum = fp.cumsum(dim=0)
            recalls = tp_cumsum / total_gts
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            ap = 0.0
            for t in torch.linspace(0, 1, 11):
                if (recalls >= t).any():
                    ap += precisions[recalls >= t].max().item()
            all_aps.append(ap / 11.0)

        return float(np.mean(all_aps)) if all_aps else 0.0


class OCRMetrics(MetricCalculator):
    """OCR任务指标"""

    def __init__(self):
        super().__init__("ocr")

    def compute(self) -> Dict[str, float]:
        if not self.predictions or not self.references:
            return {}
        metrics: Dict[str, float] = {}
        metrics.update(self._compute_character_metrics())
        metrics.update(self._compute_word_metrics())
        metrics["edit_distance"] = self._compute_edit_distance()
        return metrics

    def _compute_character_metrics(self) -> Dict[str, float]:
        total_chars = correct_chars = 0
        for pred, ref in zip(self.predictions, self.references):
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)
            total_chars += len(ref_clean)
            for i, char in enumerate(ref_clean):
                if i < len(pred_clean) and pred_clean[i] == char:
                    correct_chars += 1
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        return {
            "character_accuracy": char_accuracy,
            "total_characters": total_chars,
            "correct_characters": correct_chars,
        }

    def _compute_word_metrics(self) -> Dict[str, float]:
        total_words = correct_words = 0
        for pred, ref in zip(self.predictions, self.references):
            pred_words = self._clean_text(pred).split()
            ref_words = self._clean_text(ref).split()
            total_words += len(ref_words)
            for i, word in enumerate(ref_words):
                if i < len(pred_words) and pred_words[i] == word:
                    correct_words += 1
        word_accuracy = correct_words / total_words if total_words > 0 else 0.0
        return {
            "word_accuracy": word_accuracy,
            "total_words": total_words,
            "correct_words": correct_words,
        }

    def _compute_edit_distance(self) -> float:
        distances = []
        for pred, ref in zip(self.predictions, self.references):
            distances.append(
                self._levenshtein_distance(self._clean_text(pred), self._clean_text(ref))
            )
        return float(np.mean(distances))

    def _clean_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                current_row.append(
                    min(
                        previous_row[j + 1] + 1,
                        current_row[j] + 1,
                        previous_row[j] + (c1 != c2),
                    )
                )
            previous_row = current_row
        return previous_row[-1]


class SegmentationMetrics(MetricCalculator):
    """分割任务指标"""

    def __init__(self):
        super().__init__("segmentation")

    def compute(self) -> Dict[str, float]:
        if not self.predictions or not self.references:
            return {}
        parsed_predictions = [self._parse_segmentation_result(p) for p in self.predictions]
        parsed_references = [self._parse_segmentation_result(r) for r in self.references]
        return self._compute_segmentation_metrics(parsed_predictions, parsed_references)

    def _parse_segmentation_result(self, result: str) -> Optional[np.ndarray]:
        try:
            if "<poly>" in result:
                poly_pattern = r"<poly>(.*?)</poly>"
                matches = re.findall(poly_pattern, result)
                if matches and CV2_AVAILABLE:
                    return self._polygon_to_mask(matches[0], (224, 224))
        except Exception as e:
            logger.warning(f"解析分割结果失败: {e}")
        return None

    def _polygon_to_mask(self, polygon_str: str, image_size: Tuple[int, int]) -> np.ndarray:
        import cv2

        if not CV2_AVAILABLE:
            return np.zeros(image_size, dtype=np.uint8)
        coords = [[int(x), int(y)] for x, y in re.findall(r"(\d+),(\d+)", polygon_str)]
        if not coords:
            return np.zeros(image_size, dtype=np.uint8)
        mask = np.zeros(image_size, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 1)
        return mask

    def _compute_segmentation_metrics(
        self,
        predictions: List[Optional[np.ndarray]],
        references: List[Optional[np.ndarray]],
    ) -> Dict[str, float]:
        ious: List[float] = []
        dice_scores: List[float] = []
        for pred_mask, ref_mask in zip(predictions, references):
            if pred_mask is None or ref_mask is None or pred_mask.shape != ref_mask.shape:
                continue
            intersection = np.logical_and(pred_mask, ref_mask).sum()
            union = np.logical_or(pred_mask, ref_mask).sum()
            if union > 0:
                ious.append(intersection / union)
                dice_scores.append(2 * intersection / (pred_mask.sum() + ref_mask.sum()))
        return {
            "mean_iou": np.mean(ious) if ious else 0.0,
            "mean_dice": np.mean(dice_scores) if dice_scores else 0.0,
            "num_valid_samples": len(ious),
        }
