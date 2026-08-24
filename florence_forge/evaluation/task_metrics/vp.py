"""Visual primitive detection metrics."""

from __future__ import annotations

from typing import Dict, List

from .calculators import DetectionMetrics


class VisualPrimitiveDetectionMetrics(DetectionMetrics):
    """扩展 DetectionMetrics，增加 VP 格式质量与结构化解码指标。"""

    def __init__(self, task_type: str = "OD_VP"):
        super().__init__()
        self.task_type = task_type
        self._vp_format_valid = 0
        self._vp_coordinate_valid = 0
        self._vp_ref_covered = 0
        self._vp_pred_box_counts: List[int] = []
        self._vp_ref_box_counts: List[int] = []
        self._vp_box_count_exact_match = 0
        self._structured_vp_format_valid = 0
        self._structured_vp_decoder_ok = 0
        self._structured_vp_source_florence_native = 0
        self._structured_box_count_exact_match = 0
        self._structured_pred_box_counts: List[int] = []
        self._structured_ref_box_counts: List[int] = []
        self._structured_parsed_preds: List[List[Dict]] = []
        self._structured_parsed_refs: List[List[Dict]] = []

    def add_batch(self, predictions: List[str], references: List[str]) -> None:
        super().add_batch(predictions, references)
        from ..visual_primitive_parser import VisualPrimitiveParser
        from ..structured_vp_decoder import (
            StructuredVisualPrimitiveDecoder,
            FlorenceNativeDetectionParser,
        )

        vp_parser = VisualPrimitiveParser()
        native_parser = FlorenceNativeDetectionParser()
        structured_decoder = StructuredVisualPrimitiveDecoder()

        for pred, ref in zip(predictions, references):
            pred_dets = vp_parser.parse_detections(pred)
            ref_dets = vp_parser.parse_detections(ref)

            if pred_dets:
                self._vp_format_valid += 1
            if ref_dets:
                self._vp_ref_covered += 1

            pred_coords_valid = all(
                all(isinstance(v, (int, float)) and v >= 0 for v in d.get("bbox", []))
                for d in pred_dets
            )
            if pred_dets and pred_coords_valid:
                self._vp_coordinate_valid += 1

            self._vp_pred_box_counts.append(len(pred_dets))
            self._vp_ref_box_counts.append(len(ref_dets))
            if len(pred_dets) == len(ref_dets):
                self._vp_box_count_exact_match += 1

            native_dets = native_parser.parse(pred)
            structured = structured_decoder.decode(pred)

            if native_dets or structured:
                self._structured_vp_format_valid += 1
            if structured:
                self._structured_vp_decoder_ok += 1
            if native_dets:
                self._structured_vp_source_florence_native += 1

            if native_dets:
                struct_pred_dets = native_dets
            elif hasattr(structured, "detections"):
                struct_pred_dets = structured.detections
            else:
                struct_pred_dets = []

            self._structured_pred_box_counts.append(len(struct_pred_dets))
            self._structured_ref_box_counts.append(len(ref_dets))
            if len(struct_pred_dets) == len(ref_dets):
                self._structured_box_count_exact_match += 1

            self._structured_parsed_preds.append(struct_pred_dets)
            self._structured_parsed_refs.append(ref_dets)

    def compute(self) -> Dict[str, float]:
        metrics = super().compute()
        n = len(self.predictions)
        if n == 0:
            return metrics

        metrics["vp_format_valid_ratio"] = self._vp_format_valid / n
        metrics["vp_coordinate_valid_ratio"] = self._vp_coordinate_valid / n
        metrics["vp_ref_coverage_ratio"] = self._vp_ref_covered / n
        metrics["vp_avg_pred_boxes"] = sum(self._vp_pred_box_counts) / n
        metrics["vp_box_count_exact_match"] = self._vp_box_count_exact_match / n
        metrics["structured_vp_format_valid_ratio"] = self._structured_vp_format_valid / n
        metrics["structured_vp_decoder_ratio"] = self._structured_vp_decoder_ok / n
        metrics["structured_vp_source_florence_native_ratio"] = (
            self._structured_vp_source_florence_native / n
        )
        metrics["structured_vp_box_count_exact_match"] = (
            self._structured_box_count_exact_match / n
        )

        if self._structured_parsed_preds and self._structured_parsed_refs:
            total_pred = sum(len(p) for p in self._structured_parsed_preds)
            total_ref = sum(len(r) for r in self._structured_parsed_refs)
            matched = 0
            for preds, refs in zip(self._structured_parsed_preds, self._structured_parsed_refs):
                for p in preds:
                    for r in refs:
                        if (
                            p.get("label", "").lower() == r.get("label", "").lower()
                            and self._bbox_iou(p.get("bbox", []), r.get("bbox", [])) > 0.5
                        ):
                            matched += 1
                            break
            metrics["structured_precision"] = matched / total_pred if total_pred else 0.0
            metrics["structured_recall"] = matched / total_ref if total_ref else 0.0

        return metrics

    @staticmethod
    def _bbox_iou(box1: List, box2: List) -> float:
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
