"""Structured decoder for turning Florence native loc output into VP output."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ..core.visual_primitives import (
    VISUAL_PRIMITIVE_MARKER_SETS,
    format_ref_box,
    format_ref_box_loc_tokens,
    validate_normalized_bbox,
)
from .visual_primitive_parser import VisualPrimitiveParser


_LOC_GROUP_PATTERN = re.compile(r"((?:<loc_\d+>){4})")
_LOC_VALUE_PATTERN = re.compile(r"<loc_(\d+)>")
_PLAIN_REF_SPAN_PATTERN = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
_SPECIAL_REF_SPAN_PATTERN = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_SINGLE_TARGET_FILTER_TASKS = {
    "OD_VP",
    "PHRASE_GROUNDING_VP",
    "CAPTION_TO_PHRASE_GROUNDING",
}


@dataclass(frozen=True)
class StructuredVPDecodeResult:
    """Result of a structured VP decoding pass."""

    text: str
    detections: List[Dict[str, object]]
    source: str
    input_text: str
    raw_detection_count: int = 0
    filtered_detection_count: int = 0
    repaired_tail_detection_count: int = 0

    @property
    def used_structured_decoder(self) -> bool:
        return self.source == "florence_native"

    @property
    def used_tail_repair(self) -> bool:
        return self.repaired_tail_detection_count > 0


class FlorenceNativeDetectionParser:
    """Parse Florence native ``label<loc_*>`` object detection output."""

    def parse(self, text: str) -> List[Dict[str, object]]:
        if not text:
            return []

        detections: List[Dict[str, object]] = []
        last_label = ""
        cursor = 0
        for match in _LOC_GROUP_PATTERN.finditer(str(text)):
            label = self._clean_label(str(text)[cursor:match.start()])
            if not label:
                label = last_label
            values = [int(value) for value in _LOC_VALUE_PATTERN.findall(match.group(1))]
            cursor = match.end()
            if len(values) != 4 or not label:
                continue
            if not validate_normalized_bbox(values):
                continue
            detections.append({"label": label, "bbox": values, "confidence": 1.0})
            last_label = label
        return detections

    @staticmethod
    def _clean_label(text: str) -> str:
        explicit_ref = FlorenceNativeDetectionParser._extract_last_ref_label(text)
        if explicit_ref:
            return explicit_ref
        return FlorenceNativeDetectionParser._strip_label_noise(text)

    @staticmethod
    def _extract_last_ref_label(text: str) -> str:
        matches = []
        for pattern in (_SPECIAL_REF_SPAN_PATTERN, _PLAIN_REF_SPAN_PATTERN):
            matches.extend(pattern.finditer(text))
        if not matches:
            return ""
        last_match = max(matches, key=lambda match: match.start())
        return FlorenceNativeDetectionParser._strip_label_noise(last_match.group(1))

    @staticmethod
    def _strip_label_noise(text: str) -> str:
        text = re.sub(r"</?s>|<pad>|<bos>|<eos>", " ", text)
        text = re.sub(r"<OD>|<CAPTION_TO_PHRASE_GROUNDING>", " ", text)
        text = re.sub(r"<\|/?(?:ref|box|point)\|>", " ", text)
        text = re.sub(r"</?(?:ref|box|point)>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n:;,")


class StructuredVisualPrimitiveDecoder:
    """Convert native Florence outputs into deterministic VP evidence chains."""

    def __init__(
        self,
        box_format: str = "loc_tokens",
        marker_style: str = "special",
        max_boxes_per_label: Optional[int] = None,
        max_total_boxes: Optional[int] = None,
        nms_iou_threshold: Optional[float] = None,
        allowed_labels: Optional[Union[Sequence[str], str]] = None,
        allowed_label_match_mode: str = "strict",
        repair_malformed_tail: bool = False,
    ):
        self.box_format = box_format
        self.marker_style = marker_style
        self.max_boxes_per_label = _validate_positive_optional(max_boxes_per_label, "max_boxes_per_label")
        self.max_total_boxes = _validate_positive_optional(max_total_boxes, "max_total_boxes")
        self.nms_iou_threshold = _validate_nms_threshold_optional(nms_iou_threshold)
        self.allowed_labels = normalize_allowed_labels(allowed_labels)
        self.allowed_label_match_mode = _validate_label_match_mode(allowed_label_match_mode)
        self.repair_malformed_tail = bool(repair_malformed_tail)
        self.vp_parser = VisualPrimitiveParser()
        self.native_parser = FlorenceNativeDetectionParser()

    def decode(
        self,
        text: str,
        box_format: str | None = None,
        marker_style: str | None = None,
        max_boxes_per_label: Optional[int] = None,
        max_total_boxes: Optional[int] = None,
        nms_iou_threshold: Optional[float] = None,
        allowed_labels: Optional[Union[Sequence[str], str]] = None,
        allowed_label_match_mode: Optional[str] = None,
        repair_malformed_tail: Optional[bool] = None,
    ) -> StructuredVPDecodeResult:
        """Return VP text, preserving valid VP text and wrapping native output."""

        desired_format = box_format or self.box_format
        desired_marker_style = marker_style or self.marker_style
        desired_max_boxes_per_label = (
            _validate_positive_optional(max_boxes_per_label, "max_boxes_per_label")
            if max_boxes_per_label is not None else self.max_boxes_per_label
        )
        desired_max_total_boxes = (
            _validate_positive_optional(max_total_boxes, "max_total_boxes")
            if max_total_boxes is not None else self.max_total_boxes
        )
        desired_nms_iou_threshold = (
            _validate_nms_threshold_optional(nms_iou_threshold)
            if nms_iou_threshold is not None else self.nms_iou_threshold
        )
        desired_allowed_labels = (
            normalize_allowed_labels(allowed_labels)
            if allowed_labels is not None else self.allowed_labels
        )
        desired_allowed_label_match_mode = (
            _validate_label_match_mode(allowed_label_match_mode)
            if allowed_label_match_mode is not None else self.allowed_label_match_mode
        )
        desired_repair_malformed_tail = (
            bool(repair_malformed_tail)
            if repair_malformed_tail is not None else self.repair_malformed_tail
        )
        existing = self.vp_parser.parse_detections(text)
        if existing:
            if desired_repair_malformed_tail:
                repaired = _merge_malformed_tail_detections(
                    str(text),
                    existing,
                    native_parser=self.native_parser,
                )
                if len(repaired) > len(existing):
                    filtered_repaired = filter_native_detections(
                        repaired,
                        max_boxes_per_label=desired_max_boxes_per_label,
                        max_total_boxes=desired_max_total_boxes,
                        nms_iou_threshold=desired_nms_iou_threshold,
                        allowed_labels=desired_allowed_labels,
                        allowed_label_match_mode=desired_allowed_label_match_mode,
                    )
                    return StructuredVPDecodeResult(
                        text=native_detections_to_vp(
                            filtered_repaired,
                            box_format=desired_format,
                            marker_style=desired_marker_style,
                        ),
                        detections=filtered_repaired,
                        source="visual_primitive_repaired_tail",
                        input_text=str(text),
                        raw_detection_count=len(repaired),
                        filtered_detection_count=max(
                            0, len(repaired) - len(filtered_repaired)
                        ),
                        repaired_tail_detection_count=len(repaired) - len(existing),
                    )
            if _has_active_filter(
                desired_max_boxes_per_label,
                desired_max_total_boxes,
                desired_nms_iou_threshold,
                desired_allowed_labels,
            ):
                filtered_existing = filter_native_detections(
                    existing,
                    max_boxes_per_label=desired_max_boxes_per_label,
                    max_total_boxes=desired_max_total_boxes,
                    nms_iou_threshold=desired_nms_iou_threshold,
                    allowed_labels=desired_allowed_labels,
                    allowed_label_match_mode=desired_allowed_label_match_mode,
                )
                return StructuredVPDecodeResult(
                    text=native_detections_to_vp(
                        filtered_existing,
                        box_format=desired_format,
                        marker_style=desired_marker_style,
                    ),
                    detections=filtered_existing,
                    source="visual_primitive",
                    input_text=str(text),
                    raw_detection_count=len(existing),
                    filtered_detection_count=max(0, len(existing) - len(filtered_existing)),
                )
            return StructuredVPDecodeResult(
                text=str(text),
                detections=existing,
                source="visual_primitive",
                input_text=str(text),
                raw_detection_count=len(existing),
                filtered_detection_count=0,
            )

        native = self.native_parser.parse(text)
        if not native:
            return StructuredVPDecodeResult(
                text=str(text),
                detections=[],
                source="unparsed",
                input_text=str(text),
                raw_detection_count=0,
                filtered_detection_count=0,
            )

        filtered_native = filter_native_detections(
            native,
            max_boxes_per_label=desired_max_boxes_per_label,
            max_total_boxes=desired_max_total_boxes,
            nms_iou_threshold=desired_nms_iou_threshold,
            allowed_labels=desired_allowed_labels,
            allowed_label_match_mode=desired_allowed_label_match_mode,
        )
        vp_text = native_detections_to_vp(
            filtered_native,
            box_format=desired_format,
            marker_style=desired_marker_style,
        )
        detections = self.vp_parser.parse_detections(vp_text)
        return StructuredVPDecodeResult(
            text=vp_text,
            detections=detections,
            source="florence_native",
            input_text=str(text),
            raw_detection_count=len(native),
            filtered_detection_count=max(0, len(native) - len(filtered_native)),
        )


def _merge_malformed_tail_detections(
    text: str,
    existing: Sequence[Dict[str, object]],
    *,
    native_parser: FlorenceNativeDetectionParser,
) -> List[Dict[str, object]]:
    if not existing:
        return []

    tail_start = _last_complete_vp_box_end(text)
    if tail_start is None:
        return list(existing)
    tail = str(text)[tail_start:]
    if not _LOC_GROUP_PATTERN.search(tail):
        return list(existing)

    last_label = str(existing[-1].get("label", "") or "").strip()
    context = f"<ref>{last_label}</ref> " if last_label else ""
    tail_detections = native_parser.parse(f"{context}{tail}")
    if not tail_detections:
        return list(existing)
    return [dict(detection) for detection in existing] + tail_detections


def _last_complete_vp_box_end(text: str) -> Optional[int]:
    ends: List[int] = []
    for markers in VISUAL_PRIMITIVE_MARKER_SETS.values():
        pattern = re.compile(
            rf"{re.escape(markers['box_open'])}.*?{re.escape(markers['box_close'])}",
            flags=re.DOTALL,
        )
        ends.extend(match.end() for match in pattern.finditer(str(text or "")))
    return max(ends) if ends else None


def resolve_structured_vp_filter_caps(
    *,
    policy: str = "none",
    task_prompt: object = None,
    max_boxes_per_label: Optional[int] = None,
    max_total_boxes: Optional[int] = None,
    nms_iou_threshold: Optional[float] = None,
    allowed_labels: Optional[Union[Sequence[str], str]] = None,
) -> Dict[str, object]:
    """Resolve structured VP post-filter caps from policy and explicit caps."""

    resolved_max_boxes_per_label = _validate_positive_optional(
        max_boxes_per_label,
        "max_boxes_per_label",
    )
    resolved_max_total_boxes = _validate_positive_optional(max_total_boxes, "max_total_boxes")
    resolved_nms_iou_threshold = _validate_nms_threshold_optional(nms_iou_threshold)
    resolved_allowed_labels = normalize_allowed_labels(allowed_labels)
    normalized_policy = str(policy or "none").lower().replace("_", "-")

    if normalized_policy in {"none", "off"}:
        return {
            "max_boxes_per_label": resolved_max_boxes_per_label,
            "max_total_boxes": resolved_max_total_boxes,
            "nms_iou_threshold": resolved_nms_iou_threshold,
            "allowed_labels": resolved_allowed_labels,
        }
    if normalized_policy in {"single-target", "single"}:
        return {
            "max_boxes_per_label": resolved_max_boxes_per_label,
            "max_total_boxes": resolved_max_total_boxes or 1,
            "nms_iou_threshold": resolved_nms_iou_threshold,
            "allowed_labels": resolved_allowed_labels,
        }
    if normalized_policy == "nms":
        return {
            "max_boxes_per_label": resolved_max_boxes_per_label,
            "max_total_boxes": resolved_max_total_boxes,
            "nms_iou_threshold": resolved_nms_iou_threshold or 0.5,
            "allowed_labels": resolved_allowed_labels,
        }
    if normalized_policy == "auto":
        normalized_task = normalize_structured_vp_task_prompt(task_prompt)
        if normalized_task in _SINGLE_TARGET_FILTER_TASKS:
            resolved_max_total_boxes = resolved_max_total_boxes or 1
        return {
            "max_boxes_per_label": resolved_max_boxes_per_label,
            "max_total_boxes": resolved_max_total_boxes,
            "nms_iou_threshold": resolved_nms_iou_threshold,
            "allowed_labels": resolved_allowed_labels,
        }
    raise ValueError("structured VP filter policy must be one of: none, auto, single-target, nms")


def normalize_structured_vp_task_prompt(task_prompt: object) -> str:
    """Normalize Florence task prompts for structured VP routing."""

    text = str(task_prompt or "").strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text.strip().upper()


def filter_native_detections(
    detections: Iterable[Dict[str, object]],
    *,
    max_boxes_per_label: Optional[int] = None,
    max_total_boxes: Optional[int] = None,
    nms_iou_threshold: Optional[float] = None,
    allowed_labels: Optional[Union[Sequence[str], str]] = None,
    allowed_label_match_mode: str = "strict",
) -> List[Dict[str, object]]:
    """Apply deterministic GT-free caps and optional per-label NMS.

    The filter preserves generation order and keeps the first boxes. It is
    intentionally simple and opt-in so experiments can compare raw native
    behavior against constrained structured VP decoding.
    """

    per_label_cap = _validate_positive_optional(max_boxes_per_label, "max_boxes_per_label")
    total_cap = _validate_positive_optional(max_total_boxes, "max_total_boxes")
    nms_threshold = _validate_nms_threshold_optional(nms_iou_threshold)
    allowed_label_list = normalize_allowed_labels(allowed_labels) or []
    label_match_mode = _validate_label_match_mode(allowed_label_match_mode)
    kept: List[Dict[str, object]] = []
    label_counts: Dict[str, int] = {}

    for detection in detections:
        label = str(detection.get("label", "")).strip()
        if allowed_label_list and not label_matches_allowed_labels(
            label,
            allowed_label_list,
            mode=label_match_mode,
        ):
            continue
        bbox = detection.get("bbox")
        if nms_threshold is not None and _is_suppressed_by_nms(
            label=label,
            bbox=bbox,
            kept=kept,
            threshold=nms_threshold,
        ):
            continue
        if per_label_cap is not None:
            count = label_counts.get(label, 0)
            if count >= per_label_cap:
                continue
            label_counts[label] = count + 1
        kept.append(dict(detection))
        if total_cap is not None and len(kept) >= total_cap:
            break

    return kept


def native_detections_to_vp(
    detections: Iterable[Dict[str, object]],
    *,
    box_format: str = "loc_tokens",
    marker_style: str = "special",
) -> str:
    """Format native detections as grouped VP ref/box spans."""

    grouped: "OrderedDict[str, List[Sequence[int]]]" = OrderedDict()
    for detection in detections:
        label = str(detection.get("label", "")).strip()
        bbox = detection.get("bbox")
        if not label or not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        box = [int(value) for value in bbox]
        if not validate_normalized_bbox(box):
            continue
        grouped.setdefault(label, []).append(box)

    formatter = _get_formatter(box_format, marker_style=marker_style)
    return "\n".join(formatter(label, boxes) for label, boxes in grouped.items())


def _get_formatter(box_format: str, marker_style: str = "special"):
    if box_format == "json":
        return lambda label, boxes: format_ref_box(label, boxes, marker_style=marker_style)
    if box_format in {"loc_tokens", "loc"}:
        return lambda label, boxes: format_ref_box_loc_tokens(label, boxes, marker_style=marker_style)
    raise ValueError("box_format must be 'json' or 'loc_tokens'")


def _validate_positive_optional(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1 when provided")
    return value


def _validate_nms_threshold_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0 or value > 1.0:
        raise ValueError("nms_iou_threshold must be in (0, 1] when provided")
    return value


def _has_active_filter(*values: object) -> bool:
    return any(value is not None for value in values)


def normalize_allowed_labels(value: Optional[Union[Sequence[str], str]]) -> Optional[List[str]]:
    """Normalize comma/newline separated label allow-lists."""

    if value is None:
        return None
    if isinstance(value, str):
        raw_items = re.split(r"[,;\n|]+", value)
    else:
        raw_items = [str(item) for item in value]
    labels: List[str] = []
    seen = set()
    for item in raw_items:
        label = _normalize_label(item)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels or None


def labels_match(left: object, right: object, *, mode: str = "strict") -> bool:
    """Return whether two labels are compatible under a conservative mode."""

    match_mode = _validate_label_match_mode(mode)
    left_label = _normalize_label(left)
    right_label = _normalize_label(right)
    if not left_label or not right_label:
        return False
    if left_label == right_label:
        return True
    if match_mode == "strict":
        return False

    left_tokens = _label_tokens(left_label)
    right_tokens = _label_tokens(right_label)
    if not left_tokens or not right_tokens:
        return False
    return _contains_token_phrase(left_tokens, right_tokens) or _contains_token_phrase(
        right_tokens,
        left_tokens,
    )


def label_matches_allowed_labels(
    label: object,
    allowed_labels: Optional[Union[Sequence[str], str]],
    *,
    mode: str = "strict",
) -> bool:
    """Return whether a label is accepted by an allow-list."""

    normalized_allowed_labels = normalize_allowed_labels(allowed_labels) or []
    if not normalized_allowed_labels:
        return True
    return any(labels_match(label, allowed_label, mode=mode) for allowed_label in normalized_allowed_labels)


def _normalize_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _validate_label_match_mode(value: str) -> str:
    mode = str(value or "strict").strip().lower().replace("_", "-")
    if mode in {"strict", "exact"}:
        return "strict"
    if mode in {"contains", "contain", "substring", "phrase-contains", "phrase"}:
        return "contains"
    raise ValueError("label match mode must be one of: strict, contains")


def _label_tokens(value: object) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_label(value))


def _contains_token_phrase(container: Sequence[str], phrase: Sequence[str]) -> bool:
    if len(phrase) > len(container):
        return False
    return any(
        list(container[start:start + len(phrase)]) == list(phrase)
        for start in range(0, len(container) - len(phrase) + 1)
    )


def _is_suppressed_by_nms(
    *,
    label: str,
    bbox: object,
    kept: Sequence[Dict[str, object]],
    threshold: float,
) -> bool:
    if not _is_bbox_sequence(bbox):
        return False
    for kept_detection in kept:
        if str(kept_detection.get("label", "")).strip() != label:
            continue
        kept_bbox = kept_detection.get("bbox")
        if not _is_bbox_sequence(kept_bbox):
            continue
        if _bbox_iou(bbox, kept_bbox) >= threshold:
            return True
    return False


def _bbox_iou(box1: Sequence[object], box2: Sequence[object]) -> float:
    x1_1, y1_1, x2_1, y2_1 = [float(value) for value in box1]
    x1_2, y1_2, x2_2, y2_2 = [float(value) for value in box2]
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _is_bbox_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(isinstance(coord, (int, float)) for coord in value)
    )
