"""Parser for visual primitive model outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..core.visual_primitives import (
    COORDINATE_MAX,
    VISUAL_PRIMITIVE_MARKER_SETS,
    VISUAL_PRIMITIVE_TOKENS,
    validate_normalized_bbox,
)


class VisualPrimitiveParser:
    """Parse VP ref/box/point spans into standard Python structures."""

    def __init__(self, coord_max: int = COORDINATE_MAX):
        self.coord_max = coord_max
        self._ref_open = re.escape(VISUAL_PRIMITIVE_TOKENS["ref_open"])
        self._ref_close = re.escape(VISUAL_PRIMITIVE_TOKENS["ref_close"])
        self._box_open = re.escape(VISUAL_PRIMITIVE_TOKENS["box_open"])
        self._box_close = re.escape(VISUAL_PRIMITIVE_TOKENS["box_close"])
        self._point_open = re.escape(VISUAL_PRIMITIVE_TOKENS["point_open"])
        self._point_close = re.escape(VISUAL_PRIMITIVE_TOKENS["point_close"])
        self._marker_sets = tuple(VISUAL_PRIMITIVE_MARKER_SETS.values())

    def parse_detections(self, text: str) -> List[Dict[str, Any]]:
        """Parse VP boxes into detection dicts.

        Returns dictionaries with ``label``, ``bbox`` and ``confidence`` keys so
        the result can be consumed by the existing detection metrics.
        """

        if not text:
            return []

        detections: List[Dict[str, Any]] = []
        consumed_box_spans = set()

        for markers in self._marker_sets:
            ref_box_pattern = re.compile(
                rf"{re.escape(markers['ref_open'])}(.*?){re.escape(markers['ref_close'])}.*?"
                rf"{re.escape(markers['box_open'])}(.*?){re.escape(markers['box_close'])}",
                flags=re.DOTALL,
            )
            for match in ref_box_pattern.finditer(text):
                label = self._clean_label(match.group(1))
                box_payload = match.group(2)
                for bbox in self._parse_boxes(box_payload):
                    detections.append({"label": label, "bbox": bbox, "confidence": 1.0})
                consumed_box_spans.add(match.span(2))

        # Also allow standalone <|box|> spans, which are useful for region
        # proposal style tasks without an explicit ref label.
        for markers in self._marker_sets:
            box_pattern = re.compile(
                rf"{re.escape(markers['box_open'])}(.*?){re.escape(markers['box_close'])}",
                flags=re.DOTALL,
            )
            for match in box_pattern.finditer(text):
                if match.span(1) in consumed_box_spans:
                    continue
                for bbox in self._parse_boxes(match.group(1)):
                    detections.append({"label": "", "bbox": bbox, "confidence": 1.0})

        return detections

    def parse_points(self, text: str) -> List[Dict[str, Any]]:
        """Parse VP point spans into ``{"label": str, "point": [x, y]}`` dicts."""

        if not text:
            return []

        points: List[Dict[str, Any]] = []
        for markers in self._marker_sets:
            ref_point_pattern = re.compile(
                rf"{re.escape(markers['ref_open'])}(.*?){re.escape(markers['ref_close'])}.*?"
                rf"{re.escape(markers['point_open'])}(.*?){re.escape(markers['point_close'])}",
                flags=re.DOTALL,
            )
            for match in ref_point_pattern.finditer(text):
                label = self._clean_label(match.group(1))
                for point in self._parse_points(match.group(2)):
                    points.append({"label": label, "point": point})

        if not points:
            for markers in self._marker_sets:
                point_pattern = re.compile(
                    rf"{re.escape(markers['point_open'])}(.*?){re.escape(markers['point_close'])}",
                    flags=re.DOTALL,
                )
                for match in point_pattern.finditer(text):
                    for point in self._parse_points(match.group(1)):
                        points.append({"label": "", "point": point})

        return points

    def _parse_boxes(self, payload: str) -> List[List[int]]:
        parsed = self._parse_json_payload(payload)
        boxes = self._coerce_nested_number_list(parsed, width=4)
        if not boxes:
            boxes = self._parse_loc_token_boxes(payload)
        return [
            box for box in boxes
            if validate_normalized_bbox(box, coord_max=self.coord_max)
        ]

    @staticmethod
    def _parse_loc_token_boxes(payload: str) -> List[List[int]]:
        values = [int(value) for value in re.findall(r"<loc_(\d+)>", payload)]
        return [
            values[i:i + 4]
            for i in range(0, len(values) - 3, 4)
        ]

    def _parse_points(self, payload: str) -> List[List[int]]:
        parsed = self._parse_json_payload(payload)
        points = self._coerce_nested_number_list(parsed, width=2)
        return [
            point for point in points
            if all(0 <= value <= self.coord_max for value in point)
        ]

    def _parse_json_payload(self, payload: str) -> Any:
        payload = payload.strip()
        if not payload:
            return []
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            # Tolerate accidental Python tuple/list formatting by extracting
            # bracketed numeric rows.
            rows = re.findall(r"\[([^\[\]]+)\]", payload)
            parsed_rows = []
            for row in rows:
                values = re.findall(r"-?\d+(?:\.\d+)?", row)
                if values:
                    parsed_rows.append([float(value) for value in values])
            return parsed_rows

    def _coerce_nested_number_list(self, value: Any, *, width: int) -> List[List[int]]:
        if not isinstance(value, list):
            return []
        if value and all(isinstance(item, (int, float)) for item in value):
            value = [value]

        coerced: List[List[int]] = []
        for item in value:
            if not isinstance(item, list) or len(item) != width:
                continue
            if not all(isinstance(coord, (int, float)) for coord in item):
                continue
            coerced.append([int(round(coord)) for coord in item])
        return coerced

    @staticmethod
    def _clean_label(label: str) -> str:
        return re.sub(r"\s+", " ", label).strip()


def parse_visual_primitive_detections(text: str) -> List[Dict[str, Any]]:
    """Convenience wrapper for parsing VP detections."""

    return VisualPrimitiveParser().parse_detections(text)


def parse_visual_primitive_points(text: str) -> List[Dict[str, Any]]:
    """Convenience wrapper for parsing VP points."""

    return VisualPrimitiveParser().parse_points(text)
