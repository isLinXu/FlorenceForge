"""Visual primitive helpers for FlorenceForge.

This module keeps the VP format intentionally small and deterministic so it can
be shared by converters, parsers, metrics, and tests.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple


COORDINATE_MIN = 0
COORDINATE_MAX = 999
COORDINATE_SPACE = (COORDINATE_MIN, COORDINATE_MAX)

VISUAL_PRIMITIVE_TOKENS = {
    "ref_open": "<|ref|>",
    "ref_close": "<|/ref|>",
    "box_open": "<|box|>",
    "box_close": "<|/box|>",
    "point_open": "<|point|>",
    "point_close": "<|/point|>",
}

VISUAL_PRIMITIVE_PLAIN_TOKENS = {
    "ref_open": "<ref>",
    "ref_close": "</ref>",
    "box_open": "<box>",
    "box_close": "</box>",
    "point_open": "<point>",
    "point_close": "</point>",
}

VISUAL_PRIMITIVE_MARKER_SETS = {
    "special": VISUAL_PRIMITIVE_TOKENS,
    "plain": VISUAL_PRIMITIVE_PLAIN_TOKENS,
}

VISUAL_PRIMITIVE_SPECIAL_TOKENS = tuple(VISUAL_PRIMITIVE_TOKENS.values())

_BOX_SPAN_PATTERNS = (
    re.compile(r"<\|box\|>(.*?)<\|/box\|>", re.DOTALL),
    re.compile(r"<box>(.*?)</box>", re.DOTALL),
)
_POINT_SPAN_PATTERNS = (
    re.compile(r"<\|point\|>(.*?)<\|/point\|>", re.DOTALL),
    re.compile(r"<point>(.*?)</point>", re.DOTALL),
)
_BOX_COORD_PATTERN = re.compile(
    r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]"
)
_POINT_COORD_PATTERN = re.compile(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")
_LOC_VALUE_PATTERN = re.compile(r"<loc_(\d+)>")


@dataclass(frozen=True)
class VisualPrimitiveBox:
    """A normalized visual primitive box in ``[0, 999]`` coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    confidence: float = 1.0

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_detection(self) -> dict:
        return {
            "label": self.label,
            "bbox": self.as_list(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class VisualPrimitivePoint:
    """A normalized visual primitive point in ``[0, 999]`` coordinates."""

    x: int
    y: int
    label: str = ""

    def as_list(self) -> List[int]:
        return [self.x, self.y]


def clamp_coordinate(value: float, coord_max: int = COORDINATE_MAX) -> int:
    """Clamp and round a coordinate into the visual primitive integer space."""

    return int(max(COORDINATE_MIN, min(coord_max, round(value))))


def normalize_coordinate(value: float, size: float, coord_max: int = COORDINATE_MAX) -> int:
    """Normalize one image coordinate into ``[0, coord_max]``."""

    if size <= 0:
        raise ValueError("Image dimension must be positive")
    return clamp_coordinate((float(value) / float(size)) * coord_max, coord_max=coord_max)


def denormalize_coordinate(value: float, size: float, coord_max: int = COORDINATE_MAX) -> float:
    """Convert one normalized VP coordinate back to image coordinates."""

    if size <= 0:
        raise ValueError("Image dimension must be positive")
    return (float(value) / float(coord_max)) * float(size)


def normalize_bbox(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    *,
    input_format: str = "xyxy",
    coord_max: int = COORDINATE_MAX,
) -> List[int]:
    """Normalize a bbox into ``[x1, y1, x2, y2]`` VP coordinates.

    Args:
        bbox: Input bbox.
        image_size: ``(width, height)``.
        input_format: Either ``"xyxy"`` or ``"xywh"``.
        coord_max: Maximum VP coordinate value.
    """

    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")

    if input_format == "xyxy":
        x1, y1, x2, y2 = bbox
    elif input_format == "xywh":
        x1, y1, box_w, box_h = bbox
        x2 = x1 + box_w
        y2 = y1 + box_h
    else:
        raise ValueError("input_format must be 'xyxy' or 'xywh'")

    x1_n = normalize_coordinate(x1, width, coord_max)
    y1_n = normalize_coordinate(y1, height, coord_max)
    x2_n = normalize_coordinate(x2, width, coord_max)
    y2_n = normalize_coordinate(y2, height, coord_max)

    x_low, x_high = sorted((x1_n, x2_n))
    y_low, y_high = sorted((y1_n, y2_n))
    return [x_low, y_low, x_high, y_high]


def denormalize_bbox(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    *,
    coord_max: int = COORDINATE_MAX,
) -> List[float]:
    """Convert a normalized VP bbox back to image ``xyxy`` coordinates."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        denormalize_coordinate(x1, width, coord_max),
        denormalize_coordinate(y1, height, coord_max),
        denormalize_coordinate(x2, width, coord_max),
        denormalize_coordinate(y2, height, coord_max),
    ]


def validate_normalized_bbox(bbox: Sequence[int], coord_max: int = COORDINATE_MAX) -> bool:
    """Return whether a normalized bbox is well-formed."""

    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return (
        COORDINATE_MIN <= x1 <= coord_max
        and COORDINATE_MIN <= y1 <= coord_max
        and COORDINATE_MIN <= x2 <= coord_max
        and COORDINATE_MIN <= y2 <= coord_max
        and x1 <= x2
        and y1 <= y2
    )


def get_visual_primitive_tokens(marker_style: str = "special") -> dict:
    """Return marker tokens for a VP wrapper style."""

    try:
        return VISUAL_PRIMITIVE_MARKER_SETS[marker_style]
    except KeyError as exc:
        raise ValueError("marker_style must be 'special' or 'plain'") from exc


def format_ref(label: str, *, marker_style: str = "special") -> str:
    """Format a VP reference span."""

    tokens = get_visual_primitive_tokens(marker_style)
    return f"{tokens['ref_open']}{label}{tokens['ref_close']}"


def format_box(boxes: Iterable[Sequence[int]], *, marker_style: str = "special") -> str:
    """Format one or more normalized boxes as a VP box span."""

    box_list = [list(map(int, box)) for box in boxes]
    payload = json.dumps(box_list, separators=(",", ":"))
    tokens = get_visual_primitive_tokens(marker_style)
    return f"{tokens['box_open']}{payload}{tokens['box_close']}"


def format_loc_tokens(boxes: Iterable[Sequence[int]]) -> str:
    """Format normalized boxes as Florence-style ``<loc_*>`` tokens."""

    chunks = []
    for box in boxes:
        coords = [clamp_coordinate(value) for value in box]
        if len(coords) != 4:
            raise ValueError("Each box must contain exactly four coordinates")
        chunks.append("".join(f"<loc_{coord}>" for coord in coords))
    return "".join(chunks)


def format_box_loc_tokens(boxes: Iterable[Sequence[int]], *, marker_style: str = "special") -> str:
    """Format one or more normalized boxes as a VP box span with loc tokens."""

    payload = format_loc_tokens(boxes)
    tokens = get_visual_primitive_tokens(marker_style)
    return f"{tokens['box_open']}{payload}{tokens['box_close']}"


def format_point(points: Iterable[Sequence[int]], *, marker_style: str = "special") -> str:
    """Format one or more normalized points as a VP point span."""

    point_list = [list(map(int, point)) for point in points]
    payload = json.dumps(point_list, separators=(",", ":"))
    tokens = get_visual_primitive_tokens(marker_style)
    return f"{tokens['point_open']}{payload}{tokens['point_close']}"


def format_ref_box(
    label: str,
    boxes: Iterable[Sequence[int]],
    *,
    marker_style: str = "special",
) -> str:
    """Format ``<|ref|>label<|/ref|><|box|>...<|/box|>``."""

    separator = " " if marker_style == "plain" else ""
    return (
        f"{format_ref(label, marker_style=marker_style)}"
        f"{separator}"
        f"{format_box(boxes, marker_style=marker_style)}"
    )


@dataclass(frozen=True)
class VisualPrimitiveLine:
    """A normalized visual primitive line segment in ``[0, 999]`` coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def length(self) -> float:
        """Euclidean length in normalized coordinate space."""
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


def normalize_point(
    x: float,
    y: float,
    image_size: Tuple[int, int],
    *,
    coord_max: int = COORDINATE_MAX,
) -> Tuple[int, int]:
    """Normalize a single (x, y) point into the VP integer space."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    return (
        normalize_coordinate(x, width, coord_max),
        normalize_coordinate(y, height, coord_max),
    )


def normalize_bboxes_batch(
    bboxes: List[Sequence[float]],
    image_size: Tuple[int, int],
    *,
    input_format: str = "xyxy",
    coord_max: int = COORDINATE_MAX,
) -> List[List[int]]:
    """Normalize a list of bboxes into VP coordinates.

    Convenience wrapper around :func:`normalize_bbox` for batch processing.
    """
    return [
        normalize_bbox(bbox, image_size, input_format=input_format, coord_max=coord_max)
        for bbox in bboxes
    ]


def iou_normalized(
    box_a: Sequence[int],
    box_b: Sequence[int],
) -> float:
    """Compute IoU between two normalized VP boxes ``[x1, y1, x2, y2]``."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / max(union, 1)


def parse_vp_boxes(text: str) -> List[List[int]]:
    """Parse all VP box annotations from *text*.

    Supports both ``<|box|>...</|box|>`` and ``<box>...</box>`` styles.
    Payloads may be JSON lists, Python-like tuples, or Florence ``<loc_*>``
    groups. Returns a list of ``[x1, y1, x2, y2]`` integer lists.
    """
    boxes: List[List[int]] = []
    for payload in _iter_marker_payloads(str(text or ""), _BOX_SPAN_PATTERNS):
        boxes.extend(
            _parse_coordinate_payload(
                payload,
                expected_length=4,
                fallback_pattern=_BOX_COORD_PATTERN,
            )
        )
    return boxes


def parse_vp_points(text: str) -> List[List[int]]:
    """Parse all VP point annotations from *text*.

    Supports both ``<|point|>...</|point|>`` and ``<point>...</point>`` styles.
    Returns a list of ``[x, y]`` integer lists.
    """
    points: List[List[int]] = []
    for payload in _iter_marker_payloads(str(text or ""), _POINT_SPAN_PATTERNS):
        points.extend(
            _parse_coordinate_payload(
                payload,
                expected_length=2,
                fallback_pattern=_POINT_COORD_PATTERN,
            )
        )
    return points


def _iter_marker_payloads(text: str, patterns: Sequence[re.Pattern[str]]) -> Iterable[str]:
    for pattern in patterns:
        for match in pattern.finditer(text):
            yield match.group(1)


def _parse_coordinate_payload(
    payload: str,
    *,
    expected_length: int,
    fallback_pattern: re.Pattern[str],
) -> List[List[int]]:
    groups = _parse_json_coordinate_payload(payload, expected_length=expected_length)
    if groups:
        return _filter_valid_coordinate_groups(groups, expected_length=expected_length)

    groups = [
        [int(match.group(index)) for index in range(1, expected_length + 1)]
        for match in fallback_pattern.finditer(payload)
    ]
    if groups:
        return _filter_valid_coordinate_groups(groups, expected_length=expected_length)

    groups = _parse_loc_coordinate_groups(payload, group_size=expected_length)
    return _filter_valid_coordinate_groups(groups, expected_length=expected_length)


def _parse_json_coordinate_payload(payload: str, *, expected_length: int) -> List[List[int]]:
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return []

    groups = _coerce_coordinate_groups(decoded, expected_length=expected_length)
    return [group for group in groups if group is not None]


def _coerce_coordinate_groups(value: Any, *, expected_length: int) -> List[List[int]]:
    if _is_coordinate_sequence(value, expected_length=expected_length):
        return [[int(item) for item in value]]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    groups: List[List[int]] = []
    for item in value:
        if _is_coordinate_sequence(item, expected_length=expected_length):
            groups.append([int(coord) for coord in item])
    return groups


def _is_coordinate_sequence(value: Any, *, expected_length: int) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != expected_length:
        return False
    return all(_is_int_like(item) for item in value)


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        return value.strip().isdigit()
    return False


def _parse_loc_coordinate_groups(payload: str, *, group_size: int) -> List[List[int]]:
    values = [int(value) for value in _LOC_VALUE_PATTERN.findall(payload)]
    return [
        values[index:index + group_size]
        for index in range(0, len(values) - group_size + 1, group_size)
    ]


def _filter_valid_coordinate_groups(
    groups: Iterable[Sequence[int]],
    *,
    expected_length: int,
    coord_max: int = COORDINATE_MAX,
) -> List[List[int]]:
    return [
        list(group)
        for group in groups
        if _is_valid_coordinate_group(
            group,
            expected_length=expected_length,
            coord_max=coord_max,
        )
    ]


def _is_valid_coordinate_group(
    group: Sequence[int],
    *,
    expected_length: int,
    coord_max: int,
) -> bool:
    if len(group) != expected_length:
        return False
    if not all(COORDINATE_MIN <= int(value) <= coord_max for value in group):
        return False
    if expected_length == 4:
        x1, y1, x2, y2 = [int(value) for value in group]
        return x1 <= x2 and y1 <= y2
    return True


def resolve_marker_style(marker_style: str) -> str:
    """Normalize CLI / config marker style names to supported values."""

    normalized = str(marker_style or "special").strip().lower()
    aliases = {
        "angle_bracket": "plain",
        "angle-bracket": "plain",
        "plain": "plain",
        "special": "special",
    }
    if normalized not in aliases:
        raise ValueError("marker_style must be 'special' or 'plain'")
    return aliases[normalized]


def sort_boxes_left_to_right(boxes: Iterable[Sequence[int]]) -> List[List[int]]:
    """Sort normalized boxes from left to right, then top to bottom (TVP convention)."""

    return sorted(
        [list(map(int, box)) for box in boxes],
        key=lambda box: (box[0], box[1]),
    )


def format_ref_box_loc_tokens(
    label: str,
    boxes: Iterable[Sequence[int]],
    *,
    marker_style: str = "special",
) -> str:
    """Format ``ref + box`` using Florence native location tokens."""

    separator = " " if marker_style == "plain" else ""
    return (
        f"{format_ref(label, marker_style=marker_style)}"
        f"{separator}"
        f"{format_box_loc_tokens(boxes, marker_style=marker_style)}"
    )
