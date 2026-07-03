"""Reward model implementations (split from legacy monolith)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


from ...core.visual_primitives import COORDINATE_MAX

class FormatRewardModel:
    """Rule-based reward model checking visual primitive format.

    Checks:
      - Valid box content (numeric coordinates in [0, 999])
      - Valid point content
      - Duplicate detection penalties
      - Empty output penalties

    Output: score in [0, 1]
    """

    def __init__(self, coord_max: int = COORDINATE_MAX):
        self.coord_max = coord_max
        self._box_patterns = [
            re.compile(r"<\|box\|>(.*?)<\|/box\|>", re.DOTALL),
            re.compile(r"<box>(.*?)</box>", re.DOTALL),
        ]
        self._point_patterns = [
            re.compile(r"<\|point\|>(.*?)<\|/point\|>", re.DOTALL),
            re.compile(r"<point>(.*?)</point>", re.DOTALL),
        ]

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        score = 1.0

        # Check box format validity + coordinate range
        for pattern in self._box_patterns:
            for m in pattern.finditer(text):
                content = m.group(1).strip()
                if content:
                    box_match = re.search(r"\[(\d+),(\d+),(\d+),(\d+)\]", content)
                    if not box_match:
                        score -= 0.3
                    else:
                        x1, y1, x2, y2 = [int(box_match.group(i)) for i in range(1, 5)]
                        # Penalize out-of-range coordinates
                        if any(v > self.coord_max for v in (x1, y1, x2, y2)):
                            score -= 0.2
                        # Penalize degenerate boxes (x1 > x2 or y1 > y2)
                        if x1 > x2 or y1 > y2:
                            score -= 0.15

        # Check point format validity + coordinate range
        for pattern in self._point_patterns:
            for m in pattern.finditer(text):
                content = m.group(1).strip()
                if content:
                    pt_match = re.search(r"\[(\d+),(\d+)\]", content)
                    if not pt_match:
                        score -= 0.3
                    else:
                        px, py = int(pt_match.group(1)), int(pt_match.group(2))
                        if px > self.coord_max or py > self.coord_max:
                            score -= 0.2

        # Penalize duplicate boxes
        boxes = self._parse_all_boxes(text)
        if len(boxes) != len(set(boxes)):
            score -= 0.2

        # Penalize empty output
        if not boxes and not self._parse_all_points(text):
            score -= 0.3

        return max(0.0, score)

    def _parse_all_boxes(self, text: str) -> List[Tuple[int, ...]]:
        boxes = []
        for pattern in self._box_patterns:
            for m in pattern.finditer(text):
                for bm in re.finditer(r"\[(\d+),(\d+),(\d+),(\d+)\]", m.group(1)):
                    boxes.append(tuple(int(bm.group(i)) for i in range(1, 5)))
        return boxes

    def _parse_all_points(self, text: str) -> List[Tuple[int, ...]]:
        points = []
        for pattern in self._point_patterns:
            for m in pattern.finditer(text):
                for pm in re.finditer(r"\[(\d+),(\d+)\]", m.group(1)):
                    points.append((int(pm.group(1)), int(pm.group(2))))
        return points


