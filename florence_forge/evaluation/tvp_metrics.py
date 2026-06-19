"""TVP-specific evaluation metrics for Visual Primitive reasoning.

Provides metrics beyond standard AP/mAP for evaluating
thinking-with-visual-primitive capabilities:

  - TrajectorySimilarityMetric: bidirectional trajectory distance
  - MazeNavigationMetric: maze solving accuracy with path analysis
  - PathTracingMetric: path tracing fidelity
  - ChainOfThoughtMetric: structural quality of CoT traces
  - TVPCompositeMetric: composite metric combining all above
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np



class TrajectorySimilarityMetric:
    """Bidirectional trajectory similarity metric.

    Computes:
      - Forward similarity: for each predicted point, min distance to GT polyline
      - Reverse similarity: for each GT point, min distance to predicted polyline
      - Combined: harmonic mean of forward and reverse scores
    """

    def __init__(self, tolerance: float = 100.0):
        self.tolerance = tolerance

    def compute(
        self,
        pred_points: List[Tuple[int, int]],
        gt_points: List[Tuple[int, int]],
    ) -> Dict[str, float]:
        if not pred_points or not gt_points:
            return {"forward_sim": 0.0, "reverse_sim": 0.0, "combined_sim": 0.0}

        forward_dists = [
            self._point_to_polyline_distance(p, gt_points)
            for p in pred_points
        ]
        forward_sim = max(0.0, 1.0 - (sum(forward_dists) / len(forward_dists)) / self.tolerance)

        reverse_dists = [
            self._point_to_polyline_distance(g, pred_points)
            for g in gt_points
        ]
        reverse_sim = max(0.0, 1.0 - (sum(reverse_dists) / len(reverse_dists)) / self.tolerance)

        combined = 2 * forward_sim * reverse_sim / max(forward_sim + reverse_sim, 1e-8)
        return {
            "forward_sim": forward_sim,
            "reverse_sim": reverse_sim,
            "combined_sim": combined,
        }

    @staticmethod
    def _point_to_segment_distance(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    @classmethod
    def _point_to_polyline_distance(cls, p, polyline):
        if len(polyline) < 2:
            if polyline:
                return math.hypot(p[0] - polyline[0][0], p[1] - polyline[0][1])
            return float("inf")
        return min(
            cls._point_to_segment_distance(p, polyline[i], polyline[i + 1])
            for i in range(len(polyline) - 1)
        )


class MazeNavigationMetric:
    """Maze navigation evaluation metric.

    Components:
      - answer_accuracy: correct solvable/unsolvable classification
      - path_validity: connectivity of proposed path
      - wall_violation_rate: fraction of path segments crossing walls
      - exploration_coverage: fraction of reachable cells explored
    """

    def compute(
        self,
        pred_text: str,
        solvable: bool,
        grid: Optional[np.ndarray] = None,
        gt_path: Optional[List[Tuple[int, int]]] = None,
        grid_height: int = 1,
        grid_width: int = 1,
    ) -> Dict[str, float]:
        # Answer accuracy
        final = pred_text.split("</think>")[-1] if "</think>" in pred_text else pred_text
        final = final.strip().lower()
        pred_solvable = "true" in final
        answer_acc = 1.0 if pred_solvable == solvable else 0.0

        # Parse predicted points
        pred_points = self._parse_points(pred_text)

        # Path validity
        path_validity = 0.0
        if pred_points and len(pred_points) >= 2 and grid is not None:
            cells = self._points_to_cells(pred_points, grid_height, grid_width)
            valid = 0
            total = len(cells) - 1
            for i in range(total):
                r1, c1 = cells[i]
                r2, c2 = cells[i + 1]
                if abs(r1 - r2) + abs(c1 - c2) <= 1:
                    mid_r = (2 * r1 + 2 * r2) // 2
                    mid_c = (2 * c1 + 2 * c2) // 2
                    if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]
                            and grid[mid_r, mid_c] == 1):
                        valid += 1
            path_validity = valid / max(total, 1)

        # Wall violation rate
        wall_violation = 0.0
        if pred_points and grid is not None:
            cells = self._points_to_cells(pred_points, grid_height, grid_width)
            violations = 0
            for i in range(len(cells) - 1):
                r1, c1 = cells[i]
                r2, c2 = cells[i + 1]
                mid_r = (2 * r1 + 2 * r2) // 2
                mid_c = (2 * c1 + 2 * c2) // 2
                if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]):
                    if grid[mid_r, mid_c] == 0:
                        violations += 1
                else:
                    violations += 1
            wall_violation = violations / max(len(cells) - 1, 1)

        # Exploration coverage
        coverage = 0.0
        if pred_points and grid is not None:
            cells = set(self._points_to_cells(pred_points, grid_height, grid_width))
            reachable = sum(
                1 for r in range(grid_height) for c in range(grid_width)
                if 2 * r < grid.shape[0] and 2 * c < grid.shape[1]
                and grid[2 * r, 2 * c] == 1
            )
            explored = sum(
                1 for r, c in cells
                if 2 * r < grid.shape[0] and 2 * c < grid.shape[1]
                and grid[2 * r, 2 * c] == 1
            )
            coverage = explored / max(reachable, 1)

        return {
            "answer_accuracy": answer_acc,
            "path_validity": path_validity,
            "wall_violation_rate": wall_violation,
            "exploration_coverage": coverage,
        }

    @staticmethod
    def _parse_points(text: str) -> List[Tuple[int, int]]:
        points = []
        for pattern in [
            re.compile(r"<\|point\|>(.*?)<\|/point\|>", re.DOTALL),
            re.compile(r"<point>(.*?)</point>", re.DOTALL),
        ]:
            for m in pattern.finditer(text):
                for pm in re.finditer(r"\[(\d+),(\d+)\]", m.group(1)):
                    points.append((int(pm.group(1)), int(pm.group(2))))
        return points

    @staticmethod
    def _points_to_cells(points, grid_h, grid_w):
        cells = []
        for x, y in points:
            col = int(round(x / 999 * (grid_w - 1))) if grid_w > 1 else 0
            row = int(round(y / 999 * (grid_h - 1))) if grid_h > 1 else 0
            cells.append((row, col))
        return cells


class PathTracingMetric:
    """Path tracing evaluation metric.

    Components:
      - endpoint_accuracy: distance between predicted and GT endpoints
      - trajectory_similarity: bidirectional trajectory distance
      - continuity_score: penalizes large jumps in trajectory
    """

    def __init__(self, endpoint_tolerance: float = 50.0):
        self.endpoint_tolerance = endpoint_tolerance
        self.traj_metric = TrajectorySimilarityMetric()

    def compute(
        self,
        pred_points: List[Tuple[int, int]],
        gt_points: List[Tuple[int, int]],
        end_label_pred: str = "",
        end_label_gt: str = "",
    ) -> Dict[str, float]:
        # Endpoint accuracy
        endpoint_acc = 0.0
        if pred_points and gt_points:
            dist = math.hypot(
                pred_points[-1][0] - gt_points[-1][0],
                pred_points[-1][1] - gt_points[-1][1],
            )
            endpoint_acc = max(0.0, 1.0 - dist / self.endpoint_tolerance)

        # Label accuracy
        label_acc = 1.0 if end_label_pred.lower() == end_label_gt.lower() else 0.0

        # Trajectory similarity
        traj_scores = self.traj_metric.compute(pred_points, gt_points)

        # Continuity score
        continuity = 1.0
        if len(pred_points) >= 2:
            jumps = sum(
                1 for i in range(1, len(pred_points))
                if math.hypot(
                    pred_points[i][0] - pred_points[i-1][0],
                    pred_points[i][1] - pred_points[i-1][1],
                ) > 100
            )
            continuity = max(0.0, 1.0 - jumps * 0.2)

        return {
            "endpoint_accuracy": endpoint_acc,
            "label_accuracy": label_acc,
            "trajectory_similarity": traj_scores["combined_sim"],
            "continuity_score": continuity,
        }


class ChainOfThoughtMetric:
    """Structural quality metric for chain-of-thought traces.

    Evaluates:
      - step_presence: whether expected CoT steps are present
      - step_ordering: whether steps appear in correct order
      - grounding_consistency: whether VP markers align with text claims
      - conciseness: penalizes overly verbose traces
    """

    EXPECTED_STEPS = {
        "counting": ["Deconstructing the query", "Sweeping the scene", "Tallying"],
        "counting_fine": ["What am I looking for", "Evaluating each", "Tally"],
        "maze": ["Exploration", "Answer"],
        "path": ["starting point", "visual path", "Answer"],
        "spatial": ["Analyzing the request", "Reasoning", "Conclusion"],
    }

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        normalized = str(task_type or "counting").strip().lower().replace("-", "_")
        aliases = {
            "count_vp_cot": "counting",
            "count_vp": "counting",
            "od_vp": "counting",
            "phrase_grounding_vp": "counting",
            "spatial_vp": "spatial",
            "maze_vp": "maze",
            "path_vp": "path",
        }
        return aliases.get(normalized, normalized)

    def compute(
        self,
        text: str,
        task_type: str = "counting",
    ) -> Dict[str, float]:
        task_type = self._normalize_task_type(task_type)
        expected = self.EXPECTED_STEPS.get(task_type, [])

        # Step presence
        step_present = [step in text for step in expected]
        presence_score = sum(step_present) / max(len(expected), 1)

        # Step ordering
        ordering_score = 1.0
        positions = []
        for step in expected:
            pos = text.find(step)
            if pos >= 0:
                positions.append(pos)
        if len(positions) >= 2:
            inversions = sum(
                1 for i in range(len(positions))
                for j in range(i + 1, len(positions))
                if positions[i] > positions[j]
            )
            max_inv = len(positions) * (len(positions) - 1) / 2
            ordering_score = 1.0 - inversions / max(max_inv, 1)

        # Grounding consistency: check if VP markers exist when expected
        has_box = bool(re.search(r"<\|box\|>|<box>", text))
        has_point = bool(re.search(r"<\|point\|>|<point>", text))
        needs_box = task_type in ("counting", "od_vp", "phrase_grounding_vp")
        needs_point = task_type in ("maze", "path")

        grounding_score = 1.0
        if needs_box and not has_box:
            grounding_score -= 0.5
        if needs_point and not has_point:
            grounding_score -= 0.5
        grounding_score = max(0.0, grounding_score)

        # Conciseness
        conciseness = 1.0
        if len(text) > 3000:
            conciseness = max(0.0, 1.0 - (len(text) - 3000) / 3000)

        return {
            "step_presence": presence_score,
            "step_ordering": ordering_score,
            "grounding_consistency": grounding_score,
            "conciseness": conciseness,
        }


class CountingDetectionMetric:
    """Counting and detection accuracy metric for VP-grounded outputs.

    Evaluates:
      - count_accuracy: exp(-|pred_count - gt_count| / (gt_count + 1))
      - box_recall: fraction of GT boxes matched by predicted boxes (IoU >= threshold)
      - box_precision: fraction of predicted boxes matching GT boxes
      - count_f1: harmonic mean of recall and precision
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute(
        self,
        pred_text: str,
        gt_count: Optional[int] = None,
        gt_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        pred_boxes = self._parse_boxes(pred_text)

        # Count accuracy
        pred_count = len(pred_boxes)
        if gt_count is not None:
            count_acc = math.exp(
                -abs(pred_count - gt_count) / (abs(gt_count) + 1)
            )
        elif gt_boxes is not None:
            count_acc = math.exp(
                -abs(pred_count - len(gt_boxes)) / (len(gt_boxes) + 1)
            )
        else:
            count_acc = 1.0 if pred_count > 0 else 0.0

        if not gt_boxes:
            return {
                "count_accuracy": count_acc,
                "box_recall": 0.0,
                "box_precision": 1.0 if not pred_boxes else 0.0,
                "count_f1": 0.0,
            }

        # Box matching
        matched_gt = set()
        matched_pred = set()
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if self._iou(pred, gt) >= self.iou_threshold:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        recall = len(matched_gt) / max(len(gt_boxes), 1)
        precision = len(matched_pred) / max(len(pred_boxes), 1) if pred_boxes else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "count_accuracy": count_acc,
            "box_recall": recall,
            "box_precision": precision,
            "count_f1": f1,
        }

    @staticmethod
    def _parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
        boxes = []
        for pattern in [
            re.compile(r"<\|box\|>(.*?)<\|/box\|>", re.DOTALL),
            re.compile(r"<box>(.*?)</box>", re.DOTALL),
        ]:
            for m in pattern.finditer(text):
                for bm in re.finditer(r"\[(\d+),(\d+),(\d+),(\d+)\]", m.group(1)):
                    boxes.append(tuple(int(bm.group(i)) for i in range(1, 5)))
        return boxes

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1)


class TVPCompositeMetric:
    """Composite metric combining all TVP evaluation aspects.

    Aggregates task-specific metrics with CoT structural quality
    into a single composite score.
    """

    def __init__(self, endpoint_tolerance: float = 50.0):
        self.traj_metric = TrajectorySimilarityMetric()
        self.maze_metric = MazeNavigationMetric()
        self.path_metric = PathTracingMetric(endpoint_tolerance)
        self.cot_metric = ChainOfThoughtMetric()
        self.counting_metric = CountingDetectionMetric()

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        normalized = str(task_type or "counting").strip().lower().replace("-", "_")
        aliases = {
            "count_vp_cot": "counting",
            "count_vp": "counting",
            "od_vp": "counting",
            "phrase_grounding_vp": "counting",
            "spatial_vp": "spatial",
            "maze_vp": "maze",
            "path_vp": "path",
        }
        return aliases.get(normalized, normalized)

    def compute(
        self,
        pred_text: str,
        task_type: str = "counting",
        **kwargs,
    ) -> Dict[str, float]:
        """Compute composite metric.

        Args:
            pred_text: Model prediction text.
            task_type: Task type for metric selection.
            **kwargs: Additional task-specific arguments
                      (gt_points, solvable, grid, etc.)

        Returns:
            Dictionary of metric scores.
        """
        task_type = self._normalize_task_type(task_type)
        results: Dict[str, float] = {}

        # CoT structural quality (always applicable)
        cot_scores = self.cot_metric.compute(pred_text, task_type)
        results.update({f"cot_{k}": v for k, v in cot_scores.items()})

        # Task-specific metrics
        if task_type == "maze":
            maze_scores = self.maze_metric.compute(
                pred_text,
                solvable=kwargs.get("solvable", True),
                grid=kwargs.get("grid"),
                grid_height=kwargs.get("grid_height", 1),
                grid_width=kwargs.get("grid_width", 1),
            )
            results.update(maze_scores)

        elif task_type == "path":
            pred_points = self._parse_points(pred_text)
            gt_points = kwargs.get("gt_points", [])
            path_scores = self.path_metric.compute(
                pred_points, gt_points,
                end_label_pred=kwargs.get("pred_label", ""),
                end_label_gt=kwargs.get("gt_label", ""),
            )
            results.update(path_scores)

        elif task_type == "spatial":
            gt_answer = str(kwargs.get("gt_answer", "")).strip().lower()
            if gt_answer:
                pred_lower = pred_text.lower()
                results["answer_accuracy"] = (
                    1.0 if gt_answer in pred_lower or pred_lower.endswith(gt_answer) else 0.0
                )

        elif task_type in ("counting", "od_vp", "phrase_grounding_vp"):
            count_scores = self.counting_metric.compute(pred_text, **kwargs)
            results.update(count_scores)

        # Composite score
        metric_values = [v for v in results.values() if 0.0 <= v <= 1.0]
        if metric_values:
            results["composite"] = sum(metric_values) / len(metric_values)
        else:
            results["composite"] = 0.0

        return results

    @staticmethod
    def _parse_points(text: str) -> List[Tuple[int, int]]:
        points = []
        for pattern in [
            re.compile(r"<\|point\|>(.*?)<\|/point\|>", re.DOTALL),
            re.compile(r"<point>(.*?)</point>", re.DOTALL),
        ]:
            for m in pattern.finditer(text):
                for pm in re.finditer(r"\[(\d+),(\d+)\]", m.group(1)):
                    points.append((int(pm.group(1)), int(pm.group(2))))
        return points
