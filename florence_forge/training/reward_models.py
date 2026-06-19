"""Reward models for GRPO / RL training of Visual Primitive reasoning.

Three types of reward models (following the TVP paper):
  1. Format RM (rule-based): checks syntax, duplicates, etc.
  2. Quality RM (LLM-based GRM): evaluates thinking consistency, redundancy, etc.
  3. Accuracy RM (task-specific): checks correctness against ground truth.

All reward models operate on the FlorenceForge VP coordinate space [0, 999]
and are compatible with both special and plain marker styles.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..core.visual_primitives import (
    COORDINATE_MAX,
)


# ---------------------------------------------------------------------------
# 1. Format Reward Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Quality Reward Model (LLM-based GRM)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Accuracy Reward Models (Task-specific)
# ---------------------------------------------------------------------------

class DetectionAccuracyRewardModel:
    """Accuracy reward for object detection / grounding tasks.

    Computes IoU between predicted and ground-truth bounding boxes,
    averaged over matched pairs.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def __call__(self, text: str, metadata: Dict) -> float:
        gt_boxes = metadata.get("gt_boxes", [])
        metadata.get("gt_labels", [])

        pred_boxes = self._parse_boxes(text)
        if not gt_boxes:
            return 0.5  # neutral if no GT

        matched = 0
        for gt_box in gt_boxes:
            best_iou = 0.0
            for pred_box in pred_boxes:
                iou = self._compute_iou(gt_box, pred_box)
                best_iou = max(best_iou, iou)
            if best_iou >= self.iou_threshold:
                matched += 1

        recall = matched / max(len(gt_boxes), 1)
        precision = matched / max(len(pred_boxes), 1) if pred_boxes else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return f1

    @staticmethod
    def _parse_boxes(text: str) -> List[List[int]]:
        boxes = []
        for pattern in [
            re.compile(r"<\|box\|>(.*?)<\|/box\|>", re.DOTALL),
            re.compile(r"<box>(.*?)</box>", re.DOTALL),
        ]:
            for m in pattern.finditer(text):
                for bm in re.finditer(r"\[(\d+),(\d+),(\d+),(\d+)\]", m.group(1)):
                    boxes.append([int(bm.group(i)) for i in range(1, 5)])
        return boxes

    @staticmethod
    def _compute_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        return intersection / max(union, 1)


class CountingRewardModel:
    """Accuracy reward for counting tasks.

    R = α * exp(-β * |ŷ - y| / (|y| + 1))

    When the text contains agentic meta-cognitive tokens, the count is
    extracted from the ``<DECIDE>`` phase first (to avoid VP coordinate
    pollution). Falls back to the last standalone number otherwise.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 3.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, text: str, metadata: Dict) -> float:
        gt = metadata.get("count", None)
        if gt is None:
            return 0.0

        # Try extracting from <DECIDE> phase first (avoids VP coord pollution)
        decide_match = re.search(r"<DECIDE>(.*?)</DECIDE>", text, re.DOTALL)
        if decide_match:
            decide_text = decide_match.group(1)
            decide_numbers = re.findall(r"\b\d+\b", decide_text)
            if decide_numbers:
                pred = int(decide_numbers[-1])
                error = abs(pred - gt)
                return self.alpha * math.exp(-self.beta * error / (abs(gt) + 1))

        # Fallback: extract from text after </think> or full text
        final = text.split("</think>")[-1] if "</think>" in text else text
        numbers = re.findall(r"\b\d+\b", final)
        if not numbers:
            return 0.0
        pred = int(numbers[-1])
        error = abs(pred - gt)
        reward = self.alpha * math.exp(-self.beta * error / (abs(gt) + 1))
        return reward


class SpatialReasoningRewardModel:
    """Accuracy reward for spatial reasoning / VQA tasks."""

    def __call__(self, text: str, metadata: Dict) -> float:
        gt_answer = metadata.get("answer", "").strip().lower()
        if not gt_answer:
            return 0.5

        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()

        if gt_answer in final or final in gt_answer:
            return 1.0
        if gt_answer in ["true", "false"]:
            pred = "true" if "true" in final else "false" if "false" in final else ""
            return 1.0 if pred == gt_answer else 0.0
        return 0.0


class MazeRewardModel:
    """Accuracy reward for maze navigation tasks.

    Components (per TVP paper Section 2.5.2):
      - causal exploration progress (solvable only)
      - exploration completeness (unsolvable only)
      - wall violation penalty
      - final path validity
      - answer correctness
    """

    def __call__(self, text: str, metadata: Dict) -> float:
        solvable = metadata.get("solvable", True)
        gt_answer = "true" if solvable else "false"
        grid = metadata.get("grid", None)
        gt_path = metadata.get("solution", [])

        # 1. Answer correctness
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        pred_answer = "true" if "true" in final else "false" if "false" in final else ""
        answer_score = 1.0 if pred_answer == gt_answer else 0.0

        # 2. Parse point sequences
        all_points = self._parse_points(text)

        # 3. Final path validity
        final_path_score = 0.0
        if all_points and len(all_points) >= 2:
            final_path_score = 1.0
            if grid is not None and len(gt_path) > 0:
                final_path_score = self._check_path_connectivity(
                    all_points, grid, metadata
                )

        # 4. Wall violation penalty
        wall_violation_score = 1.0
        if grid is not None and all_points:
            wall_violation_score = self._compute_wall_violation_score(
                all_points, grid, metadata
            )

        # 5. Exploration scoring
        exploration_score = 1.0
        if grid is not None and gt_path:
            if solvable:
                exploration_score = self._compute_exploration_progress(
                    all_points, gt_path, grid, metadata
                )
            else:
                exploration_score = self._compute_exploration_completeness(
                    all_points, grid, metadata
                )

        if solvable:
            return (0.2 * answer_score + 0.2 * exploration_score +
                    0.2 * wall_violation_score + 0.3 * final_path_score +
                    0.1 * (1.0 if pred_answer == gt_answer and final_path_score > 0 else 0.0))
        else:
            return (0.3 * answer_score + 0.3 * exploration_score +
                    0.2 * wall_violation_score + 0.2 * final_path_score)

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
    def _points_to_grid_cells(points, metadata):
        grid_h = metadata.get("grid_height", 1)
        grid_w = metadata.get("grid_width", 1)
        cells = []
        for x, y in points:
            col = int(round(x / 999 * (grid_w - 1))) if grid_w > 1 else 0
            row = int(round(y / 999 * (grid_h - 1))) if grid_h > 1 else 0
            cells.append((row, col))
        return cells

    def _check_path_connectivity(self, points, grid, metadata):
        cells = self._points_to_grid_cells(points, metadata)
        if len(cells) < 2:
            return 0.0
        valid_transitions = 0
        total_transitions = len(cells) - 1
        for i in range(total_transitions):
            r1, c1 = cells[i]
            r2, c2 = cells[i + 1]
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                gr1, gc1 = 2 * r1, 2 * c1
                gr2, gc2 = 2 * r2, 2 * c2
                mid_r, mid_c = (gr1 + gr2) // 2, (gc1 + gc2) // 2
                if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]
                        and grid[mid_r, mid_c] == 1):
                    valid_transitions += 1
        return valid_transitions / max(total_transitions, 1)

    def _compute_wall_violation_score(self, points, grid, metadata):
        cells = self._points_to_grid_cells(points, metadata)
        if len(cells) < 2:
            return 1.0
        violations = 0
        for i in range(len(cells) - 1):
            r1, c1 = cells[i]
            r2, c2 = cells[i + 1]
            gr1, gc1 = 2 * r1, 2 * c1
            gr2, gc2 = 2 * r2, 2 * c2
            mid_r, mid_c = (gr1 + gr2) // 2, (gc1 + gc2) // 2
            if (0 <= mid_r < grid.shape[0] and 0 <= mid_c < grid.shape[1]):
                if grid[mid_r, mid_c] == 0:
                    violations += 1
            else:
                violations += 1
        total_legal = grid.sum()
        return max(0.0, 1.0 - violations / max(total_legal, 1))

    def _compute_exploration_progress(self, points, gt_path, grid, metadata):
        cells = self._points_to_grid_cells(points, metadata)
        if not cells or not gt_path:
            return 0.0
        end_r, end_c = gt_path[-1]
        min_dist = float("inf")
        for r, c in cells:
            d = abs(r - end_r) + abs(c - end_c)
            min_dist = min(min_dist, d)
        path_len = len(gt_path)
        return max(0.0, 1.0 - min_dist / max(path_len, 1))

    def _compute_exploration_completeness(self, points, grid, metadata):
        cells = set(self._points_to_grid_cells(points, metadata))
        grid_h = metadata.get("grid_height", 1)
        grid_w = metadata.get("grid_width", 1)
        reachable = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if grid[2 * r, 2 * c] == 1:
                    reachable += 1
        explored = len(cells.intersection(
            {(r, c) for r in range(grid_h) for c in range(grid_w)
             if 2 * r < grid.shape[0] and 2 * c < grid.shape[1] and grid[2 * r, 2 * c] == 1}
        ))
        return explored / max(reachable, 1)


class PathTracingRewardModel:
    """Accuracy reward for path tracing tasks.

    Components:
      - bidirectional trajectory distance
      - endpoint accuracy
      - trajectory continuity penalty
      - answer correctness
    """

    def __init__(self, endpoint_tolerance: float = 50.0):
        self.endpoint_tolerance = endpoint_tolerance

    def __call__(self, text: str, metadata: Dict) -> float:
        gt_points = metadata.get("points", [])
        gt_end_label = metadata.get("end_label", "").lower()

        pred_points = self._parse_points(text)

        # Answer correctness
        final = text.split("</think>")[-1] if "</think>" in text else text
        final = final.strip().lower()
        answer_score = 1.0 if gt_end_label in final else 0.0

        # Endpoint accuracy
        endpoint_score = 0.0
        if gt_points and pred_points:
            gt_end = gt_points[-1]
            pred_end = pred_points[-1]
            dist = math.hypot(gt_end[0] - pred_end[0], gt_end[1] - pred_end[1])
            endpoint_score = max(0.0, 1.0 - dist / self.endpoint_tolerance)

        # Bidirectional trajectory distance
        traj_score = 0.0
        if gt_points and pred_points:
            forward_dists = [
                self._point_to_polyline_distance(p, gt_points)
                for p in pred_points
            ]
            forward_score = max(0.0, 1.0 - (sum(forward_dists) / len(forward_dists)) / 100.0)

            reverse_dists = [
                self._point_to_polyline_distance(g, pred_points)
                for g in gt_points
            ]
            reverse_score = max(0.0, 1.0 - (sum(reverse_dists) / len(reverse_dists)) / 100.0)

            traj_score = (forward_score + reverse_score) / 2.0

        # Continuity penalty
        continuity_score = 1.0
        if len(pred_points) >= 2:
            jumps = sum(
                1 for i in range(1, len(pred_points))
                if math.hypot(
                    pred_points[i][0] - pred_points[i-1][0],
                    pred_points[i][1] - pred_points[i-1][1],
                ) > 100
            )
            continuity_score = max(0.0, 1.0 - jumps * 0.2)

        return (0.3 * answer_score + 0.3 * endpoint_score +
                0.25 * traj_score + 0.15 * continuity_score)

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
    def _point_to_segment_distance(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

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


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class MixedAccuracyRewardModel:
    """Dispatch to the appropriate accuracy RM based on task_type metadata."""

    def __init__(self):
        self.counting_rm = CountingRewardModel()
        self.spatial_rm = SpatialReasoningRewardModel()
        self.maze_rm = MazeRewardModel()
        self.path_rm = PathTracingRewardModel()
        self.detection_rm = DetectionAccuracyRewardModel()

    def __call__(self, text: str, meta: Dict) -> float:
        subtask = str(meta.get("task_type") or meta.get("base_task") or meta.get("vp_task_type") or "").strip().lower()
        aliases = {
            "count_vp_cot": "counting",
            "count_vp": "counting",
            "spatial_vp": "spatial",
            "phrase_grounding_vp": "grounding",
            "od_vp": "od_vp",
            "maze_vp": "maze",
            "path_vp": "path",
            "agentic_count": "counting",
            "agentic_spatial": "spatial",
            "agentic_maze": "maze",
            "agentic_grounding": "grounding",
        }
        subtask = aliases.get(subtask, subtask)
        if subtask == "counting":
            return self.counting_rm(text, meta)
        elif subtask in ("spatial", "vqa"):
            return self.spatial_rm(text, meta)
        elif subtask == "maze":
            return self.maze_rm(text, meta)
        elif subtask == "path":
            return self.path_rm(text, meta)
        elif subtask in ("od", "grounding", "od_vp", "phrase_grounding_vp"):
            return self.detection_rm(text, meta)
        return 0.0


def build_reward_models(
    task_type: str = "mixed",
    judge_model: Optional[Any] = None,
    judge_tokenizer: Optional[Any] = None,
) -> List[Callable[[str, Dict], float]]:
    """Build a list of reward functions for the given task type.

    Args:
        task_type: One of "counting", "spatial", "maze", "path",
                   "od", "grounding", or "mixed".
        judge_model: Optional LLM judge for QualityRewardModel.
        judge_tokenizer: Optional tokenizer for the judge model.

    Returns:
        List of reward callables [(text, metadata) -> float].
    """
    format_rm = FormatRewardModel()
    quality_rm = QualityRewardModel(judge_model, judge_tokenizer)

    accuracy_map = {
        "counting": CountingRewardModel,
        "spatial": SpatialReasoningRewardModel,
        "maze": MazeRewardModel,
        "path": PathTracingRewardModel,
        "od": DetectionAccuracyRewardModel,
        "grounding": DetectionAccuracyRewardModel,
    }

    if task_type in accuracy_map:
        acc_rm = accuracy_map[task_type]()
    else:
        acc_rm = MixedAccuracyRewardModel()

    return [format_rm, quality_rm, acc_rm]

# ---------------------------------------------------------------------------
# 4. Agentic Reward Models (meta-cognitive token structure)
# ---------------------------------------------------------------------------

class AgenticFormatRewardModel:
    """Reward model for agentic meta-cognitive token structure.

    Checks:
      - Presence of required phases (<ACT> and <DECIDE>)
      - Proper closing of all opened tags
      - Phase ordering (PLAN before ACT, ACT before DECIDE)
      - Non-empty phase content

    Output: score in [0, 1]
    """

    def __init__(self):
        from ..core.agentic_tokens import (
            AGENTIC_PHASE_TOKENS,
            AGENTIC_PHASE_ORDER,
            REQUIRED_PHASES,
            extract_phase,
            extract_all_phases,
            has_required_phases,
            get_phase_order,
        )
        self._phase_tokens = AGENTIC_PHASE_TOKENS
        self._phase_order = AGENTIC_PHASE_ORDER
        self._required_phases = REQUIRED_PHASES
        self._extract_phase = extract_phase
        self._extract_all_phases = extract_all_phases
        self._has_required_phases = has_required_phases
        self._get_phase_order = get_phase_order

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        score = 1.0

        # 1. Must have required phases
        if not self._has_required_phases(text):
            score -= 0.5

        # 2. Check all opened tags are closed
        for phase, (open_tok, close_tok) in self._phase_tokens.items():
            open_count = text.count(open_tok)
            close_count = text.count(close_tok)
            if open_count != close_count:
                score -= 0.15 * abs(open_count - close_count)

        # 3. Check phase ordering (PLAN before ACT before DECIDE)
        phase_seq = self._get_phase_order(text)
        canonical = [p for p in self._phase_order if p in phase_seq]
        if phase_seq != canonical:
            score -= 0.2

        # 4. Check non-empty phase content
        all_phases = self._extract_all_phases(text)
        for phase, contents in all_phases.items():
            for content in contents:
                if not content.strip():
                    score -= 0.1

        # 5. Bonus for having a REFLECT phase (self-correction signal)
        if all_phases.get("reflect"):
            score += 0.1

        return max(0.0, min(1.0, score))


class AgenticQualityRewardModel:
    """Evaluate the quality of agentic reasoning content.

    Heuristic checks:
      - VERIFY content references previous ACT output
      - REFLECT content acknowledges a specific error pattern
      - DECIDE content is concise and matches task output format
      - No reward hacking (e.g., repeating VERIFY without new info)

    Falls back to LLM judge if available, otherwise uses heuristics.
    Output: score in [0, 1]
    """

    def __init__(self, judge_model: Optional[Any] = None, judge_tokenizer: Optional[Any] = None):
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        from ..core.agentic_tokens import extract_all_phases
        self._extract_all_phases = extract_all_phases

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        if self.judge_model is not None:
            return self._judge_inference(text)
        return self._heuristic_score(text)

    def _heuristic_score(self, text: str) -> float:
        phases = self._extract_all_phases(text)
        score = 1.0

        # 1. VERIFY should reference prior findings
        act_contents = phases.get("act", [])
        verify_contents = phases.get("verify", [])
        if act_contents and verify_contents:
            # Check if verify mentions numbers/coordinates from act
            act_text = " ".join(act_contents)
            verify_text = " ".join(verify_contents)
            # Simple check: verify should contain at least one number from act
            act_numbers = set(re.findall(r"\d+", act_text))
            verify_numbers = set(re.findall(r"\d+", verify_text))
            if act_numbers and not act_numbers.intersection(verify_numbers):
                score -= 0.2  # verify doesn't reference act's specifics

        # 2. REFLECT should mention error type
        reflect_contents = phases.get("reflect", [])
        if reflect_contents:
            reflect_text = " ".join(reflect_contents).lower()
            error_keywords = ["miss", "wrong", "error", "mistake", "confus",
                              "incorrect", "forgot", "skip", "overlook"]
            if not any(kw in reflect_text for kw in error_keywords):
                score -= 0.15  # reflect doesn't acknowledge an error

        # 3. DECIDE should be concise (< 200 chars typically)
        decide_contents = phases.get("decide", [])
        if decide_contents:
            decide_text = " ".join(decide_contents)
            if len(decide_text) > 300:
                score -= 0.15  # overly verbose decision

        # 4. Penalize repeated identical VERIFY content
        if len(verify_contents) > 1:
            unique_verifies = set(verify_contents)
            if len(unique_verifies) < len(verify_contents):
                score -= 0.2  # repeating verify without new info

        # 5. Penalize extreme total length
        total_len = len(text)
        if total_len > 4000:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _judge_inference(self, text: str) -> float:
        prompt = (
            "Evaluate the following agentic visual reasoning response.\n"
            "Check:\n"
            "1. Does VERIFY reference findings from ACT?\n"
            "2. Does REFLECT acknowledge a specific error?\n"
            "3. Is DECIDE concise and task-appropriate?\n"
            "4. Any reward hacking (repetitive VERIFY, empty phases)?\n\n"
            f"Response:\n{text}\n\n"
            "Score: 0.0 (poor), 0.5 (fair), or 1.0 (good). Output only the score."
        )
        try:
            inputs = self.judge_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024,
            )
            device = next(self.judge_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=self.judge_tokenizer.eos_token_id,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = self.judge_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            return QualityRewardModel._parse_judge_score(decoded)
        except Exception:
            return self._heuristic_score(text)


class AgenticSelfCorrectionRewardModel:
    """Reward model for self-correction behavior.

    Specifically rewards trajectories where:
      - An error is detected in VERIFY
      - A REFLECT phase acknowledges the error
      - The DECIDE phase contains the corrected answer

    This is the key reward signal for training self-correcting agents.
    Output: score in [0, 1]
    """

    def __init__(self):
        from ..core.agentic_tokens import extract_all_phases
        self._extract_all_phases = extract_all_phases

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> float:
        phases = self._extract_all_phases(text)
        score = 0.0

        # Check for error detection in VERIFY
        verify_text = " ".join(phases.get("verify", "")).lower()
        error_detected = any(
            kw in verify_text
            for kw in ["wrong", "error", "mistake", "incorrect", "not", "but", "wait"]
        )

        # Check for reflection on the error
        reflect_present = bool(phases.get("reflect"))
        reflect_text = " ".join(phases.get("reflect", "")).lower()
        error_acknowledged = reflect_present and any(
            kw in reflect_text
            for kw in ["miss", "wrong", "error", "mistake", "confus", "incorrect", "forgot", "skip"]
        )

        # Check if DECIDE contains a correction (different from initial ACT answer)
        act_text = " ".join(phases.get("act", ""))
        decide_text = " ".join(phases.get("decide", ""))

        if error_detected:
            score += 0.3
        if error_acknowledged:
            score += 0.3
        # Check if the decide phase differs from act (correction happened)
        if act_text and decide_text and act_text.strip() != decide_text.strip():
            score += 0.2

        # If error_injected metadata is available, check correction matches GT
        if metadata and metadata.get("error_injected"):
            # The trajectory should show self-correction
            if error_detected and error_acknowledged:
                score += 0.2
            elif not error_detected:
                # Failed to detect an injected error
                score -= 0.1

        return max(0.0, min(1.0, score))


def build_agentic_reward_models(
    judge_model: Optional[Any] = None,
    judge_tokenizer: Optional[Any] = None,
) -> List[Callable[[str, Dict], float]]:
    """Build reward function list for agentic meta-cognitive training.

    Returns [format_rm, quality_rm, self_correction_rm, accuracy_rm].
    """
    format_rm = AgenticFormatRewardModel()
    quality_rm = AgenticQualityRewardModel(judge_model, judge_tokenizer)
    self_correction_rm = AgenticSelfCorrectionRewardModel()
    accuracy_rm = MixedAccuracyRewardModel()  # reuse existing dispatcher

    return [format_rm, quality_rm, self_correction_rm, accuracy_rm]

