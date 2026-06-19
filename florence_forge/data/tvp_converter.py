"""TVP (Thinking with Visual Primitives) chain-of-thought data format and converters.

Extends FlorenceForge's existing VP converter with TVP-specific
thinking chain formats for:
  - Maze navigation (exploration + solution)
  - Path tracing (trajectory + endpoint identification)
  - Spatial reasoning (observation + deduction + conclusion)
  - Counting with chain-of-thought (scan + tally + conclude)

All coordinates remain in the [0, 999] integer space consistent
with FlorenceForge's VP coordinate convention.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from ..core.tasks import get_task_config
from ..core.visual_primitives import (
    format_point,
    format_ref_box,
    normalize_bbox,
    resolve_marker_style,
    sort_boxes_left_to_right,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain-of-thought template builders
# ---------------------------------------------------------------------------

class TVPChainBuilder:
    """Build chain-of-thought formatted VP training samples.

    Each method generates a structured thinking trace followed by
    a final answer, suitable for SFT training of TVP reasoning.
    """

    @staticmethod
    def _format_point_span(points: List[Tuple[int, int]], marker_style: str = "special") -> str:
        if not points:
            return ""
        return format_point(points, marker_style=resolve_marker_style(marker_style))

    @staticmethod
    def build_maze_chain(
        solvable: bool,
        exploration_points: List[Tuple[int, int]],
        solution_points: Optional[List[Tuple[int, int]]] = None,
        answer: str = "",
        *,
        start_point: Optional[Tuple[int, int]] = None,
        end_point: Optional[Tuple[int, int]] = None,
        exploration_steps: Optional[List[Dict[str, Any]]] = None,
        marker_style: str = "special",
    ) -> str:
        """Build a maze navigation chain-of-thought aligned with the TVP paper."""
        marker_style = resolve_marker_style(marker_style)
        parts: List[str] = []

        if start_point and end_point:
            parts.append(
                "I'll use a trial-and-error strategy to explore this maze. "
                f"First locate the starting point: {TVPChainBuilder._format_point_span([start_point], marker_style)}, "
                f"and the destination: {TVPChainBuilder._format_point_span([end_point], marker_style)}."
            )
            parts.append("**Start Exploring**:")
        else:
            parts.extend([
                "1. Observation\nI see a maze with walls and passages.",
                "2. Exploration",
            ])

        if exploration_steps:
            for index, step in enumerate(exploration_steps, start=1):
                step_points = [tuple(point) for point in step.get("points", [])]
                note = str(step.get("note", "")).strip()
                if step_points:
                    parts.append(
                        f"**Step{index}**: {note} "
                        f"{TVPChainBuilder._format_point_span(step_points, marker_style)}".strip()
                    )
                elif note:
                    parts.append(f"**Step{index}**: {note}")
        elif exploration_points:
            parts.append(TVPChainBuilder._format_point_span(exploration_points, marker_style))

        if solvable and solution_points:
            parts.append("**Final Path**: After exploration, the correct route is:")
            parts.append(TVPChainBuilder._format_point_span(solution_points, marker_style))
            if end_point:
                parts.append(
                    f"Successfully reaching the destination: "
                    f"{TVPChainBuilder._format_point_span([end_point], marker_style)}!"
                )
        elif not solvable:
            parts.append("**Analysis**: No valid path exists after exhaustive exploration.")

        parts.append(f"**Answer**\n{answer or ('true' if solvable else 'false')}")
        return "\n".join(part for part in parts if part)

    @staticmethod
    def build_path_chain(
        trajectory_points: List[Tuple[int, int]],
        endpoint: Optional[Tuple[int, int]] = None,
        end_label: str = "",
        *,
        start_point: Optional[Tuple[int, int]] = None,
        marker_style: str = "special",
    ) -> str:
        """Build a path tracing chain-of-thought aligned with the TVP paper."""
        marker_style = resolve_marker_style(marker_style)
        parts: List[str] = []

        if start_point:
            parts.append(
                "I find the starting point you mentioned, it's located here:\n"
                f"{TVPChainBuilder._format_point_span([start_point], marker_style)}."
            )
        else:
            parts.append("1. Observation\nI see a path to trace.")

        if trajectory_points:
            parts.append(
                "Following this line, the visual path I observe is:\n"
                f"{TVPChainBuilder._format_point_span(trajectory_points, marker_style)}"
            )

        if endpoint:
            parts.append(
                "Following this path, it connects to:\n"
                f"{TVPChainBuilder._format_point_span([endpoint], marker_style)}."
            )

        if end_label:
            parts.append(f"4. Answer\n{end_label}")
        return "\n".join(parts)

    @staticmethod
    def build_counting_chain(
        label: str,
        boxes: List[List[int]],
        count: int,
        marker_style: str = "special",
        *,
        mode: str = "coarse",
        query_hint: str = "",
    ) -> str:
        """Build coarse-grained counting CoT (batch grounding)."""
        marker_style = resolve_marker_style(marker_style)
        sorted_boxes = sort_boxes_left_to_right(boxes)
        if mode == "fine":
            return TVPChainBuilder.build_fine_grained_counting_chain(
                label=label,
                boxes=sorted_boxes,
                count=count,
                query_hint=query_hint,
                marker_style=marker_style,
            )

        return (
            "1. Deconstructing the query\n"
            f"The user wants me to count the total number of {label} in the image.\n"
            "2. Sweeping the scene for targets\n"
            f"Here they are: {format_ref_box(label, sorted_boxes, marker_style=marker_style)}\n"
            "3. Tallying the group\n"
            f"The total number is {count}."
        )

    @staticmethod
    def build_fine_grained_counting_chain(
        label: str,
        boxes: List[List[int]],
        count: int,
        *,
        query_hint: str = "",
        marker_style: str = "special",
        instance_notes: Optional[List[str]] = None,
    ) -> str:
        """Build fine-grained counting CoT (sequential scan per instance)."""
        marker_style = resolve_marker_style(marker_style)
        sorted_boxes = sort_boxes_left_to_right(boxes)
        intent = query_hint or (
            f"The question asks me to count the {label}. "
            "I need to scan the scene and verify each candidate instance."
        )
        parts = [
            f"1. What am I looking for\n{intent}",
            f"2. Evaluating each {label}",
        ]
        for index, box in enumerate(sorted_boxes):
            note = ""
            if instance_notes and index < len(instance_notes):
                note = f" {instance_notes[index]}"
            parts.append(
                f"- Instance {index + 1}:{note}\n"
                f"{format_ref_box(label, [box], marker_style=marker_style)}"
            )
        parts.append(
            f"3. Tally\nThere are {count} {label} in this image."
        )
        return "\n".join(parts)

    @staticmethod
    def build_spatial_chain(
        observation: str,
        reasoning: str,
        answer: str,
        supporting_boxes: Optional[Dict[str, List[List[int]]]] = None,
        *,
        object_groundings: Optional[List[Tuple[str, List[List[int]], str]]] = None,
        marker_style: str = "special",
    ) -> str:
        """Build a spatial reasoning chain-of-thought with optional multi-hop grounding."""
        marker_style = resolve_marker_style(marker_style)
        parts = [
            f"1. Analyzing the request\n{observation}",
            "2. Reasoning",
        ]

        if object_groundings:
            for label, boxes, note in object_groundings:
                sorted_boxes = sort_boxes_left_to_right(boxes)
                parts.append(f"- {note}\n{format_ref_box(label, sorted_boxes, marker_style=marker_style)}")
        else:
            parts.append(reasoning)
            if supporting_boxes:
                for lbl, bxs in supporting_boxes.items():
                    parts.append(format_ref_box(lbl, sort_boxes_left_to_right(bxs), marker_style=marker_style))

        parts.append(f"3. Conclusion\n{answer}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# TVP-specific data converters
# ---------------------------------------------------------------------------

class TVPDataConverter:
    """Convert standard annotations into TVP chain-of-thought VP samples.

    Extends VisualPrimitiveConverter with TVP-specific chain-of-thought
    formatting for maze, path, spatial, and enhanced counting tasks.
    """

    @staticmethod
    def maze_jsonl_to_vp(
        input_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "MAZE_VP",
        marker_style: str = "special",
    ) -> None:
        """Convert maze navigation JSONL to VP chain-of-thought samples.

        Expected input format per line:
          {
            "image": "path/to/maze.png",
            "solvable": true/false,
            "exploration_points": [[x1,y1], ...],
            "solution_points": [[x1,y1], ...] or [],
            "grid": [...],
            "grid_height": H,
            "grid_width": W
          }
        """
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = TVPDataConverter._get_prompt(task_type)
        chain_builder = TVPChainBuilder()

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="Maze to VP chain"):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                solvable = data.get("solvable", True)
                exploration_pts = [
                    tuple(p) for p in data.get("exploration_points", [])
                ]
                solution_pts = [
                    tuple(p) for p in data.get("solution_points", [])
                ] if solvable else None
                start_point = tuple(data["start_point"]) if "start_point" in data else None
                end_point = tuple(data["end_point"]) if "end_point" in data else None
                exploration_steps = data.get("exploration_steps")
                answer = data.get("answer") or ("true" if solvable else "false")

                chain = chain_builder.build_maze_chain(
                    solvable=solvable,
                    exploration_points=exploration_pts,
                    solution_points=solution_pts,
                    answer=answer,
                    start_point=start_point,
                    end_point=end_point,
                    exploration_steps=exploration_steps,
                    marker_style=marker_style,
                )

                image_path = (image_dir / data.get("image", "")).resolve()
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "maze",
                    "vp_task_type": task_type,
                    "solvable": solvable,
                    "grid_height": data.get("grid_height", 1),
                    "grid_width": data.get("grid_width", 1),
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("Maze to VP chain conversion completed: %s", output_path)

    @staticmethod
    def path_jsonl_to_vp(
        input_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "PATH_VP",
        marker_style: str = "special",
    ) -> None:
        """Convert path tracing JSONL to VP chain-of-thought samples.

        Expected input format per line:
          {
            "image": "path/to/path_img.png",
            "points": [[x1,y1], ...],
            "endpoint": [x, y],
            "end_label": "A"
          }
        """
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = TVPDataConverter._get_prompt(task_type)
        chain_builder = TVPChainBuilder()

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="Path to VP chain"):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                trajectory_pts = [
                    tuple(p) for p in data.get("points", [])
                ]
                endpoint = tuple(data["endpoint"]) if "endpoint" in data else None
                start_point = tuple(data["start_point"]) if "start_point" in data else None
                end_label = data.get("end_label", "")

                chain = chain_builder.build_path_chain(
                    trajectory_points=trajectory_pts,
                    endpoint=endpoint,
                    end_label=end_label,
                    start_point=start_point,
                    marker_style=marker_style,
                )

                image_path = (image_dir / data.get("image", "")).resolve()
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "path",
                    "vp_task_type": task_type,
                    "end_label": end_label,
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("Path to VP chain conversion completed: %s", output_path)

    @staticmethod
    def coco_to_tvp_counting(
        coco_json_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "COUNT_VP_COT",
        marker_style: str = "special",
        counting_mode: str = "coarse",
    ) -> None:
        """Convert COCO detection annotations to TVP chain-of-thought counting samples.

        Produces chain-of-thought formatted counting samples with
        explicit grounding steps, unlike the flat format produced by
        VisualPrimitiveConverter.coco_to_vp_counting().
        """
        coco_json_path = Path(coco_json_path).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()

        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
        images = {img["id"]: img for img in coco_data.get("images", [])}
        image_annotations: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            image_annotations[ann["image_id"]].append(ann)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = TVPDataConverter._get_prompt(task_type)
        chain_builder = TVPChainBuilder()

        with open(output_path, "w", encoding="utf-8") as f:
            for image_id, annotations in tqdm(
                image_annotations.items(), desc="COCO to TVP counting chain"
            ):
                image_info = images.get(image_id)
                if not image_info:
                    continue

                image_path = image_dir / image_info["file_name"]
                image_size = TVPDataConverter._resolve_image_size(image_info, image_path)

                grouped = TVPDataConverter._group_coco_annotations(
                    annotations, categories, image_size
                )
                if not grouped:
                    continue

                for label, boxes in grouped.items():
                    sorted_boxes = sort_boxes_left_to_right(boxes)
                    chain = chain_builder.build_counting_chain(
                        label=label,
                        boxes=sorted_boxes,
                        count=len(sorted_boxes),
                        marker_style=marker_style,
                        mode=counting_mode,
                    )
                    sample = {
                        "image": str(image_path.absolute()),
                        "prefix": prompt,
                        "suffix": chain,
                        "task_family": "tvprimitives",
                        "base_task": "counting",
                        "vp_task_type": task_type,
                        "count_label": label,
                        "count": len(sorted_boxes),
                        "counting_mode": counting_mode,
                        "vp_marker_style": marker_style,
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("COCO to TVP counting chain conversion completed: %s", output_path)

    @staticmethod
    def spatial_reasoning_jsonl_to_vp(
        input_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "SPATIAL_VP",
        marker_style: str = "special",
    ) -> None:
        """Convert spatial reasoning JSONL to VP chain-of-thought samples.

        Expected input format per line:
          {
            "image": "path/to/image.png",
            "observation": "There are two objects ...",
            "reasoning": "Object A is to the left of ...",
            "answer": "left",
            "supporting_boxes": {"object_A": [[x1,y1,x2,y2]], ...}  (optional)
          }
        """
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = TVPDataConverter._get_prompt(task_type)
        chain_builder = TVPChainBuilder()

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="Spatial reasoning to VP chain"):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                observation = data.get("observation", "")
                reasoning = data.get("reasoning", "")
                answer = data.get("answer", "")
                supporting_boxes = data.get("supporting_boxes")
                object_groundings = data.get("object_groundings")

                chain = chain_builder.build_spatial_chain(
                    observation=observation,
                    reasoning=reasoning,
                    answer=answer,
                    supporting_boxes=supporting_boxes,
                    object_groundings=object_groundings,
                    marker_style=marker_style,
                )

                image_path = (image_dir / data.get("image", "")).resolve()
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "spatial",
                    "vp_task_type": task_type,
                    "answer": answer,
                    "vp_marker_style": marker_style,
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("Spatial reasoning to VP chain conversion completed: %s", output_path)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_prompt(task_type: str) -> str:
        try:
            return get_task_config(task_type).get("prompt", f"<{task_type}>")
        except KeyError:
            return f"<{task_type}>"

    @staticmethod
    def _resolve_image_size(
        image_info: Dict[str, Any],
        image_path: Path,
    ) -> Tuple[int, int]:
        width = image_info.get("width")
        height = image_info.get("height")
        if width and height:
            return int(width), int(height)
        return TVPDataConverter._read_image_size(image_path)

    @staticmethod
    def _read_image_size(image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size

    @staticmethod
    def _group_coco_annotations(
        annotations: Iterable[Dict[str, Any]],
        categories: Dict[Any, str],
        image_size: Tuple[int, int],
    ) -> Dict[str, List[List[int]]]:
        grouped: Dict[str, List[List[int]]] = defaultdict(list)
        for ann in annotations:
            label = categories.get(ann.get("category_id"))
            if not label:
                continue
            try:
                bbox = normalize_bbox(ann["bbox"], image_size, input_format="xywh")
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping invalid COCO annotation %s: %s", ann, exc)
                continue
            grouped[str(label).strip()].append(bbox)
        return {
            label: sort_boxes_left_to_right(boxes)
            for label, boxes in grouped.items()
        }
