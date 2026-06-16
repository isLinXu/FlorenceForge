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
    format_ref_box,
    normalize_bbox,
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
    def build_maze_chain(
        solvable: bool,
        exploration_points: List[Tuple[int, int]],
        solution_points: Optional[List[Tuple[int, int]]] = None,
        answer: str = "",
    ) -> str:
        """Build a maze navigation chain-of-thought.

        Format:
          1. Observation
          I see a maze with walls and passages.
          2. Exploration
          <|point|>[[x1,y1],[x2,y2],...]<|/point|>
          3. Solution
          [path or conclusion]
          4. Answer
          [true/false]
        """
        parts = [
            "1. Observation\nI see a maze with walls and passages.",
            "2. Exploration\n",
        ]

        if exploration_points:
            point_str = ",".join(
                f"[{x},{y}]" for x, y in exploration_points
            )
            parts.append(f"<|point|>[{point_str}]<|/point|>")

        if solvable and solution_points:
            parts.append("\n3. Solution\n")
            sol_str = ",".join(
                f"[{x},{y}]" for x, y in solution_points
            )
            parts.append(f"<|point|>[{sol_str}]<|/point|>")
        else:
            parts.append("\n3. Analysis\nNo valid path exists.")

        parts.append(f"\n4. Answer\n{answer}")
        return "\n".join(parts)

    @staticmethod
    def build_path_chain(
        trajectory_points: List[Tuple[int, int]],
        endpoint: Optional[Tuple[int, int]] = None,
        end_label: str = "",
    ) -> str:
        """Build a path tracing chain-of-thought.

        Format:
          1. Observation
          I see a path to trace.
          2. Trajectory
          <|point|>[[x1,y1],...] <|/point|>
          3. Endpoint identification
          [endpoint or label]
          4. Answer
          [endpoint label]
        """
        parts = [
            "1. Observation\nI see a path to trace.",
            "2. Trajectory\n",
        ]

        if trajectory_points:
            pt_str = ",".join(f"[{x},{y}]" for x, y in trajectory_points)
            parts.append(f"<|point|>[{pt_str}]<|/point|>")

        if endpoint:
            parts.append(
                f"\n3. Endpoint identification\n"
                f"The path ends at <|point|>[[{endpoint[0]},{endpoint[1]}]]<|/point|>"
            )

        parts.append(f"\n4. Answer\n{end_label}")
        return "\n".join(parts)

    @staticmethod
    def build_counting_chain(
        label: str,
        boxes: List[List[int]],
        count: int,
        marker_style: str = "special",
    ) -> str:
        """Build a counting chain-of-thought with VP grounding.

        Format:
          1. Analyzing the request
          The visual target is [label].
          2. Object grounding
          <|ref|>[label]<|/ref|><|box|>[[x1,y1,x2,y2],...] <|/box|>
          3. Conclusion
          There are [count] [label] in this image.
        """
        def formatter(lbl, bxs):
            return (format_ref_box(lbl, bxs, marker_style=marker_style))

        parts = [
            "1. Analyzing the request\n"
            f"The visual target is {label}.",
            "2. Object grounding\n"
            + formatter(label, boxes),
            "3. Conclusion\n"
            f"There are {count} {label} in this image.",
        ]
        return "\n".join(parts)

    @staticmethod
    def build_spatial_chain(
        observation: str,
        reasoning: str,
        answer: str,
        supporting_boxes: Optional[Dict[str, List[List[int]]]] = None,
        marker_style: str = "special",
    ) -> str:
        """Build a spatial reasoning chain-of-thought.

        Format:
          1. Observation
          [observation]
          2. Reasoning
          [reasoning with optional VP grounding]
          3. Answer
          [answer]
        """
        parts = [
            f"1. Observation\n{observation}",
            "2. Reasoning\n" + reasoning,
        ]

        if supporting_boxes:
            for lbl, bxs in supporting_boxes.items():
                parts.append(format_ref_box(lbl, bxs, marker_style=marker_style))

        parts.append(f"3. Answer\n{answer}")
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
                answer = "true" if solvable else "false"

                chain = chain_builder.build_maze_chain(
                    solvable=solvable,
                    exploration_points=exploration_pts,
                    solution_points=solution_pts,
                    answer=answer,
                )

                image_path = image_dir / data.get("image", "")
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "maze",
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
                end_label = data.get("end_label", "")

                chain = chain_builder.build_path_chain(
                    trajectory_points=trajectory_pts,
                    endpoint=endpoint,
                    end_label=end_label,
                )

                image_path = image_dir / data.get("image", "")
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "path",
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
                    chain = chain_builder.build_counting_chain(
                        label=label,
                        boxes=boxes,
                        count=len(boxes),
                        marker_style=marker_style,
                    )
                    sample = {
                        "image": str(image_path.absolute()),
                        "prefix": prompt,
                        "suffix": chain,
                        "task_family": "tvprimitives",
                        "base_task": "counting",
                        "count_label": label,
                        "count": len(boxes),
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

                chain = chain_builder.build_spatial_chain(
                    observation=observation,
                    reasoning=reasoning,
                    answer=answer,
                    supporting_boxes=supporting_boxes,
                    marker_style=marker_style,
                )

                image_path = image_dir / data.get("image", "")
                sample = {
                    "image": str(image_path),
                    "prefix": prompt,
                    "suffix": chain,
                    "task_family": "tvprimitives",
                    "base_task": "spatial",
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
        return dict(grouped)
