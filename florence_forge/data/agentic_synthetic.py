"""Synthetic Agentic meta-cognitive dataset generators.

Wraps the existing ``tvp_synthetic`` image/maze generators (so we reuse the
same image rendering pipeline) but outputs chains in agentic
``<PLAN>``/``<ACT>``/``<VERIFY>``/``<DECIDE>`` format via ``AgenticChainBuilder``.

This produces ready-to-train JSONL files that drop directly into
``MultiTaskDataset`` with no further conversion needed.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.tasks import get_task_config
from .agentic_trajectory_expander import AgenticChainBuilder
from .tvp_synthetic import (
    generate_maze_sample,
    generate_spatial_sample,
    pixel_box_to_vp_box,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-sample agentic generators
# ---------------------------------------------------------------------------

def _get_prompt(task_type: str) -> str:
    try:
        return get_task_config(task_type).get("prompt", f"<{task_type}>")
    except KeyError:
        return f"<{task_type}>"


def generate_agentic_maze(
    rng: random.Random,
    *,
    rows: int = 5,
    cols: int = 5,
    image_size: Tuple[int, int] = (384, 384),
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Tuple[Dict[str, Any], Any]:
    """Generate a maze image + agentic chain sample."""
    maze_sample, image = generate_maze_sample(rng, rows=rows, cols=cols)
    inject_error = rng.random() < error_injection_rate

    chain = AgenticChainBuilder.build_maze_chain(
        solvable=maze_sample.solvable,
        exploration_points=maze_sample.exploration_points,
        solution_points=maze_sample.solution_points if maze_sample.solvable else None,
        answer="true" if maze_sample.solvable else "false",
        start_point=maze_sample.start_point,
        end_point=maze_sample.end_point,
        exploration_steps=maze_sample.exploration_steps,
        marker_style=marker_style,
        inject_error=inject_error,
    )

    record = {
        "image": "",  # filled by caller
        "prefix": _get_prompt("AGENTIC_MAZE"),
        "suffix": chain,
        "task_family": "agentic",
        "base_task": "maze",
        "vp_task_type": "AGENTIC_MAZE",
        "agentic": True,
        "error_injected": inject_error,
        "solvable": maze_sample.solvable,
        "grid_height": maze_sample.grid_height,
        "grid_width": maze_sample.grid_width,
    }
    return record, image


def generate_agentic_spatial(
    rng: random.Random,
    *,
    image_size: Tuple[int, int] = (384, 384),
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Tuple[Dict[str, Any], Any]:
    """Generate a spatial reasoning image + agentic chain sample."""
    spatial_sample, image = generate_spatial_sample(rng, width=image_size[0], height=image_size[1])
    inject_error = rng.random() < error_injection_rate

    chain = AgenticChainBuilder.build_spatial_chain(
        observation=spatial_sample.observation,
        reasoning=spatial_sample.reasoning,
        answer=spatial_sample.answer,
        supporting_boxes=spatial_sample.supporting_boxes,
        marker_style=marker_style,
        inject_error=inject_error,
    )

    record = {
        "image": "",
        "prefix": _get_prompt("AGENTIC_SPATIAL"),
        "suffix": chain,
        "task_family": "agentic",
        "base_task": "spatial",
        "vp_task_type": "AGENTIC_SPATIAL",
        "agentic": True,
        "error_injected": inject_error,
        "answer": spatial_sample.answer,
        "supporting_boxes": spatial_sample.supporting_boxes,
    }
    return record, image


def generate_agentic_counting(
    rng: random.Random,
    *,
    image_size: Tuple[int, int] = (384, 384),
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Tuple[Dict[str, Any], Any]:
    """Generate a counting image + agentic chain sample.

    Draws N random colored rectangles on a white background.
    """
    from PIL import Image, ImageDraw

    width, height = image_size
    count = rng.randint(3, 8)
    label = "box"

    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)

    boxes_pixel: List[Tuple[int, int, int, int]] = []
    box_w, box_h = 50, 50
    for _ in range(count):
        x1 = rng.randint(10, width - box_w - 10)
        y1 = rng.randint(10, height - box_h - 10)
        x2, y2 = x1 + box_w, y1 + box_h
        color = (
            rng.randint(50, 200),
            rng.randint(50, 200),
            rng.randint(50, 200),
        )
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(20, 20, 20), width=2)
        boxes_pixel.append((x1, y1, x2, y2))

    # Sort left-to-right
    boxes_pixel.sort(key=lambda b: (b[0], b[1]))
    vp_boxes = [pixel_box_to_vp_box(*b, width, height) for b in boxes_pixel]

    inject_error = rng.random() < error_injection_rate
    chain = AgenticChainBuilder.build_counting_chain(
        label=label, boxes=vp_boxes, count=count,
        marker_style=marker_style, inject_error=inject_error,
    )

    record = {
        "image": "",
        "prefix": _get_prompt("AGENTIC_COUNT"),
        "suffix": chain,
        "task_family": "agentic",
        "base_task": "counting",
        "vp_task_type": "AGENTIC_COUNT",
        "agentic": True,
        "error_injected": inject_error,
        "count_label": label,
        "count": count,
        "vp_boxes": vp_boxes,
    }
    return record, image


def generate_agentic_grounding(
    rng: random.Random,
    *,
    image_size: Tuple[int, int] = (384, 384),
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Tuple[Dict[str, Any], Any]:
    """Generate a phrase grounding image + agentic chain sample.

    Places 2-3 colored objects with distinct labels, asks to locate one.
    """
    from PIL import Image, ImageDraw

    width, height = image_size
    labels_pool = ["red block", "blue block", "green block", "yellow block"]
    colors_pool = [
        (220, 53, 69),
        (13, 110, 253),
        (25, 135, 84),
        (255, 193, 7),
    ]

    num_objects = rng.randint(2, 3)
    chosen_indices = rng.sample(range(len(labels_pool)), num_objects)
    target_idx = rng.randint(0, num_objects - 1)

    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)

    box_w, box_h = 60, 60
    placed: List[Tuple[str, List[int]]] = []
    for idx in chosen_indices:
        label = labels_pool[idx]
        color = colors_pool[idx]
        x1 = rng.randint(10, width - box_w - 10)
        y1 = rng.randint(10, height - box_h - 10)
        x2, y2 = x1 + box_w, y1 + box_h
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(20, 20, 20), width=2)
        draw.text((x1 + 6, y1 + 4), label[0].upper(), fill=(255, 255, 255))
        vp_box = pixel_box_to_vp_box(x1, y1, x2, y2, width, height)
        placed.append((label, vp_box))

    target_label, target_box = placed[target_idx]
    # Pass ALL placed boxes so error injection can ground the wrong region
    all_boxes = [vp_box for _, vp_box in placed]
    inject_error = rng.random() < error_injection_rate

    chain = AgenticChainBuilder.build_grounding_chain(
        caption=f"Locate the {target_label}.",
        label=target_label,
        boxes=all_boxes,
        marker_style=marker_style,
        inject_error=inject_error,
    )

    record = {
        "image": "",
        "prefix": _get_prompt("AGENTIC_GROUNDING"),
        "suffix": chain,
        "task_family": "agentic",
        "base_task": "phrase_grounding",
        "vp_task_type": "AGENTIC_GROUNDING",
        "agentic": True,
        "error_injected": inject_error,
        "label": target_label,
        "caption": f"Locate the {target_label}.",
        "vp_boxes": [target_box],
    }
    return record, image


# ---------------------------------------------------------------------------
# Batch JSONL writers
# ---------------------------------------------------------------------------

def write_agentic_maze_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    rows: int = 5,
    cols: int = 5,
    seed: int = 42,
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Path:
    """Write agentic maze samples to JSONL + images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "agentic_maze.jsonl"
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for i in range(num_samples):
            record, image = generate_agentic_maze(
                rng, rows=rows, cols=cols,
                marker_style=marker_style,
                error_injection_rate=error_injection_rate,
            )
            image_name = f"agentic_maze_{i:04d}.png"
            record["image"] = image_name
            image.save(output_dir / image_name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d agentic maze samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_agentic_spatial_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Path:
    """Write agentic spatial reasoning samples to JSONL + images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "agentic_spatial.jsonl"
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for i in range(num_samples):
            record, image = generate_agentic_spatial(
                rng,
                marker_style=marker_style,
                error_injection_rate=error_injection_rate,
            )
            image_name = f"agentic_spatial_{i:04d}.png"
            record["image"] = image_name
            image.save(output_dir / image_name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d agentic spatial samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_agentic_counting_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Path:
    """Write agentic counting samples to JSONL + images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "agentic_counting.jsonl"
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for i in range(num_samples):
            record, image = generate_agentic_counting(
                rng,
                marker_style=marker_style,
                error_injection_rate=error_injection_rate,
            )
            image_name = f"agentic_count_{i:04d}.png"
            record["image"] = image_name
            image.save(output_dir / image_name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d agentic counting samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_agentic_grounding_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    marker_style: str = "special",
    error_injection_rate: float = 0.3,
) -> Path:
    """Write agentic phrase grounding samples to JSONL + images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "agentic_grounding.jsonl"
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for i in range(num_samples):
            record, image = generate_agentic_grounding(
                rng,
                marker_style=marker_style,
                error_injection_rate=error_injection_rate,
            )
            image_name = f"agentic_ground_{i:04d}.png"
            record["image"] = image_name
            image.save(output_dir / image_name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d agentic grounding samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_all_agentic_synthetic(
    output_dir: str | Path,
    *,
    num_samples: int = 8,
    seed: int = 42,
    error_injection_rate: float = 0.3,
) -> Dict[str, Path]:
    """Generate all agentic synthetic datasets under one directory."""
    base = Path(output_dir)
    return {
        "maze": write_agentic_maze_jsonl(
            base / "maze", num_samples=num_samples, seed=seed,
            error_injection_rate=error_injection_rate,
        ),
        "spatial": write_agentic_spatial_jsonl(
            base / "spatial", num_samples=num_samples, seed=seed + 1,
            error_injection_rate=error_injection_rate,
        ),
        "counting": write_agentic_counting_jsonl(
            base / "counting", num_samples=num_samples, seed=seed + 2,
            error_injection_rate=error_injection_rate,
        ),
        "grounding": write_agentic_grounding_jsonl(
            base / "grounding", num_samples=num_samples, seed=seed + 3,
            error_injection_rate=error_injection_rate,
        ),
    }
