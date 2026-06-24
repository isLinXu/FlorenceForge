"""Synthetic TVP maze and path-tracing dataset generators."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

Coord = Tuple[int, int]


@dataclass
class MazeSample:
    image_name: str
    solvable: bool
    start_point: Coord
    end_point: Coord
    exploration_points: List[Coord]
    solution_points: List[Coord]
    exploration_steps: List[Dict[str, Any]]
    grid_height: int
    grid_width: int


@dataclass
class PathSample:
    image_name: str
    points: List[Coord]
    start_point: Coord
    endpoint: Coord
    end_label: str


@dataclass
class SpatialSample:
    image_name: str
    observation: str
    reasoning: str
    answer: str
    supporting_boxes: Dict[str, List[List[int]]]


def to_vp_coord(x: float, y: float, width: int, height: int) -> Coord:
    """Map pixel coordinates into FlorenceForge [0, 999] VP space."""
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)
    return (
        max(0, min(999, int(round(x / max_x * 999)))),
        max(0, min(999, int(round(y / max_y * 999)))),
    )


def _cell_center(row: int, col: int, cell_size: int, margin: int) -> Tuple[float, float]:
    x = margin + col * cell_size + cell_size / 2
    y = margin + row * cell_size + cell_size / 2
    return x, y


def generate_maze_grid(
    rng: random.Random,
    rows: int,
    cols: int,
) -> Tuple[List[List[bool]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Generate a perfect maze and return (passage_grid, solution_cells, dead_end_cells)."""
    rows = max(rows, 2)
    cols = max(cols, 2)
    visited = [[False] * cols for _ in range(rows)]
    passages = [[False] * cols for _ in range(rows)]
    stack: List[Tuple[int, int]] = [(0, 0)]
    visited[0][0] = True
    passages[0][0] = True
    dead_ends: List[Tuple[int, int]] = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        row, col = stack[-1]
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            if len(stack) > 2:
                dead_ends.append((row, col))
            stack.pop()
            continue
        next_row, next_col = rng.choice(neighbors)
        visited[next_row][next_col] = True
        passages[next_row][next_col] = True
        stack.append((next_row, next_col))

    start = (0, 0)
    end = (rows - 1, cols - 1)
    solution = _bfs_path(passages, start, end)
    if not solution:
        passages[end[0]][end[1]] = True
        solution = _bfs_path(passages, start, end) or [start, end]
    return passages, solution, dead_ends


def _bfs_path(
    passages: Sequence[Sequence[bool]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    rows = len(passages)
    cols = len(passages[0]) if rows else 0
    queue = [start]
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row, col = queue.pop(0)
        if (row, col) == end:
            break
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and passages[nr][nc] and (nr, nc) not in parents:
                parents[(nr, nc)] = (row, col)
                queue.append((nr, nc))
    if end not in parents:
        return []
    path: List[Tuple[int, int]] = []
    cursor: Optional[Tuple[int, int]] = end
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return path


def render_maze_image(
    passages: Sequence[Sequence[bool]],
    *,
    cell_size: int = 48,
    margin: int = 24,
    wall_color: Tuple[int, int, int] = (20, 20, 20),
    passage_color: Tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    rows = len(passages)
    cols = len(passages[0]) if rows else 0
    width = margin * 2 + cols * cell_size
    height = margin * 2 + rows * cell_size
    image = Image.new("RGB", (width, height), wall_color)
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        for col in range(cols):
            if not passages[row][col]:
                continue
            x0 = margin + col * cell_size
            y0 = margin + row * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=passage_color,
            )
    return image


def build_maze_exploration_steps(
    solution_cells: Sequence[Tuple[int, int]],
    dead_end_cells: Sequence[Tuple[int, int]],
    *,
    cell_size: int,
    margin: int,
    image_size: Tuple[int, int],
    rng: random.Random,
    max_steps: int = 4,
) -> Tuple[List[Coord], List[Coord], List[Dict[str, Any]]]:
    width, height = image_size
    solution_points = [
        to_vp_coord(*_cell_center(row, col, cell_size, margin), width, height)
        for row, col in solution_cells
    ]
    dead_points = [
        to_vp_coord(*_cell_center(row, col, cell_size, margin), width, height)
        for row, col in dead_end_cells
    ]
    rng.shuffle(dead_points)
    exploration_points = dead_points[: max(1, min(len(dead_points), max_steps - 1))]
    steps: List[Dict[str, Any]] = []
    for index, point in enumerate(exploration_points, start=1):
        steps.append({
            "points": [list(point)],
            "note": "Dead end reached, backtracking." if index % 2 else "Trying another branch.",
        })
    if solution_points:
        mid = solution_points[len(solution_points) // 2]
        steps.append({"points": [list(mid)], "note": "Found a promising corridor."})
    return exploration_points, solution_points, steps


def generate_maze_sample(
    rng: random.Random,
    *,
    rows: int = 8,
    cols: int = 8,
    solvable: Optional[bool] = None,
) -> Tuple[MazeSample, Image.Image]:
    passages, solution_cells, dead_ends = generate_maze_grid(rng, rows, cols)
    make_solvable = solvable if solvable is not None else rng.random() > 0.15
    if not make_solvable and len(solution_cells) > 2:
        blocked = solution_cells[len(solution_cells) // 2]
        passages[blocked[0]][blocked[1]] = False
        solution_cells = []

    cell_size = 48
    margin = 24
    image = render_maze_image(passages, cell_size=cell_size, margin=margin)
    width, height = image.size
    start_px = _cell_center(solution_cells[0][0], solution_cells[0][1], cell_size, margin) if solution_cells else _cell_center(0, 0, cell_size, margin)
    end_px = _cell_center(rows - 1, cols - 1, cell_size, margin)
    start_point = to_vp_coord(*start_px, width, height)
    end_point = to_vp_coord(*end_px, width, height)

    exploration_points, solution_points, exploration_steps = build_maze_exploration_steps(
        solution_cells,
        dead_ends,
        cell_size=cell_size,
        margin=margin,
        image_size=(width, height),
        rng=rng,
    )

    sample = MazeSample(
        image_name="",
        solvable=bool(solution_cells),
        start_point=start_point,
        end_point=end_point,
        exploration_points=exploration_points,
        solution_points=solution_points if solution_cells else [],
        exploration_steps=exploration_steps,
        grid_height=rows,
        grid_width=cols,
    )
    return sample, image


def _bezier_points(
    rng: random.Random,
    width: int,
    height: int,
    num_points: int = 12,
) -> List[Tuple[float, float]]:
    start = (rng.uniform(width * 0.1, width * 0.9), rng.uniform(height * 0.1, height * 0.9))
    control1 = (rng.uniform(0, width), rng.uniform(0, height))
    control2 = (rng.uniform(0, width), rng.uniform(0, height))
    end = (rng.uniform(width * 0.1, width * 0.9), rng.uniform(height * 0.1, height * 0.9))
    points: List[Tuple[float, float]] = []
    for index in range(num_points):
        t = index / max(num_points - 1, 1)
        u = 1 - t
        x = (
            u ** 3 * start[0]
            + 3 * u ** 2 * t * control1[0]
            + 3 * u * t ** 2 * control2[0]
            + t ** 3 * end[0]
        )
        y = (
            u ** 3 * start[1]
            + 3 * u ** 2 * t * control1[1]
            + 3 * u * t ** 2 * control2[1]
            + t ** 3 * end[1]
        )
        points.append((x, y))
    return points


def generate_path_sample(
    rng: random.Random,
    *,
    width: int = 768,
    height: int = 768,
    num_lines: int = 3,
) -> Tuple[PathSample, Image.Image]:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    palette = [(220, 53, 69), (13, 110, 253), (25, 135, 84), (255, 193, 7)]
    labels = ["A", "B", "C", "D", "E"]
    trajectories: List[List[Coord]] = []
    endpoints: List[Coord] = []
    start_points: List[Coord] = []

    for index in range(max(2, num_lines)):
        color = palette[index % len(palette)]
        pixel_points = _bezier_points(rng, width, height)
        draw.line(pixel_points, fill=color, width=6)
        vp_points = [to_vp_coord(x, y, width, height) for x, y in pixel_points]
        trajectories.append(vp_points)
        start_points.append(vp_points[0])
        endpoints.append(vp_points[-1])
        label = labels[index]
        ex, ey = pixel_points[-1]
        draw.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=color, outline=(0, 0, 0))
        draw.text((ex + 12, ey - 8), label, fill=(0, 0, 0))

    target_index = rng.randrange(len(trajectories))
    sample = PathSample(
        image_name="",
        points=trajectories[target_index],
        start_point=start_points[target_index],
        endpoint=endpoints[target_index],
        end_label=labels[target_index],
    )
    return sample, image


def write_maze_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    rows: int = 8,
    cols: int = 8,
    seed: int = 42,
    jsonl_name: str = "maze_data.jsonl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for index in range(num_samples):
            sample, image = generate_maze_sample(rng, rows=rows, cols=cols)
            image_name = f"maze_{index:04d}.png"
            sample.image_name = image_name
            image.save(output_dir / image_name)
            record = {
                "image": image_name,
                "solvable": sample.solvable,
                "start_point": list(sample.start_point),
                "end_point": list(sample.end_point),
                "exploration_points": [list(point) for point in sample.exploration_points],
                "solution_points": [list(point) for point in sample.solution_points],
                "exploration_steps": sample.exploration_steps,
                "grid_height": sample.grid_height,
                "grid_width": sample.grid_width,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %s maze samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_path_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    jsonl_name: str = "path_data.jsonl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for index in range(num_samples):
            sample, image = generate_path_sample(rng)
            image_name = f"path_{index:04d}.png"
            sample.image_name = image_name
            image.save(output_dir / image_name)
            record = {
                "image": image_name,
                "points": [list(point) for point in sample.points],
                "start_point": list(sample.start_point),
                "endpoint": list(sample.endpoint),
                "end_label": sample.end_label,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %s path-tracing samples to %s", num_samples, jsonl_path)
    return jsonl_path


def pixel_box_to_vp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> List[int]:
    """Convert a pixel axis-aligned box to VP [0, 999] space."""
    left, top = to_vp_coord(min(x1, x2), min(y1, y2), width, height)
    right, bottom = to_vp_coord(max(x1, x2), max(y1, y2), width, height)
    return [left, top, right, bottom]


def generate_spatial_sample(
    rng: random.Random,
    *,
    width: int = 640,
    height: int = 640,
) -> Tuple[SpatialSample, Image.Image]:
    """Generate a two-object spatial reasoning sample with a known relation."""
    relation = rng.choice(["left", "right", "above", "below"])
    object_a = "object_A"
    object_b = "object_B"
    box_a_w, box_a_h = rng.randint(90, 150), rng.randint(90, 150)
    box_b_w, box_b_h = rng.randint(90, 150), rng.randint(90, 150)
    margin = 36

    bx1 = rng.randint(width // 3, width // 3 + width // 6)
    by1 = rng.randint(height // 3, height // 3 + height // 6)
    bx2, by2 = bx1 + box_b_w, by1 + box_b_h

    gap = rng.randint(48, 120)
    if relation == "left":
        ax2 = max(margin, bx1 - gap)
        ax1 = max(margin, ax2 - box_a_w)
        ay1 = by1 + (box_b_h - box_a_h) // 2
        ax2, ay2 = ax1 + box_a_w, ay1 + box_a_h
    elif relation == "right":
        ax1 = min(width - margin - box_a_w, bx2 + gap)
        ax2 = ax1 + box_a_w
        ay1 = by1 + (box_b_h - box_a_h) // 2
        ax2, ay2 = ax1 + box_a_w, ay1 + box_a_h
    elif relation == "above":
        ax1 = bx1 + (box_b_w - box_a_w) // 2
        ay2 = max(margin, by1 - gap)
        ay1 = max(margin, ay2 - box_a_h)
        ax2, ay2 = ax1 + box_a_w, ay1 + box_a_h
    else:
        ax1 = bx1 + (box_b_w - box_a_w) // 2
        ay1 = min(height - margin - box_a_h, by2 + gap)
        ax2, ay2 = ax1 + box_a_w, ay1 + box_a_h

    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    palette = {"object_A": (220, 53, 69), "object_B": (13, 110, 253)}
    for label, box, short in (
        (object_a, (ax1, ay1, ax2, ay2), "A"),
        (object_b, (bx1, by1, bx2, by2), "B"),
    ):
        draw.rectangle(box, fill=palette[label], outline=(20, 20, 20), width=3)
        draw.text((box[0] + 8, box[1] + 8), short, fill=(255, 255, 255))

    vp_a = pixel_box_to_vp_box(ax1, ay1, ax2, ay2, width, height)
    vp_b = pixel_box_to_vp_box(bx1, by1, bx2, by2, width, height)
    observation = (
        f"The image contains two colored objects labeled A and B. "
        f"Determine where {object_a} is relative to {object_b}."
    )
    reasoning = (
        f"I locate {object_a} and {object_b} first, then compare their centers "
        f"to decide whether A is to the left/right/above/below B."
    )
    sample = SpatialSample(
        image_name="",
        observation=observation,
        reasoning=reasoning,
        answer=relation,
        supporting_boxes={object_a: [vp_a], object_b: [vp_b]},
    )
    return sample, image


def write_spatial_jsonl(
    output_dir: str | Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    jsonl_name: str = "spatial_data.jsonl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name
    rng = random.Random(seed)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for index in range(num_samples):
            sample, image = generate_spatial_sample(rng)
            image_name = f"spatial_{index:04d}.png"
            sample.image_name = image_name
            image.save(output_dir / image_name)
            record = {
                "image": image_name,
                "observation": sample.observation,
                "reasoning": sample.reasoning,
                "answer": sample.answer,
                "supporting_boxes": sample.supporting_boxes,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %s spatial reasoning samples to %s", num_samples, jsonl_path)
    return jsonl_path


def write_all_tvp_synthetic(
    output_dir: str | Path,
    *,
    num_samples: int = 8,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate maze, path, and spatial raw JSONL datasets under one directory."""
    base = Path(output_dir)
    return {
        "maze": write_maze_jsonl(base / "maze", num_samples=num_samples, rows=5, cols=5, seed=seed),
        "path": write_path_jsonl(base / "path", num_samples=num_samples, seed=seed + 1),
        "spatial": write_spatial_jsonl(base / "spatial", num_samples=num_samples, seed=seed + 2),
    }
