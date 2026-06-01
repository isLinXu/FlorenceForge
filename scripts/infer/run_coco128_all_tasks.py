#!/usr/bin/env python3
"""Run Florence-2 inference for every FlorenceForge task on COCO128.

The script writes:
  - results.jsonl: one record per image/task
  - summary.json: run metadata and success/failure counts
  - gallery.html: quick visual review page
  - visualizations/<image_stem>/<task>.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import html
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from florence_forge.core.backends.base_vlm import (
    _patch_transformers_config_defaults,
    _patch_transformers_import_check,
)
from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.tasks import FLORENCE2_TASKS


LOGGER = logging.getLogger("coco128_florence_all_tasks")

COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

TEXT_INPUT_TASKS = {
    "CAPTION_TO_PHRASE_GROUNDING",
    "OPEN_VOCABULARY_DETECTION",
    "REGION_TO_CATEGORY",
    "REGION_TO_DESCRIPTION",
    "REGION_TO_SEGMENTATION",
    "REFERRING_EXPRESSION_SEGMENTATION",
}

COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (0, 150, 199),
    (106, 153, 78),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="COCO128 root with images/<split> and labels/<split>.",
    )
    parser.add_argument("--data-yaml", type=Path, default=None, help="Optional coco128.yaml with class names.")
    parser.add_argument("--split", default="train2017", help="Split directory name, e.g. train2017 or val2017.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/coco128_florence_all_tasks"))
    parser.add_argument("--model-name", default="microsoft/Florence-2-base")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--max-images", type=int, default=None, help="Limit image count for smoke runs.")
    parser.add_argument("--tasks", default="all", help="Comma-separated task names, or 'all'.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens-cap", type=int, default=256)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing results.jsonl and skip image/task pairs already recorded.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, rerun existing records whose status is not ok.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_names(data_yaml: Optional[Path]) -> Dict[int, str]:
    if not data_yaml:
        return dict(COCO_NAMES)
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get("names", COCO_NAMES)
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def list_images(dataset_root: Path, split: str, max_images: Optional[int]) -> List[Path]:
    image_dir = dataset_root / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split not found: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return images


def yolo_label_path(dataset_root: Path, split: str, image_path: Path) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def load_yolo_labels(
    label_path: Path,
    image_size: Tuple[int, int],
    names: Dict[int, str],
) -> List[Dict[str, Any]]:
    width, height = image_size
    if not label_path.exists():
        return []
    annotations = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, box_w, box_h = map(float, parts[1:5])
        x1 = (x_center - box_w / 2) * width
        y1 = (y_center - box_h / 2) * height
        x2 = (x_center + box_w / 2) * width
        y2 = (y_center + box_h / 2) * height
        annotations.append(
            {
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "bbox": [x1, y1, x2, y2],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            }
        )
    return annotations


def bbox_to_loc_tokens(bbox: Sequence[float], image_size: Tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    coords = [
        round(max(0, min(999, x1 / width * 999))),
        round(max(0, min(999, y1 / height * 999))),
        round(max(0, min(999, x2 / width * 999))),
        round(max(0, min(999, y2 / height * 999))),
    ]
    return "".join(f"<loc_{coord}>" for coord in coords)


def unique_labels(annotations: Sequence[Dict[str, Any]], limit: int = 8) -> List[str]:
    labels = []
    for annotation in annotations:
        label = annotation["label"]
        if label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def selected_region(annotations: Sequence[Dict[str, Any]], image_size: Tuple[int, int]) -> Tuple[str, str, List[float]]:
    if annotations:
        chosen = max(annotations, key=lambda item: item.get("area", 0.0))
        return chosen["label"], bbox_to_loc_tokens(chosen["bbox"], image_size), chosen["bbox"]
    width, height = image_size
    bbox = [0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height]
    return "object", bbox_to_loc_tokens(bbox, image_size), bbox


def build_text_inputs(
    annotations: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    labels = unique_labels(annotations)
    region_label, region_tokens, region_bbox = selected_region(annotations, image_size)
    label_text = ", ".join(labels) if labels else "objects"
    caption_hint = f"A photo containing {label_text}."
    open_vocab_query = labels[0] if labels else "object"
    text_inputs = {
        "CAPTION_TO_PHRASE_GROUNDING": caption_hint,
        "OPEN_VOCABULARY_DETECTION": open_vocab_query,
        "REGION_TO_CATEGORY": region_tokens,
        "REGION_TO_DESCRIPTION": region_tokens,
        "REGION_TO_SEGMENTATION": region_tokens,
        "REFERRING_EXPRESSION_SEGMENTATION": region_label,
    }
    metadata = {
        "unique_labels": labels,
        "region_label": region_label,
        "region_tokens": region_tokens,
        "region_bbox": region_bbox,
    }
    return text_inputs, metadata


def parse_task_names(tasks_arg: str) -> List[str]:
    if tasks_arg.strip().lower() == "all":
        return list(FLORENCE2_TASKS.keys())
    names = [name.strip() for name in tasks_arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in FLORENCE2_TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return names


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def record_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record.get("image_path", "")), str(record.get("task", ""))


def load_existing_records(results_path: Path) -> List[Dict[str, Any]]:
    if not results_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Ignoring malformed JSONL line %s in %s: %s", line_number, results_path, exc)
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def get_answer(parsed: Dict[str, Any], task_prompt: str) -> Any:
    return parsed.get(task_prompt, parsed)


def font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color: Tuple[int, int, int]) -> None:
    text_font = font(14)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=text_font)
    pad = 3
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=color + (220,),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=text_font)


def wrap_text(text: str, max_chars: int = 72) -> str:
    words = text.replace("\n", " ").split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        if sum(len(x) + 1 for x in current) + len(word) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:8])


def draw_text_panel(image: Image.Image, task: str, text: str) -> Image.Image:
    base = image.convert("RGBA")
    width, height = base.size
    panel_height = min(max(height // 4, 120), 260)
    canvas = Image.new("RGBA", (width, height + panel_height), (255, 255, 255, 255))
    canvas.alpha_composite(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, height, width, height + panel_height], fill=(20, 24, 31, 245))
    title_font = font(18)
    body_font = font(15)
    draw.text((16, height + 12), task, fill=(255, 255, 255, 255), font=title_font)
    draw.text((16, height + 42), wrap_text(text), fill=(230, 234, 240, 255), font=body_font)
    return canvas.convert("RGB")


def extract_bboxes(answer: Any) -> Tuple[List[List[float]], List[str]]:
    if not isinstance(answer, dict):
        return [], []
    boxes = answer.get("bboxes") or []
    labels = answer.get("labels") or answer.get("bboxes_labels") or []
    if labels and len(labels) < len(boxes):
        labels = labels + [""] * (len(boxes) - len(labels))
    if not labels:
        labels = [""] * len(boxes)
    return boxes, [str(label) for label in labels]


def extract_quad_boxes(answer: Any) -> Tuple[List[List[float]], List[str]]:
    if not isinstance(answer, dict):
        return [], []
    boxes = answer.get("quad_boxes") or []
    labels = answer.get("labels") or [""] * len(boxes)
    return boxes, [str(label) for label in labels]


def flatten_polygon(poly: Any) -> Iterable[List[Tuple[float, float]]]:
    if not poly:
        return
    if isinstance(poly, (list, tuple)) and all(isinstance(x, (int, float)) for x in poly):
        points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
        if len(points) >= 3:
            yield points
        return
    if (
        isinstance(poly, (list, tuple))
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in poly)
        and all(isinstance(coord, (int, float)) for point in poly for coord in point)
    ):
        points = [(float(point[0]), float(point[1])) for point in poly]
        if len(points) >= 3:
            yield points
        return
    if isinstance(poly, (list, tuple)):
        for child in poly:
            yield from flatten_polygon(child)


def extract_polygons(answer: Any) -> Tuple[List[List[Tuple[float, float]]], List[str]]:
    if not isinstance(answer, dict):
        return [], []
    raw_polygons = answer.get("polygons") or []
    raw_labels = answer.get("labels") or answer.get("polygons_labels") or []
    polygons: List[List[Tuple[float, float]]] = []
    labels: List[str] = []
    for index, item in enumerate(raw_polygons):
        label = str(raw_labels[index]) if index < len(raw_labels) else ""
        for polygon in flatten_polygon(item):
            polygons.append(polygon)
            labels.append(label)
    return polygons, labels


def visualize_answer(
    image: Image.Image,
    task_name: str,
    task_prompt: str,
    parsed: Dict[str, Any],
    selected_bbox: Sequence[float],
    save_path: Path,
) -> None:
    answer = get_answer(parsed, task_prompt)
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")

    boxes, box_labels = extract_bboxes(answer)
    quads, quad_labels = extract_quad_boxes(answer)
    polygons, polygon_labels = extract_polygons(answer)

    if boxes:
        for idx, box in enumerate(boxes):
            color = COLORS[idx % len(COLORS)]
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
            label = box_labels[idx] if idx < len(box_labels) else ""
            if label:
                draw_label(draw, (x1 + 2, max(2, y1 + 2)), label, color)
        canvas.convert("RGB").save(save_path)
        return

    if quads:
        for idx, quad in enumerate(quads):
            color = COLORS[idx % len(COLORS)]
            points = [(quad[i], quad[i + 1]) for i in range(0, len(quad) - 1, 2)]
            draw.polygon(points, outline=color + (255,), fill=color + (60,))
            label = quad_labels[idx] if idx < len(quad_labels) else ""
            if label and points:
                draw_label(draw, points[0], label, color)
        canvas.convert("RGB").save(save_path)
        return

    if polygons:
        for idx, polygon in enumerate(polygons):
            color = COLORS[idx % len(COLORS)]
            draw.polygon(polygon, outline=color + (255,), fill=color + (70,))
            label = polygon_labels[idx] if idx < len(polygon_labels) else ""
            if label:
                draw_label(draw, polygon[0], label, color)
        canvas.convert("RGB").save(save_path)
        return

    if task_name in {
        "REGION_TO_CATEGORY",
        "REGION_TO_DESCRIPTION",
        "REGION_TO_SEGMENTATION",
    }:
        x1, y1, x2, y2 = selected_bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 255), width=5)
        draw.rectangle([x1, y1, x2, y2], outline=(230, 57, 70, 255), width=3)

    text = answer if isinstance(answer, str) else json.dumps(jsonable(answer), ensure_ascii=False)
    draw_text_panel(canvas.convert("RGB"), task_name, text).save(save_path)


def make_model(model_name: str, device: str) -> Florence2MultiTaskModel:
    _patch_transformers_config_defaults()
    _patch_transformers_import_check()
    config = ModelConfig(
        model_name=model_name,
        backend_name="florence-2",
        trust_remote_code=True,
        torch_dtype="float32",
        device=device,
        device_map="auto",
        attn_implementation="eager",
        use_lora=False,
        gradient_checkpointing=False,
    )
    model = Florence2MultiTaskModel(config)
    model.load()
    model.eval()
    return model


def run_task(
    model: Florence2MultiTaskModel,
    image: Image.Image,
    task_name: str,
    text_input: Optional[str],
    num_beams: int,
    max_new_tokens_cap: int,
) -> Tuple[str, Dict[str, Any], float]:
    task_cfg = FLORENCE2_TASKS[task_name]
    task_prompt = task_cfg["prompt"]
    started = time.time()
    raw_text = model.generate(
        images=image,
        task_prompt=task_prompt,
        text_input=text_input,
        max_new_tokens=min(int(task_cfg.get("max_new_tokens", 256)), max_new_tokens_cap),
        num_beams=num_beams,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    parsed = model.processor.post_process_generation(
        raw_text,
        task=task_prompt,
        image_size=image.size,
    )
    return raw_text, parsed, time.time() - started


def write_gallery(output_dir: Path, records: Sequence[Dict[str, Any]]) -> Path:
    gallery_path = output_dir / "gallery.html"
    rows = []
    for record in records:
        image_rel = os.path.relpath(record["image_path"], output_dir)
        vis_rel = os.path.relpath(record["visualization_path"], output_dir) if record.get("visualization_path") else ""
        status = record["status"]
        answer = record.get("parsed", {})
        answer_text = html.escape(json.dumps(answer, ensure_ascii=False)[:1200])
        raw = html.escape(str(record.get("raw_text", ""))[:600])
        vis_html = (
            f"<a href='{html.escape(vis_rel)}'><img src='{html.escape(vis_rel)}'></a>"
            if vis_rel
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['image_stem'])}</td>"
            f"<td>{html.escape(record['task'])}</td>"
            f"<td>{status}</td>"
            f"<td>{record.get('duration_sec', 0):.2f}s</td>"
            f"<td><a href='{html.escape(image_rel)}'><img src='{html.escape(image_rel)}'></a></td>"
            f"<td>{vis_html}</td>"
            f"<td><pre>{answer_text}</pre><details><summary>raw</summary><pre>{raw}</pre></details></td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Florence-2 COCO128 All Tasks</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #fff; text-align: left; }}
    img {{ max-width: 260px; max-height: 220px; object-fit: contain; }}
    pre {{ white-space: pre-wrap; max-width: 520px; margin: 0; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Florence-2 COCO128 All Tasks</h1>
  <p>Generated at {html.escape(dt.datetime.now().isoformat(timespec="seconds"))}</p>
  <table>
    <thead>
      <tr><th>Image</th><th>Task</th><th>Status</th><th>Time</th><th>Source</th><th>Visualization</th><th>Parsed / Raw</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    gallery_path.write_text(html_text, encoding="utf-8")
    return gallery_path


def cleanup_device(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    device = resolve_device(args.device)
    tasks = parse_task_names(args.tasks)
    names = load_names(args.data_yaml)
    images = list_images(args.dataset_root, args.split, args.max_images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_root = args.output_dir / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    LOGGER.info("Dataset: %s split=%s images=%s", args.dataset_root, args.split, len(images))
    LOGGER.info("Tasks: %s", ", ".join(tasks))
    LOGGER.info("Device: %s", device)

    model = make_model(args.model_name, device)

    records: List[Dict[str, Any]] = load_existing_records(results_path) if args.resume else []
    completed_keys = {
        record_key(record)
        for record in records
        if record_key(record) != ("", "")
        and (record.get("status") == "ok" or not args.retry_failed)
    }
    if args.resume:
        LOGGER.info("Resume: loaded %s existing records, skipping %s recorded image/task pairs", len(records), len(completed_keys))

    new_successes = 0
    new_failures = 0
    skipped = 0
    started = time.time()

    write_mode = "a" if args.resume and results_path.exists() else "w"
    with results_path.open(write_mode, encoding="utf-8") as out:
        for image_index, image_path in enumerate(images, start=1):
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            annotations = load_yolo_labels(yolo_label_path(args.dataset_root, args.split, image_path), image.size, names)
            text_inputs, prompt_metadata = build_text_inputs(annotations, image.size)
            image_vis_dir = vis_root / image_path.stem
            image_vis_dir.mkdir(parents=True, exist_ok=True)

            LOGGER.info("Image %s/%s: %s", image_index, len(images), image_path.name)
            for task_name in tasks:
                key = (str(image_path), task_name)
                task_prompt = FLORENCE2_TASKS[task_name]["prompt"]
                text_input = text_inputs.get(task_name)
                visualization_path = image_vis_dir / f"{task_name}.png"
                if args.resume and key in completed_keys:
                    skipped += 1
                    LOGGER.info("  %-34s skip recorded", task_name)
                    continue
                if args.skip_existing and visualization_path.exists():
                    skipped += 1
                    LOGGER.info("  %-34s skip existing", task_name)
                    continue

                record: Dict[str, Any] = {
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "image_stem": image_path.stem,
                    "image_size": list(image.size),
                    "task": task_name,
                    "task_prompt": task_prompt,
                    "text_input": text_input,
                    "annotations": jsonable(annotations),
                    "prompt_metadata": jsonable(prompt_metadata),
                }
                try:
                    raw_text, parsed, duration = run_task(
                        model=model,
                        image=image,
                        task_name=task_name,
                        text_input=text_input,
                        num_beams=args.num_beams,
                        max_new_tokens_cap=args.max_new_tokens_cap,
                    )
                    record.update(
                        {
                            "status": "ok",
                            "duration_sec": duration,
                            "raw_text": raw_text,
                            "parsed": jsonable(parsed),
                        }
                    )
                    if not args.no_visualizations:
                        visualize_answer(
                            image=image,
                            task_name=task_name,
                            task_prompt=task_prompt,
                            parsed=parsed,
                            selected_bbox=prompt_metadata["region_bbox"],
                            save_path=visualization_path,
                        )
                        record["visualization_path"] = str(visualization_path)
                    new_successes += 1
                    LOGGER.info("  %-34s ok %.2fs", task_name, duration)
                except Exception as exc:
                    new_failures += 1
                    record.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "visualization_path": None,
                        }
                    )
                    LOGGER.exception("  %-34s failed", task_name)

                out.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                out.flush()
                records.append(record)
                cleanup_device(device)

    gallery_path = write_gallery(args.output_dir, records)
    successes = sum(1 for record in records if record.get("status") == "ok")
    failures = sum(1 for record in records if record.get("status") not in {"ok", None})
    summary = {
        "model_name": args.model_name,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "image_count": len(images),
        "tasks": tasks,
        "device": device,
        "successes": successes,
        "failures": failures,
        "new_successes": new_successes,
        "new_failures": new_failures,
        "skipped": skipped,
        "duration_sec": time.time() - started,
        "results_path": str(results_path),
        "gallery_path": str(gallery_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info(
        "Done: %s total successes, %s total failures (%s new successes, %s new failures, %s skipped)",
        successes,
        failures,
        new_successes,
        new_failures,
        skipped,
    )
    LOGGER.info("Results: %s", results_path)
    LOGGER.info("Gallery: %s", gallery_path)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
