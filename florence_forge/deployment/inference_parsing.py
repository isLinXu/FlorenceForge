"""Florence-2 推理输出解析（从 ``inference.py`` 抽出）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def clean_text_prefix(text: str) -> str:
    return re.split(r"[>＞]", text)[-1].strip()


def parse_florence2_output(
    output_text: str, image_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    detections = []
    pattern = r"(?P<label>[^<]+)<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)><loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"
    image_width, image_height = image_size

    for match in re.finditer(pattern, output_text):
        label = match.group("label")
        label = label.replace("</s>", "").replace("<s>", "").strip()
        label = clean_text_prefix(label)
        label = label.strip(" _")
        if not label:
            continue
        x1 = int(match.group("x1")) * image_width / 1000
        y1 = int(match.group("y1")) * image_height / 1000
        x2 = int(match.group("x2")) * image_width / 1000
        y2 = int(match.group("y2")) * image_height / 1000
        detections.append(
            {
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": 1.0,
            }
        )
    return detections


def parse_ocr_with_region(
    model_output: str, image_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    width, height = image_size
    results = []
    cleaned_output = model_output.replace("<s>", "").replace("</s>", "").strip()
    pattern = r"([^<]+)((?:<loc_\d+>){8})"
    for text, loc_str in re.findall(pattern, cleaned_output):
        text = text.replace("</s>", "").replace("<s>", "").strip()
        text = clean_text_prefix(text)
        if not text:
            continue
        loc_matches = re.findall(r"\d+", loc_str)
        if len(loc_matches) == 8:
            coords = [int(c) for c in loc_matches]
            polygon = [
                (coords[0] * width // 1000, coords[1] * height // 1000),
                (coords[2] * width // 1000, coords[3] * height // 1000),
                (coords[4] * width // 1000, coords[5] * height // 1000),
                (coords[6] * width // 1000, coords[7] * height // 1000),
            ]
            results.append({"text": text, "polygon": polygon})
    return results


def parse_bboxes(
    model_output: str, image_size: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    width, height = image_size
    bboxes = []
    bbox_matches = re.findall(
        r"\<loc_(\d+)>\<loc_(\d+)>\<loc_(\d+)>\<loc_(\d+)>", model_output
    )
    for match in bbox_matches:
        xmin = int(match[0]) * width // 1000
        ymin = int(match[1]) * height // 1000
        xmax = int(match[2]) * width // 1000
        ymax = int(match[3]) * height // 1000
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


def parse_segmentation_output(
    output_text: str, image_size: Tuple[int, int]
) -> List[List[Tuple[int, int]]]:
    all_polygons = []
    image_width, image_height = image_size
    polygon_texts = re.findall(r"(?:<loc_\d+><loc_\d+>)+", output_text)
    for poly_text in polygon_texts:
        polygon = []
        for match in re.finditer(r"<loc_(?P<x>\d+)><loc_(?P<y>\d+)>", poly_text):
            x = int(int(match.group("x")) * image_width / 1000)
            y = int(int(match.group("y")) * image_height / 1000)
            polygon.append((x, y))
        if polygon:
            all_polygons.append(polygon)
    return all_polygons
