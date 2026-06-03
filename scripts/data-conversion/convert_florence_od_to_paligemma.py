#!/usr/bin/env python3
"""Convert Florence OD JSONL targets to PaliGemma detection targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


LOC_BINS = 1024


def _clamp_loc(value: float) -> int:
    return max(0, min(LOC_BINS - 1, int(round(value))))


def _scale_bbox_to_paligemma_locs(
    bbox: Iterable[float],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Return PaliGemma locs in y_min, x_min, y_max, x_max order."""
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (
        _clamp_loc(y1 / height * (LOC_BINS - 1)),
        _clamp_loc(x1 / width * (LOC_BINS - 1)),
        _clamp_loc(y2 / height * (LOC_BINS - 1)),
        _clamp_loc(x2 / width * (LOC_BINS - 1)),
    )


def _loc_token(value: int) -> str:
    return f"<loc{value:04d}>"


def _format_detection_target(
    labels: List[str],
    bboxes: List[List[float]],
    image_size: Tuple[int, int],
) -> str:
    parts: List[str] = []
    for label, bbox in zip(labels, bboxes):
        y1, x1, y2, x2 = _scale_bbox_to_paligemma_locs(bbox, image_size)
        parts.append(
            f"{_loc_token(y1)}{_loc_token(x1)}{_loc_token(y2)}{_loc_token(x2)} {label}"
        )
    return " ; ".join(parts)


def _load_florence_od_payload(suffix: str) -> Dict[str, Any]:
    payload = json.loads(suffix)
    if "<OD>" in payload:
        return payload["<OD>"]
    if "bboxes" in payload and "labels" in payload:
        return payload
    raise ValueError("suffix does not contain Florence OD payload")


def convert_file(input_jsonl: Path, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    converted = 0

    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image"])
            with Image.open(image_path) as image:
                image_size = image.size

            od_payload = _load_florence_od_payload(record["suffix"])
            labels = od_payload.get("labels", [])
            bboxes = od_payload.get("bboxes", [])
            if len(labels) != len(bboxes):
                raise ValueError(
                    f"{input_jsonl}:{line_number} has {len(labels)} labels "
                    f"but {len(bboxes)} bboxes"
                )

            out_record = dict(record)
            out_record["prefix"] = "detect"
            out_record["suffix"] = _format_detection_target(labels, bboxes, image_size)
            out_record["source_format"] = "florence_od"
            out_record["target_format"] = "paligemma_detection"
            dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            converted += 1

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Florence OD JSONL into PaliGemma detection JSONL."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_file(args.input_jsonl, args.output_jsonl)
    print(f"Converted {count} samples -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
