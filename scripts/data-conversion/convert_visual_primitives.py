#!/usr/bin/env python3
"""Convert detection annotations into FlorenceForge visual primitive JSONL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.data import VisualPrimitiveConverter


def _add_marker_style_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--marker-style",
        default="special",
        choices=["special", "plain"],
        help="VP wrapper marker style",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert COCO/YOLO detection data to visual primitive JSONL.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    coco_od = subparsers.add_parser("coco-od", help="COCO detection -> OD_VP")
    coco_od.add_argument("--json-file", required=True, help="COCO JSON annotation file")
    coco_od.add_argument("--images-dir", required=True, help="Image directory")
    coco_od.add_argument("--output", "-o", required=True, help="Output JSONL path")
    coco_od.add_argument("--task-type", default="OD_VP", choices=["OD_VP"], help="VP task type")
    coco_od.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(coco_od)

    coco_count = subparsers.add_parser("coco-count", help="COCO detection -> COUNT_VP")
    coco_count.add_argument("--json-file", required=True, help="COCO JSON annotation file")
    coco_count.add_argument("--images-dir", required=True, help="Image directory")
    coco_count.add_argument("--output", "-o", required=True, help="Output JSONL path")
    coco_count.add_argument("--task-type", default="COUNT_VP", choices=["COUNT_VP"], help="VP task type")
    coco_count.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(coco_count)

    coco_grounding = subparsers.add_parser("coco-grounding", help="COCO detection -> query grounding VP")
    coco_grounding.add_argument("--json-file", required=True, help="COCO JSON annotation file")
    coco_grounding.add_argument("--images-dir", required=True, help="Image directory")
    coco_grounding.add_argument("--output", "-o", required=True, help="Output JSONL path")
    coco_grounding.add_argument(
        "--task-type",
        default="PHRASE_GROUNDING_VP",
        choices=["PHRASE_GROUNDING_VP", "OPEN_VOCABULARY_DETECTION"],
        help="VP query task type",
    )
    coco_grounding.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(coco_grounding)

    yolo_od = subparsers.add_parser("yolo-od", help="YOLO labels -> OD_VP")
    yolo_od.add_argument("--labels-dir", required=True, help="YOLO label directory")
    yolo_od.add_argument("--images-dir", required=True, help="Image directory")
    yolo_od.add_argument("--classes-file", required=True, help="Class names file")
    yolo_od.add_argument("--output", "-o", required=True, help="Output JSONL path")
    yolo_od.add_argument("--image-ext", default=".jpg", help="Image extension")
    yolo_od.add_argument("--task-type", default="OD_VP", choices=["OD_VP"], help="VP task type")
    yolo_od.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(yolo_od)

    yolo_count = subparsers.add_parser("yolo-count", help="YOLO labels -> COUNT_VP")
    yolo_count.add_argument("--labels-dir", required=True, help="YOLO label directory")
    yolo_count.add_argument("--images-dir", required=True, help="Image directory")
    yolo_count.add_argument("--classes-file", required=True, help="Class names file")
    yolo_count.add_argument("--output", "-o", required=True, help="Output JSONL path")
    yolo_count.add_argument("--image-ext", default=".jpg", help="Image extension")
    yolo_count.add_argument("--task-type", default="COUNT_VP", choices=["COUNT_VP"], help="VP task type")
    yolo_count.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(yolo_count)

    yolo_grounding = subparsers.add_parser("yolo-grounding", help="YOLO labels -> query grounding VP")
    yolo_grounding.add_argument("--labels-dir", required=True, help="YOLO label directory")
    yolo_grounding.add_argument("--images-dir", required=True, help="Image directory")
    yolo_grounding.add_argument("--classes-file", required=True, help="Class names file")
    yolo_grounding.add_argument("--output", "-o", required=True, help="Output JSONL path")
    yolo_grounding.add_argument("--image-ext", default=".jpg", help="Image extension")
    yolo_grounding.add_argument(
        "--task-type",
        default="PHRASE_GROUNDING_VP",
        choices=["PHRASE_GROUNDING_VP", "OPEN_VOCABULARY_DETECTION"],
        help="VP query task type",
    )
    yolo_grounding.add_argument("--box-format", default="json", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(yolo_grounding)

    jsonl_grounding = subparsers.add_parser("jsonl-grounding", help="OD_VP JSONL -> query grounding VP")
    jsonl_grounding.add_argument("--input", required=True, help="Input OD_VP JSONL path")
    jsonl_grounding.add_argument("--output", "-o", required=True, help="Output JSONL path")
    jsonl_grounding.add_argument(
        "--task-type",
        default="PHRASE_GROUNDING_VP",
        choices=["PHRASE_GROUNDING_VP", "OPEN_VOCABULARY_DETECTION"],
        help="VP query task type",
    )
    jsonl_grounding.add_argument("--box-format", default="loc_tokens", choices=["json", "loc_tokens"], help="VP box payload format")
    _add_marker_style_arg(jsonl_grounding)

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "coco-od":
        VisualPrimitiveConverter.coco_to_vp_od(
            coco_json_path=args.json_file,
            output_path=args.output,
            image_dir=args.images_dir,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "coco-count":
        VisualPrimitiveConverter.coco_to_vp_counting(
            coco_json_path=args.json_file,
            output_path=args.output,
            image_dir=args.images_dir,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "coco-grounding":
        VisualPrimitiveConverter.coco_to_vp_grounding(
            coco_json_path=args.json_file,
            output_path=args.output,
            image_dir=args.images_dir,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "yolo-od":
        VisualPrimitiveConverter.yolo_to_vp_od(
            yolo_labels_dir=args.labels_dir,
            output_path=args.output,
            image_dir=args.images_dir,
            classes_file=args.classes_file,
            image_ext=args.image_ext,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "yolo-count":
        VisualPrimitiveConverter.yolo_to_vp_counting(
            yolo_labels_dir=args.labels_dir,
            output_path=args.output,
            image_dir=args.images_dir,
            classes_file=args.classes_file,
            image_ext=args.image_ext,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "yolo-grounding":
        VisualPrimitiveConverter.yolo_to_vp_grounding(
            yolo_labels_dir=args.labels_dir,
            output_path=args.output,
            image_dir=args.images_dir,
            classes_file=args.classes_file,
            image_ext=args.image_ext,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    elif args.mode == "jsonl-grounding":
        VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
            input_path=args.input,
            output_path=args.output,
            task_type=args.task_type,
            box_format=args.box_format,
            marker_style=args.marker_style,
        )
    else:
        parser.error(f"Unsupported mode: {args.mode}")

    logging.info("Visual primitive conversion completed: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
