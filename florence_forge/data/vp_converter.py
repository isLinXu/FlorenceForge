"""Converters for visual primitive training data."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from ..core.tasks import get_task_config
from ..core.visual_primitives import (
    format_ref_box,
    format_ref_box_loc_tokens,
    normalize_bbox,
    resolve_marker_style,
    sort_boxes_left_to_right,
)
from ..evaluation.visual_primitive_parser import VisualPrimitiveParser

logger = logging.getLogger(__name__)


class VisualPrimitiveConverter:
    """Convert standard annotations into VP-style JSONL samples."""

    @staticmethod
    def coco_to_vp_od(
        coco_json_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "OD_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert COCO detection annotations to VP object detection samples."""

        logger.info("Converting COCO detection data to VP: %s -> %s", coco_json_path, output_path)
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
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for image_id, annotations in tqdm(image_annotations.items(), desc="COCO to VP OD"):
                image_info = images.get(image_id)
                if not image_info:
                    logger.warning("Image id %s not found in COCO images", image_id)
                    continue

                image_path = image_dir / image_info["file_name"]
                image_size = VisualPrimitiveConverter._resolve_image_size(image_info, image_path)
                grouped = VisualPrimitiveConverter._group_coco_annotations(
                    annotations,
                    categories,
                    image_size,
                )
                if not grouped:
                    continue

                sample = VisualPrimitiveConverter._build_sample(
                    image_path=image_path,
                    prefix=prompt,
                    suffix=VisualPrimitiveConverter._format_grouped_ref_boxes(
                        grouped,
                        box_format=box_format,
                        marker_style=marker_style,
                    ),
                    metadata={
                        "task_family": "visual_primitive",
                        "base_task": "OD",
                        "source_format": "coco",
                        "vp_box_format": box_format,
                        "vp_marker_style": marker_style,
                    },
                )
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("COCO to VP OD conversion completed: %s", output_path)

    @staticmethod
    def coco_to_vp_counting(
        coco_json_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "COUNT_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert COCO detection annotations to one counting sample per label."""

        logger.info("Converting COCO detection data to VP counting: %s -> %s", coco_json_path, output_path)
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
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for image_id, annotations in tqdm(image_annotations.items(), desc="COCO to VP count"):
                image_info = images.get(image_id)
                if not image_info:
                    logger.warning("Image id %s not found in COCO images", image_id)
                    continue

                image_path = image_dir / image_info["file_name"]
                image_size = VisualPrimitiveConverter._resolve_image_size(image_info, image_path)
                grouped = VisualPrimitiveConverter._group_coco_annotations(
                    annotations,
                    categories,
                    image_size,
                )

                for label, boxes in grouped.items():
                    count = len(boxes)
                    suffix = VisualPrimitiveConverter._format_counting_suffix(
                        label,
                        boxes,
                        box_format=box_format,
                        marker_style=marker_style,
                    )
                    sample = VisualPrimitiveConverter._build_sample(
                        image_path=image_path,
                        prefix=prompt,
                        suffix=suffix,
                        metadata={
                            "task_family": "visual_primitive",
                            "base_task": "OD",
                            "source_format": "coco",
                            "count_label": label,
                            "count": count,
                            "vp_box_format": box_format,
                            "vp_marker_style": marker_style,
                        },
                    )
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("COCO to VP counting conversion completed: %s", output_path)

    @staticmethod
    def coco_to_vp_grounding(
        coco_json_path: str,
        output_path: str,
        image_dir: str,
        task_type: str = "PHRASE_GROUNDING_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert COCO detection annotations to one query-grounding VP sample per label."""

        logger.info("Converting COCO detection data to VP grounding: %s -> %s", coco_json_path, output_path)
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
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for image_id, annotations in tqdm(image_annotations.items(), desc="COCO to VP grounding"):
                image_info = images.get(image_id)
                if not image_info:
                    logger.warning("Image id %s not found in COCO images", image_id)
                    continue

                image_path = image_dir / image_info["file_name"]
                image_size = VisualPrimitiveConverter._resolve_image_size(image_info, image_path)
                grouped = VisualPrimitiveConverter._group_coco_annotations(
                    annotations,
                    categories,
                    image_size,
                )
                VisualPrimitiveConverter._write_grounding_samples(
                    f,
                    image_path=image_path,
                    grouped=grouped,
                    prompt=prompt,
                    task_type=task_type,
                    source_format="coco",
                    box_format=box_format,
                    marker_style=marker_style,
                )

        logger.info("COCO to VP grounding conversion completed: %s", output_path)

    @staticmethod
    def yolo_to_vp_od(
        yolo_labels_dir: str,
        output_path: str,
        image_dir: str,
        classes_file: str,
        image_ext: str = ".jpg",
        task_type: str = "OD_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert YOLO detection labels to VP object detection samples."""

        logger.info("Converting YOLO labels to VP: %s -> %s", yolo_labels_dir, output_path)
        yolo_labels_dir = Path(yolo_labels_dir).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        classes_file = Path(classes_file).absolute()

        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for label_file in tqdm(list(yolo_labels_dir.glob("*.txt")), desc="YOLO to VP OD"):
                image_path = image_dir / f"{label_file.stem}{image_ext}"
                if not image_path.exists():
                    logger.warning("Image file does not exist: %s", image_path)
                    continue

                image_size = VisualPrimitiveConverter._read_image_size(image_path)
                grouped = VisualPrimitiveConverter._read_yolo_grouped_boxes(
                    label_file,
                    classes,
                    image_size,
                )
                if not grouped:
                    continue

                sample = VisualPrimitiveConverter._build_sample(
                    image_path=image_path,
                    prefix=prompt,
                    suffix=VisualPrimitiveConverter._format_grouped_ref_boxes(
                        grouped,
                        box_format=box_format,
                        marker_style=marker_style,
                    ),
                    metadata={
                        "task_family": "visual_primitive",
                        "base_task": "OD",
                        "source_format": "yolo",
                        "vp_box_format": box_format,
                        "vp_marker_style": marker_style,
                    },
                )
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("YOLO to VP OD conversion completed: %s", output_path)

    @staticmethod
    def yolo_to_vp_counting(
        yolo_labels_dir: str,
        output_path: str,
        image_dir: str,
        classes_file: str,
        image_ext: str = ".jpg",
        task_type: str = "COUNT_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert YOLO detection labels to one VP counting sample per label."""

        logger.info("Converting YOLO labels to VP counting: %s -> %s", yolo_labels_dir, output_path)
        yolo_labels_dir = Path(yolo_labels_dir).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        classes_file = Path(classes_file).absolute()

        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for label_file in tqdm(list(yolo_labels_dir.glob("*.txt")), desc="YOLO to VP count"):
                image_path = image_dir / f"{label_file.stem}{image_ext}"
                if not image_path.exists():
                    logger.warning("Image file does not exist: %s", image_path)
                    continue

                image_size = VisualPrimitiveConverter._read_image_size(image_path)
                grouped = VisualPrimitiveConverter._read_yolo_grouped_boxes(
                    label_file,
                    classes,
                    image_size,
                )

                for label, boxes in grouped.items():
                    count = len(boxes)
                    sample = VisualPrimitiveConverter._build_sample(
                        image_path=image_path,
                        prefix=prompt,
                        suffix=VisualPrimitiveConverter._format_counting_suffix(
                            label,
                            boxes,
                            box_format=box_format,
                            marker_style=marker_style,
                        ),
                        metadata={
                            "task_family": "visual_primitive",
                            "base_task": "OD",
                            "source_format": "yolo",
                            "count_label": label,
                            "count": count,
                            "vp_box_format": box_format,
                            "vp_marker_style": marker_style,
                        },
                    )
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("YOLO to VP counting conversion completed: %s", output_path)

    @staticmethod
    def yolo_to_vp_grounding(
        yolo_labels_dir: str,
        output_path: str,
        image_dir: str,
        classes_file: str,
        image_ext: str = ".jpg",
        task_type: str = "PHRASE_GROUNDING_VP",
        box_format: str = "json",
        marker_style: str = "special",
    ) -> None:
        """Convert YOLO labels to one query-grounding VP sample per label."""

        logger.info("Converting YOLO labels to VP grounding: %s -> %s", yolo_labels_dir, output_path)
        yolo_labels_dir = Path(yolo_labels_dir).absolute()
        output_path = Path(output_path).absolute()
        image_dir = Path(image_dir).absolute()
        classes_file = Path(classes_file).absolute()

        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = VisualPrimitiveConverter._get_prompt(task_type)

        with open(output_path, "w", encoding="utf-8") as f:
            for label_file in tqdm(list(yolo_labels_dir.glob("*.txt")), desc="YOLO to VP grounding"):
                image_path = image_dir / f"{label_file.stem}{image_ext}"
                if not image_path.exists():
                    logger.warning("Image file does not exist: %s", image_path)
                    continue

                image_size = VisualPrimitiveConverter._read_image_size(image_path)
                grouped = VisualPrimitiveConverter._read_yolo_grouped_boxes(
                    label_file,
                    classes,
                    image_size,
                )
                VisualPrimitiveConverter._write_grounding_samples(
                    f,
                    image_path=image_path,
                    grouped=grouped,
                    prompt=prompt,
                    task_type=task_type,
                    source_format="yolo",
                    box_format=box_format,
                    marker_style=marker_style,
                )

        logger.info("YOLO to VP grounding conversion completed: %s", output_path)

    @staticmethod
    def vp_od_jsonl_to_query_grounding(
        input_path: str,
        output_path: str,
        task_type: str = "PHRASE_GROUNDING_VP",
        box_format: str = "loc_tokens",
        marker_style: str = "plain",
    ) -> None:
        """Derive one query-grounding VP sample per label from OD_VP JSONL."""

        input_path = Path(input_path).expanduser()
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parser = VisualPrimitiveParser()
        prompt = VisualPrimitiveConverter._get_prompt(task_type)
        rows_written = 0

        with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                source_row = json.loads(line)
                detections = parser.parse_detections(str(source_row.get("suffix", "")))
                grouped: Dict[str, List[List[int]]] = defaultdict(list)
                for detection in detections:
                    label = str(detection.get("label", "")).strip()
                    bbox = detection.get("bbox")
                    if not label or not isinstance(bbox, Sequence) or len(bbox) != 4:
                        continue
                    grouped[label].append([int(value) for value in bbox])
                for sample in VisualPrimitiveConverter._iter_grounding_samples(
                    image_path=Path(str(source_row.get("image", ""))),
                    grouped=dict(grouped),
                    prompt=prompt,
                    task_type=task_type,
                    source_format=str(source_row.get("source_format", "vp_jsonl")),
                    box_format=box_format,
                    marker_style=marker_style,
                ):
                    for key, value in source_row.items():
                        if key not in sample and key not in {"prefix", "suffix", "text_input"}:
                            sample[key] = value
                    sample["source_task"] = source_row.get("prefix")
                    dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    rows_written += 1

        logger.info("VP OD JSONL to query grounding conversion completed: %s (%s rows)", output_path, rows_written)

    @staticmethod
    def _get_prompt(task_type: str) -> str:
        try:
            return get_task_config(task_type).get("prompt", f"<{task_type}>")
        except KeyError:
            return f"<{task_type}>"

    @staticmethod
    def _build_sample(
        *,
        image_path: Path,
        prefix: str,
        suffix: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        sample = {
            "image": str(image_path.absolute()),
            "prefix": prefix,
            "suffix": suffix,
        }
        sample.update(metadata)
        return sample

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

    @staticmethod
    def _read_yolo_grouped_boxes(
        label_file: Path,
        classes: Sequence[str],
        image_size: Tuple[int, int],
    ) -> Dict[str, List[List[int]]]:
        grouped: Dict[str, List[List[int]]] = defaultdict(list)
        width, height = image_size

        with open(label_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                except ValueError:
                    logger.warning("Skipping invalid YOLO row %s:%s", label_file, line_num)
                    continue
                if class_id < 0 or class_id >= len(classes):
                    logger.warning("Skipping unknown YOLO class id %s in %s:%s", class_id, label_file, line_num)
                    continue

                x1 = (x_center - box_w / 2) * width
                y1 = (y_center - box_h / 2) * height
                x2 = (x_center + box_w / 2) * width
                y2 = (y_center + box_h / 2) * height
                grouped[classes[class_id]].append(
                    normalize_bbox([x1, y1, x2, y2], image_size, input_format="xyxy")
                )

        return {
            label: sort_boxes_left_to_right(boxes)
            for label, boxes in grouped.items()
        }

    @staticmethod
    def _format_grouped_ref_boxes(
        grouped: Dict[str, List[List[int]]],
        box_format: str = "json",
        marker_style: str = "special",
    ) -> str:
        marker_style = resolve_marker_style(marker_style)
        formatter = VisualPrimitiveConverter._get_ref_box_formatter(box_format, marker_style=marker_style)
        lines = []
        for label, boxes in grouped.items():
            sorted_boxes = sort_boxes_left_to_right(boxes)
            lines.append(formatter(label, sorted_boxes))
        return "\n".join(lines)

    @staticmethod
    def _format_counting_suffix(
        label: str,
        boxes: List[List[int]],
        box_format: str = "json",
        marker_style: str = "special",
        counting_mode: str = "coarse",
        query_hint: str = "",
    ) -> str:
        sorted_boxes = sort_boxes_left_to_right(boxes)
        count = len(sorted_boxes)
        if counting_mode == "fine":
            from .tvp_converter import TVPChainBuilder
            return TVPChainBuilder.build_fine_grained_counting_chain(
                label=label,
                boxes=sorted_boxes,
                count=count,
                query_hint=query_hint,
                marker_style=resolve_marker_style(marker_style),
            )

        formatter = VisualPrimitiveConverter._get_ref_box_formatter(
            box_format,
            marker_style=resolve_marker_style(marker_style),
        )
        return (
            "1. Analyzing the request\n"
            f"The visual target is {label}.\n"
            "2. Object grounding\n"
            f"{formatter(label, sorted_boxes)}\n"
            "3. Conclusion\n"
            f"There are {count} {label} in this image."
        )

    @staticmethod
    def _get_ref_box_formatter(box_format: str, marker_style: str = "special"):
        if box_format == "json":
            return lambda label, boxes: format_ref_box(label, boxes, marker_style=marker_style)
        if box_format in {"loc_tokens", "loc"}:
            return lambda label, boxes: format_ref_box_loc_tokens(label, boxes, marker_style=marker_style)
        raise ValueError("box_format must be 'json' or 'loc_tokens'")

    @staticmethod
    def _iter_grounding_samples(
        *,
        image_path: Path,
        grouped: Dict[str, List[List[int]]],
        prompt: str,
        task_type: str,
        source_format: str,
        box_format: str,
        marker_style: str,
    ) -> Iterable[Dict[str, Any]]:
        for label, boxes in grouped.items():
            if not boxes:
                continue
            yield VisualPrimitiveConverter._build_sample(
                image_path=image_path,
                prefix=prompt,
                suffix=VisualPrimitiveConverter._format_grouped_ref_boxes(
                    {label: boxes},
                    box_format=box_format,
                    marker_style=marker_style,
                ),
                metadata={
                    "task_family": "visual_primitive",
                    "base_task": "CAPTION_TO_PHRASE_GROUNDING",
                    "source_format": source_format,
                    "query_label": label,
                    "text_input": label,
                    "query_box_count": len(boxes),
                    "vp_box_format": box_format,
                    "vp_marker_style": marker_style,
                    "vp_task_type": task_type,
                },
            )

    @staticmethod
    def _write_grounding_samples(
        fp,
        *,
        image_path: Path,
        grouped: Dict[str, List[List[int]]],
        prompt: str,
        task_type: str,
        source_format: str,
        box_format: str,
        marker_style: str,
    ) -> None:
        for sample in VisualPrimitiveConverter._iter_grounding_samples(
            image_path=image_path,
            grouped=grouped,
            prompt=prompt,
            task_type=task_type,
            source_format=source_format,
            box_format=box_format,
            marker_style=marker_style,
        ):
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _resolve_image_size(image_info: Dict[str, Any], image_path: Path) -> Tuple[int, int]:
        width = image_info.get("width")
        height = image_info.get("height")
        if width and height:
            return int(width), int(height)
        return VisualPrimitiveConverter._read_image_size(image_path)

    @staticmethod
    def _read_image_size(image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size
