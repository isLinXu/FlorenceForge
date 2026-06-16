#!/usr/bin/env python3
"""Real Florence-VP training smoke on real weights and real YOLO data.

This is intentionally tiny: it converts a small COCO128/YOLO slice to VP JSONL,
loads a real Florence-2 checkpoint, runs a real forward pass, and performs one
or a few gradient updates. It supports both a narrow parameter-slice smoke and a
real LoRA smoke without writing a full checkpoint by default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.core.config import LoRAConfig
from florence_forge.evaluation.metrics import VisualPrimitiveDetectionMetrics
from florence_forge.utils.diagnostics import DEFAULT_MODEL_ID, find_local_hf_snapshot


DEFAULT_DATASET_ROOT = Path.home() / "PycharmProjects" / "datasets" / "coco128"
DEFAULT_LOCAL_MODEL_CANDIDATES = (
    Path.home() / "Downloads" / "Florence2_det_base_ovd-v3-1751283651704-model",
)
DEFAULT_TRAINABLE_MATCH = "language_model.model.decoder.layers.5.fc2"
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
DEFAULT_LORA_MODULES_TO_SAVE = ("lm_head", "model.shared")

COCO80_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_model_path(model_id: str, model_path: Optional[str]) -> str:
    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Explicit --model-path does not exist: {path}")
        return str(path)

    for candidate in DEFAULT_LOCAL_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    snapshot = find_local_hf_snapshot(model_id)
    if snapshot is not None:
        return str(snapshot)

    raise FileNotFoundError(
        f"No local Florence checkpoint found for {model_id}. "
        "Pass --model-path or download the model before running this smoke."
    )


def _dataset_paths(dataset_root: Path) -> Dict[str, Path]:
    images_dir = dataset_root / "images" / "train2017"
    labels_dir = dataset_root / "labels" / "train2017"
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO128 images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"COCO128 YOLO labels directory not found: {labels_dir}")
    return {"images_dir": images_dir, "labels_dir": labels_dir}


def _write_coco80_classes(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(COCO80_CLASSES) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_short_rows(rows: Sequence[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            len(str(row.get("suffix", ""))),
            str(row.get("image", "")),
            str(row.get("count_label", "")),
        ),
    )[:total]


def _select_query_grounding_rows(
    rows: Sequence[Dict[str, Any]],
    total: int,
    selection: str,
) -> List[Dict[str, Any]]:
    if total <= 0:
        return []
    if selection == "shortest-query":
        return _select_short_rows(rows, total)
    if selection != "multi-instance":
        raise ValueError(f"Unsupported grounding selection mode: {selection}")

    return sorted(
        rows,
        key=lambda row: (
            -_query_box_count(row),
            str(row.get("query_label", row.get("text_input", ""))),
            str(row.get("image", "")),
            len(str(row.get("suffix", ""))),
        ),
    )[:total]


def _query_box_count(row: Dict[str, Any]) -> int:
    for key in ("query_box_count", "curriculum_query_box_count", "count"):
        value = row.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass

    suffix = str(row.get("suffix", ""))
    loc_token_count = suffix.count("<loc_")
    if loc_token_count:
        return loc_token_count // 4
    return 0


def _summarize_query_grounding_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = {"single": 0, "medium": 0, "dense": 0}
    box_counts: List[int] = []
    for row in rows:
        box_count = _query_box_count(row)
        bucket_counts[_query_bucket(box_count)] += 1
        box_counts.append(box_count)
    return {
        "rows": len(rows),
        "bucket_counts": bucket_counts,
        "avg_query_box_count": _mean(box_counts),
        "max_query_box_count": max(box_counts) if box_counts else 0,
    }


def _filter_query_grounding_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_query_boxes: Optional[int],
    max_query_boxes: Optional[int],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if min_query_boxes is not None and min_query_boxes < 0:
        raise ValueError("--grounding-min-query-boxes must be >= 0")
    if max_query_boxes is not None and max_query_boxes < 0:
        raise ValueError("--grounding-max-query-boxes must be >= 0")
    if (
        min_query_boxes is not None
        and max_query_boxes is not None
        and min_query_boxes > max_query_boxes
    ):
        raise ValueError("--grounding-min-query-boxes cannot exceed --grounding-max-query-boxes")

    kept_rows: List[Dict[str, Any]] = []
    skipped_rows = 0
    for row in rows:
        box_count = _query_box_count(row)
        if min_query_boxes is not None and box_count < min_query_boxes:
            skipped_rows += 1
            continue
        if max_query_boxes is not None and box_count > max_query_boxes:
            skipped_rows += 1
            continue
        kept_rows.append(dict(row))

    summary = {
        "input_rows": len(rows),
        "output_rows": len(kept_rows),
        "skipped_rows": skipped_rows,
        "min_query_boxes": min_query_boxes,
        "max_query_boxes": max_query_boxes,
        "input_stats": _summarize_query_grounding_rows(rows),
        "output_stats": _summarize_query_grounding_rows(kept_rows),
    }
    return kept_rows, summary


def _apply_query_count_hints(
    rows: Sequence[Dict[str, Any]],
    *,
    template: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rewrite query ``text_input`` with an explicit target box-count hint."""

    hinted_rows: List[Dict[str, Any]] = []
    changed_rows = 0
    box_counts: List[int] = []
    for row in rows:
        item = dict(row)
        original_text_input = str(
            item.get("text_input")
            or item.get("query_label")
            or item.get("count_label")
            or ""
        )
        query_label = str(item.get("query_label") or original_text_input).strip()
        box_count = _query_box_count(item)
        hinted_text_input = _format_query_count_hint(
            template,
            label=query_label,
            text_input=original_text_input,
            query_box_count=box_count,
        )
        if hinted_text_input != original_text_input:
            changed_rows += 1
        item["query_label"] = query_label
        item["text_input"] = hinted_text_input
        item["count_hint_original_text_input"] = original_text_input
        item["count_hint_text_input"] = hinted_text_input
        item["count_hint_query_box_count"] = box_count
        item["count_hint_template"] = template
        hinted_rows.append(item)
        box_counts.append(box_count)

    return hinted_rows, {
        "template": template,
        "input_rows": len(rows),
        "output_rows": len(hinted_rows),
        "changed_text_input_rows": changed_rows,
        "avg_query_box_count": _mean(box_counts),
        "max_query_box_count": max(box_counts) if box_counts else 0,
        "stats": _summarize_query_grounding_rows(hinted_rows),
    }


def _apply_query_count_hints_for_splits(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    template: str,
    splits: str,
) -> tuple[Path, Path, Dict[str, Any]]:
    if splits not in {"both", "train", "val"}:
        raise ValueError("splits must be one of: both, train, val")

    effective_train_path = train_path
    effective_val_path = val_path
    summary: Dict[str, Any] = {
        "template": template,
        "splits": splits,
        "train": None,
        "val": None,
    }

    if splits in {"both", "train"}:
        count_hint_train_path = output_dir / "train_grounding_count_hint_vp.jsonl"
        count_hint_train_rows, count_hint_train_summary = _apply_query_count_hints(
            _read_jsonl(effective_train_path),
            template=template,
        )
        _write_jsonl(count_hint_train_path, count_hint_train_rows)
        count_hint_train_summary["path"] = str(count_hint_train_path)
        count_hint_train_summary["source_path"] = str(effective_train_path)
        effective_train_path = count_hint_train_path
        summary["train"] = count_hint_train_summary

    if splits in {"both", "val"}:
        count_hint_val_path = output_dir / "val_grounding_count_hint_vp.jsonl"
        count_hint_val_rows, count_hint_val_summary = _apply_query_count_hints(
            _read_jsonl(effective_val_path),
            template=template,
        )
        _write_jsonl(count_hint_val_path, count_hint_val_rows)
        count_hint_val_summary["path"] = str(count_hint_val_path)
        count_hint_val_summary["source_path"] = str(effective_val_path)
        effective_val_path = count_hint_val_path
        summary["val"] = count_hint_val_summary

    return effective_train_path, effective_val_path, summary


def _format_query_count_hint(
    template: str,
    *,
    label: str,
    text_input: str,
    query_box_count: int,
) -> str:
    return (
        str(template)
        .replace("{label}", str(label))
        .replace("{text_input}", str(text_input))
        .replace("{query_box_count}", str(query_box_count))
        .replace("{count}", str(query_box_count))
    )


def _resolve_optional_file(value: Optional[str], name: str) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path


def _order_data_configs(
    data_configs: Sequence[Dict[str, Any]],
    *,
    training_data_order: str,
    grounding_task_type: str,
) -> List[Dict[str, Any]]:
    if training_data_order == "as-is":
        return list(data_configs)
    if training_data_order != "grounding-first":
        raise ValueError(f"Unsupported training data order: {training_data_order}")
    return sorted(
        data_configs,
        key=lambda config: 0 if config.get("task_type") == grounding_task_type else 1,
    )


def _query_bucket(box_count: int) -> str:
    if box_count <= 1:
        return "single"
    if box_count <= 3:
        return "medium"
    return "dense"


def _build_query_curriculum_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    single_weight: int,
    medium_weight: int,
    dense_weight: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    weights = {
        "single": _validate_non_negative_int(single_weight, "single_weight"),
        "medium": _validate_non_negative_int(medium_weight, "medium_weight"),
        "dense": _validate_non_negative_int(dense_weight, "dense_weight"),
    }
    bucket_counts = {bucket: 0 for bucket in weights}
    bucket_output_counts = {bucket: 0 for bucket in weights}
    box_counts: List[int] = []
    output_box_counts: List[int] = []
    curriculum_rows: List[Dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        box_count = _safe_int(row.get("query_box_count"), default=0)
        bucket = _query_bucket(box_count)
        repeat_total = weights[bucket]
        bucket_counts[bucket] += 1
        box_counts.append(box_count)
        for repeat_index in range(repeat_total):
            item = dict(row)
            item["curriculum_bucket"] = bucket
            item["curriculum_query_box_count"] = box_count
            item["curriculum_repeat_index"] = repeat_index
            item["curriculum_repeat_total"] = repeat_total
            item["curriculum_source_index"] = source_index
            curriculum_rows.append(item)
            bucket_output_counts[bucket] += 1
            output_box_counts.append(box_count)

    summary = {
        "weights": weights,
        "input_rows": len(rows),
        "output_rows": len(curriculum_rows),
        "bucket_counts": bucket_counts,
        "bucket_output_counts": bucket_output_counts,
        "avg_query_box_count_input": _mean(box_counts),
        "avg_query_box_count_output": _mean(output_box_counts),
        "max_query_box_count": max(box_counts) if box_counts else 0,
    }
    return curriculum_rows, summary


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_non_negative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _materialize_real_vp_data(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    from florence_forge.data import VisualPrimitiveConverter

    dataset_root = Path(args.dataset_root).expanduser()
    paths = _dataset_paths(dataset_root)
    classes_file = Path(args.classes_file).expanduser() if args.classes_file else output_dir / "coco80.names"
    if not args.classes_file:
        _write_coco80_classes(classes_file)
    elif not classes_file.exists():
        raise FileNotFoundError(f"--classes-file does not exist: {classes_file}")

    all_od_path = output_dir / "coco128_yolo_od_vp_all.jsonl"
    VisualPrimitiveConverter.yolo_to_vp_od(
        yolo_labels_dir=str(paths["labels_dir"]),
        output_path=str(all_od_path),
        image_dir=str(paths["images_dir"]),
        classes_file=str(classes_file),
        image_ext=args.image_ext,
        task_type="OD_VP",
        box_format=args.vp_box_format,
        marker_style=args.vp_marker_style,
    )

    od_rows = _select_short_rows(
        _read_jsonl(all_od_path),
        args.max_train_samples + args.max_val_samples,
    )
    train_od_path = output_dir / "train_od_vp.jsonl"
    val_od_path = output_dir / "val_od_vp.jsonl"
    _write_jsonl(train_od_path, od_rows[: args.max_train_samples])
    _write_jsonl(val_od_path, od_rows[args.max_train_samples:])

    data_configs = []
    skip_od_training_data = bool(args.skip_od_training_data)
    if not skip_od_training_data:
        data_configs.append({"task_type": "OD_VP", "data_path": str(train_od_path)})
    manifest: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "images_dir": str(paths["images_dir"]),
        "labels_dir": str(paths["labels_dir"]),
        "classes_file": str(classes_file),
        "all_od_path": str(all_od_path),
        "train_od_path": str(train_od_path),
        "val_od_path": str(val_od_path),
        "all_od_rows": len(_read_jsonl(all_od_path)),
        "train_od_rows": len(_read_jsonl(train_od_path)),
        "val_od_rows": len(_read_jsonl(val_od_path)),
        "skip_od_training_data": skip_od_training_data,
    }

    if args.include_count:
        all_count_path = output_dir / "coco128_yolo_count_vp_all.jsonl"
        VisualPrimitiveConverter.yolo_to_vp_counting(
            yolo_labels_dir=str(paths["labels_dir"]),
            output_path=str(all_count_path),
            image_dir=str(paths["images_dir"]),
            classes_file=str(classes_file),
            image_ext=args.image_ext,
            task_type="COUNT_VP",
            box_format=args.vp_box_format,
            marker_style=args.vp_marker_style,
        )
        count_rows = _select_short_rows(
            _read_jsonl(all_count_path),
            args.max_train_samples + args.max_val_samples,
        )
        train_count_path = output_dir / "train_count_vp.jsonl"
        val_count_path = output_dir / "val_count_vp.jsonl"
        _write_jsonl(train_count_path, count_rows[: args.max_train_samples])
        _write_jsonl(val_count_path, count_rows[args.max_train_samples:])
        data_configs.append({"task_type": "COUNT_VP", "data_path": str(train_count_path)})
        manifest.update({
            "all_count_path": str(all_count_path),
            "train_count_path": str(train_count_path),
            "val_count_path": str(val_count_path),
            "all_count_rows": len(_read_jsonl(all_count_path)),
            "train_count_rows": len(_read_jsonl(train_count_path)),
            "val_count_rows": len(_read_jsonl(val_count_path)),
        })

    if args.include_grounding:
        external_train_grounding_path = _resolve_optional_file(
            args.grounding_train_path,
            "--grounding-train-path",
        )
        external_val_grounding_path = _resolve_optional_file(
            args.grounding_val_path,
            "--grounding-val-path",
        )
        train_grounding_path = external_train_grounding_path or output_dir / "train_grounding_vp.jsonl"
        val_grounding_path = external_val_grounding_path or output_dir / "val_grounding_vp.jsonl"
        all_grounding_path: Optional[Path] = None
        grounding_source = "external-train" if external_train_grounding_path else "derived-od-split"

        if external_train_grounding_path is None:
            if args.grounding_selection == "od-split":
                VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
                    input_path=str(train_od_path),
                    output_path=str(train_grounding_path),
                    task_type=args.grounding_task_type,
                    box_format=args.vp_box_format,
                    marker_style=args.vp_marker_style,
                )
            else:
                all_grounding_path = output_dir / "all_grounding_vp.jsonl"
                VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
                    input_path=str(all_od_path),
                    output_path=str(all_grounding_path),
                    task_type=args.grounding_task_type,
                    box_format=args.vp_box_format,
                    marker_style=args.vp_marker_style,
                )
                selected_grounding_rows = _select_query_grounding_rows(
                    _read_jsonl(all_grounding_path),
                    args.max_train_samples + args.max_val_samples,
                    args.grounding_selection,
                )
                _write_jsonl(train_grounding_path, selected_grounding_rows[: args.max_train_samples])
                if external_val_grounding_path is None:
                    _write_jsonl(val_grounding_path, selected_grounding_rows[args.max_train_samples:])
                grounding_source = f"derived-{args.grounding_selection}"

        if external_val_grounding_path is None and (
            external_train_grounding_path is not None or args.grounding_selection == "od-split"
        ):
            VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
                input_path=str(val_od_path),
                output_path=str(val_grounding_path),
                task_type=args.grounding_task_type,
                box_format=args.vp_box_format,
                marker_style=args.vp_marker_style,
            )

        effective_train_grounding_path = train_grounding_path
        effective_val_grounding_path = val_grounding_path
        grounding_curriculum_summary: Optional[Dict[str, Any]] = None
        grounding_curriculum_skipped_reason: Optional[str] = None
        if external_train_grounding_path is not None and args.grounding_curriculum != "none":
            grounding_curriculum_skipped_reason = "external_train_path_used_as_effective_train_path"
        elif args.grounding_curriculum == "multi-instance":
            curriculum_path = output_dir / "train_grounding_curriculum_vp.jsonl"
            curriculum_rows, grounding_curriculum_summary = _build_query_curriculum_rows(
                _read_jsonl(train_grounding_path),
                single_weight=args.grounding_single_weight,
                medium_weight=args.grounding_medium_weight,
                dense_weight=args.grounding_dense_weight,
            )
            _write_jsonl(curriculum_path, curriculum_rows)
            grounding_curriculum_summary["path"] = str(curriculum_path)
            effective_train_grounding_path = curriculum_path

        grounding_filter_summary: Optional[Dict[str, Any]] = None
        if args.grounding_min_query_boxes is not None or args.grounding_max_query_boxes is not None:
            filtered_path = output_dir / "train_grounding_filtered_vp.jsonl"
            filtered_rows, grounding_filter_summary = _filter_query_grounding_rows(
                _read_jsonl(effective_train_grounding_path),
                min_query_boxes=args.grounding_min_query_boxes,
                max_query_boxes=args.grounding_max_query_boxes,
            )
            if not filtered_rows:
                raise RuntimeError(
                    "Grounding query-box filter removed all train rows; "
                    "adjust --grounding-min-query-boxes/--grounding-max-query-boxes."
                )
            _write_jsonl(filtered_path, filtered_rows)
            grounding_filter_summary["path"] = str(filtered_path)
            effective_train_grounding_path = filtered_path

        grounding_count_hint_summary: Optional[Dict[str, Any]] = None
        if args.grounding_count_hint_template:
            (
                effective_train_grounding_path,
                effective_val_grounding_path,
                grounding_count_hint_summary,
            ) = _apply_query_count_hints_for_splits(
                train_path=effective_train_grounding_path,
                val_path=effective_val_grounding_path,
                output_dir=output_dir,
                template=args.grounding_count_hint_template,
                splits=args.grounding_count_hint_splits,
            )

        data_configs.append({"task_type": args.grounding_task_type, "data_path": str(effective_train_grounding_path)})
        manifest.update({
            "all_grounding_path": str(all_grounding_path) if all_grounding_path else None,
            "train_grounding_path": str(train_grounding_path),
            "train_grounding_effective_path": str(effective_train_grounding_path),
            "val_grounding_path": str(effective_val_grounding_path),
            "val_grounding_original_path": str(val_grounding_path),
            "train_grounding_rows": len(_read_jsonl(train_grounding_path)),
            "train_grounding_effective_rows": len(_read_jsonl(effective_train_grounding_path)),
            "val_grounding_rows": len(_read_jsonl(effective_val_grounding_path)),
            "grounding_task_type": args.grounding_task_type,
            "grounding_curriculum": args.grounding_curriculum,
            "grounding_selection": args.grounding_selection,
            "grounding_source": grounding_source,
            "grounding_train_path_external": str(external_train_grounding_path) if external_train_grounding_path else None,
            "grounding_val_path_external": str(external_val_grounding_path) if external_val_grounding_path else None,
            "train_grounding_stats": _summarize_query_grounding_rows(_read_jsonl(train_grounding_path)),
            "train_grounding_effective_stats": _summarize_query_grounding_rows(_read_jsonl(effective_train_grounding_path)),
            "val_grounding_stats": _summarize_query_grounding_rows(_read_jsonl(effective_val_grounding_path)),
        })
        if grounding_curriculum_summary is not None:
            manifest["grounding_curriculum_summary"] = grounding_curriculum_summary
        if grounding_curriculum_skipped_reason is not None:
            manifest["grounding_curriculum_skipped_reason"] = grounding_curriculum_skipped_reason
        if grounding_filter_summary is not None:
            manifest["grounding_filter_summary"] = grounding_filter_summary
        if grounding_count_hint_summary is not None:
            manifest["grounding_count_hint_summary"] = grounding_count_hint_summary

    if not data_configs:
        raise RuntimeError(
            "No training data configs were materialized; disable --skip-od-training-data "
            "or enable another training task such as --include-grounding."
        )

    data_configs = _order_data_configs(
        data_configs,
        training_data_order=args.training_data_order,
        grounding_task_type=args.grounding_task_type,
    )
    manifest["data_configs"] = data_configs
    manifest["training_data_order"] = args.training_data_order
    manifest_path = output_dir / "vp_real_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _float_loss(output: Any):
    if hasattr(output, "loss"):
        return output.loss
    return output["loss"]


def _select_trainable_parameters(model: Any, name_fragment: str) -> List[Any]:
    selected = []
    for param in model.parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if name_fragment in name:
            param.requires_grad_(True)
            selected.append(param)

    if not selected:
        suggestions = [
            name for name, _ in model.named_parameters()
            if "decoder.layers" in name or "language_model" in name
        ][:20]
        raise RuntimeError(
            f"No parameters matched --trainable-match={name_fragment!r}. "
            f"Example parameter names: {suggestions}"
        )
    return selected


def _select_existing_trainable_parameters(model: Any) -> List[Any]:
    selected = [param for param in model.parameters() if param.requires_grad]
    if not selected:
        raise RuntimeError("No trainable parameters are enabled")
    return selected


def _count_parameter_values(parameters: Iterable[Any]) -> int:
    return int(sum(param.numel() for param in parameters))


def _count_model_parameter_values(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _clone_parameters(parameters: Iterable[Any]) -> List[Any]:
    return [param.detach().clone() for param in parameters]


def _parameter_delta_norm(before: Iterable[Any], after: Iterable[Any]) -> float:
    total = 0.0
    for before_param, after_param in zip(before, after):
        delta = after_param.detach() - before_param
        total += float(delta.float().norm().detach().cpu())
    return total


def _reference_vp_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metrics = VisualPrimitiveDetectionMetrics("OD_VP")
    refs = [str(row.get("suffix", "")) for row in rows]
    metrics.add_batch(refs, refs)
    return {
        key: float(value)
        for key, value in metrics.compute().items()
        if isinstance(value, (int, float))
    }


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from florence_forge.core.config import DataConfig, ModelConfig
    from florence_forge.core.model import Florence2MultiTaskModel
    from florence_forge.data.collate import Florence2Collator
    from florence_forge.data.dataset import MultiTaskDataset

    device = _choose_device(args.device)
    model_name = _resolve_model_path(args.model_id, args.model_path)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="florenceforge_real_vp_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()
    data_manifest = _materialize_real_vp_data(args, output_dir)
    reference_rows = _read_jsonl(Path(data_manifest["train_od_path"]))

    summary: Dict[str, Any] = {
        "ok": False,
        "model_name": model_name,
        "device": device,
        "output_dir": str(output_dir),
        "data": data_manifest,
        "reference_vp_metrics": _reference_vp_metrics(reference_rows),
        "training_mode": args.training_mode,
        "trainable_match": args.trainable_match,
        "lora_task_type": args.lora_task_type,
        "lora_target_modules": args.lora_target_modules,
        "lora_modules_to_save": args.lora_modules_to_save,
        "vp_box_format": args.vp_box_format,
        "vp_marker_style": args.vp_marker_style,
        "max_steps": args.max_steps,
    }

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        modules_to_save=args.lora_modules_to_save,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=args.lora_task_type,
    )
    config = ModelConfig(
        model_name=model_name,
        backend_name="florence-2",
        device=device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        use_lora=args.training_mode == "lora",
        lora_config=lora_config,
        attn_implementation="eager",
        enable_visual_primitives=True,
    )

    start = time.time()
    model = Florence2MultiTaskModel(config).load()
    summary["model_load_sec"] = round(time.time() - start, 3)
    summary["backend_device"] = str(model._backend.device)
    summary["backend_dtype"] = str(model._backend.dtype)
    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is not None:
        summary["tokenizer_vocab_size"] = int(len(tokenizer))

    dataset = MultiTaskDataset(
        data_configs=data_manifest["data_configs"],
        image_base_path="",
        config=DataConfig(use_cache=False, num_workers=0, batch_size=args.batch_size),
        processor=model.processor,
        backend=model._backend,
    )
    if len(dataset) == 0:
        raise RuntimeError("VP training dataset is empty after conversion")

    collator = Florence2Collator(pad_token_id=dataset._get_pad_token_id())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train_data,
        num_workers=0,
        collate_fn=collator,
        generator=(
            torch.Generator().manual_seed(args.shuffle_seed)
            if args.shuffle_train_data and args.shuffle_seed is not None else None
        ),
    )
    summary["dataset_size"] = len(dataset)
    summary["batch_size"] = args.batch_size
    summary["training_data_order"] = args.training_data_order
    summary["shuffle_train_data"] = bool(args.shuffle_train_data)
    summary["shuffle_seed"] = args.shuffle_seed if args.shuffle_train_data else None

    if args.training_mode == "slice":
        selected = _select_trainable_parameters(model, args.trainable_match)
    else:
        selected = _select_existing_trainable_parameters(model)
    before_params = _clone_parameters(selected)
    optimizer = torch.optim.AdamW(selected, lr=args.learning_rate)
    total_param_values = _count_model_parameter_values(model)
    trainable_param_values = _count_parameter_values(selected)
    summary.update({
        "total_parameter_values": total_param_values,
        "trainable_parameter_values": trainable_param_values,
        "trainable_parameter_ratio": (
            trainable_param_values / total_param_values
            if total_param_values else 0.0
        ),
    })

    model.train()
    losses: List[float] = []
    grad_norms: List[float] = []
    start = time.time()
    for step, batch in zip(range(args.max_steps), itertools.cycle(dataloader)):
        output = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = _float_loss(output)
        loss.backward()
        grad_norm = 0.0
        for param in selected:
            if param.grad is not None:
                grad_norm += float(param.grad.detach().float().norm().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(grad_norm)

        if step == 0:
            summary["first_batch_shapes"] = {
                key: list(value.shape)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            summary["first_batch_task_type"] = batch.get("task_type")
            summary["first_batch_answer_preview"] = str(batch.get("answer", [""])[0])[:300]

    if not losses:
        raise RuntimeError("No training steps were executed")

    summary.update({
        "ok": True,
        "train_sec": round(time.time() - start, 3),
        "total_sec": round(time.time() - start_total, 3),
        "losses": losses,
        "final_loss": losses[-1],
        "grad_norms": grad_norms,
        "steps_executed": len(losses),
        "trainable_param_tensors": len(selected),
        "trainable_param_delta_norm": _parameter_delta_norm(before_params, selected),
    })

    if args.save_adapter:
        if args.training_mode != "lora":
            raise ValueError("--save-adapter is only supported with --training-mode lora")
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        summary["adapter_dir"] = str(adapter_dir)

    summary_path = output_dir / "real_florence_vp_training_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="COCO128/YOLO dataset root")
    parser.add_argument("--classes-file", default=None, help="Optional class names file; defaults to generated COCO80 names")
    parser.add_argument("--image-ext", default=".jpg", help="Image extension used by YOLO labels")
    parser.add_argument("--vp-box-format", default="json", choices=["json", "loc_tokens"])
    parser.add_argument("--vp-marker-style", default="special", choices=["special", "plain"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", default=None, help="Explicit local Florence-2 checkpoint path")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_real_training")
    parser.add_argument("--max-train-samples", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--training-data-order",
        default="as-is",
        choices=["as-is", "grounding-first"],
        help="Deterministic task ordering for short runs; grounding-first consumes query grounding rows first.",
    )
    parser.add_argument(
        "--skip-od-training-data",
        action="store_true",
        help="Do not include the generated OD_VP train split in MultiTaskDataset; useful for grounding-only overfit probes.",
    )
    parser.add_argument("--shuffle-train-data", action="store_true", help="Shuffle the materialized multi-task dataset.")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="DataLoader seed used when --shuffle-train-data is set.")
    parser.add_argument("--training-mode", default="slice", choices=["slice", "lora"])
    parser.add_argument("--trainable-match", default=DEFAULT_TRAINABLE_MATCH)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-task-type",
        default="SEQ_2_SEQ_LM",
        choices=["CAUSAL_LM", "SEQ_2_SEQ_LM"],
        help="PEFT task type. Florence-2 VP training is encoder-decoder, so SEQ_2_SEQ_LM is the default.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="LoRA target module suffixes for Florence-2.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        nargs="*",
        default=None,
        help=(
            "Extra modules to train and save with LoRA. Use an empty value to disable; "
            "for VP format learning try: lm_head model.shared"
        ),
    )
    parser.add_argument(
        "--train-vp-head",
        action="store_true",
        help="Shortcut for --lora-modules-to-save lm_head model.shared.",
    )
    parser.add_argument("--save-adapter", action="store_true", help="Save PEFT adapter under output-dir/adapter")
    parser.add_argument("--include-count", action="store_true", help="Also materialize COUNT_VP YOLO samples")
    parser.add_argument("--include-grounding", action="store_true", help="Also materialize query grounding VP samples")
    parser.add_argument(
        "--grounding-train-path",
        default=None,
        help="Optional query-grounding JSONL to use directly as the effective grounding training data.",
    )
    parser.add_argument(
        "--grounding-val-path",
        default=None,
        help="Optional query-grounding JSONL to expose as val_grounding_path in the manifest.",
    )
    parser.add_argument(
        "--grounding-task-type",
        default="PHRASE_GROUNDING_VP",
        choices=["PHRASE_GROUNDING_VP", "OPEN_VOCABULARY_DETECTION"],
        help="Task prompt used for derived query grounding VP samples.",
    )
    parser.add_argument(
        "--grounding-curriculum",
        default="none",
        choices=["none", "multi-instance"],
        help="Optional over-sampling curriculum for train grounding samples.",
    )
    parser.add_argument(
        "--grounding-selection",
        default="od-split",
        choices=["od-split", "shortest-query", "multi-instance"],
        help=(
            "How to choose derived grounding train/val rows. "
            "od-split preserves the OD split; multi-instance ranks query rows by box count."
        ),
    )
    parser.add_argument("--grounding-single-weight", type=int, default=1)
    parser.add_argument("--grounding-medium-weight", type=int, default=2)
    parser.add_argument("--grounding-dense-weight", type=int, default=3)
    parser.add_argument(
        "--grounding-min-query-boxes",
        type=int,
        default=None,
        help="Optional minimum query box count for effective grounding train rows.",
    )
    parser.add_argument(
        "--grounding-max-query-boxes",
        type=int,
        default=None,
        help="Optional maximum query box count for effective grounding train rows.",
    )
    parser.add_argument(
        "--grounding-count-hint-template",
        default=None,
        help=(
            "Optional template used to rewrite query-grounding text_input with an explicit "
            "target count, e.g. '{label} | count={query_box_count}'. "
            "query_label remains clean for evaluation allow-lists."
        ),
    )
    parser.add_argument(
        "--grounding-count-hint-splits",
        default="both",
        choices=["both", "train", "val"],
        help=(
            "Which query-grounding splits should receive --grounding-count-hint-template. "
            "Use 'train' to keep validation/inference prompts as clean labels."
        ),
    )
    args = parser.parse_args(argv)
    if args.train_vp_head and args.lora_modules_to_save is None:
        args.lora_modules_to_save = list(DEFAULT_LORA_MODULES_TO_SAVE)
    if args.grounding_train_path or args.grounding_val_path or args.grounding_selection != "od-split":
        args.include_grounding = True
    if args.grounding_count_hint_template:
        args.include_grounding = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_smoke(parse_args(argv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
