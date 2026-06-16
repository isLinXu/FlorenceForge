#!/usr/bin/env python3
"""Offline smoke test for the FlorenceForge visual primitive MVP.

The script creates a tiny synthetic COCO detection dataset, converts it to
OD_VP and COUNT_VP JSONL, loads the result through MultiTaskDataset, parses the
VP suffix, and computes VP-aware detection metrics. It does not load a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from florence_forge.data import MultiTaskDataset, VisualPrimitiveConverter
from florence_forge.evaluation.metrics import get_metric_calculator
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline VP MVP smoke test.")
    parser.add_argument(
        "--work-dir",
        help="Directory for generated smoke artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated artifacts when --work-dir is not provided.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def _write_synthetic_coco(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20]},
        ],
    }
    coco_path = root / "annotations.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path, image_dir


def run_smoke(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    coco_path, image_dir = _write_synthetic_coco(root)
    od_output = root / "od_vp.jsonl"
    count_output = root / "count_vp.jsonl"

    VisualPrimitiveConverter.coco_to_vp_od(
        coco_json_path=str(coco_path),
        output_path=str(od_output),
        image_dir=str(image_dir),
    )
    VisualPrimitiveConverter.coco_to_vp_counting(
        coco_json_path=str(coco_path),
        output_path=str(count_output),
        image_dir=str(image_dir),
    )

    od_dataset = MultiTaskDataset(
        data_configs=[{"task_type": "OD_VP", "data_path": str(od_output)}],
        image_base_path="",
    )
    count_dataset = MultiTaskDataset(
        data_configs=[{"task_type": "COUNT_VP", "data_path": str(count_output)}],
        image_base_path="",
    )

    parser = VisualPrimitiveParser()
    od_suffix = od_dataset[0]["answer"]
    count_suffix = count_dataset[0]["answer"]
    od_detections = parser.parse_detections(od_suffix)
    count_detections = parser.parse_detections(count_suffix)

    metric = get_metric_calculator("OD_VP")
    metric.add_batch([od_suffix], [od_suffix])
    metrics = metric.compute()

    checks = {
        "od_rows": len(od_dataset),
        "count_rows": len(count_dataset),
        "od_detections": len(od_detections),
        "count_detections": len(count_detections),
        "vp_format_valid_ratio": metrics.get("vp_format_valid_ratio", 0.0),
        "vp_coordinate_valid_ratio": metrics.get("vp_coordinate_valid_ratio", 0.0),
        "vp_box_count_exact_match": metrics.get("vp_box_count_exact_match", 0.0),
    }

    if checks["od_rows"] != 1:
        raise AssertionError(f"Expected one OD_VP row, got {checks['od_rows']}")
    if checks["count_rows"] != 1:
        raise AssertionError(f"Expected one COUNT_VP row, got {checks['count_rows']}")
    if checks["od_detections"] != 2:
        raise AssertionError(f"Expected two OD detections, got {checks['od_detections']}")
    if checks["count_detections"] != 2:
        raise AssertionError(f"Expected two count detections, got {checks['count_detections']}")
    if checks["vp_format_valid_ratio"] != 1.0:
        raise AssertionError("VP format validity should be 1.0")
    if checks["vp_coordinate_valid_ratio"] != 1.0:
        raise AssertionError("VP coordinate validity should be 1.0")

    report = {
        "ok": True,
        "artifacts": {
            "coco": str(coco_path),
            "od_vp": str(od_output),
            "count_vp": str(count_output),
        },
        "checks": checks,
    }
    (root / "vp_smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.work_dir:
        report = run_smoke(Path(args.work_dir))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.keep_artifacts:
        root = REPO_ROOT / ".codex_reports" / "visual_primitive_mvp_smoke"
        report = run_smoke(root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    with TemporaryDirectory(prefix="florence_vp_smoke_") as tmpdir:
        report = run_smoke(Path(tmpdir))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
