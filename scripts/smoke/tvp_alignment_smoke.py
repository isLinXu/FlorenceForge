#!/usr/bin/env python3
"""Offline smoke test for the FlorenceForge x TVP alignment stack.

Generates tiny synthetic maze/path/spatial datasets, converts them to TVP
chain-of-thought JSONL, validates dataset loading, config bridging, and
benchmark evaluation. Does not load a Florence model unless --with-training
is explicitly requested.
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

from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.data.tvp_converter import TVPDataConverter
from florence_forge.data.tvp_synthetic import write_all_tvp_synthetic
from florence_forge.evaluation.tvp_benchmark import evaluate_tvp_predictions
from florence_forge.training.tvp_training import build_training_config_from_tvp, load_tvp_yaml


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline TVP alignment smoke test.")
    parser.add_argument(
        "--work-dir",
        help="Directory for generated smoke artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated artifacts when --work-dir is not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Synthetic samples per TVP task family.",
    )
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="Also validate TVP SFT config materialization (still no model load).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def _convert_all_to_vp(raw_root: Path, vp_root: Path) -> dict[str, Path]:
    vp_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    maze_raw = raw_root / "maze"
    maze_vp = vp_root / "maze_vp.jsonl"
    TVPDataConverter.maze_jsonl_to_vp(
        input_path=str(maze_raw / "maze_data.jsonl"),
        output_path=str(maze_vp),
        image_dir=str(maze_raw),
    )
    outputs["maze"] = maze_vp

    path_raw = raw_root / "path"
    path_vp = vp_root / "path_vp.jsonl"
    TVPDataConverter.path_jsonl_to_vp(
        input_path=str(path_raw / "path_data.jsonl"),
        output_path=str(path_vp),
        image_dir=str(path_raw),
    )
    outputs["path"] = path_vp

    spatial_raw = raw_root / "spatial"
    spatial_vp = vp_root / "spatial_vp.jsonl"
    TVPDataConverter.spatial_reasoning_jsonl_to_vp(
        input_path=str(spatial_raw / "spatial_data.jsonl"),
        output_path=str(spatial_vp),
        image_dir=str(spatial_raw),
    )
    outputs["spatial"] = spatial_vp

    return outputs


def _load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dataset_counts(vp_outputs: dict[str, Path]) -> dict[str, int]:
    counts = {}
    for name, path in vp_outputs.items():
        task_type = {
            "maze": "MAZE_VP",
            "path": "PATH_VP",
            "spatial": "SPATIAL_VP",
        }[name]
        dataset = MultiTaskDataset(
            data_configs=[{"task_type": task_type, "data_path": str(path)}],
            image_base_path="",
        )
        counts[name] = len(dataset)
    return counts


def _benchmark_smoke(vp_outputs: dict[str, Path]) -> dict[str, float]:
    all_records = []
    all_predictions = []
    for name, path in vp_outputs.items():
        for record in _load_records(path):
            all_records.append(record)
            all_predictions.append(record.get("suffix", ""))
    results = evaluate_tvp_predictions(all_records, all_predictions)
    return {
        "sample_count": results["overall_metrics"]["sample_count"],
        "composite_mean": results["overall_metrics"]["composite_mean"],
    }


def _write_smoke_sft_yaml(root: Path, vp_outputs: dict[str, Path]) -> Path:
    yaml_path = root / "sft_smoke.yaml"
    yaml_path.write_text(
        f"""
model_name_or_path: "microsoft/Florence-2-base"
output_dir: "{(root / 'outputs/tvp/sft_smoke').as_posix()}"
mixed_training:
  enabled: false
epochs: 1
batch_size: 1
num_workers: 0
datasets:
  - path: "{vp_outputs['maze'].as_posix()}"
    image_root: "."
    task_type: "maze"
    weight: 1.0
  - path: "{vp_outputs['path'].as_posix()}"
    image_root: "."
    task_type: "path"
    weight: 1.0
  - path: "{vp_outputs['spatial'].as_posix()}"
    image_root: "."
    task_type: "spatial"
    weight: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def run_smoke(root: Path, *, num_samples: int = 3, with_training: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "raw"
    vp_root = root / "vp"

    write_all_tvp_synthetic(raw_root, num_samples=num_samples, seed=17)
    vp_outputs = _convert_all_to_vp(raw_root, vp_root)
    dataset_counts = _dataset_counts(vp_outputs)
    benchmark = _benchmark_smoke(vp_outputs)

    checks = {
        "raw_tasks": len(dataset_counts),
        "dataset_rows": sum(dataset_counts.values()),
        "benchmark_sample_count": benchmark["sample_count"],
        "benchmark_composite_mean": benchmark["composite_mean"],
    }

    if dataset_counts["maze"] != num_samples:
        raise AssertionError(f"Expected {num_samples} maze rows, got {dataset_counts['maze']}")
    if dataset_counts["path"] != num_samples:
        raise AssertionError(f"Expected {num_samples} path rows, got {dataset_counts['path']}")
    if dataset_counts["spatial"] != num_samples:
        raise AssertionError(f"Expected {num_samples} spatial rows, got {dataset_counts['spatial']}")
    if benchmark["sample_count"] != num_samples * 3:
        raise AssertionError("Benchmark sample count mismatch")
    if benchmark["composite_mean"] <= 0:
        raise AssertionError("Expected positive composite score for gold predictions")

    training_info = None
    if with_training:
        smoke_yaml = _write_smoke_sft_yaml(root, vp_outputs)
        cfg = load_tvp_yaml(smoke_yaml)
        training_config = build_training_config_from_tvp(cfg)
        training_info = {
            "yaml": str(smoke_yaml),
            "tasks": training_config.tasks,
            "num_epochs": training_config.num_epochs,
            "dataset_entries": len(training_config._tvp_data_configs),
        }
        if len(training_config.tasks) != 3:
            raise AssertionError("Expected three TVP tasks in smoke training config")

    report = {
        "ok": True,
        "artifacts": {
            "raw_root": str(raw_root),
            "vp_root": str(vp_root),
            **{f"{name}_vp": str(path) for name, path in vp_outputs.items()},
        },
        "checks": checks,
        "training": training_info,
    }
    (root / "tvp_alignment_smoke_report.json").write_text(
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
        report = run_smoke(
            Path(args.work_dir),
            num_samples=args.num_samples,
            with_training=args.with_training,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.keep_artifacts:
        root = REPO_ROOT / ".codex_reports" / "tvp_alignment_smoke"
        report = run_smoke(
            root,
            num_samples=args.num_samples,
            with_training=args.with_training,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    with TemporaryDirectory(prefix="florence_tvp_smoke_") as tmpdir:
        report = run_smoke(
            Path(tmpdir),
            num_samples=args.num_samples,
            with_training=args.with_training,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
