#!/usr/bin/env python3
"""Real TVP SFT smoke on Apple MPS.

Prepares tiny synthetic maze/path/spatial TVP JSONL, then runs one TVP SFT
training step through the MultiTaskTrainer bridge on MPS.

Defaults to the local Florence checkpoint already used in this repo's prior
MPS smoke runs. Override with --model-path if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = (
    "/Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model"
)
DEFAULT_WORK_DIR = REPO_ROOT / ".codex_reports" / "tvp_alignment_smoke"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TVP SFT smoke on Apple MPS.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local Florence checkpoint directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory containing TVP smoke artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "tvp" / "sft_mps_smoke",
        help="Training output directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Stop after this many optimizer steps.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Synthetic samples per TVP task family.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate synthetic TVP data even if artifacts already exist.",
    )
    parser.add_argument("--device", default="mps", choices=["mps", "auto", "cpu"])
    return parser


def _ensure_smoke_data(work_dir: Path, *, num_samples: int, regenerate: bool) -> Path:
    vp_maze = work_dir / "vp" / "maze_vp.jsonl"
    if regenerate or not vp_maze.exists():
        import importlib.util

        script_path = REPO_ROOT / "scripts" / "smoke" / "tvp_alignment_smoke.py"
        spec = importlib.util.spec_from_file_location("tvp_alignment_smoke", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.run_smoke(work_dir, num_samples=num_samples, with_training=True)
    smoke_yaml = work_dir / "sft_smoke.yaml"
    if not smoke_yaml.exists():
        raise FileNotFoundError(f"Missing smoke YAML: {smoke_yaml}")
    return smoke_yaml


def run_mps_smoke(args: argparse.Namespace) -> dict:
    import torch

    device = args.device
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    smoke_yaml = _ensure_smoke_data(
        args.work_dir,
        num_samples=args.num_samples,
        regenerate=args.regenerate_data,
    )

    from florence_forge.training.tvp_training import run_tvp_sft_with_multitask_trainer

    started = time.time()
    summary = run_tvp_sft_with_multitask_trainer(
        smoke_yaml,
        checkpoint_dir=str(args.output_dir),
        overrides={
            "model_name_or_path": str(model_path),
            "device": device,
            "max_steps": args.max_steps,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "torch_dtype": "float32",
            "use_lora": True,
        },
    )
    elapsed = time.time() - started

    report = {
        "ok": True,
        "device": device,
        "model_path": str(model_path),
        "smoke_yaml": str(smoke_yaml),
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "elapsed_seconds": round(elapsed, 2),
        "training_summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "tvp_mps_training_smoke_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    report = run_mps_smoke(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
