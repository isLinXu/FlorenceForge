#!/usr/bin/env python3
"""Run PaliGemma multi-task inference on one image or an image directory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


LOGGER = logging.getLogger("paligemma_multitask_infer")

TASK_PROMPTS = {
    "caption": "caption en",
    "detect": "detect",
    "ocr": "ocr",
    "vqa": "answer en {question}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local PaliGemma model directory.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory.")
    parser.add_argument("--output-dir", type=Path, default=Path(".codex_reports/paligemma_multitask_infer"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--tasks", default="caption,detect,ocr,vqa", help="Comma-separated task names.")
    parser.add_argument("--question", default="What is in the image?", help="Question used by the vqa task.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
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


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def iter_images(input_path: Path, max_images: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(path for path in input_path.iterdir() if path.suffix.lower() in suffixes)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No images found under {input_path}")
    return images


def parse_tasks(tasks_arg: str) -> List[str]:
    tasks = [task.strip().lower() for task in tasks_arg.split(",") if task.strip()]
    unknown = [task for task in tasks if task not in TASK_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown PaliGemma task(s): {', '.join(unknown)}")
    return tasks


def build_prompt(task: str, question: str) -> str:
    prompt = TASK_PROMPTS[task].format(question=question)
    return prompt if prompt.startswith("<image>") else f"<image>{prompt}"


def load_model(model_path: Path, device: str, dtype: torch.dtype):
    LOGGER.info("Loading PaliGemma model: %s", model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    if device != "cpu":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


def parse_paligemma_detections(text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    width, height = image_size
    pattern = re.compile(
        r"<loc(?P<y1>\d{4})><loc(?P<x1>\d{4})><loc(?P<y2>\d{4})><loc(?P<x2>\d{4})>\s*(?P<label>[^;<]+)"
    )
    detections: List[Dict[str, Any]] = []
    for match in pattern.finditer(text):
        y1 = int(match.group("y1")) / 1023 * height
        x1 = int(match.group("x1")) / 1023 * width
        y2 = int(match.group("y2")) / 1023 * height
        x2 = int(match.group("x2")) / 1023 * width
        detections.append(
            {
                "label": match.group("label").strip(),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return detections


def infer_one(
    model,
    processor,
    image: Image.Image,
    task: str,
    question: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Dict[str, Any]:
    prompt = build_prompt(task, question)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    started = time.time()
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    answer_ids = generated[0][input_len:]
    text = processor.decode(answer_ids, skip_special_tokens=True).strip()
    duration = time.time() - started
    parsed: Dict[str, Any] = {}
    if task == "detect":
        parsed["detections"] = parse_paligemma_detections(text, image.size)
    return {
        "task": task,
        "prompt": prompt,
        "text": text,
        "parsed": parsed,
        "duration_sec": duration,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    tasks = parse_tasks(args.tasks)
    images = iter_images(args.input, args.max_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, device, dtype)
    records: List[Dict[str, Any]] = []
    results_path = args.output_dir / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            for task in tasks:
                try:
                    result = infer_one(
                        model=model,
                        processor=processor,
                        image=image,
                        task=task,
                        question=args.question,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                    )
                    record = {
                        "status": "ok",
                        "image_path": str(image_path),
                        "image_size": list(image.size),
                        **result,
                    }
                except Exception as exc:
                    LOGGER.exception("Failed %s on %s", task, image_path)
                    record = {
                        "status": "error",
                        "image_path": str(image_path),
                        "task": task,
                        "error": str(exc),
                    }
                handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                LOGGER.info("%s | %s | %s", image_path.name, task, record.get("text", record.get("error", ""))[:160])

    ok_count = sum(1 for record in records if record.get("status") == "ok")
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "model": str(args.model),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "device": device,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "tasks": tasks,
        "total_records": len(records),
        "ok_records": ok_count,
        "error_records": len(records) - ok_count,
        "results_path": str(results_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
