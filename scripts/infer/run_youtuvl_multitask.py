#!/usr/bin/env python3
"""Run Youtu-VL-4B-Instruct inference on one image or an image directory.

This script follows the official Youtu-VL chat path:
AutoProcessor + AutoModelForCausalLM + apply_chat_template + img_input.
"""

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
from transformers import AutoModelForCausalLM, AutoProcessor


LOGGER = logging.getLogger("youtuvl_multitask_infer")

DEFAULT_MODEL_ID = "tencent/Youtu-VL-4B-Instruct"
MODELSCOPE_MODEL_ID = "Tencent-YouTu-Research/Youtu-VL-4B-Instruct"

TASK_PROMPTS = {
    "caption": "Describe the image in detail.",
    "detect": "Detect all objects in the provided image.",
    "ocr": "Read all text present in the image.",
    "vqa": "{question}",
    "grounding": "Please provide the bounding box coordinate of the region this sentence describes: {question}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="HF model id or local Youtu-VL model directory.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".codex_reports/youtuvl_multitask_infer"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="eager", help="Use eager/sdpa/flash_attention_2.")
    parser.add_argument("--tasks", default="caption,vqa", help="Comma-separated task names.")
    parser.add_argument("--question", default="What is in the image?", help="Question for vqa/grounding.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.001)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype | str:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return "auto"
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
        raise ValueError(f"Unknown Youtu-VL task(s): {', '.join(unknown)}")
    return tasks


def build_prompt(task: str, question: str) -> str:
    return TASK_PROMPTS[task].format(question=question)


def normalize_model_ref(model: str) -> str:
    """Accept the ModelScope URL/id used by the user and map it to HF id.

    Transformers can load a local ModelScope snapshot directory directly. For
    remote ids it expects the Hugging Face repo id, whose files mirror the
    ModelScope model card for this checkpoint.
    """
    model_ref = model.rstrip("/")
    if model_ref.startswith("http"):
        if "modelscope.cn/models/" in model_ref and MODELSCOPE_MODEL_ID in model_ref:
            return DEFAULT_MODEL_ID
        return model_ref
    if model_ref == MODELSCOPE_MODEL_ID:
        return DEFAULT_MODEL_ID
    return model_ref


def load_model_and_processor(args: argparse.Namespace, device: str, dtype: torch.dtype | str):
    model_ref = normalize_model_ref(args.model)
    LOGGER.info("Loading Youtu-VL model: %s", model_ref)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs).eval()
    if device != "cpu" and model_kwargs["device_map"] is None:
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_ref,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    return model, processor


def parse_youtuvl_boxes(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"<ref>(?P<label>.*?)</ref>.*?<box>.*?"
        r"<x_min>(?P<x1>\d+)</x_min>.*?"
        r"<y_min>(?P<y1>\d+)</y_min>.*?"
        r"<x_max>(?P<x2>\d+)</x_max>.*?"
        r"<y_max>(?P<y2>\d+)</y_max>.*?</box>",
        re.DOTALL,
    )
    boxes: List[Dict[str, Any]] = []
    for match in pattern.finditer(text):
        boxes.append(
            {
                "label": match.group("label").strip(),
                "bbox": [
                    int(match.group("x1")),
                    int(match.group("y1")),
                    int(match.group("x2")),
                    int(match.group("y2")),
                ],
            }
        )
    return boxes


def infer_one(
    model,
    processor,
    image_path: Path,
    task: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> Dict[str, Any]:
    prompt = build_prompt(task, question)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "img_input": str(image_path),
    }

    started = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = outputs[0].strip() if outputs else ""
    parsed: Dict[str, Any] = {}
    if task in {"detect", "grounding"}:
        parsed["boxes"] = parse_youtuvl_boxes(text)

    with Image.open(image_path) as image:
        image_size: Tuple[int, int] = image.size

    return {
        "task": task,
        "prompt": prompt,
        "text": text,
        "parsed": parsed,
        "duration_sec": round(time.time() - started, 4),
        "image_size": list(image_size),
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

    model, processor = load_model_and_processor(args, device, dtype)
    records: List[Dict[str, Any]] = []
    results_path = args.output_dir / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            for task in tasks:
                try:
                    result = infer_one(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        task=task,
                        question=args.question,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=args.do_sample,
                    )
                    record = {"status": "ok", "image_path": str(image_path), **result}
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
        "attn_implementation": args.attn_implementation,
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
