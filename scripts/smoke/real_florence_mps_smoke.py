#!/usr/bin/env python3
"""Real Florence-2 smoke test for FlorenceForge on local devices.

This script validates the real backend path without starting a long fine-tune:

1. Load a Florence-2 checkpoint through FlorenceForge.
2. Optionally run image-level generation.
3. Encode a tiny JSONL dataset through MultiTaskDataset.
4. Run a real forward pass and loss.
5. Optionally freeze the model and train a tiny parameter slice for one step.
6. Optionally run FlorenceForge's TrainingLoop on one real batch.

It defaults to MPS when available, then CUDA, then CPU. By default it resolves a
local Hugging Face cache snapshot for microsoft/Florence-2-base to avoid network
downloads. Pass --model-path to use an explicit local path.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from florence_forge.utils.diagnostics import DEFAULT_MODEL_ID, find_local_hf_snapshot


DEFAULT_TRAINABLE_MATCH = "language_model.model.decoder.layers.5.fc2"


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
            raise FileNotFoundError("Explicit --model-path does not exist: %s" % path)
        return str(path)

    snapshot = find_local_hf_snapshot(model_id)
    if snapshot is not None:
        return str(snapshot)

    raise FileNotFoundError(
        "No local snapshot found for %s. Run `huggingface-cli download %s` "
        "or pass --model-path." % (model_id, model_id)
    )


def _make_tiny_dataset(root: Path) -> Dict[str, Path]:
    from PIL import Image

    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(image_dir / "red.png")

    data_path = root / "caption.jsonl"
    data_path.write_text(
        json.dumps({
            "image": "red.png",
            "prefix": "<CAPTION>",
            "suffix": "a red image",
        }) + "\n",
        encoding="utf-8",
    )
    return {"image_dir": image_dir, "data_path": data_path}


def _float_loss(output: Any) -> torch.Tensor:
    if hasattr(output, "loss"):
        return output.loss
    return output["loss"]


def _select_trainable_parameters(
    model: torch.nn.Module,
    name_fragment: str,
) -> Iterable[torch.nn.Parameter]:
    selected = []
    for param in model.parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if name_fragment in name:
            param.requires_grad_(True)
            selected.append(param)

    if not selected:
        raise RuntimeError("No parameters matched --trainable-match=%r" % name_fragment)
    return selected


def _clone_trainable_parameters(parameters: Iterable["torch.nn.Parameter"]) -> list["torch.Tensor"]:
    return [param.detach().clone() for param in parameters]


def _parameter_delta_norm(
    before: Iterable["torch.Tensor"],
    after: Iterable["torch.nn.Parameter"],
) -> float:
    import torch

    total = 0.0
    for before_param, after_param in zip(before, after):
        total += float((after_param.detach() - before_param).float().norm().cpu())
    return total


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from PIL import Image

    from florence_forge.core.config import DataConfig, ModelConfig
    from florence_forge.core.model import Florence2MultiTaskModel
    from florence_forge.data.collate import Florence2Collator
    from florence_forge.data.dataset import MultiTaskDataset

    device = _choose_device(args.device)
    model_name = _resolve_model_path(args.model_id, args.model_path)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="florenceforge_real_smoke_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "device": device,
        "output_dir": str(output_dir),
        "mode": args.mode,
    }

    config = ModelConfig(
        model_name=model_name,
        backend_name="florence-2",
        device=device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        use_lora=False,
        attn_implementation="eager",
    )

    start = time.time()
    model = Florence2MultiTaskModel(config).load()
    summary["model_load_sec"] = round(time.time() - start, 3)
    summary["backend_device"] = str(model._backend.device)
    summary["backend_dtype"] = str(model._backend.dtype)

    if args.mode in ("generate", "all"):
        image = Image.new("RGB", (args.image_size, args.image_size), color="red")
        start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                images=image,
                task_prompt="<CAPTION>",
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
        summary["generate_sec"] = round(time.time() - start, 3)
        summary["generated_text"] = generated

    if args.mode in ("forward", "backward", "train-loop", "all"):
        data_paths = _make_tiny_dataset(output_dir)
        dataset = MultiTaskDataset(
            data_configs=[{"task_type": "CAPTION", "data_path": str(data_paths["data_path"])}],
            image_base_path=str(data_paths["image_dir"]),
            config=DataConfig(use_cache=False, num_workers=0, batch_size=1),
            processor=model.processor,
            backend=model._backend,
        )

        start = time.time()
        sample = dataset[0]
        batch = Florence2Collator(pad_token_id=dataset._get_pad_token_id())([sample])
        summary["encode_sec"] = round(time.time() - start, 3)
        summary["batch_devices"] = {
            key: str(value.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        summary["input_shape"] = list(batch["input_ids"].shape)
        summary["pixel_shape"] = list(batch["pixel_values"].shape)

        model.train()
        start = time.time()
        output = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = _float_loss(output)
        summary["forward_sec"] = round(time.time() - start, 3)
        summary["loss"] = float(loss.detach().cpu())

        if args.mode in ("backward", "all"):
            selected = list(_select_trainable_parameters(model, args.trainable_match))
            optimizer = torch.optim.AdamW(selected, lr=args.learning_rate)
            start = time.time()
            loss.backward()
            grad_norm = 0.0
            for param in selected:
                if param.grad is not None:
                    grad_norm += float(param.grad.detach().float().norm().cpu())
            optimizer.step()
            optimizer.zero_grad()
            summary["backward_step_sec"] = round(time.time() - start, 3)
            summary["trainable_param_tensors"] = len(selected)
            summary["grad_norm_sum"] = grad_norm

        if args.mode in ("train-loop", "all"):
            from torch.utils.data import DataLoader

            from florence_forge.core.config import TrainingConfig
            from florence_forge.training.training_loop import TrainingLoop

            selected = list(_select_trainable_parameters(model, args.trainable_match))
            before_params = _clone_trainable_parameters(selected)
            optimizer = torch.optim.AdamW(selected, lr=args.learning_rate)
            train_config = TrainingConfig(
                device=device,
                output_dir=str(output_dir),
                batch_size=1,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=1,
                max_grad_norm=args.max_grad_norm,
                gradient_checkpointing=False,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=Florence2Collator(pad_token_id=dataset._get_pad_token_id()),
            )

            loop = TrainingLoop(model=model, config=train_config, accelerator=None)
            start = time.time()
            metrics = loop.train_epoch(
                train_dataloader=dataloader,
                optimizer=optimizer,
                lr_scheduler=None,
                epoch=0,
            )
            summary["train_loop_sec"] = round(time.time() - start, 3)
            summary["train_loop_metrics"] = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            summary["train_loop_global_step"] = loop.global_step
            summary["train_loop_param_delta_norm"] = _parameter_delta_norm(before_params, selected)
            summary["train_loop_trainable_param_tensors"] = len(selected)

    summary_path = output_dir / "real_florence_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--mode", default="all", choices=["generate", "forward", "backward", "train-loop", "all"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--trainable-match", default=DEFAULT_TRAINABLE_MATCH)
    return parser.parse_args()


def main() -> None:
    summary = run_smoke(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
