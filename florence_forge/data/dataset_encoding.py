"""多任务数据集样本编码（processor / backend 路径）。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.tasks import FLORENCE2_TASKS
from .dataset_types import TaskSample

logger = logging.getLogger(__name__)


def get_task_prompt(task_type: str, backend: Any) -> str:
    if backend is not None:
        try:
            return backend.get_task_prompt(task_type)
        except Exception:
            pass
    task_config = FLORENCE2_TASKS.get(task_type)
    return task_config.prompt if task_config else ""


def build_prompt_and_answer(
    sample: TaskSample,
    *,
    backend: Any,
) -> Tuple[str, str]:
    task_prompt = get_task_prompt(sample.task_type, backend)
    prefix = sample.prefix or ""
    if task_prompt and prefix:
        prompt = prefix if prefix.startswith(task_prompt) else f"{task_prompt}{prefix}"
    else:
        prompt = prefix or task_prompt

    for extra_key in ("text_input", "region"):
        extra_value = sample.metadata.get(extra_key)
        if extra_value is None:
            continue
        if not isinstance(extra_value, str):
            extra_value = json.dumps(extra_value, ensure_ascii=False)
        if extra_value and extra_value not in prompt:
            prompt = f"{prompt}{extra_value}"

    return prompt, sample.suffix


def default_prepare_labels(
    encoded_prompt: Dict[str, torch.Tensor],
    encoded_full: Dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_ids = encoded_prompt.get("input_ids")
    full_ids = encoded_full["input_ids"]

    if prompt_ids is None:
        logger.warning("prompt 编码未返回 input_ids，回退为仅监督完整序列")
        return full_ids.clone()

    if prompt_ids.dim() == 2:
        prompt_ids = prompt_ids.squeeze(0)
    if full_ids.dim() == 2:
        full_ids = full_ids.squeeze(0)

    prompt_length = len(prompt_ids)
    labels = torch.full_like(full_ids, -100)
    if len(full_ids) > prompt_length:
        labels[prompt_length:] = full_ids[prompt_length:]
    return labels


def unencoded_sample_dict(sample: TaskSample) -> Dict[str, Any]:
    return {
        "image_path": sample.image_path,
        "prompt": sample.prefix,
        "answer": sample.suffix,
        "task_type": sample.task_type,
        "weight": sample.weight,
        "metadata": sample.metadata,
        "_needs_encoding": True,
    }


def raw_image_result(
    image: Any,
    prompt: str,
    answer: str,
    sample: TaskSample,
) -> Dict[str, Any]:
    return {
        "image": image,
        "prompt": prompt,
        "answer": answer,
        "task_type": sample.task_type,
        "weight": sample.weight,
        "metadata": sample.metadata,
    }


def _extract_reference_ids_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """从带 ``-100`` mask 的 labels 中提取可解码的参考答案 token。"""
    if labels.dim() > 1:
        labels = labels.squeeze(0)
    reference_ids = labels[labels != -100]
    if reference_ids.numel() == 0:
        return labels.new_empty((0,), dtype=labels.dtype)
    return reference_ids.clone()


def _encode_via_backend(
    *,
    backend: Any,
    image: Any,
    sample: TaskSample,
    prompt: str,
    answer: str,
) -> Optional[Dict[str, Any]]:
    if not hasattr(backend, "encode_with_task"):
        return None
    try:
        backend_encoded = backend.encode_with_task(
            images=[image],
            task_name=sample.task_type,
            text_input=answer,
            return_tensors="pt",
        )
        full_processed = {
            k: (
                v.squeeze(0)
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1
                else v
            )
            for k, v in backend_encoded.items()
        }

        if hasattr(backend, "prepare_labels"):
            try:
                labels = backend.prepare_labels({}, backend_encoded)
                if isinstance(labels, torch.Tensor) and labels.dim() > 0 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)
            except Exception as exc:
                logger.debug("backend.prepare_labels 失败，回退到默认: %s", exc)
                labels = full_processed["input_ids"].clone()
        else:
            labels = full_processed["input_ids"].clone()

        result = {
            "input_ids": full_processed["input_ids"],
            "pixel_values": full_processed["pixel_values"],
            "labels": labels,
            "reference_ids": _extract_reference_ids_from_labels(labels),
            "prompt": prompt,
            "answer": answer,
            "task_type": sample.task_type,
            "weight": sample.weight,
            "metadata": sample.metadata,
        }
        prompt_lengths = full_processed.get("prompt_lengths")
        if isinstance(prompt_lengths, torch.Tensor):
            if prompt_lengths.dim() > 0:
                prompt_length = int(prompt_lengths.reshape(-1)[0].item())
            else:
                prompt_length = int(prompt_lengths.item())
            result["prompt_input_ids"] = full_processed["input_ids"][:prompt_length].clone()
            if "attention_mask" in full_processed:
                result["prompt_attention_mask"] = full_processed["attention_mask"][:prompt_length].clone()
        if "attention_mask" in full_processed:
            result["attention_mask"] = full_processed["attention_mask"]
        for extra_key in ("token_type_ids", "position_ids", "mm_token_type_ids"):
            if extra_key in full_processed:
                result[extra_key] = full_processed[extra_key]
        return result
    except AssertionError:
        return None
    except Exception as exc:
        logger.debug("backend.encode_with_task 失败，回退到 processor 拼接: %s", exc)
        return None


def _encode_via_processor(
    *,
    processor: Any,
    backend: Any,
    image: Any,
    sample: TaskSample,
    prompt: str,
    answer: str,
    prompt_text: str,
) -> Dict[str, Any]:
    full_text = prompt_text + answer
    full_processed = processor(text=full_text, images=image, return_tensors="pt")
    full_processed = {
        k: v.squeeze(0) if hasattr(v, "squeeze") else v
        for k, v in full_processed.items()
    }

    tokenizer = getattr(processor, "tokenizer", None) or getattr(
        processor, "text_processor", None
    )
    prompt_id_len = None
    if tokenizer is not None:
        try:
            answer_token_len = len(
                tokenizer(answer, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ][0]
            )
            full_ids_1d = full_processed["input_ids"]
            if full_ids_1d.dim() == 2:
                full_ids_1d = full_ids_1d.squeeze(0)
            prompt_id_len = max(0, full_ids_1d.shape[0] - answer_token_len)
        except Exception:
            prompt_id_len = None

    full_ids_1d = full_processed["input_ids"]
    if full_ids_1d.dim() == 2:
        full_ids_1d = full_ids_1d.squeeze(0)

    prompt_input_ids = (
        full_ids_1d[:prompt_id_len]
        if prompt_id_len is not None
        else full_ids_1d.clone()
    )
    prompt_processed = {
        "input_ids": prompt_input_ids,
        "pixel_values": full_processed.get("pixel_values"),
    }

    if backend is not None and hasattr(backend, "prepare_labels"):
        try:
            labels = backend.prepare_labels(prompt_processed, full_processed)
        except Exception:
            labels = default_prepare_labels(prompt_processed, full_processed)
    else:
        labels = default_prepare_labels(prompt_processed, full_processed)

    result = {
        "input_ids": full_processed["input_ids"],
        "attention_mask": full_processed["attention_mask"],
        "pixel_values": full_processed["pixel_values"],
        "labels": labels,
        "reference_ids": _extract_reference_ids_from_labels(labels),
        "prompt_input_ids": prompt_input_ids.clone(),
        "prompt_attention_mask": prompt_processed.get("attention_mask"),
        "prompt": prompt,
        "answer": answer,
        "task_type": sample.task_type,
        "weight": sample.weight,
        "metadata": sample.metadata,
    }
    for extra_key in ("token_type_ids", "position_ids", "mm_token_type_ids"):
        if extra_key in full_processed:
            result[extra_key] = full_processed[extra_key]
    return result


def encode_training_sample(
    *,
    sample: TaskSample,
    image: Any,
    processor: Any,
    backend: Any,
) -> Dict[str, Any]:
    """将单张图像 + 样本编码为训练用字典。

    If the sample suffix contains agentic meta-cognitive tokens, a
    ``loss_weights`` tensor is added to the output for phase-aware
    loss weighting during training.
    """
    prompt, answer = build_prompt_and_answer(sample, backend=backend)
    prompt_text = prompt or sample.prefix or get_task_prompt(sample.task_type, backend)

    backend_result = _encode_via_backend(
        backend=backend,
        image=image,
        sample=sample,
        prompt=prompt,
        answer=answer,
    )
    if backend_result is not None:
        _maybe_add_phase_weights(backend_result, answer, processor, sample)
        return backend_result

    result = _encode_via_processor(
        processor=processor,
        backend=backend,
        image=image,
        sample=sample,
        prompt=prompt,
        answer=answer,
        prompt_text=prompt_text,
    )
    _maybe_add_phase_weights(result, answer, processor, sample)
    return result


def _maybe_add_phase_weights(
    result: Dict[str, Any],
    answer: str,
    processor: Any,
    sample: TaskSample,
) -> None:
    """Add ``loss_weights`` to result if the answer contains agentic tokens.

    This is a no-op for non-agentic samples, so it has zero overhead
    for standard Florence-2 training.
    """
    # Quick check: does the answer contain any agentic token?
    if "<PLAN>" not in answer and "<ACT>" not in answer and "<DECIDE>" not in answer:
        return

    labels = result.get("labels")
    if labels is None or not hasattr(labels, "shape"):
        return

    tokenizer = getattr(processor, "tokenizer", None) or getattr(
        processor, "text_processor", None
    )
    if tokenizer is None:
        return

    try:
        from .phase_aware_loss import build_phase_weight_tensor

        labels_1d = labels.squeeze(0) if labels.dim() > 1 else labels
        weights = build_phase_weight_tensor(
            labels=labels_1d,
            answer_text=answer,
            tokenizer=tokenizer,
        )
        if labels.dim() > 1:
            weights = weights.unsqueeze(0)
        result["loss_weights"] = weights
    except Exception as exc:
        logger.debug("Phase-aware weight computation skipped: %s", exc)
