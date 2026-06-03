"""单次推理运行时：Florence2 生成、PIL/tensor 前向与可视化分发。"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import inference_parsing as parsing
from . import inference_visualization as visualization

logger = logging.getLogger(__name__)


def is_florence2_model(model: nn.Module) -> bool:
    if not hasattr(model, "__class__"):
        return False
    return (
        hasattr(model, "generate")
        and hasattr(model, "processor")
        and (
            "Florence2MultiTaskModel" in str(model.__class__)
            or "florence" in str(model.__class__).lower()
        )
    )


def format_generate_output(generated_text: Any) -> str:
    if generated_text is None:
        return ""
    if isinstance(generated_text, torch.Tensor):
        return str(generated_text)
    if isinstance(generated_text, (list, tuple)):
        return str(generated_text[0]) if generated_text else ""
    return str(generated_text)


def generate_florence2_text(
    model: nn.Module,
    image: Any,
    *,
    task_prompt: Optional[str],
    text_input: Optional[str],
    device: torch.device,
    use_amp: bool,
) -> str:
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type=device.type):
                generated = model.generate(
                    images=image,
                    task_prompt=task_prompt,
                    text_input=text_input,
                )
        else:
            generated = model.generate(
                images=image,
                task_prompt=task_prompt,
                text_input=text_input,
            )
    return format_generate_output(generated)


def pil_to_batched_tensor(image: Any, device: torch.device) -> torch.Tensor:
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor.to(device).unsqueeze(0)


def forward_tensor(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
) -> Any:
    inputs = inputs.to(device)
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type=device.type):
                return model(inputs)
        return model(inputs)


def ensure_tensor(inputs: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs)
    inputs = inputs.to(device)
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    return inputs


def visualize_florence2_output(
    image: Any,
    outputs: str,
    task_prompt: Optional[str],
    *,
    save_path: Optional[str] = None,
) -> None:
    if not outputs or task_prompt is None:
        return
    size: Tuple[int, int] = image.size

    try:
        if "<OD>" in task_prompt or "detection" in task_prompt.lower():
            detections = parsing.parse_florence2_output(outputs, size)
            if detections:
                visualization.visualize_detections(image, detections, save_path)
                logger.info("检测到 %d 个目标并已可视化", len(detections))
            else:
                logger.warning("未检测到任何目标")
        elif "segmentation" in task_prompt.lower() or "REGION_TO_SEGMENTATION" in task_prompt:
            seg = parsing.parse_segmentation_output(outputs, size)
            if seg:
                visualization.visualize_segmentation(image.copy(), seg, save_path)
                logger.info("分割结果已可视化")
            else:
                logger.warning("未解析到分割数据")
        elif "<REGION_PROPOSAL>" in task_prompt:
            bboxes = parsing.parse_bboxes(outputs, size)
            if bboxes:
                visualization.visualize_bboxes(image.copy(), bboxes, save_path)
                logger.info("区域提议结果已可视化")
            else:
                logger.warning("未解析到区域提议")
        elif "OCR_WITH_REGION" in task_prompt:
            ocr_results = parsing.parse_ocr_with_region(outputs, size)
            if ocr_results:
                visualization.visualize_ocr_with_region(image.copy(), ocr_results, save_path)
                logger.info("OCR区域结果已可视化")
            else:
                logger.warning("未解析到OCR区域结果")
        elif "<REGION_TO_CATEGORY>" in task_prompt:
            detections = parsing.parse_florence2_output(outputs, size)
            if detections:
                visualization.visualize_detections(image, detections, save_path)
                logger.info("检测到 %d 个目标并已可视化", len(detections))
            else:
                logger.warning("未检测到任何目标")
        else:
            visualization.visualize_caption(image.copy(), outputs, save_path)
    except Exception as exc:
        logger.error("可视化失败: %s", exc)


def predict_pil_image(
    model: nn.Module,
    image: Any,
    *,
    device: torch.device,
    use_amp: bool,
    task_prompt: Optional[str],
    text_input: Optional[str],
    visualize: bool,
    save_path: Optional[str],
) -> Any:
    if is_florence2_model(model):
        try:
            outputs = generate_florence2_text(
                model,
                image,
                task_prompt=task_prompt,
                text_input=text_input,
                device=device,
                use_amp=use_amp,
            )
            if visualize and outputs:
                visualize_florence2_output(
                    image, outputs, task_prompt, save_path=save_path
                )
            return outputs
        except Exception as exc:
            logger.error("Florence2模型推理失败: %s", exc)
            if "embedding" in str(exc).lower() or "indices" in str(exc).lower():
                logger.warning("检测到embedding相关错误，返回空结果")
                return ""
            logger.info("尝试回退到普通tensor处理方式")
            try:
                tensor_in = pil_to_batched_tensor(image, device)
                return forward_tensor(model, tensor_in, device=device, use_amp=use_amp)
            except Exception as fallback_exc:
                logger.error("回退处理也失败: %s", fallback_exc)
                return ""

    tensor_in = pil_to_batched_tensor(image, device)
    return forward_tensor(model, tensor_in, device=device, use_amp=use_amp)


def predict_batch_non_florence(
    model: nn.Module,
    inputs_list: list,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    preprocessor: Optional[Callable],
    postprocessor: Optional[Callable],
    update_stats: Callable[[float, int], None],
) -> list:
    """普通模型的批量 tensor 推理。"""
    results: list = []
    for start in range(0, len(inputs_list), batch_size):
        batch_inputs = inputs_list[start : start + batch_size]
        if preprocessor is not None:
            batch_inputs = [preprocessor(inp) for inp in batch_inputs]

        processed: list = []
        for inp in batch_inputs:
            if hasattr(inp, "mode") and hasattr(inp, "size"):
                try:
                    from PIL import Image

                    if isinstance(inp, Image.Image):
                        inp = pil_to_batched_tensor(inp, device).squeeze(0)
                except ImportError:
                    logger.warning("PIL未安装，无法处理PIL Image")
            if not isinstance(inp, torch.Tensor):
                inp = torch.tensor(inp)
            processed.append(inp)

        batch_tensor = torch.stack(processed).to(device)
        step_start = time.time()
        batch_outputs = forward_tensor(
            model, batch_tensor, device=device, use_amp=use_amp
        )
        update_stats(time.time() - step_start, len(batch_inputs))

        if postprocessor is not None:
            results.extend(
                postprocessor(output.unsqueeze(0)) for output in batch_outputs
            )
        else:
            results.extend(output for output in batch_outputs)
    return results


def run_predict_core(
    model: nn.Module,
    inputs: Any,
    *,
    device: torch.device,
    use_amp: bool,
    preprocessor: Optional[Callable],
    postprocessor: Optional[Callable],
    task_prompt: Optional[str],
    text_input: Optional[str],
    return_raw: bool,
    visualize: bool,
    save_path: Optional[str],
) -> Any:
    if preprocessor:
        inputs = preprocessor(inputs, task_prompt=task_prompt, text_input=text_input)

    if hasattr(inputs, "mode") and hasattr(inputs, "size"):
        try:
            from PIL import Image

            if isinstance(inputs, Image.Image):
                outputs = predict_pil_image(
                    model,
                    inputs,
                    device=device,
                    use_amp=use_amp,
                    task_prompt=task_prompt,
                    text_input=text_input,
                    visualize=visualize,
                    save_path=save_path,
                )
                if not return_raw and postprocessor is not None:
                    outputs = postprocessor(outputs)
                return outputs
        except ImportError:
            logger.warning("PIL未安装，无法处理PIL Image")

    tensor_in = ensure_tensor(inputs, device)
    outputs = forward_tensor(model, tensor_in, device=device, use_amp=use_amp)
    if not return_raw and postprocessor is not None:
        outputs = postprocessor(outputs)
    return outputs
