"""CLI 推理命令处理。

处理 ``infer`` 子命令：单张图像和批量目录推理，支持结构化 VP 解码。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from ._helpers import (
    _is_supported_image_file,
    _iter_image_files,
    _normalize_inference_stats,
)

logger = logging.getLogger(__name__)


def _apply_structured_vp_decode(result, args) -> dict:
    """Apply structured VP decoding to an inference result if --structured-vp is enabled."""
    if not getattr(args, "structured_vp", False):
        return {"result": result}

    from florence_forge.evaluation.structured_vp_decoder import StructuredVisualPrimitiveDecoder

    decoder = StructuredVisualPrimitiveDecoder(
        box_format=getattr(args, "vp_box_format", "loc_tokens"),
        marker_style=getattr(args, "vp_marker_style", "special"),
        max_boxes_per_label=getattr(args, "vp_max_boxes_per_label", None),
        nms_iou_threshold=getattr(args, "vp_nms_iou_threshold", None),
    )

    text = result if isinstance(result, str) else str(result)
    decoded = decoder.decode(text)

    return {
        "raw_result": result,
        "vp_text": decoded.vp_text,
        "vp_labels": decoded.labels,
        "vp_boxes": decoded.boxes,
        "vp_counts": decoded.counts,
    }


def run_inference_task(args) -> bool:
    """运行推理任务"""
    try:
        from florence_forge.deployment.inference import InferenceEngine
    except ImportError:
        logger.error("❌ 无法导入推理引擎，请检查安装")
        return False

    model_path_str = args.model
    is_hf_hub_id = "/" in model_path_str and not os.path.exists(model_path_str)

    if not is_hf_hub_id:
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.error(f"❌ 模型文件或目录不存在: {model_path}")
            return False
    else:
        logger.info(f"ℹ️  将从Hugging Face Hub加载模型: {model_path_str}")
        model_path = model_path_str

    logger.info("🚀 开始推理任务")
    logger.info(f"   模型路径: {model_path}")
    logger.info(f"   输入路径: {args.input}")
    logger.info(f"   输出目录: {args.output}")
    logger.info(f"   设备: {args.device}")

    try:
        inference_engine = InferenceEngine(
            model=str(model_path),
            device=args.device,
            batch_size=args.batch_size,
            use_amp=args.use_amp,
        )
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(args.input)
        results: List[Dict[str, Any]] = []

        if input_path.is_file():
            results = _run_single_image_inference(
                input_path, inference_engine, output_dir, args
            )
        elif input_path.is_dir():
            results = _run_batch_inference(
                input_path, inference_engine, output_dir, args
            )
        else:
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return False

        _save_inference_summary(results, inference_engine, output_dir, model_path, input_path)
        return True

    except Exception as e:
        logger.error(f"❌ 推理任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def _run_single_image_inference(
    input_path: Path,
    inference_engine: Any,
    output_dir: Path,
    args,
) -> List[Dict[str, Any]]:
    """处理单张图像推理。"""
    if not _is_supported_image_file(input_path):
        logger.error(f"❌ 不支持的文件格式: {input_path.suffix}")
        raise ValueError(f"Unsupported image format: {input_path.suffix}")

    logger.info("📸 处理单张图像...")
    with Image.open(input_path) as img:
        image = img.convert("RGB")

    task_prompt = getattr(args, "task_prompt", "<OD>")
    visualize = getattr(args, "visualize", False)
    save_path = None
    if visualize and getattr(args, "save_visualizations", False):
        save_path = output_dir / f"{input_path.stem}_visualization.png"

    text_input = _resolve_text_input(args, task_prompt)
    result = inference_engine.predict(
        image,
        task_prompt=task_prompt,
        text_input=text_input,
        visualize=visualize,
        save_path=str(save_path) if save_path else None,
    )

    output_data = {"image_path": str(input_path)}
    output_data.update(_apply_structured_vp_decode(result, args))
    result_file = output_dir / f"{input_path.stem}_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✅ 推理完成: {result_file}")
    return [{"image_path": str(input_path), "result_file": str(result_file), "result": result}]


def _run_batch_inference(
    input_path: Path,
    inference_engine: Any,
    output_dir: Path,
    args,
) -> List[Dict[str, Any]]:
    """处理批量目录推理。"""
    image_files = list(_iter_image_files(input_path))
    if not image_files:
        logger.error(f"❌ 在目录中未找到图像文件: {input_path}")
        raise ValueError(f"No image files found in: {input_path}")

    task_prompt = getattr(args, "task_prompt", "<OD>")
    text_input = _resolve_text_input(args, task_prompt)
    logger.info(f"📸 处理 {len(image_files)} 张图像...")

    results: List[Dict[str, Any]] = []
    for i, image_file in enumerate(image_files, 1):
        try:
            image_path = Path(image_file)
            logger.info(f"处理 {i}/{len(image_files)}: {image_path.name}")
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            visualize = getattr(args, "visualize", False)
            save_path = None
            if visualize and getattr(args, "save_visualizations", False):
                save_path = output_dir / f"{image_path.stem}_visualization.png"

            result = inference_engine.predict(
                image,
                task_prompt=task_prompt,
                text_input=text_input,
                visualize=visualize,
                save_path=str(save_path) if save_path else None,
            )

            output_data = {"image_path": str(image_path)}
            output_data.update(_apply_structured_vp_decode(result, args))
            result_file = output_dir / f"{image_path.stem}_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

            results.append({
                "image_path": str(image_path),
                "result_file": str(result_file),
                "result": result,
            })
        except Exception as e:
            logger.error(f"❌ 处理图像失败 {image_path.name}: {e}")
            continue

    logger.info(f"✅ 批量推理完成，处理了 {len(results)} 张图像")
    return results


def _resolve_text_input(args, task_prompt: str) -> str:
    """根据任务提示推断是否需要文本输入。"""
    if task_prompt == "<OPEN_VOCABULARY_DETECTION>" and not getattr(args, "text_input", None):
        logger.error(f"❌ 任务 '{task_prompt}' 需要 --text-input 参数.")
        raise ValueError("--text-input required for OPEN_VOCABULARY_DETECTION")
    return getattr(args, "text_input", None)


def _save_inference_summary(
    results: List[Dict[str, Any]],
    inference_engine: Any,
    output_dir: Path,
    model_path: Any,
    input_path: Path,
) -> None:
    """保存推理汇总统计。"""
    stats = _normalize_inference_stats(inference_engine.get_stats())
    summary_file = output_dir / "inference_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": str(model_path),
            "input_path": str(input_path),
            "total_images": len(results),
            "results": results,
            "stats": stats,
        }, f, indent=2, ensure_ascii=False)

    logger.info("📊 推理统计:")
    logger.info(f"   总推理次数: {stats['total_inferences']}")
    logger.info(f"   总耗时: {stats['total_time']:.2f}s")
    logger.info(f"   平均推理时间: {stats['avg_inference_time']:.3f}s")
    logger.info(f"   吞吐量: {stats['throughput']:.2f} images/s")
    logger.info(f"   汇总文件: {summary_file}")
