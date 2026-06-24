"""CLI agentic 命令处理。

处理 ``agentic`` 子命令：使用 ``AgenticOrchestrator`` 对单张或批量图像
执行多步视觉推理。Orchestrator 将自然语言目标分解为子任务序列，
依次调用 Florence-2 原生工具（<OD>、<OCR>、<CAPTION> …）并汇总结果。

设计要点
  * ``InferenceEngineAdapter`` 将现有 ``InferenceEngine`` 适配为
    ``ToolBackend`` 协议（``predict_task``），无需修改引擎代码。
  * 支持单图和目录批量模式，与 ``infer`` 命令的 UX 一致。
  * 输出 JSON 结果文件 + 可选 transcript 文件，方便后续 SFT 轨迹采集。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ._helpers import _is_supported_image_file, _iter_image_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InferenceEngine → ToolBackend adapter
# ---------------------------------------------------------------------------

class InferenceEngineAdapter:
    """Adapter wrapping ``InferenceEngine`` to satisfy the ``ToolBackend`` protocol.

    ``AgenticOrchestrator`` calls ``backend.predict_task(images, task_name, text_input)``
    — this adapter translates that into ``InferenceEngine.predict(image, task_prompt, text_input)``.
    """

    def __init__(self, engine: Any):
        self._engine = engine

    def predict_task(
        self,
        images: Any,
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        from florence_forge.core.tasks import get_task_config_typed

        task_config = get_task_config_typed(task_name)
        result = self._engine.predict(
            images,
            task_prompt=task_config.prompt,
            text_input=text_input,
        )
        # predict() may return a tensor or str; coerce to str
        if isinstance(result, str):
            return result
        return str(result)


# ---------------------------------------------------------------------------
# run_agentic_task — the CLI handler
# ---------------------------------------------------------------------------

def run_agentic_task(args) -> bool:
    """运行 Agentic 多步推理任务。"""
    try:
        from florence_forge.deployment.inference import InferenceEngine
    except ImportError:
        logger.error("❌ 无法导入推理引擎，请检查安装")
        return False

    from florence_forge.agentic import (
        AgenticOrchestrator,
        OrchestratorConfig,
    )

    # --- 解析模型路径 ---
    model_path_str = args.model
    is_hf_hub_id = "/" in model_path_str and not os.path.exists(model_path_str)

    if not is_hf_hub_id:
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.error(f"❌ 模型文件或目录不存在: {model_path}")
            return False
    else:
        logger.info(f"ℹ️  将从 Hugging Face Hub 加载模型: {model_path_str}")
        model_path = model_path_str

    # --- 构建 OrchestratorConfig ---
    orch_config = OrchestratorConfig(
        max_steps=getattr(args, "max_steps", 12),
        max_retries=getattr(args, "max_retries", 1),
        summarize_every=getattr(args, "summarize_every", 3),
        emit_transcript=True,
    )

    logger.info("🤖 Agentic 多步推理")
    logger.info(f"   模型: {model_path}")
    logger.info(f"   输入: {args.input}")
    logger.info(f"   输出: {args.output}")
    logger.info(f"   目标: {args.goal}")
    logger.info(f"   设备: {args.device}")
    logger.info(f"   max_steps={orch_config.max_steps}, max_retries={orch_config.max_retries}")

    try:
        inference_engine = InferenceEngine(
            model=str(model_path),
            device=args.device,
            batch_size=1,
            use_amp=getattr(args, "use_amp", False),
        )
        backend = InferenceEngineAdapter(inference_engine)
        orchestrator = AgenticOrchestrator(backend, orch_config)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(args.input)

        if input_path.is_file():
            results = _run_single_image_agentic(
                input_path, orchestrator, output_dir, args,
            )
        elif input_path.is_dir():
            results = _run_batch_agentic(
                input_path, orchestrator, output_dir, args,
            )
        else:
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return False

        _save_agentic_summary(results, output_dir, model_path, input_path)
        return True

    except Exception as e:
        logger.error(f"❌ Agentic 任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_single_image_agentic(
    input_path: Path,
    orchestrator: Any,
    output_dir: Path,
    args,
) -> List[Dict[str, Any]]:
    """处理单张图像的 agentic 推理。"""
    if not _is_supported_image_file(input_path):
        logger.error(f"❌ 不支持的文件格式: {input_path.suffix}")
        raise ValueError(f"Unsupported image format: {input_path.suffix}")

    logger.info("📸 处理单张图像...")
    with Image.open(input_path) as img:
        image = img.convert("RGB")

    goal = args.goal
    result = orchestrator.run(image=image, goal=goal)

    result_dict = result.to_dict()
    result_dict["image_path"] = str(input_path)

    result_file = output_dir / f"{input_path.stem}_agentic.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✅ Agentic 推理完成: {result_file}")
    logger.info(f"   最终答案: {result.final_answer}")
    logger.info(f"   步骤数: {len(result.steps)}")
    logger.info(f"   成功: {result.success}")

    # 保存 transcript（可选）
    if getattr(args, "save_transcript", False) and result.transcript:
        transcript_file = output_dir / f"{input_path.stem}_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(result.transcript)
        logger.info(f"   Transcript: {transcript_file}")

    return [result_dict]


def _run_batch_agentic(
    input_path: Path,
    orchestrator: Any,
    output_dir: Path,
    args,
) -> List[Dict[str, Any]]:
    """处理批量目录的 agentic 推理。"""
    image_files = list(_iter_image_files(input_path))
    if not image_files:
        logger.error(f"❌ 在目录中未找到图像文件: {input_path}")
        raise ValueError(f"No image files found in: {input_path}")

    goal = args.goal
    logger.info(f"📸 处理 {len(image_files)} 张图像...")

    results: List[Dict[str, Any]] = []
    for i, image_file in enumerate(image_files, 1):
        try:
            image_path = Path(image_file)
            logger.info(f"处理 {i}/{len(image_files)}: {image_path.name}")
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            result = orchestrator.run(image=image, goal=goal)
            result_dict = result.to_dict()
            result_dict["image_path"] = str(image_path)

            result_file = output_dir / f"{image_path.stem}_agentic.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

            results.append(result_dict)
        except Exception as e:
            logger.error(f"❌ 处理图像失败 {image_path.name}: {e}")
            continue

    logger.info(f"✅ 批量 Agentic 推理完成，处理了 {len(results)} 张图像")
    return results


def _save_agentic_summary(
    results: List[Dict[str, Any]],
    output_dir: Path,
    model_path: Any,
    input_path: Path,
) -> None:
    """保存 agentic 汇总统计。"""
    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    total_steps = sum(len(r.get("steps", [])) for r in results)

    summary_file = output_dir / "agentic_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": str(model_path),
            "input_path": str(input_path),
            "total_images": total,
            "successful": success_count,
            "total_steps": total_steps,
            "avg_steps": total_steps / total if total else 0,
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info("📊 Agentic 统计:")
    logger.info(f"   总图像数: {total}")
    logger.info(f"   成功: {success_count}/{total}")
    logger.info(f"   总步骤数: {total_steps}")
    logger.info(f"   平均步骤数: {total_steps / total:.1f}" if total else "   平均步骤数: N/A")
    logger.info(f"   汇总文件: {summary_file}")
