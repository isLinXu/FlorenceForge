"""CLI 评估命令处理。

处理 ``eval`` 子命令：模型评估与 TVP benchmark 评估。
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from florence_forge.data.dataset import MultiTaskDataset

logger = logging.getLogger(__name__)


def _build_eval_dataset_from_jsonl(data_path: str, model) -> "MultiTaskDataset":
    """从 Florence-2 JSONL 评估文件构建 ``MultiTaskDataset``。

    JSONL 每行形如 ``{"image": ..., "prefix": "<OD>", "suffix": ...}``，
    任务类型由 ``prefix`` 推断（匹配 ``FLORENCE2_TASKS`` 中的 prompt）。
    """
    from florence_forge.core.tasks import FLORENCE2_TASKS
    from florence_forge.data.dataset import MultiTaskDataset

    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"评估数据文件不存在: {source}")

    prompt_to_task = sorted(
        ((cfg.prompt, name) for name, cfg in FLORENCE2_TASKS.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def _infer_task_type(prefix: str) -> Optional[str]:
        prefix = (prefix or "").strip()
        for prompt, name in prompt_to_task:
            if prompt and prefix.startswith(prompt):
                return name
        return None

    grouped_lines: "defaultdict[str, list]" = defaultdict(list)
    skipped = 0
    with open(source, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            task_type = _infer_task_type(record.get("prefix", ""))
            if task_type is None:
                skipped += 1
                continue
            grouped_lines[task_type].append(line)

    if not grouped_lines:
        raise ValueError(
            f"无法从 {source} 推断出任何受支持的任务类型，"
            f"请确认每行包含可识别的 prefix（如 <OD>、<CAPTION>）。"
        )
    if skipped:
        logger.warning(f"评估数据中有 {skipped} 行无法解析或无法识别任务类型，已跳过")

    temp_dir = Path(tempfile.mkdtemp(prefix="florence_eval_"))
    data_configs = []
    for task_type, lines in grouped_lines.items():
        task_file = temp_dir / f"{task_type}.jsonl"
        with open(task_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        data_configs.append({"task_type": task_type, "data_path": str(task_file)})
        logger.info(f"   任务 {task_type}: {len(lines)} 个样本")

    # 评估阶段优先复用模型 backend。
    # 对 Florence 原生 OD 等任务，backend.encode_with_task() 会正确处理
    # “task token + answer token” 的拼接；仅传 processor 会退回到旧的
    # prompt+answer 直接拼接路径，并触发 processing_florence2 的断言。
    dataset = MultiTaskDataset(
        data_configs,
        processor=model.processor,
        backend=getattr(model, "_backend", None),
    )
    dataset._eval_temp_dir = str(temp_dir)  # type: ignore[attr-defined]
    return dataset


def _run_tvp_eval_task(args) -> bool:
    """Run TVP benchmark evaluation over a JSONL dataset."""
    try:
        from florence_forge.core.model import Florence2MultiTaskModel
        from florence_forge.core.config import ModelConfig
        from florence_forge.evaluation.tvp_benchmark import run_tvp_benchmark
    except ImportError as exc:
        logger.error(f"❌ 无法导入 TVP 评估模块: {exc}")
        return False

    model_path = args.model
    data_path = args.data
    logger.info("📊 开始 TVP benchmark 评估")
    logger.info(f"   模型路径: {model_path}")
    logger.info(f"   评估数据: {data_path}")
    logger.info(f"   设备: {args.device}")

    try:
        model_config = ModelConfig(
            model_name=model_path,
            device=args.device,
            use_lora=False,
            enable_visual_primitives=True,
        )
        model = Florence2MultiTaskModel(model_config)
        model.load()

        results = run_tvp_benchmark(
            model,
            data_path,
            max_samples=getattr(args, "max_samples", None),
        )

        logger.info("📈 TVP benchmark 结果:")
        overall = results.get("overall_metrics", {})
        logger.info(
            f"   composite_mean: {overall.get('composite_mean', 0.0):.4f} "
            f"(n={overall.get('sample_count', 0)})"
        )
        for task_type, task_result in results.get("task_metrics", {}).items():
            logger.info(
                f"   {task_type}: composite_mean="
                f"{task_result.get('composite_mean', 0.0):.4f} "
                f"(n={task_result.get('sample_count', 0)})"
            )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, ensure_ascii=False, default=str)
            logger.info(f"   评估结果已保存: {output_path}")

        return True
    except Exception as exc:
        logger.error(f"❌ TVP 评估任务失败: {exc}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def run_eval_task(args) -> bool:
    """运行模型评估"""
    benchmark = getattr(args, "benchmark", "default")
    if benchmark == "tvp":
        return _run_tvp_eval_task(args)

    try:
        from florence_forge.evaluation.evaluator import MultiTaskEvaluator
    except ImportError as e:
        logger.error(f"❌ 无法导入评估模块: {e}")
        return False

    model_path = args.model
    data_path = args.data

    logger.info("📊 开始模型评估")
    logger.info(f"   模型路径: {model_path}")
    logger.info(f"   评估数据: {data_path}")
    logger.info(f"   设备: {args.device}")

    dataset = None
    try:
        from florence_forge.core.model import Florence2MultiTaskModel
        from florence_forge.core.config import ModelConfig

        model_config = ModelConfig(
            model_name=model_path,
            device=args.device,
            use_lora=False,
        )
        model = Florence2MultiTaskModel(model_config)
        model.load()

        dataset = _build_eval_dataset_from_jsonl(data_path, model)
        evaluator = MultiTaskEvaluator(model, device=args.device)
        results = evaluator.evaluate_dataset(dataset)

        logger.info("📈 评估结果:")
        overall_metrics = results.get("overall_metrics", {})
        for metric_name, metric_value in overall_metrics.items():
            logger.info(
                f"   {metric_name}: {metric_value:.4f}"
                if isinstance(metric_value, float)
                else f"   {metric_name}: {metric_value}"
            )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"   评估结果已保存: {output_path}")

        return True
    except Exception as e:
        logger.error(f"❌ 评估任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False
    finally:
        temp_dir = getattr(dataset, "_eval_temp_dir", None) if dataset is not None else None
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
