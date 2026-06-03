"""CLI 命令处理函数。

抽离自 ``cli/main.py`` 的重型子命令实现：train / infer / serve / eval / convert。
所有外部重依赖均在函数内部惰性导入，保持包导入轻量，并兼容测试中的
``unittest.mock.patch`` 打桩（如 ``florence_forge.deployment.inference.InferenceEngine``）。

``main.py`` 会从本模块回导这些 handler，因此
``from florence_forge.cli.main import run_inference_task`` 等历史导入路径仍然有效。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ._helpers import (
    TASK_CONFIG_MAPPING,
    _is_supported_image_file,
    _iter_image_files,
    _normalize_inference_stats,
)

if TYPE_CHECKING:
    from florence_forge.core.config import TrainingConfig
    from florence_forge.data.dataset import MultiTaskDataset

logger = logging.getLogger(__name__)


def _select_trainer_class(version: str):
    """Select the training stack requested by CLI or programmatic callers."""
    normalized = (version or "v2").strip().lower()
    if normalized in {"v1", "legacy"}:
        raise ValueError(
            "训练器 v1 已在 v2.0.0 移除；请省略 --trainer-version 或显式使用 v2。"
        )
    if normalized in {"v2", "refactored", "modular", ""}:
        from florence_forge.training.trainer_refactored import MultiTaskTrainer

        return MultiTaskTrainer
    raise ValueError(f"不支持的训练器版本: {version}")


def run_inference_task(args) -> bool:
    """运行推理任务"""
    try:
        import json
        from PIL import Image

        # 导入推理引擎
        try:
            from florence_forge.deployment.inference import InferenceEngine
        except ImportError:
            logger.error("❌ 无法导入推理引擎，请检查安装")
            return False

        # 验证模型路径
        model_path_str = args.model
        is_hf_hub_id = '/' in model_path_str and not os.path.exists(model_path_str)

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

        # 创建推理引擎
        logger.info("🤖 初始化推理引擎...")
        inference_engine = InferenceEngine(
            model=str(model_path),
            device=args.device,
            batch_size=args.batch_size,
            use_amp=args.use_amp
        )

        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理输入
        input_path = Path(args.input)
        results = []

        if input_path.is_file():
            # 单个文件推理
            if _is_supported_image_file(input_path):
                logger.info("📸 处理单张图像...")
                with Image.open(input_path) as img:
                    image = img.convert('RGB')
                # 为Florence2模型添加默认任务提示
                task_prompt = getattr(args, 'task_prompt', '<OD>')  # 默认为目标检测

                # 设置可视化参数
                visualize = getattr(args, 'visualize', False)
                save_path = None
                if visualize and getattr(args, 'save_visualizations', False):
                    save_path = output_dir / f"{input_path.stem}_visualization.png"

                # 检查是否需要文本输入
                if task_prompt == '<OPEN_VOCABULARY_DETECTION>' and not args.text_input:
                    logger.error(f"❌ 任务 '{task_prompt}' 需要 --text-input 参数.")
                    return False
                text_input = args.text_input

                result = inference_engine.predict(
                    image,
                    task_prompt=task_prompt,
                    text_input=text_input,
                    visualize=visualize,
                    save_path=str(save_path) if save_path else None
                )

                # 保存结果
                result_file = output_dir / f"{input_path.stem}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "image_path": str(input_path),
                        "result": str(result) if not isinstance(result, (dict, list)) else result
                    }, f, indent=2, ensure_ascii=False)

                results.append({
                    "image_path": str(input_path),
                    "result_file": str(result_file),
                    "result": result
                })

                logger.info(f"✅ 推理完成: {result_file}")
            else:
                logger.error(f"❌ 不支持的文件格式: {input_path.suffix}")
                return False

        elif input_path.is_dir():
            # 批量推理：支持常见图像格式，大小写不敏感
            image_files = _iter_image_files(input_path)

            if not image_files:
                logger.error(f"❌ 在目录中未找到图像文件: {input_path}")
                return False

            # 为Florence2模型添加默认任务提示
            task_prompt = getattr(args, 'task_prompt', '<OD>')  # 默认为目标检测

            # 预先检查是否需要文本输入
            if task_prompt == '<OPEN_VOCABULARY_DETECTION>' and not args.text_input:
                logger.error(f"❌ 任务 '{task_prompt}' 需要 --text-input 参数.")
                return False
            text_input = args.text_input

            logger.info(f"📸 处理 {len(image_files)} 张图像...")

            # 批量处理
            for i, image_file in enumerate(image_files, 1):
                try:
                    image_path = Path(image_file)
                    logger.info(f"处理 {i}/{len(image_files)}: {image_path.name}")

                    with Image.open(image_path) as img:
                        image = img.convert('RGB')

                    # 设置可视化参数
                    visualize = getattr(args, 'visualize', False)
                    save_path = None
                    if visualize and getattr(args, 'save_visualizations', False):
                        save_path = output_dir / f"{image_path.stem}_visualization.png"

                    result = inference_engine.predict(
                        image,
                        task_prompt=task_prompt,
                        text_input=text_input,
                        visualize=visualize,
                        save_path=str(save_path) if save_path else None
                    )

                    # 保存结果
                    result_file = output_dir / f"{image_path.stem}_result.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "image_path": str(image_path),
                            "result": str(result) if not isinstance(result, (dict, list)) else result
                        }, f, indent=2, ensure_ascii=False)

                    results.append({
                        "image_path": str(image_path),
                        "result_file": str(result_file),
                        "result": result
                    })

                except Exception as e:
                    logger.error(f"❌ 处理图像失败 {image_path.name}: {e}")
                    continue

            logger.info(f"✅ 批量推理完成，处理了 {len(results)} 张图像")
        else:
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return False

        # 保存汇总结果
        stats = _normalize_inference_stats(inference_engine.get_stats())
        summary_file = output_dir / "inference_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": str(model_path),
                "input_path": str(input_path),
                "total_images": len(results),
                "results": results,
                "stats": stats
            }, f, indent=2, ensure_ascii=False)

        # 输出统计信息
        logger.info("📊 推理统计:")
        logger.info(f"   总推理次数: {stats['total_inferences']}")
        logger.info(f"   总耗时: {stats['total_time']:.2f}s")
        logger.info(f"   平均推理时间: {stats['avg_inference_time']:.3f}s")
        logger.info(f"   吞吐量: {stats['throughput']:.2f} images/s")
        logger.info(f"   汇总文件: {summary_file}")

        return True

    except Exception as e:
        logger.error(f"❌ 推理任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def run_serve_task(args) -> bool:
    """运行模型推理服务器"""
    try:
        from florence_forge.deployment.server import create_server
    except ImportError as e:
        logger.error(f"❌ 无法导入服务模块: {e}")
        logger.error("请确保已安装 FastAPI: pip install fastapi uvicorn")
        return False

    model_path = args.model
    host = args.host
    port = args.port

    logger.info("🚀 启动模型推理服务器")
    logger.info(f"   模型路径: {model_path}")
    logger.info(f"   监听地址: {host}:{port}")
    logger.info(f"   设备: {args.device}")
    logger.info(f"   后端: {getattr(args, 'backend', 'native')}")
    model_revision = getattr(args, 'model_revision', None)
    if model_revision:
        logger.info(f"   模型 revision: {model_revision}")

    server = create_server(
        model_path=model_path,
        host=host,
        port=port,
        device=args.device,
        backend=getattr(args, 'backend', 'native'),
        batch_size=getattr(args, 'batch_size', 1),
        use_amp=getattr(args, 'use_amp', False),
        model_revision=model_revision,
    )
    server.run(host=host, port=port)
    return True


def _build_eval_dataset_from_jsonl(data_path: str, model) -> "MultiTaskDataset":
    """从 Florence-2 JSONL 评估文件构建 ``MultiTaskDataset``。

    JSONL 每行形如 ``{"image": ..., "prefix": "<OD>", "suffix": ...}``，
    任务类型由 ``prefix`` 推断（匹配 ``FLORENCE2_TASKS`` 中的 prompt）。
    文件可能混合多种任务，这里按任务类型分组写入临时文件，再构建数据集。

    Args:
        data_path: JSONL 评估数据文件路径
        model: 已加载的模型，提供 processor 用于在线编码

    Returns:
        构建好的 MultiTaskDataset 实例
    """
    import json
    import tempfile
    from collections import defaultdict

    from florence_forge.core.tasks import FLORENCE2_TASKS
    from florence_forge.data.dataset import MultiTaskDataset

    source = Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"评估数据文件不存在: {source}")

    # prompt -> task_type 映射，按 prompt 长度降序匹配（避免短 prompt 误匹配）
    prompt_to_task = sorted(
        ((cfg.get("prompt", ""), name) for name, cfg in FLORENCE2_TASKS.items()),
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

    # 按任务类型分组写入临时文件
    temp_dir = Path(tempfile.mkdtemp(prefix="florence_eval_"))
    data_configs = []
    for task_type, lines in grouped_lines.items():
        task_file = temp_dir / f"{task_type}.jsonl"
        with open(task_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        data_configs.append({"task_type": task_type, "data_path": str(task_file)})
        logger.info(f"   任务 {task_type}: {len(lines)} 个样本")

    dataset = MultiTaskDataset(data_configs, processor=model.processor)
    # 记录临时目录，便于调用方在评估结束后清理
    dataset._eval_temp_dir = str(temp_dir)  # type: ignore[attr-defined]
    return dataset


def run_eval_task(args) -> bool:
    """运行模型评估"""
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

        # 加载模型
        model_config = ModelConfig(
            model_name=model_path,
            device=args.device,
            use_lora=False,
        )
        model = Florence2MultiTaskModel(model_config)
        model.load()

        # 从评估数据构建数据集
        dataset = _build_eval_dataset_from_jsonl(data_path, model)

        # 创建评估器并评估整个数据集
        evaluator = MultiTaskEvaluator(model, device=args.device)
        results = evaluator.evaluate_dataset(dataset)

        # 输出评估结果（总体指标）
        logger.info("📈 评估结果:")
        overall_metrics = results.get("overall_metrics", {})
        for metric_name, metric_value in overall_metrics.items():
            logger.info(
                f"   {metric_name}: {metric_value:.4f}"
                if isinstance(metric_value, float)
                else f"   {metric_name}: {metric_value}"
            )

        # 保存结果
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"   评估结果已保存: {output_path}")

        return True

    except Exception as e:
        logger.error(f"❌ 评估任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

    finally:
        # 清理评估期间生成的临时文件
        temp_dir = getattr(dataset, "_eval_temp_dir", None) if dataset is not None else None
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_data_conversion(args) -> bool:
    """运行数据转换任务"""
    try:
        # 导入数据转换器
        from florence_forge.data.converter import DataFormatConverter

        logger.info(f"开始数据转换: {args.convert_type}")

        if args.convert_type == 'yolo':
            DataFormatConverter.yolo_to_florence2_od(
                yolo_labels_dir=args.labels_dir,
                output_path=args.output,
                image_dir=args.images_dir,
                classes_file=args.classes_file,
                image_ext=args.image_ext,
                task_type=args.task_type
            )

        elif args.convert_type == 'coco':
            DataFormatConverter.coco_to_florence2_od(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir
            )

        elif args.convert_type == 'coco-caption':
            DataFormatConverter.coco_caption_to_florence2(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir
            )

        elif args.convert_type == 'csv':
            DataFormatConverter.csv_caption_to_florence2(
                csv_path=args.csv_file,
                output_path=args.output,
                image_column=args.image_column,
                caption_column=args.caption_column,
                task_type=args.task_type
            )

        elif args.convert_type == 'xml':
            DataFormatConverter.xml_to_florence2_od(
                xml_dir=args.xml_dir,
                output_path=args.output,
                image_dir=args.images_dir
            )

        elif args.convert_type == 'ocr':
            DataFormatConverter.txt_ocr_to_florence2(
                image_dir=args.images_dir,
                txt_dir=args.texts_dir,
                output_path=args.output,
                task_type=args.task_type
            )

        elif args.convert_type == 'ocr-txt':
            DataFormatConverter.txt_file_ocr_to_florence2(
                txt_file_path=args.txt_file,
                image_dir=args.images_dir,
                output_path=args.output,
                task_type=args.task_type
            )

        else:
            logger.error(f"❌ 不支持的转换类型: {args.convert_type}")
            return False

        logger.info(f"✅ 数据转换完成: {args.output}")
        return True

    except ImportError as e:
        logger.error(f"❌ 导入数据转换器失败: {e}")
        logger.error("请确保已正确安装florence_forge或数据转换器模块")
        return False

    except Exception as e:
        logger.error(f"❌ 数据转换失败: {e}")
        return False


def run_training_task(
    task: Optional[str] = None,
    config: Optional[str] = None,
    override: Optional[list] = None,
    **overrides
) -> bool:
    """运行训练任务"""
    # 处理 --override 参数
    if override:
        for key, value in override:
            # 尝试将值转换为适当的类型
            try:
                # 尝试转换为数字
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 如果转换失败，保持原始字符串
                pass
            overrides[key] = value
    try:
        from florence_forge.core.model import Florence2MultiTaskModel
        from florence_forge.training.config import load_config_from_file
        trainer_version = overrides.pop("trainer_version", "v2")
        MultiTaskTrainer = _select_trainer_class(trainer_version)

        # 确定配置文件路径
        if config:
            config_path = Path(config)
        elif task:
            if task not in TASK_CONFIG_MAPPING:
                logger.error(f"❌ 未知任务类型: {task}")
                logger.info(f"可用任务: {', '.join(TASK_CONFIG_MAPPING.keys())}")
                return False
            config_path = Path(TASK_CONFIG_MAPPING[task])
        else:
            logger.error("❌ 必须指定任务类型或配置文件")
            return False

        # 查找配置文件
        possible_paths = [
            config_path,
            Path.cwd() / config_path,
            Path(__file__).parent.parent / config_path
        ]

        actual_config_path = None
        for path in possible_paths:
            if path.exists():
                actual_config_path = path
                break

        if not actual_config_path:
            logger.error(f"❌ 找不到配置文件: {config_path}")
            return False

        # 将任务类型映射为正确的任务名称
        task_type_mapping = {
            'od': 'OD',
            'detection': 'OD',
            'caption': 'CAPTION',
            'detailed_caption': 'DETAILED_CAPTION',
            'more_detailed_caption': 'MORE_DETAILED_CAPTION',
            'open_vocabulary_detection': 'OPEN_VOCABULARY_DETECTION',
            'phrase_grounding': 'CAPTION_TO_PHRASE_GROUNDING',
            'dense_region_caption': 'DENSE_REGION_CAPTION',
            'region_proposal': 'REGION_PROPOSAL',
            'region_to_category': 'REGION_TO_CATEGORY',
            'region_to_description': 'REGION_TO_DESCRIPTION',
            'ocr': 'OCR',
            'ocr_with_region': 'OCR_WITH_REGION',
            'segmentation': 'REFERRING_EXPRESSION_SEGMENTATION',
            'seg': 'REFERRING_EXPRESSION_SEGMENTATION',
            'region_to_segmentation': 'REGION_TO_SEGMENTATION',
            'referring_expression_segmentation': 'REFERRING_EXPRESSION_SEGMENTATION'
        }

        # 添加任务类型到覆盖参数中
        if task and task in task_type_mapping:
            overrides['task_type'] = task_type_mapping[task]

        logger.info("🚀 开始训练任务")
        logger.info(f"   任务类型: {task or 'custom'}")
        logger.info(f"   配置文件: {actual_config_path}")
        logger.info(f"   训练器版本: {trainer_version}")

        if overrides:
            logger.info(f"   参数覆盖: {overrides}")

        # 加载训练配置
        logger.info("📋 加载训练配置...")
        training_config = load_config_from_file(str(actual_config_path))

        # 应用命令行参数覆盖
        if overrides:
            _apply_config_overrides(training_config, overrides)

        # 确保训练和验证数据路径被设置
        if 'train_data' in overrides and overrides['train_data'] is not None:
            training_config.train_data_path = overrides['train_data']
            logger.info(f"设置训练数据路径: {training_config.train_data_path}")

        if 'val_data' in overrides and overrides['val_data'] is not None:
            training_config.val_data_path = overrides['val_data']
            logger.info(f"设置验证数据路径: {training_config.val_data_path}")

        # 验证配置（Pydantic model_validate 已在 load_config_from_file 中完成校验）
        # 此处做补充的运行时检查
        logger.info("✅ 验证训练配置...")
        if training_config.train_data_path and not Path(training_config.train_data_path).exists():
            logger.warning(f"⚠️ 训练数据路径不存在: {training_config.train_data_path}")

        # 初始化模型（延迟加载模式，需显式调用 load()）
        logger.info("🤖 初始化模型...")
        model = Florence2MultiTaskModel(training_config.model_settings)
        logger.info("📥 加载模型和处理器...")
        model.load()

        # 准备数据集
        logger.info("📊 准备训练数据...")
        train_dataset, val_dataset = _prepare_datasets(training_config, model=model)

        # 创建训练器
        logger.info("🏋️ 创建训练器...")
        trainer = MultiTaskTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=training_config
        )

        # 断点续训：从检查点恢复
        resume_path = overrides.pop('resume', None)
        if resume_path:
            resume_checkpoint = Path(resume_path)
            if not resume_checkpoint.exists():
                logger.error(f"❌ 检查点路径不存在: {resume_checkpoint}")
                return False
            logger.info(f"📂 从检查点恢复训练: {resume_checkpoint}")
            trainer.load_checkpoint(resume_checkpoint)
            logger.info(f"   恢复到 Epoch {trainer.current_epoch}, Step {trainer.global_step}")

        # 开始训练
        logger.info("🚀 开始训练...")
        training_summary = trainer.train()

        # 输出训练结果
        logger.info("✅ 训练完成!")
        logger.info(f"   最终损失: {training_summary.get('final_loss', 'N/A')}")
        logger.info(f"   最佳指标: {training_summary.get('best_metric', 'N/A')}")
        logger.info(f"   训练轮数: {training_summary.get('epochs_completed', 'N/A')}")
        logger.info(f"   输出目录: {training_config.output_dir}")

        return True

    except Exception as e:
        logger.error(f"❌ 运行训练任务时出错: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def _set_nested_attr(obj, attr_str, value):
    """递归设置对象的嵌套属性"""
    alias_map = {
        "model_config": "model_settings",
        "data_config": "data_settings",
        "optimization_config": "optimization_settings",
        "task_scheduling_config": "task_scheduling_settings",
        "distributed_config": "distributed_settings",
    }
    attrs = attr_str.split('.')
    attrs[0] = alias_map.get(attrs[0], attrs[0])
    for attr in attrs[:-1]:
        attr = alias_map.get(attr, attr)
        obj = getattr(obj, attr)
    setattr(obj, alias_map.get(attrs[-1], attrs[-1]), value)


def _apply_config_overrides(config: 'TrainingConfig', overrides: Dict[str, Any]) -> None:
    """应用命令行参数覆盖到配置"""
    try:
        if 'epochs' in overrides and overrides['epochs'] is not None:
            config.num_epochs = overrides['epochs']
            logger.info(f"覆盖训练轮数: {config.num_epochs}")

        if 'batch_size' in overrides and overrides['batch_size'] is not None:
            config.data_settings.batch_size = overrides['batch_size']
            logger.info(f"覆盖批次大小: {config.data_settings.batch_size}")

        if 'lr' in overrides and overrides['lr'] is not None:
            config.optimization_settings.learning_rate = overrides['lr']
            logger.info(f"覆盖学习率: {config.optimization_settings.learning_rate}")

        if 'output_dir' in overrides and overrides['output_dir'] is not None:
            config.output_dir = overrides['output_dir']
            logger.info(f"覆盖输出目录: {config.output_dir}")

        if 'device' in overrides and overrides['device'] is not None:
            config.device = overrides['device']
            if hasattr(config, 'model_settings'):
                config.model_settings.device = overrides['device']
            logger.info(f"覆盖训练设备: {config.device}")

        if 'model' in overrides and overrides['model'] is not None:
            config.model_settings.model_name = overrides['model']
            logger.info(f"覆盖模型名称: {config.model_settings.model_name}")

        if 'train_data_path' in overrides and overrides['train_data_path'] is not None:
            config.train_data_path = overrides['train_data_path']
            logger.info(f"覆盖训练数据路径: {config.train_data_path}")

        if 'val_data_path' in overrides and overrides['val_data_path'] is not None:
            config.val_data_path = overrides['val_data_path']
            logger.info(f"覆盖验证数据路径: {config.val_data_path}")

        # 添加任务类型覆盖
        if 'task_type' in overrides and overrides['task_type'] is not None:
            config.tasks = [overrides['task_type']]
            config.task_weights = {overrides['task_type']: 1.0}
            logger.info(f"覆盖任务类型: {config.tasks}")

        # 处理所有其他以.分隔的覆盖
        for key, value in overrides.items():
            if '.' in key and value is not None:
                try:
                    _set_nested_attr(config, key, value)
                    logger.info(f"覆盖配置: {key} = {value}")
                except AttributeError:
                    logger.warning(f"无法设置配置属性: {key}")

    except Exception as e:
        logger.warning(f"应用配置覆盖时出错: {e}")


def _prepare_datasets(
    config: 'TrainingConfig', model=None
) -> Tuple['MultiTaskDataset', Optional['MultiTaskDataset']]:
    """准备训练和验证数据集"""
    try:
        from florence_forge.data.dataset import MultiTaskDataset

        try:
            from transformers import AutoProcessor
        except ImportError:
            AutoProcessor = None

        # 构建数据配置列表
        data_configs = []
        for task in config.tasks:
            # 确保任务类型格式正确
            task_type = task.upper() if task else "CAPTION"
            if config.train_data_path:
                data_configs.append({
                    "task_type": task_type,
                    "data_path": config.train_data_path,
                    "weight": config.task_weights.get(task, 1.0)
                })

        if not data_configs:
            # 如果没有配置数据路径，使用默认的示例数据
            data_configs = [{
                "task_type": "CAPTION",
                "data_path": "./data/sample_data.jsonl",
                "weight": 1.0
            }]
            logger.warning("未配置训练数据路径，使用默认示例数据")

        # 创建processor（优先复用模型已有的 processor，避免双重加载）
        processor = None
        if hasattr(model, 'processor') and model.processor is not None:
            processor = model.processor
            logger.info("复用模型已有的 processor")
        elif AutoProcessor is not None:
            try:
                processor = AutoProcessor.from_pretrained(
                    config.model_settings.model_name,
                    trust_remote_code=config.model_settings.trust_remote_code
                )
            except Exception as e:
                logger.error(f"处理器加载失败: {e}")
                processor = None
        else:
            logger.warning("AutoProcessor不可用，跳过处理器加载")

        # 获取 VLM backend（关键：dataset 通过 backend.encode_with_task 走特定后端的编码路径，
        # 否则会回退到裸 processor 拼接，对 Florence-2 会触发 task token 独占断言）
        backend = None
        if model is not None and hasattr(model, "_backend") and model._backend is not None:
            backend = model._backend
            logger.info(f"复用模型 backend: {type(backend).__name__}")

        # 创建训练数据集
        train_dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path="./data/images",
            config=config.data_settings,
            processor=processor,
            backend=backend,
        )

        # 创建验证数据集（如果配置了验证数据）
        val_dataset = None
        if config.val_data_path:
            val_data_configs = []
            for task in config.tasks:
                # 确保任务类型格式正确
                task_type = task.upper() if task else "CAPTION"
                val_data_configs.append({
                    "task_type": task_type,
                    "data_path": config.val_data_path,
                    "weight": config.task_weights.get(task, 1.0)
                })

            val_dataset = MultiTaskDataset(
                data_configs=val_data_configs,
                image_base_path="./data/images",
                config=config.data_settings,
                processor=processor,
                backend=backend,
            )

        logger.info(f"训练数据集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"验证数据集大小: {len(val_dataset)}")

        return train_dataset, val_dataset

    except Exception as e:
        logger.error(f"准备数据集时出错: {e}")
        raise
