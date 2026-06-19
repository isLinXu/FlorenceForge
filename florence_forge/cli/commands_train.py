"""CLI 训练命令处理。

处理 ``train`` 子命令：常规训练、TVP 训练、配置覆盖、数据集准备。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import TASK_CONFIG_MAPPING

logger = logging.getLogger(__name__)


def _select_trainer_class(version: str | None = None):
    """Return the canonical training stack (v1 removed in v2.0.0)."""
    if version and version.strip().lower() in {"v1", "legacy"}:
        raise ValueError(
            "v1 训练栈已在 v2.0.0 移除；请使用默认 v2 MultiTaskTrainer"
        )
    from florence_forge.training.trainer import MultiTaskTrainer

    return MultiTaskTrainer


def run_training_task(
    task: Optional[str] = None,
    config: Optional[str] = None,
    override: Optional[list] = None,
    **overrides
) -> bool:
    """运行训练任务"""
    if override:
        for key, value in override:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides[key] = value

    try:
        from florence_forge.core.model import Florence2MultiTaskModel
        from florence_forge.training.config import load_config_from_file
        MultiTaskTrainer = _select_trainer_class(overrides.pop("trainer_version", None))

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

        possible_paths = [
            config_path,
            Path.cwd() / config_path,
            Path(__file__).parent.parent / config_path,
        ]
        actual_config_path = None
        for path in possible_paths:
            if path.exists():
                actual_config_path = path
                break

        if not actual_config_path:
            logger.error(f"❌ 找不到配置文件: {config_path}")
            return False

        task_type_mapping = {
            "od": "OD",
            "detection": "OD",
            "caption": "CAPTION",
            "detailed_caption": "DETAILED_CAPTION",
            "more_detailed_caption": "MORE_DETAILED_CAPTION",
            "open_vocabulary_detection": "OPEN_VOCABULARY_DETECTION",
            "phrase_grounding": "CAPTION_TO_PHRASE_GROUNDING",
            "dense_region_caption": "DENSE_REGION_CAPTION",
            "region_proposal": "REGION_PROPOSAL",
            "region_to_category": "REGION_TO_CATEGORY",
            "region_to_description": "REGION_TO_DESCRIPTION",
            "ocr": "OCR",
            "ocr_with_region": "OCR_WITH_REGION",
            "segmentation": "REFERRING_EXPRESSION_SEGMENTATION",
            "seg": "REFERRING_EXPRESSION_SEGMENTATION",
            "region_to_segmentation": "REGION_TO_SEGMENTATION",
            "referring_expression_segmentation": "REFERRING_EXPRESSION_SEGMENTATION",
        }

        if task and task in task_type_mapping:
            overrides["task_type"] = task_type_mapping[task]

        logger.info("🚀 开始训练任务")
        logger.info(f"   任务类型: {task or 'custom'}")
        logger.info(f"   配置文件: {actual_config_path}")
        logger.info(f"   训练器: MultiTaskTrainer (模块化训练栈)")
        if overrides:
            logger.info(f"   参数覆盖: {overrides}")

        logger.info("📋 加载训练配置...")
        training_config = load_config_from_file(str(actual_config_path))
        if overrides:
            _apply_config_overrides(training_config, overrides)

        if "train_data" in overrides and overrides["train_data"] is not None:
            training_config.train_data_path = overrides["train_data"]
            logger.info(f"设置训练数据路径: {training_config.train_data_path}")
        if "val_data" in overrides and overrides["val_data"] is not None:
            training_config.val_data_path = overrides["val_data"]
            logger.info(f"设置验证数据路径: {training_config.val_data_path}")

        if training_config.train_data_path and not Path(training_config.train_data_path).exists():
            logger.warning(f"⚠️ 训练数据路径不存在: {training_config.train_data_path}")

        model_name = training_config.model_settings.model_name
        if "/" in model_name and not Path(model_name).exists():
            try:
                from florence_forge.utils.diagnostics import find_local_hf_snapshot
                snapshot = find_local_hf_snapshot(model_name)
                if snapshot is not None:
                    logger.info("使用本地 Hugging Face 缓存: %s", snapshot)
                    training_config.model_settings.model_name = str(snapshot)
            except Exception as exc:
                logger.debug("本地模型快照解析跳过: %s", exc)

        target_device = training_config.model_settings.device or training_config.device
        if target_device in ("mps", "cpu"):
            training_config.model_settings.device_map = None

        logger.info("🤖 初始化模型...")
        model = Florence2MultiTaskModel(training_config.model_settings)
        logger.info("📥 加载模型和处理器...")
        model.load()

        logger.info("📊 准备训练数据...")
        train_dataset, val_dataset = _prepare_datasets(training_config, model=model)

        logger.info("🏋️ 创建训练器...")
        trainer = MultiTaskTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=training_config,
        )

        resume_path = overrides.pop("resume", None)
        if resume_path:
            resume_checkpoint = Path(resume_path)
            if not resume_checkpoint.exists():
                logger.error(f"❌ 检查点路径不存在: {resume_checkpoint}")
                return False
            logger.info(f"📂 从检查点恢复训练: {resume_checkpoint}")
            trainer.load_checkpoint(resume_checkpoint)
            logger.info(f"   恢复到 Epoch {trainer.current_epoch}, Step {trainer.global_step}")

        logger.info("🚀 开始训练...")
        training_summary = trainer.train()

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


def run_tvp_training_task(
    *,
    tvp_config: Optional[str] = None,
    tvp_pipeline: Optional[str] = None,
    tvp_stage: Optional[str] = None,
    override: Optional[list] = None,
    **overrides,
) -> bool:
    """Run TVP SFT/OPD/GRPO or full pipeline via training bridges."""
    if override:
        for key, value in override:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides[key] = value

    try:
        if tvp_pipeline:
            from florence_forge.training.tvp_pipeline import create_pipeline_from_yaml
            pipeline_path = Path(tvp_pipeline)
            if not pipeline_path.exists():
                logger.error(f"❌ TVP pipeline 配置不存在: {pipeline_path}")
                return False
            logger.info("启动 TVP 三阶段 pipeline: %s", pipeline_path)
            pipeline = create_pipeline_from_yaml(str(pipeline_path))
            results = pipeline.run()
            logger.info("✅ TVP pipeline 完成: %s", results)
            return True

        if not tvp_config:
            logger.error("❌ 必须指定 --tvp-config 或 --tvp-pipeline")
            return False

        from florence_forge.training.tvp_training import (
            run_tvp_grpo,
            run_tvp_opd,
            run_tvp_sft_with_multitask_trainer,
        )
        config_path = Path(tvp_config)
        if not config_path.exists():
            logger.error(f"❌ TVP 配置不存在: {config_path}")
            return False

        stage_overrides: Dict[str, Any] = {}
        if overrides.get("epochs") is not None:
            stage_overrides["epochs"] = overrides["epochs"]
        if overrides.get("batch_size") is not None:
            stage_overrides["batch_size"] = overrides["batch_size"]
        if overrides.get("lr") is not None:
            stage_overrides["learning_rate"] = overrides["lr"]
        if overrides.get("output_dir") is not None:
            stage_overrides["output_dir"] = overrides["output_dir"]
        if overrides.get("model") is not None:
            stage_overrides["model_name_or_path"] = overrides["model"]
        if overrides.get("device") is not None:
            stage_overrides["device"] = overrides["device"]

        stage = (tvp_stage or "sft").lower()
        logger.info("启动 TVP %s 训练: %s", stage.upper(), config_path)
        if stage == "opd":
            summary = run_tvp_opd(
                config_path,
                checkpoint_dir=stage_overrides.get("output_dir"),
                overrides=stage_overrides or None,
            )
        elif stage == "grpo":
            summary = run_tvp_grpo(
                config_path,
                checkpoint_dir=stage_overrides.get("output_dir"),
                overrides=stage_overrides or None,
            )
        else:
            summary = run_tvp_sft_with_multitask_trainer(
                config_path,
                checkpoint_dir=stage_overrides.get("output_dir"),
                overrides=stage_overrides or None,
            )
        logger.info("✅ TVP %s 完成: %s", stage.upper(), summary)
        return True
    except Exception as exc:
        logger.error(f"❌ TVP 训练失败: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _coerce_override_value(value: Any) -> Any:
    """将 CLI --override 字符串转为合适的 Python 值。"""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in ("null", "none", ""):
        return None
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    return value


def _set_nested_attr(obj, attr_str, value):
    """递归设置对象的嵌套属性"""
    value = _coerce_override_value(value)
    alias_map = {
        "model_config": "model_settings",
        "data_config": "data_settings",
        "optimization_config": "optimization_settings",
        "task_scheduling_config": "task_scheduling_settings",
        "distributed_config": "distributed_settings",
    }
    attrs = attr_str.split(".")
    attrs[0] = alias_map.get(attrs[0], attrs[0])
    for attr in attrs[:-1]:
        attr = alias_map.get(attr, attr)
        obj = getattr(obj, attr)
    setattr(obj, alias_map.get(attrs[-1], attrs[-1]), value)


def _apply_config_overrides(config: Any, overrides: Dict[str, Any]) -> None:
    """应用命令行参数覆盖到配置"""
    try:
        if "epochs" in overrides and overrides["epochs"] is not None:
            config.num_epochs = overrides["epochs"]
            logger.info(f"覆盖训练轮数: {config.num_epochs}")
        if "batch_size" in overrides and overrides["batch_size"] is not None:
            config.data_settings.batch_size = overrides["batch_size"]
            logger.info(f"覆盖批次大小: {config.data_settings.batch_size}")
        if "lr" in overrides and overrides["lr"] is not None:
            config.optimization_settings.learning_rate = overrides["lr"]
            logger.info(f"覆盖学习率: {config.optimization_settings.learning_rate}")
        if "output_dir" in overrides and overrides["output_dir"] is not None:
            config.output_dir = overrides["output_dir"]
            logger.info(f"覆盖输出目录: {config.output_dir}")
        if "device" in overrides and overrides["device"] is not None:
            config.device = overrides["device"]
            if hasattr(config, "model_settings"):
                config.model_settings.device = overrides["device"]
            logger.info(f"覆盖训练设备: {config.device}")
        if "model" in overrides and overrides["model"] is not None:
            config.model_settings.model_name = overrides["model"]
            logger.info(f"覆盖模型名称: {config.model_settings.model_name}")
        if "train_data" in overrides and overrides["train_data"] is not None:
            config.train_data_path = overrides["train_data"]
            logger.info(f"覆盖训练数据路径: {config.train_data_path}")
        if "train_data_path" in overrides and overrides["train_data_path"] is not None:
            config.train_data_path = overrides["train_data_path"]
            logger.info(f"覆盖训练数据路径: {config.train_data_path}")
        if "val_data" in overrides and overrides["val_data"] is not None:
            config.val_data_path = overrides["val_data"]
            logger.info(f"覆盖验证数据路径: {config.val_data_path}")
        if "max_steps" in overrides and overrides["max_steps"] is not None:
            config.max_steps = int(overrides["max_steps"])
            logger.info(f"覆盖 max_steps: {config.max_steps}")
        if "val_data_path" in overrides and overrides["val_data_path"] is not None:
            config.val_data_path = overrides["val_data_path"]
            logger.info(f"覆盖验证数据路径: {config.val_data_path}")
        if "task_type" in overrides and overrides["task_type"] is not None:
            config.tasks = [overrides["task_type"]]
            config.task_weights = {overrides["task_type"]: 1.0}
            logger.info(f"覆盖任务类型: {config.tasks}")
        for key, value in overrides.items():
            if "." in key and value is not None:
                try:
                    _set_nested_attr(config, key, _coerce_override_value(value))
                    logger.info(f"覆盖配置: {key} = {value}")
                except AttributeError:
                    logger.warning(f"无法设置配置属性: {key}")
    except Exception as e:
        logger.warning(f"应用配置覆盖时出错: {e}")


def _resolve_image_base_path(config: Any, data_path: Optional[str] = None) -> str:
    """从配置或 JSONL 路径推断图像根目录。"""
    explicit = getattr(config, "image_base_path", None)
    if explicit:
        return str(explicit)
    candidate_path = data_path or config.train_data_path
    if candidate_path:
        parent = Path(candidate_path).expanduser().resolve().parent
        images_dir = parent / "images"
        if images_dir.is_dir():
            return str(images_dir)
        return str(parent)
    return "./data/images"


def _prepare_datasets(
    config: Any, model=None
) -> Tuple[Any, Optional[Any]]:
    """准备训练和验证数据集"""
    from florence_forge.data.dataset import MultiTaskDataset

    try:
        from transformers import AutoProcessor
    except ImportError:
        AutoProcessor = None

    data_configs = []
    for task in config.tasks:
        task_type = task.upper() if task else "CAPTION"
        if config.train_data_path:
            data_configs.append({
                "task_type": task_type,
                "data_path": config.train_data_path,
                "weight": config.task_weights.get(task, 1.0),
            })

    if not data_configs:
        data_configs = [{
            "task_type": "CAPTION",
            "data_path": "./data/sample_data.jsonl",
            "weight": 1.0,
        }]
        logger.warning("未配置训练数据路径，使用默认示例数据")

    processor = None
    if hasattr(model, "processor") and model.processor is not None:
        processor = model.processor
        logger.info("复用模型已有的 processor")
    elif AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                config.model_settings.model_name,
                trust_remote_code=config.model_settings.trust_remote_code,
            )
        except Exception as e:
            logger.error(f"处理器加载失败: {e}")
            processor = None
    else:
        logger.warning("AutoProcessor不可用，跳过处理器加载")

    backend = None
    if model is not None and hasattr(model, "_backend") and model._backend is not None:
        backend = model._backend
        logger.info(f"复用模型 backend: {type(backend).__name__}")

    image_base = _resolve_image_base_path(config, config.train_data_path)
    logger.info("图像基础路径: %s", image_base)
    train_dataset = MultiTaskDataset(
        data_configs=data_configs,
        image_base_path=image_base,
        config=config.data_settings,
        processor=processor,
        backend=backend,
    )

    val_dataset = None
    if config.val_data_path:
        val_data_configs = []
        for task in config.tasks:
            task_type = task.upper() if task else "CAPTION"
            val_data_configs.append({
                "task_type": task_type,
                "data_path": config.val_data_path,
                "weight": config.task_weights.get(task, 1.0),
            })
        val_image_base = _resolve_image_base_path(config, config.val_data_path)
        val_dataset = MultiTaskDataset(
            data_configs=val_data_configs,
            image_base_path=val_image_base,
            config=config.data_settings,
            processor=processor,
            backend=backend,
        )

    logger.info(f"训练数据集大小: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"验证数据集大小: {len(val_dataset)}")

    return train_dataset, val_dataset
