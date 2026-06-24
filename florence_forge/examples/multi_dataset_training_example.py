#!/usr/bin/env python3
"""
Florence Forge - 多数据集多任务训练示例

展示如何使用MultiDatasetManager和MultiDatasetTrainer进行多数据集多任务训练
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from transformers import AutoProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入Florence Forge组件
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.config import TrainingConfig, ModelConfig
from florence_forge.data.multi_dataset_manager import MultiDatasetManager, DatasetInfo, TaskDatasetMapping
from florence_forge.training.multi_dataset_trainer import MultiDatasetTrainer

def create_sample_dataset_config() -> Dict[str, Any]:
    """创建示例数据集配置
    
    Returns:
        数据集配置字典
    """
    config = {
        "datasets": {
            "coco_captions": {
                "name": "coco_captions",
                "path": "/path/to/coco/captions",
                "format": "coco",
                "task_types": ["CAPTION"],
                "priority": 1.0,
                "max_samples": 50000,
                "preprocessing": {
                    "image_size": [384, 384],
                    "normalize": True
                },
                "metadata": {
                    "description": "COCO图像描述数据集",
                    "version": "2017",
                    "language": "en"
                }
            },
            "vqa_v2": {
                "name": "vqa_v2",
                "path": "/path/to/vqa/v2",
                "format": "vqa",
                "task_types": ["visual_question_answering"],
                "priority": 1.2,
                "max_samples": 30000,
                "preprocessing": {
                    "image_size": [384, 384],
                    "normalize": True,
                    "max_question_length": 128
                },
                "metadata": {
                    "description": "Visual Question Answering v2数据集",
                    "version": "2.0",
                    "language": "en"
                }
            },
            "refcoco": {
                "name": "refcoco",
                "path": "/path/to/refcoco",
                "format": "referring_expression",
                "task_types": ["referring_expression_comprehension"],
                "priority": 0.8,
                "max_samples": 20000,
                "preprocessing": {
                    "image_size": [384, 384],
                    "normalize": True,
                    "max_expression_length": 64
                },
                "metadata": {
                    "description": "RefCOCO指代表达理解数据集",
                    "version": "1.0",
                    "language": "en"
                }
            },
            "docvqa": {
                "name": "docvqa",
                "path": "/path/to/docvqa",
                "format": "document_qa",
                "task_types": ["document_question_answering"],
                "priority": 1.5,
                "max_samples": 15000,
                "preprocessing": {
                    "image_size": [768, 768],  # 文档需要更高分辨率
                    "normalize": True,
                    "max_question_length": 256
                },
                "metadata": {
                    "description": "文档问答数据集",
                    "version": "1.0",
                    "language": "en"
                }
            },
            "chinese_captions": {
                "name": "chinese_captions",
                "path": "/path/to/chinese/captions",
                "format": "custom",
                "task_types": ["CAPTION"],
                "priority": 0.9,
                "max_samples": 25000,
                "preprocessing": {
                    "image_size": [384, 384],
                    "normalize": True
                },
                "metadata": {
                    "description": "中文图像描述数据集",
                    "version": "1.0",
                    "language": "zh"
                }
            }
        },
        "task_mappings": {
            "CAPTION": {
                "datasets": ["coco_captions", "chinese_captions"],
                "weights": [0.7, 0.3],
                "sampling_strategy": "weighted_random"
            },
            "visual_question_answering": {
                "datasets": ["vqa_v2"],
                "weights": [1.0],
                "sampling_strategy": "sequential"
            },
            "referring_expression_comprehension": {
                "datasets": ["refcoco"],
                "weights": [1.0],
                "sampling_strategy": "sequential"
            },
            "document_question_answering": {
                "datasets": ["docvqa"],
                "weights": [1.0],
                "sampling_strategy": "sequential"
            }
        },
        "global_settings": {
            "enable_balanced_sampling": True,
            "cross_dataset_validation": True,
            "adaptive_dataset_weighting": True,
            "task_curriculum_learning": False,
            "max_total_samples": 100000,
            "validation_ratio": 0.15,
            "test_ratio": 0.05
        }
    }
    
    return config

def setup_multi_dataset_manager(config_path: Optional[str] = None) -> MultiDatasetManager:
    """设置多数据集管理器
    
    Args:
        config_path: 配置文件路径，None表示使用示例配置
        
    Returns:
        配置好的多数据集管理器
    """
    logger.info("正在设置多数据集管理器...")
    
    if config_path and Path(config_path).exists():
        # 从文件加载配置
        manager = MultiDatasetManager.load_configuration(config_path)
    else:
        # 使用示例配置
        manager = MultiDatasetManager()
        
        # 注册数据集
        datasets_config = create_sample_dataset_config()["datasets"]
        for dataset_name, dataset_config in datasets_config.items():
            dataset_info = DatasetInfo(
                name=dataset_config["name"],
                path=dataset_config["path"],
                format=dataset_config["format"],
                task_types=dataset_config["task_types"],
                priority=dataset_config["priority"],
                max_samples=dataset_config.get("max_samples"),
                preprocessing=dataset_config.get("preprocessing", {}),
                metadata=dataset_config.get("metadata", {})
            )
            manager.register_dataset(dataset_info)
        
        # 添加任务映射
        task_mappings = create_sample_dataset_config()["task_mappings"]
        for task_type, mapping_config in task_mappings.items():
            mapping = TaskDatasetMapping(
                task_type=task_type,
                datasets=mapping_config["datasets"],
                weights=mapping_config["weights"],
                sampling_strategy=mapping_config["sampling_strategy"]
            )
            manager.add_task_mapping(mapping)
        
        # 应用全局设置
        global_settings = create_sample_dataset_config()["global_settings"]
        if global_settings.get("enable_balanced_sampling"):
            manager.enable_balanced_sampling()
        
        if global_settings.get("max_total_samples"):
            manager.limit_total_samples(global_settings["max_total_samples"])
    
    logger.info("多数据集管理器设置完成")
    return manager

def create_training_config() -> TrainingConfig:
    """创建训练配置
    
    Returns:
        训练配置对象
    """
    config = TrainingConfig(
        # 基础训练参数
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-5,
        warmup_steps=1000,
        max_steps=50000,
        
        # 评估参数
        eval_steps=1000,
        eval_ratio=0.15,
        save_steps=2000,
        logging_steps=100,
        
        # 优化参数
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        weight_decay=0.01,
        
        # 多数据集特定参数
        adaptive_dataset_weighting=True,
        cross_dataset_validation=True,
        task_curriculum_learning=False,
        
        # 输出设置
        output_dir="./outputs/multi_dataset_training",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 其他设置
        seed=42,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    
    return config

def setup_model_and_processor() -> tuple:
    """设置模型和处理器
    
    Returns:
        (模型, 处理器) 元组
    """
    logger.info("正在加载模型和处理器...")
    
    # 模型配置
    model_config = ModelConfig(
        model_name="microsoft/Florence-2-base",
        use_lora=True,
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
            "lora_dropout": 0.1
        },
        gradient_checkpointing=True
    )
    
    # 加载模型
    model = Florence2MultiTaskModel.from_pretrained(
        model_config.model_name,
        config=model_config
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_config.model_name)
    
    logger.info("模型和处理器加载完成")
    return model, processor

def main():
    """主函数 - 多数据集多任务训练示例"""
    logger.info("开始多数据集多任务训练示例")
    
    try:
        # 1. 设置多数据集管理器
        dataset_manager = setup_multi_dataset_manager()
        
        # 2. 创建训练配置
        training_config = create_training_config()
        
        # 3. 设置模型和处理器
        model, processor = setup_model_and_processor()
        
        # 4. 创建多数据集训练器
        trainer = MultiDatasetTrainer(
            model=model,
            dataset_manager=dataset_manager,
            config=training_config,
            task_types=[  # 指定要训练的任务类型
                "CAPTION",
                "visual_question_answering",
                "referring_expression_comprehension",
                "document_question_answering"
            ]
        )
        
        # 5. 开始训练
        logger.info("开始训练...")
        training_results = trainer.train()
        
        # 6. 输出训练结果
        logger.info("训练完成！")
        logger.info(f"最终训练损失: {training_results.get('train_loss', 'N/A')}")
        logger.info(f"最终验证损失: {training_results.get('eval_loss', 'N/A')}")
        
        # 7. 保存数据集性能分析
        performance_summary = trainer.get_dataset_performance_summary()
        logger.info("数据集性能摘要:")
        for dataset_name, performance in performance_summary["dataset_performance"].items():
            logger.info(f"  {dataset_name}: {performance}")
        
        # 8. 保存配置以供后续使用
        config_save_path = Path(training_config.output_dir) / "dataset_config.json"
        dataset_manager.save_configuration(config_save_path)
        logger.info(f"数据集配置已保存到: {config_save_path}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

def create_config_file_example():
    """创建配置文件示例"""
    config = create_sample_dataset_config()
    
    # 保存到文件
    config_path = "multi_dataset_config_example.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"示例配置文件已创建: {config_path}")
    return config_path

def load_and_train_from_config(config_path: str):
    """从配置文件加载并训练
    
    Args:
        config_path: 数据集配置文件路径
    """
    logger.info(f"从配置文件加载训练: {config_path}")
    
    # 设置模型
    model, processor = setup_model_and_processor()
    
    # 创建训练配置
    training_config = create_training_config()
    
    # 从配置文件创建训练器
    trainer = MultiDatasetTrainer.from_config(
        model=model,
        dataset_config_path=config_path,
        training_config=training_config,
        task_types=["CAPTION", "OD"]
    )
    
    # 开始训练
    results = trainer.train()
    
    logger.info("从配置文件的训练完成")
    return results

if __name__ == "__main__":
    # 示例1: 直接创建和训练
    print("=== 示例1: 直接创建多数据集训练 ===")
    main()
    
    # 示例2: 创建配置文件
    print("\n=== 示例2: 创建配置文件 ===")
    config_file = create_config_file_example()
    
    # 示例3: 从配置文件加载训练
    print("\n=== 示例3: 从配置文件加载训练 ===")
    # load_and_train_from_config(config_file)
    
    print("\n所有示例完成！")