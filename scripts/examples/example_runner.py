#!/usr/bin/env python3
"""
示例运行器 - 提供完整的使用示例和教程脚本
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"警告: 无法导入必要的依赖: {e}")
    print("请运行: pip install -r requirements.txt")

from florence_forge.utils.logging import setup_logging

logger = logging.getLogger(__name__)

class ExampleRunner:
    """示例运行器
    
    提供各种使用示例和教程，包括：
    - 基础配置示例
    - 数据处理示例
    - 模型使用示例
    - 训练流程示例
    - 评估示例
    """
    
    def __init__(self, output_dir: str = "./example_outputs"):
        """初始化示例运行器
        
        Args:
            output_dir: 示例输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        setup_logging(
            level=logging.INFO,
            log_file=self.output_dir / "examples.log"
        )
        
        logger.info("示例运行器初始化完成")
    
    def example_basic_config(self) -> Dict[str, Any]:
        """基础配置示例"""
        logger.info("运行基础配置示例...")
        
        try:
            from florence_forge.core.config import (
                ModelConfig,
                TrainingConfig,
                DataConfig,
                LoRAConfig
            )
            
            # 1. 创建模型配置
            model_config = ModelConfig(
                model_name="microsoft/Florence-2-base",
                use_lora=True,
                device_map="auto"
            )
            
            # 2. 创建LoRA配置
            lora_config = LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1
            )
            
            # 3. 创建训练配置
            training_config = TrainingConfig(
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-4,
                warmup_steps=100,
                save_steps=500,
                eval_steps=250,
                output_dir="./outputs/example_training"
            )
            
            # 4. 创建数据配置
            data_config = DataConfig(
                max_length=512,
                image_size=224,
                num_workers=4
            )
            
            # 5. 保存配置示例
            configs = {
                "model_config": model_config.to_dict(),
                "lora_config": lora_config.to_dict(),
                "training_config": training_config.__dict__,
                "data_config": data_config.__dict__
            }
            
            config_file = self.output_dir / "example_configs.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置示例已保存到: {config_file}")
            
            return {
                "status": "success",
                "message": "基础配置示例运行成功",
                "config_file": str(config_file),
                "configs": configs
            }
            
        except Exception as e:
            logger.error(f"基础配置示例失败: {e}")
            return {
                "status": "error",
                "message": f"基础配置示例失败: {str(e)}"
            }
    
    def example_data_processing(self) -> Dict[str, Any]:
        """数据处理示例"""
        logger.info("运行数据处理示例...")
        
        try:
            from florence_forge.data.dataset import TaskSample, MultiTaskDataset
            from florence_forge.data.builder import DatasetBuilder
            
            # 1. 创建任务样本
            samples = []
            
            # 图像描述任务样本
            caption_sample = TaskSample(
                task_type="CAPTION",
                image_path="example_images/cat.jpg",
                prefix="<CAPTION>",
                suffix="A cute cat sitting on a windowsill",
                weight=1.0,
                metadata={"source": "example", "quality": "high"}
            )
            samples.append(caption_sample)
            
            # 详细描述任务样本
            detailed_caption_sample = TaskSample(
                task_type="DETAILED_CAPTION",
                image_path="example_images/dog.jpg",
                prefix="<DETAILED_CAPTION>",
                suffix="A golden retriever dog running happily in a green park with trees in the background",
                weight=1.2,
                metadata={"source": "example", "quality": "high"}
            )
            samples.append(detailed_caption_sample)
            
            # 目标检测任务样本
            od_sample = TaskSample(
                task_type="OD",
                image_path="example_images/street.jpg",
                prefix="<OD>",
                suffix="car<loc_123><loc_456><loc_789><loc_012>person<loc_234><loc_567><loc_890><loc_123>",
                weight=1.5,
                metadata={"source": "example", "objects": ["car", "person"]}
            )
            samples.append(od_sample)
            
            # 2. 使用数据集构建器
            builder = DatasetBuilder()
            
            # 添加图像描述数据
            builder.add_task_data('caption', {
                'images': ['example_images/cat.jpg', 'example_images/dog.jpg'],
                'texts': ['A cute cat', 'A happy dog']
            })
            
            # 添加目标检测数据
            builder.add_task_data('object_detection', {
                'images': ['example_images/street.jpg'],
                'annotations': [{
                    'objects': [{'name': 'car', 'bbox': [123, 456, 789, 12]},
                               {'name': 'person', 'bbox': [234, 567, 890, 123]}]
                }]
            })
            
            # 3. 保存样本示例
            samples_data = [sample.to_dict() for sample in samples]
            samples_file = self.output_dir / "example_samples.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, indent=2, ensure_ascii=False)
            
            # 4. 创建示例图像（用于演示）
            self._create_example_images()
            
            logger.info(f"数据处理示例已保存到: {samples_file}")
            
            return {
                "status": "success",
                "message": "数据处理示例运行成功",
                "samples_file": str(samples_file),
                "samples_count": len(samples),
                "task_types": list(set(sample.task_type for sample in samples))
            }
            
        except Exception as e:
            logger.error(f"数据处理示例失败: {e}")
            return {
                "status": "error",
                "message": f"数据处理示例失败: {str(e)}"
            }
    
    def example_model_usage(self) -> Dict[str, Any]:
        """模型使用示例（轻量级）"""
        logger.info("运行模型使用示例...")
        
        try:
            from florence_forge.core.model import Florence2MultiTaskModel
            from florence_forge.core.config import ModelConfig
            from florence_forge.core.tasks import FLORENCE2_TASKS, get_task_config
            
            # 1. 展示任务配置
            available_tasks = list(FLORENCE2_TASKS.keys())
            task_configs = {}
            
            for task in available_tasks[:5]:  # 只展示前5个任务
                task_config = get_task_config(task)
                task_configs[task] = task_config
            
            # 2. 创建模型配置（不实际加载模型）
            model_config = ModelConfig(
                model_name="microsoft/Florence-2-base",
                use_lora=True,
                device_map="cpu"  # 使用CPU避免GPU内存问题
            )
            
            # 3. 模拟推理流程
            inference_example = {
                "input": {
                    "image_path": "example_images/cat.jpg",
                    "task": "<CAPTION>",
                    "max_new_tokens": 50
                },
                "expected_output": {
                    "text": "A cute cat sitting on a windowsill",
                    "confidence": 0.95
                }
            }
            
            # 4. 保存模型使用示例
            model_usage_data = {
                "model_config": model_config.to_dict(),
                "available_tasks": available_tasks,
                "task_configs": task_configs,
                "inference_example": inference_example,
                "usage_notes": [
                    "模型支持多种视觉任务",
                    "可以使用LoRA进行高效微调",
                    "支持批量推理",
                    "自动设备管理"
                ]
            }
            
            usage_file = self.output_dir / "example_model_usage.json"
            with open(usage_file, 'w', encoding='utf-8') as f:
                json.dump(model_usage_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型使用示例已保存到: {usage_file}")
            
            return {
                "status": "success",
                "message": "模型使用示例运行成功",
                "usage_file": str(usage_file),
                "available_tasks_count": len(available_tasks),
                "model_name": model_config.model_name
            }
            
        except Exception as e:
            logger.error(f"模型使用示例失败: {e}")
            return {
                "status": "error",
                "message": f"模型使用示例失败: {str(e)}"
            }
    
    def example_training_workflow(self) -> Dict[str, Any]:
        """训练流程示例"""
        logger.info("运行训练流程示例...")
        
        try:
            from florence_forge.training.trainer_refactored import MultiTaskTrainer
            from florence_forge.training.scheduler import TaskScheduler
            from florence_forge.core.config import TrainingConfig
            
            # 1. 创建训练配置
            training_config = TrainingConfig(
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-4,
                warmup_steps=100,
                save_steps=500,
                eval_steps=250,
                output_dir=str(self.output_dir / "training_output"),
                logging_steps=50,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # 2. 任务调度配置
            task_schedule = {
                "CAPTION": {"weight": 1.0, "frequency": 0.4},
                "DETAILED_CAPTION": {"weight": 1.2, "frequency": 0.3},
                "OD": {"weight": 1.5, "frequency": 0.2},
                "DENSE_REGION_CAPTION": {"weight": 1.1, "frequency": 0.1}
            }
            
            # 3. 训练流程步骤
            training_steps = [
                "1. 数据预处理和验证",
                "2. 模型初始化和LoRA配置",
                "3. 优化器和学习率调度器设置",
                "4. 训练循环开始",
                "5. 每个epoch的训练步骤",
                "6. 定期评估和检查点保存",
                "7. 早停和最佳模型选择",
                "8. 训练完成和结果保存"
            ]
            
            # 4. 监控指标
            monitoring_metrics = {
                "training_metrics": [
                    "train_loss",
                    "train_accuracy",
                    "learning_rate",
                    "gradient_norm"
                ],
                "evaluation_metrics": [
                    "eval_loss",
                    "eval_accuracy",
                    "bleu_score",
                    "rouge_score"
                ],
                "system_metrics": [
                    "memory_usage",
                    "gpu_utilization",
                    "training_speed"
                ]
            }
            
            # 5. 保存训练流程示例
            training_workflow_data = {
                "training_config": training_config.__dict__,
                "task_schedule": task_schedule,
                "training_steps": training_steps,
                "monitoring_metrics": monitoring_metrics,
                "best_practices": [
                    "使用梯度累积处理大批量",
                    "定期保存检查点",
                    "监控过拟合",
                    "使用验证集进行早停",
                    "记录详细的训练日志"
                ]
            }
            
            workflow_file = self.output_dir / "example_training_workflow.json"
            with open(workflow_file, 'w', encoding='utf-8') as f:
                json.dump(training_workflow_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练流程示例已保存到: {workflow_file}")
            
            return {
                "status": "success",
                "message": "训练流程示例运行成功",
                "workflow_file": str(workflow_file),
                "num_epochs": training_config.num_epochs,
                "task_count": len(task_schedule)
            }
            
        except Exception as e:
            logger.error(f"训练流程示例失败: {e}")
            return {
                "status": "error",
                "message": f"训练流程示例失败: {str(e)}"
            }
    
    def example_evaluation_workflow(self) -> Dict[str, Any]:
        """评估流程示例"""
        logger.info("运行评估流程示例...")
        
        try:
            from florence_forge.evaluation.evaluator import MultiTaskEvaluator
            from florence_forge.evaluation.metrics import MetricCalculator
            
            # 1. 评估配置
            evaluation_config = {
                "batch_size": 8,
                "num_beams": 4,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True
            }
            
            # 2. 评估指标配置
            metrics_config = {
                "CAPTION": ["bleu", "rouge", "meteor", "cider"],
                "DETAILED_CAPTION": ["bleu", "rouge", "meteor"],
                "OD": ["map", "precision", "recall"],
                "DENSE_REGION_CAPTION": ["bleu", "rouge"]
            }
            
            # 3. 模拟评估结果
            mock_evaluation_results = {
                "CAPTION": {
                    "bleu_1": 0.75,
                    "bleu_4": 0.45,
                    "rouge_l": 0.68,
                    "meteor": 0.52,
                    "cider": 1.23,
                    "sample_count": 1000
                },
                "DETAILED_CAPTION": {
                    "bleu_1": 0.72,
                    "bleu_4": 0.42,
                    "rouge_l": 0.65,
                    "meteor": 0.49,
                    "sample_count": 500
                },
                "OD": {
                    "map_50": 0.68,
                    "map_75": 0.45,
                    "precision": 0.72,
                    "recall": 0.69,
                    "sample_count": 800
                }
            }
            
            # 4. 评估流程步骤
            evaluation_steps = [
                "1. 加载训练好的模型",
                "2. 准备评估数据集",
                "3. 配置评估参数",
                "4. 执行模型推理",
                "5. 计算评估指标",
                "6. 生成评估报告",
                "7. 可视化结果",
                "8. 保存评估结果"
            ]
            
            # 5. 保存评估流程示例
            evaluation_workflow_data = {
                "evaluation_config": evaluation_config,
                "metrics_config": metrics_config,
                "evaluation_steps": evaluation_steps,
                "mock_results": mock_evaluation_results,
                "analysis_tips": [
                    "比较不同任务的性能",
                    "分析错误案例",
                    "检查模型偏差",
                    "评估泛化能力",
                    "监控推理速度"
                ]
            }
            
            eval_file = self.output_dir / "example_evaluation_workflow.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_workflow_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估流程示例已保存到: {eval_file}")
            
            return {
                "status": "success",
                "message": "评估流程示例运行成功",
                "eval_file": str(eval_file),
                "task_count": len(metrics_config),
                "metric_types": len(set().union(*metrics_config.values()))
            }
            
        except Exception as e:
            logger.error(f"评估流程示例失败: {e}")
            return {
                "status": "error",
                "message": f"评估流程示例失败: {str(e)}"
            }
    
    def _create_example_images(self) -> None:
        """创建示例图像文件"""
        try:
            # 创建示例图像目录
            image_dir = self.output_dir / "example_images"
            image_dir.mkdir(exist_ok=True)
            
            # 创建简单的示例图像
            for name, color in [("cat", (255, 200, 100)), ("dog", (200, 150, 100)), ("street", (150, 150, 150))]:
                img_array = np.full((224, 224, 3), color, dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(image_dir / f"{name}.jpg")
            
            logger.info(f"示例图像已创建在: {image_dir}")
            
        except Exception as e:
            logger.warning(f"创建示例图像失败: {e}")
    
    def run_all_examples(self) -> Dict[str, Any]:
        """运行所有示例
        
        Returns:
            完整的示例运行结果
        """
        logger.info("开始运行完整示例套件")
        
        examples = [
            ("basic_config", self.example_basic_config),
            ("data_processing", self.example_data_processing),
            ("model_usage", self.example_model_usage),
            ("training_workflow", self.example_training_workflow),
            ("evaluation_workflow", self.example_evaluation_workflow)
        ]
        
        results = {}
        
        for example_name, example_func in examples:
            try:
                result = example_func()
                results[example_name] = result
                logger.info(f"{example_name}: {result['status']}")
            except Exception as e:
                logger.error(f"示例 {example_name} 执行失败: {e}")
                results[example_name] = {
                    "status": "error",
                    "message": f"示例执行失败: {str(e)}"
                }
        
        # 生成总结
        summary = {
            "total_examples": len(examples),
            "successful": sum(1 for r in results.values() if r["status"] == "success"),
            "failed": sum(1 for r in results.values() if r["status"] == "error"),
            "output_directory": str(self.output_dir),
            "results": results
        }
        
        # 保存总结
        summary_file = self.output_dir / "examples_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"示例运行完成，结果已保存到: {summary_file}")
        return summary

def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlorenceForge示例运行器")
    parser.add_argument("--example", choices=["config", "data", "model", "training", "evaluation", "all"], 
                       default="all", help="要运行的示例类型")
    parser.add_argument("--output-dir", default="./example_outputs", help="示例输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # 创建示例运行器
    runner = ExampleRunner(output_dir=args.output_dir)
    
    # 运行指定的示例
    if args.example == "config":
        results = runner.example_basic_config()
    elif args.example == "data":
        results = runner.example_data_processing()
    elif args.example == "model":
        results = runner.example_model_usage()
    elif args.example == "training":
        results = runner.example_training_workflow()
    elif args.example == "evaluation":
        results = runner.example_evaluation_workflow()
    else:  # all
        results = runner.run_all_examples()
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("示例运行结果")
    print("="*50)
    
    if args.example == "all":
        print(f"总计: {results['total_examples']}")
        print(f"成功: {results['successful']}")
        print(f"失败: {results['failed']}")
        print(f"输出目录: {results['output_directory']}")
        
        if args.verbose:
            print("\n详细结果:")
            for name, result in results['results'].items():
                status_symbol = "✓" if result['status'] == "success" else "✗"
                print(f"  {status_symbol} {name}: {result['message']}")
    else:
        status_symbol = "✓" if results['status'] == "success" else "✗"
        print(f"{status_symbol} {args.example}: {results['message']}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()