#!/usr/bin/env python3
"""模型合并和Benchmark评估示例脚本

展示如何使用FlorenceForge的模型合并和benchmark评估功能
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.training.model_merger import ModelMerger
from florence_forge.training.lora_manager import LoRAManager
from florence_forge.evaluation.benchmark import BenchmarkEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMergeAndBenchmarkDemo:
    """模型合并和Benchmark评估演示类"""
    
    def __init__(self, output_dir: str = "./demo_outputs"):
        """初始化演示类
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.lora_manager = LoRAManager()
        self.model_merger = ModelMerger(self.lora_manager)
        
        logger.info(f"演示初始化完成，输出目录: {self.output_dir}")
    
    def demo_lora_merge(self) -> None:
        """演示LoRA权重合并"""
        logger.info("=== LoRA权重合并演示 ===")
        
        try:
            # 1. 加载基础模型
            logger.info("加载基础模型...")
            base_model = Florence2MultiTaskModel(
                model_name="microsoft/Florence-2-base",
                device="cpu"  # 演示使用CPU
            )
            
            # 2. 模拟LoRA训练后的模型
            logger.info("创建LoRA模型...")
            lora_model = self.lora_manager.apply_lora(
                base_model, 
                tasks=["caption", "detection"]
            )
            
            # 3. 合并LoRA权重
            logger.info("合并LoRA权重...")
            merged_model_dir = self.output_dir / "merged_model"
            
            # 使用merge_and_unload方法
            self.model_merger.merge_and_unload(
                lora_model,
                merged_model_dir,
                save_tokenizer=True,
                save_processor=True
            )
            
            # 4. 验证合并后的模型
            logger.info("验证合并后的模型...")
            validation_results = self.model_merger.validate_merged_model(
                base_model  # 简化演示，实际应该加载合并后的模型
            )
            
            logger.info(f"验证结果: {validation_results}")
            
            # 5. 导出不同格式
            logger.info("导出模型到不同格式...")
            export_dir = self.output_dir / "exported_models"
            
            # 导出PyTorch格式
            self.model_merger.export_merged_model(
                base_model,
                export_dir / "pytorch",
                export_format="pytorch"
            )
            
            # 尝试导出ONNX格式（如果支持）
            try:
                self.model_merger.export_merged_model(
                    base_model,
                    export_dir / "onnx",
                    export_format="onnx"
                )
            except Exception as e:
                logger.warning(f"ONNX导出失败: {e}")
            
            logger.info("LoRA权重合并演示完成")
            
        except Exception as e:
            logger.error(f"LoRA权重合并演示失败: {e}")
    
    def demo_multi_adapter_merge(self) -> None:
        """演示多适配器合并"""
        logger.info("=== 多适配器合并演示 ===")
        
        try:
            # 1. 加载基础模型
            base_model = Florence2MultiTaskModel(  # noqa: F841
                model_name="microsoft/Florence-2-base",
                device="cpu"
            )
            
            # 2. 模拟多个适配器路径
            adapter_paths = {
                "caption": "./adapters/caption_lora",
                "detection": "./adapters/detection_lora",
                "segmentation": "./adapters/segmentation_lora"
            }
            
            # 3. 设置适配器权重
            adapter_weights = {
                "caption": 1.0,
                "detection": 0.8,
                "segmentation": 0.6
            }
            
            # 注意：这里只是演示API，实际需要真实的适配器文件
            logger.info("多适配器合并功能已准备就绪")
            logger.info(f"适配器路径: {adapter_paths}")
            logger.info(f"适配器权重: {adapter_weights}")
            
            # 实际合并代码（需要真实适配器文件）:
            # merged_model = self.model_merger.merge_multiple_adapters(
            #     base_model,
            #     adapter_paths,
            #     adapter_weights,
            #     self.output_dir / "multi_merged_model"
            # )
            
            logger.info("多适配器合并演示完成")
            
        except Exception as e:
            logger.error(f"多适配器合并演示失败: {e}")
    
    def demo_benchmark_evaluation(self) -> None:
        """演示Benchmark评估"""
        logger.info("=== Benchmark评估演示 ===")
        
        try:
            # 1. 加载模型
            model = Florence2MultiTaskModel(
                model_name="microsoft/Florence-2-base",
                device="cpu"
            )
            
            # 2. 创建Benchmark评估器
            benchmark_config = {
                'batch_size': 4,
                'num_workers': 2,
                'max_samples_per_task': 100,
                'save_predictions': True,
                'compute_detailed_metrics': True
            }
            
            benchmark_evaluator = BenchmarkEvaluator(
                model=model,
                device=model.device,
                benchmark_config=benchmark_config
            )
            
            # 3. 准备评估数据集（演示用）
            logger.info("准备评估数据集...")
            
            # 注意：这里需要真实的数据集文件
            datasets = {
                # "coco_caption": MultiTaskDataset(
                #     data_configs=[
                #         {
                #             "task_type": "CAPTION",
                #             "data_path": "./data/coco_caption_val.jsonl",
                #             "weight": 1.0
                #         }
                #     ]
                # ),
                # "voc_detection": MultiTaskDataset(
                #     data_configs=[
                #         {
                #             "task_type": "DETECTION",
                #             "data_path": "./data/voc_detection_val.jsonl",
                #             "weight": 1.0
                #         }
                #     ]
                # )
            }
            
            if not datasets:
                logger.info("演示数据集为空，跳过实际评估")
                logger.info("实际使用时，请提供真实的数据集文件")
                return
            
            # 4. 运行Benchmark评估
            logger.info("运行Benchmark评估...")
            benchmark_results = benchmark_evaluator.run_benchmark(
                datasets=datasets,
                output_dir=self.output_dir / "benchmark_results",
                save_detailed=True
            )
            
            # 5. 生成报告
            logger.info("生成Benchmark报告...")
            report_dir = self.output_dir / "benchmark_reports"
            report_dir.mkdir(exist_ok=True)
            
            # 生成不同格式的报告
            benchmark_evaluator.generate_benchmark_report(
                benchmark_results,
                report_dir / "benchmark_report.md",
                format="markdown"
            )
            
            benchmark_evaluator.generate_benchmark_report(
                benchmark_results,
                report_dir / "benchmark_report.json",
                format="json"
            )
            
            logger.info("Benchmark评估演示完成")
            
        except Exception as e:
            logger.error(f"Benchmark评估演示失败: {e}")
    
    def demo_single_task_evaluation(self) -> None:
        """演示单任务评估"""
        logger.info("=== 单任务评估演示 ===")
        
        try:
            # 1. 加载模型
            model = Florence2MultiTaskModel(
                model_name="microsoft/Florence-2-base",
                device="cpu"
            )
            
            # 2. 创建Benchmark评估器
            benchmark_evaluator = BenchmarkEvaluator(model=model)
            
            # 3. 演示标准指标计算
            logger.info("演示标准指标计算...")
            
            # 模拟预测和参考数据
            predictions = [
                "a cat sitting on a table",
                "a dog running in the park",
                "a bird flying in the sky"
            ]
            
            references = [
                "a cat is sitting on the table",
                "a dog is running in the park", 
                "a bird is flying in the sky"
            ]
            
            # 计算图像描述任务指标
            caption_metrics = benchmark_evaluator.compute_standard_metrics(
                predictions=predictions,
                references=references,
                task_type="caption"
            )
            
            logger.info(f"图像描述指标: {caption_metrics}")
            
            # 4. 保存单任务评估结果
            task_output_dir = self.output_dir / "single_task_evaluation"
            task_output_dir.mkdir(exist_ok=True)
            
            import json
            with open(task_output_dir / "caption_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(caption_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info("单任务评估演示完成")
            
        except Exception as e:
            logger.error(f"单任务评估演示失败: {e}")
    
    def demo_baseline_comparison(self) -> None:
        """演示基线比较"""
        logger.info("=== 基线比较演示 ===")
        
        try:
            # 1. 模拟当前模型结果
            current_results = {
                'overall_summary': {
                    'average_metrics': {
                        'bleu_4': 0.25,
                        'rouge_l': 0.45,
                        'cider': 0.85
                    }
                },
                'task_performance': {
                    'caption': {
                        'average_metrics': {
                            'bleu_4': {'mean': 0.25, 'std': 0.05},
                            'rouge_l': {'mean': 0.45, 'std': 0.08}
                        }
                    }
                }
            }
            
            # 2. 模拟基线结果
            baseline_results = {
                'overall_summary': {
                    'average_metrics': {
                        'bleu_4': 0.20,
                        'rouge_l': 0.40,
                        'cider': 0.75
                    }
                },
                'task_performance': {
                    'caption': {
                        'average_metrics': {
                            'bleu_4': {'mean': 0.20, 'std': 0.04},
                            'rouge_l': {'mean': 0.40, 'std': 0.07}
                        }
                    }
                }
            }
            
            # 3. 创建评估器并比较
            model = Florence2MultiTaskModel(
                model_name="microsoft/Florence-2-base",
                device="cpu"
            )
            
            benchmark_evaluator = BenchmarkEvaluator(model=model)
            
            # 使用内部方法进行比较
            comparison_results = benchmark_evaluator._compare_with_baseline(
                current_results, baseline_results
            )
            
            logger.info("基线比较结果:")
            for metric, comparison in comparison_results.get('overall_improvement', {}).items():
                improvement = comparison['relative'] * 100
                logger.info(f"  {metric}: {improvement:+.2f}% 改进")
            
            # 4. 保存比较结果
            comparison_dir = self.output_dir / "baseline_comparison"
            comparison_dir.mkdir(exist_ok=True)
            
            import json
            with open(comparison_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            logger.info("基线比较演示完成")
            
        except Exception as e:
            logger.error(f"基线比较演示失败: {e}")
    
    def run_all_demos(self) -> None:
        """运行所有演示"""
        logger.info("开始运行所有演示...")
        
        demos = [
            ("LoRA权重合并", self.demo_lora_merge),
            ("多适配器合并", self.demo_multi_adapter_merge),
            ("Benchmark评估", self.demo_benchmark_evaluation),
            ("单任务评估", self.demo_single_task_evaluation),
            ("基线比较", self.demo_baseline_comparison)
        ]
        
        for demo_name, demo_func in demos:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"运行演示: {demo_name}")
                logger.info(f"{'='*50}")
                demo_func()
                logger.info(f"演示 '{demo_name}' 完成")
            except Exception as e:
                logger.error(f"演示 '{demo_name}' 失败: {e}")
        
        logger.info("\n所有演示运行完成！")
        logger.info(f"输出目录: {self.output_dir}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型合并和Benchmark评估演示")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./demo_outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--demo", 
        type=str,
        choices=["lora", "multi", "benchmark", "single", "baseline", "all"],
        default="all",
        help="要运行的演示类型"
    )
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = ModelMergeAndBenchmarkDemo(args.output_dir)
    
    # 运行指定演示
    if args.demo == "lora":
        demo.demo_lora_merge()
    elif args.demo == "multi":
        demo.demo_multi_adapter_merge()
    elif args.demo == "benchmark":
        demo.demo_benchmark_evaluation()
    elif args.demo == "single":
        demo.demo_single_task_evaluation()
    elif args.demo == "baseline":
        demo.demo_baseline_comparison()
    else:
        demo.run_all_demos()

if __name__ == "__main__":
    main()