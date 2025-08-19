"""FlorenceForge Benchmark评估模块

提供标准化的benchmark指标计算和评估功能
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.model import Florence2MultiTaskModel
from ..data.dataset import MultiTaskDataset
from .evaluator import MultiTaskEvaluator
from .metrics import (
    CaptionMetrics, DetectionMetrics, OCRMetrics, 
    SegmentationMetrics, get_metric_calculator
)

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    """Benchmark评估器
    
    提供标准化的benchmark评估功能，支持多种评估协议和指标计算
    """
    
    def __init__(
        self,
        model: Florence2MultiTaskModel,
        device: Optional[torch.device] = None,
        benchmark_config: Optional[Dict[str, Any]] = None
    ):
        """初始化Benchmark评估器
        
        Args:
            model: Florence2多任务模型
            device: 计算设备
            benchmark_config: Benchmark配置
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # 默认benchmark配置
        self.benchmark_config = benchmark_config or self._get_default_config()
        
        # 初始化评估器
        self.evaluator = MultiTaskEvaluator(model, device)
        
        # 结果存储
        self.benchmark_results = {}
        self.detailed_results = {}
        
        logger.info(f"Benchmark评估器初始化完成，设备: {self.device}")
    
    def run_benchmark(
        self,
        datasets: Dict[str, MultiTaskDataset],
        output_dir: Union[str, Path],
        save_detailed: bool = True,
        compare_baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行完整的benchmark评估
        
        Args:
            datasets: 评估数据集字典 {dataset_name: dataset}
            output_dir: 输出目录
            save_detailed: 是否保存详细结果
            compare_baseline: 基线结果用于比较
            
        Returns:
            Benchmark评估结果
        """
        logger.info(f"开始运行benchmark评估，数据集: {list(datasets.keys())}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        benchmark_results = {
            'benchmark_info': {
                'model_name': self.model.model_name,
                'device': str(self.device),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.benchmark_config
            },
            'dataset_results': {},
            'overall_summary': {},
            'task_performance': {}
        }
        
        total_start_time = time.time()
        
        # 评估每个数据集
        for dataset_name, dataset in datasets.items():
            logger.info(f"评估数据集: {dataset_name}")
            
            dataset_result = self._evaluate_dataset(
                dataset, dataset_name, output_path / dataset_name
            )
            
            benchmark_results['dataset_results'][dataset_name] = dataset_result
        
        # 计算总体摘要
        benchmark_results['overall_summary'] = self._compute_overall_summary(
            benchmark_results['dataset_results']
        )
        
        # 计算任务性能统计
        benchmark_results['task_performance'] = self._compute_task_performance(
            benchmark_results['dataset_results']
        )
        
        # 与基线比较
        if compare_baseline:
            benchmark_results['baseline_comparison'] = self._compare_with_baseline(
                benchmark_results, compare_baseline
            )
        
        # 计算总评估时间
        total_time = time.time() - total_start_time
        benchmark_results['benchmark_info']['total_evaluation_time'] = total_time
        
        # 保存结果
        self._save_benchmark_results(
            benchmark_results, output_path, save_detailed
        )
        
        self.benchmark_results = benchmark_results
        
        logger.info(f"Benchmark评估完成，总耗时: {total_time:.2f}秒")
        
        return benchmark_results
    
    def evaluate_single_task(
        self,
        dataset: MultiTaskDataset,
        task_type: str,
        output_dir: Optional[Union[str, Path]] = None,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """评估单个任务
        
        Args:
            dataset: 评估数据集
            task_type: 任务类型
            output_dir: 输出目录
            detailed_analysis: 是否进行详细分析
            
        Returns:
            任务评估结果
        """
        logger.info(f"开始评估任务: {task_type}")
        
        # 筛选任务数据
        task_dataset = self._filter_task_dataset(dataset, task_type)
        
        if len(task_dataset) == 0:
            logger.warning(f"数据集中没有任务 {task_type} 的数据")
            return {}
        
        # 运行评估
        task_result = self.evaluator.evaluate_task(
            task_dataset, task_type,
            batch_size=self.benchmark_config.get('batch_size', 8),
            max_samples=self.benchmark_config.get('max_samples_per_task')
        )
        
        # 详细分析
        if detailed_analysis:
            task_result['detailed_analysis'] = self._analyze_task_performance(
                task_result, task_type
            )
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / f"{task_type}_results.json", 'w', encoding='utf-8') as f:
                json.dump(task_result, f, indent=2, ensure_ascii=False)
        
        return task_result
    
    def compute_standard_metrics(
        self,
        predictions: List[Any],
        references: List[Any],
        task_type: str,
        metric_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """计算标准指标
        
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            task_type: 任务类型
            metric_config: 指标配置
            
        Returns:
            标准指标字典
        """
        if len(predictions) != len(references):
            raise ValueError("预测结果和参考答案数量不匹配")
        
        # 获取指标计算器
        calculator = get_metric_calculator(task_type)
        
        # 添加数据
        calculator.add_batch(predictions, references)
        
        # 计算指标
        metrics = calculator.compute()
        
        # 应用指标配置
        if metric_config:
            metrics = self._apply_metric_config(metrics, metric_config)
        
        return metrics
    
    def generate_benchmark_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "markdown"
    ) -> None:
        """生成benchmark报告
        
        Args:
            results: Benchmark结果
            output_path: 输出路径
            format: 报告格式 ('markdown', 'html', 'json')
        """
        output_file = Path(output_path)
        
        if format == "markdown":
            self._generate_markdown_report(results, output_file)
        elif format == "html":
            self._generate_html_report(results, output_file)
        elif format == "json":
            self._generate_json_report(results, output_file)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
        
        logger.info(f"Benchmark报告已生成: {output_file}")
    
    def _evaluate_dataset(
        self,
        dataset: MultiTaskDataset,
        dataset_name: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """评估单个数据集"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # 运行评估
        eval_results = self.evaluator.evaluate_dataset(
            dataset,
            batch_size=self.benchmark_config.get('batch_size', 8),
            num_workers=self.benchmark_config.get('num_workers', 4),
            max_samples_per_task=self.benchmark_config.get('max_samples_per_task'),
            save_predictions=self.benchmark_config.get('save_predictions', False),
            output_dir=output_dir
        )
        
        evaluation_time = time.time() - start_time
        
        # 添加数据集特定信息
        dataset_result = {
            'dataset_name': dataset_name,
            'evaluation_time': evaluation_time,
            'dataset_info': {
                'total_samples': len(dataset),
                'task_distribution': dataset.get_task_statistics(),
                'data_path': getattr(dataset, 'data_path', 'unknown')
            },
            'metrics': eval_results.get('task_metrics', {}),
            'overall_metrics': eval_results.get('overall_metrics', {})
        }
        
        return dataset_result
    
    def _compute_overall_summary(
        self,
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算总体摘要"""
        summary = {
            'total_datasets': len(dataset_results),
            'total_samples': 0,
            'total_evaluation_time': 0.0,
            'average_metrics': {},
            'best_performance': {},
            'worst_performance': {}
        }
        
        # 收集所有指标
        all_metrics = defaultdict(list)
        
        for dataset_name, result in dataset_results.items():
            summary['total_samples'] += result['dataset_info']['total_samples']
            summary['total_evaluation_time'] += result['evaluation_time']
            
            # 收集指标
            for metric_name, value in result.get('overall_metrics', {}).items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append((dataset_name, value))
        
        # 计算平均指标
        for metric_name, values in all_metrics.items():
            if values:
                metric_values = [v[1] for v in values]
                summary['average_metrics'][metric_name] = np.mean(metric_values)
                
                # 最佳和最差性能
                best_idx = np.argmax(metric_values)
                worst_idx = np.argmin(metric_values)
                
                summary['best_performance'][metric_name] = {
                    'dataset': values[best_idx][0],
                    'value': values[best_idx][1]
                }
                
                summary['worst_performance'][metric_name] = {
                    'dataset': values[worst_idx][0],
                    'value': values[worst_idx][1]
                }
        
        return summary
    
    def _compute_task_performance(
        self,
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算任务性能统计"""
        task_stats = defaultdict(lambda: {
            'datasets': [],
            'total_samples': 0,
            'metrics': defaultdict(list)
        })
        
        # 收集任务统计
        for dataset_name, result in dataset_results.items():
            for task_type, task_data in result.get('metrics', {}).items():
                task_stats[task_type]['datasets'].append(dataset_name)
                task_stats[task_type]['total_samples'] += task_data.get('sample_count', 0)
                
                # 收集指标
                for metric_name, value in task_data.get('metrics', {}).items():
                    if isinstance(value, (int, float)):
                        task_stats[task_type]['metrics'][metric_name].append(value)
        
        # 计算任务平均性能
        task_performance = {}
        for task_type, stats in task_stats.items():
            task_performance[task_type] = {
                'participating_datasets': len(stats['datasets']),
                'total_samples': stats['total_samples'],
                'average_metrics': {}
            }
            
            for metric_name, values in stats['metrics'].items():
                if values:
                    task_performance[task_type]['average_metrics'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        return task_performance
    
    def _compare_with_baseline(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """与基线结果比较"""
        comparison = {
            'overall_improvement': {},
            'task_improvements': {},
            'dataset_improvements': {}
        }
        
        # 总体改进
        current_overall = current_results.get('overall_summary', {}).get('average_metrics', {})
        baseline_overall = baseline_results.get('overall_summary', {}).get('average_metrics', {})
        
        for metric_name in current_overall:
            if metric_name in baseline_overall:
                current_val = current_overall[metric_name]
                baseline_val = baseline_overall[metric_name]
                
                improvement = current_val - baseline_val
                relative_improvement = improvement / baseline_val if baseline_val != 0 else 0
                
                comparison['overall_improvement'][metric_name] = {
                    'absolute': improvement,
                    'relative': relative_improvement,
                    'current': current_val,
                    'baseline': baseline_val
                }
        
        # 任务级别改进
        current_tasks = current_results.get('task_performance', {})
        baseline_tasks = baseline_results.get('task_performance', {})
        
        for task_type in current_tasks:
            if task_type in baseline_tasks:
                comparison['task_improvements'][task_type] = self._compare_task_metrics(
                    current_tasks[task_type].get('average_metrics', {}),
                    baseline_tasks[task_type].get('average_metrics', {})
                )
        
        return comparison
    
    def _compare_task_metrics(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """比较任务指标"""
        comparison = {}
        
        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current_mean = current_metrics[metric_name].get('mean', 0)
                baseline_mean = baseline_metrics[metric_name].get('mean', 0)
                
                improvement = current_mean - baseline_mean
                relative_improvement = improvement / baseline_mean if baseline_mean != 0 else 0
                
                comparison[metric_name] = {
                    'absolute': improvement,
                    'relative': relative_improvement,
                    'current': current_mean,
                    'baseline': baseline_mean
                }
        
        return comparison
    
    def _filter_task_dataset(
        self,
        dataset: MultiTaskDataset,
        task_type: str
    ) -> MultiTaskDataset:
        """筛选特定任务的数据集"""
        # 这里简化实现，实际应该创建新的数据集实例
        # 只包含指定任务的数据
        return dataset  # 临时返回原数据集
    
    def _analyze_task_performance(
        self,
        task_result: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """分析任务性能"""
        analysis = {
            'task_type': task_type,
            'performance_level': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        metrics = task_result.get('metrics', {})
        
        # 简单的性能分析逻辑
        if task_type.lower() in ['caption', 'detailed_caption']:
            bleu_score = metrics.get('bleu_4', 0)
            if bleu_score > 0.3:
                analysis['performance_level'] = 'good'
                analysis['strengths'].append('良好的BLEU分数')
            elif bleu_score > 0.2:
                analysis['performance_level'] = 'fair'
            else:
                analysis['performance_level'] = 'poor'
                analysis['weaknesses'].append('BLEU分数较低')
                analysis['recommendations'].append('考虑增加训练数据或调整模型参数')
        
        elif task_type.lower() in ['detection', 'object_detection']:
            map_score = metrics.get('mAP', 0)
            if map_score > 0.5:
                analysis['performance_level'] = 'good'
                analysis['strengths'].append('良好的mAP分数')
            elif map_score > 0.3:
                analysis['performance_level'] = 'fair'
            else:
                analysis['performance_level'] = 'poor'
                analysis['weaknesses'].append('mAP分数较低')
                analysis['recommendations'].append('考虑调整检测阈值或增强数据')
        
        return analysis
    
    def _apply_metric_config(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """应用指标配置"""
        # 筛选指标
        if 'include_metrics' in config:
            metrics = {
                k: v for k, v in metrics.items() 
                if k in config['include_metrics']
            }
        
        # 排除指标
        if 'exclude_metrics' in config:
            metrics = {
                k: v for k, v in metrics.items() 
                if k not in config['exclude_metrics']
            }
        
        return metrics
    
    def _save_benchmark_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        save_detailed: bool
    ) -> None:
        """保存benchmark结果"""
        # 保存主要结果
        with open(output_dir / 'benchmark_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存摘要
        summary = {
            'benchmark_info': results['benchmark_info'],
            'overall_summary': results['overall_summary'],
            'task_performance': results['task_performance']
        }
        
        with open(output_dir / 'benchmark_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存详细结果
        if save_detailed:
            detailed_dir = output_dir / 'detailed_results'
            detailed_dir.mkdir(exist_ok=True)
            
            for dataset_name, dataset_result in results['dataset_results'].items():
                with open(detailed_dir / f'{dataset_name}_detailed.json', 'w', encoding='utf-8') as f:
                    json.dump(dataset_result, f, indent=2, ensure_ascii=False)
    
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        output_file: Path
    ) -> None:
        """生成Markdown格式报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Florence Forge Benchmark Report\n\n")
            
            # 基本信息
            info = results['benchmark_info']
            f.write(f"**Model:** {info['model_name']}\n")
            f.write(f"**Device:** {info['device']}\n")
            f.write(f"**Timestamp:** {info['timestamp']}\n")
            f.write(f"**Total Time:** {info.get('total_evaluation_time', 0):.2f}s\n\n")
            
            # 总体摘要
            summary = results['overall_summary']
            f.write("## Overall Summary\n\n")
            f.write(f"- **Total Datasets:** {summary['total_datasets']}\n")
            f.write(f"- **Total Samples:** {summary['total_samples']}\n")
            f.write(f"- **Evaluation Time:** {summary['total_evaluation_time']:.2f}s\n\n")
            
            # 平均指标
            if summary.get('average_metrics'):
                f.write("### Average Metrics\n\n")
                for metric, value in summary['average_metrics'].items():
                    f.write(f"- **{metric}:** {value:.4f}\n")
                f.write("\n")
            
            # 任务性能
            task_perf = results['task_performance']
            f.write("## Task Performance\n\n")
            for task_type, perf in task_perf.items():
                f.write(f"### {task_type}\n\n")
                f.write(f"- **Datasets:** {perf['participating_datasets']}\n")
                f.write(f"- **Samples:** {perf['total_samples']}\n")
                
                if perf.get('average_metrics'):
                    f.write("- **Metrics:**\n")
                    for metric, stats in perf['average_metrics'].items():
                        f.write(f"  - {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
    
    def _generate_html_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成HTML格式报告"""
        # 简化的HTML报告生成
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Florence Forge Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; }}
                .task {{ margin: 20px 0; border: 1px solid #ccc; padding: 15px; }}
            </style>
        </head>
        <body>
            <h1>Florence Forge Benchmark Report</h1>
            <p><strong>Model:</strong> {results['benchmark_info']['model_name']}</p>
            <p><strong>Timestamp:</strong> {results['benchmark_info']['timestamp']}</p>
            <!-- 更多内容... -->
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_json_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成JSON格式报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'batch_size': 8,
            'num_workers': 4,
            'max_samples_per_task': None,
            'save_predictions': False,
            'compute_detailed_metrics': True,
            'metric_config': {
                'include_metrics': None,  # None表示包含所有指标
                'exclude_metrics': [],
                'custom_thresholds': {}
            }
        }