"""FlorenceForge结果分析器模块

提供评估结果的深度分析和可视化功能
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from . import analyzer_deps
from . import analyzer_diagnostics
from . import analyzer_plotting
from . import analyzer_scoring

logger = logging.getLogger(__name__)

# 向后兼容：历史测试/脚本可能 patch 或引用这些名称
_load_plotting_dependencies = analyzer_deps.load_plotting_dependencies
_load_clustering_dependencies = analyzer_deps.load_clustering_dependencies
_load_psutil = analyzer_deps.load_psutil

class ResultAnalyzer:
    """结果分析器
    
    提供评估结果的深度分析和可视化功能
    """
    
    def __init__(self, evaluation_results: Optional[Dict[str, Any]] = None):
        """初始化结果分析器
        
        Args:
            evaluation_results: 评估结果字典
        """
        self.evaluation_results = evaluation_results or {}
        self.analysis_cache = {}
    
    def load_results(self, results_path: Union[str, Path]) -> None:
        """从文件加载评估结果
        
        Args:
            results_path: 结果文件路径
        """
        results_path = Path(results_path)
        
        if results_path.suffix == '.json':
            with open(results_path, 'r', encoding='utf-8') as f:
                self.evaluation_results = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {results_path.suffix}")
        
        # 清空缓存
        self.analysis_cache = {}
        
        logger.info(f"评估结果已从 {results_path} 加载")
    
    def analyze_task_performance(self) -> Dict[str, Any]:
        """分析任务性能
        
        Returns:
            任务性能分析结果
        """
        if 'task_performance' in self.analysis_cache:
            return self.analysis_cache['task_performance']
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        if not task_metrics:
            return {}
        
        analysis = {
            'task_ranking': self._rank_tasks_by_performance(task_metrics),
            'metric_distribution': self._analyze_metric_distribution(task_metrics),
            'performance_categories': self._categorize_task_performance(task_metrics),
            'correlation_analysis': self._analyze_metric_correlations(task_metrics)
        }
        
        self.analysis_cache['task_performance'] = analysis
        return analysis
    
    def analyze_error_patterns(
        self,
        predictions_dir: Optional[Union[str,
        Path]] = None
    ) -> Dict[str, Any]:
        """分析错误模式
        
        Args:
            predictions_dir: 预测结果目录
            
        Returns:
            错误模式分析结果
        """
        if 'error_patterns' in self.analysis_cache:
            return self.analysis_cache['error_patterns']
        
        analysis = {
            'common_errors': {},
            'error_frequency': {},
            'task_specific_errors': {}
        }
        
        # 如果提供了预测结果目录，分析具体错误
        if predictions_dir:
            predictions_dir = Path(predictions_dir)
            
            for pred_file in predictions_dir.glob('predictions_*.json'):
                task_type = pred_file.stem.replace('predictions_', '')
                
                with open(pred_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                
                task_errors = self._analyze_task_errors(predictions, task_type)
                analysis['task_specific_errors'][task_type] = task_errors
        
        self.analysis_cache['error_patterns'] = analysis
        return analysis
    
    def analyze_sample_difficulty(self) -> Dict[str, Any]:
        """分析样本难度
        
        Returns:
            样本难度分析结果
        """
        if 'sample_difficulty' in self.analysis_cache:
            return self.analysis_cache['sample_difficulty']
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        analysis = {
            'difficulty_distribution': {},
            'challenging_tasks': [],
            'easy_tasks': []
        }
        
        # 基于性能指标推断难度
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            sample_count = task_data['sample_count']
            
            # 计算综合难度分数
            difficulty_score = self._calculate_difficulty_score(metrics)
            
            analysis['difficulty_distribution'][task_type] = {
                'difficulty_score': difficulty_score,
                'sample_count': sample_count,
                'key_metrics': self._extract_key_metrics(metrics)
            }
            
            # 分类任务难度
            if difficulty_score > 0.7:
                analysis['challenging_tasks'].append(task_type)
            elif difficulty_score < 0.3:
                analysis['easy_tasks'].append(task_type)
        
        self.analysis_cache['sample_difficulty'] = analysis
        return analysis
    
    def generate_performance_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """生成性能报告
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            报告内容
        """
        report_lines = []
        
        # 报告标题
        report_lines.append("# Florence2 多任务模型性能报告\n")
        
        # 总体性能
        overall_metrics = self.evaluation_results.get('overall_metrics', {})
        if overall_metrics:
            report_lines.append("## 总体性能")
            report_lines.append("")
            
            for metric_name, value in overall_metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"- **{metric_name}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric_name}**: {value}")
            
            report_lines.append("")
        
        # 任务性能分析
        task_analysis = self.analyze_task_performance()
        if task_analysis:
            report_lines.append("## 任务性能分析")
            report_lines.append("")
            
            # 任务排名
            if 'task_ranking' in task_analysis:
                report_lines.append("### 任务性能排名")
                report_lines.append("")
                
                for i, (task_type, score) in enumerate(task_analysis['task_ranking'][:10], 1):
                    report_lines.append(f"{i}. **{task_type}**: {score:.4f}")
                
                report_lines.append("")
            
            # 性能分类
            if 'performance_categories' in task_analysis:
                categories = task_analysis['performance_categories']
                
                report_lines.append("### 性能分类")
                report_lines.append("")
                
                for category, tasks in categories.items():
                    if tasks:
                        report_lines.append(f"**{category}**: {', '.join(tasks)}")
                
                report_lines.append("")
        
        # 样本难度分析
        difficulty_analysis = self.analyze_sample_difficulty()
        if difficulty_analysis:
            report_lines.append("## 样本难度分析")
            report_lines.append("")
            
            if difficulty_analysis['challenging_tasks']:
                report_lines.append(f"**挑战性任务**: {', '.join(difficulty_analysis['challenging_tasks'])}")
            
            if difficulty_analysis['easy_tasks']:
                report_lines.append(f"**简单任务**: {', '.join(difficulty_analysis['easy_tasks'])}")
            
            report_lines.append("")
        
        # 评估信息
        eval_info = self.evaluation_results.get('evaluation_info', {})
        if eval_info:
            report_lines.append("## 评估信息")
            report_lines.append("")
            
            for key, value in eval_info.items():
                report_lines.append(f"- **{key}**: {value}")
            
            report_lines.append("")
        
        # 建议和改进方向
        recommendations = self._generate_recommendations()
        if recommendations:
            report_lines.append("## 建议和改进方向")
            report_lines.append("")
            
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"性能报告已保存到: {output_path}")
        
        return report_content
    
    def plot_task_performance(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        analyzer_plotting.plot_task_performance(self.evaluation_results, output_dir)

    def plot_metric_comparison(
        self,
        metric_names: List[str],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        analyzer_plotting.plot_metric_comparison(
            self.evaluation_results, metric_names, output_dir
        )

    def _rank_tasks_by_performance(self, task_metrics: Dict[str, Dict]) -> List[Tuple[str, float]]:
        return analyzer_scoring.rank_tasks_by_performance(task_metrics)

    def _analyze_metric_distribution(self, task_metrics: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        return analyzer_scoring.analyze_metric_distribution(task_metrics)

    def _categorize_task_performance(self, task_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
        return analyzer_scoring.categorize_task_performance(task_metrics)

    def _analyze_metric_correlations(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        return analyzer_scoring.analyze_metric_correlations(task_metrics)

    def _analyze_task_errors(self, predictions: List[Dict], task_type: str) -> Dict[str, Any]:
        return analyzer_scoring.analyze_task_errors(predictions, task_type)

    def _calculate_difficulty_score(self, metrics: Dict[str, float]) -> float:
        return analyzer_scoring.calculate_difficulty_score(metrics)

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        return analyzer_scoring.calculate_performance_score(metrics)

    def _extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return analyzer_scoring.extract_key_metrics(metrics)
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于任务性能分析生成建议
        task_analysis = self.analyze_task_performance()
        
        if 'performance_categories' in task_analysis:
            categories = task_analysis['performance_categories']
            
            if categories.get('需要改进'):
                recommendations.append(
                    f"重点关注以下需要改进的任务: {', '.join(categories['需要改进'][:3])}"
                )
            
            if len(categories.get('优秀', [])) < len(categories.get('需要改进', [])):
                recommendations.append("考虑增加训练数据或调整模型架构以提升整体性能")
        
        # 基于样本难度分析生成建议
        difficulty_analysis = self.analyze_sample_difficulty()
        
        if difficulty_analysis.get('challenging_tasks'):
            recommendations.append(
                f"对于挑战性任务 {', '.join(difficulty_analysis['challenging_tasks'][:2])}，"
                "建议增加训练时间或使用更复杂的模型结构"
            )
        
        # 基于评估信息生成建议
        eval_info = self.evaluation_results.get('evaluation_info', {})
        task_counts = eval_info.get('task_sample_counts', {})
        
        if task_counts:
            min_samples = min(task_counts.values())
            max_samples = max(task_counts.values())
            
            if max_samples > min_samples * 3:
                recommendations.append("数据分布不均衡，建议平衡各任务的训练样本数量")
        
        return recommendations
    
    def cluster_error_patterns(
        self, clustering_method: str = "kmeans", n_clusters: int = 5
    ) -> Dict[str, Any]:
        return analyzer_diagnostics.cluster_error_patterns(
            self.evaluation_results, clustering_method, n_clusters
        )

    def diagnose_performance_bottlenecks(self) -> Dict[str, Any]:
        return analyzer_diagnostics.diagnose_performance_bottlenecks(
            self.evaluation_results
        )

    def assess_data_quality(self) -> Dict[str, Any]:
        return analyzer_diagnostics.assess_data_quality(self.evaluation_results)

    def analyze_model_behavior(self) -> Dict[str, Any]:
        return analyzer_diagnostics.analyze_model_behavior(self.evaluation_results)
