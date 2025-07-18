"""FlorenceForge结果分析器模块

提供评估结果的深度分析和可视化功能
"""

import json
import logging

try:
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn未安装，可视化功能将不可用")

logger = logging.getLogger(__name__)

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
        
        # 设置绘图样式
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")
    
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
        """绘制任务性能图表
        
        Args:
            output_dir: 输出目录
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("绘图库不可用，跳过可视化")
            return
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        if not task_metrics:
            logger.warning("没有任务指标数据")
            return
        
        # 准备数据
        task_names = []
        performance_scores = []
        sample_counts = []
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            
            task_names.append(task_type)
            performance_scores.append(score)
            sample_counts.append(task_data['sample_count'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('任务性能分析', fontsize=16)
        
        # 1. 性能分数条形图
        axes[0, 0].bar(range(len(task_names)), performance_scores)
        axes[0, 0].set_title('任务性能分数')
        axes[0, 0].set_xlabel('任务')
        axes[0, 0].set_ylabel('性能分数')
        axes[0, 0].set_xticks(range(len(task_names)))
        axes[0, 0].set_xticklabels(task_names, rotation=45, ha='right')
        
        # 2. 样本数量分布
        axes[0, 1].bar(range(len(task_names)), sample_counts, color='orange')
        axes[0, 1].set_title('样本数量分布')
        axes[0, 1].set_xlabel('任务')
        axes[0, 1].set_ylabel('样本数量')
        axes[0, 1].set_xticks(range(len(task_names)))
        axes[0, 1].set_xticklabels(task_names, rotation=45, ha='right')
        
        # 3. 性能vs样本数量散点图
        axes[1, 0].scatter(sample_counts, performance_scores, alpha=0.7)
        axes[1, 0].set_title('性能 vs 样本数量')
        axes[1, 0].set_xlabel('样本数量')
        axes[1, 0].set_ylabel('性能分数')
        
        # 添加任务标签
        for i, task in enumerate(task_names):
            axes[1, 0].annotate(task, (sample_counts[i], performance_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 性能分数分布直方图
        axes[1, 1].hist(performance_scores, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_title('性能分数分布')
        axes[1, 1].set_xlabel('性能分数')
        axes[1, 1].set_ylabel('任务数量')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_dir / 'task_performance.png', dpi=300, bbox_inches='tight')
            logger.info(f"性能图表已保存到: {output_dir / 'task_performance.png'}")
        
        plt.show()
    
    def plot_metric_comparison(
        self,
        metric_names: List[str],
        output_dir: Optional[Union[str,
        Path]] = None
    ) -> None:
        """绘制指标比较图
        
        Args:
            metric_names: 要比较的指标名称列表
            output_dir: 输出目录
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("绘图库不可用，跳过可视化")
            return
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        if not task_metrics:
            logger.warning("没有任务指标数据")
            return
        
        # 准备数据
        data = []
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            
            for metric_name in metric_names:
                if metric_name in metrics:
                    data.append({
                        'task': task_type,
                        'metric': metric_name,
                        'value': metrics[metric_name]
                    })
        
        if not data:
            logger.warning("没有找到指定的指标数据")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 使用seaborn绘制分组条形图
        sns.barplot(data=df, x='task', y='value', hue='metric')
        
        plt.title('任务指标比较')
        plt.xlabel('任务')
        plt.ylabel('指标值')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='指标')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"指标比较图已保存到: {output_dir / 'metric_comparison.png'}")
        
        plt.show()
    
    def _rank_tasks_by_performance(self, task_metrics: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """按性能对任务排名"""
        task_scores = []
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            task_scores.append((task_type, score))
        
        # 按分数降序排列
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores
    
    def _analyze_metric_distribution(
        self,
        task_metrics: Dict[str,
        Dict]
    ) -> Dict[str, Dict[str, float]]:
        """分析指标分布"""
        metric_values = defaultdict(list)
        
        # 收集所有指标值
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)
        
        # 计算统计信息
        distribution = {}
        for metric_name, values in metric_values.items():
            if values:
                distribution[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return distribution
    
    def _categorize_task_performance(self, task_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
        """将任务按性能分类"""
        categories = {
            '优秀': [],
            '良好': [],
            '一般': [],
            '需要改进': []
        }
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            
            if score >= 0.8:
                categories['优秀'].append(task_type)
            elif score >= 0.6:
                categories['良好'].append(task_type)
            elif score >= 0.4:
                categories['一般'].append(task_type)
            else:
                categories['需要改进'].append(task_type)
        
        return categories
    
    def _analyze_metric_correlations(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """分析指标相关性"""
        # 收集指标数据
        metric_data = defaultdict(list)
        
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_data[metric_name].append(value)
        
        # 计算相关性（简化版本）
        correlations = {}
        metric_names = list(metric_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                if len(metric_data[metric1]) == len(metric_data[metric2]):
                    corr = np.corrcoef(metric_data[metric1], metric_data[metric2])[0, 1]
                    correlations[f'{metric1}_vs_{metric2}'] = corr
        
        return correlations
    
    def _analyze_task_errors(self, predictions: List[Dict], task_type: str) -> Dict[str, Any]:
        """分析任务特定错误"""
        errors = {
            'total_samples': len(predictions),
            'error_types': defaultdict(int),
            'common_mistakes': []
        }
        
        for pred_data in predictions:
            prediction = pred_data.get('prediction', '')
            reference = pred_data.get('reference', '')
            
            # 简单的错误分析
            if prediction != reference:
                # 长度差异
                if len(prediction) < len(reference) * 0.5:
                    errors['error_types']['too_short'] += 1
                elif len(prediction) > len(reference) * 1.5:
                    errors['error_types']['too_long'] += 1
                
                # 空预测
                if not prediction.strip():
                    errors['error_types']['empty_prediction'] += 1
                
                # 格式错误（针对特定任务）
                if task_type in ['object_detection', 'phrase_grounding']:
                    if '<loc_' not in prediction:
                        errors['error_types']['missing_location'] += 1
        
        return errors
    
    def _calculate_difficulty_score(self, metrics: Dict[str, float]) -> float:
        """计算难度分数（0-1，1表示最难）"""
        # 基于性能指标推断难度
        performance_score = self._calculate_performance_score(metrics)
        
        # 难度与性能成反比
        difficulty_score = 1.0 - performance_score
        
        return max(0.0, min(1.0, difficulty_score))
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算综合性能分数"""
        # 定义关键指标权重
        key_metrics = {
            'accuracy': 0.3,
            'f1': 0.25,
            'bleu': 0.2,
            'precision': 0.15,
            'recall': 0.15,
            'rouge1_f1': 0.2,
            'mean_iou': 0.25,
            'character_accuracy': 0.3,
            'word_accuracy': 0.25
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in key_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    weighted_score += value * weight
                    total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            # 如果没有关键指标，使用所有可用指标的平均值
            valid_values = []
            for value in metrics.values():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    valid_values.append(value)
            
            return np.mean(valid_values) if valid_values else 0.0
    
    def _extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """提取关键指标"""
        key_metric_names = [
            'accuracy', 'f1', 'bleu', 'precision', 'recall',
            'rouge1_f1', 'mean_iou', 'character_accuracy', 'word_accuracy'
        ]
        
        key_metrics = {}
        for metric_name in key_metric_names:
            if metric_name in metrics:
                key_metrics[metric_name] = metrics[metric_name]
        
        return key_metrics
    
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