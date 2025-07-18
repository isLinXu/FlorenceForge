"""FlorenceForge多任务评估器模块

提供完整的多任务模型评估功能
"""

import json
import time
import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.model import Florence2MultiTaskModel
from ..core.tasks import FLORENCE2_TASKS
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader

logger = logging.getLogger(__name__)

class MultiTaskEvaluator:
    """多任务评估器
    
    提供完整的多任务模型评估功能
    """
    
    def __init__(
        self,
        model: Florence2MultiTaskModel,
        device: Optional[torch.device] = None
    ):
        """初始化评估器
        
        Args:
            model: Florence2多任务模型
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cpu')  # 强制使用CPU
        
        # 将模型移到指定设备
        self.model.to(self.device)
        
        # 评估结果存储
        self.evaluation_results = {}
        self.task_metrics = {}
        
        logger.info(f"多任务评估器初始化完成，使用设备: {self.device}")
    
    def evaluate_dataset(
        self,
        dataset: MultiTaskDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples_per_task: Optional[int] = None,
        save_predictions: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """评估整个数据集
        
        Args:
            dataset: 评估数据集
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            max_samples_per_task: 每个任务的最大样本数
            save_predictions: 是否保存预测结果
            output_dir: 输出目录
            
        Returns:
            评估结果字典
        """
        logger.info("开始数据集评估...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn
        )
        
        # 按任务组织指标计算器
        task_calculators = {}
        for task_type in dataset.task_indices.keys():
            task_calculators[task_type] = get_metric_calculator(task_type)
        
        # 存储预测结果
        all_predictions = defaultdict(list)
        all_references = defaultdict(list)
        
        # 评估统计
        total_samples = 0
        task_sample_counts = defaultdict(int)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 获取批次中的任务类型
                batch_task_types = batch.get('task_types', [])
                
                # 生成预测
                predictions = self.model.generate(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['pixel_values'],
                    attention_mask=batch.get('attention_mask'),
                    max_new_tokens=512,
                    do_sample=False
                )
                
                # 解码预测结果
                decoded_predictions = self.model.processor.batch_decode(
                    predictions, skip_special_tokens=True
                )
                
                # 解码参考答案
                reference_ids = batch.get('labels', batch['input_ids'])
                decoded_references = self.model.processor.batch_decode(
                    reference_ids, skip_special_tokens=True
                )
                
                # 按任务类型组织结果
                for i, (pred, ref, task_type) in enumerate(
                    zip(decoded_predictions, decoded_references, batch_task_types)
                ):
                    # 检查样本数限制
                    if (max_samples_per_task and 
                        task_sample_counts[task_type] >= max_samples_per_task):
                        continue
                    
                    # 清理预测和参考文本
                    pred_clean = self._clean_prediction(pred, task_type)
                    ref_clean = self._clean_reference(ref, task_type)
                    
                    # 添加到指标计算器
                    if task_type in task_calculators:
                        task_calculators[task_type].add_batch([pred_clean], [ref_clean])
                    
                    # 存储预测结果
                    if save_predictions:
                        all_predictions[task_type].append({
                            'prediction': pred_clean,
                            'reference': ref_clean,
                            'sample_id': total_samples
                        })
                    
                    task_sample_counts[task_type] += 1
                    total_samples += 1
        
        # 计算各任务指标
        task_metrics = {}
        for task_type, calculator in task_calculators.items():
            if task_sample_counts[task_type] > 0:
                metrics = calculator.compute()
                task_metrics[task_type] = {
                    'metrics': metrics,
                    'sample_count': task_sample_counts[task_type]
                }
        
        # 计算总体指标
        overall_metrics = self._compute_overall_metrics(task_metrics)
        
        # 构建评估结果
        evaluation_time = time.time() - start_time
        
        results = {
            'overall_metrics': overall_metrics,
            'task_metrics': task_metrics,
            'evaluation_info': {
                'total_samples': total_samples,
                'task_sample_counts': dict(task_sample_counts),
                'evaluation_time': evaluation_time,
                'device': str(self.device)
            }
        }
        
        # 保存结果
        if output_dir:
            self._save_evaluation_results(results, all_predictions, output_dir)
        
        # 存储到实例变量
        self.evaluation_results = results
        self.task_metrics = task_metrics
        
        logger.info(f"数据集评估完成，耗时: {evaluation_time:.2f}秒")
        
        return results
    
    def evaluate_task(
        self,
        dataset: MultiTaskDataset,
        task_type: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """评估特定任务
        
        Args:
            dataset: 评估数据集
            task_type: 任务类型
            batch_size: 批次大小
            max_samples: 最大样本数
            
        Returns:
            任务评估结果
        """
        logger.info(f"开始评估任务: {task_type}")
        
        # 创建任务特定的子集
        task_subset = dataset.create_task_subset(task_type, max_samples)
        
        if len(task_subset) == 0:
            logger.warning(f"任务 {task_type} 没有可用样本")
            return {}
        
        # 创建数据加载器
        dataloader = DataLoader(
            task_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        # 创建指标计算器
        calculator = get_metric_calculator(task_type)
        
        # 设置模型为评估模式
        self.model.eval()
        
        predictions = []
        references = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估 {task_type}"):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 生成预测
                pred_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['pixel_values'],
                    attention_mask=batch.get('attention_mask'),
                    max_new_tokens=512,
                    do_sample=False
                )
                
                # 解码结果
                batch_predictions = self.model.processor.batch_decode(
                    pred_ids, skip_special_tokens=True
                )
                
                reference_ids = batch.get('labels', batch['input_ids'])
                batch_references = self.model.processor.batch_decode(
                    reference_ids, skip_special_tokens=True
                )
                
                # 清理和收集结果
                for pred, ref in zip(batch_predictions, batch_references):
                    pred_clean = self._clean_prediction(pred, task_type)
                    ref_clean = self._clean_reference(ref, task_type)
                    
                    predictions.append(pred_clean)
                    references.append(ref_clean)
                    
                    calculator.add_batch([pred_clean], [ref_clean])
        
        # 计算指标
        metrics = calculator.compute()
        evaluation_time = time.time() - start_time
        
        results = {
            'task_type': task_type,
            'metrics': metrics,
            'sample_count': len(predictions),
            'evaluation_time': evaluation_time,
            'predictions': predictions[:10],  # 保存前10个预测示例
            'references': references[:10]
        }
        
        logger.info(f"任务 {task_type} 评估完成，耗时: {evaluation_time:.2f}秒")
        
        return results
    
    def evaluate_single_sample(
        self,
        image: Union[str, Path, 'PIL.Image.Image'],
        task_type: str,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """评估单个样本
        
        Args:
            image: 输入图像
            task_type: 任务类型
            reference: 参考答案（可选）
            
        Returns:
            评估结果
        """
        # 使用模型进行预测
        prediction = self.model.predict_task(image, task_type)
        
        result = {
            'task_type': task_type,
            'prediction': prediction,
            'reference': reference
        }
        
        # 如果有参考答案，计算指标
        if reference:
            calculator = get_metric_calculator(task_type)
            
            pred_clean = self._clean_prediction(prediction, task_type)
            ref_clean = self._clean_reference(reference, task_type)
            
            calculator.add_batch([pred_clean], [ref_clean])
            metrics = calculator.compute()
            
            result['metrics'] = metrics
        
        return result
    
    def _clean_prediction(self, prediction: str, task_type: str) -> str:
        """清理预测结果
        
        Args:
            prediction: 原始预测结果
            task_type: 任务类型
            
        Returns:
            清理后的预测结果
        """
        # 移除特殊标记
        cleaned = prediction.strip()
        
        # 根据任务类型进行特定清理
        if task_type in FLORENCE2_TASKS:
            task_config = FLORENCE2_TASKS[task_type]
            
            # 移除任务前缀
            if 'prefix' in task_config and task_config['prefix']:
                prefix = task_config['prefix']
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def _clean_reference(self, reference: str, task_type: str) -> str:
        """清理参考答案
        
        Args:
            reference: 原始参考答案
            task_type: 任务类型
            
        Returns:
            清理后的参考答案
        """
        # 基本清理
        cleaned = reference.strip()
        
        # 根据任务类型进行特定清理
        if task_type in FLORENCE2_TASKS:
            task_config = FLORENCE2_TASKS[task_type]
            
            # 移除任务前缀
            if 'prefix' in task_config and task_config['prefix']:
                prefix = task_config['prefix']
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def _compute_overall_metrics(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """计算总体指标
        
        Args:
            task_metrics: 各任务指标
            
        Returns:
            总体指标字典
        """
        overall_metrics = {}
        
        # 收集所有指标名称
        all_metric_names = set()
        for task_data in task_metrics.values():
            all_metric_names.update(task_data['metrics'].keys())
        
        # 计算每个指标的加权平均
        total_samples = sum(task_data['sample_count'] for task_data in task_metrics.values())
        
        for metric_name in all_metric_names:
            weighted_sum = 0.0
            valid_tasks = 0
            
            for task_type, task_data in task_metrics.items():
                if metric_name in task_data['metrics']:
                    metric_value = task_data['metrics'][metric_name]
                    sample_count = task_data['sample_count']
                    
                    if isinstance(metric_value, (int, float)):
                        weight = sample_count / total_samples
                        weighted_sum += metric_value * weight
                        valid_tasks += 1
            
            if valid_tasks > 0:
                overall_metrics[f'avg_{metric_name}'] = weighted_sum
        
        # 添加任务数量统计
        overall_metrics['num_tasks'] = len(task_metrics)
        overall_metrics['total_samples'] = total_samples
        
        return overall_metrics
    
    def _save_evaluation_results(
        self,
        results: Dict[str, Any],
        predictions: Dict[str, List[Dict]],
        output_dir: Union[str, Path]
    ) -> None:
        """保存评估结果
        
        Args:
            results: 评估结果
            predictions: 预测结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存评估指标
        with open(output_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存预测结果
        if predictions:
            for task_type, task_predictions in predictions.items():
                task_file = output_dir / f'predictions_{task_type}.json'
                with open(task_file, 'w', encoding='utf-8') as f:
                    json.dump(task_predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {output_dir}")
    
    def get_task_performance_summary(self) -> Dict[str, Any]:
        """获取任务性能摘要
        
        Returns:
            任务性能摘要
        """
        if not self.task_metrics:
            return {}
        
        summary = {}
        
        for task_type, task_data in self.task_metrics.items():
            metrics = task_data['metrics']
            sample_count = task_data['sample_count']
            
            # 选择关键指标
            key_metrics = {}
            
            # 通用指标
            for metric_name in ['accuracy', 'f1', 'bleu', 'rouge1_f1', 'precision', 'recall']:
                if metric_name in metrics:
                    key_metrics[metric_name] = metrics[metric_name]
            
            # 任务特定指标
            if 'caption' in task_type.lower():
                for metric_name in ['word_overlap', 'length_ratio']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'detection' in task_type.lower():
                for metric_name in ['mAP', 'true_positives', 'false_positives']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'ocr' in task_type.lower():
                for metric_name in ['character_accuracy', 'word_accuracy', 'edit_distance']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'segmentation' in task_type.lower():
                for metric_name in ['mean_iou', 'mean_dice']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            summary[task_type] = {
                'key_metrics': key_metrics,
                'sample_count': sample_count,
                'all_metrics': metrics
            }
        
        return summary
    
    def compare_with_baseline(
        self,
        baseline_results: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """与基线结果比较
        
        Args:
            baseline_results: 基线评估结果
            output_file: 输出文件路径
            
        Returns:
            比较结果
        """
        if not self.evaluation_results:
            raise ValueError("请先运行评估")
        
        comparison = {
            'current_results': self.evaluation_results,
            'baseline_results': baseline_results,
            'improvements': {},
            'regressions': {}
        }
        
        # 比较总体指标
        current_overall = self.evaluation_results.get('overall_metrics', {})
        baseline_overall = baseline_results.get('overall_metrics', {})
        
        for metric_name in current_overall:
            if metric_name in baseline_overall:
                current_value = current_overall[metric_name]
                baseline_value = baseline_overall[metric_name]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    improvement = current_value - baseline_value
                    relative_improvement = improvement / baseline_value if baseline_value != 0 else 0
                    
                    comparison['improvements'][metric_name] = {
                        'absolute': improvement,
                        'relative': relative_improvement,
                        'current': current_value,
                        'baseline': baseline_value
                    }
        
        # 比较任务级指标
        current_tasks = self.evaluation_results.get('task_metrics', {})
        baseline_tasks = baseline_results.get('task_metrics', {})
        
        task_comparisons = {}
        for task_type in current_tasks:
            if task_type in baseline_tasks:
                task_comparisons[task_type] = self._compare_task_metrics(
                    current_tasks[task_type]['metrics'],
                    baseline_tasks[task_type]['metrics']
                )
        
        comparison['task_comparisons'] = task_comparisons
        
        # 保存比较结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
        
        return comparison
    
    def _compare_task_metrics(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """比较任务指标"""
        comparison = {}
        
        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    improvement = current_value - baseline_value
                    relative_improvement = improvement / baseline_value if baseline_value != 0 else 0
                    
                    comparison[metric_name] = {
                        'absolute': improvement,
                        'relative': relative_improvement,
                        'current': current_value,
                        'baseline': baseline_value
                    }
        
        return comparison