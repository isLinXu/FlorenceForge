"""效率评估指标计算器

基于MetricCalculator基类实现的效率评估指标，
包括推理速度、内存使用、GPU使用、模型大小等效率评估功能。
"""

import logging
from typing import Dict, List, Any, Optional, Callable
import time
import numpy as np

from ..metrics import MetricCalculator
from .efficiency_metrics import EfficiencyEvaluator

logger = logging.getLogger(__name__)

class EfficiencyMetricsCalculator(MetricCalculator):
    """效率评估指标计算器

    继承自MetricCalculator基类，专门用于计算效率指标，
    包括推理速度、内存使用、GPU使用、模型大小等效率评估功能。
    """

    def __init__(self, task_type: str = "efficiency", **kwargs):
        """初始化效率指标计算器

        Args:
            task_type: 任务类型，默认为"efficiency"
            **kwargs: 传递给EfficiencyEvaluator的额外参数
        """
        super().__init__(task_type)
        self.efficiency_evaluator = EfficiencyEvaluator(**kwargs)
        self.model: Optional[Any] = None
        self.images: List[Any] = []
        self.inference_times: List[float] = []
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        self.batch_sizes: List[int] = []
        self.start_time: Optional[float] = None
        self.total_samples: int = 0

    def set_model(self, model: Any) -> None:
        """设置要评估的模型

        Args:
            model: 要评估的模型对象
        """
        self.model = model
        self.efficiency_evaluator.model = model

    def add_batch(
        self,
        predictions: List[str],
        references: List[str],
        images: Optional[List[Any]] = None,
        inference_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """添加一批预测和参考数据

        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            images: 图像数据列表
            inference_time: 推理时间（秒）
            batch_size: 批次大小
            **kwargs: 其他参数
        """
        super().add_batch(predictions, references, **kwargs)

        # 记录开始时间
        if self.start_time is None:
            self.start_time = time.time()

        # 存储图像数据
        if images is not None:
            self.images.extend(images)
            current_batch_size = len(images)
        else:
            current_batch_size = len(predictions)
            self.images.extend([None] * current_batch_size)

        # 记录批次信息
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
        else:
            self.batch_sizes.append(current_batch_size)

        # 记录推理时间
        if inference_time is not None:
            self.inference_times.append(inference_time)

        # 更新总样本数
        self.total_samples += current_batch_size

        # 尝试测量内存和GPU使用
        try:
            memory_usage = self.efficiency_evaluator.measure_memory_usage()
            if memory_usage is not None:
                self.memory_usage.append(memory_usage)

            gpu_usage = self.efficiency_evaluator.get_gpu_usage()
            if gpu_usage is not None:
                self.gpu_usage.append(gpu_usage)
        except Exception as e:
            logger.debug(f"无法测量资源使用: {e}")

    def compute(self) -> Dict[str, float]:
        """计算效率评估指标

        Returns:
            包含各种效率指标的字典
        """
        if not self.predictions or not self.references:
            logger.warning("没有足够的数据进行效率指标计算")
            return {}

        metrics = {}

        try:
            # 计算基础指标
            base_metrics = super().compute()
            metrics.update(base_metrics)

            # 计算时间相关指标
            if self.inference_times:
                metrics.update(self._compute_timing_metrics())

            # 计算吞吐量指标
            if self.start_time is not None:
                total_time = time.time() - self.start_time
                if total_time > 0:
                    metrics['throughput_samples_per_second'] = self.total_samples / total_time
                    metrics['total_processing_time'] = total_time

            # 计算内存使用指标
            if self.memory_usage:
                metrics.update(self._compute_memory_metrics())

            # 计算GPU使用指标
            if self.gpu_usage:
                metrics.update(self._compute_gpu_metrics())

            # 计算批次效率指标
            if self.batch_sizes:
                metrics.update(self._compute_batch_metrics())

            # 如果有模型，计算模型相关指标
            if self.model is not None:
                model_metrics = self._compute_model_metrics()
                metrics.update(model_metrics)

            # 计算效率分数
            efficiency_score = self._calculate_efficiency_score(metrics)
            if efficiency_score is not None:
                metrics['efficiency_score'] = efficiency_score

            # 添加统计信息
            metrics.update({
                'total_samples': self.total_samples,
                'num_batches': len(self.batch_sizes),
                'avg_batch_size': np.mean(self.batch_sizes) if self.batch_sizes else 0
            })

        except Exception as e:
            logger.error(f"效率指标计算失败: {e}")
            # 返回基础指标作为后备
            metrics = super().compute()

        return metrics

    def _compute_timing_metrics(self) -> Dict[str, float]:
        """计算时间相关指标

        Returns:
            时间指标字典
        """
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)

        return {
            'avg_inference_time': np.mean(times),
            'median_inference_time': np.median(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'p95_inference_time': np.percentile(times, 95),
            'p99_inference_time': np.percentile(times, 99),
            'total_inference_time': np.sum(times)
        }

    def _compute_memory_metrics(self) -> Dict[str, float]:
        """计算内存使用指标

        Returns:
            内存指标字典
        """
        if not self.memory_usage:
            return {}

        memory = np.array(self.memory_usage)

        return {
            'avg_memory_usage_mb': np.mean(memory),
            'max_memory_usage_mb': np.max(memory),
            'min_memory_usage_mb': np.min(memory),
            'std_memory_usage_mb': np.std(memory),
            'memory_efficiency': self.total_samples / np.mean(memory) if np.mean(memory) > 0 else 0
        }

    def _compute_gpu_metrics(self) -> Dict[str, float]:
        """计算GPU使用指标

        Returns:
            GPU指标字典
        """
        if not self.gpu_usage:
            return {}

        gpu = np.array(self.gpu_usage)

        return {
            'avg_gpu_usage_percent': np.mean(gpu),
            'max_gpu_usage_percent': np.max(gpu),
            'min_gpu_usage_percent': np.min(gpu),
            'std_gpu_usage_percent': np.std(gpu),
            'gpu_utilization_efficiency': np.mean(gpu) / 100.0
        }

    def _compute_batch_metrics(self) -> Dict[str, float]:
        """计算批次效率指标

        Returns:
            批次指标字典
        """
        if not self.batch_sizes or not self.inference_times:
            return {}

        # 计算每个样本的平均处理时间
        per_sample_times = []
        for i, batch_size in enumerate(self.batch_sizes):
            if i < len(self.inference_times) and batch_size > 0:
                per_sample_time = self.inference_times[i] / batch_size
                per_sample_times.append(per_sample_time)

        if not per_sample_times:
            return {}

        per_sample_times = np.array(per_sample_times)

        return {
            'avg_per_sample_time': np.mean(per_sample_times),
            'median_per_sample_time': np.median(per_sample_times),
            'std_per_sample_time': np.std(per_sample_times),
            'samples_per_second': 1.0 / np.mean(per_sample_times) if np.mean(per_sample_times) > 0 else 0
        }

    def _compute_model_metrics(self) -> Dict[str, float]:
        """计算模型相关指标

        Returns:
            模型指标字典
        """
        metrics = {}

        try:
            # 获取模型大小
            model_size = self.efficiency_evaluator.get_model_size()
            if model_size is not None:
                metrics['model_size_mb'] = model_size

            # 获取参数数量
            num_params = self.efficiency_evaluator.count_parameters()
            if num_params is not None:
                metrics['num_parameters'] = num_params
                metrics['num_parameters_millions'] = num_params / 1e6

            # 计算参数效率（每个参数的性能）
            if num_params is not None and self.total_samples > 0:
                if self.start_time is not None:
                    total_time = time.time() - self.start_time
                    if total_time > 0:
                        metrics['parameter_efficiency'] = (self.total_samples / total_time) / num_params

        except Exception as e:
            logger.warning(f"计算模型指标失败: {e}")

        return metrics

    def _calculate_efficiency_score(self, metrics: Dict[str, float]) -> Optional[float]:
        """计算综合效率分数

        Args:
            metrics: 已计算的指标字典

        Returns:
            效率分数（0-1之间）
        """
        try:
            score_components = []

            # 速度分数（基于吞吐量）
            if 'throughput_samples_per_second' in metrics:
                throughput = metrics['throughput_samples_per_second']
                # 归一化到0-1范围，假设100 samples/s为满分
                speed_score = min(throughput / 100.0, 1.0)
                score_components.append(('speed', speed_score, 0.4))

            # 内存效率分数
            if 'memory_efficiency' in metrics:
                memory_eff = metrics['memory_efficiency']
                # 归一化内存效率
                memory_score = min(memory_eff / 10.0, 1.0)  # 假设10 samples/MB为满分
                score_components.append(('memory', memory_score, 0.3))

            # GPU利用率分数
            if 'gpu_utilization_efficiency' in metrics:
                gpu_score = metrics['gpu_utilization_efficiency']
                score_components.append(('gpu', gpu_score, 0.2))

            # 参数效率分数
            if 'parameter_efficiency' in metrics:
                param_eff = metrics['parameter_efficiency']
                # 归一化参数效率
                param_score = min(param_eff * 1e6, 1.0)  # 调整缩放因子
                score_components.append(('parameter', param_score, 0.1))

            if not score_components:
                return None

            # 计算加权平均分数
            total_weight = sum(weight for _, _, weight in score_components)
            if total_weight == 0:
                return None

            weighted_score = sum(score * weight for _, score, weight in score_components) / total_weight

            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            logger.warning(f"计算效率分数失败: {e}")
            return None

    def reset(self) -> None:
        """重置计算器状态"""
        super().reset()
        self.images.clear()
        self.inference_times.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.batch_sizes.clear()
        self.start_time = None
        self.total_samples = 0

    def get_detailed_results(self) -> Dict[str, Any]:
        """获取详细的评估结果

        Returns:
            包含详细评估信息的字典
        """
        if not self.predictions or not self.references:
            return {}

        results = {
            'summary': self.compute(),
            'timing_details': [],
            'resource_usage': {},
            'efficiency_analysis': {}
        }

        try:
            # 添加时间详情
            for i, inference_time in enumerate(self.inference_times):
                batch_size = self.batch_sizes[i] if i < len(self.batch_sizes) else 1
                timing_detail = {
                    'batch_index': i,
                    'inference_time': inference_time,
                    'batch_size': batch_size,
                    'per_sample_time': inference_time / batch_size if batch_size > 0 else inference_time
                }
                results['timing_details'].append(timing_detail)

            # 资源使用详情
            results['resource_usage'] = {
                'memory_usage_history': self.memory_usage,
                'gpu_usage_history': self.gpu_usage,
                'batch_sizes': self.batch_sizes
            }

            # 效率分析
            if self.inference_times:
                times = np.array(self.inference_times)
                results['efficiency_analysis'] = {
                    'timing_stability': {
                        'coefficient_of_variation': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0,
                        'timing_trend': self._analyze_timing_trend(times)
                    },
                    'bottleneck_analysis': self._analyze_bottlenecks(),
                    'optimization_suggestions': self._generate_optimization_suggestions()
                }

        except Exception as e:
            logger.error(f"获取详细结果失败: {e}")

        return results

    def compute_inference_speed(self) -> float:
        """计算推理速度（样本/秒）

        Returns:
            推理速度
        """
        if not self.inference_times or not self.batch_sizes:
            return 0.0

        total_samples = sum(self.batch_sizes)
        total_time = sum(self.inference_times)

        if total_time <= 0:
            return 0.0

        return total_samples / total_time

    def compute_memory_usage(self) -> float:
        """计算平均内存使用（MB）

        Returns:
            平均内存使用
        """
        if not self.memory_usage:
            return 0.0

        return np.mean(self.memory_usage)

    def compute_gpu_utilization(self) -> float:
        """计算GPU利用率（%）

        Returns:
            GPU利用率
        """
        if not self.gpu_usage:
            return 0.0

        return np.mean(self.gpu_usage)

    def measure_inference_speed(self, model_function: Callable, test_data: List[Any], num_iterations: int = 10) -> Dict[str, float]:
        """测量推理速度

        Args:
            model_function: 模型推理函数
            test_data: 测试数据
            num_iterations: 迭代次数

        Returns:
            推理速度测量结果
        """
        if not test_data or not model_function:
            return {'inference_speed': 0.0}

        inference_times = []

        # 预热运行
        for _ in range(min(3, num_iterations // 3)):
            try:
                for data in test_data:
                    model_function(data)
            except Exception:
                pass

        for _ in range(num_iterations):
            start_time = time.time()
            try:
                # 执行推理
                for data in test_data:
                    model_function(data)
                end_time = time.time()
                inference_times.append(end_time - start_time)
            except Exception as e:
                logger.warning(f"推理速度测量失败: {e}")
                continue

        if not inference_times:
            return {'inference_speed': 0.0}

        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        samples_per_second = len(test_data) / avg_time if avg_time > 0 else 0

        return {
            'inference_speed': samples_per_second,
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'total_samples': len(test_data) * num_iterations
        }

    def measure_memory_usage(self, model: Any, inputs: Any) -> Dict[str, float]:
        """测量模型内存使用情况

        Args:
            model: 要测试的模型
            inputs: 输入数据

        Returns:
            内存使用指标
        """
        if model is None or inputs is None:
            return {'peak_memory': 0.0, 'memory_efficiency': 0.0}

        try:
            import psutil
            import gc

            # 获取初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 清理内存
            gc.collect()

            # 运行模型推理
            _ = model(inputs)

            # 获取峰值内存使用
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory

            # 计算内存效率（简化版本）
            model_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
            param_memory = model_params * 4 / 1024 / 1024  # 假设float32，MB

            memory_efficiency = param_memory / memory_used if memory_used > 0 else 0.0

            return {
                'peak_memory': peak_memory,
                'memory_used': memory_used,
                'initial_memory': initial_memory,
                'memory_efficiency': memory_efficiency,
                'model_params': model_params,
                'param_memory_mb': param_memory
            }
        except Exception as e:
            logger.warning(f"内存使用测量失败: {e}")
            return {'peak_memory': 0.0, 'memory_efficiency': 0.0}

    def _analyze_timing_trend(self, times: np.ndarray) -> str:
        """分析时间趋势

        Args:
            times: 时间数组

        Returns:
            趋势描述
        """
        if len(times) < 3:
            return "数据不足"

        # 简单的线性趋势分析
        x = np.arange(len(times))
        slope = np.polyfit(x, times, 1)[0]

        if abs(slope) < 0.001:
            return "稳定"
        elif slope > 0:
            return "递增（可能存在内存泄漏或性能退化）"
        else:
            return "递减（可能存在预热效应）"

    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """分析性能瓶颈

        Returns:
            瓶颈分析结果
        """
        bottlenecks = {
            'identified_bottlenecks': [],
            'resource_constraints': []
        }

        try:
            # 分析内存瓶颈
            if self.memory_usage:
                max_memory = max(self.memory_usage)
                avg_memory = np.mean(self.memory_usage)
                if max_memory > 8000:  # 8GB
                    bottlenecks['identified_bottlenecks'].append("高内存使用")
                if max_memory / avg_memory > 2.0:
                    bottlenecks['identified_bottlenecks'].append("内存使用不稳定")

            # 分析GPU瓶颈
            if self.gpu_usage:
                avg_gpu = np.mean(self.gpu_usage)
                if avg_gpu > 90:
                    bottlenecks['resource_constraints'].append("GPU使用率过高")
                elif avg_gpu < 30:
                    bottlenecks['resource_constraints'].append("GPU利用率不足")

            # 分析时间瓶颈
            if self.inference_times:
                times = np.array(self.inference_times)
                cv = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
                if cv > 0.5:
                    bottlenecks['identified_bottlenecks'].append("推理时间不稳定")

        except Exception as e:
            logger.warning(f"瓶颈分析失败: {e}")

        return bottlenecks

    def _generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议

        Returns:
            优化建议列表
        """
        suggestions = []

        try:
            # 基于批次大小的建议
            if self.batch_sizes:
                avg_batch_size = np.mean(self.batch_sizes)
                if avg_batch_size < 4:
                    suggestions.append("考虑增加批次大小以提高吞吐量")
                elif avg_batch_size > 32:
                    suggestions.append("考虑减少批次大小以降低内存使用")

            # 基于内存使用的建议
            if self.memory_usage:
                max_memory = max(self.memory_usage)
                if max_memory > 6000:  # 6GB
                    suggestions.append("考虑使用混合精度训练或模型量化")

            # 基于GPU使用的建议
            if self.gpu_usage:
                avg_gpu = np.mean(self.gpu_usage)
                if avg_gpu < 50:
                    suggestions.append("GPU利用率较低，考虑增加批次大小或使用多GPU")

            # 基于推理时间的建议
            if self.inference_times:
                times = np.array(self.inference_times)
                if np.mean(times) > 1.0:  # 1秒
                    suggestions.append("推理时间较长，考虑模型优化或硬件升级")

        except Exception as e:
            logger.warning(f"生成优化建议失败: {e}")

        return suggestions

    def benchmark_throughput(
        self,
        test_data: List[Any],
        batch_sizes: Optional[List[int]] = None,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """基准吞吐量测试

        Args:
            test_data: 测试数据
            batch_sizes: 要测试的批次大小列表
            num_iterations: 每个批次大小的迭代次数

        Returns:
            基准测试结果
        """
        if self.model is None:
            return {'error': '未设置模型'}

        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        results = {
            'batch_size_results': {},
            'optimal_batch_size': None,
            'max_throughput': 0
        }

        try:
            for batch_size in batch_sizes:
                if batch_size > len(test_data):
                    continue

                batch_results = self.efficiency_evaluator.benchmark_throughput(
                    test_data[:batch_size],
                    batch_size=batch_size,
                    num_iterations=num_iterations
                )

                if batch_results:
                    results['batch_size_results'][batch_size] = batch_results

                    # 更新最优批次大小
                    throughput = batch_results.get('throughput', 0)
                    if throughput > results['max_throughput']:
                        results['max_throughput'] = throughput
                        results['optimal_batch_size'] = batch_size

        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            results['error'] = str(e)

        return results
