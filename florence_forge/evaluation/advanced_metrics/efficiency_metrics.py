"""效率评估指标

提供模型效率评估功能，包括推理速度、内存使用、计算复杂度等
"""

import logging
import time
import psutil
import gc
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

from ...utils.optional_dependencies import missing_dependency_message

try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug(
        missing_dependency_message("GPU效率监控", "GPUtil")
    )
logger = logging.getLogger(__name__)

@dataclass
class EfficiencyMetrics:
    """效率指标结果"""
    inference_time: float  # 推理时间（秒）
    throughput: float  # 吞吐量（样本/秒）
    memory_usage: Dict[str, float]  # 内存使用情况
    gpu_usage: Dict[str, float]  # GPU使用情况
    cpu_usage: float  # CPU使用率
    model_size: float  # 模型大小（MB）
    flops: Optional[float] = None  # 浮点运算次数
    params_count: Optional[int] = None  # 参数数量
    energy_consumption: Optional[float] = None  # 能耗估计

@dataclass
class BatchEfficiencyResult:
    """批次效率测试结果"""
    batch_size: int
    metrics: EfficiencyMetrics
    latency_percentiles: Dict[str, float]  # 延迟百分位数
    memory_peak: float  # 内存峰值
    stability_score: float  # 稳定性评分

class EfficiencyEvaluator:
    """效率评估器
    
    提供全面的模型效率评估功能
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        warmup_runs: int = 5,
        measurement_runs: int = 20
    ):
        """初始化效率评估器
        
        Args:
            device: 计算设备
            warmup_runs: 预热运行次数
            measurement_runs: 测量运行次数
        """
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        
        # 初始化监控工具
        self.process = psutil.Process()
        
    def evaluate_inference_speed(
        self,
        model: Any,
        inputs: Any,
        batch_sizes: List[int] = None
    ) -> Dict[int, BatchEfficiencyResult]:
        """评估推理速度
        
        Args:
            model: 待评估模型
            inputs: 输入数据
            batch_sizes: 批次大小列表
            
        Returns:
            不同批次大小的效率结果
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        results = {}
        
        try:
            model.eval()
            
            for batch_size in batch_sizes:
                logger.info(f"评估批次大小 {batch_size} 的推理速度")
                
                # 准备批次数据
                batch_inputs = self._prepare_batch_inputs(inputs, batch_size)
                
                # 预热
                self._warmup_model(model, batch_inputs)
                
                # 测量推理时间
                latencies = self._measure_inference_latency(model, batch_inputs)
                
                # 测量资源使用
                memory_usage = self._measure_memory_usage(model, batch_inputs)
                gpu_usage = self._measure_gpu_usage(model, batch_inputs)
                
                # 计算指标
                avg_latency = np.mean(latencies)
                throughput = batch_size / avg_latency
                
                # 计算延迟百分位数
                latency_percentiles = {
                    "p50": float(np.percentile(latencies, 50)),
                    "p90": float(np.percentile(latencies, 90)),
                    "p95": float(np.percentile(latencies, 95)),
                    "p99": float(np.percentile(latencies, 99))
                }
                
                # 计算稳定性评分
                stability_score = 1.0 - (np.std(latencies) / np.mean(latencies))
                
                # 获取模型信息
                model_size = self._get_model_size(model)
                params_count = self._count_parameters(model)
                
                efficiency_metrics = EfficiencyMetrics(
                    inference_time=avg_latency,
                    throughput=throughput,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage,
                    cpu_usage=self._get_cpu_usage(),
                    model_size=model_size,
                    params_count=params_count
                )
                
                results[batch_size] = BatchEfficiencyResult(
                    batch_size=batch_size,
                    metrics=efficiency_metrics,
                    latency_percentiles=latency_percentiles,
                    memory_peak=memory_usage.get("peak_memory", 0.0),
                    stability_score=stability_score
                )
                
        except Exception as e:
            logger.error(f"推理速度评估失败: {e}")
        
        return results
    
    def _prepare_batch_inputs(self, inputs: Any, batch_size: int) -> Any:
        """准备批次输入数据"""
        if TORCH_AVAILABLE and isinstance(inputs, torch.Tensor):
            # 重复输入以达到指定批次大小
            if inputs.size(0) < batch_size:
                repeat_times = (batch_size + inputs.size(0) - 1) // inputs.size(0)
                inputs = inputs.repeat(repeat_times, *([1] * (inputs.dim() - 1)))
            return inputs[:batch_size]
        else:
            # 处理其他类型的输入
            if isinstance(inputs, (list, tuple)):
                return inputs[:batch_size] if len(inputs) >= batch_size else inputs * batch_size
            return inputs
    
    def _warmup_model(self, model: Any, inputs: Any) -> None:
        """模型预热"""
        try:
            with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: iter([None]))():
                for _ in range(self.warmup_runs):
                    _ = model(inputs)
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def _measure_inference_latency(self, model: Any, inputs: Any) -> List[float]:
        """测量推理延迟"""
        latencies = []
        
        try:
            with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: iter([None]))():
                for _ in range(self.measurement_runs):
                    start_time = time.perf_counter()
                    
                    _ = model(inputs)
                    
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    latencies.append(end_time - start_time)
                    
        except Exception as e:
            logger.error(f"延迟测量失败: {e}")
            latencies = [0.0] * self.measurement_runs
        
        return latencies
    
    def _measure_memory_usage(self, model: Any, inputs: Any) -> Dict[str, float]:
        """测量内存使用情况"""
        memory_info = {}
        
        try:
            # 系统内存
            process_memory = self.process.memory_info()
            memory_info["system_memory_mb"] = process_memory.rss / 1024 / 1024
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # GPU内存
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
                
                with torch.no_grad():
                    _ = model(inputs)
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                memory_info["gpu_memory_mb"] = peak_memory - initial_memory
                memory_info["peak_memory"] = peak_memory
                memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024
                
                torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            logger.error(f"内存使用测量失败: {e}")
            memory_info = {"system_memory_mb": 0.0, "gpu_memory_mb": 0.0}
        
        return memory_info
    
    def _measure_gpu_usage(self, model: Any, inputs: Any) -> Dict[str, float]:
        """测量GPU使用情况"""
        gpu_info = {}
        
        try:
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    gpu_info["gpu_utilization"] = gpu.load * 100
                    gpu_info["gpu_memory_utilization"] = gpu.memoryUtil * 100
                    gpu_info["gpu_temperature"] = gpu.temperature
            
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                # 基本GPU信息
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["current_device"] = torch.cuda.current_device()
                
        except Exception as e:
            logger.error(f"GPU使用测量失败: {e}")
            gpu_info = {"gpu_utilization": 0.0}
        
        return gpu_info
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _get_model_size(self, model: Any) -> float:
        """获取模型大小（MB）"""
        try:
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                param_size = 0
                buffer_size = 0
                
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                
                return (param_size + buffer_size) / 1024 / 1024
            
        except Exception as e:
            logger.error(f"模型大小计算失败: {e}")
        
        return 0.0
    
    def _count_parameters(self, model: Any) -> Optional[int]:
        """计算模型参数数量"""
        try:
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
        except Exception as e:
            logger.error(f"参数计数失败: {e}")
        
        return None
    
    def profile_model(
        self,
        model: Any,
        inputs: Any,
        profile_memory: bool = True,
        profile_shapes: bool = True
    ) -> Dict[str, Any]:
        """使用PyTorch Profiler分析模型
        
        Args:
            model: 待分析模型
            inputs: 输入数据
            profile_memory: 是否分析内存
            profile_shapes: 是否分析张量形状
            
        Returns:
            分析结果
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch未安装，无法进行模型分析")
            return {}
        
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            
            with torch.profiler.profile(
                activities=activities,
                record_shapes=profile_shapes,
                profile_memory=profile_memory,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    _ = model(inputs)
            
            # 分析结果
            analysis = {
                "key_averages": prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
                "memory_profile": prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_memory_usage", row_limit=10
                ) if profile_memory else None,
                "trace_file": None  # 可以保存trace文件
            }
            
            # 提取关键指标
            events = prof.key_averages()
            total_time = sum(event.cuda_time_total for event in events if event.cuda_time_total > 0)
            
            analysis["summary"] = {
                "total_cuda_time": total_time,
                "num_events": len(events),
                "avg_event_time": total_time / len(events) if events else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"模型分析失败: {e}")
            return {}
    
    def benchmark_throughput(
        self,
        model: Any,
        inputs: Any,
        duration_seconds: float = 30.0
    ) -> Dict[str, float]:
        """基准吞吐量测试
        
        Args:
            model: 待测试模型
            inputs: 输入数据
            duration_seconds: 测试持续时间
            
        Returns:
            吞吐量基准结果
        """
        try:
            model.eval()
            
            # 预热
            self._warmup_model(model, inputs)
            
            start_time = time.perf_counter()
            end_time = start_time + duration_seconds
            
            total_samples = 0
            inference_times = []
            
            with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: iter([None]))():
                while time.perf_counter() < end_time:
                    inference_start = time.perf_counter()
                    
                    _ = model(inputs)
                    
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    inference_end = time.perf_counter()
                    
                    batch_size = inputs.size(0) if TORCH_AVAILABLE and isinstance(inputs, torch.Tensor) else 1
                    total_samples += batch_size
                    inference_times.append(inference_end - inference_start)
            
            actual_duration = time.perf_counter() - start_time
            
            return {
                "throughput_samples_per_sec": total_samples / actual_duration,
                "avg_inference_time": np.mean(inference_times),
                "total_samples": total_samples,
                "actual_duration": actual_duration,
                "total_inferences": len(inference_times)
            }
            
        except Exception as e:
            logger.error(f"吞吐量基准测试失败: {e}")
            return {"throughput_samples_per_sec": 0.0}
    
    def compare_efficiency(
        self,
        models: Dict[str, Any],
        inputs: Any,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """比较多个模型的效率
        
        Args:
            models: 模型字典 {name: model}
            inputs: 输入数据
            metrics: 要比较的指标列表
            
        Returns:
            效率比较结果
        """
        if metrics is None:
            metrics = ["inference_time", "throughput", "memory_usage", "model_size"]
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"评估模型: {model_name}")
            
            # 评估单个批次大小
            batch_results = self.evaluate_inference_speed(model, inputs, [1])
            
            if 1 in batch_results:
                batch_result = batch_results[1]
                model_metrics = batch_result.metrics
                
                results[model_name] = {
                    "inference_time": model_metrics.inference_time,
                    "throughput": model_metrics.throughput,
                    "memory_usage": model_metrics.memory_usage.get("gpu_memory_mb", 0.0),
                    "model_size": model_metrics.model_size,
                    "params_count": model_metrics.params_count or 0,
                    "stability_score": batch_result.stability_score
                }
        
        return results
    
    def generate_efficiency_report(
        self,
        batch_results: Dict[int, BatchEfficiencyResult]
    ) -> Dict[str, Any]:
        """生成效率评估报告
        
        Args:
            batch_results: 批次效率结果
            
        Returns:
            详细的效率报告
        """
        if not batch_results:
            return {"error": "没有可用的评估结果"}
        
        # 提取关键指标
        batch_sizes = list(batch_results.keys())
        throughputs = [result.metrics.throughput for result in batch_results.values()]
        latencies = [result.metrics.inference_time for result in batch_results.values()]
        memory_usage = [result.memory_peak for result in batch_results.values()]
        
        # 找到最优批次大小
        optimal_batch_idx = np.argmax(throughputs)
        optimal_batch_size = batch_sizes[optimal_batch_idx]
        
        # 计算效率评分
        efficiency_score = self._calculate_efficiency_score(batch_results)
        
        # 生成建议
        recommendations = self._generate_efficiency_recommendations(batch_results)
        
        return {
            "summary": {
                "optimal_batch_size": optimal_batch_size,
                "max_throughput": max(throughputs),
                "min_latency": min(latencies),
                "efficiency_score": efficiency_score,
                "model_size_mb": list(batch_results.values())[0].metrics.model_size,
                "total_parameters": list(batch_results.values())[0].metrics.params_count
            },
            "detailed_results": {
                batch_size: {
                    "throughput": result.metrics.throughput,
                    "latency": result.metrics.inference_time,
                    "memory_peak": result.memory_peak,
                    "stability_score": result.stability_score,
                    "latency_percentiles": result.latency_percentiles
                }
                for batch_size, result in batch_results.items()
            },
            "recommendations": recommendations,
            "evaluation_metadata": {
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": self.device,
                "warmup_runs": self.warmup_runs,
                "measurement_runs": self.measurement_runs
            }
        }
    
    def _calculate_efficiency_score(
        self,
        batch_results: Dict[int, BatchEfficiencyResult]
    ) -> float:
        """计算综合效率评分"""
        if not batch_results:
            return 0.0
        
        # 归一化指标
        throughputs = [result.metrics.throughput for result in batch_results.values()]
        latencies = [result.metrics.inference_time for result in batch_results.values()]
        stability_scores = [result.stability_score for result in batch_results.values()]
        
        # 计算归一化分数
        max_throughput = max(throughputs)
        min_latency = min(latencies)
        avg_stability = np.mean(stability_scores)
        
        # 综合评分（权重可调整）
        throughput_score = max_throughput / (max_throughput + 1)  # 归一化到0-1
        latency_score = min_latency / (min_latency + 1)  # 越小越好，归一化
        latency_score = 1 - latency_score  # 反转，使得越小分数越高
        
        efficiency_score = 0.4 * throughput_score + 0.3 * latency_score + 0.3 * avg_stability
        
        return float(efficiency_score)
    
    def _generate_efficiency_recommendations(
        self,
        batch_results: Dict[int, BatchEfficiencyResult]
    ) -> List[str]:
        """生成效率优化建议"""
        recommendations = []
        
        if not batch_results:
            return recommendations
        
        # 分析批次大小效果
        batch_sizes = sorted(batch_results.keys())
        throughputs = [batch_results[bs].metrics.throughput for bs in batch_sizes]
        
        # 找到吞吐量峰值
        max_throughput_idx = np.argmax(throughputs)
        optimal_batch_size = batch_sizes[max_throughput_idx]
        
        recommendations.append(f"建议使用批次大小 {optimal_batch_size} 以获得最佳吞吐量")
        
        # 内存使用分析
        memory_peaks = [batch_results[bs].memory_peak for bs in batch_sizes]
        if max(memory_peaks) > 8000:  # 8GB
            recommendations.append("内存使用较高，考虑减小批次大小或使用梯度检查点")
        
        # 稳定性分析
        stability_scores = [batch_results[bs].stability_score for bs in batch_sizes]
        if min(stability_scores) < 0.8:
            recommendations.append("推理时间不稳定，建议增加预热时间或检查系统负载")
        
        # 延迟分析
        avg_latency = np.mean([batch_results[bs].metrics.inference_time for bs in batch_sizes])
        if avg_latency > 1.0:  # 1秒
            recommendations.append("推理延迟较高，考虑模型量化或使用更快的硬件")
        
        return recommendations
