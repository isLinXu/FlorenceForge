"""FlorenceForge内存管理工具模块

提供内存监控、优化和管理功能
"""

import gc
import math
import psutil
import torch
import threading
import time
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
from collections import deque
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path

@dataclass
class MemoryInfo:
    """内存信息数据类"""
    total: float  # 总内存 (GB)
    available: float  # 可用内存 (GB)
    used: float  # 已用内存 (GB)
    percent: float  # 使用百分比
    
    def __str__(self) -> str:
        """返回内存信息的字符串表示
        
        Returns:
            str: 格式化的内存使用情况字符串，显示已用/总内存和使用百分比
        """
        return f"Memory: {self.used:.1f}GB/{self.total:.1f}GB ({self.percent:.1f}%)"

@dataclass
class GPUMemoryInfo:
    """GPU内存信息数据类"""
    device_id: int
    name: str
    total: float  # 总显存 (GB)
    allocated: float  # 已分配显存 (GB)
    reserved: float  # 已保留显存 (GB)
    free: float  # 空闲显存 (GB)
    
    def __str__(self) -> str:
        """返回GPU内存信息的字符串表示
        
        Returns:
            str: 格式化的GPU内存使用情况字符串，显示设备ID、名称和已分配/总显存
        """
        return f"GPU {self.device_id} ({self.name}): {self.allocated:.1f}GB/{self.total:.1f}GB"

def get_memory_usage() -> MemoryInfo:
    """获取系统内存使用情况
    
    Returns:
        内存信息
    """
    memory = psutil.virtual_memory()
    
    return MemoryInfo(
        total=memory.total / (1024**3),
        available=memory.available / (1024**3),
        used=memory.used / (1024**3),
        percent=memory.percent
    )

def get_gpu_memory_usage() -> List[GPUMemoryInfo]:
    """获取GPU内存使用情况
    
    Returns:
        GPU内存信息列表
    """
    gpu_info = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            free = total - reserved
            
            gpu_info.append(GPUMemoryInfo(
                device_id=i,
                name=props.name,
                total=total,
                allocated=allocated,
                reserved=reserved,
                free=free
            ))
    
    return gpu_info

def clear_cache() -> None:
    """清理缓存"""
    # 清理Python垃圾回收
    gc.collect()
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def clear_gpu_cache() -> None:
    """清理GPU缓存
    
    专门用于清理GPU内存缓存的函数
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 强制垃圾回收
        gc.collect()

def optimize_memory(
    aggressive: bool = False,
    clear_gradients: bool = True,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """优化内存使用
    
    Args:
        aggressive: 是否进行激进优化
        clear_gradients: 是否清理梯度
        model: 可选模型；提供时仅遍历其参数释放梯度，避免全局对象扫描
        
    Returns:
        优化前后的内存使用情况
    """
    # 记录优化前的内存使用
    before_memory = get_memory_usage()
    before_gpu = get_gpu_memory_usage()
    
    # 基础优化
    if clear_gradients:
        if model is not None:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 清理缓存
    clear_cache()
    
    # 激进优化
    if aggressive:
        # 强制垃圾回收多次
        for _ in range(3):
            gc.collect()
        
        # 清理未使用的张量
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # 记录优化后的内存使用
    after_memory = get_memory_usage()
    after_gpu = get_gpu_memory_usage()
    
    result = {
        'system_memory_freed': before_memory.used - after_memory.used,
        'system_memory_before': before_memory.used,
        'system_memory_after': after_memory.used
    }
    
    if before_gpu and after_gpu:
        result.update({
            'gpu_memory_freed': before_gpu[0].allocated - after_gpu[0].allocated,
            'gpu_memory_before': before_gpu[0].allocated,
            'gpu_memory_after': after_gpu[0].allocated
        })
    
    return result

@contextmanager
def memory_monitor(name: str = "Operation", log_func=None):
    """内存监控上下文管理器
    
    Args:
        name: 操作名称
        log_func: 日志函数
    """
    if log_func is None:
        log_func = print
    
    # 记录开始时的内存
    start_memory = get_memory_usage()
    start_gpu = get_gpu_memory_usage()
    
    log_func(f"[{name}] 开始 - {start_memory}")
    if start_gpu:
        log_func(f"[{name}] 开始 - {start_gpu[0]}")
    
    try:
        yield
    finally:
        # 记录结束时的内存
        end_memory = get_memory_usage()
        end_gpu = get_gpu_memory_usage()
        
        memory_diff = end_memory.used - start_memory.used
        log_func(f"[{name}] 结束 - {end_memory} (变化: {memory_diff:+.1f}GB)")
        
        if start_gpu and end_gpu:
            gpu_diff = end_gpu[0].allocated - start_gpu[0].allocated
            log_func(f"[{name}] 结束 - {end_gpu[0]} (变化: {gpu_diff:+.1f}GB)")

class MemoryTracker:
    """内存跟踪器
    
    持续监控内存使用情况
    """
    
    def __init__(
        self,
        interval: float = 1.0,
        max_history: int = 1000
    ):
        """初始化内存跟踪器
        
        Args:
            interval: 监控间隔（秒）
            max_history: 最大历史记录数
        """
        self.interval = interval
        self.max_history = max(0, max_history)
        self.history = deque(maxlen=self.max_history)
        self.is_tracking = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """开始跟踪"""
        if self.is_tracking:
            return
        
        self.is_tracking = True
        self.thread = threading.Thread(target=self._track_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """停止跟踪"""
        self.is_tracking = False
        if self.thread:
            self.thread.join()
    
    def _track_loop(self) -> None:
        """跟踪循环"""
        while self.is_tracking:
            timestamp = time.time()
            memory_info = get_memory_usage()
            gpu_info = get_gpu_memory_usage()
            
            record = {
                'timestamp': timestamp,
                'system_memory': {
                    'total': memory_info.total,
                    'used': memory_info.used,
                    'percent': memory_info.percent
                }
            }
            
            if gpu_info:
                record['gpu_memory'] = [
                    {
                        'device_id': gpu.device_id,
                        'allocated': gpu.allocated,
                        'reserved': gpu.reserved,
                        'total': gpu.total
                    }
                    for gpu in gpu_info
                ]
            
            self.history.append(record)
            
            time.sleep(self.interval)
    
    def get_peak_usage(self) -> Dict[str, float]:
        """获取峰值使用情况
        
        Returns:
            峰值使用情况
        """
        if not self.history:
            return {}
        
        peak_system = max(record['system_memory']['used'] for record in self.history)
        result = {'peak_system_memory': peak_system}
        
        # GPU峰值
        if self.history[0].get('gpu_memory'):
            for i in range(len(self.history[0]['gpu_memory'])):
                peak_gpu = max(
                    record['gpu_memory'][i]['allocated'] 
                    for record in self.history 
                    if 'gpu_memory' in record and i < len(record['gpu_memory'])
                )
                result[f'peak_gpu_{i}_memory'] = peak_gpu
        
        return result
    
    def get_average_usage(self) -> Dict[str, float]:
        """获取平均使用情况
        
        Returns:
            平均使用情况
        """
        if not self.history:
            return {}
        
        avg_system = sum(record['system_memory']['used'] for record in self.history) / len(self.history)
        result = {'avg_system_memory': avg_system}
        
        # GPU平均
        if self.history[0].get('gpu_memory'):
            for i in range(len(self.history[0]['gpu_memory'])):
                gpu_values = [
                    record['gpu_memory'][i]['allocated'] 
                    for record in self.history 
                    if 'gpu_memory' in record and i < len(record['gpu_memory'])
                ]
                if gpu_values:
                    result[f'avg_gpu_{i}_memory'] = sum(gpu_values) / len(gpu_values)
        
        return result
    
    def export_history(self, file_path: Union[str, Path]) -> None:
        """导出历史记录
        
        Args:
            file_path: 文件路径
        """
        import json
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(list(self.history), f, indent=2)

class MemoryPool:
    """内存池
    
    管理可重用的张量内存
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """初始化内存池
        
        Args:
            device: 设备
        """
        self.device = device or torch.device('cpu')
        self.pools: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        self.max_pool_size = 10
    
    def get_tensor(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """获取张量
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            
        Returns:
            张量
        """
        if isinstance(shape, tuple):
            shape = torch.Size(shape)
        
        key = (shape, dtype)
        
        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()  # 清零
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """归还张量
        
        Args:
            tensor: 要归还的张量
        """
        if tensor.device != self.device:
            return
        
        key = (tensor.shape, tensor.dtype)
        
        if key not in self.pools:
            self.pools[key] = []
        
        if len(self.pools[key]) < self.max_pool_size:
            # 分离梯度并移动到CPU（如果需要）
            tensor = tensor.detach()
            self.pools[key].append(tensor)
    
    def clear(self) -> None:
        """清空内存池"""
        self.pools.clear()
        clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息
        
        Returns:
            统计信息
        """
        total_tensors = sum(len(pool) for pool in self.pools.values())
        total_memory = 0
        
        for (shape, dtype), pool in self.pools.items():
            tensor_size = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            total_memory += len(pool) * tensor_size
        
        return {
            'total_pools': len(self.pools),
            'total_tensors': total_tensors,
            'total_memory_mb': total_memory / (1024**2),
            'pools': {
                f"{shape}_{dtype}": len(pool)
                for (shape, dtype), pool in self.pools.items()
            }
        }

def check_memory_requirements(
    model_size_gb: float,
    batch_size: int,
    sequence_length: int,
    safety_factor: float = 1.5
) -> Dict[str, Any]:
    """检查内存需求
    
    Args:
        model_size_gb: 模型大小（GB）
        batch_size: 批次大小
        sequence_length: 序列长度
        safety_factor: 安全系数
        
    Returns:
        内存需求分析
    """
    # 估算内存需求
    # 模型参数 + 梯度 + 优化器状态 + 激活值
    estimated_memory = model_size_gb * 4  # 参数 + 梯度 + Adam状态
    
    # 激活值内存（粗略估算）
    activation_memory = batch_size * sequence_length * 1024 * 4 / (1024**3)  # 假设1024维隐藏层
    
    total_estimated = (estimated_memory + activation_memory) * safety_factor
    
    # 获取当前可用内存
    current_memory = get_memory_usage()
    gpu_memory = get_gpu_memory_usage()
    
    result = {
        'estimated_memory_gb': total_estimated,
        'model_memory_gb': model_size_gb,
        'activation_memory_gb': activation_memory,
        'safety_factor': safety_factor,
        'system_available_gb': current_memory.available,
        'system_sufficient': current_memory.available > total_estimated
    }
    
    if gpu_memory:
        gpu = gpu_memory[0]
        result.update({
            'gpu_available_gb': gpu.free,
            'gpu_sufficient': gpu.free > total_estimated
        })
    
    return result

def suggest_batch_size(
    model_size_gb: float,
    available_memory_gb: float,
    sequence_length: int = 512,
    safety_factor: float = 0.8
) -> int:
    """建议批次大小
    
    Args:
        model_size_gb: 模型大小（GB）
        available_memory_gb: 可用内存（GB）
        sequence_length: 序列长度
        safety_factor: 安全系数
        
    Returns:
        建议的批次大小
    """
    # 保留内存用于模型和其他开销
    usable_memory = (available_memory_gb - model_size_gb * 3) * safety_factor
    
    if usable_memory <= 0:
        return 1
    
    # 估算每个样本的内存需求
    memory_per_sample = sequence_length * 1024 * 4 / (1024**3)  # 粗略估算
    
    suggested_batch_size = max(1, int(usable_memory / memory_per_sample))
    
    return suggested_batch_size

@contextmanager
def low_memory_mode():
    """低内存模式上下文管理器"""
    # 保存原始设置
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    
    try:
        # 设置为低内存模式
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 清理缓存
        clear_cache()
        
        yield
        
    finally:
        # 恢复原始设置
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

def monitor_memory_leaks(
    threshold_mb: float = 100.0,
    check_interval: float = 60.0
) -> None:
    """监控内存泄漏
    
    Args:
        threshold_mb: 内存增长阈值（MB）
        check_interval: 检查间隔（秒）
    """
    initial_memory = get_memory_usage().used * 1024  # 转换为MB
    
    def check_leak():
        """检查内存泄漏的内部函数
        
        检查当前内存使用量与初始内存的差值，如果超过阈值则记录警告日志。
        """
        current_memory = get_memory_usage().used * 1024
        growth = current_memory - initial_memory
        
        if growth > threshold_mb:
            warnings.warn(
                f"检测到可能的内存泄漏: 内存增长 {growth:.1f}MB",
                UserWarning
            )
    
    # 启动监控线程
    def monitor_loop():
        """内存监控循环函数
        
        在后台线程中持续运行，按指定间隔检查内存泄漏情况。
        """
        while True:
            time.sleep(check_interval)
            check_leak()
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
