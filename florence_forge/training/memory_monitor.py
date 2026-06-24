"""内存监控模块

提供训练过程中的内存使用监控和优化建议。
"""

import gc
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Any
import json

from ..utils.training_logging import format_memory_snapshot

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """内存统计信息"""
    timestamp: float
    cpu_memory_mb: float
    cpu_memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None
    gpu_memory_cached_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None  # 峰值显存（自上次重置以来）
    process_memory_mb: Optional[float] = None
    step: Optional[int] = None
    phase: Optional[str] = None  # 'forward', 'backward', 'optimizer', 'validation'


@dataclass
class MemoryMonitorConfig:
    """内存监控配置"""
    enable_monitoring: bool = True
    log_frequency: int = 100  # 每多少步记录一次
    warning_threshold_percent: float = 85.0  # 内存使用警告阈值
    critical_threshold_percent: float = 95.0  # 内存使用严重警告阈值
    enable_gpu_monitoring: bool = True
    enable_continuous_monitoring: bool = False  # 是否启用连续监控
    monitoring_interval: float = 5.0  # 连续监控间隔（秒）
    auto_cleanup: bool = True  # 是否自动清理内存
    save_stats: bool = True  # 是否保存统计信息
    max_stats_history: int = 10000  # 最大历史记录数
    enable_peak_tracking: bool = True  # 是否跟踪峰值显存
    suggest_gradient_accumulation: bool = True  # 是否自动建议梯度累积调整


class MemoryMonitor:
    """内存监控器
    
    监控训练过程中的内存使用情况，提供警告和优化建议。
    """
    
    def __init__(self, config: MemoryMonitorConfig):
        self.config = config
        self.stats_history: Deque[MemoryStats] = deque(
            maxlen=max(1, self.config.max_stats_history)
        )
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # 峰值显存跟踪
        self._gpu_peak_mb: Optional[float] = None
        self._gpu_peak_reset_step: int = 0

        # 检查依赖
        if psutil is None:
            logger.warning("psutil未安装，CPU内存监控功能受限")

        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA不可用，GPU内存监控功能禁用")
            self.config.enable_gpu_monitoring = False

        # 获取系统信息
        self.total_cpu_memory = self._get_total_cpu_memory()
        self.total_gpu_memory = self._get_total_gpu_memory()

        logger.info(f"内存监控器初始化完成 - CPU: {self.total_cpu_memory:.1f}MB, GPU: {self.total_gpu_memory:.1f}MB")
    
    def _get_total_cpu_memory(self) -> float:
        """获取总CPU内存"""
        if psutil:
            return psutil.virtual_memory().total / 1024 / 1024
        return 0.0
    
    def _get_total_gpu_memory(self) -> float:
        """获取总GPU内存"""
        if torch and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 0.0
    
    def get_current_stats(self, step: Optional[int] = None, phase: Optional[str] = None) -> MemoryStats:
        """获取当前内存统计信息"""
        timestamp = time.time()
        
        # CPU内存
        cpu_memory_mb = 0.0
        cpu_memory_percent = 0.0
        process_memory_mb = None
        
        if psutil:
            memory = psutil.virtual_memory()
            cpu_memory_mb = memory.used / 1024 / 1024
            cpu_memory_percent = memory.percent
            
            # 当前进程内存
            try:
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        
        # GPU内存
        gpu_memory_mb = None
        gpu_memory_percent = None
        gpu_memory_reserved_mb = None
        gpu_memory_cached_mb = None
        gpu_memory_peak_mb = None

        if self.config.enable_gpu_monitoring and torch and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                gpu_memory_cached_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                # 跟踪峰值显存
                if self.config.enable_peak_tracking:
                    current_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                    if self._gpu_peak_mb is None or current_peak > self._gpu_peak_mb:
                        self._gpu_peak_mb = current_peak
                    gpu_memory_peak_mb = self._gpu_peak_mb

                if self.total_gpu_memory > 0:
                    gpu_memory_percent = (gpu_memory_mb / self.total_gpu_memory) * 100
            except Exception as e:
                logger.debug(f"获取GPU内存信息失败: {e}")

        return MemoryStats(
            timestamp=timestamp,
            cpu_memory_mb=cpu_memory_mb,
            cpu_memory_percent=cpu_memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            gpu_memory_cached_mb=gpu_memory_cached_mb,
            gpu_memory_peak_mb=gpu_memory_peak_mb,
            process_memory_mb=process_memory_mb,
            step=step,
            phase=phase
        )
    
    def log_memory_usage(self, step: Optional[int] = None, phase: Optional[str] = None, force: bool = False) -> MemoryStats:
        """记录内存使用情况
        
        Args:
            step: 当前训练步数
            phase: 当前阶段
            force: 是否强制记录（忽略频率限制）
            
        Returns:
            当前内存统计信息
        """
        if not self.config.enable_monitoring:
            return self.get_current_stats(step, phase)
        
        # 检查是否需要记录
        should_log = force
        if not should_log and step is not None:
            should_log = step % self.config.log_frequency == 0
        
        stats = self.get_current_stats(step, phase)
        
        if should_log:
            # 检查警告阈值
            warnings = self._check_memory_warnings(stats)
            
            log_msg = format_memory_snapshot(
                step=step,
                phase=phase,
                cpu_percent=stats.cpu_memory_percent,
                cpu_mb=stats.cpu_memory_mb,
                gpu_percent=stats.gpu_memory_percent,
                gpu_mb=stats.gpu_memory_mb,
                gpu_peak_mb=stats.gpu_memory_peak_mb,
                process_mb=stats.process_memory_mb,
            )

            if warnings:
                logger.warning("%s | warning=%s", log_msg, ", ".join(warnings))
            elif phase in {None, "after_optimizer", "validation", "continuous"}:
                logger.info(log_msg)
            else:
                logger.debug(log_msg)

            # 自动清理
            if self.config.auto_cleanup and warnings:
                self.cleanup_memory()

            # 自动建议梯度累积调整
            if self.config.suggest_gradient_accumulation and stats.gpu_memory_percent is not None:
                if stats.gpu_memory_percent >= self.config.critical_threshold_percent:
                    suggested_accum = self._suggest_gradient_accumulation(stats)
                    if suggested_accum > 1:
                        logger.warning(
                            f"💡 内存优化建议: 当前显存使用过高 ({stats.gpu_memory_percent:.1f}%)，"
                            f"建议将 gradient_accumulation_steps 增加到 {suggested_accum} "
                            f"以减小 batch_size 并降低显存占用"
                        )
        
        # 保存统计信息
        if self.config.save_stats:
            self.stats_history.append(stats)
        
        return stats
    
    def _check_memory_warnings(self, stats: MemoryStats) -> List[str]:
        """检查内存警告"""
        warnings = []
        
        # CPU内存警告
        if stats.cpu_memory_percent >= self.config.critical_threshold_percent:
            warnings.append(f"CPU内存严重不足 ({stats.cpu_memory_percent:.1f}%)")
        elif stats.cpu_memory_percent >= self.config.warning_threshold_percent:
            warnings.append(f"CPU内存不足 ({stats.cpu_memory_percent:.1f}%)")
        
        # GPU内存警告
        if stats.gpu_memory_percent is not None:
            if stats.gpu_memory_percent >= self.config.critical_threshold_percent:
                warnings.append(f"GPU内存严重不足 ({stats.gpu_memory_percent:.1f}%)")
            elif stats.gpu_memory_percent >= self.config.warning_threshold_percent:
                warnings.append(f"GPU内存不足 ({stats.gpu_memory_percent:.1f}%)")
        
        return warnings
    
    def _suggest_gradient_accumulation(self, stats: MemoryStats) -> int:
        """根据当前显存使用建议梯度累积步数

        基于经验规则：当显存超过 95% 时，建议将累积步数翻倍
        直到显存降到安全范围或达到上限（16步）。

        Returns:
            建议的 gradient_accumulation_steps
        """
        if stats.gpu_memory_percent is None or self.total_gpu_memory <= 0:
            return 1

        # 简单的经验规则：显存每超过阈值 10%，累积步数翻倍
        over_ratio = (stats.gpu_memory_percent - self.config.warning_threshold_percent) / 10.0
        if over_ratio <= 0:
            return 1

        suggested = int(2 ** over_ratio)
        return min(suggested, 16)  # 上限 16 步

    def cleanup_memory(self) -> Dict[str, float]:
        """清理内存（含显存碎片整理）

        Returns:
            清理前后的内存使用情况
        """
        before_stats = self.get_current_stats()

        # Python垃圾回收
        gc.collect()

        # GPU内存清理与碎片整理
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 重置峰值统计以便下次准确跟踪
            torch.cuda.reset_peak_memory_stats()
            self._gpu_peak_mb = None

        after_stats = self.get_current_stats()

        # 计算清理效果
        cleanup_info = {
            'cpu_memory_freed_mb': before_stats.cpu_memory_mb - after_stats.cpu_memory_mb,
            'cpu_memory_freed_percent': before_stats.cpu_memory_percent - after_stats.cpu_memory_percent
        }

        if before_stats.gpu_memory_mb is not None and after_stats.gpu_memory_mb is not None:
            cleanup_info.update({
                'gpu_memory_freed_mb': before_stats.gpu_memory_mb - after_stats.gpu_memory_mb,
                'gpu_memory_freed_percent': (before_stats.gpu_memory_percent or 0) - (after_stats.gpu_memory_percent or 0)
            })
        
        logger.info(f"内存清理完成 - CPU释放: {cleanup_info['cpu_memory_freed_mb']:.1f}MB, "
                   f"GPU释放: {cleanup_info.get('gpu_memory_freed_mb', 0):.1f}MB")
        
        return cleanup_info
    
    def start_continuous_monitoring(self) -> None:
        """启动连续监控"""
        if not self.config.enable_continuous_monitoring or self.monitoring_thread is not None:
            return
        
        def monitor_loop():
            while not self.stop_monitoring.wait(self.config.monitoring_interval):
                try:
                    self.log_memory_usage(phase="continuous")
                except Exception as e:
                    logger.error(f"连续监控过程中出错: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("连续内存监控已启动")
    
    def stop_continuous_monitoring(self) -> None:
        """停止连续监控"""
        if self.monitoring_thread is None:
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        self.stop_monitoring.clear()
        logger.info("连续内存监控已停止")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        if not self.stats_history:
            return {'error': '没有内存统计数据'}
        
        # 计算统计信息
        cpu_usage = [s.cpu_memory_percent for s in self.stats_history]
        gpu_usage = [s.gpu_memory_percent for s in self.stats_history if s.gpu_memory_percent is not None]
        
        summary = {
            'total_records': len(self.stats_history),
            'monitoring_duration_hours': (self.stats_history[-1].timestamp - self.stats_history[0].timestamp) / 3600,
            'cpu_memory': {
                'max_usage_percent': max(cpu_usage),
                'avg_usage_percent': sum(cpu_usage) / len(cpu_usage),
                'min_usage_percent': min(cpu_usage),
                'total_memory_mb': self.total_cpu_memory
            }
        }
        
        if gpu_usage:
            summary['gpu_memory'] = {
                'max_usage_percent': max(gpu_usage),
                'avg_usage_percent': sum(gpu_usage) / len(gpu_usage),
                'min_usage_percent': min(gpu_usage),
                'total_memory_mb': self.total_gpu_memory
            }
        
        # 警告统计
        warning_count = 0
        critical_count = 0
        
        for stats in self.stats_history:
            warnings = self._check_memory_warnings(stats)
            if any('严重' in w for w in warnings):
                critical_count += 1
            elif warnings:
                warning_count += 1
        
        summary['warnings'] = {
            'warning_count': warning_count,
            'critical_count': critical_count,
            'warning_rate': warning_count / len(self.stats_history),
            'critical_rate': critical_count / len(self.stats_history)
        }
        
        return summary
    
    def export_stats(self, output_path: Path) -> Path:
        """导出内存统计信息
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备导出数据
        export_data = {
            'config': {
                'log_frequency': self.config.log_frequency,
                'warning_threshold_percent': self.config.warning_threshold_percent,
                'critical_threshold_percent': self.config.critical_threshold_percent,
                'enable_gpu_monitoring': self.config.enable_gpu_monitoring
            },
            'system_info': {
                'total_cpu_memory_mb': self.total_cpu_memory,
                'total_gpu_memory_mb': self.total_gpu_memory
            },
            'summary': self.get_memory_summary(),
            'stats_history': [
                {
                    'timestamp': stats.timestamp,
                    'cpu_memory_mb': stats.cpu_memory_mb,
                    'cpu_memory_percent': stats.cpu_memory_percent,
                    'gpu_memory_mb': stats.gpu_memory_mb,
                    'gpu_memory_percent': stats.gpu_memory_percent,
                    'gpu_memory_reserved_mb': stats.gpu_memory_reserved_mb,
                    'gpu_memory_cached_mb': stats.gpu_memory_cached_mb,
                    'process_memory_mb': stats.process_memory_mb,
                    'step': stats.step,
                    'phase': stats.phase
                }
                for stats in self.stats_history
            ]
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"内存统计信息已导出到: {output_path}")
        return output_path
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取内存优化建议"""
        if not self.stats_history:
            return ["没有足够的内存统计数据来提供建议"]
        
        recommendations = []
        summary = self.get_memory_summary()
        
        # CPU内存建议
        cpu_max = summary['cpu_memory']['max_usage_percent']
        if cpu_max > 90:
            recommendations.append("CPU内存使用率过高，建议减少批次大小或增加系统内存")
        elif cpu_max > 80:
            recommendations.append("CPU内存使用率较高，建议监控内存泄漏")
        
        # GPU内存建议
        if 'gpu_memory' in summary:
            gpu_max = summary['gpu_memory']['max_usage_percent']
            if gpu_max > 90:
                recommendations.append("GPU内存使用率过高，建议减少批次大小或使用梯度累积")
            elif gpu_max > 80:
                recommendations.append("GPU内存使用率较高，建议启用混合精度训练")
        
        # 警告频率建议
        warning_rate = summary['warnings']['warning_rate']
        if warning_rate > 0.1:
            recommendations.append("内存警告频率过高，建议优化内存使用策略")
        
        if not recommendations:
            recommendations.append("内存使用情况良好，无需特别优化")
        
        return recommendations
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.config.enable_continuous_monitoring:
            self.start_continuous_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_continuous_monitoring()
