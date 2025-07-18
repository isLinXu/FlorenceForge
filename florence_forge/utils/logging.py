"""FlorenceForge日志工具模块

提供统一的日志配置和管理功能
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_name: bool = True
) -> None:
    """设置日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        log_format: 自定义日志格式
        include_timestamp: 是否包含时间戳
        include_level: 是否包含日志级别
        include_name: 是否包含记录器名称
    """
    # 构建日志格式
    if log_format is None:
        format_parts = []
        
        if include_timestamp:
            format_parts.append('%(asctime)s')
        
        if include_level:
            format_parts.append('[%(levelname)s]')
        
        if include_name:
            format_parts.append('%(name)s')
        
        format_parts.append('%(message)s')
        
        log_format = ' - '.join(format_parts)
    
    # 创建格式化器
    formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的记录器
    
    Args:
        name: 记录器名称
        
    Returns:
        记录器实例
    """
    return logging.getLogger(name)

def create_experiment_logger(
    experiment_name: str,
    log_dir: Union[str, Path],
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """创建实验专用记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        level: 日志级别
        
    Returns:
        实验记录器
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # 创建记录器
    logger = logging.getLogger(f"experiment.{experiment_name}")
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"实验记录器已创建: {experiment_name}")
    logger.info(f"日志文件: {log_file}")
    
    return logger

class LoggerMixin:
    """日志记录器混入类
    
    为类提供便捷的日志记录功能
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取类专用记录器"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """记录信息日志"""
        self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """记录警告日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """记录错误日志"""
        self.logger.error(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """记录调试日志"""
        self.logger.debug(message, *args, **kwargs)

class ProgressLogger:
    """进度日志记录器
    
    用于记录长时间运行任务的进度
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int, log_interval: int = 100):
        """初始化进度记录器
        
        Args:
            logger: 日志记录器
            total_steps: 总步数
            log_interval: 日志记录间隔
        """
        self.logger = logger
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = None
    
    def start(self) -> None:
        """开始记录进度"""
        self.start_time = datetime.now()
        self.logger.info(f"开始任务，总步数: {self.total_steps}")
    
    def update(self, step: Optional[int] = None, **kwargs) -> None:
        """更新进度
        
        Args:
            step: 当前步数（如果不提供则自动递增）
            **kwargs: 额外的进度信息
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if self.current_step % self.log_interval == 0 or self.current_step == self.total_steps:
            progress = self.current_step / self.total_steps * 100
            
            # 计算剩余时间
            if self.start_time and self.current_step > 0:
                elapsed = datetime.now() - self.start_time
                estimated_total = elapsed * self.total_steps / self.current_step
                remaining = estimated_total - elapsed
                
                time_info = f"剩余时间: {str(remaining).split('.')[0]}"
            else:
                time_info = ""
            
            # 构建进度信息
            progress_info = f"进度: {self.current_step}/{self.total_steps} ({progress:.1f}%)"
            
            if kwargs:
                extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                progress_info += f", {extra_info}"
            
            if time_info:
                progress_info += f", {time_info}"
            
            self.logger.info(progress_info)
    
    def finish(self) -> None:
        """完成进度记录"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"任务完成，总耗时: {str(elapsed).split('.')[0]}")
        else:
            self.logger.info("任务完成")

def log_function_call(func):
    """函数调用日志装饰器
    
    记录函数的调用和执行时间
    """
    def wrapper(*args, **kwargs):
        """包装函数，记录目标函数的调用和执行时间
        
        Args:
            *args: 目标函数的位置参数
            **kwargs: 目标函数的关键字参数
            
        Returns:
            目标函数的返回值
        """
        logger = get_logger(func.__module__)
        
        # 记录函数调用
        logger.debug(f"调用函数: {func.__name__}")
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            
            # 记录成功完成
            elapsed = datetime.now() - start_time
            logger.debug(f"函数 {func.__name__} 执行完成，耗时: {elapsed.total_seconds():.3f}秒")
            
            return result
        
        except Exception as e:
            # 记录异常
            elapsed = datetime.now() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {elapsed.total_seconds():.3f}秒，错误: {e}")
            raise
    
    return wrapper

def log_memory_usage(logger: logging.Logger) -> None:
    """记录内存使用情况"""
    try:
        import psutil
        import torch
        
        # 系统内存
        memory = psutil.virtual_memory()
        logger.info(f"系统内存使用: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        
        # GPU内存（如果可用）
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                logger.info(
                    f"GPU {i} 内存使用: 已分配 {memory_allocated:.1f}GB, "
                    f"已保留 {memory_reserved:.1f}GB, 总计 {memory_total:.1f}GB"
                )
    
    except ImportError:
        logger.warning("psutil未安装，无法获取内存使用信息")
    except Exception as e:
        logger.warning(f"获取内存使用信息失败: {e}")