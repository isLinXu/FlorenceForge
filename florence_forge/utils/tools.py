#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具模块

提供各种常用的辅助工具和实用函数
"""

import time
import hashlib
import pickle
import json
import logging
from pathlib import Path
from typing import Union, Optional, Callable, Any, Dict, List
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """计时器工具类"""
    
    def __init__(self, name: str = "Timer"):
        """初始化计时器
        
        Args:
            name (str, optional): 计时器名称，用于标识不同的计时器实例。默认为"Timer"
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        logger.info(f"{self.name} 开始计时")
        return self
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            raise ValueError("计时器尚未启动")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(f"{self.name} 计时结束: {self.elapsed_time:.4f}秒")
        return self.elapsed_time
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


def timing_decorator(func_name: Optional[str] = None):
    """计时装饰器
    
    Args:
        func_name: 函数名称，用于日志显示
    """
    def decorator(func: Callable) -> Callable:
        """装饰器内部函数，用于包装目标函数
        
        Args:
            func (Callable): 要被装饰的函数
            
        Returns:
            Callable: 包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """包装函数，在执行目标函数时进行计时
            
            Args:
                *args: 目标函数的位置参数
                **kwargs: 目标函数的关键字参数
                
            Returns:
                目标函数的返回值
            """
            name = func_name or f"{func.__module__}.{func.__name__}"
            with Timer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class FileHasher:
    """文件哈希计算工具"""
    
    @staticmethod
    def md5_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """计算文件MD5哈希值
        
        Args:
            file_path: 文件路径
            chunk_size: 读取块大小
            
        Returns:
            MD5哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def sha256_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """计算文件SHA256哈希值
        
        Args:
            file_path: 文件路径
            chunk_size: 读取块大小
            
        Returns:
            SHA256哈希值
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def verify_file_integrity(
        file_path: Union[str, Path], 
        expected_hash: str, 
        hash_type: str = "md5"
    ) -> bool:
        """验证文件完整性
        
        Args:
            file_path: 文件路径
            expected_hash: 期望的哈希值
            hash_type: 哈希类型 (md5, sha256)
            
        Returns:
            是否验证通过
        """
        if hash_type.lower() == "md5":
            actual_hash = FileHasher.md5_hash(file_path)
        elif hash_type.lower() == "sha256":
            actual_hash = FileHasher.sha256_hash(file_path)
        else:
            raise ValueError(f"不支持的哈希类型: {hash_type}")
        
        return actual_hash.lower() == expected_hash.lower()


class ConfigManager:
    """配置管理器
    
    支持多种格式的配置文件管理
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """初始化配置管理器
        
        Args:
            config_path (Union[str, Path]): 配置文件路径，支持JSON、YAML等格式
            
        Note:
            初始化时会自动加载配置文件内容到config_data属性中
        """
        self.config_path = Path(config_path)
        self.config_data = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return
        
        suffix = self.config_path.suffix.lower()
        
        try:
            if suffix == ".json":
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            elif suffix in [".yml", ".yaml"]:
                import yaml
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
            elif suffix == ".pkl":
                with open(self.config_path, 'rb') as f:
                    self.config_data = pickle.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
            
            logger.info(f"成功加载配置文件: {self.config_path}")
        
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config_data = {}
    
    def save_config(self):
        """保存配置文件"""
        suffix = self.config_path.suffix.lower()
        
        # 确保目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if suffix == ".json":
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            elif suffix in [".yml", ".yaml"]:
                import yaml
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            elif suffix == ".pkl":
                with open(self.config_path, 'wb') as f:
                    pickle.dump(self.config_data, f)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
            
            logger.info(f"成功保存配置文件: {self.config_path}")
        
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config_data
        
        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """批量更新配置
        
        Args:
            updates: 更新字典
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        return self.config_data.copy()


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Progress"):
        """初始化进度跟踪器
        
        Args:
            total (int): 总步数或总任务数
            description (str, optional): 进度描述信息。默认为"Progress"
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = 1):
        """更新进度
        
        Args:
            step: 步长
        """
        self.current += step
        self._print_progress()
    
    def _print_progress(self):
        """打印进度信息"""
        if self.total <= 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        progress_bar = "█" * int(percentage // 2) + "░" * (50 - int(percentage // 2))
        
        print(f"\r{self.description}: [{progress_bar}] {percentage:.1f}% ({self.current}/{self.total}) {eta_str}", end="")
        
        if self.current >= self.total:
            print()  # 换行


@contextmanager
def suppress_warnings():
    """抑制警告的上下文管理器"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def retry_on_failure(
    max_retries: int = 3, 
    delay: float = 1.0, 
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        """装饰器内部函数，用于包装目标函数以支持重试机制
        
        Args:
            func (Callable): 要被装饰的函数
            
        Returns:
            Callable: 包装后的函数，具备重试功能
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """包装函数，在目标函数失败时执行重试逻辑
            
            Args:
                *args: 目标函数的位置参数
                **kwargs: 目标函数的关键字参数
                
            Returns:
                目标函数的返回值
                
            Raises:
                Exception: 当达到最大重试次数后仍然失败时抛出最后一次的异常
            """
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {e}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，{current_delay}秒后重试")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


def ensure_list(value: Union[Any, List[Any]]) -> List[Any]:
    """确保值为列表格式
    
    Args:
        value: 输入值
        
    Returns:
        列表格式的值
    """
    if isinstance(value, list):
        return value
    elif value is None:
        return []
    else:
        return [value]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键名
        sep: 分隔符
        
    Returns:
        展平后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """反展平字典
    
    Args:
        d: 展平的字典
        sep: 分隔符
        
    Returns:
        嵌套字典
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result