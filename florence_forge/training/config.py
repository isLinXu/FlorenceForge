"""Training configuration management module.

This module provides configuration validation and management for training processes.
Re-exports core Pydantic config classes and adds training-specific helpers.
"""

import logging
from typing import Union
from pathlib import Path

# 核心配置类：直接导入，失败即报错（Pydantic v2 已重构）
from ..core.config import (
    TrainingConfig,
)

# 监控配置：可选依赖
try:
    from .monitoring import MonitoringConfig
except ImportError:
    MonitoringConfig = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 向后兼容：旧代码使用 dataclass 版 TrainingConfig 时，重定向到 Pydantic 版
# ---------------------------------------------------------------------------
# Pydantic 版 TrainingConfig 已包含完整的字段校验和序列化能力，
# 此处不再维护重复的 dataclass 版本。
#
# 如需从旧代码迁移：
#   from florence_forge.training.config import TrainingConfig  # ✅ 仍然可用
#   config = TrainingConfig.from_dict({...})                   # ✅ Pydantic 自动校验
#   config = TrainingConfig.load_from_yaml("config.yaml")      # ✅ 新增方法
# ---------------------------------------------------------------------------


def load_config_from_file(config_path: Union[str, Path]) -> TrainingConfig:
    """加载训练配置文件（YAML/JSON）

    Args:
        config_path: 配置文件路径

    Returns:
        TrainingConfig 实例（Pydantic v2 校验）
    """
    return TrainingConfig.load_from_file(config_path)


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """验证配置文件是否有效

    Args:
        config_path: 配置文件路径

    Returns:
        配置是否有效
    """
    try:
        TrainingConfig.load_from_file(config_path)
        return True
    except Exception as e:
        logger.error(f"配置文件验证失败: {e}")
        return False


def create_default_config() -> TrainingConfig:
    """创建默认训练配置"""
    return TrainingConfig()