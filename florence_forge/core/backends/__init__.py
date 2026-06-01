"""VLM 后端包导出入口。

本包仅导出 4 个**已验证的真实 VLM 后端**：
    - Florence2Backend
    - PaliGemmaBackend
    - YouTuVLBackend
    - GenericHFBackend（通用 HF VLM 自动适配）

并配套：
    - BaseVLMBackend：抽象基类（7 大接口）
    - VLMBackendRegistry：动态注册表
    - create_backend / auto_select_backend：工厂函数

MoE / SSM 等实验性组件已迁移到 `florence_forge.experimental.moe`，
不会从这里导出。如需使用：

    >>> from florence_forge.experimental.moe import MoELayer  # 实验性 API
"""

from .base_vlm import (
    BaseVLMBackend,
    VLMBackendRegistry,
    create_backend,
    _check_flash_attn_availability,
    _patch_transformers_import_check,
)
from .florence2_backend import Florence2Backend
from .generic_hf_backend import GenericHFBackend, _guess_architecture_type
from .paligemma_backend import PaliGemmaBackend
from .youtuvl_backend import YouTuVLBackend


def auto_select_backend(config):
    """根据配置直接创建后端实例。"""
    return VLMBackendRegistry.create(getattr(config, "backend_name", "florence-2"), config)


__all__ = [
    "BaseVLMBackend",
    "VLMBackendRegistry",
    "create_backend",
    "auto_select_backend",
    "_check_flash_attn_availability",
    "_patch_transformers_import_check",
    "Florence2Backend",
    "GenericHFBackend",
    "PaliGemmaBackend",
    "YouTuVLBackend",
    "_guess_architecture_type",
]
