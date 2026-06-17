#!/usr/bin/env python3
"""向后兼容别名：``fsds_plugin`` → ``fsdp_plugin``。

历史拼写错误保留此模块，新代码请使用 ``florence_forge.training.fsdp_plugin``。
"""

from .fsdp_plugin import FSDPPlugin

__all__ = ["FSDPPlugin"]
