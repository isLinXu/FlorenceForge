"""ArchitectureResolver —— 后端路由策略门面

历史背景
--------
本类最初用于替换 ``_build_mixer`` 中 5 分支的 ``if-elif-elif-elif-else`` 后端
分派，但与 ``VLMBackendRegistry`` 形成了两套**功能高度重叠**的注册机制
（详见 ``docs/Deep_Analysis_2026-05-28.md`` 的 P1-4）。

当前职责（重构后，2026-05-29）
-----------------------------
``ArchitectureResolver`` 现在是 ``VLMBackendRegistry`` 之上的一层薄壳，
保持原有 API 兼容的同时让 **VLM 后端从单一来源派生**：

* 注册的类若是 :class:`BaseVLMBackend` 子类 → 自动同步到 ``VLMBackendRegistry``
  （单一事实源）；
* 注册的类若不是 VLM 后端（例如测试 / 第三方扩展） → 仅保留在本类的局部表中；
* ``resolve()`` 找不到本地条目时会回退到 ``VLMBackendRegistry``，
  让 ``florence-2`` / ``paligemma`` 等"原生"后端在不显式 import 路由器时也能解析；
* ``clear()`` 只清空本类**自身**的注册表，避免误清空全局 VLM 注册。

向后兼容
--------
- ``register(name, cls)``、``register_builder(name, fn)``、``resolve(name, **kw)``、
  ``clear()``、``get_builder(name)`` 行为与旧版本完全一致。
- 新增 ``list_backends()`` / ``sync_from_vlm_registry()`` 便于检视全局可用后端。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


def _try_import_vlm_registry():
    """惰性引入 VLMBackendRegistry，避免循环依赖与早期 import 失败。"""
    try:
        from florence_forge.core.backends.base_vlm import (  # type: ignore
            BaseVLMBackend,
            VLMBackendRegistry,
        )

        return BaseVLMBackend, VLMBackendRegistry
    except Exception as exc:  # pragma: no cover - 仅在 backends 子包构建失败时触发
        logger.debug("VLMBackendRegistry 不可用: %s", exc)
        return None, None


class ArchitectureResolver:
    """路由表 + 策略模式的后端解析门面。

    设计目标：
        * 提供统一的入口 ``resolve(name, **kw)`` 屏蔽创建细节；
        * VLM 后端的注册以 :class:`VLMBackendRegistry` 为单一事实源；
        * 仍支持注册非 VLM 类或纯构建函数（向后兼容旧用法）。
    """

    # 局部注册表：用于非 VLM 类、定制 builder、第三方扩展
    _registry: Dict[str, Type[Any]] = {}
    _builders: Dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, backend_class: Type[Any]) -> None:
        """注册一个后端类到路由表。

        若 ``backend_class`` 是 :class:`BaseVLMBackend` 子类，会**同时**注册到
        :class:`VLMBackendRegistry`，避免两套注册表数据不一致。
        """
        cls._registry[name] = backend_class

        BaseVLMBackend, VLMBackendRegistry = _try_import_vlm_registry()
        if BaseVLMBackend is None or VLMBackendRegistry is None:
            return
        try:
            if isinstance(backend_class, type) and issubclass(backend_class, BaseVLMBackend):
                if not VLMBackendRegistry.is_registered(name):
                    VLMBackendRegistry.register(name, backend_class)
                    logger.debug("同步注册 VLM 后端到 VLMBackendRegistry: %s", name)
        except TypeError:
            # backend_class 不是普通 class（例如 Mock），忽略同步
            pass

    @classmethod
    def register_builder(
        cls, name: str, builder: Callable[..., Any]
    ) -> None:
        """注册构造函数（用于装饰、包装、依赖注入等复杂创建路径）。"""
        cls._builders[name] = builder

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls, name: str, **kwargs: Any) -> Any:
        """根据名称解析后端实例。

        解析顺序：
            1. 本地 ``_registry`` + 可选 ``_builders`` 包装
            2. 仅 ``_builders``（无对应类）
            3. 回退到 ``VLMBackendRegistry``（全局 VLM 注册）

        Raises:
            ValueError: 三处都找不到时抛出。
        """
        if name in cls._registry:
            backend_cls = cls._registry[name]
            if name in cls._builders:
                builder = cls._builders[name]
                return builder(backend_cls, **kwargs)
            return backend_cls(**kwargs)

        if name in cls._builders:
            return cls._builders[name](**kwargs)

        _, VLMBackendRegistry = _try_import_vlm_registry()
        if VLMBackendRegistry is not None and VLMBackendRegistry.is_registered(name):
            backend_cls = VLMBackendRegistry._backends[name.lower()]
            try:
                return backend_cls(**kwargs)
            except TypeError:
                # 多数 VLM 后端只接受一个 positional config
                config = kwargs.get("config") or next(iter(kwargs.values()), None)
                return backend_cls(config)

        available_local = sorted(set(cls._registry) | set(cls._builders))
        available_global: List[str] = []
        if VLMBackendRegistry is not None:
            available_global = VLMBackendRegistry.list_backends()
        raise ValueError(
            f"未知的后端名称: {name}。"
            f"本地已注册: {available_local}; "
            f"全局 VLM 注册表: {available_global}"
        )

    # ------------------------------------------------------------------
    # 检视 / 维护
    # ------------------------------------------------------------------

    @classmethod
    def clear(cls) -> None:
        """清空**本类自身**的路由表与策略映射。

        ⚠ 不会清空全局 ``VLMBackendRegistry``，避免影响其他模块。
        """
        cls._registry.clear()
        cls._builders.clear()

    @classmethod
    def get_builder(cls, name: str) -> Optional[Callable[..., Any]]:
        """获取指定后端的构建函数。"""
        return cls._builders.get(name)

    @classmethod
    def list_backends(cls) -> List[str]:
        """返回所有可解析的后端名称（本地 + 全局 VLM 注册表的并集）。"""
        names = set(cls._registry) | set(cls._builders)
        _, VLMBackendRegistry = _try_import_vlm_registry()
        if VLMBackendRegistry is not None:
            names.update(VLMBackendRegistry.list_backends())
        return sorted(names)

    @classmethod
    def sync_from_vlm_registry(cls) -> int:
        """将 :class:`VLMBackendRegistry` 中已注册但本地未登记的后端同步过来。

        Returns:
            同步条目数量。便于诊断脚本输出。
        """
        _, VLMBackendRegistry = _try_import_vlm_registry()
        if VLMBackendRegistry is None:
            return 0
        added = 0
        for name in VLMBackendRegistry.list_backends():
            if name not in cls._registry:
                cls._registry[name] = VLMBackendRegistry._backends[name]
                added += 1
        return added
