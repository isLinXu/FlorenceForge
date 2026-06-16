"""Helpers for opt-in ``torch.compile`` support."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compile_module_if_requested(
    model: nn.Module,
    *,
    enabled: bool,
    mode: Optional[str] = None,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    backend: Optional[str] = None,
    context: str = "model",
) -> nn.Module:
    """Compile a module when requested, falling back to the original model.

    ``torch.compile`` can fail for some remote-code model classes or deployment
    environments. Because this is a performance optimization rather than a
    correctness requirement, failures are logged and the original module is
    returned unchanged.
    """
    if not enabled:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.warning("torch.compile 不可用，跳过 %s 编译", context)
        return model

    compile_kwargs = {"fullgraph": fullgraph}
    if mode and mode != "default":
        compile_kwargs["mode"] = mode
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    if backend:
        compile_kwargs["backend"] = backend

    try:
        compiled = compile_fn(model, **compile_kwargs)
    except Exception as exc:
        logger.warning("%s 编译失败，继续使用未编译模型: %s", context, exc)
        return model

    logger.info("%s 编译完成", context)
    return compiled
