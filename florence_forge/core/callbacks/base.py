"""Callback base class and manager.

Defines the lifecycle-hook contract (:class:`TrainerCallback`) and the
exception-isolating dispatcher (:class:`CallbackManager`) shared by every
built-in callback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid cycles
    from ...training.trainer import MultiTaskTrainer


# ---------------------------------------------------------------------------
# Callback 基类
# ---------------------------------------------------------------------------


class TrainerCallback:
    """训练器回调基类

    所有钩子均为空实现，子类按需重写。
    钩子调用顺序（每个 step）：
    1. on_step_begin
    2. 【forward / backward / optimizer step】
    3. on_step_end
    """

    # ---------- 训练级 ----------
    def on_train_begin(
        self,
        trainer: "MultiTaskTrainer",
        config: Any,
    ) -> None:
        """训练开始前触发"""
        pass

    def on_train_end(
        self,
        trainer: "MultiTaskTrainer",
        config: Any,
    ) -> None:
        """训练结束后触发"""
        pass

    # ---------- Epoch 级 ----------
    def on_epoch_begin(
        self,
        trainer: "MultiTaskTrainer",
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "MultiTaskTrainer",
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    # ---------- Step 级 ----------
    def on_step_begin(
        self,
        trainer: "MultiTaskTrainer",
        step: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def on_step_end(
        self,
        trainer: "MultiTaskTrainer",
        step: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    # ---------- 评估级 ----------
    def on_eval_begin(
        self,
        trainer: "MultiTaskTrainer",
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def on_eval_end(
        self,
        trainer: "MultiTaskTrainer",
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    # ---------- 检查点 ----------
    def on_save(
        self,
        trainer: "MultiTaskTrainer",
        output_dir: str,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def on_load(
        self,
        trainer: "MultiTaskTrainer",
        checkpoint_dir: str,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Callback 管理器
# ---------------------------------------------------------------------------


class CallbackManager:
    """统一管理所有 Callback 的调用

    每个 callback 调用都被隔离在 try-except 中，单个 callback 的异常
    不会影响其他 callback 的执行，确保训练流程的健壮性。
    """

    def __init__(self, callbacks: Optional[List[TrainerCallback]] = None):
        self.callbacks = callbacks or []
        self.logger = logging.getLogger(__name__ + ".CallbackManager")

    def append(self, callback: TrainerCallback) -> None:
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def _safe_call(self, callback: TrainerCallback, method_name: str, *args, **kwargs) -> None:
        """安全调用 callback 方法，捕获异常并记录日志"""
        try:
            method = getattr(callback, method_name)
            method(*args, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Callback {callback.__class__.__name__}.{method_name} 执行失败: {e}"
            )

    # ========== 训练级 ==========
    def on_train_begin(self, trainer: "MultiTaskTrainer", config: Any) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_train_begin", trainer, config)

    def on_train_end(self, trainer: "MultiTaskTrainer", config: Any) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_train_end", trainer, config)

    # ========== Epoch 级 ==========
    def on_epoch_begin(
        self,
        trainer: "MultiTaskTrainer",
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_epoch_begin", trainer, epoch, logs)

    def on_epoch_end(
        self,
        trainer: "MultiTaskTrainer",
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_epoch_end", trainer, epoch, logs)

    # ========== Step 级 ==========
    def on_step_begin(
        self,
        trainer: "MultiTaskTrainer",
        step: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_step_begin", trainer, step, logs)

    def on_step_end(
        self,
        trainer: "MultiTaskTrainer",
        step: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_step_end", trainer, step, logs)

    # ========== 评估级 ==========
    def on_eval_begin(
        self,
        trainer: "MultiTaskTrainer",
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_eval_begin", trainer, logs)

    def on_eval_end(
        self,
        trainer: "MultiTaskTrainer",
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_eval_end", trainer, logs)

    # ========== 检查点 ==========
    def on_save(
        self,
        trainer: "MultiTaskTrainer",
        output_dir: str,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_save", trainer, output_dir, logs)

    def on_load(
        self,
        trainer: "MultiTaskTrainer",
        checkpoint_dir: str,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_load", trainer, checkpoint_dir, logs)
