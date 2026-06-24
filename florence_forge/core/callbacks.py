"""
FlorenceForge 统一 Callback 系统（P1-2 优化）

为训练器/评估器提供生命周期钩子，支持：
- 日志/监控（TensorBoard、WandB、打印）
- 检查点策略（按 step/epoch 保存、最大保留数）
- 早停
- 梯度裁剪/监控
- 自定义扩展

设计参考：HuggingFace TrainerCallback + Keras Callback
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional


if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid cycles
    from ..training.trainer import MultiTaskTrainer

from ..utils.training_logging import (
    format_training_complete,
    format_training_start,
    format_training_step,
    resolve_total_steps,
    should_log_step,
)

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


# ---------------------------------------------------------------------------
# 内置 Callback 实现
# ---------------------------------------------------------------------------


class LoggingCallback(TrainerCallback):
    """日志回调：以紧凑、可检索的格式打印训练进度。"""

    def __init__(self, logging_steps: int = 100):
        self.logging_steps = logging_steps
        self.train_start_time: Optional[float] = None
        self.logger = logging.getLogger(__name__ + ".LoggingCallback")

    @staticmethod
    def _should_emit(trainer) -> bool:
        accelerator = getattr(trainer, "accelerator", None)
        return accelerator is None or getattr(accelerator, "is_local_main_process", True)

    def on_train_begin(self, trainer, config):
        self.train_start_time = time.time()
        if not self._should_emit(trainer):
            return
        data_settings = getattr(config, "data_settings", None)
        batch_size = getattr(data_settings, "batch_size", getattr(config, "batch_size", None))
        self.logger.info(
            format_training_start(
                epochs=getattr(config, "num_epochs", 1),
                batch_size=batch_size,
                gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", None),
                logging_steps=self.logging_steps,
            )
        )

    def on_step_end(self, trainer, step, logs=None):
        if not self._should_emit(trainer):
            return
        completed_step = step + 1
        config = getattr(trainer, "config", None)
        total_steps = resolve_total_steps(
            getattr(trainer, "train_dataloader", None),
            getattr(config, "num_epochs", None),
            getattr(config, "max_steps", None),
        )
        if not should_log_step(completed_step, self.logging_steps, total_steps):
            return
        self.logger.info(
            format_training_step(
                completed_step=completed_step,
                total_steps=total_steps,
                epoch=getattr(trainer, "current_epoch", 0) + 1,
                total_epochs=getattr(config, "num_epochs", None),
                metrics=logs,
                task_type=(logs or {}).get("task_type"),
                elapsed_seconds=(
                    time.time() - self.train_start_time if self.train_start_time else None
                ),
            )
        )

    def on_train_end(self, trainer, config):
        if self.train_start_time and self._should_emit(trainer):
            self.logger.info(
                format_training_complete(
                    completed_steps=getattr(trainer, "global_step", None),
                    elapsed_seconds=time.time() - self.train_start_time,
                    best_metric=getattr(trainer, "best_metric", None),
                )
            )


class CheckpointCallback(TrainerCallback):
    """检查点回调：按 step 保存，限制保留数"""

    def __init__(self, save_steps: int = 1000, save_total_limit: int = 3):
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

    def on_step_end(self, trainer, step, logs=None):
        if step > 0 and step % self.save_steps == 0:
            output_dir = trainer.config.output_dir
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                trainer.accelerator.save_model(trainer.model, checkpoint_dir)
            except Exception as e:
                logging.getLogger(__name__).error(f"CheckpointCallback: 保存检查点失败: {e}")
            self.on_save(trainer, checkpoint_dir, logs)
            # 清理旧检查点
            self._cleanup_old_checkpoints(trainer)

    def _cleanup_old_checkpoints(self, trainer) -> None:
        if self.save_total_limit <= 0:
            return
        output_dir = Path(trainer.config.output_dir)
        checkpoints = sorted([
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ], key=os.path.getmtime)
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)


class EarlyStoppingCallback(TrainerCallback):
    """早停回调"""

    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 5,
        threshold: float = 0.001,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.threshold = threshold
        self.mode = mode  # "min" or "max"

        self.wait = 0
        self.best: Optional[float] = None
        self.stopped = False
        self.logger = logging.getLogger(__name__ + ".EarlyStoppingCallback")

    def on_eval_end(self, trainer, logs=None):
        if self.stopped:
            return
        current = logs.get(self.monitor, None) if logs else None
        if current is None:
            return

        if self.best is None:
            self.best = current
            return

        improve = (
            current < self.best - self.threshold
            if self.mode == "min"
            else current > self.best + self.threshold
        )

        if improve:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.logger.info(
                    f"[EarlyStopping] 已 {self.patience} 次评估未改善，触发早停。"
                )
                self.stopped = True
                trainer._stop_training = True


class GradientClipCallback(TrainerCallback):
    """梯度裁剪监控回调

    注意：实际的梯度裁剪由训练器通过 accelerator.clip_grad_norm_() 完成。
    此回调仅记录梯度范数，不执行裁剪，以避免重复裁剪导致梯度异常。
    """

    def __init__(self, max_grad_norm: float = 1.0, log_frequency: int = 100):
        self.max_grad_norm = max_grad_norm
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__ + ".GradientClipCallback")

    def on_step_end(self, trainer, step, logs=None):
        if step % self.log_frequency != 0:
            return
        # 仅记录梯度范数，不执行裁剪
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > self.max_grad_norm * 2:
            self.logger.warning(
                f"Step {step}: 梯度范数 {total_norm:.4f} 远超阈值 {self.max_grad_norm}，"
                f"训练器已自动裁剪"
            )


class TensorBoardCallback(TrainerCallback):
    """TensorBoard 日志回调（可选依赖）"""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.writer = None
        self.logger = logging.getLogger(__name__ + ".TensorBoardCallback")

    def on_train_begin(self, trainer, config):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir = self.log_dir or config.logging_dir or f"{config.output_dir}/logs"
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info(f"[TensorBoardCallback] 日志目录: {self.log_dir}")
        except ImportError:
            self.logger.warning("[TensorBoardCallback] tensorboard 未安装，跳过")

    def on_step_end(self, trainer, step, logs=None):
        if self.writer and logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", v, step)

    def on_train_end(self, trainer, config):
        if self.writer:
            self.writer.close()
            self.writer = None


class MonitoringCallback(TrainerCallback):
    """统一监控回调——将 TrainingMonitor 集成到 Callback 体系

    此回调消除了 Callback 系统和 TrainingMonitor 之间的并行问题。
    所有监控操作（WandB、SwanLab、TensorBoard）都通过此回调统一管理，
    训练器无需直接调用 TrainingMonitor。

    使用方式：
        from florence_forge.training.monitoring import MonitoringConfig, TrainingMonitor

        monitor_config = MonitoringConfig(enable_wandb=True, wandb_project="my-project")
        monitor = TrainingMonitor(monitor_config, output_dir="./output")
        callback = MonitoringCallback(monitor, log_frequency=10)
        trainer.callbacks.append(callback)
    """

    def __init__(self, monitor=None, log_frequency: int = 10, log_gradients: bool = False):
        """初始化监控回调

        Args:
            monitor: TrainingMonitor 实例。如果为 None，则创建一个默认的
                     TensorBoard-only 监控器。
            log_frequency: 每多少步记录一次指标
            log_gradients: 是否记录梯度信息
        """
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        self.logger = logging.getLogger(__name__ + ".MonitoringCallback")

        # 延迟初始化 monitor（避免在 import 时触发依赖检查）
        self._monitor = monitor
        self._owns_monitor = False

    def _ensure_monitor(self, output_dir: Optional[str] = None):
        """确保 monitor 已初始化"""
        if self._monitor is not None:
            return
        try:
            from ..training.monitoring import MonitoringConfig, TrainingMonitor
            config = MonitoringConfig(enable_tensorboard=True)
            self._monitor = TrainingMonitor(config, output_dir=output_dir or "./outputs")
            self._owns_monitor = True
        except Exception as e:
            self.logger.warning(f"MonitoringCallback: 无法创建默认监控器: {e}")

    def on_train_begin(self, trainer, config):
        output_dir = getattr(config, 'output_dir', './outputs')
        self._ensure_monitor(output_dir)
        if self._monitor is not None and hasattr(self._monitor, 'config'):
            if getattr(self._monitor.config, 'log_model_architecture', False):
                try:
                    self._monitor.log_model_architecture(trainer.model)
                except Exception as e:
                    self.logger.warning(f"记录模型架构失败: {e}")

    def on_step_end(self, trainer, step, logs=None):
        if self._monitor is None or step % self.log_frequency != 0:
            return
        if logs:
            try:
                self._monitor.log_metrics(logs, step, prefix="train")
            except Exception as e:
                self.logger.warning(f"记录训练指标失败: {e}")

        # 记录梯度
        if self.log_gradients and hasattr(trainer, 'model'):
            try:
                self._monitor.log_gradients(trainer.model, step)
            except Exception as e:
                self.logger.warning(f"记录梯度失败: {e}")

    def on_eval_end(self, trainer, logs=None):
        if self._monitor is None or not logs:
            return
        step = getattr(trainer, 'global_step', 0)
        try:
            self._monitor.log_metrics(logs, step, prefix="eval")
        except Exception as e:
            self.logger.warning(f"记录评估指标失败: {e}")

    def on_epoch_end(self, trainer, epoch, logs=None):
        """记录 epoch 级指标到监控器"""
        if self._monitor is None or not logs:
            return
        try:
            # 提取训练和验证指标
            train_metrics = logs.get("train_metrics", {})
            val_metrics = logs.get("val_metrics", {})
            epoch_metrics = {}
            if train_metrics:
                epoch_metrics.update({f"epoch/{k}": v for k, v in train_metrics.items()})
            if val_metrics:
                epoch_metrics.update({f"epoch/{k}": v for k, v in val_metrics.items()})
            if epoch_metrics:
                self._monitor.log_metrics(epoch_metrics, epoch, prefix="epoch")
        except Exception as e:
            self.logger.warning(f"记录 epoch 指标失败: {e}")

    def on_train_end(self, trainer, config):
        if self._monitor is not None:
            try:
                self._monitor.finish()
            except Exception as e:
                self.logger.warning(f"结束监控失败: {e}")


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------


def create_default_callbacks(config, monitor=None) -> List[TrainerCallback]:
    """根据 TrainingConfig 创建默认 Callback 列表

    Args:
        config: 训练配置
        monitor: 可选的 TrainingMonitor 实例。如果提供，将创建
                 MonitoringCallback 替代独立的 TensorBoardCallback。
    """
    callbacks: List[TrainerCallback] = []

    # 日志
    if hasattr(config, "logging_steps"):
        callbacks.append(LoggingCallback(logging_steps=config.logging_steps))

    # 检查点
    if hasattr(config, "save_steps") and hasattr(config, "save_total_limit"):
        callbacks.append(
            CheckpointCallback(
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
            )
        )

    # 早停（如果配置了）
    if getattr(config, "early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                monitor=config.metric_for_best_model,
                patience=config.early_stopping_patience,
                threshold=config.early_stopping_threshold,
                mode="min" if not config.greater_is_better else "max",
            )
        )

    # 监控——统一入口（WandB/SwanLab/TensorBoard）
    log_frequency = getattr(config, "logging_steps", 10)
    if monitor is not None:
        # 使用 MonitoringCallback 统一管理所有监控工具
        callbacks.append(MonitoringCallback(
            monitor=monitor,
            log_frequency=log_frequency,
            log_gradients=getattr(config, "log_gradients", False),
        ))
    elif getattr(config, "use_tensorboard", True):
        # 向后兼容：没有 monitor 时使用独立的 TensorBoardCallback
        callbacks.append(TensorBoardCallback())

    return callbacks
