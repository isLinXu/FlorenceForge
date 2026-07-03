"""Built-in training callbacks: logging, checkpointing, early stopping,
gradient monitoring and TensorBoard.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from ...utils.training_logging import (
    format_training_complete,
    format_training_start,
    format_training_step,
    resolve_total_steps,
    should_log_step,
)
from .base import TrainerCallback


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
