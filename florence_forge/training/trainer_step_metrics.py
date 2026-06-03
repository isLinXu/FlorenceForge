"""训练步指标记录（供 MultiDatasetTrainer 等扩展训练器使用）。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def init_step_metrics_state(trainer: Any) -> None:
    """在训练器上初始化步级 CSV / 内存指标状态。"""
    output_dir = Path(trainer.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.output_dir = output_dir

    trainer.step_csv_buffer = []
    trainer.csv_buffer_size = int(getattr(trainer.config, "csv_buffer_size", 100) or 100)
    trainer.step_metrics = []
    trainer.step_metrics_history_limit = int(
        getattr(trainer.config, "step_metrics_history_limit", 1000) or 0
    )
    trainer.step_csv_path = output_dir / "step_metrics.csv"
    if not trainer.step_csv_path.exists():
        with open(trainer.step_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "epoch",
                    "task_type",
                    "loss",
                    "learning_rate",
                    "grad_norm",
                    "time_per_step",
                    "data_time",
                    "forward_time",
                    "backward_time",
                    "optim_time",
                ]
            )


def flush_csv_buffer(trainer: Any) -> None:
    if not getattr(trainer, "step_csv_buffer", None):
        return
    if not trainer.step_csv_buffer:
        return
    try:
        with open(trainer.step_csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(trainer.step_csv_buffer)
        trainer.step_csv_buffer.clear()
    except Exception as exc:
        logger.warning("CSV缓冲区刷新失败: %s", exc)


def record_step_metrics(
    trainer: Any,
    *,
    step: int,
    epoch: int,
    task_type: str,
    loss: float,
    learning_rate: float,
    grad_norm: float,
    time_per_step: float,
    data_time: float = 0.0,
    forward_time: float = 0.0,
    backward_time: float = 0.0,
    optim_time: float = 0.0,
) -> None:
    trainer.step_csv_buffer.append(
        [
            step,
            epoch,
            task_type,
            loss,
            learning_rate,
            grad_norm,
            time_per_step,
            data_time,
            forward_time,
            backward_time,
            optim_time,
        ]
    )
    if len(trainer.step_csv_buffer) >= trainer.csv_buffer_size:
        flush_csv_buffer(trainer)

    trainer.step_metrics.append(
        {
            "step": step,
            "epoch": epoch,
            "task_type": task_type,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "time_per_step": time_per_step,
            "data_time": data_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optim_time": optim_time,
        }
    )
    limit = trainer.step_metrics_history_limit
    if limit > 0 and len(trainer.step_metrics) > limit:
        del trainer.step_metrics[:-limit]
    elif limit == 0:
        trainer.step_metrics.clear()

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and getattr(accelerator, "is_local_main_process", True):
        accelerator.log(
            {
                "train/loss": loss,
                "train/learning_rate": learning_rate,
                "train/grad_norm": grad_norm,
                "train/time_per_step": time_per_step,
                "train/data_time": data_time,
                "train/forward_time": forward_time,
                "train/backward_time": backward_time,
                "train/optim_time": optim_time,
            },
            step=step,
        )
