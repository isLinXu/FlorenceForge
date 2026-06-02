"""Concise, consistent console messages for training progress."""

from __future__ import annotations

import math
import os
import re
import sys
from typing import Any, List, Mapping, Optional

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_RESET = "\033[0m"
_STYLES = {
    "tag_train": "\033[36m",
    "tag_epoch": "\033[35m",
    "tag_complete": "\033[32m",
    "tag_memory": "\033[33m",
    "key": "\033[2m",
    "task": "\033[1m",
    "loss": "\033[33m",
    "lr": "\033[34m",
    "grad": "\033[35m",
    "time": "\033[36m",
    "good": "\033[32m",
    "memory": "\033[33m",
}


def should_colorize_logs(color: Optional[bool] = None) -> bool:
    """Return whether console training logs should include ANSI colors."""
    if color is not None:
        return bool(color)

    mode = os.getenv("FLORENCE_FORGE_COLOR_LOGS", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "always", "force"}:
        return True
    if mode in {"0", "false", "no", "off", "never"}:
        return False

    if os.getenv("NO_COLOR") or os.getenv("CI"):
        return False
    if os.getenv("CLICOLOR_FORCE") == "1":
        return True
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return _is_tty(sys.stdout) or _is_tty(sys.stderr)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color escapes from a formatted log line."""
    return ANSI_PATTERN.sub("", text)


def format_duration(seconds: Optional[float]) -> str:
    """Format elapsed or remaining time for compact log output."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def resolve_total_steps(
    dataloader: Any,
    num_epochs: Optional[int],
    max_steps: Optional[int] = None,
) -> Optional[int]:
    """Resolve the intended run step count without requiring a concrete loader."""
    if max_steps:
        return int(max_steps)
    if dataloader is None or not num_epochs:
        return None
    try:
        return len(dataloader) * int(num_epochs)
    except (TypeError, AttributeError):
        return None


def should_log_step(
    completed_step: int,
    logging_steps: int,
    total_steps: Optional[int] = None,
) -> bool:
    """Return whether a user-facing progress message should be emitted."""
    interval = max(1, int(logging_steps))
    return (
        completed_step == 1
        or completed_step % interval == 0
        or (total_steps is not None and completed_step >= total_steps)
    )


def format_training_start(
    *,
    epochs: int,
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    logging_steps: Optional[int] = None,
    color: Optional[bool] = None,
) -> str:
    """Render a stable training-run header."""
    fields = [f"epochs={epochs}"]
    if batch_size is not None:
        fields.append(f"batch_size={batch_size}")
    if gradient_accumulation_steps is not None:
        fields.append(f"grad_accum={gradient_accumulation_steps}")
    if logging_steps is not None:
        fields.append(f"log_every={logging_steps} steps")
    return _format_line("[train]", fields, color=color, tag_style="tag_train", event="start")


def format_training_step(
    *,
    completed_step: int,
    total_steps: Optional[int] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    task_type: Optional[str] = None,
    dataset_name: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    color: Optional[bool] = None,
) -> str:
    """Render a compact progress line with optional ETA."""
    fields = []
    if epoch is not None:
        epoch_text = f"{epoch}/{total_epochs}" if total_epochs else str(epoch)
        fields.append(f"epoch={epoch_text}")

    step_text = str(completed_step)
    if total_steps:
        percentage = min(100.0, completed_step / total_steps * 100.0)
        step_text = f"{completed_step}/{total_steps} ({percentage:.1f}%)"
    fields.append(f"step={step_text}")

    if task_type:
        fields.append(f"task={task_type}")
    if dataset_name and dataset_name != "unknown":
        fields.append(f"dataset={dataset_name}")

    values = metrics or {}
    _append_metric(fields, "loss", values.get("loss"), ".4f")
    _append_metric(fields, "lr", values.get("learning_rate"), ".2e")
    _append_metric(fields, "grad", values.get("grad_norm"), ".3f")
    _append_metric(fields, "step_time", values.get("time_per_step"), ".2f", suffix="s")

    eta = _estimate_eta(elapsed_seconds, completed_step, total_steps)
    if eta is not None:
        fields.append(f"eta={format_duration(eta)}")
    return _format_line("[train]", fields, color=color, tag_style="tag_train")


def format_epoch_summary(
    *,
    epoch: int,
    total_epochs: Optional[int] = None,
    train_metrics: Optional[Mapping[str, Any]] = None,
    val_metrics: Optional[Mapping[str, Any]] = None,
    color: Optional[bool] = None,
) -> str:
    """Render the epoch-level train/evaluation result line."""
    epoch_text = f"{epoch}/{total_epochs}" if total_epochs else str(epoch)
    fields = [f"epoch={epoch_text}"]
    train_values = train_metrics or {}
    val_values = val_metrics or {}

    _append_metric(fields, "train_loss", train_values.get("loss"), ".4f")
    val_loss = val_values.get("loss", val_values.get("val_loss"))
    if val_loss is None:
        fields.append("val_loss=--")
    else:
        _append_metric(fields, "val_loss", val_loss, ".4f")
    _append_metric(fields, "lr", train_values.get("learning_rate"), ".2e")
    return _format_line("[epoch]", fields, color=color, tag_style="tag_epoch")


def format_training_complete(
    *,
    completed_steps: Optional[int],
    elapsed_seconds: Optional[float],
    best_metric: Optional[float] = None,
    color: Optional[bool] = None,
) -> str:
    """Render a single run-completion summary."""
    fields = []
    if completed_steps is not None:
        fields.append(f"steps={completed_steps}")
    fields.append(f"elapsed={format_duration(elapsed_seconds)}")
    if best_metric is not None and math.isfinite(best_metric):
        fields.append(f"best_metric={best_metric:.4f}")
    return _format_line(
        "[train]",
        fields,
        color=color,
        tag_style="tag_complete",
        event="complete",
    )


def format_memory_snapshot(
    *,
    step: Optional[int],
    phase: Optional[str],
    cpu_percent: float,
    cpu_mb: float,
    gpu_percent: Optional[float] = None,
    gpu_mb: Optional[float] = None,
    gpu_peak_mb: Optional[float] = None,
    process_mb: Optional[float] = None,
    color: Optional[bool] = None,
) -> str:
    """Render memory usage with the same field-oriented training style."""
    fields = [f"step={step if step is not None else '--'}"]
    if phase:
        fields.append(f"phase={phase}")
    fields.append(f"cpu={cpu_percent:.1f}%/{cpu_mb:.0f}MB")
    if gpu_percent is not None and gpu_mb is not None:
        fields.append(f"gpu={gpu_percent:.1f}%/{gpu_mb:.0f}MB")
    if gpu_peak_mb is not None:
        fields.append(f"gpu_peak={gpu_peak_mb:.0f}MB")
    if process_mb is not None:
        fields.append(f"rss={process_mb:.0f}MB")
    return _format_line("[memory]", fields, color=color, tag_style="tag_memory")


def _estimate_eta(
    elapsed_seconds: Optional[float],
    completed_step: int,
    total_steps: Optional[int],
) -> Optional[float]:
    if elapsed_seconds is None or not total_steps or completed_step <= 0:
        return None
    remaining_steps = max(0, total_steps - completed_step)
    return elapsed_seconds / completed_step * remaining_steps


def _append_metric(
    fields: List[str],
    label: str,
    value: Any,
    spec: str,
    *,
    suffix: str = "",
) -> None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        fields.append(f"{label}={format(float(value), spec)}{suffix}")


def _format_line(
    tag: str,
    fields: List[str],
    *,
    color: Optional[bool],
    tag_style: str,
    event: Optional[str] = None,
) -> str:
    enabled = should_colorize_logs(color)
    prefix = _paint(tag, tag_style, enabled)
    if event:
        prefix = f"{prefix} {_paint(event, tag_style, enabled)}"
        return prefix + " | " + " | ".join(_format_field(field, enabled) for field in fields)
    return prefix + " " + " | ".join(_format_field(field, enabled) for field in fields)


def _format_field(field: str, enabled: bool) -> str:
    if not enabled or "=" not in field:
        return field

    key, value = field.split("=", 1)
    key_text = _paint(key, "key", enabled)
    value_style = _field_value_style(key)
    value_text = _paint(value, value_style, enabled) if value_style else value
    return f"{key_text}={value_text}"


def _field_value_style(key: str) -> Optional[str]:
    if key in {"task", "dataset"}:
        return "task"
    if key in {"loss", "train_loss", "val_loss"}:
        return "loss"
    if key == "lr":
        return "lr"
    if key == "grad":
        return "grad"
    if key in {"step_time", "eta", "elapsed"}:
        return "time"
    if key == "best_metric":
        return "good"
    if key.startswith(("cpu", "gpu", "rss")):
        return "memory"
    return None


def _paint(text: str, style: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_STYLES[style]}{text}{_RESET}"


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False
