"""Training console log formatting and integration tests."""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from florence_forge.core.callbacks import LoggingCallback
from florence_forge.core.config import TrainingConfig
from florence_forge.training.memory_monitor import MemoryMonitor, MemoryMonitorConfig, MemoryStats
from florence_forge.training.training_loop import TrainingLoop
from florence_forge.utils.training_logging import (
    format_duration,
    format_epoch_summary,
    format_memory_snapshot,
    format_training_complete,
    format_training_step,
    should_log_step,
    should_colorize_logs,
    strip_ansi_codes,
)


@pytest.fixture(autouse=True)
def disable_auto_training_colors(monkeypatch):
    monkeypatch.setenv("FLORENCE_FORGE_COLOR_LOGS", "0")


def test_training_message_helpers_render_compact_progress_and_eta():
    message = format_training_step(
        completed_step=2,
        total_steps=10,
        epoch=1,
        total_epochs=2,
        metrics={
            "loss": 0.123456,
            "learning_rate": 1e-5,
            "grad_norm": 0.75,
            "time_per_step": 0.4,
        },
        task_type="CAPTION",
        elapsed_seconds=10.0,
    )

    assert message == (
        "[train] epoch=1/2 | step=2/10 (20.0%) | task=CAPTION | "
        "loss=0.1235 | lr=1.00e-05 | grad=0.750 | step_time=0.40s | eta=00:40"
    )
    assert format_duration(3661) == "01:01:01"
    assert should_log_step(1, 100)
    assert should_log_step(100, 100)
    assert not should_log_step(2, 100)


def test_training_messages_can_be_colorized_and_stripped():
    message = format_training_step(
        completed_step=1,
        total_steps=4,
        metrics={"loss": 0.25, "learning_rate": 1e-4, "grad_norm": 0.5},
        task_type="OCR",
        elapsed_seconds=2.0,
        color=True,
    )
    plain = strip_ansi_codes(message)

    assert "\033[" in message
    assert plain == (
        "[train] step=1/4 (25.0%) | task=OCR | loss=0.2500 | "
        "lr=1.00e-04 | grad=0.500 | eta=00:06"
    )

    assert "\033[" in format_training_complete(
        completed_steps=4,
        elapsed_seconds=9,
        best_metric=0.9,
        color=True,
    )
    assert "\033[" in format_epoch_summary(
        epoch=1,
        train_metrics={"loss": 0.2},
        val_metrics={},
        color=True,
    )
    assert "\033[" in format_memory_snapshot(
        step=1,
        phase="after_optimizer",
        cpu_percent=10.0,
        cpu_mb=2048.0,
        color=True,
    )


def test_color_auto_respects_environment(monkeypatch):
    monkeypatch.delenv("FLORENCE_FORGE_COLOR_LOGS", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert should_colorize_logs() is False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("CI", "1")
    assert should_colorize_logs() is False

    monkeypatch.setenv("FLORENCE_FORGE_COLOR_LOGS", "1")
    assert should_colorize_logs() is True

    monkeypatch.setenv("FLORENCE_FORGE_COLOR_LOGS", "0")
    assert should_colorize_logs() is False


def test_epoch_and_completion_messages_keep_stable_fields():
    assert format_epoch_summary(
        epoch=1,
        total_epochs=3,
        train_metrics={"loss": 0.8, "learning_rate": 2e-5},
        val_metrics={"loss": 0.7},
    ) == "[epoch] epoch=1/3 | train_loss=0.8000 | val_loss=0.7000 | lr=2.00e-05"

    assert format_training_complete(
        completed_steps=12,
        elapsed_seconds=65,
        best_metric=0.7,
    ) == "[train] complete | steps=12 | elapsed=01:05 | best_metric=0.7000"


def test_logging_callback_adds_context_and_uses_human_step_numbers(caplog):
    callback = LoggingCallback(logging_steps=5)
    trainer = SimpleNamespace(
        config=SimpleNamespace(num_epochs=2, max_steps=None),
        train_dataloader=[object()] * 5,
        current_epoch=0,
        global_step=1,
        best_metric=0.5,
    )
    config = SimpleNamespace(
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=2,
    )

    with caplog.at_level(logging.INFO):
        callback.on_train_begin(trainer, config)
        callback.on_step_end(
            trainer,
            0,
            {"loss": 1.0, "learning_rate": 1e-4, "task_type": "OD"},
        )
        callback.on_train_end(trainer, config)

    output = caplog.text
    assert "[train] start | epochs=2 | batch_size=4 | grad_accum=2 | log_every=5 steps" in output
    assert "step=1/10 (10.0%) | task=OD | loss=1.0000 | lr=1.00e-04" in output
    assert "[train] complete | steps=1" in output


def test_logging_callback_suppresses_duplicate_worker_progress(caplog):
    callback = LoggingCallback(logging_steps=1)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(is_local_main_process=False),
        config=SimpleNamespace(num_epochs=1, max_steps=1),
        train_dataloader=[object()],
        current_epoch=0,
        global_step=1,
    )
    config = SimpleNamespace(num_epochs=1, batch_size=1, gradient_accumulation_steps=1)

    with caplog.at_level(logging.INFO):
        callback.on_train_begin(trainer, config)
        callback.on_step_end(trainer, 0, {"loss": 1.0})
        callback.on_train_end(trainer, config)

    assert "[train]" not in caplog.text


def test_memory_monitor_keeps_intermediate_phases_at_debug(caplog):
    monitor = MemoryMonitor.__new__(MemoryMonitor)
    monitor.config = MemoryMonitorConfig(
        enable_monitoring=True,
        log_frequency=1,
        enable_gpu_monitoring=False,
        auto_cleanup=False,
        save_stats=False,
        suggest_gradient_accumulation=False,
    )
    stats = MemoryStats(
        timestamp=1.0,
        cpu_memory_mb=512.0,
        cpu_memory_percent=20.0,
        process_memory_mb=100.0,
    )
    monitor.get_current_stats = lambda step=None, phase=None: stats
    monitor._check_memory_warnings = lambda current: []

    with caplog.at_level(logging.DEBUG):
        monitor.log_memory_usage(step=10, phase="before_forward")
        monitor.log_memory_usage(step=10, phase="after_optimizer")

    info_messages = [
        record.message for record in caplog.records if record.levelno == logging.INFO
    ]
    debug_messages = [
        record.message for record in caplog.records if record.levelno == logging.DEBUG
    ]
    assert any("phase=before_forward" in message for message in debug_messages)
    assert any("phase=after_optimizer" in message for message in info_messages)


def test_v2_training_loop_emits_compact_step_log(caplog):
    class LossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, labels=None, **kwargs):
            return SimpleNamespace(loss=self.weight * input_ids.sum())

    config = TrainingConfig(
        num_epochs=1,
        logging_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
    )
    model = LossModel()
    loop = TrainingLoop(model=model, config=config, accelerator=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    batches = [
        {
            "input_ids": torch.ones(1),
            "labels": torch.ones(1, dtype=torch.long),
            "task_type": "OCR",
        }
    ]

    with caplog.at_level(logging.INFO):
        loop.train_epoch(batches, optimizer, scheduler, epoch=0)

    assert "[train] epoch=1/1 | step=1/1 (100.0%) | task=OCR" in caplog.text
