"""Unit tests for reusable application logging helpers."""

import builtins
import io
import logging
import sys
from types import SimpleNamespace

import pytest

from florence_forge.utils.logging import (
    LoggerMixin,
    ProgressLogger,
    create_experiment_logger,
    get_logger,
    log_function_call,
    log_memory_usage,
    setup_logging,
)


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    library_levels = {
        name: logging.getLogger(name).level
        for name in ("transformers", "torch", "PIL", "matplotlib")
    }
    yield
    for handler in root.handlers[:]:
        if handler not in original_handlers:
            handler.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)
    for name, level in library_levels.items():
        logging.getLogger(name).setLevel(level)


def _stream_logger(name: str):
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers[:] = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(stream))
    return logger, stream


def test_setup_logging_writes_console_and_file_without_duplicate_handlers(
    tmp_path, capsys, restore_logging
):
    log_file = tmp_path / "logs" / "training.log"

    setup_logging(
        level="INFO",
        log_file=log_file,
        include_timestamp=False,
        include_name=False,
        force=True,
    )
    setup_logging(level="INFO", log_file=log_file)
    get_logger("training").info("visible progress")

    assert "visible progress" in capsys.readouterr().out
    assert "visible progress" in log_file.read_text(encoding="utf-8")
    root = logging.getLogger()
    assert sum(isinstance(h, logging.FileHandler) for h in root.handlers) == 1
    assert logging.getLogger("transformers").level == logging.WARNING


def test_create_experiment_logger_and_mixin_methods(tmp_path):
    experiment = create_experiment_logger("caption", tmp_path, force=True)
    experiment.propagate = False
    experiment.info("epoch finished")

    log_files = list(tmp_path.glob("caption_*.log"))
    assert len(log_files) == 1
    assert "epoch finished" in log_files[0].read_text(encoding="utf-8")

    for handler in experiment.handlers[:]:
        experiment.removeHandler(handler)
        handler.close()

    class TrainComponent(LoggerMixin):
        pass

    component = TrainComponent()
    logger, stream = _stream_logger(component.logger.name)
    component._logger = logger
    component.log_debug("debug")
    component.log_info("info")
    component.log_warning("warn")
    component.log_error("error")

    assert component.logger.name.endswith("TrainComponent")
    assert stream.getvalue().splitlines() == ["debug", "info", "warn", "error"]


def test_progress_logger_reports_interval_eta_and_finish():
    logger, stream = _stream_logger("test.progress")
    progress = ProgressLogger(logger, total_steps=2, log_interval=1)

    progress.update(step=1, phase="warmup")
    progress.start()
    progress.update(step=1, loss="0.5000")
    progress.update()
    progress.finish()
    ProgressLogger(logger, total_steps=1).finish()

    output = stream.getvalue()
    assert "进度: 1/2 (50.0%), phase=warmup" in output
    assert "开始任务，总步数: 2" in output
    assert "loss=0.5000" in output
    assert "剩余时间:" in output
    assert "进度: 2/2 (100.0%)" in output
    assert "任务完成，总耗时:" in output
    assert "任务完成\n" in output


def test_log_function_call_records_success_and_error():
    logger, stream = _stream_logger(__name__)

    @log_function_call
    def successful(value):
        return value + 1

    @log_function_call
    def failing():
        raise RuntimeError("broken")

    assert successful(2) == 3
    with pytest.raises(RuntimeError, match="broken"):
        failing()

    output = stream.getvalue()
    assert "调用函数: successful" in output
    assert "函数 successful 执行完成" in output
    assert "函数 failing 执行失败" in output

    logger.handlers[:] = []


def test_log_memory_usage_includes_system_and_cuda_metrics(monkeypatch):
    logger, stream = _stream_logger("test.memory.gpu")
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(percent=25.0, used=2 * 1024**3, total=8 * 1024**3)
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        memory_allocated=lambda index: 1 * 1024**3,
        memory_reserved=lambda index: 2 * 1024**3,
        get_device_properties=lambda index: SimpleNamespace(total_memory=8 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    log_memory_usage(logger)

    output = stream.getvalue()
    assert "系统内存使用: 25.0% (2.0GB / 8.0GB)" in output
    assert "GPU 0 内存使用: 已分配 1.0GB" in output


def test_log_memory_usage_handles_missing_dependency_and_runtime_error(monkeypatch):
    logger, stream = _stream_logger("test.memory.error")
    original_import = builtins.__import__

    def missing_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_psutil)
    log_memory_usage(logger)
    assert "psutil未安装" in stream.getvalue()

    monkeypatch.setattr(builtins, "__import__", original_import)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: (_ for _ in ()).throw(RuntimeError("probe failed"))),
    )
    log_memory_usage(logger)
    assert "获取内存使用信息失败: probe failed" in stream.getvalue()
