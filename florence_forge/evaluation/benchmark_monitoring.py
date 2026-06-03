"""Benchmark runtime monitoring helpers."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import torch

try:
    import GPUtil
except ImportError:  # pragma: no cover - optional dependency
    GPUtil = None

logger = logging.getLogger(__name__)


def make_monitoring_data(history_size: int) -> Dict[str, Any]:
    """Create the mutable monitoring data structure used by benchmark runs."""
    maxlen = max(0, int(history_size or 0)) or None
    return {
        "start_time": None,
        "current_progress": 0,
        "estimated_time": None,
        "resource_usage": deque(maxlen=maxlen),
        "performance_metrics": deque(maxlen=maxlen),
    }


def monitoring_data_snapshot(monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of benchmark monitoring data."""
    return {
        **monitoring_data,
        "resource_usage": list(monitoring_data.get("resource_usage", [])),
        "performance_metrics": list(monitoring_data.get("performance_metrics", [])),
    }


def collect_resource_usage(cpu_interval: float = 1.0) -> Dict[str, Any]:
    """Collect one CPU, memory, and optional GPU usage sample."""
    cpu_percent = psutil.cpu_percent(interval=cpu_interval)
    memory = psutil.virtual_memory()

    gpu_info = []
    if torch.cuda.is_available() and GPUtil is not None:
        try:
            for gpu in GPUtil.getGPUs():
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature,
                    }
                )
        except Exception as exc:  # pragma: no cover - depends on host GPU tooling
            logger.debug("GPU usage sampling failed: %s", exc)

    return {
        "timestamp": time.time(),
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "gpu_info": gpu_info,
    }


def get_real_time_status(
    monitoring_data: Dict[str, Any],
    *,
    enabled: bool = True,
    is_running: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a real-time status payload from monitoring state."""
    if not enabled:
        return {"error": "监控未启用"}

    current_time = time.time()
    start_time = monitoring_data.get("start_time")
    running = start_time is not None if is_running is None else is_running
    status = {
        "is_running": running,
        "progress": monitoring_data.get("current_progress", 0),
        "elapsed_time": current_time - start_time if start_time else 0,
        "estimated_remaining_time": None,
        "current_resource_usage": None,
    }

    progress = status["progress"]
    if 0 < progress < 1 and status["elapsed_time"] > 0:
        estimated_total = status["elapsed_time"] / progress
        status["estimated_remaining_time"] = estimated_total - status["elapsed_time"]

    resource_usage = monitoring_data.get("resource_usage", [])
    if resource_usage:
        status["current_resource_usage"] = resource_usage[-1]

    return status


def export_monitoring_data(
    monitoring_data: Dict[str, Any],
    output_file: Union[str, Path],
) -> bool:
    """Export monitoring data to JSON. Returns False when no data exists."""
    if not monitoring_data.get("resource_usage"):
        logger.warning("无监控数据可导出")
        return False

    output_path = Path(output_file)
    snapshot = monitoring_data_snapshot(monitoring_data)
    start_time = snapshot.get("start_time")
    monitoring_export = {
        "metadata": {
            "start_time": start_time,
            "total_duration": time.time() - start_time if start_time else 0,
            "data_points": len(snapshot["resource_usage"]),
        },
        "resource_usage": snapshot["resource_usage"],
        "performance_metrics": snapshot.get("performance_metrics", []),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(monitoring_export, f, indent=2, ensure_ascii=False, default=str)

    logger.info("监控数据已导出: %s", output_path)
    return True


class BenchmarkMonitor:
    """Owns benchmark resource monitoring state and background sampling."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        history_size: int = 100,
        sample_interval: float = 5.0,
        cpu_interval: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.cpu_interval = cpu_interval
        self.data = make_monitoring_data(history_size)
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> Optional[threading.Thread]:
        """Start background resource monitoring when enabled."""
        if not self.enabled:
            return None
        if self.is_running:
            return self.thread

        self.data["start_time"] = time.time()
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.thread.start()
        return self.thread

    def stop(self, timeout: float = 1.0) -> None:
        """Stop background monitoring without disabling future runs."""
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def snapshot(self) -> Dict[str, Any]:
        return monitoring_data_snapshot(self.data)

    def status(self) -> Dict[str, Any]:
        return get_real_time_status(
            self.data,
            enabled=self.enabled,
            is_running=self.is_running,
        )

    def export(self, output_file: Union[str, Path]) -> bool:
        return export_monitoring_data(self.data, output_file)

    def _monitor_resources(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.data["resource_usage"].append(
                    collect_resource_usage(cpu_interval=self.cpu_interval)
                )
            except Exception as exc:
                logger.warning("监控线程错误: %s", exc)

            self._stop_event.wait(self.sample_interval)
