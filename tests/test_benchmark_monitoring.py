"""Benchmark monitoring helper regression tests."""

import json
import time
from collections import deque

from florence_forge.evaluation.benchmark_monitoring import (
    BenchmarkMonitor,
    export_monitoring_data,
    get_real_time_status,
    make_monitoring_data,
    monitoring_data_snapshot,
)


def test_make_monitoring_data_bounds_resource_history():
    data = make_monitoring_data(history_size=1)
    data["resource_usage"].append({"cpu_percent": 10.0})
    data["resource_usage"].append({"cpu_percent": 20.0})

    assert list(data["resource_usage"]) == [{"cpu_percent": 20.0}]


def test_monitoring_data_snapshot_converts_deques_to_lists():
    snapshot = monitoring_data_snapshot(
        {
            "start_time": 1.0,
            "current_progress": 0.5,
            "estimated_time": None,
            "resource_usage": deque([{"cpu_percent": 10.0}], maxlen=1),
            "performance_metrics": deque([{"latency": 0.1}], maxlen=1),
        }
    )

    assert snapshot["resource_usage"] == [{"cpu_percent": 10.0}]
    assert snapshot["performance_metrics"] == [{"latency": 0.1}]


def test_get_real_time_status_uses_explicit_running_state():
    status = get_real_time_status(
        {
            "start_time": time.time() - 10,
            "current_progress": 0.5,
            "resource_usage": deque([{"cpu_percent": 10.0}], maxlen=1),
        },
        enabled=True,
        is_running=False,
    )

    assert status["is_running"] is False
    assert status["estimated_remaining_time"] is not None
    assert status["current_resource_usage"] == {"cpu_percent": 10.0}


def test_export_monitoring_data_writes_json(tmp_path):
    data = make_monitoring_data(history_size=2)
    data["start_time"] = time.time() - 1
    data["resource_usage"].append({"cpu_percent": 10.0})
    data["performance_metrics"].append({"latency": 0.1})
    output_file = tmp_path / "monitoring.json"

    assert export_monitoring_data(data, output_file) is True

    exported = json.loads(output_file.read_text(encoding="utf-8"))
    assert exported["metadata"]["data_points"] == 1
    assert exported["resource_usage"] == [{"cpu_percent": 10.0}]
    assert exported["performance_metrics"] == [{"latency": 0.1}]


def test_export_monitoring_data_returns_false_without_samples(tmp_path):
    data = make_monitoring_data(history_size=2)

    assert export_monitoring_data(data, tmp_path / "empty.json") is False


def test_disabled_monitor_does_not_start_thread():
    monitor = BenchmarkMonitor(enabled=False, sample_interval=0.01, cpu_interval=0)

    assert monitor.start() is None
    assert monitor.thread is None
    assert monitor.status() == {"error": "监控未启用"}
