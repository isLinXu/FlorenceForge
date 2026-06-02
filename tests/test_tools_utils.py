"""General tools utility regression tests."""

import hashlib

import pytest

from florence_forge.utils.tools import (
    ConfigManager,
    FileHasher,
    ProgressTracker,
    Timer,
    ensure_list,
    flatten_dict,
    retry_on_failure,
    suppress_warnings,
    timing_decorator,
    unflatten_dict,
)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


def test_timer_start_stop_returns_elapsed():
    timer = Timer("t").start()
    elapsed = timer.stop()
    assert elapsed >= 0.0
    assert timer.elapsed_time == elapsed


def test_timer_stop_without_start_raises():
    with pytest.raises(ValueError, match="尚未启动"):
        Timer().stop()


def test_timer_context_manager():
    with Timer("ctx") as timer:
        assert timer.start_time is not None
    assert timer.elapsed_time is not None


def test_timing_decorator_runs_function():
    @timing_decorator("custom")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


# ---------------------------------------------------------------------------
# FileHasher
# ---------------------------------------------------------------------------


def test_file_hasher_md5_and_sha256(tmp_path):
    path = tmp_path / "f.bin"
    payload = b"florence forge"
    path.write_bytes(payload)

    assert FileHasher.md5_hash(path) == hashlib.md5(payload).hexdigest()
    assert FileHasher.sha256_hash(path) == hashlib.sha256(payload).hexdigest()


def test_file_hasher_verify_integrity(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"data")
    md5 = hashlib.md5(b"data").hexdigest()

    assert FileHasher.verify_file_integrity(path, md5.upper(), "md5") is True
    assert FileHasher.verify_file_integrity(path, "deadbeef", "sha256") is False


def test_file_hasher_invalid_type(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="不支持的哈希类型"):
        FileHasher.verify_file_integrity(path, "x", "crc32")


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


def test_config_manager_missing_file_starts_empty(tmp_path):
    cm = ConfigManager(tmp_path / "missing.json")
    assert cm.to_dict() == {}


def test_config_manager_json_roundtrip(tmp_path):
    path = tmp_path / "cfg.json"
    cm = ConfigManager(path)
    cm.set("model.name", "florence")
    cm.set("model.lora.r", 16)
    cm.save_config()

    reloaded = ConfigManager(path)
    assert reloaded.get("model.name") == "florence"
    assert reloaded.get("model.lora.r") == 16


def test_config_manager_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    cm = ConfigManager(path)
    cm.update({"a.b": 1, "c": 2})
    cm.save_config()

    reloaded = ConfigManager(path)
    assert reloaded.get("a.b") == 1
    assert reloaded.get("c") == 2


def test_config_manager_pickle_roundtrip(tmp_path):
    path = tmp_path / "cfg.pkl"
    cm = ConfigManager(path)
    cm.set("x", [1, 2, 3])
    cm.save_config()

    reloaded = ConfigManager(path)
    assert reloaded.get("x") == [1, 2, 3]


def test_config_manager_get_default_on_missing_key(tmp_path):
    cm = ConfigManager(tmp_path / "cfg.json")
    cm.set("a", 1)
    assert cm.get("a.b.c", default="fallback") == "fallback"


def test_config_manager_unsupported_format_keeps_empty(tmp_path):
    path = tmp_path / "cfg.ini"
    path.write_text("[x]\n", encoding="utf-8")
    cm = ConfigManager(path)  # load failure is swallowed -> empty
    assert cm.to_dict() == {}


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


def test_progress_tracker_updates_to_completion(capsys):
    tracker = ProgressTracker(total=2, description="job")
    tracker.update()
    tracker.update()
    assert tracker.current == 2
    out = capsys.readouterr().out
    assert "job" in out


def test_progress_tracker_zero_total_is_noop():
    tracker = ProgressTracker(total=0)
    tracker.update()  # should not raise or divide by zero
    assert tracker.current == 1


# ---------------------------------------------------------------------------
# suppress_warnings / retry_on_failure
# ---------------------------------------------------------------------------


def test_suppress_warnings_context():
    import warnings

    with suppress_warnings():
        warnings.warn("should be suppressed")


def test_retry_on_failure_eventually_succeeds():
    calls = {"n": 0}

    @retry_on_failure(max_retries=3, delay=0.0, backoff=1.0, exceptions=(ValueError,))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_on_failure_reraises_after_max():
    @retry_on_failure(max_retries=2, delay=0.0, backoff=1.0, exceptions=(RuntimeError,))
    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        always_fails()


# ---------------------------------------------------------------------------
# dict helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [([1, 2], [1, 2]), (None, []), ("x", ["x"]), (5, [5])],
)
def test_ensure_list(value, expected):
    assert ensure_list(value) == expected


def test_flatten_and_unflatten_roundtrip():
    nested = {"model": {"name": "florence", "lora": {"r": 16}}, "lr": 0.001}
    flat = flatten_dict(nested)
    assert flat == {"model.name": "florence", "model.lora.r": 16, "lr": 0.001}
    assert unflatten_dict(flat) == nested
