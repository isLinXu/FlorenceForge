"""Tests for the extensible task registry (register_task / unregister_task)."""

from __future__ import annotations

import pytest

from florence_forge.core.tasks import (
    TaskCategory,
    TaskConfig,
    TaskOutputType,
    get_task_config_typed,
    is_tvp_task,
    register_task,
    unregister_task,
    validate_task_name,
)
import florence_forge.core.tasks as tasks_mod


@pytest.fixture
def cleanup_task():
    registered: list[str] = []
    yield registered
    for name in registered:
        if validate_task_name(name):
            unregister_task(name)


def test_register_task_with_kwargs(cleanup_task):
    cleanup_task.append("MY_CUSTOM_TASK")
    cfg = register_task(
        "MY_CUSTOM_TASK",
        prompt="<MY_CUSTOM>",
        category=TaskCategory.IMAGE_CAPTIONING,
        description="custom task",
    )
    assert isinstance(cfg, TaskConfig)
    assert validate_task_name("MY_CUSTOM_TASK")
    assert get_task_config_typed("MY_CUSTOM_TASK").prompt == "<MY_CUSTOM>"


def test_register_task_with_config_object(cleanup_task):
    cleanup_task.append("CFG_TASK")
    cfg = TaskConfig(
        prompt="<CFG>",
        category=TaskCategory.OBJECT_DETECTION,
        description="via config object",
        output_type=TaskOutputType.STRUCTURED,
    )
    register_task("CFG_TASK", cfg)
    assert get_task_config_typed("CFG_TASK") is cfg


def test_register_duplicate_requires_overwrite(cleanup_task):
    cleanup_task.append("DUP_TASK")
    register_task("DUP_TASK", prompt="<A>", category=TaskCategory.TEXT_RECOGNITION,
                  description="first")
    with pytest.raises(ValueError):
        register_task("DUP_TASK", prompt="<B>", category=TaskCategory.TEXT_RECOGNITION,
                      description="second")
    # overwrite=True succeeds
    register_task("DUP_TASK", prompt="<B>", category=TaskCategory.TEXT_RECOGNITION,
                  description="second", overwrite=True)
    assert get_task_config_typed("DUP_TASK").prompt == "<B>"


def test_register_cannot_overwrite_builtin_without_flag():
    with pytest.raises(ValueError):
        register_task("CAPTION", prompt="<X>", category=TaskCategory.IMAGE_CAPTIONING,
                      description="hijack")


def test_unregister_unknown_raises():
    with pytest.raises(KeyError):
        unregister_task("NOT_A_REAL_TASK_XYZ")


def test_derived_tvp_names_refresh(cleanup_task):
    cleanup_task.append("CUSTOM_TVP")
    register_task(
        "CUSTOM_TVP",
        prompt="<COUNT>",
        category=TaskCategory.OBJECT_DETECTION,
        description="custom tvp",
        is_tvp=True,
    )
    assert is_tvp_task("CUSTOM_TVP")
    # module-level derived tuple reflects the new task
    assert "CUSTOM_TVP" in tasks_mod.TVP_TASK_NAMES
    unregister_task("CUSTOM_TVP")
    cleanup_task.clear()
    assert "CUSTOM_TVP" not in tasks_mod.TVP_TASK_NAMES


def test_invalid_name_rejected():
    with pytest.raises(ValueError):
        register_task("", prompt="<X>", category=TaskCategory.IMAGE_CAPTIONING,
                      description="empty")
