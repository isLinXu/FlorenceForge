"""可选依赖提示文案测试。"""

import builtins

import pytest
import torch

from florence_forge.deployment.exporter import ModelExporter
from florence_forge.utils.optional_dependencies import (
    format_install_hint,
    missing_dependency_message,
)


def test_format_install_hint_with_extra():
    hint = format_install_hint("transformers", "evaluation")
    assert 'pip install -e ".[evaluation]"' in hint


def test_missing_dependency_message_is_consistent():
    message = missing_dependency_message("可视化功能", "matplotlib")
    assert message.startswith("可视化功能需要 matplotlib")
    assert "请安装 `matplotlib`" in message


@pytest.mark.parametrize(
    ("module_name", "feature", "package_name"),
    [
        ("tensorrt", "TensorRT导出", "tensorrt"),
        ("coremltools", "Core ML导出", "coremltools"),
    ],
)
def test_exporter_raises_consistent_import_error(monkeypatch, module_name, feature, package_name):
    model = torch.nn.Linear(2, 2)
    exporter = ModelExporter(model)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"missing {module_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    expected = missing_dependency_message(feature, package_name)
    if module_name == "tensorrt":
        with pytest.raises(ImportError, match=expected):
            exporter.export_tensorrt("dummy.trt")
    else:
        with pytest.raises(ImportError, match=expected):
            exporter.export_coreml("dummy.mlmodel", torch.randn(1, 2))
