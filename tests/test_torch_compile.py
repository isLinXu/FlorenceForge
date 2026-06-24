"""torch.compile helper tests."""

import logging

import torch.nn as nn

import florence_forge.utils.torch_compile as compile_mod
from florence_forge.utils.torch_compile import compile_module_if_requested


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def test_compile_helper_noops_when_disabled(monkeypatch):
    model = TinyModel()

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("compile should not be called")

    monkeypatch.setattr(compile_mod.torch, "compile", unexpected_compile, raising=False)

    assert compile_module_if_requested(model, enabled=False) is model


def test_compile_helper_passes_supported_options(monkeypatch):
    model = TinyModel()
    calls = []

    def fake_compile(compiled_model, **kwargs):
        calls.append((compiled_model, kwargs))
        return compiled_model

    monkeypatch.setattr(compile_mod.torch, "compile", fake_compile, raising=False)

    result = compile_module_if_requested(
        model,
        enabled=True,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True,
        backend="inductor",
        context="test model",
    )

    assert result is model
    assert calls == [
        (
            model,
            {
                "fullgraph": True,
                "mode": "reduce-overhead",
                "dynamic": True,
                "backend": "inductor",
            },
        )
    ]


def test_compile_helper_falls_back_on_compile_error(monkeypatch, caplog):
    model = TinyModel()

    def fake_compile(*args, **kwargs):
        raise RuntimeError("unsupported graph")

    monkeypatch.setattr(compile_mod.torch, "compile", fake_compile, raising=False)

    with caplog.at_level(logging.WARNING):
        result = compile_module_if_requested(model, enabled=True, context="test model")

    assert result is model
    assert "编译失败" in caplog.text


def test_compile_helper_falls_back_when_torch_compile_missing(monkeypatch, caplog):
    model = TinyModel()
    monkeypatch.delattr(compile_mod.torch, "compile", raising=False)

    with caplog.at_level(logging.WARNING):
        result = compile_module_if_requested(model, enabled=True, context="test model")

    assert result is model
    assert "torch.compile 不可用" in caplog.text
