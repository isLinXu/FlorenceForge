"""Tests for LoRA manager wrapper compatibility."""

from types import SimpleNamespace

import torch
import torch.nn as nn

import florence_forge.training.lora_manager as lora_module
from florence_forge.core.config import LoRAConfig
from florence_forge.training.lora_manager import LoRAManager


class CaptureLoraConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_create_peft_config_uses_configured_task_type(monkeypatch):
    monkeypatch.setattr(lora_module, "LoraConfig", CaptureLoraConfig)
    monkeypatch.setattr(
        lora_module,
        "TaskType",
        SimpleNamespace(CAUSAL_LM="CAUSAL_LM", SEQ_2_SEQ_LM="SEQ_2_SEQ_LM"),
    )

    manager = LoRAManager(LoRAConfig(task_type="SEQ_2_SEQ_LM"))

    peft_config = manager.create_peft_config("CAPTION")

    assert peft_config.task_type == "SEQ_2_SEQ_LM"


def test_apply_and_switch_lora_use_wrapped_backend_model(monkeypatch):
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class DummyPeftModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.adapters = []
            self.active_adapter = None
            self.weight = nn.Parameter(torch.ones(1))

        def add_adapter(self, adapter_name, peft_config):
            self.adapters.append((adapter_name, peft_config))

        def set_adapter(self, adapter_name):
            self.active_adapter = adapter_name

    class ForgeWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self._backend = object()
            self._inner_model = BaseModel()

        @property
        def model(self):
            return self._inner_model

        @model.setter
        def model(self, value):
            self._inner_model = value

    seen = {}

    def fake_get_peft_model(model, peft_config, adapter_name=None):
        seen["model"] = model
        seen["adapter_name"] = adapter_name
        return DummyPeftModel(model)

    monkeypatch.setattr(lora_module, "LoraConfig", CaptureLoraConfig)
    monkeypatch.setattr(lora_module, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(lora_module, "get_memory_usage", lambda: {"gpu_allocated": 0})

    manager = LoRAManager(LoRAConfig())
    wrapper = ForgeWrapper()
    original_inner = wrapper.model

    returned = manager.apply_lora_to_model(wrapper, "CAPTION")

    assert returned is wrapper
    assert seen["model"] is original_inner
    assert seen["adapter_name"] == "lora_CAPTION"
    assert isinstance(wrapper.model, DummyPeftModel)

    manager.add_adapter_to_model(wrapper, "OD")
    manager.switch_adapter(wrapper, "OD")

    assert wrapper.model.adapters[0][0] == "lora_OD"
    assert wrapper.model.active_adapter == "lora_OD"


def test_apply_lora_reuses_existing_peft_wrapper(monkeypatch):
    class ExistingPeftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.peft_config = {"default": object()}
            self.active_adapter = "default"
            self.set_calls = []

        def set_adapter(self, adapter_name):
            self.set_calls.append(adapter_name)
            self.active_adapter = adapter_name

    class ForgeWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self._backend = object()
            self._inner_model = ExistingPeftModel()

        @property
        def model(self):
            return self._inner_model

        @model.setter
        def model(self, value):
            self._inner_model = value

    def fail_get_peft_model(*args, **kwargs):
        raise AssertionError("existing PEFT models should not be wrapped again")

    monkeypatch.setattr(lora_module, "LoraConfig", CaptureLoraConfig)
    monkeypatch.setattr(lora_module, "get_peft_model", fail_get_peft_model)
    monkeypatch.setattr(lora_module, "get_memory_usage", lambda: {"gpu_allocated": 0})

    manager = LoRAManager(LoRAConfig())
    wrapper = ForgeWrapper()

    returned = manager.apply_lora_to_model(wrapper, "OD")

    assert returned is wrapper
    assert wrapper.model.set_calls == ["default"]
    assert manager.active_adapters["OD"] == "default"
