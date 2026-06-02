"""推理引擎安全加载测试。"""

import pytest
import torch

from florence_forge.deployment.inference import InferenceEngine


def test_local_state_dict_is_rejected_with_clear_error(tmp_path):
    model_path = tmp_path / "state_dict.pt"
    torch.save({"weight": torch.ones(1)}, model_path)

    with pytest.raises(TypeError, match="不是 nn.Module"):
        InferenceEngine(model_path, device="cpu")


def test_auto_device_handles_missing_mps_backend(monkeypatch):
    engine = InferenceEngine.__new__(InferenceEngine)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(torch.backends, "mps", raising=False)

    assert engine._setup_device("auto").type == "cpu"


def test_hf_directory_inference_load_disables_lora(monkeypatch, tmp_path):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    created_configs = []

    class DummyModelConfig:
        def __init__(self, **kwargs):
            created_configs.append(kwargs)

    class DummyFlorence2MultiTaskModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.loaded = False

        def load(self):
            self.loaded = True
            return self

    monkeypatch.setattr("florence_forge.core.model.ModelConfig", DummyModelConfig)
    monkeypatch.setattr(
        "florence_forge.core.model.Florence2MultiTaskModel",
        DummyFlorence2MultiTaskModel,
    )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine.compile_model = False
    engine.model_revision = None

    model = engine._load_model(str(model_dir))

    assert isinstance(model, DummyFlorence2MultiTaskModel)
    assert model.loaded is True
    assert created_configs == [
        {
            "model_name": str(model_dir),
            "device": "cpu",
            "use_lora": False,
        }
    ]


def test_hf_directory_inference_load_pins_model_revision(monkeypatch, tmp_path):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    created_configs = []

    class DummyModelConfig:
        def __init__(self, **kwargs):
            created_configs.append(kwargs)

    class DummyFlorence2MultiTaskModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.loaded = False

        def load(self):
            self.loaded = True
            return self

    monkeypatch.setattr("florence_forge.core.model.ModelConfig", DummyModelConfig)
    monkeypatch.setattr(
        "florence_forge.core.model.Florence2MultiTaskModel",
        DummyFlorence2MultiTaskModel,
    )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine.compile_model = False
    engine.model_revision = "abc123"

    model = engine._load_model(str(model_dir))

    assert isinstance(model, DummyFlorence2MultiTaskModel)
    assert model.loaded is True
    assert created_configs == [
        {
            "model_name": str(model_dir),
            "device": "cpu",
            "use_lora": False,
            "revision": "abc123",
        }
    ]


def test_lora_directory_inference_load_uses_adapter_revision(monkeypatch, tmp_path):
    model_dir = tmp_path / "lora_model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "base-model", "revision": "base-rev"}',
        encoding="utf-8",
    )
    created_configs = []
    load_calls = []

    class DummyModelConfig:
        def __init__(self, **kwargs):
            created_configs.append(kwargs)

    class DummyFlorence2MultiTaskModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        @classmethod
        def load_pretrained(cls, model_identifier, config, is_peft_model=False):
            load_calls.append({
                "model_identifier": model_identifier,
                "config": config,
                "is_peft_model": is_peft_model,
            })
            return cls(config)

    monkeypatch.setattr("florence_forge.core.model.ModelConfig", DummyModelConfig)
    monkeypatch.setattr(
        "florence_forge.core.model.Florence2MultiTaskModel",
        DummyFlorence2MultiTaskModel,
    )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine.compile_model = False
    engine.model_revision = None

    model = engine._load_model(str(model_dir))

    assert isinstance(model, DummyFlorence2MultiTaskModel)
    assert created_configs == [
        {
            "model_name": "base-model",
            "device": "cpu",
            "use_lora": False,
            "revision": "base-rev",
        }
    ]
    assert load_calls == [
        {
            "model_identifier": str(model_dir),
            "config": model.config,
            "is_peft_model": True,
        }
    ]
