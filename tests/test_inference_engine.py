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
