"""测试 VLM 后端抽象层

覆盖 BaseVLMBackend 接口契约、Florence2Backend 实现、
VLMBackendRegistry 注册与发现机制。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from florence_forge.core.backends.base_vlm import (
    BaseVLMBackend,
    VLMBackendRegistry,
    _is_cpu_fallback_candidate,
    _patch_transformers_config_defaults,
)


class DummyBackend(BaseVLMBackend):
    """用于测试的最小后端实现"""

    def load_model(self):
        self._model = MagicMock()

    def load_processor(self):
        self._processor = MagicMock()

    def encode(self, images, text, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

    def generate(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        return torch.tensor([[4, 5, 6]])

    def decode(self, token_ids, skip_special_tokens=True):
        return ["hello world"]

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, **kwargs):
        out = MagicMock()
        out.loss = torch.tensor(1.0)
        return out

    def save_pretrained(self, save_directory):
        pass

    def load_pretrained(self, model_path, **kwargs):
        pass

    def get_model_info(self):
        return {"backend": "dummy"}

    def get_task_prompt(self, task_name):
        return f"<{task_name}>"

    def supports_task(self, task_name):
        return True


@pytest.fixture(autouse=True)
def reset_registry():
    """每个测试前清理注册表"""
    original = dict(VLMBackendRegistry._backends)
    VLMBackendRegistry._backends.clear()
    yield
    VLMBackendRegistry._backends.clear()
    VLMBackendRegistry._backends.update(original)


class TestVLMBackendRegistry:
    def test_register_and_create(self):
        VLMBackendRegistry.register("dummy", DummyBackend)
        config = MagicMock()
        backend = VLMBackendRegistry.create("dummy", config)
        assert isinstance(backend, DummyBackend)

    def test_register_invalid_class(self):
        with pytest.raises(TypeError):
            VLMBackendRegistry.register("bad", str)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="未知后端"):
            VLMBackendRegistry.create("nonexistent", MagicMock())

    def test_list_backends(self):
        VLMBackendRegistry.register("dummy1", DummyBackend)
        VLMBackendRegistry.register("dummy2", DummyBackend)
        assert sorted(VLMBackendRegistry.list_backends()) == ["dummy1", "dummy2"]

    def test_is_registered(self):
        VLMBackendRegistry.register("dummy", DummyBackend)
        assert VLMBackendRegistry.is_registered("dummy") is True
        assert VLMBackendRegistry.is_registered("DUMMY") is True  # 大小写不敏感
        assert VLMBackendRegistry.is_registered("other") is False


class TestDummyBackend:
    def test_init_and_load(self):
        config = MagicMock()
        backend = DummyBackend(config)
        # BaseVLMBackend.__init__ 不会自动调用 load_model
        backend.load_model()
        backend.load_processor()
        assert backend.model is not None
        assert backend.processor is not None

    def test_encode_returns_tensors(self):
        backend = DummyBackend(MagicMock())
        result = backend.encode(images=[], text="test")
        assert "input_ids" in result
        assert "pixel_values" in result

    def test_generate_returns_tensor(self):
        backend = DummyBackend(MagicMock())
        out = backend.generate(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.randn(1, 3, 224, 224),
        )
        assert isinstance(out, torch.Tensor)

    def test_decode_returns_list(self):
        backend = DummyBackend(MagicMock())
        texts = backend.decode(torch.tensor([[1, 2]]))
        assert isinstance(texts, list)
        assert len(texts) == 1

    def test_forward_returns_loss(self):
        backend = DummyBackend(MagicMock())
        out = backend.forward(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.randn(1, 3, 224, 224),
        )
        assert hasattr(out, "loss")

    def test_get_model_info(self):
        backend = DummyBackend(MagicMock())
        info = backend.get_model_info()
        assert info["backend"] == "dummy"

    def test_task_methods(self):
        backend = DummyBackend(MagicMock())
        assert backend.get_task_prompt("OD") == "<OD>"
        assert backend.supports_task("ANY") is True

    def test_patch_transformers_config_defaults_sets_forced_bos_token_id(self):
        from transformers import PretrainedConfig

        if hasattr(PretrainedConfig, "forced_bos_token_id"):
            delattr(PretrainedConfig, "forced_bos_token_id")

        _patch_transformers_config_defaults()

        assert hasattr(PretrainedConfig, "forced_bos_token_id")
        assert PretrainedConfig.forced_bos_token_id is None

    def test_cpu_fallback_only_for_device_like_errors(self):
        assert _is_cpu_fallback_candidate(RuntimeError("CUDA out of memory"))
        assert _is_cpu_fallback_candidate(RuntimeError("bfloat16 not supported on this device"))
        assert not _is_cpu_fallback_candidate(RuntimeError("Read timed out while downloading model"))
        assert not _is_cpu_fallback_candidate(AttributeError("forced_bos_token_id missing"))


class TestFlorence2BackendRegistration:
    def test_florence2_registered(self, reset_registry):
        """验证 Florence2Backend 可被手动注册并在导入路径中存在"""
        from florence_forge.core.backends.florence2_backend import Florence2Backend
        from florence_forge.core.backends.base_vlm import VLMBackendRegistry
        # 手动注册（因为 fixture 清除了全局注册表）
        VLMBackendRegistry.register("florence-2", Florence2Backend)
        VLMBackendRegistry.register("florence2", Florence2Backend)
        assert VLMBackendRegistry.is_registered("florence-2")
        assert VLMBackendRegistry.is_registered("florence2")
