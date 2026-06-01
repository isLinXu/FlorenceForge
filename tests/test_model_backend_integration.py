"""测试 Florence2MultiTaskModel 的 VLM 后端集成

验证 model.py 在后端可用和不可用两种路径下的行为一致性。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel


class TestModelBackendIntegration:
    def test_model_uses_backend_when_available(self):
        """当后端系统可用时，model 应委托给后端"""
        config = ModelConfig(model_name="dummy", backend_name="florence-2")

        mock_backend = MagicMock(spec=["model", "processor", "forward", "load", "get_model_info"])
        mock_backend.model = MagicMock()
        mock_backend.processor = MagicMock()
        mock_backend.forward.return_value = MagicMock(loss=torch.tensor(1.0))
        mock_backend.get_model_info.return_value = {
            "total_parameters": 1000,
            "trainable_parameters": 100,
            "trainable_ratio": 0.1,
        }

        with patch("florence_forge.core.model.VLMBackendRegistry") as MockRegistry:
            MockRegistry.is_registered.return_value = True
            MockRegistry.create.return_value = mock_backend

            # patch 掉 model.py 顶层的 transformers 和 peft 导入
            with patch("florence_forge.core.model.AutoProcessor"), \
                 patch("florence_forge.core.model.AutoModelForCausalLM"), \
                 patch("florence_forge.core.model.LoraConfig"), \
                 patch("florence_forge.core.model.get_peft_model"), \
                 patch("florence_forge.core.model.PeftModel"):
                from florence_forge.core.model import Florence2MultiTaskModel
                model = Florence2MultiTaskModel(config)
                assert model._backend is mock_backend
                # 测试 forward 委托
                out = model.forward(
                    input_ids=torch.tensor([[1, 2]]),
                    pixel_values=torch.randn(1, 3, 224, 224),
                )
                mock_backend.forward.assert_called_once()

    def test_model_fallback_when_backend_unavailable(self):
        """当后端系统不可用时，model 初始化应抛出 ValueError"""
        config = ModelConfig(model_name="dummy", backend_name="unknown")
        with patch("florence_forge.core.model.AutoProcessor"), \
             patch("florence_forge.core.model.AutoModelForCausalLM"), \
             patch("florence_forge.core.model.LoraConfig"), \
             patch("florence_forge.core.model.get_peft_model"), \
             patch("florence_forge.core.model.PeftModel"):
            from florence_forge.core.model import Florence2MultiTaskModel
            with pytest.raises(ValueError, match="未注册"):
                Florence2MultiTaskModel(config)

    def test_to_method(self):
        config = ModelConfig()
        model = Florence2MultiTaskModel.__new__(Florence2MultiTaskModel)
        model.config = config
        model._backend = MagicMock()
        model._backend._model = MagicMock()
        model._backend._device = "cpu"
        result = model.to("cpu")
        assert result is model

    def test_train_eval_modes(self):
        config = ModelConfig()
        model = Florence2MultiTaskModel.__new__(Florence2MultiTaskModel)
        model.config = config
        mock_model = MagicMock()
        model._backend = MagicMock()
        model._backend.model = mock_model
        model.train()
        mock_model.train.assert_called_once_with(True)
        model.eval()
        mock_model.eval.assert_called_once()

    def test_get_model_info_delegation(self):
        config = ModelConfig()
        model = Florence2MultiTaskModel.__new__(Florence2MultiTaskModel)
        model.config = config
        model.is_peft_model = False
        model._backend = MagicMock()
        model._backend.get_model_info.return_value = {
            "backend": "test",
            "is_peft_model": False,
            "total_parameters": 1000,
            "trainable_parameters": 100,
            "trainable_ratio": 0.1,
        }
        info = model.get_model_info()
        assert info["backend"] == "test"
