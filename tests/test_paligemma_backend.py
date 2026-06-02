"""PaliGemmaBackend 单元测试

验证 PaliGemma 后端是否满足 BaseVLMBackend 接口契约。
使用 Mock 对象模拟 transformers 依赖，无需真实模型权重。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path

# 标记需要可选依赖的测试
pytestmark = [pytest.mark.integration]


class MockPaliGemmaModel(nn.Module):
    """模拟 PaliGemmaForConditionalGeneration"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, token_type_ids=None, **kwargs):
        batch_size = input_ids.shape[0]
        loss = torch.tensor(1.0 - 0.01 * batch_size, requires_grad=True)
        return MagicMock(loss=loss)

    def generate(self, **kwargs):
        return torch.randint(0, 100, (kwargs.get("input_ids").shape[0], 10))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


class MockProcessor:
    """模拟 PaliGemma Processor"""

    def __init__(self):
        pass

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        batch_size = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
            "token_type_ids": torch.zeros(batch_size, 10, dtype=torch.long),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["mock paligemma output"] * token_ids.shape[0]

    def save_pretrained(self, path):
        pass

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


@pytest.fixture
def mock_config():
    """创建 PaliGemma 测试配置"""
    from florence_forge.core.config import ModelConfig

    return ModelConfig(
        model_name="google/paligemma-3b-pt-224",
        backend_name="paligemma",
        use_lora=False,
        trust_remote_code=False,
    )


def _create_loaded_backend(config):
    """创建并加载 Mock PaliGemma 后端的辅助函数

    PaliGemmaBackend 采用延迟加载策略，构造函数不加载模型/处理器。
    需要在 patch 上下文中显式调用 backend.load() 来触发加载。
    """
    from florence_forge.core.backends.paligemma_backend import PaliGemmaBackend

    with patch(
        "florence_forge.core.backends.paligemma_backend.PaliGemmaForConditionalGeneration",
        MockPaliGemmaModel,
    ), patch(
        "florence_forge.core.backends.paligemma_backend.AutoProcessor",
        MockProcessor,
    ), patch.object(
        PaliGemmaBackend, "_get_optimal_device", return_value="cpu"
    ):
        backend = PaliGemmaBackend(config)
        backend.load()
    return backend


class TestPaliGemmaBackend:
    """PaliGemmaBackend 接口测试"""

    def test_backend_registered(self):
        """验证 PaliGemmaBackend 已注册到 Registry"""
        from florence_forge.core.backends import VLMBackendRegistry

        assert VLMBackendRegistry.is_registered("paligemma")
        assert VLMBackendRegistry.is_registered("paligemma-3b")

    def test_task_prompt_mapping(self, mock_config):
        """验证 PaliGemma 任务 prompt 映射"""
        from florence_forge.core.backends.paligemma_backend import PALIGEMMA_TASK_PROMPTS

        # 任务映射不需要加载模型，直接测试字典
        assert PALIGEMMA_TASK_PROMPTS.get("CAPTION") == "caption"
        assert PALIGEMMA_TASK_PROMPTS.get("OD") == "detect"
        assert PALIGEMMA_TASK_PROMPTS.get("OCR") == "ocr"
        assert PALIGEMMA_TASK_PROMPTS.get("VQA") == "answer"

        # 通过 backend 实例测试
        backend = _create_loaded_backend(mock_config)
        assert backend.get_task_prompt("CAPTION") == "caption"
        assert backend.get_task_prompt("OD") == "detect"
        assert backend.get_task_prompt("OCR") == "ocr"
        assert backend.get_task_prompt("VQA") == "answer"
        assert backend.supports_task("CAPTION")
        assert not backend.supports_task("UNKNOWN_TASK")

    def test_encode_decode(self, mock_config):
        """验证编码和解码流程"""
        backend = _create_loaded_backend(mock_config)

        # 编码
        inputs = backend.encode(
            images=[MagicMock()],
            text=["caption"],
        )
        assert "input_ids" in inputs
        assert "pixel_values" in inputs
        assert "attention_mask" in inputs
        assert "token_type_ids" in inputs

        # 解码
        token_ids = torch.randint(0, 100, (2, 10))
        texts = backend.decode(token_ids)
        assert len(texts) == 2
        assert texts[0] == "mock paligemma output"

    def test_forward_pass(self, mock_config):
        """验证前向传播"""
        backend = _create_loaded_backend(mock_config)

        input_ids = torch.randint(0, 100, (2, 10))
        pixel_values = torch.randn(2, 3, 224, 224)
        labels = torch.full((2, 10), -100)
        labels[:, 5:] = torch.randint(0, 100, (2, 5))

        outputs = backend.forward(input_ids, pixel_values, labels=labels)
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_generate(self, mock_config):
        """验证生成接口"""
        backend = _create_loaded_backend(mock_config)

        input_ids = torch.randint(0, 100, (1, 5))
        pixel_values = torch.randn(1, 3, 224, 224)

        generated = backend.generate(input_ids, pixel_values, max_new_tokens=10)
        assert generated.shape[0] == 1  # batch_size=1

    def test_get_model_info(self, mock_config):
        """验证模型信息查询"""
        backend = _create_loaded_backend(mock_config)
        info = backend.get_model_info()

        assert info["backend"] == "paligemma"
        assert info["model_name"] == "google/paligemma-3b-pt-224"
        assert "total_parameters" in info
        assert "trainable_parameters" in info

    def test_save_pretrained(self, mock_config, tmp_path):
        """验证保存功能"""
        backend = _create_loaded_backend(mock_config)
        save_path = tmp_path / "test_paligemma"
        backend.save_pretrained(str(save_path))
        assert save_path.exists()

    def test_model_property(self, mock_config):
        """验证 model 和 processor property"""
        backend = _create_loaded_backend(mock_config)
        assert backend.model is not None
        assert backend.processor is not None

    def test_backend_via_registry(self, mock_config):
        """验证通过 Registry 创建 PaliGemma 后端"""
        from florence_forge.core.backends.paligemma_backend import PaliGemmaBackend
        from florence_forge.core.backends import VLMBackendRegistry

        with patch(
            "florence_forge.core.backends.paligemma_backend.PaliGemmaForConditionalGeneration",
            MockPaliGemmaModel,
        ), patch(
            "florence_forge.core.backends.paligemma_backend.AutoProcessor",
            MockProcessor,
        ), patch.object(
            PaliGemmaBackend, "_get_optimal_device", return_value="cpu"
        ):
            backend = VLMBackendRegistry.create("paligemma", mock_config)
            backend.load()
            assert backend is not None
            info = backend.get_model_info()
            assert info["backend"] == "paligemma"

    def test_task_prompts_coverage(self):
        """验证 PaliGemma 任务提示映射覆盖主要任务"""
        from florence_forge.core.backends.paligemma_backend import PALIGEMMA_TASK_PROMPTS

        expected_tasks = [
            "CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION",
            "OD", "DENSE_CAPTION", "REGION_PROPOSAL",
            "OCR", "OCR_WITH_REGION", "VQA",
        ]
        for task in expected_tasks:
            assert task in PALIGEMMA_TASK_PROMPTS
