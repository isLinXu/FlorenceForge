"""GenericHFBackend 单元测试

测试通用 HuggingFace VLM 后端的各项功能。
由于 GenericHFBackend 依赖 transformers，测试使用 Mock 对象模拟。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from florence_forge.core.backends.base_vlm import VLMBackendRegistry


class MockHFLM(nn.Module):
    """模拟 HuggingFace 语言模型"""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(10, 10))

    def forward(self, **kwargs):
        output = MagicMock()
        output.loss = torch.tensor(1.5) if "labels" in kwargs else None
        return output

    def generate(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return torch.randint(0, 100, (batch_size, 10))

    def save_pretrained(self, path):
        pass


class MockHFProcessor:
    """模拟 HuggingFace Processor"""

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        batch_size = 1 if isinstance(text, str) else len(text)
        return {
            "input_ids": torch.randint(1, 100, (batch_size, 8)),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        if token_ids.dim() == 1:
            return ["decoded text"]
        return ["decoded text"] * token_ids.shape[0]

    def save_pretrained(self, path):
        pass


class MockConfig:
    """模拟 ModelConfig"""

    def __init__(self, model_name="test/model", backend_name="generic-hf",
                 architecture_type="auto", task_prompts=None, supported_tasks=None):
        self.model_name = model_name
        self.backend_name = backend_name
        self.architecture_type = architecture_type
        self.task_prompts = task_prompts or {}
        self.supported_tasks = supported_tasks
        self.trust_remote_code = True
        self.device = "cpu"
        self.use_bf16 = False
        self.use_fp16 = False


@pytest.fixture
def mock_hf_backend():
    """创建模拟的 GenericHFBackend"""
    from florence_forge.core.backends.generic_hf_backend import GenericHFBackend

    config = MockConfig()
    backend = object.__new__(GenericHFBackend)
    nn.Module.__init__(backend)
    backend.config = config
    backend._model = MockHFLM()
    backend._processor = MockHFProcessor()
    backend._tokenizer = None
    backend._image_processor = None
    backend._device = "cpu"
    backend._dtype = torch.float32
    backend.is_peft_model = False
    backend._architecture_type = "causal_lm"
    backend._task_prompts = {
        "CAPTION": "Describe this image.",
        "OD": "Detect all objects.",
        "CUSTOM": "custom prompt",
    }
    backend._supports_tasks = ["CAPTION", "OD", "CUSTOM"]
    return backend


class TestGenericHFBackend:
    """GenericHFBackend 测试类"""

    def test_backend_registered(self):
        """验证 GenericHFBackend 已注册"""
        assert VLMBackendRegistry.is_registered("generic-hf")
        assert VLMBackendRegistry.is_registered("auto")
        assert VLMBackendRegistry.is_registered("hf")

    def test_get_task_prompt(self, mock_hf_backend):
        """测试任务 prompt 获取"""
        assert mock_hf_backend.get_task_prompt("CAPTION") == "Describe this image."
        assert mock_hf_backend.get_task_prompt("OD") == "Detect all objects."
        assert mock_hf_backend.get_task_prompt("UNKNOWN") == "UNKNOWN"

    def test_supports_task(self, mock_hf_backend):
        """测试任务支持检查"""
        assert mock_hf_backend.supports_task("CAPTION") is True
        assert mock_hf_backend.supports_task("OD") is True
        assert mock_hf_backend.supports_task("UNKNOWN") is False

    def test_get_supported_tasks(self, mock_hf_backend):
        """测试获取支持的任务列表"""
        tasks = mock_hf_backend.get_supported_tasks()
        assert "CAPTION" in tasks
        assert "OD" in tasks
        assert "CUSTOM" in tasks

    def test_set_task_prompt(self, mock_hf_backend):
        """测试动态设置任务 prompt"""
        mock_hf_backend.set_task_prompt("NEW_TASK", "new prompt")
        assert mock_hf_backend.get_task_prompt("NEW_TASK") == "new prompt"
        assert "NEW_TASK" in mock_hf_backend.get_supported_tasks()

    def test_encode(self, mock_hf_backend):
        """测试编码"""
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = mock_hf_backend.encode(
            images=[image],
            text="Describe this image."
        )
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs

    def test_encode_with_task(self, mock_hf_backend):
        """测试使用任务名称编码"""
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = mock_hf_backend.encode_with_task(
            images=[image],
            task_name="CAPTION",
        )
        assert "input_ids" in inputs
        assert "pixel_values" in inputs

    def test_generate(self, mock_hf_backend):
        """测试生成"""
        input_ids = torch.randint(0, 100, (2, 5))
        pixel_values = torch.randn(2, 3, 224, 224)

        generated = mock_hf_backend.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=10,
        )
        assert generated.shape[0] == 2

    def test_decode(self, mock_hf_backend):
        """测试解码"""
        token_ids = torch.randint(0, 100, (2, 10))
        texts = mock_hf_backend.decode(token_ids)
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_forward(self, mock_hf_backend):
        """测试前向传播"""
        input_ids = torch.randint(0, 100, (2, 8))
        pixel_values = torch.randn(2, 3, 224, 224)
        labels = torch.full((2, 8), -100)
        labels[:, 4:] = torch.randint(0, 100, (2, 4))

        outputs = mock_hf_backend.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_get_model_info(self, mock_hf_backend):
        """测试模型信息"""
        info = mock_hf_backend.get_model_info()
        assert info["backend"] == "generic-hf"
        assert info["architecture_type"] == "causal_lm"
        assert "total_parameters" in info
        assert "trainable_parameters" in info

    def test_prepare_labels(self, mock_hf_backend):
        """测试 labels 构建"""
        prompt_ids = torch.tensor([1, 2, 3])
        full_ids = torch.tensor([1, 2, 3, 4, 5])

        encoded_prompt = {"input_ids": prompt_ids}
        encoded_full = {"input_ids": full_ids}

        labels = mock_hf_backend.prepare_labels(encoded_prompt, encoded_full)
        assert labels[0].item() == -100
        assert labels[1].item() == -100
        assert labels[2].item() == -100
        assert labels[3].item() == 4
        assert labels[4].item() == 5

    def test_architecture_type_property(self, mock_hf_backend):
        """验证架构类型属性"""
        assert mock_hf_backend.ARCHITECTURE_TYPE == "auto"
        assert mock_hf_backend._architecture_type == "causal_lm"


class TestAutoSelectBackend:
    """自动后端选择测试"""

    def test_auto_select_florence(self):
        """测试自动选择 Florence-2 后端"""
        from florence_forge.core.backends import auto_select_backend
        from florence_forge.core.backends.florence2_backend import Florence2Backend

        # florence-2 后端已注册，auto_select_backend 直接创建实例
        config = MockConfig(model_name="microsoft/florence-2-large", backend_name="florence-2")
        backend = auto_select_backend(config)
        assert isinstance(backend, Florence2Backend)

    def test_auto_select_generic(self):
        """测试自动选择 GenericHFBackend"""
        from florence_forge.core.backends import auto_select_backend
        from florence_forge.core.backends.generic_hf_backend import GenericHFBackend

        # backend_name="auto" 已注册为 GenericHFBackend
        config = MockConfig(model_name="unknown/custom-model", backend_name="auto")
        backend = auto_select_backend(config)
        assert isinstance(backend, GenericHFBackend)


class TestGenericHFBackendGuessArchitecture:
    """架构推断测试"""

    def test_guess_florence(self):
        """测试推断 Florence 架构"""
        from florence_forge.core.backends.generic_hf_backend import _guess_architecture_type
        assert _guess_architecture_type("microsoft/florence-2-large") == "vision2seq"

    def test_guess_paligemma(self):
        """测试推断 PaliGemma 架构"""
        from florence_forge.core.backends.generic_hf_backend import _guess_architecture_type
        assert _guess_architecture_type("google/paligemma-3b") == "causal_lm"

    def test_guess_llava(self):
        """测试推断 LLaVA 架构"""
        from florence_forge.core.backends.generic_hf_backend import _guess_architecture_type
        assert _guess_architecture_type("llava-hf/llava-1.5") == "causal_lm"

    def test_guess_unknown(self):
        """测试未知模型回退到 auto"""
        from florence_forge.core.backends.generic_hf_backend import _guess_architecture_type
        assert _guess_architecture_type("some/random-model") == "auto"
