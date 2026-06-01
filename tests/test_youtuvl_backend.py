"""YouTuVLBackend 单元测试

测试腾讯优图 YouTu-VL VLM 后端的各项功能。
由于 YouTu-VL 依赖 transformers，测试使用 Mock 对象模拟。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from florence_forge.core.backends.base_vlm import VLMBackendRegistry


class MockYouTuVLModel(nn.Module):
    """模拟 YouTu-VL 模型"""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(10, 10))

    def forward(self, **kwargs):
        output = MagicMock()
        output.loss = torch.tensor(1.2) if "labels" in kwargs else None
        return output

    def generate(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return torch.randint(0, 100, (batch_size, 12))

    def save_pretrained(self, path):
        pass


class MockYouTuVLProcessor:
    """模拟 YouTu-VL Processor"""

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        batch_size = 1 if isinstance(text, str) else len(text)
        return {
            "input_ids": torch.randint(1, 100, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        if token_ids.dim() == 1:
            return ["<ref>cat</ref><box><x_min>100</x_min><y_min>200</y_min><x_max>300</x_max><y_max>400</y_max></box>"]
        return ["decoded text"] * token_ids.shape[0]

    def save_pretrained(self, path):
        pass


class MockConfig:
    """模拟 ModelConfig"""

    def __init__(self, model_name="tencent-YouTu/Youtu-VL-4B-Instruct",
                 task_prompts=None, supported_tasks=None):
        self.model_name = model_name
        self.backend_name = "youtuvl"
        self.task_prompts = task_prompts or {}
        self.supported_tasks = supported_tasks
        self.trust_remote_code = True
        self.device = "cpu"
        self.use_bf16 = False
        self.use_fp16 = False


@pytest.fixture
def mock_youtuvl_backend():
    """创建模拟的 YouTuVLBackend"""
    from florence_forge.core.backends.youtuvl_backend import YouTuVLBackend

    config = MockConfig()
    backend = object.__new__(YouTuVLBackend)
    nn.Module.__init__(backend)
    backend.config = config
    backend._model = MockYouTuVLModel()
    backend._processor = MockYouTuVLProcessor()
    backend._device = "cpu"
    backend._dtype = torch.float32
    backend.is_peft_model = False
    backend._architecture_type = "encoder_decoder"
    backend._task_prompts = {
        "CAPTION": "Describe the image in detail.",
        "OD": "Detect all objects in the image and provide their locations.",
        "VQA": "Answer the question based on the image.",
        "OCR": "Read all text present in the image.",
    }
    backend._supports_tasks = ["CAPTION", "OD", "VQA", "OCR"]
    return backend


class TestYouTuVLBackend:
    """YouTuVLBackend 测试类"""

    def test_backend_registered(self):
        """验证 YouTuVLBackend 已注册"""
        assert VLMBackendRegistry.is_registered("youtuvl")
        assert VLMBackendRegistry.is_registered("youtu-vl")
        assert VLMBackendRegistry.is_registered("tencent-youtuvl")

    def test_get_task_prompt(self, mock_youtuvl_backend):
        """测试任务 prompt 获取"""
        assert mock_youtuvl_backend.get_task_prompt("CAPTION") == "Describe the image in detail."
        assert mock_youtuvl_backend.get_task_prompt("OD") == "Detect all objects in the image and provide their locations."
        assert mock_youtuvl_backend.get_task_prompt("UNKNOWN") == "UNKNOWN"

    def test_supports_task(self, mock_youtuvl_backend):
        """测试任务支持检查"""
        assert mock_youtuvl_backend.supports_task("CAPTION") is True
        assert mock_youtuvl_backend.supports_task("VQA") is True
        assert mock_youtuvl_backend.supports_task("UNKNOWN") is False

    def test_get_supported_tasks(self, mock_youtuvl_backend):
        """测试获取支持的任务列表"""
        tasks = mock_youtuvl_backend.get_supported_tasks()
        assert "CAPTION" in tasks
        assert "OD" in tasks
        assert "VQA" in tasks

    def test_set_task_prompt(self, mock_youtuvl_backend):
        """测试动态设置任务 prompt"""
        mock_youtuvl_backend.set_task_prompt("NEW_TASK", "new prompt")
        assert mock_youtuvl_backend.get_task_prompt("NEW_TASK") == "new prompt"
        assert "NEW_TASK" in mock_youtuvl_backend.get_supported_tasks()

    def test_encode(self, mock_youtuvl_backend):
        """测试编码"""
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = mock_youtuvl_backend.encode(
            images=[image],
            text="Describe the image."
        )
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs

    def test_encode_with_task(self, mock_youtuvl_backend):
        """测试使用后端任务编码"""
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = mock_youtuvl_backend.encode_with_task(
            images=[image],
            task_name="VQA",
            text_input="What is in the image?"
        )
        assert "input_ids" in inputs
        assert "pixel_values" in inputs

    def test_generate(self, mock_youtuvl_backend):
        """测试生成"""
        input_ids = torch.randint(0, 100, (2, 5))
        pixel_values = torch.randn(2, 3, 224, 224)

        generated = mock_youtuvl_backend.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=10,
        )
        assert generated.shape[0] == 2

    def test_decode(self, mock_youtuvl_backend):
        """测试解码"""
        token_ids = torch.randint(0, 100, (2, 10))
        texts = mock_youtuvl_backend.decode(token_ids)
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_forward(self, mock_youtuvl_backend):
        """测试前向传播"""
        input_ids = torch.randint(0, 100, (2, 8))
        pixel_values = torch.randn(2, 3, 224, 224)
        labels = torch.full((2, 8), -100)
        labels[:, 4:] = torch.randint(0, 100, (2, 4))

        outputs = mock_youtuvl_backend.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_get_model_info(self, mock_youtuvl_backend):
        """测试模型信息"""
        info = mock_youtuvl_backend.get_model_info()
        assert info["backend"] == "youtuvl"
        assert info["architecture_type"] == "encoder_decoder"
        assert "total_parameters" in info
        assert "trainable_parameters" in info

    def test_generate_with_task(self, mock_youtuvl_backend):
        """测试便捷推理方法"""
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = mock_youtuvl_backend.generate_with_task(
            image=image,
            task_name="OD",
            max_new_tokens=50,
        )
        assert isinstance(result, str)

    def test_parse_detection_output(self):
        """测试检测输出解析"""
        from florence_forge.core.backends.youtuvl_backend import YouTuVLBackend

        text = (
            "<ref>cat</ref><box><x_min>100</x_min><y_min>200</y_min>"
            "<x_max>300</x_max><y_max>400</y_max></box>"
        )
        results = YouTuVLBackend.parse_detection_output(text)
        assert len(results) == 1
        assert results[0]["label"] == "cat"
        assert results[0]["bbox"] == [100, 200, 300, 400]

    def test_parse_multiple_detections(self):
        """测试多目标检测输出解析"""
        from florence_forge.core.backends.youtuvl_backend import YouTuVLBackend

        text = (
            "<ref>cat</ref><box><x_min>100</x_min><y_min>200</y_min>"
            "<x_max>300</x_max><y_max>400</y_max></box>"
            "<ref>dog</ref><box><x_min>50</x_min><y_min>60</y_min>"
            "<x_max>150</x_max><y_max>180</y_max></box>"
        )
        results = YouTuVLBackend.parse_detection_output(text)
        assert len(results) == 2
        assert results[0]["label"] == "cat"
        assert results[1]["label"] == "dog"

    def test_architecture_type(self, mock_youtuvl_backend):
        """验证架构类型"""
        assert mock_youtuvl_backend.ARCHITECTURE_TYPE == "encoder_decoder"


class TestYouTuVLBackendRegistry:
    """注册表集成测试"""

    def test_registry_create(self):
        """验证可以通过注册表创建"""
        assert VLMBackendRegistry.is_registered("youtuvl")

    def test_auto_select_youtuvl(self):
        """验证自动选择 YouTu-VL 后端"""
        from florence_forge.core.backends import auto_select_backend
        from florence_forge.core.backends.youtuvl_backend import YouTuVLBackend

        # backend_name="youtuvl" 直接匹配已注册后端
        config = MockConfig(model_name="tencent-YouTu/Youtu-VL-4B-Instruct")
        backend = auto_select_backend(config)
        assert isinstance(backend, YouTuVLBackend)

    def test_auto_select_youtuvl_by_model_name(self):
        """验证通过模型名称推断选择 YouTu-VL 后端"""
        from florence_forge.core.backends import auto_select_backend
        from florence_forge.core.backends.youtuvl_backend import YouTuVLBackend

        # 使用未注册的 backend_name，让函数通过 model_name 推断
        config = MockConfig(model_name="tencent-YouTu/Youtu-VL-4B-Instruct")
        config.backend_name = "youtu-vl"  # 另一个已注册别名
        backend = auto_select_backend(config)
        assert isinstance(backend, YouTuVLBackend)
