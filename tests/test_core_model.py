"""测试核心模型和后端功能"""
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock

from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.config import ModelConfig
from florence_forge.core.backends.base_vlm import BaseVLMBackend, VLMBackendRegistry


class MockBackend(BaseVLMBackend):
    """测试用的 Mock 后端"""
    BACKEND_NAME = "mock"
    ARCHITECTURE_TYPE = "encoder_decoder"
    
    def __init__(self, config):
        super().__init__(config)
        self._task_prompts = {
            "CAPTION": "<CAPTION>",
            "OD": "<OD>"
        }
        self._supports_tasks = ["CAPTION", "OD"]
    
    def load_model(self):
        """加载 mock 模型"""
        self._model = MagicMock()
        self._model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        self._device = "cpu"
        self._dtype = torch.float32
    
    def load_processor(self):
        """加载 mock processor"""
        self._processor = MagicMock()
    
    def get_task_prompt(self, task_name: str) -> str:
        return self._task_prompts.get(task_name, "")
    
    def supports_task(self, task_name: str) -> bool:
        return task_name in self._supports_tasks


@pytest.fixture
def mock_config():
    """创建测试配置"""
    config = ModelConfig()
    config.model_name = "mock-model"
    config.device = "cpu"
    config.backend_name = "mock"
    return config


@pytest.fixture
def register_mock_backend():
    """注册 mock 后端"""
    VLMBackendRegistry.register("mock", MockBackend)
    yield
    # 清理
    if "mock" in VLMBackendRegistry._backends:
        del VLMBackendRegistry._backends["mock"]


class TestFlorence2MultiTaskModel:
    """测试 Florence2MultiTaskModel"""
    
    def test_model_initialization(self, mock_config, register_mock_backend):
        """测试模型初始化"""
        model = Florence2MultiTaskModel(mock_config)
        assert model.config == mock_config
        assert model._backend is not None
        assert model._backend.BACKEND_NAME == "mock"
    
    def test_model_load(self, mock_config, register_mock_backend):
        """测试模型加载"""
        model = Florence2MultiTaskModel(mock_config)
        model.load()
        assert model._backend.model is not None
        assert model._backend.processor is not None
    
    def test_model_to_device(self, mock_config, register_mock_backend):
        """测试设备转移并验证设备同步"""
        model = Florence2MultiTaskModel(mock_config)
        model.load()
        
        # 转移到 cuda（mock 模型会保持在 cpu）
        model.to("cuda")
        
        # 验证后端设备状态已同步
        assert model._backend._device == "cuda"
    
    def test_model_info(self, mock_config, register_mock_backend):
        """测试模型信息获取"""
        model = Florence2MultiTaskModel(mock_config)
        model.load()
        
        info = model.get_model_info()
        assert "model_name" in info
        assert "backend" in info
        assert info["backend"] == "mock"
        assert "total_parameters" in info

    def test_generate_tensor_inputs_delegates_to_backend(
        self,
        mock_config,
        register_mock_backend,
    ):
        """评估器使用的张量级 generate 应直接委托给后端。"""
        model = Florence2MultiTaskModel(mock_config)
        expected = torch.tensor([[4, 5, 6]])
        model._backend.generate = MagicMock(return_value=expected)

        input_ids = torch.tensor([[1, 2]])
        pixel_values = torch.randn(1, 3, 224, 224)
        attention_mask = torch.tensor([[1, 1]])

        output = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=16,
        )

        assert output is expected
        model._backend.generate.assert_called_once()
        call_kwargs = model._backend.generate.call_args.kwargs
        assert call_kwargs["input_ids"] is input_ids
        assert call_kwargs["pixel_values"] is pixel_values
        assert call_kwargs["attention_mask"] is attention_mask
        assert call_kwargs["max_new_tokens"] == 16

    def test_generate_tensor_inputs_moves_to_backend_device(
        self,
        mock_config,
        register_mock_backend,
    ):
        """张量级 generate 在门面层先对齐到后端设备。"""
        model = Florence2MultiTaskModel(mock_config)
        model._backend._device = "meta"
        expected = torch.empty((1, 1), device="meta", dtype=torch.long)
        model._backend.generate = MagicMock(return_value=expected)

        output = model.generate(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.randn(1, 3, 4, 4),
            attention_mask=torch.tensor([[1, 1]]),
        )

        assert output is expected
        call_kwargs = model._backend.generate.call_args.kwargs
        assert call_kwargs["input_ids"].device.type == "meta"
        assert call_kwargs["pixel_values"].device.type == "meta"
        assert call_kwargs["attention_mask"].device.type == "meta"

    def test_generate_image_inputs_preserves_decoded_text_api(
        self,
        mock_config,
        register_mock_backend,
    ):
        """图片级 generate 仍返回清理后的文本结果。"""
        model = Florence2MultiTaskModel(mock_config)
        model._backend.encode = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        })
        model._backend.generate = MagicMock(return_value=torch.tensor([[3, 4]]))
        model._backend.decode = MagicMock(return_value=["<CAPTION>a cat"])

        image = Image.new("RGB", (8, 8))
        output = model.generate(images=image, task_prompt="<CAPTION>")

        assert output == "a cat"
        model._backend.encode.assert_called_once()
        model._backend.generate.assert_called_once()
        model._backend.decode.assert_called_once()


class TestVLMBackendRegistry:
    """测试后端注册表"""
    
    def test_backend_registration(self):
        """测试后端注册"""
        VLMBackendRegistry.register("test-backend", MockBackend)
        assert "test-backend" in VLMBackendRegistry.list_backends()
        
        # 清理
        del VLMBackendRegistry._backends["test-backend"]
    
    def test_backend_creation(self, mock_config):
        """测试后端创建"""
        VLMBackendRegistry.register("test-backend", MockBackend)
        
        backend = VLMBackendRegistry.create("test-backend", mock_config)
        assert isinstance(backend, MockBackend)
        assert backend.BACKEND_NAME == "mock"
        
        # 清理
        del VLMBackendRegistry._backends["test-backend"]
    
    def test_unknown_backend_raises_error(self, mock_config):
        """测试未知后端抛出错误"""
        with pytest.raises(ValueError, match="未知后端"):
            VLMBackendRegistry.create("nonexistent-backend", mock_config)


class TestBaseVLMBackend:
    """测试 BaseVLMBackend 基类功能"""
    
    def test_optimal_device_detection(self, mock_config, register_mock_backend):
        """测试设备检测"""
        backend = MockBackend(mock_config)
        device = backend._get_optimal_device()
        
        # 在测试环境中应该选择可用设备
        assert device in ["cpu", "cuda", "mps"]
    
    def test_optimal_dtype_selection(self, mock_config, register_mock_backend):
        """测试数据类型选择"""
        backend = MockBackend(mock_config)
        dtype = backend._get_optimal_dtype("cpu")
        
        # CPU 应该使用 float32
        assert dtype == torch.float32

    def test_load_with_cpu_fallback_moves_mps_model_to_mps(self, mock_config, register_mock_backend):
        """配置为 MPS 时，模型应真实移动到 MPS，而不是保留在 CPU。"""
        class FakeModel:
            def __init__(self):
                self.moved_to = None

            def to(self, device):
                self.moved_to = device
                return self

        fake_model = FakeModel()
        mock_config.device = "mps"
        backend = MockBackend(mock_config)

        backend._load_with_cpu_fallback(
            lambda model_name, **kwargs: fake_model,
            "mock-model",
            {"device_map": None, "torch_dtype": torch.float32},
        )

        assert fake_model.moved_to == "mps"
        assert backend._device == "mps"

    def test_florence_generate_omits_none_attention_mask_and_disables_cache(
        self,
        mock_config,
        register_mock_backend,
    ):
        """Florence-2 推理兼容当前 transformers：不传 None mask，并默认关闭 cache。"""
        class RecordingModel:
            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 3]])

        backend = MockBackend(mock_config)
        backend.BACKEND_NAME = "florence-2"
        backend.GENERATE_DEFAULTS = {"use_cache": False}
        backend._model = RecordingModel()

        backend.generate(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.randn(1, 3, 224, 224),
            attention_mask=None,
        )

        assert "attention_mask" not in backend._model.kwargs
        assert backend._model.kwargs["use_cache"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
