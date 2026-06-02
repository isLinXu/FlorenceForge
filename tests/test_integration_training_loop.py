"""端到端集成测试：验证训练循环流程

使用 Mock 对象模拟完整的训练流程，不依赖真实模型权重。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path

# 标记需要可选依赖的测试
pytestmark = [pytest.mark.integration]


class MockVLMOutput:
    """模拟 VLM 模型输出"""
    def __init__(self, loss=None, logits=None):
        self.loss = loss if loss is not None else torch.tensor(1.0, requires_grad=True)
        self.logits = logits


class MockVLMBackend(nn.Module):
    """模拟 VLM 后端，用于集成测试"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._model = nn.Linear(10, 10)  # 简单的可训练参数
        self._processor = MagicMock()
        self._device = "cpu"
        self.is_peft_model = False

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, **kwargs):
        # 模拟前向传播，返回带 loss 的输出
        batch_size = input_ids.shape[0]
        loss = torch.tensor(1.0 - 0.01 * batch_size, requires_grad=True)
        return MockVLMOutput(loss=loss)

    def generate(self, **kwargs):
        return torch.randint(0, 100, (1, 10))

    def decode(self, token_ids, **kwargs):
        return ["mock output"]

    def encode(self, **kwargs):
        return {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

    def save_pretrained(self, path):
        # Mock 后端：创建空目录即可
        Path(path).mkdir(parents=True, exist_ok=True)

    def load_pretrained(self, path, **kwargs):
        pass

    def get_model_info(self):
        return {
            "model_name": "mock",
            "total_parameters": 100,
            "trainable_parameters": 100,
            "trainable_ratio": 1.0,
            "device": "cpu",
            "dtype": "float32",
        }

    def get_task_prompt(self, task_name):
        return f"<{task_name}>"

    def supports_task(self, task_name):
        return True


@pytest.fixture
def mock_config():
    """创建测试配置"""
    from florence_forge.core.config import ModelConfig, TrainingConfig, DataConfig, OptimizationConfig

    model_config = ModelConfig(
        model_name="mock-model",
        backend_name="mock",
        use_lora=False,
        trust_remote_code=False,
    )

    data_config = DataConfig(
        batch_size=2,
        num_workers=0,
        shuffle=True,
        drop_last=False,
        use_cache=False,
    )

    opt_config = OptimizationConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    training_config = TrainingConfig(
        num_epochs=1,
        max_steps=3,
        model_config=model_config,
        data_config=data_config,
        optimization_config=opt_config,
        logging_steps=1,
        save_steps=10,
        output_dir="/tmp/florence_forge_test",
    )

    return training_config


@pytest.fixture
def mock_dataset():
    """创建模拟数据集"""
    from florence_forge.data.dataset import MultiTaskDataset, TaskSample

    samples = [
        TaskSample(
            task_type="CAPTION",
            image_path="/tmp/fake_image_1.jpg",
            prefix="<CAPTION>",
            suffix="A cat sitting on a table",
        ),
        TaskSample(
            task_type="CAPTION",
            image_path="/tmp/fake_image_2.jpg",
            prefix="<CAPTION>",
            suffix="A dog running in the park",
        ),
        TaskSample(
            task_type="OD",
            image_path="/tmp/fake_image_3.jpg",
            prefix="<OD>",
            suffix="<loc_100><loc_200>cat",
        ),
        TaskSample(
            task_type="OD",
            image_path="/tmp/fake_image_4.jpg",
            prefix="<OD>",
            suffix="<loc_300><loc_400>dog",
        ),
    ]

    # 创建数据集并注入样本
    from florence_forge.core.config import DataConfig

    dataset = MultiTaskDataset.__new__(MultiTaskDataset)
    dataset.data_configs = []
    dataset.image_base_path = Path("/tmp")
    dataset.config = DataConfig(use_cache=False, cache_dir=None)
    dataset.processor = None
    dataset.samples = samples
    dataset.task_weights = {"CAPTION": 0.5, "OD": 0.5}
    dataset.task_indices = {"CAPTION": [0, 1], "OD": [2, 3]}
    dataset.use_cache = False
    dataset.cache_dir = None
    dataset._cache_index = {}
    dataset.backend = None
    dataset.lazy_load = False

    return dataset


class TestTrainingLoopIntegration:
    """训练循环集成测试"""

    def test_collator_assembles_batch_correctly(self, mock_dataset):
        """验证 Collator 能正确组装批次"""
        from florence_forge.data.collate import Florence2Collator

        # 手动构造模拟的编码后样本
        sample1 = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor([-100, -100, 5, 6]),
            "token_type_ids": torch.tensor([0, 0, 1, 1]),
            "task_type": "CAPTION",
        }
        sample2 = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor([-100, 7, 8]),
            "token_type_ids": torch.tensor([0, 1, 1]),
            "task_type": "CAPTION",
        }

        collator = Florence2Collator(pad_token_id=0)
        batch = collator([sample1, sample2])

        assert batch["input_ids"].shape == (2, 4)  # batch_size=2, max_seq_len=4
        assert batch["attention_mask"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)
        assert batch["token_type_ids"].shape == (2, 4)
        assert batch["pixel_values"].shape == (2, 3, 224, 224)
        assert batch["task_type"] == "CAPTION"

        # 验证 padding 正确
        assert batch["input_ids"][1, 3].item() == 0  # 第二个样本被 padding
        assert batch["labels"][1, 3].item() == -100  # labels 的 padding
        assert batch["token_type_ids"][1, 3].item() == 0

    def test_dataset_getitem_with_encoding(self, mock_dataset):
        """验证 Dataset __getitem__ 返回正确结构"""
        from florence_forge.data.dataset import MultiTaskDataset
        from unittest.mock import patch

        # 设置一个 mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        mock_dataset.processor = mock_processor

        # Patch 图像加载以避免 FileNotFoundError
        with patch("florence_forge.data.dataset._load_image_cached") as mock_load_img:
            from PIL import Image
            mock_load_img.return_value = Image.new("RGB", (224, 224), color="red")

            item = mock_dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "pixel_values" in item
        assert "labels" in item
        assert item["task_type"] == "CAPTION"

    def test_model_backend_delegation(self, mock_config):
        """验证模型能正确委托到后端"""
        from florence_forge.core.model import Florence2MultiTaskModel

        # 使用 object.__new__ 绕过 nn.Module 的 __init__ 检查
        model = object.__new__(Florence2MultiTaskModel)
        nn.Module.__init__(model)
        model.config = mock_config.model_config
        model.is_peft_model = False

        # 注入 mock 后端
        mock_backend = MockVLMBackend(mock_config.model_config)
        model._backend = mock_backend

        # 验证 property 代理
        assert model.model is mock_backend.model
        assert model.processor is mock_backend.processor

        # 验证 forward 委托
        input_ids = torch.randint(0, 100, (2, 10))
        pixel_values = torch.randn(2, 3, 224, 224)
        labels = torch.full((2, 10), -100)
        labels[:, 5:] = torch.randint(0, 100, (2, 5))

        outputs = model.forward(input_ids, pixel_values, labels=labels)
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_full_training_step_simulation(self, mock_config, mock_dataset):
        """模拟完整的训练步骤"""
        from florence_forge.core.model import Florence2MultiTaskModel
        from florence_forge.data.collate import Florence2Collator
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW

        # 创建 mock 模型
        model = object.__new__(Florence2MultiTaskModel)
        nn.Module.__init__(model)
        model.config = mock_config.model_config
        model.is_peft_model = False
        model._backend = MockVLMBackend(mock_config.model_config)

        # 创建 collator
        collator = Florence2Collator(pad_token_id=0)

        # 创建真正的 mock dataset 类（避免 __getitem__ 替换问题）
        class EncodedMockDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        # 为 mock dataset 准备编码后的样本
        encoded_samples = []
        for i in range(len(mock_dataset.samples)):
            encoded_samples.append({
                "input_ids": torch.randint(1, 100, (10,)),
                "attention_mask": torch.ones(10),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.tensor([-100, -100, -100, 10, 20, 30, 40, 50, 60, 70]),
                "task_type": mock_dataset.samples[i].task_type,
            })

        encoded_dataset = EncodedMockDataset(encoded_samples)

        # 创建数据加载器
        dataloader = DataLoader(
            encoded_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collator,
        )

        # 创建优化器
        optimizer = AdamW(model.parameters(), lr=1e-4)

        # 模拟训练步骤
        model.train()
        losses = []

        for step, batch in enumerate(dataloader):
            if step >= 2:  # 只运行 2 步
                break

            input_ids = batch["input_ids"]
            pixel_values = batch["pixel_values"]
            attention_mask = batch.get("attention_mask")
            labels = batch["labels"]

            # 前向传播
            outputs = model(input_ids, pixel_values, attention_mask, labels)
            loss = outputs.loss
            losses.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)

    def test_model_save_load_with_backend(self, mock_config, tmp_path):
        """验证基于后端的模型保存和加载"""
        from florence_forge.core.model import Florence2MultiTaskModel

        model = object.__new__(Florence2MultiTaskModel)
        nn.Module.__init__(model)
        model.config = mock_config.model_config
        model.is_peft_model = False
        model._backend = MockVLMBackend(mock_config.model_config)

        # 保存
        save_path = tmp_path / "test_model"
        model.save_pretrained(str(save_path))

        # 验证保存路径存在
        assert save_path.exists()

    def test_task_switching_with_mock_model(self, mock_config, mock_dataset):
        """验证多任务切换逻辑"""
        from florence_forge.core.model import Florence2MultiTaskModel

        model = object.__new__(Florence2MultiTaskModel)
        nn.Module.__init__(model)
        model.config = mock_config.model_config
        model.is_peft_model = False
        model._backend = MockVLMBackend(mock_config.model_config)

        # 模拟任务切换
        task_types = ["CAPTION", "OD", "CAPTION", "OD"]
        for task in task_types:
            # 验证 predict_task 委托到后端
            result = model.predict_task(
                images=MagicMock(),
                task_name=task,
            )
            assert result is not None

    def test_multiprocess_dataloader_compatibility(self, mock_dataset):
        """验证 Dataset 支持多进程 DataLoader"""
        import pickle
        from florence_forge.data.dataset import MultiTaskDataset

        # 序列化并反序列化 dataset
        serialized = pickle.dumps(mock_dataset)
        restored = pickle.loads(serialized)

        # 验证反序列化后状态正确
        assert len(restored.samples) == len(mock_dataset.samples)
        assert restored._cache_index == {}  # 内存缓存不应被序列化
        assert restored.processor is None   # processor 不应被序列化
        assert restored.use_cache == mock_dataset.use_cache
