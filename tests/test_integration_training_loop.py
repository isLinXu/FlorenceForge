"""端到端集成测试：验证训练循环流程

使用 Mock 对象模拟完整的训练流程，不依赖真实模型权重。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path

from florence_forge.core.tasks import FLORENCE2_TASKS
from florence_forge.core.config import DistributedConfig, TrainingConfig
from florence_forge.training.device_config import DeviceConfigurator

# 标记需要可选依赖的测试
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


class MockVLMOutput:
    """模拟 VLM 模型输出"""
    def __init__(self, loss=None, logits=None):
        self.loss = loss if loss is not None else torch.tensor(1.0, requires_grad=True)
        self.logits = logits

    @property
    def loss(self):
        return self._loss if self._loss is not None else torch.tensor(1.0, requires_grad=True)

    @loss.setter
    def loss(self, value):
        self._loss = value


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
    dataset.lazy_load = False
    dataset._sample_index = []  # 延迟加载索引
    dataset._sample_offset_cache = {}
    # 初始化 _sample_cache 以支持 use_cache / _cache_index 属性
    dataset._init_sample_cache()
    dataset._init_augmentation()
    dataset.backend = None

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
            "task_type": "CAPTION",
        }
        sample2 = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor([-100, 7, 8]),
            "task_type": "CAPTION",
        }

        collator = Florence2Collator(pad_token_id=0)
        batch = collator([sample1, sample2])

        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "pixel_values" in batch
        assert "labels" in batch
        assert batch["task_type"] in FLORENCE2_TASKS

    def test_dataset_getitem_with_encoding(self, mock_dataset):
        """验证 Dataset __getitem__ 返回正确结构"""

        # 设置一个 mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[6, 7]]),
        }
        mock_processor.tokenizer = mock_tokenizer
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
        assert item["task_type"] in FLORENCE2_TASKS


class TestTrainerV2AcceleratorWiring:
    def test_create_accelerator_attaches_fsdp_plugin(self):
        if not _has_accelerate():
            pytest.skip("accelerate 未安装")
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        config = TrainingConfig(device="cuda")
        config.distributed_settings = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            fsdp_sharding_strategy="FULL_SHARD",
        )
        plugin = DeviceConfigurator(config).build_distributed_plugin(
            config.distributed_settings
        )
        assert plugin is not None


class TestTrainingLoopRefactored:
    def test_create_accelerator_attaches_deepspeed_plugin(self):
        if not _has_accelerate():
            pytest.skip("accelerate 未安装")
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")


        config = TrainingConfig(device="cuda")
        config.distributed_settings = DistributedConfig(
            enabled=True,
            strategy="deepspeed",
            deepspeed_config_file=None,
        )
        plugin = DeviceConfigurator(config).build_distributed_plugin(
            config.distributed_settings
        )
        assert plugin is None

    def test_disabled_strategy_returns_none(self):
        config = TrainingConfig()
        config.distributed_settings = DistributedConfig(enabled=False, strategy="none")
        assert (
            DeviceConfigurator(config).build_distributed_plugin(
                config.distributed_settings
            )
            is None
        )


def test_plot_backend_defaults_to_no_show(monkeypatch):
    from florence_forge.utils.plot_backend import should_show_plots

    monkeypatch.delenv("FLORENCE_FORGE_SHOW_PLOTS", raising=False)
    assert should_show_plots() is False
    monkeypatch.setenv("FLORENCE_FORGE_SHOW_PLOTS", "1")
    assert should_show_plots() is True


def test_training_loop_runs_single_optimizer_step_cpu():
    """v2 训练循环：单 batch 前向 + 反向 + optimizer.step（CPU，无真实模型）。"""
    from florence_forge.training.training_loop import TrainingLoop

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 1)

        def forward(self, input_ids=None, labels=None, **kwargs):
            x = input_ids.float()
            loss = self.proj(x).mean()
            return type("Out", (), {"loss": loss})()

    config = TrainingConfig(device="cpu", num_epochs=1, batch_size=1)
    config.max_grad_norm = 0.0
    config.gradient_accumulation_steps = 1
    config.logging_steps = 999

    model = TinyModel()
    loop = TrainingLoop(model=model, config=config, accelerator=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    class OneBatchLoader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    metrics = loop.train_epoch(OneBatchLoader(), optimizer, None, epoch=0)
    assert loop.global_step == 1
    assert "loss" in metrics


@pytest.mark.skipif(not _has_accelerate(), reason="accelerate 未安装")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_accelerate_cuda_training_step_smoke():
    """Accelerate 在 CUDA 上完成一步 backward + optimizer.step。"""
    from accelerate import Accelerator

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    accelerator = Accelerator()
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    inputs = torch.randn(2, 4, device=accelerator.device)
    with accelerator.accumulate(model):
        loss = model(inputs).sum()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()

    assert accelerator.device.type == "cuda"


@pytest.mark.skipif(not _has_accelerate(), reason="accelerate 未安装")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_multitask_trainer_v2_setup_and_one_training_step_cuda():
    """v2 MultiTaskTrainer：setup_training + TrainingLoop 单步（CUDA）。"""
    from florence_forge.training.trainer import MultiTaskTrainer

    class TinyModel(nn.Module):
        def forward(self, input_ids=None, labels=None, **kwargs):
            loss = input_ids.float().mean()
            return type("Out", (), {"loss": loss})()

    batch = {
        "input_ids": torch.randint(0, 10, (2, 8)),
        "attention_mask": torch.ones(2, 8),
        "labels": torch.randint(0, 10, (2, 8)),
        "task_type": "CAPTION",
    }

    class OneBatchLoader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    train_dataset = MagicMock()
    train_dataset.__len__ = MagicMock(return_value=1)

    config = TrainingConfig(device="cuda", num_epochs=1, batch_size=2)
    config.max_grad_norm = 0.0
    config.gradient_accumulation_steps = 1
    config.logging_steps = 999
    config.use_callbacks = False
    config.use_lora = False

    trainer = MultiTaskTrainer(
        model=TinyModel(),
        train_dataset=train_dataset,
        val_dataset=None,
        config=config,
    )
    trainer.setup_training()
    trainer.train_dataloader = OneBatchLoader()

    metrics = trainer.training_loop.train_epoch(
        train_dataloader=trainer.train_dataloader,
        optimizer=trainer.optimizer,
        lr_scheduler=trainer.lr_scheduler,
        epoch=0,
    )
    assert trainer.training_loop.global_step == 1
    assert "loss" in metrics


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要至少 2 张 CUDA GPU",
)
def test_multi_gpu_cuda_tensor_placement_smoke():
    """多 GPU 环境最小冒烟：张量可在 cuda:0 / cuda:1 上创建。"""
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    x = torch.ones(2, 2, device=device0)
    y = torch.ones(2, 2, device=device1)
    assert x.device.type == "cuda"
    assert y.device.index == 1
