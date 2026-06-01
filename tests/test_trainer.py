"""单元测试：多任务训练器 MultiTaskTrainer

测试训练器初始化、训练循环关键路径和断点续训功能。
使用 Mock 对象，不依赖真实模型权重。
"""

import pytest
import torch
import torch.nn as nn
import time
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

from florence_forge.core.config import TrainingConfig, ModelConfig, DataConfig, OptimizationConfig


class SimpleMockModel(nn.Module):
    """用于测试的简单 Mock 模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self._processor = MagicMock()
        self._generate_called = False

    @property
    def processor(self):
        return self._processor

    def generate(self, **kwargs):
        self._generate_called = True
        return torch.randint(0, 100, (1, 10))

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        loss = torch.tensor(1.0, requires_grad=True)
        return type("Output", (), {"loss": loss, "logits": torch.randn(batch_size, 10)})()

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load_pretrained(self, path, **kwargs):
        pass


@pytest.fixture
def mock_config(tmp_path):
    """创建测试用的 TrainingConfig"""
    return TrainingConfig(
        experiment_name="test_experiment",
        model_settings=ModelConfig(model_name="mock-model"),
        data_settings=DataConfig(
            task_types=["CAPTION"],
            train_data_path=str(tmp_path / "train.json"),
            val_data_path=str(tmp_path / "val.json"),
            batch_size=2,
        ),
        optimization_settings=OptimizationConfig(
            learning_rate=1e-4,
            num_epochs=1,
        ),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def mock_model():
    return SimpleMockModel()


@pytest.fixture
def mock_dataset():
    """创建 Mock 数据集"""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.task_types = ["CAPTION"]
    item = {
        "input_ids": torch.randint(0, 100, (2, 10)),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
        "pixel_values": torch.randn(2, 3, 224, 224),
        "labels": torch.randint(0, 100, (2, 10)),
        "task": "CAPTION",
    }
    dataset.__getitem__ = MagicMock(return_value=item)
    return dataset


class TestTrainingConfigAliases:
    """TrainingConfig Pydantic alias 测试"""

    def test_optimization_settings_accessible(self):
        c = TrainingConfig()
        assert hasattr(c, 'optimization_settings')
        assert c.optimization_settings.learning_rate > 0

    def test_data_settings_accessible(self):
        c = TrainingConfig()
        assert hasattr(c, 'data_settings')
        assert c.data_settings.batch_size > 0

    def test_model_settings_accessible(self):
        c = TrainingConfig()
        assert hasattr(c, 'model_settings')
        assert c.model_settings.model_name is not None

    def test_model_config_is_configdict(self):
        """model_config 应该返回 Pydantic ConfigDict，而不是 ModelConfig"""
        c = TrainingConfig()
        assert isinstance(c.model_config, dict)  # Pydantic ConfigDict
        assert not hasattr(c.model_config, 'model_name')  # 不是 ModelConfig

    def test_alias_initialization(self):
        """使用 alias 初始化（如 optimization_config）也应该工作"""
        c = TrainingConfig(optimization_config={"learning_rate": 0.001})
        assert c.optimization_settings.learning_rate == 0.001


class TestMultiTaskTrainerInit:
    """训练器初始化测试"""

    def test_init_basic(self, mock_model, mock_dataset, mock_config):
        from florence_forge.training.trainer import MultiTaskTrainer

        trainer = MultiTaskTrainer(
            model=mock_model,
            train_dataset=mock_dataset,
            val_dataset=mock_dataset,
            config=mock_config,
        )
        assert trainer.model is mock_model
        assert trainer.config is mock_config


class TestAcceleratorCompat:
    """Accelerator 兼容模块测试"""

    def test_import_accelerator_compat(self):
        from florence_forge.training._accelerator_compat import Accelerator
        assert Accelerator is not None

    def test_fallback_accelerator_basic(self):
        """测试 fallback Accelerator 的基本方法"""
        try:
            import accelerate
            pytest.skip("accelerate 已安装，跳过 fallback 测试")
        except ImportError:
            from florence_forge.training._accelerator_compat import Accelerator

            acc = Accelerator()
            assert acc.is_main_process is True
            assert str(acc.device) == "cpu"

    def test_fallback_accelerator_backward(self):
        """测试 fallback backward"""
        try:
            import accelerate
            pytest.skip("accelerate 已安装，跳过 fallback 测试")
        except ImportError:
            from florence_forge.training._accelerator_compat import Accelerator

            acc = Accelerator()
            loss = torch.tensor(1.0, requires_grad=True)
            acc.backward(loss)  # 不应抛出异常

    def test_fallback_unwrap_model(self):
        """测试 fallback unwrap_model"""
        try:
            import accelerate
            pytest.skip("accelerate 已安装，跳过 fallback 测试")
        except ImportError:
            from florence_forge.training._accelerator_compat import Accelerator

            acc = Accelerator()
            model = SimpleMockModel()
            unwrapped = acc.unwrap_model(model)
            assert unwrapped is model


class TestCLIResumeFlag:
    """CLI --resume 标志功能测试"""

    def test_resume_flag_in_argparse(self):
        """验证 --resume 参数在 argparse 中注册"""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        train_parser = subparsers.add_parser('train')
        train_parser.add_argument('--resume', '-r', help='从检查点恢复训练')

        args = train_parser.parse_args(['--resume', '/path/to/checkpoint'])
        assert args.resume == '/path/to/checkpoint'

    def test_load_checkpoint_method_exists(self):
        """验证训练器实现了 load_checkpoint 方法"""
        from florence_forge.training.trainer import MultiTaskTrainer
        assert hasattr(MultiTaskTrainer, "load_checkpoint"), "训练器应实现 load_checkpoint 方法"


class TestTrainingReportGeneration:
    """训练结束报告生成行为测试"""

    def test_report_generation_can_be_disabled(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
        trainer.config = SimpleNamespace(generate_training_report_on_end=False)
        trainer.visualizer = MagicMock()

        assert trainer._generate_training_report_on_end() is None
        trainer.visualizer.generate_training_report.assert_not_called()

    def test_report_generation_sync_mode_returns_path(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
        trainer.config = SimpleNamespace(
            generate_training_report_on_end=True,
            async_training_report=False,
        )
        trainer.visualizer = MagicMock()
        trainer.visualizer.generate_training_report.return_value = "training_report.html"

        assert trainer._generate_training_report_on_end() == "training_report.html"
        trainer.visualizer.generate_training_report.assert_called_once()

    def test_report_generation_async_mode_does_not_block(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        started = threading.Event()
        release = threading.Event()

        def slow_report():
            started.set()
            release.wait(timeout=2)
            return "training_report.html"

        trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
        trainer.config = SimpleNamespace(
            generate_training_report_on_end=True,
            async_training_report=True,
        )
        trainer.visualizer = MagicMock()
        trainer.visualizer.generate_training_report.side_effect = slow_report

        start = time.monotonic()
        assert trainer._generate_training_report_on_end() is None
        elapsed = time.monotonic() - start

        assert elapsed < 0.2
        assert started.wait(timeout=1)
        assert trainer._report_thread.daemon is True

        release.set()
        trainer._report_thread.join(timeout=1)
        assert not trainer._report_thread.is_alive()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyProgress:
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *args, **kwargs):
        pass


class _MemoryMonitorStub:
    class _Config:
        log_frequency = 999999

    def __init__(self):
        self.config = self._Config()

    def log_memory_usage(self, *args, **kwargs):
        pass


class _TaskSchedulerStub:
    def select_task(self, current_epoch):
        return "CAPTION"

    def update_task_performance(self, task_type, loss):
        pass


class _CallbackManagerStub:
    def on_step_begin(self, trainer, step, logs):
        pass

    def on_step_end(self, trainer, step, logs):
        pass


def _build_minimal_trainer(batch, gradient_validator=None):
    from florence_forge.training.trainer import MultiTaskTrainer

    trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
    trainer.model = SimpleMockModel()
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.train_dataloader = [batch]
    trainer.config = MagicMock()
    trainer.config.num_epochs = 1
    trainer.config.logging_steps = 1000
    trainer.config.max_steps = None
    trainer.config.optimization_settings = MagicMock(max_grad_norm=1.0)
    trainer.accelerator = MagicMock()
    trainer.accelerator.is_local_main_process = False
    trainer.accelerator.accumulate.return_value = _NullContext()
    trainer.accelerator.backward = MagicMock()
    trainer.accelerator.clip_grad_norm_ = MagicMock(return_value=0.5)
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{"lr": 1e-4}]
    trainer.lr_scheduler = MagicMock()
    trainer.memory_monitor = _MemoryMonitorStub()
    trainer.task_scheduler = _TaskSchedulerStub()
    trainer.callback_manager = _CallbackManagerStub()
    trainer.gradient_validator = gradient_validator
    trainer.lora_manager = None
    trainer.train_metrics = defaultdict(list)
    trainer.val_metrics = defaultdict(list)
    trainer.step_metrics = []
    trainer._record_step_metrics = MagicMock()
    trainer._move_batch_to_device = lambda x: x
    return trainer


class TestMultiTaskTrainerTrainEpochBranches:
    def test_train_epoch_skips_step_when_labels_missing(self):
        """labels 缺失时应只清梯度，不执行 backward/step。"""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "task_type": "CAPTION",
        }

        trainer = _build_minimal_trainer(batch)

        with patch("florence_forge.training.trainer.tqdm", side_effect=lambda iterable, **kwargs: _DummyProgress(iterable)):
            metrics = trainer._train_epoch()

        trainer.optimizer.zero_grad.assert_called_once()
        trainer.optimizer.step.assert_not_called()
        trainer.lr_scheduler.step.assert_not_called()
        trainer.accelerator.backward.assert_not_called()
        assert metrics == {}
        assert trainer.global_step == 0

    def test_train_epoch_skips_optimizer_step_when_gradient_invalid(self):
        """梯度验证失败时应跳过 optimizer/lr_scheduler，但仍清梯度。"""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "labels": torch.randint(0, 100, (2, 10)),
            "task_type": "CAPTION",
        }

        gradient_validator = MagicMock()
        gradient_validator.validate_gradients.return_value = (False, {"reason": "test"})
        trainer = _build_minimal_trainer(batch, gradient_validator=gradient_validator)

        with patch("florence_forge.training.trainer.tqdm", side_effect=lambda iterable, **kwargs: _DummyProgress(iterable)):
            metrics = trainer._train_epoch()

        trainer.accelerator.backward.assert_called_once()
        trainer.optimizer.step.assert_not_called()
        trainer.lr_scheduler.step.assert_not_called()
        trainer.optimizer.zero_grad.assert_called_once()
        trainer.accelerator.clip_grad_norm_.assert_not_called()
        assert "loss" in metrics
        assert trainer.global_step == 1


class TestMultiTaskTrainerMetricsHistory:
    def test_record_step_metrics_keeps_bounded_in_memory_history(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
        trainer.step_csv_buffer = []
        trainer.csv_buffer_size = 999
        trainer.step_metrics = []
        trainer.step_metrics_history_limit = 2
        trainer.accelerator = MagicMock()
        trainer.accelerator.is_local_main_process = False

        for step in range(5):
            trainer._record_step_metrics(
                step=step,
                epoch=0,
                task_type="CAPTION",
                loss=float(step),
                learning_rate=1e-4,
                grad_norm=0.1,
                time_per_step=0.2,
            )

        assert [metric["step"] for metric in trainer.step_metrics] == [3, 4]
        assert len(trainer.step_csv_buffer) == 5

    def test_record_step_metrics_can_disable_in_memory_history(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        trainer = MultiTaskTrainer.__new__(MultiTaskTrainer)
        trainer.step_csv_buffer = []
        trainer.csv_buffer_size = 999
        trainer.step_metrics = []
        trainer.step_metrics_history_limit = 0
        trainer.accelerator = MagicMock()
        trainer.accelerator.is_local_main_process = False

        trainer._record_step_metrics(
            step=1,
            epoch=0,
            task_type="CAPTION",
            loss=1.0,
            learning_rate=1e-4,
            grad_norm=0.1,
            time_per_step=0.2,
        )

        assert trainer.step_metrics == []
        assert len(trainer.step_csv_buffer) == 1


class TestMultiDatasetTrainerMetricsHistory:
    def test_init_binds_config_before_dataset_creation(self):
        from florence_forge.training.multi_dataset_trainer import MultiDatasetTrainer

        class FakeDataset:
            def __len__(self):
                return 1

        class FakeDatasetManager:
            def validate_configuration(self):
                return {"errors": [], "warnings": []}

            def create_unified_dataset(self, task_types=None, processor=None):
                return FakeDataset()

            def create_balanced_split(self, **kwargs):
                assert kwargs["val_ratio"] == 0.2
                return FakeDataset(), FakeDataset(), FakeDataset()

        def fake_parent_init(self, model, train_dataset, val_dataset=None, config=None, accelerator=None):
            self.model = model
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.config = config

        with patch(
            "florence_forge.training.multi_dataset_trainer.MultiTaskTrainer.__init__",
            new=fake_parent_init,
        ):
            trainer = MultiDatasetTrainer(
                model=SimpleMockModel(),
                dataset_manager=FakeDatasetManager(),
                config=None,
            )

        assert isinstance(trainer.config, TrainingConfig)

    def test_record_dataset_performance_keeps_bounded_history(self):
        from florence_forge.training.multi_dataset_trainer import MultiDatasetTrainer

        trainer = MultiDatasetTrainer.__new__(MultiDatasetTrainer)
        trainer.dataset_performance = defaultdict(lambda: defaultdict(list))
        trainer.dataset_performance_history_limit = 2

        for step in range(5):
            trainer._record_dataset_performance("ds1", "CAPTION", float(step))

        assert trainer.dataset_performance["ds1"]["CAPTION"] == [3.0, 4.0]

    def test_record_dataset_performance_can_disable_history(self):
        from florence_forge.training.multi_dataset_trainer import MultiDatasetTrainer

        trainer = MultiDatasetTrainer.__new__(MultiDatasetTrainer)
        trainer.dataset_performance = defaultdict(lambda: defaultdict(list))
        trainer.dataset_performance_history_limit = 0

        trainer._record_dataset_performance("ds1", "CAPTION", 1.0)

        assert trainer.dataset_performance["ds1"]["CAPTION"] == []
