"""单元测试：多任务训练器 MultiTaskTrainer"""

import argparse
import pytest
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from florence_forge.core.config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)


class SimpleMockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, **kwargs):
        loss = torch.tensor(1.0, requires_grad=True)
        return type("Output", (), {"loss": loss})()

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def mock_config(tmp_path):
    return TrainingConfig(
        experiment_name="test_experiment",
        model_settings=ModelConfig(model_name="mock-model"),
        data_settings=DataConfig(
            task_types=["CAPTION"],
            train_data_path=str(tmp_path / "train.json"),
            val_data_path=str(tmp_path / "val.json"),
            batch_size=2,
        ),
        optimization_settings=OptimizationConfig(learning_rate=1e-4, num_epochs=1),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.task_indices = {"CAPTION": [0]}
    return dataset


class TestTrainingConfigAliases:
    def test_optimization_settings_accessible(self):
        c = TrainingConfig()
        assert c.optimization_settings.learning_rate > 0

    def test_data_settings_accessible(self):
        c = TrainingConfig()
        assert c.data_settings.batch_size > 0


class TestMultiTaskTrainerInit:
    def test_init_basic(self, mock_config, mock_dataset):
        from florence_forge.training.trainer import MultiTaskTrainer

        mock_accel = MagicMock()
        mock_accel.is_local_main_process = True
        trainer = MultiTaskTrainer(
            model=SimpleMockModel(),
            train_dataset=mock_dataset,
            config=mock_config,
            accelerator=mock_accel,
        )
        assert trainer.config is mock_config
        assert trainer.training_loop is not None
        assert trainer.checkpoint_manager is not None


class TestTrainerExports:
    def test_default_export_points_to_trainer_module(self):
        import florence_forge
        import florence_forge.training as training
        from florence_forge.training.trainer import MultiTaskTrainer

        assert training.MultiTaskTrainer is MultiTaskTrainer
        assert florence_forge.Trainer is MultiTaskTrainer


class TestCLIResumeFlag:
    def test_resume_flag_in_argparse(self):
        parser = argparse.ArgumentParser()
        train_parser = parser.add_subparsers(dest="command").add_parser("train")
        train_parser.add_argument("--resume", "-r")
        args = train_parser.parse_args(["--resume", "/path/to/checkpoint"])
        assert args.resume == "/path/to/checkpoint"

    def test_load_checkpoint_method_exists(self):
        from florence_forge.training.trainer import MultiTaskTrainer

        assert hasattr(MultiTaskTrainer, "load_checkpoint")


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
