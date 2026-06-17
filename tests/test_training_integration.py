"""训练流程集成测试

测试完整训练流程的各个组件协同工作
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import torch
import torch.nn as nn

from florence_forge.training.checkpoint_manager import CheckpointManager
from florence_forge.training.training_loop import TrainingLoop
from florence_forge.training.device_config import DeviceConfigurator
from florence_forge.training.gradient_checkpoint_optimizer import GradientCheckpointOptimizer
from florence_forge.core.config import TrainingConfig


@pytest.fixture
def training_config():
    """创建测试训练配置"""
    config = TrainingConfig()
    config.output_dir = tempfile.mkdtemp()
    config.num_epochs = 2
    config.batch_size = 2
    config.learning_rate = 1e-4
    config.device = "cpu"
    config.gradient_checkpointing = False
    return config


@pytest.fixture
def mock_model():
    """创建 mock 模型"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model


@pytest.fixture
def mock_dataloader():
    """创建 mock 数据加载器"""
    # 创建简单的数据批次
    batches = [
        {
            'input_ids': torch.randn(2, 10),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 10, (2,)),
            'task_type': ['CAPTION', 'CAPTION']
        }
        for _ in range(3)
    ]
    
    class MockDataLoader:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    return MockDataLoader(batches)


class TestCheckpointManager:
    """测试检查点管理器"""
    
    def test_save_checkpoint(self, training_config, mock_model):
        """测试检查点保存"""
        manager = CheckpointManager(
            model=mock_model,
            config=training_config,
            accelerator=None
        )
        
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 保存检查点（同步模式）
        manager.save_checkpoint(
            epoch=0,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics={'loss': 1.0},
            is_best=True,
            async_save=False
        )
        
        # 验证检查点文件存在
        checkpoint_dir = Path(training_config.output_dir) / "checkpoint-epoch-0"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "checkpoint.pt").exists()
        assert (checkpoint_dir / "BEST_MODEL").exists()
    
    def test_load_checkpoint(self, training_config, mock_model):
        """测试检查点加载"""
        manager = CheckpointManager(
            model=mock_model,
            config=training_config,
            accelerator=None
        )
        
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 保存检查点
        manager.save_checkpoint(
            epoch=5,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics={'loss': 0.5},
            is_best=False,
            async_save=False
        )
        
        # 创建新模型和优化器
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # 加载检查点
        new_manager = CheckpointManager(
            model=new_model,
            config=training_config,
            accelerator=None
        )
        
        checkpoint_dir = Path(training_config.output_dir) / "checkpoint-epoch-5"
        metadata = new_manager.load_checkpoint(
            checkpoint_path=checkpoint_dir,
            optimizer=new_optimizer
        )
        
        assert metadata['epoch'] == 5
        assert 'loss' in metadata['metrics']
    
    def test_checkpoint_cleanup(self, training_config, mock_model):
        """测试检查点清理"""
        training_config.keep_checkpoints = 2
        
        manager = CheckpointManager(
            model=mock_model,
            config=training_config,
            accelerator=None
        )
        
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 保存多个检查点
        for epoch in range(5):
            manager.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                metrics={'loss': 1.0 / (epoch + 1)},
                is_best=False,
                async_save=False
            )
        
        # 验证只保留最近的 N 个
        output_dir = Path(training_config.output_dir)
        checkpoints = list(output_dir.glob("checkpoint-epoch-*"))
        assert len(checkpoints) <= training_config.keep_checkpoints + 1  # +1 for potential best model


class TestTrainingLoop:
    """测试训练循环"""
    
    def test_train_epoch(self, training_config, mock_model, mock_dataloader):
        """测试训练一个 epoch"""
        # 修改模型使其输出正确格式
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.CrossEntropyLoss()
            
            def forward(self, input_ids, attention_mask, labels, **kwargs):
                output = self.base_model(input_ids)
                loss = self.loss_fn(output, labels)
                return type('Outputs', (), {'loss': loss})()
        
        wrapped_model = ModelWrapper(mock_model)
        
        loop = TrainingLoop(
            model=wrapped_model,
            config=training_config,
            accelerator=None
        )
        
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 执行训练
        metrics = loop.train_epoch(
            train_dataloader=mock_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0
        )
        
        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        assert isinstance(metrics['loss'], float)
    
    def test_validate_epoch(self, training_config, mock_model, mock_dataloader):
        """测试验证一个 epoch"""
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.CrossEntropyLoss()
            
            def forward(self, input_ids, attention_mask, labels, **kwargs):
                output = self.base_model(input_ids)
                loss = self.loss_fn(output, labels)
                return type('Outputs', (), {'loss': loss})()
        
        wrapped_model = ModelWrapper(mock_model)
        
        loop = TrainingLoop(
            model=wrapped_model,
            config=training_config,
            accelerator=None
        )
        
        # 执行验证
        metrics = loop.validate_epoch(
            val_dataloader=mock_dataloader,
            epoch=0
        )
        
        assert 'val_loss' in metrics
        assert isinstance(metrics['val_loss'], float)

    def test_train_epoch_filters_dataloader_metadata(self, training_config, mock_model):
        """训练循环调用模型前应移除 task_type/prompt/metadata 等非模型字段。"""
        class StrictModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, input_ids, attention_mask, labels):
                output = self.base_model(input_ids)
                loss = self.loss_fn(output, labels)
                return type('Outputs', (), {'loss': loss})()

        batches = [
            {
                'input_ids': torch.randn(2, 10),
                'attention_mask': torch.ones(2, 10),
                'labels': torch.randint(0, 10, (2,)),
                'task_type': 'CAPTION',
                'prompt': ['<CAPTION>', '<CAPTION>'],
                'metadata': [{'id': 1}, {'id': 2}],
            }
        ]

        class MockDataLoader:
            def __iter__(self):
                return iter(batches)

            def __len__(self):
                return len(batches)

        wrapped_model = StrictModelWrapper(mock_model)
        loop = TrainingLoop(
            model=wrapped_model,
            config=training_config,
            accelerator=None
        )
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        metrics = loop.train_epoch(
            train_dataloader=MockDataLoader(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0
        )

        assert metrics['loss'] > 0.0
        assert metrics['task_CAPTION_loss'] > 0.0

    def test_train_epoch_skips_step_when_gradient_validation_fails(self, training_config):
        """梯度验证失败时清梯度并跳过 optimizer/lr_scheduler step。"""
        class LossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                return type('Outputs', (), {'loss': self.weight * input_ids.sum()})()

        batches = [{
            'input_ids': torch.ones(2, 10),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.ones(2, dtype=torch.long),
            'task_type': 'CAPTION',
        }]

        class MockDataLoader:
            def __iter__(self):
                return iter(batches)

            def __len__(self):
                return len(batches)

        model = LossModel()
        loop = TrainingLoop(model=model, config=training_config, accelerator=None)
        optimizer = MagicMock()
        optimizer.param_groups = [{'lr': 0.001}]
        lr_scheduler = MagicMock()
        lr_scheduler.get_last_lr.return_value = [0.001]
        gradient_validator = MagicMock()
        gradient_validator.validate_gradients.return_value = (False, {'reason': 'nan'})

        metrics = loop.train_epoch(
            train_dataloader=MockDataLoader(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            gradient_validator=gradient_validator,
        )

        gradient_validator.validate_gradients.assert_called_once_with(0)
        optimizer.step.assert_not_called()
        lr_scheduler.step.assert_not_called()
        optimizer.zero_grad.assert_called_once()
        assert metrics['loss'] > 0.0
    
    def test_max_steps_stops_epoch_early(self, training_config, mock_model):
        """max_steps 达到后应提前结束当前 epoch（v1/v2 对齐）。"""
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, input_ids, attention_mask, labels, **kwargs):
                output = self.base_model(input_ids)
                loss = self.loss_fn(output, labels)
                return type('Outputs', (), {'loss': loss})()

        # 构造一个 10 个 batch 的 dataloader，但 max_steps 限制为 3 步
        batches = [
            {
                'input_ids': torch.randn(2, 10),
                'attention_mask': torch.ones(2, 10),
                'labels': torch.randint(0, 10, (2,)),
                'task_type': 'CAPTION',
            }
            for _ in range(10)
        ]

        class MockDataLoader:
            def __iter__(self):
                return iter(batches)

            def __len__(self):
                return len(batches)

        training_config.max_steps = 3
        wrapped_model = ModelWrapper(mock_model)
        loop = TrainingLoop(model=wrapped_model, config=training_config, accelerator=None)
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        loop.train_epoch(
            train_dataloader=MockDataLoader(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
        )

        assert loop.global_step == 3
        assert loop._max_steps_reached()

    def test_train_epoch_skips_batch_with_no_supervised_labels(self, training_config):
        """labels 全为 -100 时不应反传或更新参数。"""
        class NanLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                raise AssertionError("forward should not run when labels are all masked")

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.full((1, 3), -100),
            "task_type": "CAPTION",
        }

        class SingleBatchLoader:
            def __iter__(self):
                return iter([batch])

            def __len__(self):
                return 1

        model = NanLossModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loop = TrainingLoop(model=model, config=training_config, accelerator=None)
        metrics = loop.train_epoch(
            train_dataloader=SingleBatchLoader(),
            optimizer=optimizer,
            lr_scheduler=None,
            epoch=0,
        )
        assert loop.global_step == 0
        assert metrics["loss"] == 0.0

    def test_train_epoch_skips_non_finite_loss(self, training_config):
        """loss 为 nan/inf 时跳过反传，global_step 不增加。"""
        class NonFiniteLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                return type("Out", (), {"loss": torch.tensor(float("nan"))})()

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[0, 1, 2]]),
            "task_type": "CAPTION",
        }

        class SingleBatchLoader:
            def __iter__(self):
                return iter([batch])

            def __len__(self):
                return 1

        model = NonFiniteLossModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loop = TrainingLoop(model=model, config=training_config, accelerator=None)
        metrics = loop.train_epoch(
            train_dataloader=SingleBatchLoader(),
            optimizer=optimizer,
            lr_scheduler=None,
            epoch=0,
        )
        assert loop.global_step == 0
        assert metrics["loss"] == 0.0

    def test_max_steps_reached_false_when_unset(self, training_config, mock_model):
        """未设定 max_steps 时不应触发提前终止。"""
        training_config.max_steps = None
        loop = TrainingLoop(model=mock_model, config=training_config, accelerator=None)
        loop.global_step = 999
        assert not loop._max_steps_reached()

    def test_early_stopping(self, training_config, mock_model):
        """测试早停逻辑"""
        loop = TrainingLoop(
            model=mock_model,
            config=training_config,
            accelerator=None
        )
        
        # 模拟指标变化
        assert not loop.should_early_stop(1.0, patience=3)
        assert not loop.should_early_stop(0.9, patience=3)  # 改善
        assert not loop.should_early_stop(0.95, patience=3)  # 未改善
        assert not loop.should_early_stop(0.96, patience=3)  # 未改善
        assert loop.should_early_stop(0.97, patience=3)  # 触发早停


class TestDeviceConfigurator:
    """测试设备配置器"""
    
    def test_device_detection(self, training_config):
        """测试设备检测"""
        configurator = DeviceConfigurator(training_config)
        device = configurator.setup_device()
        
        assert device in ['cuda', 'mps', 'cpu']
    
    def test_mixed_precision_selection(self, training_config):
        """测试混合精度选择"""
        configurator = DeviceConfigurator(training_config)
        configurator.setup_device()
        
        mixed_precision = configurator.determine_mixed_precision()
        assert mixed_precision in ['no', 'fp16', 'bf16']


class TestGradientCheckpointOptimizer:
    """测试梯度检查点优化器"""
    
    def test_auto_strategy_selection(self, training_config, mock_model):
        """测试策略自动选择"""
        optimizer = GradientCheckpointOptimizer(
            model=mock_model,
            config=training_config
        )
        
        strategy = optimizer._auto_select_strategy()
        assert strategy in ['full', 'selective']
    
    def test_enable_gradient_checkpointing(self, training_config):
        """测试启用梯度检查点"""
        # 创建带 gradient_checkpointing_enable 方法的模型
        class MockModelWithCheckpointing(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self._checkpointing_enabled = False
            
            def gradient_checkpointing_enable(self):
                self._checkpointing_enabled = True
        
        model = MockModelWithCheckpointing()
        training_config.gradient_checkpointing = True
        
        optimizer = GradientCheckpointOptimizer(
            model=model,
            config=training_config
        )
        
        optimizer.enable_gradient_checkpointing()
        assert model._checkpointing_enabled


class TestIntegration:
    """集成测试：测试多个组件协同工作"""
    
    def test_full_training_workflow(self, training_config, mock_model, mock_dataloader):
        """测试完整训练工作流"""
        # 1. 设备配置
        device_config = DeviceConfigurator(training_config)
        device = device_config.setup_device()
        
        # 2. 梯度检查点
        grad_optimizer = GradientCheckpointOptimizer(
            model=mock_model,
            config=training_config
        )
        grad_optimizer.enable_gradient_checkpointing()
        
        # 3. 检查点管理器
        checkpoint_manager = CheckpointManager(
            model=mock_model,
            config=training_config,
            accelerator=None
        )
        
        # 4. 训练循环
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.CrossEntropyLoss()
            
            def forward(self, input_ids, attention_mask, labels, **kwargs):
                output = self.base_model(input_ids)
                loss = self.loss_fn(output, labels)
                return type('Outputs', (), {'loss': loss})()
        
        wrapped_model = ModelWrapper(mock_model)
        training_loop = TrainingLoop(
            model=wrapped_model,
            config=training_config,
            accelerator=None
        )
        
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=training_config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 执行一个完整的训练周期
        for epoch in range(2):
            # 训练
            train_metrics = training_loop.train_epoch(
                train_dataloader=mock_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch
            )
            
            # 验证
            val_metrics = training_loop.validate_epoch(
                val_dataloader=mock_dataloader,
                epoch=epoch
            )
            
            # 保存检查点
            is_best = epoch == 1  # 假设第二个 epoch 最好
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                metrics={**train_metrics, **val_metrics},
                is_best=is_best,
                async_save=False
            )
        
        # 验证结果
        output_dir = Path(training_config.output_dir)
        assert (output_dir / "checkpoint-epoch-0").exists()
        assert (output_dir / "checkpoint-epoch-1").exists()
        assert (output_dir / "checkpoint-epoch-1" / "BEST_MODEL").exists()


class TestLoadBestModelAtEnd:
    """测试 v2 trainer 的 load_best_model_at_end 行为"""

    def _build_trainer(self, training_config, mock_model):
        from florence_forge.training.trainer import (
            MultiTaskTrainer as TrainerV2,
        )

        return TrainerV2(
            model=mock_model,
            train_dataset=MagicMock(),
            val_dataset=None,
            config=training_config,
            accelerator=MagicMock(),
        )

    def test_restores_best_checkpoint_when_enabled(self, training_config, mock_model):
        training_config.load_best_model_at_end = True
        trainer = self._build_trainer(training_config, mock_model)

        best_dir = Path(training_config.output_dir) / "best-ckpt"
        best_dir.mkdir(parents=True, exist_ok=True)

        trainer.checkpoint_manager = MagicMock()
        trainer.checkpoint_manager.get_best_checkpoint_path.return_value = best_dir

        trainer._maybe_restore_best_model()

        trainer.checkpoint_manager.load_checkpoint.assert_called_once()

    def test_noop_when_disabled(self, training_config, mock_model):
        training_config.load_best_model_at_end = False
        trainer = self._build_trainer(training_config, mock_model)

        trainer.checkpoint_manager = MagicMock()
        trainer._maybe_restore_best_model()

        trainer.checkpoint_manager.load_checkpoint.assert_not_called()

    def test_noop_when_no_best_checkpoint(self, training_config, mock_model):
        training_config.load_best_model_at_end = True
        trainer = self._build_trainer(training_config, mock_model)

        trainer.checkpoint_manager = MagicMock()
        trainer.checkpoint_manager.get_best_checkpoint_path.return_value = None

        trainer._maybe_restore_best_model()

        trainer.checkpoint_manager.load_checkpoint.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
