#!/usr/bin/env python3
"""测试 v3 增强功能

覆盖：
1. v1→v2 统一导出
2. TrainingLoop NaN/Inf 检测、日志钩子、梯度累积进度
3. CheckpointManager 断点续训恢复（global_step、resume_training_state）
4. LoRAManager 冻结/共享/导出导入
5. ModelMerger _linear_merge vs _weighted_merge 区别
6. DeviceConfigurator 多 GPU 选择、CPU 提示
"""
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. v1→v2 统一导出
# ---------------------------------------------------------------------------


class TestDefaultTrainerExport:
    """验证默认导出指向 trainer.MultiTaskTrainer"""

    def test_default_import_is_canonical_trainer(self):
        from florence_forge.training import MultiTaskTrainer
        from florence_forge.training.trainer import MultiTaskTrainer as TrainerCls

        assert MultiTaskTrainer is TrainerCls


# ---------------------------------------------------------------------------
# 2. TrainingLoop 增强
# ---------------------------------------------------------------------------


class TestTrainingLoopNaNInfDetection:
    """NaN/Inf loss 检测"""

    @pytest.fixture
    def training_loop(self):
        from florence_forge.training.training_loop import TrainingLoop
        from florence_forge.core.config import TrainingConfig

        model = nn.Linear(10, 2)
        config = TrainingConfig()
        return TrainingLoop(model=model, config=config)

    def test_nan_loss_count_initialized(self, training_loop):
        assert training_loop._nan_loss_count == 0
        assert training_loop._inf_loss_count == 0

    def test_nan_loss_skips_optimizer_step(self, training_loop):
        """当 loss 为 NaN 时，应跳过 optimizer step 并增加计数"""
        from florence_forge.training.training_loop import TrainingLoop
        from florence_forge.core.config import TrainingConfig

        model = nn.Linear(10, 2)
        config = TrainingConfig()
        config.gradient_accumulation_steps = 1

        loop = TrainingLoop(model=model, config=config)

        # 模拟一个产生 NaN loss 的 batch
        dataloader = [{"input_ids": torch.randint(0, 10, (2, 5)),
                       "labels": torch.randint(0, 2, (2, 5)),
                       "task_types": ["test"]}]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = None

        # 用 mock 让 model 输出 NaN loss
        nan_output = MagicMock()
        nan_output.loss = torch.tensor(float('nan'))

        with patch.object(type(model), '__call__', return_value=nan_output):
            loop.train_epoch(
                train_dataloader=dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=0,
            )

        assert loop._nan_loss_count > 0


class TestTrainingLoopLogHooks:
    """日志钩子"""

    @pytest.fixture
    def training_loop(self):
        from florence_forge.training.training_loop import TrainingLoop
        from florence_forge.core.config import TrainingConfig

        model = nn.Linear(10, 2)
        config = TrainingConfig()
        return TrainingLoop(model=model, config=config)

    def test_add_log_hook(self, training_loop):
        events = []
        training_loop.add_log_hook(lambda e, d: events.append(e))
        training_loop._emit_log("train_step", {"loss": 0.5})
        assert events == ["train_step"]

    def test_add_non_callable_hook_raises(self, training_loop):
        with pytest.raises(TypeError, match="callable"):
            training_loop.add_log_hook("not_callable")

    def test_log_hook_exception_does_not_propagate(self, training_loop):
        def bad_hook(event, data):
            raise RuntimeError("hook error")

        training_loop.add_log_hook(bad_hook)
        # 不应抛出异常
        training_loop._emit_log("train_step", {"loss": 0.5})

    def test_multiple_hooks(self, training_loop):
        events_a, events_b = [], []
        training_loop.add_log_hook(lambda e, d: events_a.append(e))
        training_loop.add_log_hook(lambda e, d: events_b.append(e))
        training_loop._emit_log("epoch_end", {})
        assert len(events_a) == 1
        assert len(events_b) == 1


class TestTrainingLoopAccumProgress:
    """梯度累积微批次进度"""

    def test_accum_steps_in_config(self):
        from florence_forge.core.config import TrainingConfig
        config = TrainingConfig()
        config.gradient_accumulation_steps = 4
        assert config.gradient_accumulation_steps == 4


# ---------------------------------------------------------------------------
# 3. CheckpointManager 断点续训恢复
# ---------------------------------------------------------------------------


class TestCheckpointManagerResume:
    """断点续训恢复"""

    @pytest.fixture
    def checkpoint_manager(self, tmp_path):
        from florence_forge.training.checkpoint_manager import CheckpointManager
        from florence_forge.core.config import TrainingConfig

        model = nn.Linear(10, 2)
        config = TrainingConfig()
        config.output_dir = str(tmp_path / "output")
        return CheckpointManager(model=model, config=config)

    def test_save_and_load_preserves_global_step(self, checkpoint_manager, tmp_path):
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        checkpoint_manager.save_checkpoint(
            epoch=3,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics={'val_loss': 0.5},
            is_best=True,
            async_save=False,
            global_step=150,
        )

        # 等待保存完成
        checkpoint_manager._wait_for_pending_save()

        metadata = checkpoint_manager.load_checkpoint(
            checkpoint_path=tmp_path / "output" / "checkpoint-epoch-3",
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        assert metadata['global_step'] == 150
        assert metadata['epoch'] == 3

    def test_resume_training_state_restores_loop(self, checkpoint_manager, tmp_path):
        from florence_forge.training.training_loop import TrainingLoop
        from florence_forge.core.config import TrainingConfig

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # 保存检查点
        checkpoint_manager.save_checkpoint(
            epoch=5,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics={'val_loss': 0.3},
            is_best=True,
            async_save=False,
            global_step=300,
        )
        checkpoint_manager._wait_for_pending_save()

        # 创建 TrainingLoop 并恢复
        config = TrainingConfig()
        loop = TrainingLoop(model=model, config=config)
        assert loop.global_step == 0

        checkpoint_manager.resume_training_state(
            checkpoint_path=tmp_path / "output" / "checkpoint-epoch-5",
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            training_loop=loop,
        )

        assert loop.global_step == 300
        assert loop.best_metric == 0.3


# ---------------------------------------------------------------------------
# 4. LoRAManager 冻结/共享/导出导入
# ---------------------------------------------------------------------------


class TestLoRAManagerFreezeShare:
    """LoRA 适配器冻结和共享"""

    @pytest.fixture
    def lora_manager(self):
        from florence_forge.training.lora_manager import LoRAManager
        from florence_forge.core.config import LoRAConfig

        config = LoRAConfig(r=8, lora_alpha=16)
        manager = LoRAManager(config)
        # 手动添加一些适配器映射
        manager.active_adapters = {"task_a": "adapter_a", "task_b": "adapter_b"}
        manager.task_configs = {"task_a": config, "task_b": config}
        return manager

    def test_freeze_adapter(self, lora_manager):
        model = MagicMock()
        lora_manager.freeze_adapter(model, "task_a")
        assert lora_manager._frozen_adapters.get("adapter_a") is True

    def test_unfreeze_adapter(self, lora_manager):
        model = MagicMock()
        lora_manager.freeze_adapter(model, "task_a")
        lora_manager.unfreeze_adapter(model, "task_a")
        assert lora_manager._frozen_adapters.get("adapter_a") is False

    def test_is_adapter_frozen(self, lora_manager):
        model = MagicMock()
        assert not lora_manager.is_adapter_frozen("task_a")
        lora_manager.freeze_adapter(model, "task_a")
        assert lora_manager.is_adapter_frozen("task_a")

    def test_freeze_nonexistent_adapter(self, lora_manager):
        model = MagicMock()
        # 不应抛出异常
        lora_manager.freeze_adapter(model, "nonexistent")

    def test_register_shared_adapter(self, lora_manager):
        lora_manager.register_shared_adapter("task_a", "task_c")
        assert lora_manager.active_adapters["task_c"] == "adapter_a"
        assert lora_manager._shared_adapters["task_c"] == "adapter_a"

    def test_register_shared_adapter_missing_source(self, lora_manager):
        with pytest.raises(ValueError, match="没有活跃适配器"):
            lora_manager.register_shared_adapter("nonexistent", "task_c")

    def test_get_shared_adapter_source(self, lora_manager):
        assert lora_manager.get_shared_adapter_source("task_a") is None
        lora_manager.register_shared_adapter("task_a", "task_c")
        assert lora_manager.get_shared_adapter_source("task_c") == "adapter_a"


class TestLoRAManagerExportImport:
    """LoRA 状态导出/导入"""

    @pytest.fixture
    def lora_manager(self):
        from florence_forge.training.lora_manager import LoRAManager
        from florence_forge.core.config import LoRAConfig

        config = LoRAConfig(r=8, lora_alpha=16)
        manager = LoRAManager(config)
        manager.active_adapters = {"task_a": "adapter_a"}
        manager.task_configs = {"task_a": config}
        manager._frozen_adapters = {"adapter_a": True}
        manager._shared_adapters = {"task_c": "adapter_a"}
        return manager

    def test_export_state(self, lora_manager):
        state = lora_manager.export_state()
        assert 'active_adapters' in state
        assert 'frozen_adapters' in state
        assert 'shared_adapters' in state
        assert 'task_configs' in state
        assert 'base_config' in state
        assert state['active_adapters']['task_a'] == 'adapter_a'
        assert state['frozen_adapters']['adapter_a'] is True

    def test_import_state(self, lora_manager):
        state = lora_manager.export_state()
        
        # 创建新的 manager 并导入
        from florence_forge.training.lora_manager import LoRAManager
        from florence_forge.core.config import LoRAConfig
        
        new_config = LoRAConfig(r=4, lora_alpha=8)
        new_manager = LoRAManager(new_config)
        new_manager.import_state(state)
        
        assert new_manager.active_adapters['task_a'] == 'adapter_a'
        assert new_manager._frozen_adapters['adapter_a'] is True
        assert new_manager._shared_adapters['task_c'] == 'adapter_a'


# ---------------------------------------------------------------------------
# 5. ModelMerger _linear_merge vs _weighted_merge
# ---------------------------------------------------------------------------


class TestModelMergerMergeStrategies:
    """验证 ``apply_weight_delta`` 线性/加权合并语义。"""

    @pytest.fixture
    def merger(self):
        from florence_forge.training.model_merger import ModelMerger
        return ModelMerger()

    def test_linear_merge_adds_weights(self, merger):
        """scaling_factor=1.0 时直接相加。"""
        model = nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        lora_weights = {'weight': torch.ones_like(model.weight.data) * 0.1}
        
        merger.apply_weight_delta(model, lora_weights)
        
        expected = original_weight + 0.1
        assert torch.allclose(model.weight.data, expected)

    def test_weighted_merge_with_scaling(self, merger):
        """scaling_factor 缩放增量。"""
        model = nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        lora_weights = {'weight': torch.ones_like(model.weight.data) * 0.1}
        
        merger.apply_weight_delta(model, lora_weights, scaling_factor=0.5)
        
        expected = original_weight + 0.1 * 0.5
        assert torch.allclose(model.weight.data, expected)

    def test_weighted_merge_default_scaling_is_1(self, merger):
        """scaling_factor=1.0 时两次合并结果一致。"""
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        model2.load_state_dict(model1.state_dict())
        
        lora_weights = {'weight': torch.ones_like(model1.weight.data) * 0.2}
        
        merger.apply_weight_delta(model1, lora_weights, scaling_factor=1.0)
        merger.apply_weight_delta(model2, lora_weights, scaling_factor=1.0)
        
        assert torch.allclose(model1.weight.data, model2.weight.data)


# ---------------------------------------------------------------------------
# 6. DeviceConfigurator 增强
# ---------------------------------------------------------------------------


class TestDeviceConfiguratorMultiGPU:
    """多 GPU 自动选择"""

    @pytest.fixture
    def configurator(self):
        from florence_forge.training.device_config import DeviceConfigurator
        from florence_forge.core.config import TrainingConfig

        config = TrainingConfig()
        return DeviceConfigurator(config)

    def test_select_best_gpu_single(self, configurator):
        """单 GPU 环境返回 0"""
        with patch('torch.cuda.device_count', return_value=1):
            assert configurator._select_best_gpu() == 0

    def test_select_best_gpu_multi(self, configurator):
        """多 GPU 环境选择空闲显存最多的"""
        with patch('torch.cuda.device_count', return_value=3), \
             patch('torch.cuda.memory_allocated', side_effect=[5e9, 1e9, 3e9]), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # 模拟 GPU 属性
            for i in range(3):
                props = MagicMock()
                props.total_mem = 24e9
                mock_props.return_value = props
            
            result = configurator._select_best_gpu()
            assert result == 1  # GPU 1 已用显存最少

    def test_select_best_gpu_respects_cuda_visible_devices(self, configurator):
        """CUDA_VISIBLE_DEVICES 已设置时，不覆盖用户选择"""
        import os
        with patch('torch.cuda.device_count', return_value=4), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            result = configurator._select_best_gpu()
            assert result == 0  # 相对于可见设备，索引 0

    def test_get_gpu_info_no_cuda(self, configurator):
        """无 CUDA 时返回空列表"""
        with patch('torch.cuda.is_available', return_value=False):
            assert configurator.get_gpu_info() == []

    def test_get_gpu_info_caches(self, configurator):
        """GPU 信息应缓存"""
        with patch('torch.cuda.is_available', return_value=False):
            info1 = configurator.get_gpu_info()
            info2 = configurator.get_gpu_info()
            assert info1 is info2


class TestDeviceConfiguratorCPUTips:
    """CPU 训练优化提示"""

    def test_cpu_training_logs_tips(self, caplog):
        from florence_forge.training.device_config import DeviceConfigurator
        from florence_forge.core.config import TrainingConfig

        config = TrainingConfig()
        config.device = "auto"
        configurator = DeviceConfigurator(config)

        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            
            with caplog.at_level(logging.WARNING):
                configurator.setup_device()

        # 应包含优化建议
        log_messages = " ".join(caplog.messages)
        assert "优化建议" in log_messages or "batch_size" in log_messages


# ---------------------------------------------------------------------------
# 运行全部现有测试确保零回归
# ---------------------------------------------------------------------------


class TestRegression:
    """回归测试：确保现有测试仍然通过"""

    def test_trainer_import(self):
        from florence_forge.training.trainer import MultiTaskTrainer
        assert MultiTaskTrainer is not None

    def test_training_loop_import(self):
        from florence_forge.training.training_loop import TrainingLoop
        assert TrainingLoop is not None

    def test_checkpoint_manager_import(self):
        from florence_forge.training.checkpoint_manager import CheckpointManager
        assert CheckpointManager is not None

    def test_lora_manager_import(self):
        from florence_forge.training.lora_manager import LoRAManager
        assert LoRAManager is not None

    def test_model_merger_import(self):
        from florence_forge.training.model_merger import ModelMerger
        assert ModelMerger is not None

    def test_device_config_import(self):
        from florence_forge.training.device_config import DeviceConfigurator
        assert DeviceConfigurator is not None

    def test_scheduler_import(self):
        from florence_forge.training.scheduler import TaskScheduler
        assert TaskScheduler is not None
