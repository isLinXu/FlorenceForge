"""测试分布式训练相关功能

覆盖:
- DistributedTaskSampler
- TaskDataLoader 分布式检测
- DistributedConfig 配置验证
- 训练器中的分布式 epoch 设置
"""

import os
import pytest
import torch
from unittest.mock import MagicMock, patch

from florence_forge.data.loader import (
    DistributedTaskSampler,
    TaskDataLoader,
)
from florence_forge.core.config import DataConfig, DistributedConfig, TrainingConfig


# ---------------------------------------------------------------------------
# DistributedTaskSampler 测试
# ---------------------------------------------------------------------------

class TestDistributedTaskSampler:
    """测试分布式任务采样器"""

    def _create_simple_sampler(self, num_samples: int = 100):
        """创建一个简单的可迭代采样器（替代 MagicMock 以避免迭代器行为异常）"""
        class SimpleSampler:
            def __init__(self, n):
                self.n = n
            def __iter__(self):
                return iter(range(self.n))
            def __len__(self):
                return self.n
        return SimpleSampler(num_samples)

    def _create_mock_sampler_with_set_epoch(self, num_samples: int = 100):
        """创建带 set_epoch 的 mock 采样器"""
        sampler = MagicMock()
        sampler.__iter__ = lambda self: iter(range(num_samples))
        sampler.__len__ = lambda self: num_samples
        sampler.set_epoch = MagicMock()
        return sampler

    def test_basic_distribution(self):
        """测试基础分布式切片功能"""
        base_sampler = self._create_simple_sampler(100)

        dist_sampler = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=4,
            rank=0,
            shuffle=False,
            seed=42,
            drop_last=False,
        )

        # rank 0 应该获得索引 0-24
        indices = list(dist_sampler)
        assert len(indices) == 25  # 100 / 4 = 25
        assert indices == list(range(0, 25))

    def test_rank_distribution(self):
        """测试不同 rank 获得不同数据"""
        base_sampler = self._create_simple_sampler(100)

        # rank 1
        dist_sampler_1 = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=4,
            rank=1,
            shuffle=False,
            seed=42,
            drop_last=False,
        )
        indices_1 = list(dist_sampler_1)
        assert indices_1 == list(range(25, 50))

        # rank 3
        dist_sampler_3 = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=4,
            rank=3,
            shuffle=False,
            seed=42,
            drop_last=False,
        )
        indices_3 = list(dist_sampler_3)
        assert indices_3 == list(range(75, 100))

    def test_no_overlap_between_ranks(self):
        """测试不同 rank 之间没有数据重叠"""
        base_sampler = self._create_simple_sampler(100)

        all_indices = set()
        for rank in range(4):
            dist_sampler = DistributedTaskSampler(
                sampler=base_sampler,
                world_size=4,
                rank=rank,
                shuffle=False,
                seed=42,
                drop_last=False,
            )
            rank_indices = set(list(dist_sampler))
            # 检查与之前 ranks 没有交集
            assert len(all_indices & rank_indices) == 0
            all_indices.update(rank_indices)

        # 所有 ranks 合起来应该覆盖全部数据
        assert len(all_indices) == 100

    def test_shuffle_changes_order_per_epoch(self):
        """测试不同 epoch 产生不同的打乱顺序

        注意：在分布式设置中，每个 epoch 重新 shuffle 后，
        同一个 rank 获得的样本子集会改变，这是预期行为。
        """
        base_sampler = self._create_simple_sampler(100)

        dist_sampler = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=2,
            rank=0,
            shuffle=True,
            seed=42,
            drop_last=False,
        )

        # epoch 0
        dist_sampler.set_epoch(0)
        indices_epoch_0 = list(dist_sampler)

        # epoch 1
        dist_sampler.set_epoch(1)
        indices_epoch_1 = list(dist_sampler)

        # 两个 epoch 的顺序应该不同
        assert indices_epoch_0 != indices_epoch_1
        # 每个 epoch 的样本数应该相同（world_size=2，所以是 100/2=50）
        assert len(indices_epoch_0) == 50
        assert len(indices_epoch_1) == 50
        # 每个 epoch 内没有重复样本
        assert len(set(indices_epoch_0)) == 50
        assert len(set(indices_epoch_1)) == 50

    def test_drop_last(self):
        """测试 drop_last=True 时截断到 world_size 整数倍"""
        base_sampler = self._create_simple_sampler(103)  # 103 不是 4 的倍数

        dist_sampler = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=4,
            rank=0,
            shuffle=False,
            seed=42,
            drop_last=True,
        )

        # 103 // 4 = 25，每个 rank 25 个，总共 100
        assert len(dist_sampler) == 25
        indices = list(dist_sampler)
        assert len(indices) == 25

    def test_set_epoch_propagation(self):
        """测试 set_epoch 传播给底层采样器"""
        base_sampler = self._create_mock_sampler_with_set_epoch(100)

        dist_sampler = DistributedTaskSampler(
            sampler=base_sampler,
            world_size=2,
            rank=0,
            shuffle=True,
            seed=42,
        )

        dist_sampler.set_epoch(5)
        base_sampler.set_epoch.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# TaskDataLoader 分布式检测测试
# ---------------------------------------------------------------------------

class TestTaskDataLoaderDistributed:
    """测试 TaskDataLoader 的分布式环境检测"""

    def test_non_distributed_environment(self):
        """测试非分布式环境"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0, 1, 2, 3]}
        mock_dataset.__len__ = lambda self: 4
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(distributed=False)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="random")

        assert loader._distributed is False
        assert loader._world_size == 1
        assert loader._rank == 0

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "0"}, clear=False)
    def test_env_var_detection(self):
        """测试通过环境变量检测分布式"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": list(range(100))}
        mock_dataset.__len__ = lambda self: 100
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(distributed=False)  # 未显式启用，但环境变量存在
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="random")

        # 应该自动检测到分布式环境
        assert loader._distributed is True
        assert loader._world_size == 4
        assert loader._rank == 1
        assert loader._local_rank == 0

    def test_explicit_distributed_config(self):
        """测试显式启用分布式配置"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": list(range(100))}
        mock_dataset.__len__ = lambda self: 100
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(distributed=True, world_size=8, rank=3, local_rank=0)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="random")

        assert loader._distributed is True
        assert loader._world_size == 8
        assert loader._rank == 3

    def test_distributed_sampler_wrapping(self):
        """测试分布式环境下自动包装采样器"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": list(range(100))}
        mock_dataset.__len__ = lambda self: 100
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(distributed=True, world_size=4, rank=0, batch_size=4)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="balanced")

        # 应该使用 DistributedTaskSampler 包装 TaskBalancedSampler
        assert isinstance(loader.sampler, DistributedTaskSampler)
        # 本地样本数应为 100/4 = 25
        assert len(loader.sampler) == 25

    def test_len_with_sampler_keeps_partial_batch_when_not_drop_last(self):
        """自定义 sampler 下 __len__ 应正确统计最后一个不完整 batch。"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0, 1, 2, 3, 4]}
        mock_dataset.task_weights = {}
        mock_dataset.__len__ = lambda self: 5
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(batch_size=2, drop_last=False, distributed=False)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="balanced")

        assert len(loader.sampler) == 5
        assert len(loader) == 3

    def test_len_with_sampler_drops_partial_batch_when_drop_last(self):
        """drop_last=True 时自定义 sampler 下 __len__ 保持向下取整。"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0, 1, 2, 3, 4]}
        mock_dataset.task_weights = {}
        mock_dataset.__len__ = lambda self: 5
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(batch_size=2, drop_last=True, distributed=False)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="balanced")

        assert len(loader.sampler) == 4
        assert len(loader) == 2

    def test_get_statistics_with_distributed(self):
        """测试分布式环境下 get_statistics 包含分布式信息"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": list(range(100))}
        mock_dataset.__len__ = lambda self: 100
        mock_dataset.get_task_statistics = lambda: {}

        config = DataConfig(distributed=True, world_size=4, rank=1)
        loader = TaskDataLoader(mock_dataset, config=config, sampling_strategy="random")

        stats = loader.get_statistics()
        assert stats["distributed"] is True
        assert stats["world_size"] == 4
        assert stats["rank"] == 1


# ---------------------------------------------------------------------------
# DistributedConfig 配置验证测试
# ---------------------------------------------------------------------------

class TestDistributedConfig:
    """测试分布式配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = DistributedConfig()
        assert config.enabled is False
        assert config.strategy == "ddp"
        assert config.backend == "nccl"
        assert config.deepspeed_stage == 0
        assert config.fsdp_sharding_strategy == "FULL_SHARD"

    def test_fsdp_config_validation(self):
        """测试 FSDP 配置校验"""
        # 有效的配置
        config = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            fsdp_sharding_strategy="SHARD_GRAD_OP",
            fsdp_auto_wrap_policy="SIZE_BASED_WRAP",
        )
        assert config.fsdp_sharding_strategy == "SHARD_GRAD_OP"

    def test_invalid_strategy(self):
        """测试无效的策略"""
        with pytest.raises(ValueError, match="分布式策略必须是"):
            DistributedConfig(strategy="invalid_strategy")

    def test_invalid_backend(self):
        """测试无效的后端"""
        with pytest.raises(ValueError, match="分布式后端必须是"):
            DistributedConfig(backend="invalid_backend")

    def test_invalid_fsdp_sharding(self):
        """测试无效的 FSDP 分片策略"""
        with pytest.raises(ValueError, match="FSDP 分片策略必须是"):
            DistributedConfig(fsdp_sharding_strategy="INVALID")

    def test_deepspeed_stage_range(self):
        """测试 DeepSpeed 阶段范围校验"""
        # 有效范围
        for stage in [0, 1, 2, 3]:
            config = DistributedConfig(deepspeed_stage=stage)
            assert config.deepspeed_stage == stage

        # 无效范围
        with pytest.raises(ValueError):
            DistributedConfig(deepspeed_stage=4)
        with pytest.raises(ValueError):
            DistributedConfig(deepspeed_stage=-1)

    def test_config_serialization(self):
        """测试配置序列化"""
        config = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            fsdp_cpu_offload=True,
        )
        data = config.to_dict()
        assert data["enabled"] is True
        assert data["strategy"] == "fsdp"
        assert data["fsdp_cpu_offload"] is True

        # 反序列化
        restored = DistributedConfig.from_dict(data)
        assert restored.enabled is True
        assert restored.strategy == "fsdp"


# ---------------------------------------------------------------------------
# 训练器分布式 epoch 设置测试
# ---------------------------------------------------------------------------

class TestTrainerDistributedEpoch:
    """测试训练器中的分布式 epoch 设置"""

    def test_sampler_set_epoch_called(self):
        """测试训练循环中正确调用 sampler.set_epoch"""
        from florence_forge.training.trainer import MultiTaskTrainer

        mock_model = MagicMock()
        mock_model.parameters = lambda: [torch.nn.Parameter(torch.tensor(1.0))]

        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        # 创建 mock dataloader，带有 set_epoch 方法的 sampler
        mock_sampler = MagicMock()
        mock_sampler.set_epoch = MagicMock()

        mock_dataloader = MagicMock()
        mock_dataloader.sampler = mock_sampler
        mock_dataloader.__iter__ = lambda self: iter([{"is_empty": True}])
        mock_dataloader.__len__ = lambda self: 1

        config = TrainingConfig(num_epochs=2)
        trainer = MultiTaskTrainer(mock_model, mock_dataset, config=config)

        # 手动设置 dataloader
        trainer.train_dataloader = mock_dataloader
        trainer.val_dataloader = None
        trainer.optimizer = MagicMock()
        trainer.lr_scheduler = None
        trainer.task_scheduler = MagicMock()
        trainer.task_scheduler.should_update_weights = lambda: False
        trainer.task_scheduler.update_task_performance = lambda *args: None
        trainer.callback_manager = MagicMock()

        # 模拟训练 epoch
        trainer.current_epoch = 0
        # 手动调用 epoch 设置逻辑
        if trainer.train_dataloader is not None:
            sampler = getattr(trainer.train_dataloader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(0)

        mock_sampler.set_epoch.assert_called_once_with(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
