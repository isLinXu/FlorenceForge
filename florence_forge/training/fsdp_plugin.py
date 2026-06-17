#!/usr/bin/env python3
"""FSDP 分布式训练插件

职责：
- 自动检测 FSDP 可用性（PyTorch >= 2.0 + CUDA + NCCL）
- 配置 FSDP 策略（完全分片、梯度分片、包装策略）
- 与 Accelerate 桥接
"""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..core.config import TrainingConfig, DistributedConfig

logger = logging.getLogger(__name__)

__all__ = ["FSDPPlugin"]


class FSDPPlugin:
    """FSDP 插件

    特性：
    - 自动配置 FSDP 策略
    - 无缝集成 Accelerate
    - 支持 CPU offload
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._fsdp_available = self._check_fsdp_availability()

    def _check_fsdp_availability(self) -> bool:
        """检测 FSDP 是否可用（PyTorch >= 2.0 + CUDA + NCCL）。"""
        try:
            import torch.distributed as dist

            version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            return (
                version >= (2, 0)
                and torch.cuda.is_available()
                and hasattr(dist, "is_nccl_available")
                and dist.is_nccl_available()
            )
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._fsdp_available

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure_fsdp(self, config: "TrainingConfig") -> Dict[str, Any]:
        """根据训练配置生成 FSDP 策略字典。"""
        dist_config = config.distributed_settings
        return {
            "sharding_strategy": dist_config.fsdp_sharding_strategy,
            "auto_wrap_policy": dist_config.fsdp_auto_wrap_policy,
            "backward_prefetch": dist_config.fsdp_backward_prefetch,
            "cpu_offload": dist_config.fsdp_cpu_offload,
            "activation_checkpointing": dist_config.fsdp_activation_checkpointing,
            "min_num_params": dist_config.fsdp_min_num_params,
        }

    def build_accelerate_plugin(self, dist_config: "DistributedConfig"):
        """构建 Accelerate FullyShardedDataParallelPlugin 实例。"""
        if not self.is_available:
            logger.warning("FSDP 不可用（需要 PyTorch 2.0+、CUDA 与 NCCL）")
            return None

        try:
            from accelerate.utils import FullyShardedDataParallelPlugin
            from torch.distributed.fsdp import ShardingStrategy

            sharding_strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
                "full_shard": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
                "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
            }

            sharding_strategy = sharding_strategy_map.get(
                dist_config.fsdp_sharding_strategy,
                ShardingStrategy.FULL_SHARD,
            )

            plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=sharding_strategy,
                cpu_offload=dist_config.fsdp_cpu_offload,
            )
            logger.info("FSDP 插件已配置：%s", dist_config.fsdp_sharding_strategy)
            return plugin
        except ImportError as exc:
            logger.error("FSDP 不可用：%s", exc)
            return None
