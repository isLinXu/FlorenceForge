"""训练器设备和混合精度配置模块

提供设备检测、混合精度配置和分布式训练插件构建
"""
import logging
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


class DeviceConfigurator:
    """设备和混合精度配置器
    
    自动检测最优设备和混合精度设置
    """
    
    def __init__(self, config):
        """初始化配置器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        self.device = None
        self.device_type = None
    
    def setup_device(self) -> str:
        """设置设备
        
        Returns:
            设备类型字符串 ('cuda', 'mps', 'cpu')
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.device_type = "cuda"
                logger.info(f"🚀 自动检测到 CUDA 设备：{torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                self.device_type = "mps"
                logger.info("🍎 自动检测到 Apple MPS 设备")
            else:
                self.device = "cpu"
                self.device_type = "cpu"
                logger.warning("⚠️  未检测到 GPU，使用 CPU 训练（速度较慢）")
        else:
            self.device = self.config.device
            self.device_type = self.device.split(':')[0] if ':' in self.device else self.device
            logger.info(f"使用指定设备：{self.device}")
        
        return self.device_type
    
    def determine_mixed_precision(self) -> str:
        """确定混合精度设置
        
        根据设备类型、PyTorch 版本和配置自动选择混合精度
        
        Returns:
            混合精度模式 ('no', 'fp16', 'bf16')
        """
        # 如果配置中明确指定，直接使用
        if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision != "auto":
            logger.info(f"使用配置指定的混合精度：{self.config.mixed_precision}")
            return self.config.mixed_precision
        
        # 自动检测
        device_type = self.device_type or self.setup_device()
        
        # CUDA 设备：检测 BF16 支持
        if device_type == "cuda":
            try:
                # Ampere (SM80+) 及以上架构支持 BF16
                compute_capability = torch.cuda.get_device_capability()
                major, minor = compute_capability
                
                if major >= 8:  # A100, A6000, RTX 30xx/40xx
                    logger.info(f"✅ 检测到 Ampere+ 架构（SM{major}{minor}），使用 BF16")
                    return "bf16"
                else:  # V100, T4, RTX 20xx
                    logger.info(f"✅ 检测到 CUDA 设备（SM{major}{minor}），使用 FP16")
                    return "fp16"
            except Exception as e:
                logger.warning(f"⚠️  CUDA 架构检测失败，降级到 FP16：{e}")
                return "fp16"
        
        # MPS 设备：PyTorch 2.0+ 支持 FP16
        elif device_type == "mps":
            try:
                pytorch_version = torch.__version__.split('+')[0]
                major, minor = map(int, pytorch_version.split('.')[:2])
                
                if major >= 2:
                    logger.info("✅ PyTorch 2.0+，MPS 支持 FP16")
                    return "fp16"
                else:
                    logger.warning("⚠️  PyTorch < 2.0，MPS 不支持混合精度")
                    return "no"
            except Exception as e:
                logger.warning(f"⚠️  PyTorch 版本检测失败：{e}")
                return "no"
        
        # CPU：不支持混合精度
        else:
            logger.info("CPU 设备，不使用混合精度")
            return "no"
    
    def build_distributed_plugin(self, dist_config):
        """构建分布式训练插件
        
        Args:
            dist_config: 分布式训练配置
        
        Returns:
            分布式插件实例或 None
        """
        if not dist_config.enabled or dist_config.strategy == "none":
            return None
        
        if dist_config.strategy == "fsdp":
            return self._build_fsdp_plugin(dist_config)
        elif dist_config.strategy == "deepspeed":
            return self._build_deepspeed_plugin(dist_config)
        else:
            logger.warning(f"未知的分布式策略：{dist_config.strategy}")
            return None
    
    def _build_fsdp_plugin(self, dist_config):
        """构建 FSDP 插件
        
        Args:
            dist_config: 分布式训练配置
        
        Returns:
            FSDP 插件实例
        """
        try:
            from accelerate.utils import FullyShardedDataParallelPlugin
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            
            # 映射分片策略
            sharding_strategy_map = {
                "full_shard": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
                "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
            }
            
            sharding_strategy = sharding_strategy_map.get(
                dist_config.fsdp_sharding_strategy,
                ShardingStrategy.FULL_SHARD
            )
            
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=sharding_strategy,
                cpu_offload=dist_config.fsdp_cpu_offload,
            )
            
            logger.info(f"🚀 FSDP 插件已配置：{dist_config.fsdp_sharding_strategy}")
            return fsdp_plugin
        except ImportError as e:
            logger.error(f"❌ FSDP 不可用（需要 PyTorch 1.11+）：{e}")
            return None
    
    def _build_deepspeed_plugin(self, dist_config):
        """构建 DeepSpeed 插件
        
        Args:
            dist_config: 分布式训练配置
        
        Returns:
            DeepSpeed 插件实例
        """
        try:
            from accelerate.utils import DeepSpeedPlugin
            
            deepspeed_config = dist_config.deepspeed_config_file
            if deepspeed_config:
                logger.info(f"🚀 DeepSpeed 插件已配置：{deepspeed_config}")
                return DeepSpeedPlugin(
                    deepspeed_config_file=deepspeed_config,
                    zero_stage=dist_config.deepspeed_zero_stage
                )
            else:
                logger.warning("⚠️  DeepSpeed 配置文件未指定")
                return None
        except ImportError as e:
            logger.error(f"❌ DeepSpeed 不可用：{e}")
            return None
