"""训练器设备和混合精度配置模块

提供设备检测、混合精度配置和分布式训练插件构建
"""
import logging
from typing import Optional, Dict, Any, List

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
        self._gpu_info_cache: Optional[List[Dict[str, Any]]] = None
    
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
                logger.warning(
                    "💡 CPU 训练优化建议：减小 batch_size、关闭混合精度、"
                    "减少数据加载线程数 (num_workers=0)、禁用 gradient_checkpointing"
                )
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
        
        device_type = self.device_type or self.setup_device()
        use_bf16 = bool(getattr(self.config, "use_bf16", False))
        use_fp16 = bool(getattr(self.config, "use_fp16", False))

        if not use_bf16 and not use_fp16:
            logger.info("配置未启用混合精度（use_fp16/use_bf16 均为 false），使用 FP32")
            return "no"

        # CUDA：按架构与配置选择 BF16/FP16
        if device_type == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability()
                if use_bf16 and major >= 8:
                    logger.info("CUDA Ampere+ 架构，使用 BF16")
                    return "bf16"
                if use_fp16 or not use_bf16:
                    logger.info("CUDA 设备，使用 FP16")
                    return "fp16"
            except Exception as exc:
                logger.warning("CUDA 架构检测失败，降级到 FP16：%s", exc)
            return "fp16"

        # MPS：Florence-2 + LoRA 在 FP16 下易出现 loss=nan，默认 FP32
        if device_type == "mps":
            if use_fp16:
                logger.warning(
                    "MPS 已启用 FP16 混合精度；若出现 loss=nan，请关闭 use_fp16 或设置 mixed_precision: no"
                )
                return "fp16"
            if use_bf16:
                logger.info(
                    "MPS 上忽略 use_bf16，使用 FP32 全精度（Apple Silicon 上更稳定）"
                )
            else:
                logger.info("MPS 使用 FP32 全精度训练")
            return "no"

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
        """构建 FSDP 插件"""
        from .fsdp_plugin import FSDPPlugin

        return FSDPPlugin().build_accelerate_plugin(dist_config)

    def _build_deepspeed_plugin(self, dist_config):
        """构建 DeepSpeed 插件"""
        from .deepspeed_plugin import DeepSpeedPlugin

        return DeepSpeedPlugin().build_accelerate_plugin(
            dist_config, training_config=self.config
        )

    # ------------------------------------------------------------------
    # Multi-GPU selection (v3 enhancement)
    # ------------------------------------------------------------------

    def _select_best_gpu(self) -> int:
        """Select the GPU with the most free memory.

        Returns:
            GPU index (0-based relative to CUDA_VISIBLE_DEVICES).
        """
        import os

        # If CUDA_VISIBLE_DEVICES is set, respect user choice and return 0
        # (the index is relative to the visible-device subset).
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return 0

        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            return 0

        best_gpu = 0
        best_free_mem = -1.0
        for i in range(num_gpus):
            try:
                allocated = torch.cuda.memory_allocated(i)
                props = torch.cuda.get_device_properties(i)
                total_mem = getattr(props, "total_mem", getattr(props, "total_memory", 0))
                free_mem = total_mem - allocated
                if free_mem > best_free_mem:
                    best_free_mem = free_mem
                    best_gpu = i
            except Exception as e:
                logger.debug("GPU %d info query failed: %s", i, e)

        logger.info("Auto-selected GPU %d (free mem: %.1f GB)", best_gpu, best_free_mem / 1e9)
        return best_gpu

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Return a list of GPU info dicts, cached after first call.

        Returns an empty list if CUDA is not available.
        """
        if self._gpu_info_cache is not None:
            return self._gpu_info_cache

        if not torch.cuda.is_available():
            self._gpu_info_cache = []
            return self._gpu_info_cache

        info_list: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total_mem = getattr(props, "total_mem", getattr(props, "total_memory", 0))
                allocated = torch.cuda.memory_allocated(i)
                info_list.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": total_mem / 1e9,
                    "allocated_gb": allocated / 1e9,
                    "free_memory_gb": (total_mem - allocated) / 1e9,
                })
            except Exception as e:
                logger.debug("GPU %d info query failed: %s", i, e)

        self._gpu_info_cache = info_list
        return self._gpu_info_cache
