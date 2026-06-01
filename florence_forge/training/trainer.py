#!/usr/bin/env python3
"""
Florence Forge — 多任务训练器（v1，单文件主流版）

ℹ️ 当前状态（2026-05-21）
=========================
本文件是默认导出的训练器入口：

    >>> from florence_forge.training import MultiTaskTrainer

文件较大（~1579 行）含 FSDP/DeepSpeed Plugin、激活值重计算 4 档策略、
异步 checkpoint、CallbackManager、GradientValidator、MemoryMonitor、
TaskScheduler、LoRAManager、ModelMerger 等高级特性。

并存说明：仓库内同时存在 v2 训练栈（`trainer_refactored.py` + `training_loop.py`
+ `checkpoint_manager.py`），v2 模块化更清晰但功能尚未覆盖 v1 全部特性。
迁移路线见 `trainer_refactored.py` 顶部说明。

提供完整的多任务训练功能，包括训练循环、评估、检查点管理等
"""

import os
import csv
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm.auto import tqdm

from ._accelerator_compat import Accelerator

from ..core.config import TrainingConfig
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader
from .scheduler import TaskScheduler
from .lora_manager import LoRAManager
from .model_merger import ModelMerger
from .visualizer import TrainingVisualizer
from .monitoring import TrainingMonitor
from .gradient_validator import GradientValidator, GradientValidationConfig
from .memory_monitor import MemoryMonitor, MemoryMonitorConfig
from .trainer_io import TrainerIOMixin
from ..core.callbacks import CallbackManager, create_default_callbacks
from ..utils.training_logging import format_epoch_summary

logger = logging.getLogger(__name__)

class MultiTaskTrainer(TrainerIOMixin):
    """多任务训练器
    
    提供完整的多任务训练功能
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiTaskDataset,
        val_dataset: Optional[MultiTaskDataset] = None,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None
    ):
        """初始化训练器

        Args:
            model: 多任务模型（Florence2MultiTaskModel 或任何兼容 nn.Module 的模型）
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 训练配置
            accelerator: Accelerate加速器
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # 初始化加速器
        if accelerator is None:
            # 自动检测设备并配置加速器
            
            # 设备检测和配置
            self._setup_device()
            
            # 确定混合精度设置（智能检测和配置）
            mixed_precision = self._determine_mixed_precision()
            
            # 构建 Accelerator 参数
            accel_kwargs = {
                "mixed_precision": mixed_precision,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "log_with": "tensorboard" if self.config.logging_dir else None,
                "project_dir": self.config.logging_dir
            }
            
            # 分布式插件配置（FSDP / DeepSpeed）
            dist_config = self.config.distributed_settings
            if dist_config.enabled or dist_config.strategy != "none":
                plugin = self._build_distributed_plugin(dist_config)
                if plugin is not None:
                    if dist_config.strategy == "fsdp":
                        accel_kwargs["fsdp_plugin"] = plugin
                        logger.info(f"🚀 FSDP 插件已配置: {dist_config.fsdp_sharding_strategy}")
                    elif dist_config.strategy == "deepspeed":
                        accel_kwargs["deepspeed_plugin"] = plugin
                        logger.info(f"🚀 DeepSpeed ZeRO-{dist_config.deepspeed_stage} 插件已配置")
            
            self.accelerator = Accelerator(**accel_kwargs)
        else:
            self.accelerator = accelerator
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf') if not self.config.greater_is_better else float('-inf')
        self.patience_counter = 0
        
        # 组件初始化
        self.task_scheduler = None
        self.lora_manager = None
        self.model_merger = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 指标记录
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.step_metrics = []
        self.step_metrics_history_limit = int(
            getattr(self.config, "step_metrics_history_limit", 1000) or 0
        )
        
        # 异步检查点保存线程池
        self._checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint_saver")
        self._last_checkpoint_future = None
        self._report_thread = None
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化可视化器
        self.visualizer = TrainingVisualizer(str(self.output_dir))
        
        # 初始化监控器（仅用于 MonitoringCallback，不再直接在训练循环中调用）
        self.monitor = None
        if hasattr(self.config, 'monitoring_config') and self.config.monitoring_config:
            self.monitor = TrainingMonitor(
                config=self.config.monitoring_config,
                output_dir=str(self.output_dir)
            )
        
        # 初始化梯度验证器
        gradient_config = GradientValidationConfig(
            max_grad_norm_threshold=self.config.optimization_settings.max_grad_norm * 2,
            log_frequency=max(1, self.config.logging_steps // 2),
            save_stats=True,
            stats_save_frequency=self.config.save_steps
        )
        self.gradient_validator = GradientValidator(
            model=self.model,
            config=gradient_config,
            output_dir=str(self.output_dir)
        ) if self.config.optimization_settings.max_grad_norm > 0 else None
        
        # 初始化内存监控器
        memory_config = MemoryMonitorConfig(
            enable_monitoring=True,
            log_frequency=50,  # 每50步记录一次内存使用
            warning_threshold_percent=80.0,
            critical_threshold_percent=90.0,
            enable_gpu_monitoring=True,
            auto_cleanup=True,
            save_stats=True
        )
        self.memory_monitor = MemoryMonitor(memory_config)

        # 初始化 Callback 管理器（传入 monitor，统一由 MonitoringCallback 管理）
        self.callback_manager = CallbackManager(
            create_default_callbacks(self.config, monitor=self.monitor)
        )

        logger.info("多任务训练器初始化完成")
    
    def _setup_device(self) -> None:
        """设置和检测设备"""
        
        if self.config.device == "auto":
            # 自动检测最佳设备
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"检测到 {device_count} 个CUDA设备")
                
                # 选择显存最大的GPU
                best_device = 0
                max_memory = 0
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory = props.total_memory / (1024**3)  # GB
                    logger.info(f"GPU {i}: {props.name}, 显存: {memory:.1f}GB")
                    if memory > max_memory:
                        max_memory = memory
                        best_device = i
                
                self.config.device = f"cuda:{best_device}"
                logger.info(f"自动选择设备: {self.config.device} (显存: {max_memory:.1f}GB)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.config.device = "mps"
                logger.info("自动选择设备: mps (Apple Silicon GPU)")
            else:
                self.config.device = "cpu"
                logger.info("自动选择设备: cpu")
        else:
            # 验证指定的设备是否可用
            if self.config.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA不可用，回退到CPU")
                    self.config.device = "cpu"
                else:
                    device_id = int(self.config.device.split(":")[1]) if ":" in self.config.device else 0
                    if device_id >= torch.cuda.device_count():
                        logger.warning(f"GPU {device_id} 不存在，使用GPU 0")
                        self.config.device = "cuda:0"
            elif self.config.device == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    logger.warning(f"MPS不可用，回退到CPU")
                    self.config.device = "cpu"
            
            logger.info(f"使用指定设备: {self.config.device}")
        
        # 设置PyTorch默认设备
        if self.config.device.startswith("cuda"):
            device_id = int(self.config.device.split(":")[1]) if ":" in self.config.device else 0
            torch.cuda.set_device(device_id)
            # 清理GPU缓存
            torch.cuda.empty_cache()
    
    def _determine_mixed_precision(self) -> str:
        """智能确定混合精度设置"""
        
        # 如果是CPU，不使用混合精度
        if self.config.device == "cpu":
            logger.info("CPU设备，禁用混合精度")
            return "no"
        
        # 如果不是CUDA设备，不使用混合精度
        if not self.config.device.startswith("cuda") and self.config.device != "mps":
            logger.info(f"设备 {self.config.device} 不支持混合精度")
            return "no"
        
        # CUDA设备的混合精度检测
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            # 检查PyTorch版本
            pt_version = torch.__version__.split("+")[0]  # 移除+cu118等后缀
            pt_major, pt_minor = map(int, pt_version.split(".")[:2])
            pt_version_ok = (pt_major > 1) or (pt_major == 1 and pt_minor >= 10)
            
            # 检查硬件是否支持BF16
            hw_support_bf16 = False
            try:
                if hasattr(torch.cuda, 'is_bf16_supported'):
                    hw_support_bf16 = torch.cuda.is_bf16_supported()
                else:
                    # 对于旧版本PyTorch，检查GPU架构
                    device_id = int(self.config.device.split(":")[1]) if ":" in self.config.device else 0
                    props = torch.cuda.get_device_properties(device_id)
                    # Ampere架构(8.0+)及以上支持BF16
                    hw_support_bf16 = props.major >= 8
            except Exception as e:
                logger.warning(f"检查BF16支持时出错: {e}")
                hw_support_bf16 = False
            
            # 决定混合精度类型
            if hasattr(self.config, 'use_bf16') and self.config.use_bf16:
                if pt_version_ok and hw_support_bf16:
                    logger.info("✅ 使用BF16混合精度加速训练")
                    return "bf16"
                else:
                    logger.warning(f"⚠️ BF16不可用，原因: PyTorch版本({pt_version}>=1.10): {pt_version_ok}, 硬件支持: {hw_support_bf16}")
                    # 自动降级到FP16
                    if hasattr(self.config, 'use_fp16') and self.config.use_fp16:
                        logger.info("✅ 回退到FP16混合精度")
                        return "fp16"
                    else:
                        logger.info("⚠️ 禁用混合精度")
                        return "no"
            elif hasattr(self.config, 'use_fp16') and self.config.use_fp16:
                logger.info("✅ 使用FP16混合精度加速训练")
                return "fp16"
            else:
                # 自动选择最佳混合精度
                if pt_version_ok and hw_support_bf16:
                    logger.info("✅ 自动选择BF16混合精度（推荐）")
                    return "bf16"
                else:
                    logger.info("✅ 自动选择FP16混合精度")
                    return "fp16"
        
        # MPS设备（Apple Silicon）
        elif self.config.device == "mps":
            # MPS支持FP16但不支持BF16
            if hasattr(self.config, 'use_fp16') and self.config.use_fp16:
                logger.info("✅ MPS设备使用FP16混合精度")
                return "fp16"
            else:
                logger.info("MPS设备，建议启用FP16混合精度以提升性能")
                return "no"
        
        # 默认不使用混合精度
        logger.info("禁用混合精度")
        return "no"
    
    def _build_distributed_plugin(self, dist_config):
        """构建分布式训练插件（FSDP 或 DeepSpeed）

        Args:
            dist_config: DistributedConfig 实例

        Returns:
            FSDPPlugin 或 DeepSpeedPlugin 实例，或 None
        """
        if dist_config.strategy == "fsdp":
            return self._build_fsdp_plugin(dist_config)
        elif dist_config.strategy == "deepspeed":
            return self._build_deepspeed_plugin(dist_config)
        return None

    def _build_fsdp_plugin(self, dist_config):
        """构建 FSDP 插件"""
        try:
            from accelerate.utils import FullyShardedDataParallelPlugin
            
            # 映射配置到插件参数
            plugin_kwargs = {
                "sharding_strategy": dist_config.fsdp_sharding_strategy,
                "backward_prefetch": dist_config.fsdp_backward_prefetch,
                "cpu_offload": dist_config.fsdp_cpu_offload,
                "auto_wrap_policy": dist_config.fsdp_auto_wrap_policy.lower().replace("_wrap", ""),
                "min_num_params": int(dist_config.fsdp_min_num_params),
                "activation_checkpointing": dist_config.fsdp_activation_checkpointing,
            }
            
            plugin = FullyShardedDataParallelPlugin(**plugin_kwargs)
            return plugin
        except Exception as e:
            logger.warning(f"⚠️ FSDP 插件构建失败: {e}，将回退到 DDP")
            return None

    def _build_deepspeed_plugin(self, dist_config):
        """构建 DeepSpeed 插件"""
        try:
            from accelerate.utils import DeepSpeedPlugin
            
            # 如果提供了配置文件，优先使用
            if dist_config.deepspeed_config_file:
                plugin = DeepSpeedPlugin(hf_ds_config=dist_config.deepspeed_config_file)
                return plugin
            
            # 否则从配置参数构建
            plugin_kwargs = {
                "zero_stage": dist_config.deepspeed_stage,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "gradient_clipping": self.config.optimization_settings.max_grad_norm
                if self.config.optimization_settings.max_grad_norm > 0
                else None,
            }
            
            # Offload 配置
            if dist_config.deepspeed_offload_optimizer:
                plugin_kwargs["offload_optimizer_device"] = "cpu"
            if dist_config.deepspeed_offload_param:
                plugin_kwargs["offload_param_device"] = "cpu"
            
            plugin = DeepSpeedPlugin(**plugin_kwargs)
            return plugin
        except Exception as e:
            logger.warning(f"⚠️ DeepSpeed 插件构建失败: {e}，将回退到 DDP")
            return None
    
    def setup_training(self) -> None:
        """设置训练组件"""
        logger.info("正在设置训练组件...")
        
        # 设置任务调度器
        task_types = list(self.train_dataset.task_indices.keys())
        self.task_scheduler = TaskScheduler(
            task_types=task_types,
            config=self.config.task_scheduling_settings
        )
        
        # 设置LoRA管理器和模型合并器
        if self.config.model_settings.use_lora:
            self.lora_manager = LoRAManager(self.config.model_settings.lora_config)
            self.model_merger = ModelMerger(self.lora_manager)
            
            # 为第一个任务应用LoRA到模型
            if task_types:
                first_task = task_types[0]
                logger.info(f"将LoRA应用到模型，首个任务: {first_task}")
                self.model = self.lora_manager.apply_lora_to_model(
                    self.model, first_task
                )
                
                # 为其他任务添加适配器
                for task_type in task_types[1:]:
                    logger.info(f"为任务 {task_type} 添加LoRA适配器")
                    self.lora_manager.add_adapter_to_model(
                        self.model, task_type
                    )
            
            # 打印可训练参数信息
            if hasattr(self.lora_manager, 'print_trainable_parameters'):
                self.lora_manager.print_trainable_parameters(self.model)
        
        # 启用 Gradient Checkpointing（降低显存，增加计算时间约 20-30%）
        if self.config.model_settings.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # 设置数据加载器
        self._setup_dataloaders()
        
        # 设置优化器和调度器
        self._setup_optimizer()
        self._setup_lr_scheduler()
        
        # 使用accelerator准备组件
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        
        # 使用accelerate准备数据加载器（支持分布式训练）
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        # 初始化CSV日志记录器
        self._init_csv_logger()
        
        logger.info("训练组件设置完成")
    
    def _enable_gradient_checkpointing(self) -> None:
        """启用 Gradient Checkpointing / 激活值重计算以降低显存占用

        支持三种策略模式：
        - full: 全局启用（所有 Transformer 层）
        - selective: 仅对指定层或每隔 N 层启用（平衡显存与速度）
        - auto: 自动检测最佳策略（基于模型大小和可用显存）
        - none: 禁用（仅清理已有设置）

        启用后模型需设置 model.train() 以确保梯度正确传播。
        """
        model_config = self.config.model_settings
        strategy = model_config.activation_checkpointing_strategy

        # 向后兼容：旧配置 gradient_checkpointing=True 时启用 full 模式
        if strategy == "none" and model_config.gradient_checkpointing:
            strategy = "full"
            logger.info("gradient_checkpointing=True 已映射到 activation_checkpointing_strategy='full'")

        if strategy == "none":
            logger.info("激活值重计算已禁用 (strategy='none')")
            return

        try:
            if strategy == "auto":
                strategy = self._auto_select_checkpoint_strategy()
                logger.info(f"🔄 自动选择重计算策略: {strategy}")

            if strategy == "full":
                self._apply_full_gradient_checkpointing()
            elif strategy == "selective":
                self._apply_selective_gradient_checkpointing()
            else:
                logger.warning(f"未知的重计算策略: {strategy}，跳过")
                return

            # 重要：启用 gradient checkpointing 后需要确保 use_cache=False
            self._disable_kv_cache_for_training()

        except Exception as e:
            logger.warning(f"⚠️ Gradient Checkpointing 启用失败: {e}")
            logger.warning("   将继续训练，但显存占用可能较高")

    def _auto_select_checkpoint_strategy(self) -> str:
        """自动选择最佳激活值重计算策略

        基于模型大小和可用显存做出决策：
        - 大模型 (>7B) 或显存紧张: 使用 selective
        - 中等模型 (1B-7B): 使用 full
        - 小模型 (<1B): 不使用 checkpointing

        Returns:
            建议的策略名称 ("full", "selective", "none")
        """
        # 估算模型参数量
        total_params = 0
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass

        params_in_billions = total_params / 1e9

        # 检测可用显存
        available_vram_gb = float('inf')
        if torch.cuda.is_available():
            try:
                device_props = torch.cuda.get_device_properties(self.config.device if hasattr(self.config, 'device') else 0)
                total_vram = device_props.total_memory / (1024**3)
                allocated_vram = torch.cuda.memory_allocated() / (1024**3)
                available_vram_gb = total_vram - allocated_vram
            except Exception:
                pass

        logger.info(f"自动策略检测: 模型参数量={params_in_billions:.2f}B, 可用显存={available_vram_gb:.1f}GB")

        # 决策逻辑
        if params_in_billions < 1.0:
            logger.info("模型较小 (<1B)，无需激活值重计算")
            return "none"
        elif params_in_billions >= 7.0 or available_vram_gb < 10.0:
            logger.info("大模型或显存紧张，使用 selective 策略")
            # 自动设置每隔 2 层 checkpoint
            if self.config.model_settings.checkpoint_every_n_layers is None:
                self.config.model_settings.checkpoint_every_n_layers = 2
            return "selective"
        else:
            logger.info("中等模型且显存充足，使用 full 策略")
            return "full"

    def _apply_full_gradient_checkpointing(self) -> None:
        """全局启用 Gradient Checkpointing（所有层）"""
        # 方式1：HuggingFace 模型原生支持
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient Checkpointing 已启用（HF 原生接口，full 模式）")
        # 方式2：Florence2MultiTaskModel 代理到内部 backend
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient Checkpointing 已启用（通过 backend 代理，full 模式）")
        # 方式3：PyTorch 原生 gradient_checkpointing
        else:
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
            logger.info("✅ Gradient Checkpointing 已启用（PyTorch 原生模式，full 模式）")

    def _apply_selective_gradient_checkpointing(self) -> None:
        """选择性启用 Gradient Checkpointing（指定层或每隔 N 层）"""
        target_layers = self.config.model_settings.checkpoint_target_layers
        every_n = self.config.model_settings.checkpoint_every_n_layers

        # 遍历模型寻找可 checkpoint 的层
        checkpointed_count = 0
        matched_modules = []

        for name, module in self.model.named_modules():
            # 判断是否应该对此模块启用 checkpoint
            should_checkpoint = False

            if target_layers is not None:
                # 模式匹配
                if isinstance(target_layers, list):
                    should_checkpoint = any(
                        name == pattern or name.endswith(pattern)
                        for pattern in target_layers
                    )
                elif isinstance(target_layers, str):
                    # 支持通配符 *（简化实现：检查前缀/包含）
                    pattern = target_layers.replace("*", "")
                    should_checkpoint = name.startswith(pattern) or pattern in name

            elif every_n is not None:
                # 从名称中提取层索引（如 encoder.layers.0 -> 0）
                layer_idx = self._extract_layer_index(name)
                if layer_idx is not None and layer_idx % every_n == 0:
                    should_checkpoint = True

            if should_checkpoint:
                # 对匹配到的模块启用 gradient checkpointing
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    checkpointed_count += 1
                    matched_modules.append(name)
                elif isinstance(module, torch.nn.TransformerEncoderLayer) or \
                     isinstance(module, torch.nn.TransformerDecoderLayer):
                    # PyTorch 原生 Transformer 层
                    module.checkpoint = True
                    checkpointed_count += 1
                    matched_modules.append(name)

        if checkpointed_count > 0:
            logger.info(f"✅ 选择性 Gradient Checkpointing 已启用: {checkpointed_count} 个模块")
            if len(matched_modules) <= 10:
                for mod_name in matched_modules:
                    logger.info(f"   - {mod_name}")
            else:
                logger.info(f"   前 10 个: {', '.join(matched_modules[:10])}...")
        else:
            logger.warning("未匹配到任何可 checkpoint 的模块，回退到 full 模式")
            self._apply_full_gradient_checkpointing()

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        """从模块名称中提取层索引

        例如: "encoder.layers.3.self_attn" -> 3

        Returns:
            层索引或 None
        """
        import re
        # 匹配常见的层索引模式
        patterns = [
            r'layers\.(\d+)',
            r'layer\.(\d+)',
            r'blocks\.(\d+)',
            r'block\.(\d+)',
            r'h\.(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, module_name)
            if match:
                return int(match.group(1))
        return None

    def _disable_kv_cache_for_training(self) -> None:
        """训练时禁用 KV Cache（与 gradient checkpointing 兼容）"""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
            logger.info("   已将 model.config.use_cache 设为 False（与 checkpointing 兼容）")
        # 递归禁用子模块的 use_cache
        for name, module in self.model.named_modules():
            if hasattr(module, 'config') and hasattr(module.config, 'use_cache'):
                module.config.use_cache = False

    def _setup_dataloaders(self) -> None:
        """设置数据加载器"""
        # 训练数据加载器
        train_loader = TaskDataLoader(
            dataset=self.train_dataset,
            config=self.config.data_settings,
            sampling_strategy=self.config.task_scheduling_settings.strategy
        )
        self.train_dataloader = train_loader.get_dataloader()
        
        # 验证数据加载器
        if self.val_dataset is not None:
            # 创建验证配置的副本，避免修改原始配置
            val_config = deepcopy(self.config.data_settings)
            val_config.shuffle = False  # 验证时不打乱
            val_config.drop_last = False  # 验证时保留所有数据
            
            val_loader = TaskDataLoader(
                dataset=self.val_dataset,
                config=val_config,
                sampling_strategy="random"
            )
            self.val_dataloader = val_loader.get_dataloader()
            logger.info(f"验证数据加载器已创建，批次数: {len(self.val_dataloader)}")
    
    def _setup_optimizer(self) -> None:
        """设置优化器"""
        opt_config = self.config.optimization_settings
        
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=(opt_config.adam_beta1, opt_config.adam_beta2),
            eps=opt_config.adam_epsilon
        )
        
        logger.info(f"优化器设置完成，学习率: {opt_config.learning_rate}")
    
    def _setup_lr_scheduler(self) -> None:
        """设置学习率调度器"""
        opt_config = self.config.optimization_settings
        
        # 计算总训练步数
        if self.config.max_steps:
            num_training_steps = self.config.max_steps
        else:
            num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        
        # 计算预热步数
        if opt_config.warmup_steps:
            num_warmup_steps = opt_config.warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * opt_config.warmup_ratio)
        
        # 创建调度器
        if opt_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif opt_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.lr_scheduler = get_scheduler(
                opt_config.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        logger.info(
            f"学习率调度器设置完成，类型: {opt_config.lr_scheduler_type}, "
            f"预热步数: {num_warmup_steps}, 总步数: {num_training_steps}"
        )
    
    def _init_csv_logger(self) -> None:
        """初始化CSV日志记录器"""
        self.step_csv_path = self.output_dir / "step_metrics.csv"
        self.epoch_csv_path = self.output_dir / "epoch_metrics.csv"
        
        # 初始化CSV缓冲区
        self.step_csv_buffer = []
        self.csv_buffer_size = 50  # 缓冲区大小
        
        # 初始化步骤指标CSV
        with open(self.step_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'task_type', 'loss', 'learning_rate', 
                'grad_norm', 'time_per_step', 'data_time', 'forward_time',
                'backward_time', 'optim_time'
            ])
        
        # 初始化轮次指标CSV
        with open(self.epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'learning_rate',
                'best_metric', 'patience_counter'
            ])
    
    def train(self) -> Dict[str, Any]:
        """开始训练
        
        Returns:
            训练结果字典
        """
        logger.debug("准备启动多任务训练流程")
        self.callback_manager.on_train_begin(self, self.config)
        
        # 设置训练组件
        self.setup_training()
        
        # 保存配置
        self._save_config()
        
        # 注：模型架构记录已迁移到 MonitoringCallback.on_train_begin()
        
        # 训练循环
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                self.callback_manager.on_epoch_begin(self, epoch)

                # 分布式采样器：每个 epoch 设置不同的随机种子
                if self.train_dataloader is not None:
                    # 优先访问 sampler.set_epoch（Accelerate DataLoader 包装器支持）
                    sampler = getattr(self.train_dataloader, 'sampler', None)
                    if sampler is not None and hasattr(sampler, 'set_epoch'):
                        sampler.set_epoch(epoch)
                    # 兼容 TaskDataLoader 的 set_epoch
                    elif hasattr(self.train_dataloader, 'set_epoch'):
                        self.train_dataloader.set_epoch(epoch)

                # 训练一个epoch
                train_metrics = self._train_epoch()

                # 验证（epoch 级别）
                val_metrics = None
                if self.val_dataset is not None:
                    self.callback_manager.on_eval_begin(self, None)
                    # 验证条件：按 eval_steps 间隔，或最后一个 epoch 强制验证
                    is_last_epoch = (epoch + 1) == self.config.num_epochs
                    should_eval = (
                        self.config.eval_steps <= 1 or
                        (epoch + 1) % self.config.eval_steps == 0 or
                        is_last_epoch
                    )
                    if should_eval:
                        val_metrics = self._validate_epoch()
                        if self.accelerator.is_local_main_process:
                            logger.info(
                                "[eval] epoch=%s/%s | samples=%s | complete",
                                epoch + 1,
                                self.config.num_epochs,
                                len(self.val_dataloader.dataset),
                            )
                    self.callback_manager.on_eval_end(self, {"val_metrics": val_metrics})

                # 记录epoch指标
                self._record_epoch_metrics(train_metrics, val_metrics)

                # 保存检查点
                if (epoch + 1) % self.config.save_steps == 0:
                    self._save_checkpoint(epoch)
                    self.callback_manager.on_save(self, self.config.output_dir, None)

                # 更新任务权重（在 epoch 结束回调前执行）
                if self.task_scheduler.should_update_weights():
                    self.task_scheduler.auto_adjust_weights()

                # epoch 结束回调（每个 epoch 都触发，不限于权重更新时）
                self.callback_manager.on_epoch_end(self, epoch, {"train_metrics": train_metrics, "val_metrics": val_metrics})

                # 早停检查（仅检查回调设置的 _stop_training 标志；
                # 内置早停逻辑已迁移到 EarlyStoppingCallback，此处不再重复判断）
                if getattr(self, '_stop_training', False):
                    logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
        
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            # 保存检查点以便恢复
            self._save_checkpoint(self.current_epoch)
            raise
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA 显存不足: {e}")
            logger.error("请减小 batch_size 或启用梯度累积")
            raise
        
        except Exception as e:
            # 使用 exception() 记录完整 traceback，便于调试
            logger.exception(f"训练过程中发生未预期错误: {e}")
            # 尝试保存检查点（使用 self.global_step 而非局部变量）
            try:
                self._save_checkpoint(self.current_epoch)
            except Exception:
                logger.warning("保存检查点失败，可能无法从当前状态恢复")
            raise
        
        finally:
            # 保存最终模型
            self._save_final_model()
            
            self._generate_training_report_on_end()

            # 注：监控器的 finish() 已迁移到 MonitoringCallback.on_train_end()，
            # 通过 callback_manager.on_train_end() 统一调用

            # 计算训练时间
            total_time = time.time() - start_time
            logger.debug("训练清理完成，总耗时: %.2f秒", total_time)
            self.callback_manager.on_train_end(self, self.config)

        return self._get_training_summary()

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将批次数据移动到模型所在设备

        当使用 accelerator 时，accelerator.prepare() 已经处理了设备转移，
        此方法会跳过以避免冗余的 GPU→CPU→GPU 传输。

        Args:
            batch: 原始批次数据字典

        Returns:
            移动到设备后的批次数据字典
        """
        # accelerator 已自动处理设备转移，跳过冗余操作
        if self.accelerator is not None:
            return batch

        device = next(self.model.parameters()).device
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device, non_blocking=True)
            else:
                result[key] = value
        return result
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        # 细分计时统计
        total_data_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_optim_time = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            disable=not self.accelerator.is_local_main_process
        )
        
        data_start_time = time.time()
        for step, batch in enumerate(progress_bar):
            # 数据加载时间
            data_time = time.time() - data_start_time
            total_data_time += data_time
            
            step_start_time = time.time()
            logs: Dict[str, float] = {}

            # 回调：step 开始
            self.callback_manager.on_step_begin(self, self.global_step, logs)
            if batch is None or batch.get("is_empty", False):
                data_start_time = time.time()
                continue

            # 从批次数据中获取实际任务类型（支持单任务字符串或多任务列表）
            task_type = None
            if isinstance(batch, dict) and 'task_type' in batch:
                task_type = batch['task_type']
            elif isinstance(batch, dict) and 'task_types' in batch:
                task_type = batch['task_types'][0]  # 多任务批次取第一个
            else:
                task_type = self.task_scheduler.select_task(self.current_epoch)
                logger.warning(f"无法从数据中获取任务类型，使用调度器选择: {task_type}")

            # 切换LoRA适配器（如果使用）
            if self.lora_manager and task_type in self.lora_manager.active_adapters:
                self.lora_manager.switch_adapter(self.model, task_type)

            # 将批次数据移动到模型所在设备
            batch_on_device = self._move_batch_to_device(batch)
            input_ids = batch_on_device["input_ids"]
            pixel_values = batch_on_device["pixel_values"]
            attention_mask = batch_on_device.get("attention_mask")
            labels = batch_on_device.get("labels")

            # 内存监控 - 每 N 步记录一次（减少开销）
            if self.global_step % self.memory_monitor.config.log_frequency == 0:
                self.memory_monitor.log_memory_usage(self.global_step, "before_forward")

            # 在 accumulate 上下文【外】检查 labels，避免 continue 破坏梯度累积状态
            if labels is None:
                logger.warning(
                    f"步骤 {self.global_step}: 未找到labels，跳过该步骤（将在 accumulate 上下文外执行 zero_grad）"
                )
                # 在 accumulate 上下文外显式清零梯度，防止梯度累积错误
                self.optimizer.zero_grad()
                data_start_time = time.time()
                continue

            # 前向传播 + 反向传播 + 优化器步骤（必须在同一个 accumulate 上下文中）
            forward_start_time = time.time()
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                forward_time = time.time() - forward_start_time
                total_forward_time += forward_time

                # 内存监控 - 前向传播后（仅 log_frequency 步）
                if self.global_step % self.memory_monitor.config.log_frequency == 0:
                    self.memory_monitor.log_memory_usage(self.global_step, "after_forward")

                # 反向传播
                backward_start_time = time.time()
                self.accelerator.backward(loss)

                # 梯度验证（在梯度裁剪之前）
                gradient_valid = True
                validation_info = None
                if self.gradient_validator:
                    try:
                        gradient_valid, validation_info = self.gradient_validator.validate_gradients(self.global_step)

                        # 如果梯度无效，记录警告并在后续跳过优化步骤
                        # 注意：不能在 accumulate 上下文中使用 continue，会破坏梯度累积状态
                        if not gradient_valid:
                            logger.warning(f"步骤 {self.global_step}: 梯度验证失败，跳过优化步骤")

                    except Exception as e:
                        logger.error(f"梯度验证过程中出错: {e}")
                        # 验证失败时继续训练，但记录错误

                # 梯度裁剪（利用返回值获取梯度范数，避免重复计算）
                grad_norm = 0.0
                if gradient_valid and self.config.optimization_settings.max_grad_norm > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimization_settings.max_grad_norm
                    )
                    # accelerator.clip_grad_norm_ 可能返回张量，需转为 float
                    if hasattr(grad_norm, 'item'):
                        grad_norm = grad_norm.item()
                backward_time = time.time() - backward_start_time
                total_backward_time += backward_time

                # 优化器步骤（仅梯度验证通过时执行）
                optim_start_time = time.time()
                if gradient_valid:
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()
                optim_time = time.time() - optim_start_time
            
            # 内存监控 - 优化器步骤后（仅 log_frequency 步）
            if self.global_step % self.memory_monitor.config.log_frequency == 0:
                self.memory_monitor.log_memory_usage(self.global_step, "after_optimizer")
            total_optim_time += optim_time
            
            # 记录指标
            step_time = time.time() - step_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新任务性能
            self.task_scheduler.update_task_performance(task_type, loss.item())
            
            # 记录步骤指标 - 修改逻辑确保小数据集也能记录足够的指标
            should_log = (
                self.global_step % self.config.logging_steps == 0 or  # 按原有间隔记录
                self.global_step == 1 or  # 记录第一步
                (self.global_step % max(1, len(self.train_dataloader) // 3) == 0)  # 每个epoch至少记录3次
            )
            if should_log:
                self._record_step_metrics(
                    step=self.global_step,
                    epoch=self.current_epoch,
                    task_type=task_type,
                    loss=loss.item(),
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    time_per_step=step_time,
                    data_time=data_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    optim_time=optim_time
                )
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'task': task_type,
                'step_s': f'{step_time:.2f}',
                'grad': f'{grad_norm:.3f}'
            })
            
            # 收集epoch指标
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['learning_rate'].append(current_lr)

            # 触发 on_step_end 回调
            step_logs: Dict[str, Any] = {
                "loss": loss.item(),
                "learning_rate": current_lr,
                "grad_norm": grad_norm,
                "time_per_step": step_time,
                "data_time": data_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "optim_time": optim_time,
                "task_type": task_type,
            }
            self.callback_manager.on_step_end(self, self.global_step, step_logs)

            self.global_step += 1
            
            # 检查是否达到最大步数
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
            
            # 准备下一次数据加载计时
            data_start_time = time.time()
        
        # 计算epoch平均指标
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        return avg_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        val_metrics = defaultdict(list)
        
        # 使用 torch.inference_mode() 替代 no_grad()，获得更好的推理性能
        # inference_mode() 比 no_grad() 更激进，会禁用 view 跟踪，减少开销
        # 同时使用 accelerator.autocast() 匹配训练时的混合精度设置
        with torch.inference_mode(), self.accelerator.autocast():
            for batch in tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_local_main_process
            ):
                # 将批次数据移动到模型所在设备
                batch_on_device = self._move_batch_to_device(batch)
                input_ids = batch_on_device["input_ids"]
                pixel_values = batch_on_device["pixel_values"]
                attention_mask = batch_on_device.get("attention_mask")
                labels = batch_on_device.get("labels")

                # 验证时需要正确的labels，不能使用input_ids
                if labels is not None:
                    outputs = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                        val_metrics['loss'].append(loss.item())
                    else:
                        # 模型未返回 loss，跳过该批次
                        logger.warning(f"验证批次未返回 loss，跳过")
                        continue
                else:
                    # 如果没有labels，无法进行有监督的验证
                    logger.warning("验证批次缺少 labels，跳过该批次")
                    continue
        
        # 计算平均指标
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        
        return avg_metrics
    
    def _record_step_metrics(
        self,
        step: int,
        epoch: int,
        task_type: str,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        time_per_step: float,
        data_time: float = 0.0,
        forward_time: float = 0.0,
        backward_time: float = 0.0,
        optim_time: float = 0.0
    ) -> None:
        """记录步骤指标"""
        # 添加到CSV缓冲区（包含细分计时）
        self.step_csv_buffer.append([
            step, epoch, task_type, loss, learning_rate,
            grad_norm, time_per_step, data_time, forward_time,
            backward_time, optim_time
        ])
        
        # 当缓冲区满时或到达epoch结束时批量写入
        if len(self.step_csv_buffer) >= self.csv_buffer_size:
            self._flush_csv_buffer()
        
        # 记录到内存
        self.step_metrics.append({
            'step': step,
            'epoch': epoch,
            'task_type': task_type,
            'loss': loss,
            'learning_rate': learning_rate,
            'grad_norm': grad_norm,
            'time_per_step': time_per_step,
            'data_time': data_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optim_time': optim_time
        })
        if self.step_metrics_history_limit > 0 and len(self.step_metrics) > self.step_metrics_history_limit:
            del self.step_metrics[:-self.step_metrics_history_limit]
        elif self.step_metrics_history_limit == 0:
            self.step_metrics.clear()
        
        # 记录到accelerator（如果配置了）
        if self.accelerator.is_local_main_process:
            self.accelerator.log({
                'train/loss': loss,
                'train/learning_rate': learning_rate,
                'train/grad_norm': grad_norm,
                'train/time_per_step': time_per_step,
                'train/data_time': data_time,
                'train/forward_time': forward_time,
                'train/backward_time': backward_time,
                'train/optim_time': optim_time
            }, step=step)
        
        # 注：监控器（WandB, SwanLab, TensorBoard）的指标记录已迁移到 MonitoringCallback，
        # 通过 callback_manager.on_step_end() 统一调用，此处不再直接调用 self.monitor
    
    def _flush_csv_buffer(self) -> None:
        """刷新CSV缓冲区到文件"""
        if not self.step_csv_buffer:
            return
        
        try:
            with open(self.step_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.step_csv_buffer)
            self.step_csv_buffer.clear()
        except Exception as e:
            logger.warning(f"CSV缓冲区刷新失败: {e}")
    
    def _record_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ) -> None:
        """记录epoch指标"""
        # 在epoch结束时刷新步骤指标缓冲区
        self._flush_csv_buffer()
        
        train_loss = train_metrics.get('loss', 0.0)
        val_loss = val_metrics.get('loss') if val_metrics else None
        current_lr = train_metrics.get('learning_rate', 0.0)
        
        # 记录到CSV
        with open(self.epoch_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch, train_loss, val_loss if val_loss is not None else '', current_lr,
                self.best_metric, self.patience_counter
            ])
        
        # 记录到内存（限制历史长度，防止长时间训练内存膨胀）
        max_history = 100
        self.train_metrics['loss'].append(train_loss)
        if len(self.train_metrics['loss']) > max_history:
            self.train_metrics['loss'] = self.train_metrics['loss'][-max_history:]
        if val_metrics and val_loss is not None:
            self.val_metrics['loss'].append(val_loss)
            if len(self.val_metrics['loss']) > max_history:
                self.val_metrics['loss'] = self.val_metrics['loss'][-max_history:]
        
        # 记录到accelerator
        if self.accelerator.is_local_main_process:
            log_dict = {
                'epoch/train_loss': train_loss,
                'epoch/learning_rate': current_lr
            }
            if val_metrics and val_loss is not None:
                log_dict['epoch/val_loss'] = val_loss
            
            self.accelerator.log(log_dict, step=self.current_epoch)
        
        # 注：监控器（WandB, SwanLab, TensorBoard）的 epoch 指标记录
        # 已迁移到 MonitoringCallback，通过 callback_manager.on_epoch_end() 统一调用
        
        if self.accelerator.is_local_main_process:
            logger.info(
                format_epoch_summary(
                    epoch=self.current_epoch + 1,
                    total_epochs=self.config.num_epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
    
    def _should_early_stop(self, val_metrics: Optional[Dict[str, float]]) -> bool:
        """检查是否应该早停

        注意：训练循环中的早停已由 EarlyStoppingCallback 统一管理，
        通过设置 trainer._stop_training = True 触发。
        此方法保留作为兼容性接口，不建议直接调用。

        Args:
            val_metrics: 验证指标

        Returns:
            是否应该早停
        """
        if val_metrics is None or self.config.early_stopping_patience <= 0:
            return False

        current_metric = val_metrics.get(self.config.metric_for_best_model.replace('eval_', ''), 0.0)

        # 检查是否有改进
        improved = False
        if self.config.greater_is_better:
            if current_metric > self.best_metric + self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        else:
            if current_metric < self.best_metric - self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric

        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _run_training_report_generation(self) -> Optional[str]:
        """生成训练可视化报告并统一处理日志/异常。"""
        try:
            logger.info("正在生成训练可视化报告...")
            report_path = self.visualizer.generate_training_report()
            if report_path:
                logger.info(f"训练报告已生成: {report_path}")
            else:
                logger.warning("训练报告生成失败")
            return report_path
        except Exception as e:
            logger.error(f"生成可视化报告时出错: {e}")
            return None

    def _generate_training_report_on_end(self) -> Optional[str]:
        """按配置在训练结束时生成报告。

        默认异步启动后台 daemon 线程，避免 report HTML/图片生成阻塞
        train() 返回；如调用方需要确定报告已完成，可设置
        async_training_report=False。
        """
        if not getattr(self.config, "generate_training_report_on_end", True):
            logger.info("已跳过训练可视化报告生成")
            return None

        if not getattr(self.config, "async_training_report", True):
            return self._run_training_report_generation()

        report_thread = threading.Thread(
            target=self._run_training_report_generation,
            name="training_report_generator",
            daemon=True,
        )
        self._report_thread = report_thread
        report_thread.start()
        logger.info("训练可视化报告已在后台生成")
        return None
    
