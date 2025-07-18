#!/usr/bin/env python3
"""
Florence Forge - LoRA管理器

提供LoRA配置管理、参数优化和模型适配功能
"""

import logging
import json
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
except ImportError:
    # 如果PEFT不可用，定义占位符
    class LoraConfig:
        pass
    class PeftModel:
        pass
    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"
    def get_peft_model(*args, **kwargs):
        raise ImportError("PEFT library not available")

from ..core.config import LoRAConfig as ForgeLoRAConfig

logger = logging.getLogger(__name__)

class LoRAManager:
    """LoRA管理器
    
    负责LoRA配置的创建、管理和优化
    """
    
    def __init__(self, base_config: Optional[ForgeLoRAConfig] = None):
        """初始化LoRA管理器
        
        Args:
            base_config: 基础LoRA配置
        """
        self.base_config = base_config or ForgeLoRAConfig()
        self.task_configs: Dict[str, ForgeLoRAConfig] = {}
        self.active_adapters: Dict[str, str] = {}  # task -> adapter_name
        
        logger.info("LoRA管理器初始化完成")
    
    def create_task_config(
        self,
        task_name: str,
        r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        lora_dropout: Optional[float] = None,
        **kwargs
    ) -> ForgeLoRAConfig:
        """为特定任务创建LoRA配置
        
        Args:
            task_name: 任务名称
            r: LoRA秩
            lora_alpha: LoRA alpha参数
            target_modules: 目标模块列表
            lora_dropout: LoRA dropout率
            **kwargs: 其他配置参数
            
        Returns:
            任务特定的LoRA配置
        """
        # 从基础配置开始
        config = ForgeLoRAConfig(
            r=r or self.base_config.r,
            lora_alpha=lora_alpha or self.base_config.lora_alpha,
            target_modules=target_modules or self.base_config.target_modules.copy(),
            lora_dropout=lora_dropout or self.base_config.lora_dropout,
            bias=kwargs.get('bias', self.base_config.bias),
            task_type=kwargs.get('task_type', self.base_config.task_type)
        )
        
        # 根据任务类型调整配置
        config = self._optimize_config_for_task(task_name, config)
        
        self.task_configs[task_name] = config
        logger.info(f"为任务 {task_name} 创建LoRA配置: r={config.r}, alpha={config.lora_alpha}")
        
        return config
    
    def _optimize_config_for_task(self, task_name: str, config: ForgeLoRAConfig) -> ForgeLoRAConfig:
        """根据任务类型优化LoRA配置
        
        Args:
            task_name: 任务名称
            config: 基础配置
            
        Returns:
            优化后的配置
        """
        # 任务特定的优化策略
        task_optimizations = {
            # 图像描述任务 - 需要较强的语言生成能力
            'CAPTION': {'r': 32, 'lora_alpha': 32},
            'DETAILED_CAPTION': {'r': 48, 'lora_alpha': 48},
            'MORE_DETAILED_CAPTION': {'r': 64, 'lora_alpha': 64},
            
            # 目标检测任务 - 需要空间理解能力
            'OD': {'r': 32, 'lora_alpha': 32},
            'OPEN_VOCABULARY_DETECTION': {'r': 48, 'lora_alpha': 48},
            
            # OCR任务 - 需要细粒度特征提取
            'OCR': {'r': 24, 'lora_alpha': 24},
            'OCR_WITH_REGION': {'r': 32, 'lora_alpha': 32},
            
            # 分割任务 - 需要精确的空间定位
            'REGION_TO_SEGMENTATION': {'r': 48, 'lora_alpha': 48},
            'REFERRING_EXPRESSION_SEGMENTATION': {'r': 64, 'lora_alpha': 64},
            
            # 区域分析任务
            'REGION_PROPOSAL': {'r': 32, 'lora_alpha': 32},
            'REGION_TO_CATEGORY': {'r': 24, 'lora_alpha': 24},
            'REGION_TO_DESCRIPTION': {'r': 32, 'lora_alpha': 32},
            
            # 复合任务
            'CAPTION_TO_PHRASE_GROUNDING': {'r': 48, 'lora_alpha': 48},
            'DENSE_REGION_CAPTION': {'r': 56, 'lora_alpha': 56}
        }
        
        if task_name in task_optimizations:
            optimization = task_optimizations[task_name]
            config.r = optimization.get('r', config.r)
            config.lora_alpha = optimization.get('lora_alpha', config.lora_alpha)
            
            # 根据任务复杂度调整目标模块
            if optimization.get('r', 32) >= 48:
                # 复杂任务包含更多模块
                config.target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ]
        
        return config
    
    def create_peft_config(self, task_name: str) -> LoraConfig:
        """创建PEFT库兼容的LoRA配置
        
        Args:
            task_name: 任务名称
            
        Returns:
            PEFT LoraConfig对象
        """
        if task_name not in self.task_configs:
            self.create_task_config(task_name)
        
        forge_config = self.task_configs[task_name]
        
        return LoraConfig(
            r=forge_config.r,
            lora_alpha=forge_config.lora_alpha,
            target_modules=forge_config.target_modules,
            lora_dropout=forge_config.lora_dropout,
            bias=forge_config.bias,
            task_type=TaskType.CAUSAL_LM
        )
    
    def apply_lora_to_model(
        self,
        model: nn.Module,
        task_name: str,
        adapter_name: Optional[str] = None
    ) -> PeftModel:
        """将LoRA应用到模型
        
        Args:
            model: 基础模型
            task_name: 任务名称
            adapter_name: 适配器名称
            
        Returns:
            应用LoRA的模型
        """
        if adapter_name is None:
            adapter_name = f"lora_{task_name}"
        
        peft_config = self.create_peft_config(task_name)
        
        try:
            self.active_adapters[task_name] = adapter_name
            
            logger.info(f"LoRA已应用到模型，任务: {task_name}, 适配器: {adapter_name}")
            return peft_model
        
        except Exception as e:
            logger.error(f"应用LoRA失败: {e}")
            raise
    
    def add_adapter_to_model(
        self,
        model: PeftModel,
        task_name: str,
        adapter_name: Optional[str] = None
    ) -> None:
        """向已有PEFT模型添加新适配器
        
        Args:
            model: PEFT模型
            task_name: 任务名称
            adapter_name: 适配器名称
        """
        if adapter_name is None:
            adapter_name = f"lora_{task_name}"
        
        peft_config = self.create_peft_config(task_name)
        
        try:
            self.active_adapters[task_name] = adapter_name
            
            logger.info(f"新适配器已添加，任务: {task_name}, 适配器: {adapter_name}")
        
        except Exception as e:
            logger.error(f"添加适配器失败: {e}")
            raise
    
    def switch_adapter(self, model: PeftModel, task_name: str) -> None:
        """切换模型的活跃适配器
        
        Args:
            model: PEFT模型
            task_name: 任务名称
        """
        if task_name not in self.active_adapters:
            raise ValueError(f"任务 {task_name} 没有对应的适配器")
        
        adapter_name = self.active_adapters[task_name]
        
        try:
            model.set_adapter(adapter_name)
            logger.debug(f"已切换到适配器: {adapter_name} (任务: {task_name})")
        
        except Exception as e:
            logger.error(f"切换适配器失败: {e}")
            raise
    
    def get_trainable_parameters(self, model: PeftModel) -> Tuple[int, int, float]:
        """获取可训练参数信息
        
        Args:
            model: PEFT模型
            
        Returns:
            (可训练参数数, 总参数数, 可训练比例)
        """
        trainable_params = 0
        total_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        return trainable_params, total_params, trainable_ratio
    
    def print_trainable_parameters(self, model: PeftModel) -> None:
        """打印可训练参数信息
        
        Args:
            model: PEFT模型
        """
        trainable_params, total_params, trainable_ratio = self.get_trainable_parameters(model)
        
        logger.info(
            f"可训练参数: {trainable_params:,} || "
            f"总参数: {total_params:,} || "
            f"可训练比例: {trainable_ratio:.2%}"
        )
    
    def save_adapter(
        self,
        model: PeftModel,
        save_directory: Union[str, Path],
        adapter_name: Optional[str] = None
    ) -> None:
        """保存适配器
        
        Args:
            model: PEFT模型
            save_directory: 保存目录
            adapter_name: 适配器名称
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if adapter_name:
                # 保存特定适配器
                model.save_pretrained(save_path, selected_adapters=[adapter_name])
            else:
                # 保存所有适配器
                model.save_pretrained(save_path)
            
            # 保存管理器状态
            self.save_manager_state(save_path / "lora_manager_state.json")
            
            logger.info(f"适配器已保存到: {save_path}")
        
        except Exception as e:
            logger.error(f"保存适配器失败: {e}")
            raise
    
    def load_adapter(
        self,
        model: nn.Module,
        load_directory: Union[str, Path],
        adapter_name: Optional[str] = None
    ) -> PeftModel:
        """加载适配器
        
        Args:
            model: 基础模型
            load_directory: 加载目录
            adapter_name: 适配器名称
            
        Returns:
            加载适配器的PEFT模型
        """
        load_path = Path(load_directory)
        
        try:
            # 加载PEFT模型
            peft_model = PeftModel.from_pretrained(model, load_path, adapter_name=adapter_name)
            
            # 加载管理器状态
            manager_state_path = load_path / "lora_manager_state.json"
            if manager_state_path.exists():
                self.load_manager_state(manager_state_path)
            
            logger.info(f"适配器已从 {load_path} 加载")
            return peft_model
        
        except Exception as e:
            logger.error(f"加载适配器失败: {e}")
            raise
    
    def optimize_config_for_memory(
        self,
        task_name: str,
        target_memory_gb: float,
        base_memory_gb: float
    ) -> ForgeLoRAConfig:
        """根据内存限制优化LoRA配置
        
        Args:
            task_name: 任务名称
            target_memory_gb: 目标内存使用量(GB)
            base_memory_gb: 基础模型内存使用量(GB)
            
        Returns:
            优化后的配置
        """
        available_memory = target_memory_gb - base_memory_gb
        
        if available_memory <= 0:
            logger.warning("可用内存不足，使用最小配置")
            return ForgeLoRAConfig(r=8, lora_alpha=8)
        
        # 估算LoRA参数内存使用量 (简化计算)
        # 假设每个参数4字节，LoRA参数数量约为 r * (input_dim + output_dim) * num_layers
        estimated_lora_params_per_r = 1000000  # 每个r值对应的参数数量估计
        bytes_per_param = 4
        
        max_r = int(available_memory * 1024**3 / (estimated_lora_params_per_r * bytes_per_param))
        max_r = min(max_r, 128)  # 限制最大r值
        max_r = max(max_r, 8)    # 确保最小r值
        
        # 创建优化配置
        if task_name not in self.task_configs:
            self.create_task_config(task_name)
        
        config = self.task_configs[task_name]
        config.r = min(config.r, max_r)
        config.lora_alpha = config.r  # 保持alpha = r的关系
        
        logger.info(f"内存优化完成，任务: {task_name}, r: {config.r}, 预计内存: {available_memory:.2f}GB")
        
        return config
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要
        
        Returns:
            配置摘要字典
        """
        return {
            "base_config": self.base_config.to_dict(),
            "task_configs": {
                task: config.to_dict()
                for task, config in self.task_configs.items()
            },
            "active_adapters": self.active_adapters.copy(),
            "num_tasks": len(self.task_configs)
        }
    
    def save_manager_state(self, file_path: Union[str, Path]) -> None:
        """保存管理器状态
        
        Args:
            file_path: 文件路径
        """
        state = {
            "base_config": self.base_config.to_dict(),
            "task_configs": {
                task: config.to_dict()
                for task, config in self.task_configs.items()
            },
            "active_adapters": self.active_adapters
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"管理器状态已保存到: {file_path}")
    
    def load_manager_state(self, file_path: Union[str, Path]) -> None:
        """加载管理器状态
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # 重建配置对象
        self.base_config = ForgeLoRAConfig(**state["base_config"])
        
        self.task_configs = {
            task: ForgeLoRAConfig(**config_dict)
            for task, config_dict in state["task_configs"].items()
        }
        
        self.active_adapters = state["active_adapters"]
        
        logger.info(f"管理器状态已从 {file_path} 加载")
    
    def clear_all_configs(self) -> None:
        """清空所有配置"""
        self.task_configs.clear()
        self.active_adapters.clear()
        logger.info("所有LoRA配置已清空")