"""FlorenceForge模型合并器模块

提供LoRA权重合并和模型导出功能
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoProcessor

from ..core.model import Florence2MultiTaskModel
from .lora_manager import LoRAManager

logger = logging.getLogger(__name__)

class ModelMerger:
    """模型合并器
    
    提供LoRA权重与基础模型合并的功能
    """
    
    def __init__(self, lora_manager: Optional[LoRAManager] = None):
        """初始化模型合并器
        
        Args:
            lora_manager: LoRA管理器实例
        """
        self.lora_manager = lora_manager or LoRAManager()
        
        logger.info("模型合并器初始化完成")
    
    def merge_lora_weights(
        self,
        base_model: Florence2MultiTaskModel,
        lora_model: PeftModel,
        task_name: Optional[str] = None,
        merge_strategy: str = "linear"
    ) -> Florence2MultiTaskModel:
        """合并LoRA权重到基础模型
        
        Args:
            base_model: 基础模型
            lora_model: LoRA模型
            task_name: 任务名称
            merge_strategy: 合并策略 ('linear', 'weighted')
            
        Returns:
            合并后的模型
        """
        logger.info(f"开始合并LoRA权重，任务: {task_name}, 策略: {merge_strategy}")
        
        try:
            # 获取LoRA权重
            lora_state_dict = get_peft_model_state_dict(lora_model)
            
            # 创建合并后的模型
            merged_model = self._create_merged_model(
                base_model, lora_state_dict, merge_strategy
            )
            
            logger.info("LoRA权重合并完成")
            return merged_model
            
        except Exception as e:
            logger.error(f"LoRA权重合并失败: {e}")
            raise
    
    def merge_and_unload(
        self,
        peft_model: PeftModel,
        output_dir: Union[str, Path],
        save_tokenizer: bool = True,
        save_processor: bool = True
    ) -> None:
        """合并LoRA权重并卸载PEFT包装器
        
        Args:
            peft_model: PEFT模型
            output_dir: 输出目录
            save_tokenizer: 是否保存tokenizer
            save_processor: 是否保存processor
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 合并并卸载LoRA权重
            merged_model = peft_model.merge_and_unload()
            
            # 保存合并后的模型
            merged_model.save_pretrained(output_path)
            
            # 保存tokenizer和processor
            if save_tokenizer and hasattr(peft_model, 'tokenizer'):
                peft_model.tokenizer.save_pretrained(output_path)
            
            if save_processor and hasattr(peft_model, 'processor'):
                peft_model.processor.save_pretrained(output_path)
            
            # 保存合并信息
            merge_info = {
                'merge_timestamp': torch.utils.data.get_worker_info(),
                'base_model_name': getattr(merged_model.config, 'name_or_path', 'unknown'),
                'lora_merged': True,
                'merge_strategy': 'peft_merge_and_unload'
            }
            
            with open(output_path / 'merge_info.json', 'w', encoding='utf-8') as f:
                json.dump(merge_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型合并并保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"模型合并失败: {e}")
            raise
    
    def merge_multiple_adapters(
        self,
        base_model: Florence2MultiTaskModel,
        adapter_paths: Dict[str, Union[str, Path]],
        weights: Optional[Dict[str, float]] = None,
        output_dir: Union[str, Path] = "./merged_model"
    ) -> Florence2MultiTaskModel:
        """合并多个LoRA适配器
        
        Args:
            base_model: 基础模型
            adapter_paths: 适配器路径字典 {task_name: path}
            weights: 各适配器权重 {task_name: weight}
            output_dir: 输出目录
            
        Returns:
            合并后的模型
        """
        logger.info(f"开始合并多个适配器: {list(adapter_paths.keys())}")
        
        if weights is None:
            weights = {task: 1.0 for task in adapter_paths.keys()}
        
        try:
            # 加载所有适配器的权重
            all_adapter_weights = {}
            
            for task_name, adapter_path in adapter_paths.items():
                # 加载适配器
                peft_model = PeftModel.from_pretrained(base_model, adapter_path)
                adapter_weights = get_peft_model_state_dict(peft_model)
                
                # 应用权重
                task_weight = weights.get(task_name, 1.0)
                weighted_weights = {
                    k: v * task_weight for k, v in adapter_weights.items()
                }
                
                # 累加权重
                for key, value in weighted_weights.items():
                    if key in all_adapter_weights:
                        all_adapter_weights[key] += value
                    else:
                        all_adapter_weights[key] = value
            
            # 创建合并后的模型
            merged_model = self._create_merged_model(
                base_model, all_adapter_weights, "weighted"
            )
            
            # 保存合并后的模型
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(output_path)
            
            # 保存合并信息
            merge_info = {
                'merged_adapters': list(adapter_paths.keys()),
                'adapter_weights': weights,
                'merge_strategy': 'weighted_sum',
                'total_adapters': len(adapter_paths)
            }
            
            with open(output_path / 'multi_adapter_merge_info.json', 'w', encoding='utf-8') as f:
                json.dump(merge_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"多适配器合并完成，保存到: {output_path}")
            return merged_model
            
        except Exception as e:
            logger.error(f"多适配器合并失败: {e}")
            raise
    
    def _create_merged_model(
        self,
        base_model: Florence2MultiTaskModel,
        lora_weights: Dict[str, torch.Tensor],
        merge_strategy: str
    ) -> Florence2MultiTaskModel:
        """创建合并后的模型
        
        Args:
            base_model: 基础模型
            lora_weights: LoRA权重字典
            merge_strategy: 合并策略
            
        Returns:
            合并后的模型
        """
        # 创建模型副本
        merged_model = type(base_model)(
            model_name=base_model.model_name,
            device=base_model.device
        )
        
        # 复制基础模型权重
        merged_model.load_state_dict(base_model.state_dict(), strict=False)
        
        # 合并LoRA权重
        if merge_strategy == "linear":
            self._linear_merge(merged_model, lora_weights)
        elif merge_strategy == "weighted":
            self._weighted_merge(merged_model, lora_weights)
        else:
            raise ValueError(f"不支持的合并策略: {merge_strategy}")
        
        return merged_model
    
    def _linear_merge(
        self,
        model: Florence2MultiTaskModel,
        lora_weights: Dict[str, torch.Tensor]
    ) -> None:
        """线性合并LoRA权重"""
        model_state = model.state_dict()
        
        for name, lora_weight in lora_weights.items():
            if name in model_state:
                # 直接添加LoRA权重
                model_state[name] += lora_weight
        
        model.load_state_dict(model_state)
    
    def _weighted_merge(
        self,
        model: Florence2MultiTaskModel,
        lora_weights: Dict[str, torch.Tensor]
    ) -> None:
        """加权合并LoRA权重"""
        model_state = model.state_dict()
        
        for name, lora_weight in lora_weights.items():
            if name in model_state:
                # 加权合并
                model_state[name] = model_state[name] + lora_weight
        
        model.load_state_dict(model_state)
    
    def export_merged_model(
        self,
        merged_model: Florence2MultiTaskModel,
        output_dir: Union[str, Path],
        export_format: str = "pytorch",
        include_config: bool = True
    ) -> None:
        """导出合并后的模型
        
        Args:
            merged_model: 合并后的模型
            output_dir: 输出目录
            export_format: 导出格式 ('pytorch', 'onnx', 'torchscript')
            include_config: 是否包含配置文件
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if export_format == "pytorch":
                # 保存PyTorch模型
                merged_model.save_pretrained(output_path)
                
            elif export_format == "onnx":
                # 导出ONNX格式
                self._export_onnx(merged_model, output_path)
                
            elif export_format == "torchscript":
                # 导出TorchScript格式
                self._export_torchscript(merged_model, output_path)
                
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            # 保存导出信息
            if include_config:
                export_info = {
                    'export_format': export_format,
                    'model_type': 'florence2_multitask',
                    'export_timestamp': str(torch.utils.data.get_worker_info()),
                    'model_size_mb': self._get_model_size_mb(output_path)
                }
                
                with open(output_path / 'export_info.json', 'w', encoding='utf-8') as f:
                    json.dump(export_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型已导出到: {output_path} (格式: {export_format})")
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def _export_onnx(
        self,
        model: Florence2MultiTaskModel,
        output_path: Path
    ) -> None:
        """导出ONNX格式"""
        try:
            import onnx
            
            # 创建示例输入
            dummy_input_ids = torch.randint(0, 1000, (1, 512))
            dummy_pixel_values = torch.randn(1, 3, 224, 224)
            
            # 导出ONNX
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_pixel_values),
                output_path / "model.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'pixel_values'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'pixel_values': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        except ImportError:
            logger.warning("ONNX库未安装，跳过ONNX导出")
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise
    
    def _export_torchscript(
        self,
        model: Florence2MultiTaskModel,
        output_path: Path
    ) -> None:
        """导出TorchScript格式"""
        try:
            # 设置为评估模式
            model.eval()
            
            # 创建示例输入
            dummy_input_ids = torch.randint(0, 1000, (1, 512))
            dummy_pixel_values = torch.randn(1, 3, 224, 224)
            
            # 追踪模型
            traced_model = torch.jit.trace(
                model, (dummy_input_ids, dummy_pixel_values)
            )
            
            # 保存TorchScript模型
            traced_model.save(output_path / "model.pt")
            
        except Exception as e:
            logger.error(f"TorchScript导出失败: {e}")
            raise
    
    def _get_model_size_mb(self, model_path: Path) -> float:
        """获取模型大小(MB)"""
        try:
            total_size = 0
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # 转换为MB
        except Exception:
            return 0.0
    
    def merge_all_adapters(
        self,
        peft_model: PeftModel,
        adapter_weights: Optional[Dict[str, float]] = None
    ) -> Florence2MultiTaskModel:
        """合并所有适配器
        
        Args:
            peft_model: PEFT模型
            adapter_weights: 适配器权重
            
        Returns:
            合并后的模型
        """
        try:
            # 获取所有适配器名称
            adapter_names = list(peft_model.peft_config.keys())
            
            if not adapter_names:
                logger.warning("没有找到适配器，返回基础模型")
                return peft_model.base_model
            
            # 如果只有一个适配器，直接合并
            if len(adapter_names) == 1:
                return peft_model.merge_and_unload()
            
            # 多适配器合并
            if adapter_weights is None:
                adapter_weights = {name: 1.0 for name in adapter_names}
            
            # 获取所有适配器权重
            all_weights = {}
            for adapter_name in adapter_names:
                peft_model.set_adapter(adapter_name)
                adapter_state = get_peft_model_state_dict(peft_model)
                weight = adapter_weights.get(adapter_name, 1.0)
                
                for key, value in adapter_state.items():
                    if key in all_weights:
                        all_weights[key] += value * weight
                    else:
                        all_weights[key] = value * weight
            
            # 创建合并后的模型
            merged_model = self._create_merged_model(
                peft_model.base_model, all_weights, "weighted"
            )
            
            return merged_model
            
        except Exception as e:
            logger.error(f"合并所有适配器失败: {e}")
            raise
    
    def validate_merged_model(
        self,
        merged_model: Florence2MultiTaskModel,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """验证合并后的模型
        
        Args:
            merged_model: 合并后的模型
            test_inputs: 测试输入
            
        Returns:
            验证结果
        """
        logger.info("开始验证合并后的模型...")
        
        validation_results = {
            'model_loadable': False,
            'forward_pass': False,
            'parameter_count': 0,
            'model_size_mb': 0.0,
            'errors': []
        }
        
        try:
            # 检查模型是否可加载
            validation_results['model_loadable'] = True
            
            # 计算参数数量
            validation_results['parameter_count'] = sum(
                p.numel() for p in merged_model.parameters()
            )
            
            # 测试前向传播
            if test_inputs is None:
                test_inputs = {
                    'input_ids': torch.randint(0, 1000, (1, 512)),
                    'pixel_values': torch.randn(1, 3, 224, 224)
                }
            
            merged_model.eval()
            with torch.no_grad():
                output = merged_model.generate(
                    input_ids=test_inputs['input_ids'],
                    pixel_values=test_inputs['pixel_values'],
                    max_new_tokens=10
                )
                validation_results['forward_pass'] = True
            
            logger.info("模型验证通过")
            
        except Exception as e:
            error_msg = f"模型验证失败: {e}"
            validation_results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return validation_results