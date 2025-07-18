"""模型量化"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self):
        self.quantization_config = {
            'backend': 'fbgemm',  # 或 'qnnpack' for mobile
            'reduce_range': False,
            'qconfig_spec': None
        }
    
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """准备模型进行量化"""
        try:
            # 设置量化配置
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig(self.quantization_config['backend'])
            
            # 准备量化
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            logger.info("模型已准备好进行量化")
            return prepared_model
            
        except Exception as e:
            logger.error(f"准备模型量化时出错: {e}")
            return model
    
    def calibrate_model(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """校准模型"""
        try:
            model.eval()
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_data):
                    if batch_idx >= 100:  # 限制校准样本数量
                        break
                    _ = model(data)
            
            logger.info("模型校准完成")
            return model
            
        except Exception as e:
            logger.error(f"模型校准时出错: {e}")
            return model
    
    def quantize_model(self, prepared_model: nn.Module) -> nn.Module:
        """量化模型"""
        try:
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            logger.info("模型量化完成")
            return quantized_model
            
        except Exception as e:
            logger.error(f"模型量化时出错: {e}")
            return prepared_model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """获取模型大小信息"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }
