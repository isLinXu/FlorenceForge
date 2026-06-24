#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出器

支持将训练好的模型导出为不同格式，便于部署和推理
"""

import torch
import torch.nn as nn
import logging
import json
from typing import Union, Optional, List, Dict, Any
from pathlib import Path

from ..utils.optional_dependencies import missing_dependency_message

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器
    
    支持多种格式的模型导出
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """初始化导出器
        
        Args:
            model: 要导出的模型
            device: 设备类型
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def export_pytorch(self, save_path: Union[str, Path], **kwargs) -> None:
        """导出PyTorch格式模型
        
        Args:
            save_path: 保存路径
            **kwargs: 其他参数
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整模型
        if kwargs.get('save_full_model', False):
            torch.save(self.model, save_path)
            logger.info(f"完整模型已保存到: {save_path}")
        else:
            # 只保存状态字典
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"模型状态字典已保存到: {save_path}")
    
    def export_torchscript(
        self, 
        save_path: Union[str, Path],
        example_inputs: Optional[torch.Tensor] = None,
        method: str = "trace",
        **kwargs
    ) -> None:
        """导出TorchScript格式模型
        
        Args:
            save_path: 保存路径
            example_inputs: 示例输入
            method: 导出方法 (trace, script)
            **kwargs: 其他参数
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with torch.no_grad():
                if method == "trace":
                    if example_inputs is None:
                        raise ValueError("trace方法需要提供example_inputs")
                    
                    # 确保输入在正确的设备上
                    if isinstance(example_inputs, (list, tuple)):
                        example_inputs = [inp.to(self.device) for inp in example_inputs]
                    else:
                        example_inputs = example_inputs.to(self.device)
                    
                    traced_model = torch.jit.trace(self.model, example_inputs)
                    
                elif method == "script":
                    traced_model = torch.jit.script(self.model)
                    
                else:
                    raise ValueError(f"不支持的导出方法: {method}")
                
                traced_model.save(str(save_path))
                logger.info(f"TorchScript模型已保存到: {save_path}")
        
        except Exception as e:
            logger.error(f"TorchScript导出失败: {e}")
            raise
    
    def export_onnx(
        self,
        save_path: Union[str, Path],
        example_inputs: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        **kwargs
    ) -> None:
        """导出ONNX格式模型
        
        Args:
            save_path: 保存路径
            example_inputs: 示例输入
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴配置
            opset_version: ONNX操作集版本
            **kwargs: 其他参数
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 确保输入在正确的设备上
            if isinstance(example_inputs, (list, tuple)):
                example_inputs = [inp.to(self.device) for inp in example_inputs]
            else:
                example_inputs = example_inputs.to(self.device)
            
            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    example_inputs,
                    str(save_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    **kwargs
                )
            
            logger.info(f"ONNX模型已保存到: {save_path}")
            
            # 验证ONNX模型
            self._verify_onnx_model(save_path)
        
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: Union[str, Path]) -> None:
        """验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # 加载并检查ONNX模型
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话测试
            ort.InferenceSession(str(onnx_path))
            
            logger.info("ONNX模型验证通过")
            
        except ImportError:
            logger.warning(
                missing_dependency_message(
                    "ONNX模型验证",
                    "onnx, onnxruntime",
                    "evaluation",
                )
            )
        except Exception as e:
            logger.warning(f"ONNX模型验证失败: {e}")
    
    def export_tensorrt(
        self,
        save_path: Union[str, Path],
        onnx_path: Optional[Union[str, Path]] = None,
        precision: str = "fp16",
        max_batch_size: int = 1,
        **kwargs
    ) -> None:
        """导出TensorRT格式模型
        
        Args:
            save_path: 保存路径
            onnx_path: ONNX模型路径（如果提供）
            precision: 精度模式 (fp32, fp16, int8)
            max_batch_size: 最大批次大小
            **kwargs: 其他参数
        """
        try:
            import tensorrt as trt
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果没有提供ONNX路径，先导出ONNX
            if onnx_path is None:
                onnx_path = save_path.with_suffix('.onnx')
                logger.info("先导出ONNX模型...")
                # 这里需要示例输入，实际使用时需要提供
                # self.export_onnx(onnx_path, example_inputs)
            
            logger.info(f"开始转换TensorRT模型: {precision}精度")
            
            # 创建TensorRT构建器
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # 解析ONNX模型
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("ONNX解析失败")
            
            # 配置构建器
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # INT8需要校准数据，这里简化处理
                logger.warning("INT8量化需要校准数据，请确保已正确配置")
            
            # 构建引擎
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("TensorRT引擎构建失败")
            
            # 保存引擎
            with open(save_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT模型已保存到: {save_path}")
        
        except ImportError as exc:
            logger.error(
                missing_dependency_message(
                    "TensorRT导出",
                    "tensorrt",
                )
            )
            raise ImportError(
                missing_dependency_message("TensorRT导出", "tensorrt")
            ) from exc
        except Exception as e:
            logger.error(f"TensorRT导出失败: {e}")
            raise
    
    def export_coreml(
        self,
        save_path: Union[str, Path],
        example_inputs: torch.Tensor,
        **kwargs
    ) -> None:
        """导出Core ML格式模型（macOS/iOS）
        
        Args:
            save_path: 保存路径
            example_inputs: 示例输入
            **kwargs: 其他参数
        """
        try:
            import coremltools as ct
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 先转换为TorchScript
            with torch.no_grad():
                example_inputs = example_inputs.to(self.device)
                traced_model = torch.jit.trace(self.model, example_inputs)
            
            # 转换为Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_inputs.shape)],
                **kwargs
            )
            
            # 保存模型
            coreml_model.save(str(save_path))
            
            logger.info(f"Core ML模型已保存到: {save_path}")
        
        except ImportError as exc:
            logger.error(
                missing_dependency_message(
                    "Core ML导出",
                    "coremltools",
                )
            )
            raise ImportError(
                missing_dependency_message("Core ML导出", "coremltools")
            ) from exc
        except Exception as e:
            logger.error(f"Core ML导出失败: {e}")
            raise
    
    def export_multiple_formats(
        self,
        output_dir: Union[str, Path],
        formats: List[str],
        example_inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Path]:
        """批量导出多种格式
        
        Args:
            output_dir: 输出目录
            formats: 要导出的格式列表
            example_inputs: 示例输入
            **kwargs: 其他参数
            
        Returns:
            格式到文件路径的映射
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt == "pytorch":
                    save_path = output_dir / "model.pth"
                    self.export_pytorch(save_path, **kwargs)
                    
                elif fmt == "torchscript":
                    save_path = output_dir / "model.pt"
                    self.export_torchscript(save_path, example_inputs, **kwargs)
                    
                elif fmt == "onnx":
                    save_path = output_dir / "model.onnx"
                    if example_inputs is None:
                        logger.warning(f"跳过{fmt}导出：需要example_inputs")
                        continue
                    self.export_onnx(save_path, example_inputs, **kwargs)
                    
                elif fmt == "tensorrt":
                    save_path = output_dir / "model.trt"
                    self.export_tensorrt(save_path, **kwargs)
                    
                elif fmt == "coreml":
                    save_path = output_dir / "model.mlmodel"
                    if example_inputs is None:
                        logger.warning(f"跳过{fmt}导出：需要example_inputs")
                        continue
                    self.export_coreml(save_path, example_inputs, **kwargs)
                    
                else:
                    logger.warning(f"不支持的导出格式: {fmt}")
                    continue
                
                results[fmt] = save_path
                
            except Exception as e:
                logger.error(f"{fmt}格式导出失败: {e}")
        
        # 保存导出信息
        export_info = {
            "formats": list(results.keys()),
            "files": {fmt: str(path) for fmt, path in results.items()},
            "model_info": self._get_model_info()
        }
        
        info_path = output_dir / "export_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"导出完成，信息已保存到: {info_path}")
        
        return results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            "model_class": self.model.__class__.__name__,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # 尝试获取模型大小
        try:
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            info["model_size_mb"] = model_size / (1024 * 1024)
            
            # 获取参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["total_parameters"] = total_params
            info["trainable_parameters"] = trainable_params
            
            # 获取模型层数信息
            info["num_layers"] = len(list(self.model.modules()))
            
        except Exception as e:
            logger.warning(f"获取模型信息时出错: {e}")
            info["model_size_mb"] = "unknown"
            info["total_parameters"] = "unknown"
            info["trainable_parameters"] = "unknown"
        
        return info
    
    def benchmark_formats(
        self,
        formats: List[str],
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """基准测试不同格式的性能
        
        Args:
            formats: 要测试的格式列表
            example_inputs: 示例输入
            num_runs: 运行次数
            
        Returns:
            性能测试结果
        """
        import time
        import tempfile
        
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            for fmt in formats:
                try:
                    # 导出模型
                    if fmt == "pytorch":
                        # 原始PyTorch模型
                        model = self.model
                        
                    elif fmt == "torchscript":
                        save_path = temp_dir / "model.pt"
                        self.export_torchscript(save_path, example_inputs)
                        model = torch.jit.load(save_path)
                        
                    else:
                        logger.warning(f"基准测试暂不支持格式: {fmt}")
                        continue
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(10):
                            _ = model(example_inputs)
                    
                    # 计时
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(num_runs):
                            _ = model(example_inputs)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / num_runs
                    fps = 1.0 / avg_time
                    
                    results[fmt] = {
                        "avg_inference_time_ms": avg_time * 1000,
                        "fps": fps
                    }
                    
                    logger.info(f"{fmt}: {avg_time*1000:.2f}ms, {fps:.2f}FPS")
                    
                except Exception as e:
                    logger.error(f"{fmt}基准测试失败: {e}")
        
        return results
