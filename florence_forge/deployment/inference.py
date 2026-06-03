#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推理引擎 — 提供高效的模型推理和批处理功能。"""

import logging
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Union, Optional, Callable, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from .inference_loading import InferenceModelLoader, resolve_device
from . import inference_parsing as parsing
from . import inference_visualization as visualization
from . import inference_runtime as runtime

logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎"""

    def __init__(
        self,
        model: Union[nn.Module, str, Path],
        device: str = "auto",
        batch_size: int = 1,
        use_amp: bool = False,
        compile_model: bool = False,
        allow_unsafe_torch_load: bool = False,
        model_revision: Optional[str] = None,
    ):
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.compile_model = compile_model
        self.allow_unsafe_torch_load = allow_unsafe_torch_load
        self.model_revision = model_revision

        loader = InferenceModelLoader(
            device=self.device,
            compile_model=compile_model,
            allow_unsafe_torch_load=allow_unsafe_torch_load,
            model_revision=model_revision,
        )
        self._loader = loader
        self.model = loader.load(model)
        self.model.eval()

        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_inference_time": 0.0,
            "throughput": 0.0,
        }
        self.preprocessor: Optional[Callable] = None
        self.postprocessor: Optional[Callable] = None
        logger.info("推理引擎初始化完成，设备: %s", self.device)

    def _parse_florence2_output(self, output_text: str, image_size: Tuple[int, int]):
        return parsing.parse_florence2_output(output_text, image_size)

    def _parse_ocr_with_region(self, model_output: str, image_size: Tuple[int, int]):
        return parsing.parse_ocr_with_region(model_output, image_size)

    def _parse_bboxes(self, model_output: str, image_size: Tuple[int, int]):
        return parsing.parse_bboxes(model_output, image_size)

    def _parse_segmentation_output(self, output_text: str, image_size: Tuple[int, int]):
        return parsing.parse_segmentation_output(output_text, image_size)

    def _visualize_detections(self, image, detections, save_path=None):
        return visualization.visualize_detections(image, detections, save_path)

    def _visualize_bboxes(self, image, bboxes, save_path):
        return visualization.visualize_bboxes(image, bboxes, save_path)

    def _visualize_ocr_with_region(self, image, ocr_results, save_path=None):
        return visualization.visualize_ocr_with_region(image, ocr_results, save_path)

    def _visualize_caption(self, image, caption: str, save_path=None):
        return visualization.visualize_caption(image, caption, save_path)

    def _visualize_segmentation(self, image, polygons, save_path=None, **kwargs):
        return visualization.visualize_segmentation(image, polygons, save_path, **kwargs)

    def _setup_device(self, device: str) -> torch.device:
        return resolve_device(device)

    def _build_model_config_kwargs(self, model_name: str, revision: Optional[str] = None):
        loader = getattr(self, "_loader", None) or InferenceModelLoader(device=self.device)
        return loader.build_model_config_kwargs(model_name, revision)

    def _load_model(self, model: Union[nn.Module, str, Path]) -> nn.Module:
        loader = getattr(self, "_loader", None) or InferenceModelLoader(
            device=self.device,
            compile_model=getattr(self, "compile_model", False),
            allow_unsafe_torch_load=getattr(self, "allow_unsafe_torch_load", False),
            model_revision=getattr(self, "model_revision", None),
        )
        return loader.load(model)

    def _load_torch_file(self, model_identifier: str) -> nn.Module:
        loader = getattr(self, "_loader", None) or InferenceModelLoader(device=self.device)
        return loader.load_torch_file(model_identifier)

    def set_preprocessor(self, preprocessor: Callable) -> None:
        """设置预处理函数"""
        self.preprocessor = preprocessor

    def set_postprocessor(self, postprocessor: Callable) -> None:
        """设置后处理函数"""
        self.postprocessor = postprocessor

    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, List, 'PIL.Image.Image'],
        task_prompt: Optional[str] = None,
        text_input: Optional[str] = None,
        return_raw: bool = False,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Union[torch.Tensor, Any]:
        """单次预测
        
        Args:
            inputs: 输入数据（支持PIL Image、numpy数组、tensor等）
            return_raw: 是否返回原始输出
            task_prompt: 任务提示（Florence2模型使用）
            text_input: 文本输入（某些任务需要）
            visualize: 是否在原图上可视化检测结果
            save_path: 可视化结果保存路径（如果不指定则显示图像）
            
        Returns:
            预测结果
        """
        start_time = time.time()
        if hasattr(self.model, "__class__"):
            logger.debug(
                "模型类型: %s, Florence2=%s",
                self.model.__class__,
                runtime.is_florence2_model(self.model),
            )
        outputs = runtime.run_predict_core(
            self.model,
            inputs,
            device=self.device,
            use_amp=self.use_amp,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            task_prompt=task_prompt,
            text_input=text_input,
            return_raw=return_raw,
            visualize=visualize,
            save_path=save_path,
        )
        self._update_stats(time.time() - start_time)
        return outputs
    
    def predict_batch(
        self,
        inputs_list: List[Union[torch.Tensor, np.ndarray, 'PIL.Image.Image']],
        batch_size: Optional[int] = None,
        task_prompt: str = "<OD>",
        text_input: Optional[str] = None,
        visualize: bool = False,
        save_dir: Optional[str] = None
    ) -> List[Any]:
        """批量预测
        
        Args:
            inputs_list: 输入数据列表（支持PIL Image、numpy数组、tensor等）
            batch_size: 批处理大小
            task_prompt: 任务提示（Florence2模型使用）
            text_input: 文本输入（某些任务需要）
            visualize: 是否在原图上可视化检测结果
            save_dir: 可视化结果保存目录（如果不指定则显示图像）
            
        Returns:
            预测结果列表
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        is_florence2 = runtime.is_florence2_model(self.model)
        if hasattr(self.model, "__class__"):
            logger.debug(
                "批量推理 - 模型类型: %s, Florence2=%s",
                self.model.__class__,
                is_florence2,
            )

        results = []

        if is_florence2:
            # Florence2模型：逐个处理（因为processor不支持真正的批处理）
            for i, inp in enumerate(inputs_list):
                try:
                    # 为每个图像生成保存路径
                    save_path = None
                    if visualize and save_dir:
                        import os
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"result_{i:04d}.png")
                    
                    result = self.predict(
                        inp, 
                        task_prompt=task_prompt, 
                        text_input=text_input,
                        visualize=visualize,
                        save_path=save_path
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量推理中单个样本失败: {e}")
                    results.append("")
        else:
            results = runtime.predict_batch_non_florence(
                self.model,
                inputs_list,
                device=self.device,
                use_amp=self.use_amp,
                batch_size=batch_size,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                update_stats=self._update_stats,
            )

        return results
    
    def predict_stream(
        self,
        input_generator,
        max_workers: int = 4
    ):
        """流式预测
        
        Args:
            input_generator: 输入数据生成器
            max_workers: 最大工作线程数
            
        Yields:
            预测结果
        """
        input_queue = Queue(maxsize=max_workers * 2)
        result_queue = Queue()
        
        def producer():
            """生产者线程"""
            try:
                for inputs in input_generator:
                    input_queue.put(inputs)
                input_queue.put(None)  # 结束标志
            except Exception as e:
                logger.error(f"生产者线程错误: {e}")
                input_queue.put(None)
        
        def consumer():
            """消费者线程"""
            try:
                while True:
                    try:
                        inputs = input_queue.get(timeout=1.0)
                        if inputs is None:
                            break
                        
                        result = self.predict(inputs)
                        result_queue.put(result)
                        
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"消费者线程错误: {e}")
                        result_queue.put(None)
            finally:
                result_queue.put(None)  # 结束标志
        
        # 启动线程
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # 生成结果
        try:
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result is None:
                        break
                    yield result
                except Empty:
                    continue
        finally:
            producer_thread.join()
            consumer_thread.join()
    
    def benchmark(
        self,
        input_shape: tuple,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """性能基准测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            warmup_runs: 预热次数
            
        Returns:
            性能指标
        """
        # 创建随机输入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 预热
        logger.info(f"预热 {warmup_runs} 次...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        
        # 同步（如果是CUDA）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 基准测试
        logger.info(f"基准测试 {num_runs} 次...")
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # 计算统计信息
        times = np.array(times)
        
        results = {
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "median_time_ms": np.median(times) * 1000,
            "fps": 1.0 / np.mean(times),
            "throughput_samples_per_sec": input_shape[0] / np.mean(times) if len(input_shape) > 0 else 1.0 / np.mean(times)
        }
        
        logger.info(f"基准测试完成: {results['avg_time_ms']:.2f}ms, {results['fps']:.2f}FPS")
        
        return results
    
    def profile_memory(self, input_shape: tuple) -> Dict[str, float]:
        """内存使用分析
        
        Args:
            input_shape: 输入形状
            
        Returns:
            内存使用信息
        """
        if self.device.type != 'cuda':
            logger.warning("内存分析仅支持CUDA设备")
            return {}
        
        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 创建输入并进行推理
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        results = {
            "initial_memory_mb": initial_memory / (1024 ** 2),
            "peak_memory_mb": peak_memory / (1024 ** 2),
            "current_memory_mb": current_memory / (1024 ** 2),
            "memory_increase_mb": (current_memory - initial_memory) / (1024 ** 2)
        }
        
        return results
    
    def _update_stats(self, inference_time: float, batch_size: int = 1) -> None:
        """更新统计信息
        
        Args:
            inference_time: 推理时间
            batch_size: 批处理大小
        """
        self.stats["total_inferences"] += batch_size
        self.stats["total_time"] += inference_time
        self.stats["avg_inference_time"] = self.stats["total_time"] / self.stats["total_inferences"]
        self.stats["throughput"] = self.stats["total_inferences"] / self.stats["total_time"]
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_inference_time": 0.0,
            "throughput": 0.0
        }
    
    def save_engine(self, save_path: Union[str, Path]) -> None:
        """保存推理引擎配置
        
        Args:
            save_path: 保存路径
        """
        import json
        
        config = {
            "device": str(self.device),
            "batch_size": self.batch_size,
            "use_amp": self.use_amp,
            "compile_model": self.compile_model,
            "stats": self.stats
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"推理引擎配置已保存到: {save_path}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 清理资源
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
