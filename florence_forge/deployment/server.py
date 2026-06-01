#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型服务器

提供HTTP API接口用于模型推理服务
"""

import json
import logging
import time
import base64
import io
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

from ..utils.optional_dependencies import missing_dependency_message

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
    try:
        import python_multipart  # noqa: F401
        MULTIPART_AVAILABLE = True
    except ImportError:
        try:
            import multipart  # noqa: F401
            MULTIPART_AVAILABLE = True
        except ImportError:
            MULTIPART_AVAILABLE = False
except ImportError as e:
    FASTAPI_AVAILABLE = False
    MULTIPART_AVAILABLE = False
    # 延迟设置 logger，因为此时 logger 可能还未定义
    import logging as _logging
    _logging.getLogger(__name__).warning(
        f"{missing_dependency_message('FastAPI服务', 'fastapi 和 uvicorn')} ({e})"
    )

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError as e:
    PYDANTIC_AVAILABLE = False
    import logging as _logging
    _logging.getLogger(__name__).warning(
        f"{missing_dependency_message('服务请求模型', 'pydantic>=2.4.0')} ({e})"
    )
    # 如果pydantic不可用，创建一个占位符BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import numpy as np

import torch
from .backends import InferenceBackend, NativeInferenceBackend, VLLMInferenceBackend
from .inference import InferenceEngine

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """预测请求模型"""
    data: Union[str, List[float], List[List[float]]]  # base64编码的图像或数值数据
    format: str = "base64"  # 数据格式: base64, array
    return_raw: bool = False
    

class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool
    result: Any
    inference_time: float
    message: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """批量预测请求模型"""
    data_list: List[Union[str, List[float], List[List[float]]]]
    format: str = "base64"
    batch_size: Optional[int] = None
    return_raw: bool = False


class ServerStats(BaseModel):
    """服务器统计信息模型"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_inference_time: float
    uptime: float
    model_info: Dict[str, Any]


class ModelServer:
    """模型服务器
    
    提供HTTP API接口用于模型推理
    """
    
    def __init__(
        self,
        inference_engine: Union[InferenceEngine, InferenceBackend],
        host: str = "0.0.0.0",
        port: int = 8000,
        title: str = "Florence Forge Model Server",
        description: str = "Florence-2 模型推理服务",
        version: str = "1.0.0"
    ):
        """初始化模型服务器
        
        Args:
            inference_engine: 推理引擎或部署后端
            host: 服务器主机
            port: 服务器端口
            title: API标题
            description: API描述
            version: API版本
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                missing_dependency_message("FastAPI服务", "fastapi 和 uvicorn")
            )
        
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                missing_dependency_message("服务请求模型", "pydantic>=2.4.0")
            )
        
        if isinstance(inference_engine, InferenceBackend):
            self.inference_backend = inference_engine
            self.inference_engine = getattr(inference_engine, "engine", inference_engine)
        else:
            self.inference_engine = inference_engine
            self.inference_backend = NativeInferenceBackend(inference_engine)
        self.host = host
        self.port = port
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title=title,
            description=description,
            version=version
        )
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 服务器统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_inference_time": 0.0,
            "start_time": time.time()
        }
        
        # 设置路由
        self._setup_routes()
        
        logger.info(f"模型服务器初始化完成: {host}:{port}")
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/")
        async def root():
            """根路径"""
            return {
                "message": "Florence Forge Model Server",
                "status": "running",
                "endpoints": {
                    "predict": "/predict",
                    "predict_batch": "/predict/batch",
                    "health": "/health",
                    "stats": "/stats",
                    "model_info": "/model/info"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.stats["start_time"]
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """单次预测"""
            start_time = time.time()
            
            try:
                # 解析输入数据
                inputs = self._parse_input_data(request.data, request.format)
                
                # 执行推理
                result = self.inference_backend.predict(
                    inputs, 
                    return_raw=request.return_raw
                )
                
                # 转换结果为可序列化格式
                serializable_result = self._make_serializable(result)
                
                inference_time = time.time() - start_time
                
                # 更新统计信息
                self._update_stats(inference_time, success=True)
                
                return PredictionResponse(
                    success=True,
                    result=serializable_result,
                    inference_time=inference_time
                )
            
            except Exception as e:
                inference_time = time.time() - start_time
                self._update_stats(inference_time, success=False)
                
                logger.error(f"预测失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch")
        async def predict_batch(request: BatchPredictionRequest):
            """批量预测"""
            start_time = time.time()
            
            try:
                # 解析输入数据列表
                inputs_list = [
                    self._parse_input_data(data, request.format)
                    for data in request.data_list
                ]
                
                # 执行批量推理
                results = self.inference_backend.predict_batch(
                    inputs_list,
                    batch_size=request.batch_size
                )
                
                # 转换结果为可序列化格式
                serializable_results = [
                    self._make_serializable(result) for result in results
                ]
                
                inference_time = time.time() - start_time
                
                # 更新统计信息
                self._update_stats(inference_time, success=True)
                
                return {
                    "success": True,
                    "results": serializable_results,
                    "inference_time": inference_time,
                    "batch_size": len(inputs_list)
                }
            
            except Exception as e:
                inference_time = time.time() - start_time
                self._update_stats(inference_time, success=False)
                
                logger.error(f"批量预测失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        if MULTIPART_AVAILABLE:
            @self.app.post("/predict/upload")
            async def predict_upload(file: UploadFile = File(...)):
                """文件上传预测"""
                start_time = time.time()
                
                try:
                    # 读取文件内容
                    contents = await file.read()
                    content_type = file.content_type or ""
                    
                    # 根据文件类型处理
                    if content_type.startswith('image/'):
                        if not PIL_AVAILABLE:
                            raise HTTPException(
                                status_code=500, 
                                detail="需要安装PIL库处理图像文件"
                            )
                        
                        # 处理图像文件
                        image = Image.open(io.BytesIO(contents))
                        inputs = self._process_image(image)
                        
                    else:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"不支持的文件类型: {file.content_type}"
                        )
                    
                    # 执行推理
                    result = self.inference_backend.predict(inputs)
                    
                    # 转换结果
                    serializable_result = self._make_serializable(result)
                    
                    inference_time = time.time() - start_time
                    
                    # 更新统计信息
                    self._update_stats(inference_time, success=True)
                    
                    return {
                        "success": True,
                        "result": serializable_result,
                        "inference_time": inference_time,
                        "filename": file.filename
                    }
                
                except Exception as e:
                    inference_time = time.time() - start_time
                    self._update_stats(inference_time, success=False)
                    
                    logger.error(f"文件上传预测失败: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        else:
            @self.app.post("/predict/upload")
            async def predict_upload_unavailable():
                """文件上传预测依赖缺失时的占位路由"""
                raise HTTPException(
                    status_code=503,
                    detail=missing_dependency_message("文件上传预测", "python-multipart"),
                )
        
        @self.app.get("/stats", response_model=ServerStats)
        async def get_stats():
            """获取服务器统计信息"""
            uptime = time.time() - self.stats["start_time"]
            avg_inference_time = (
                self.stats["total_inference_time"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            )
            
            return ServerStats(
                total_requests=self.stats["total_requests"],
                successful_requests=self.stats["successful_requests"],
                failed_requests=self.stats["failed_requests"],
                avg_inference_time=avg_inference_time,
                uptime=uptime,
                model_info=self._get_model_info()
            )
        
        @self.app.get("/model/info")
        async def get_model_info():
            """获取模型信息"""
            return self._get_model_info()
        
        @self.app.post("/model/benchmark")
        async def benchmark_model(input_shape: List[int], num_runs: int = 100):
            """模型性能基准测试"""
            try:
                results = self.inference_backend.benchmark(
                    tuple(input_shape), 
                    num_runs=num_runs
                )
                return {
                    "success": True,
                    "benchmark_results": results
                }
            except Exception as e:
                logger.error(f"基准测试失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _parse_input_data(
        self, 
        data: Union[str, List[float], List[List[float]]], 
        format: str
    ) -> torch.Tensor:
        """解析输入数据
        
        Args:
            data: 输入数据
            format: 数据格式
            
        Returns:
            PyTorch张量
        """
        if format == "base64":
            # 解码base64图像
            if not isinstance(data, str):
                raise ValueError("base64格式数据必须是字符串")
            
            try:
                # 移除data URL前缀（如果存在）
                if data.startswith('data:'):
                    data = data.split(',')[1]
                
                image_bytes = base64.b64decode(data)
                
                if not PIL_AVAILABLE:
                    raise ValueError("需要安装PIL库处理base64图像")
                
                image = Image.open(io.BytesIO(image_bytes))
                return self._process_image(image)
                
            except Exception as e:
                raise ValueError(f"base64图像解码失败: {e}")
        
        elif format == "array":
            # 数值数组
            return torch.tensor(data, dtype=torch.float32)
        
        else:
            raise ValueError(f"不支持的数据格式: {format}")
    
    def _process_image(self, image: 'Image.Image') -> torch.Tensor:
        """处理PIL图像
        
        Args:
            image: PIL图像
            
        Returns:
            PyTorch张量
        """
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 转换为PyTorch张量 (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # 归一化到[0, 1]
        tensor = tensor / 255.0
        
        return tensor
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化格式
        
        Args:
            obj: 要转换的对象
            
        Returns:
            可序列化的对象
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def _update_stats(self, inference_time: float, success: bool) -> None:
        """更新统计信息
        
        Args:
            inference_time: 推理时间
            success: 是否成功
        """
        self.stats["total_requests"] += 1
        self.stats["total_inference_time"] += inference_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
    
    def _get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return self.inference_backend.get_model_info()
    
    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        """运行服务器
        
        Args:
            host: 主机地址
            port: 端口号
            **kwargs: uvicorn其他参数
        """
        host = host or self.host
        port = port or self.port
        
        logger.info(f"启动模型服务器: http://{host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs
        )
    
    async def start_async(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """异步启动服务器
        
        Args:
            host: 主机地址
            port: 端口号
        """
        host = host or self.host
        port = port or self.port
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def save_config(self, config_path: Union[str, Path]):
        """保存服务器配置
        
        Args:
            config_path: 配置文件路径
        """
        config = {
            "host": self.host,
            "port": self.port,
            "model_info": self._get_model_info(),
            "stats": self.stats
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"服务器配置已保存到: {config_path}")


def create_server(
    model_path: Union[str, Path],
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    backend: str = "native",
    **engine_kwargs
) -> ModelServer:
    """创建模型服务器的便捷函数
    
    Args:
        model_path: 模型路径
        host: 服务器主机
        port: 服务器端口
        device: 设备类型
        backend: 推理后端，支持 native 或 vllm
        **engine_kwargs: 推理引擎其他参数
        
    Returns:
        模型服务器实例
    """
    if backend == "native":
        inference_backend: Union[InferenceEngine, InferenceBackend] = InferenceEngine(
            model=model_path,
            device=device,
            **engine_kwargs
        )
    elif backend == "vllm":
        inference_backend = VLLMInferenceBackend(model=model_path, **engine_kwargs)
    else:
        raise ValueError(f"不支持的推理后端: {backend}")
    
    # 创建服务器
    server = ModelServer(
        inference_engine=inference_backend,
        host=host,
        port=port
    )
    
    return server


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Florence Forge Model Server")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--device", default="auto", help="设备类型")
    parser.add_argument("--backend", default="native", choices=["native", "vllm"], help="推理后端")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--use-amp", action="store_true", help="使用自动混合精度")
    parser.add_argument("--compile", action="store_true", help="编译模型")
    
    args = parser.parse_args()
    
    # 创建并运行服务器
    server = create_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        backend=args.backend,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        compile_model=args.compile
    )
    
    server.run()
