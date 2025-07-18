#!/usr/bin/env python3
"""
性能基准测试工具 - 提供完整的性能评估和基准测试功能
"""

import sys
import time
import json
import psutil
import gc
import logging
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError as e:
    print(f"警告: 无法导入必要的依赖: {e}")
    print("请运行: pip install -r requirements.txt")

try:
    from src.config import ModelConfig, TrainingConfig, DataConfig, LoRAConfig
    from src.data_structures import TaskSample
    from src.utils import setup_logging, get_memory_info, clear_cache
except ImportError as e:
    print(f"警告: 无法导入项目模块: {e}")
    # 创建模拟类以便基准测试可以运行
    class ModelConfig:
        def __init__(self, model_name="test", use_lora=True):
            self.model_name = model_name
            self.use_lora = use_lora
    
    class TrainingConfig:
        def __init__(self, num_epochs=5, batch_size=4):
            self.num_epochs = num_epochs
            self.batch_size = batch_size
    
    class DataConfig:
        def __init__(self):
            pass
    
    class LoRAConfig:
        def __init__(self, r=16):
            self.r = r
    
    class TaskSample:
        def __init__(self, task_type, image_path, prefix, suffix, weight=1.0, metadata=None):
            self.task_type = task_type
            self.image_path = image_path
            self.prefix = prefix
            self.suffix = suffix
            self.weight = weight
            self.metadata = metadata or {}
        
        def to_dict(self):
            return {
                'task_type': self.task_type,
                'image_path': self.image_path,
                'prefix': self.prefix,
                'suffix': self.suffix,
                'weight': self.weight,
                'metadata': self.metadata
            }
        
        @classmethod
        def from_dict(cls, data):
            return cls(**data)
    
    def setup_logging(level=logging.INFO, log_file=None):
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def get_memory_info():
        return {'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024}
    
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """基准测试结果数据结构"""
    test_name: str
    duration: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    gpu_usage: Optional[Dict[str, float]] = None
    throughput: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    status: str = "completed"
    error_message: Optional[str] = None

@contextmanager
def performance_monitor():
    """性能监控上下文管理器"""
    # 记录开始状态
    start_time = time.time()
    start_memory = psutil.virtual_memory()
    start_cpu_percent = psutil.cpu_percent(interval=None)
    
    # GPU监控（如果可用）
    gpu_info = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_info['start_memory'] = torch.cuda.memory_allocated()
        gpu_info['start_reserved'] = torch.cuda.memory_reserved()
    
    try:
        yield
    finally:
        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.virtual_memory()
        end_cpu_percent = psutil.cpu_percent(interval=None)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_info['end_memory'] = torch.cuda.memory_allocated()
            gpu_info['end_reserved'] = torch.cuda.memory_reserved()
        
        # 计算性能指标
        duration = end_time - start_time
        memory_diff = end_memory.used - start_memory.used
        cpu_usage = (start_cpu_percent + end_cpu_percent) / 2
        
        logger.debug(
            f"性能监控 - 耗时: {duration:.3f}s, "
            f"内存变化: {memory_diff/1024/1024:.2f}MB, "
            f"CPU使用率: {cpu_usage:.1f}%"
        )

class BenchmarkTools:
    """性能基准测试工具
    
    提供各种性能测试和基准测试功能
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """初始化基准测试工具
        
        Args:
            output_dir: 基准测试结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_results: List[BenchmarkResult] = []
        
        # 设置日志
        setup_logging(
            level=logging.INFO,
            log_file=self.output_dir / "benchmark.log"
        )
        
        logger.info("性能基准测试工具初始化完成")
    
    def _measure_performance(self, func: Callable, *args, **kwargs) -> Tuple[Any, BenchmarkResult]:
        """测量函数性能
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数返回值和性能测试结果
        """
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory()
        start_cpu_percent = psutil.cpu_percent(interval=None)
        
        gpu_info = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_info['start_memory'] = torch.cuda.memory_allocated()
            gpu_info['start_reserved'] = torch.cuda.memory_reserved()
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory()
            end_cpu_percent = psutil.cpu_percent(interval=None)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_info['end_memory'] = torch.cuda.memory_allocated()
                gpu_info['end_reserved'] = torch.cuda.memory_reserved()
            
            # 计算性能指标
            duration = end_time - start_time
            memory_usage = {
                'start_mb': start_memory.used / 1024 / 1024,
                'end_mb': end_memory.used / 1024 / 1024,
                'diff_mb': (end_memory.used - start_memory.used) / 1024 / 1024,
                'peak_mb': end_memory.used / 1024 / 1024  # 简化的峰值内存
            }
            
            cpu_usage = (start_cpu_percent + end_cpu_percent) / 2
            
            gpu_usage = None
            if torch.cuda.is_available() and gpu_info:
                gpu_usage = {
                    'start_mb': gpu_info['start_memory'] / 1024 / 1024,
                    'end_mb': gpu_info['end_memory'] / 1024 / 1024,
                    'diff_mb': (gpu_info['end_memory'] - gpu_info['start_memory']) / 1024 / 1024,
                    'reserved_mb': gpu_info['end_reserved'] / 1024 / 1024
                }
            
            benchmark_result = BenchmarkResult(
                test_name=func.__name__,
                duration=duration,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                status="completed"
            )
            
            return result, benchmark_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            benchmark_result = BenchmarkResult(
                test_name=func.__name__,
                duration=duration,
                memory_usage={'error': True},
                cpu_usage=0.0,
                status="error",
                error_message=str(e)
            )
            
            return None, benchmark_result
    
    def benchmark_config_creation(self) -> BenchmarkResult:
        """基准测试配置创建性能"""
        logger.info("基准测试配置创建性能...")
        
        def create_configs():
            """创建基准测试配置"""
            
            configs = []
            for i in range(1000):  # 创建1000个配置对象
                model_config = ModelConfig(
                    model_name=f"test-model-{i}",
                    use_lora=True
                )
                training_config = TrainingConfig(
                    num_epochs=5,
                    batch_size=4
                )
                data_config = DataConfig()
                lora_config = LoRAConfig(r=16)
                
                configs.append((model_config, training_config, data_config, lora_config))
            
            return configs
        
        result, benchmark_result = self._measure_performance(create_configs)
        
        if result is not None:
            benchmark_result.throughput = 1000 / benchmark_result.duration  # 配置/秒
            benchmark_result.additional_metrics = {
                'configs_created': len(result),
                'configs_per_second': benchmark_result.throughput
            }
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_data_processing(self) -> BenchmarkResult:
        """基准测试数据处理性能"""
        logger.info("基准测试数据处理性能...")
        
        def process_data():
            """处理数据"""
            
            samples = []
            for i in range(10000):  # 创建10000个样本
                sample = TaskSample(
                    task_type="CAPTION",
                    image_path=f"test_image_{i}.jpg",
                    prefix="<CAPTION>",
                    suffix=f"Test caption {i}",
                    weight=1.0,
                    metadata={"index": i, "source": "benchmark"}
                )
                
                # 测试字典转换
                sample_dict = sample.to_dict()
                sample_from_dict = TaskSample.from_dict(sample_dict)
                
                samples.append(sample_from_dict)
            
            return samples
        
        result, benchmark_result = self._measure_performance(process_data)
        
        if result is not None:
            benchmark_result.throughput = len(result) / benchmark_result.duration  # 样本/秒
            benchmark_result.additional_metrics = {
                'samples_processed': len(result),
                'samples_per_second': benchmark_result.throughput
            }
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_tensor_operations(self) -> BenchmarkResult:
        """基准测试张量操作性能"""
        logger.info("基准测试张量操作性能...")
        
        def tensor_operations():
            """张量操作基准测试"""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 创建测试张量
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            results = []
            for i in range(100):  # 执行100次操作
                # 矩阵乘法
                z = torch.matmul(x, y)
                
                # 激活函数
                z = torch.relu(z)
                
                # 归一化
                z = torch.nn.functional.normalize(z, dim=1)
                
                # 求和
                result = torch.sum(z)
                results.append(result.item())
            
            return results
        
        result, benchmark_result = self._measure_performance(tensor_operations)
        
        if result is not None:
            benchmark_result.throughput = len(result) / benchmark_result.duration  # 操作/秒
            benchmark_result.additional_metrics = {
                'operations_completed': len(result),
                'operations_per_second': benchmark_result.throughput,
                'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            }
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_image_processing(self) -> BenchmarkResult:
        """基准测试图像处理性能"""
        logger.info("基准测试图像处理性能...")
        
        def image_processing():
            """图像处理基准测试"""
            # 创建测试图像
            images = []
            for i in range(100):  # 处理100张图像
                # 创建随机图像
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                # 图像变换
                img_resized = img.resize((256, 256))
                img_cropped = img_resized.crop((16, 16, 240, 240))
                
                # 转换为张量
                img_array = np.array(img_cropped)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                
                images.append(img_tensor)
            
            # 批处理
            batch = torch.stack(images)
            
            return batch
        
        result, benchmark_result = self._measure_performance(image_processing)
        
        if result is not None:
            benchmark_result.throughput = result.shape[0] / benchmark_result.duration  # 图像/秒
            benchmark_result.additional_metrics = {
                'images_processed': result.shape[0],
                'images_per_second': benchmark_result.throughput,
                'batch_shape': list(result.shape)
            }
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_memory_management(self) -> BenchmarkResult:
        """基准测试内存管理性能"""
        logger.info("基准测试内存管理性能...")
        
        def memory_operations():
            """内存操作基准测试"""
            
            memory_stats = []
            
            # 分配大量内存
            large_tensors = []
            for i in range(50):
                tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)
                
                # 记录内存状态
                memory_info = get_memory_info()
                memory_stats.append(memory_info)
                
                # 定期清理
                if i % 10 == 0:
                    clear_cache()
            
            # 最终清理
            del large_tensors
            clear_cache()
            
            return memory_stats
        
        result, benchmark_result = self._measure_performance(memory_operations)
        
        if result is not None:
            benchmark_result.additional_metrics = {
                'memory_snapshots': len(result),
                'final_memory_info': result[-1] if result else None
            }
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """运行所有基准测试
        
        Returns:
            完整的基准测试结果
        """
        logger.info("开始运行完整基准测试套件")
        start_time = time.time()
        
        # 运行各项基准测试
        benchmarks = [
            self.benchmark_config_creation,
            self.benchmark_data_processing,
            self.benchmark_tensor_operations,
            self.benchmark_image_processing,
            self.benchmark_memory_management
        ]
        
        for benchmark_func in benchmarks:
            try:
                result = benchmark_func()
                logger.info(f"{result.test_name}: {result.status} ({result.duration:.3f}s)")
            except Exception as e:
                logger.error(f"基准测试 {benchmark_func.__name__} 执行失败: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 生成摘要
        summary = self._generate_benchmark_summary(total_duration)
        
        # 保存结果
        self._save_benchmark_results(summary)
        
        logger.info(f"完整基准测试套件执行完成，总耗时: {total_duration:.2f}秒")
        return summary
    
    def _generate_benchmark_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成基准测试摘要"""
        completed = sum(1 for r in self.benchmark_results if r.status == "completed")
        errors = sum(1 for r in self.benchmark_results if r.status == "error")
        
        # 计算平均性能指标
        avg_duration = np.mean([r.duration for r in self.benchmark_results if r.status == "completed"])
        avg_memory_usage = np.mean([r.memory_usage.get('diff_mb', 0) for r in self.benchmark_results if r.status == "completed"])
        avg_cpu_usage = np.mean([r.cpu_usage for r in self.benchmark_results if r.status == "completed"])
        
        return {
            "total_duration": total_duration,
            "summary": {
                "total_benchmarks": len(self.benchmark_results),
                "completed": completed,
                "errors": errors
            },
            "average_metrics": {
                "duration": avg_duration,
                "memory_usage_mb": avg_memory_usage,
                "cpu_usage_percent": avg_cpu_usage
            },
            "individual_results": [{
                "test_name": r.test_name,
                "duration": r.duration,
                "status": r.status,
                "throughput": r.throughput,
                "memory_diff_mb": r.memory_usage.get('diff_mb', 0) if isinstance(r.memory_usage, dict) else 0
            } for r in self.benchmark_results],
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
    
    def _save_benchmark_results(self, summary: Dict[str, Any]) -> None:
        """保存基准测试结果"""
        # 保存摘要
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细结果
        detailed_results = {
            "summary": summary,
            "detailed_results": [{
                "test_name": r.test_name,
                "duration": r.duration,
                "memory_usage": r.memory_usage,
                "cpu_usage": r.cpu_usage,
                "gpu_usage": r.gpu_usage,
                "throughput": r.throughput,
                "additional_metrics": r.additional_metrics,
                "status": r.status,
                "error_message": r.error_message
            } for r in self.benchmark_results]
        }
        
        detailed_file = self.output_dir / "benchmark_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"基准测试结果已保存到: {summary_file}")

def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlorenceForge性能基准测试工具")
    parser.add_argument("--output-dir", default="./benchmark_results", help="基准测试结果输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # 创建基准测试工具
    benchmark = BenchmarkTools(output_dir=args.output_dir)
    
    # 运行基准测试
    results = benchmark.run_all_benchmarks()
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("基准测试结果摘要")
    print("="*50)
    print(f"总计: {results['summary']['total_benchmarks']}")
    print(f"完成: {results['summary']['completed']}")
    print(f"错误: {results['summary']['errors']}")
    print(f"总耗时: {results['total_duration']:.2f}秒")
    print(f"平均耗时: {results['average_metrics']['duration']:.3f}秒")
    print(f"平均内存使用: {results['average_metrics']['memory_usage_mb']:.2f}MB")
    print(f"平均CPU使用率: {results['average_metrics']['cpu_usage_percent']:.1f}%")
    
    # 显示详细结果
    if args.verbose:
        print("\n详细结果:")
        for result in results['individual_results']:
            status_symbol = "✓" if result['status'] == "completed" else "✗"
            throughput_str = f" ({result['throughput']:.1f}/s)" if result['throughput'] else ""
            print(f"  {status_symbol} {result['test_name']}: {result['duration']:.3f}s{throughput_str}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()