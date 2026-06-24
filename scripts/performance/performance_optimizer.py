#!/usr/bin/env python3
"""
性能优化器 - 分析和优化FlorenceForge框架性能
"""

import sys
import time
import json
import gc
import psutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError as e:
    print(f"导入依赖失败: {e}")
    sys.exit(1)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    execution_time: float
    throughput: Optional[float]
    peak_memory: float
    
@dataclass
class OptimizationResult:
    """优化结果数据类"""
    test_name: str
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_ratio: float
    recommendations: List[str]

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        """初始化性能分析器"""
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    @contextmanager
    def profile(self):
        """性能分析上下文管理器"""
        # 记录开始状态
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # GPU信息（如果可用）
        gpu_memory_start = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
        
        peak_memory = start_memory
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.time()
            end_cpu = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            # GPU信息（如果可用）
            gpu_usage_end = None
            gpu_memory_end = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024
            
            # 计算峰值内存
            peak_memory = max(peak_memory, end_memory)
            
            # 存储指标
            self.last_metrics = PerformanceMetrics(
                cpu_usage=(start_cpu + end_cpu) / 2,
                memory_usage=end_memory - start_memory,
                gpu_usage=gpu_usage_end,
                gpu_memory=gpu_memory_end - gpu_memory_start if gpu_memory_start and gpu_memory_end else None,
                execution_time=end_time - start_time,
                throughput=None,
                peak_memory=peak_memory
            )
    
    def get_last_metrics(self) -> PerformanceMetrics:
        """获取最后一次分析的指标"""
        return getattr(self, 'last_metrics', None)

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, output_dir: str = "./performance_results"):
        """TODO: Add documentation for __init__"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiler = PerformanceProfiler()
        self.optimization_results = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "performance_optimizer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("性能优化器初始化完成")
    
    def benchmark_tensor_operations(self) -> Dict[str, Any]:
        """基准测试张量操作"""
        self.logger.info("开始张量操作基准测试...")
        
        results = {}
        
        # 测试不同大小的矩阵乘法
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            self.logger.info(f"测试矩阵大小: {size}x{size}")
            
            # CPU测试
            with self.profiler.profile():
                a = torch.randn(size, size)
                b = torch.randn(size, size)
                c = torch.mm(a, b)
                del a, b, c
            
            cpu_metrics = self.profiler.get_last_metrics()
            
            # GPU测试（如果可用）
            gpu_metrics = None
            if torch.cuda.is_available():
                with self.profiler.profile():
                    a = torch.randn(size, size, device='cuda')
                    b = torch.randn(size, size, device='cuda')
                    c = torch.mm(a, b)
                    torch.cuda.synchronize()
                    del a, b, c
                
                gpu_metrics = self.profiler.get_last_metrics()
            
            results[f"matrix_{size}"] = {
                "cpu": cpu_metrics,
                "gpu": gpu_metrics
            }
        
        return results
    
    def benchmark_image_processing(self) -> Dict[str, Any]:
        """基准测试图像处理"""
        self.logger.info("开始图像处理基准测试...")
        
        results = {}
        
        # 测试不同大小的图像处理
        sizes = [(224, 224), (512, 512), (1024, 1024)]
        batch_sizes = [1, 4, 8, 16]
        
        for img_size in sizes:
            for batch_size in batch_sizes:
                test_name = f"image_{img_size[0]}x{img_size[1]}_batch_{batch_size}"
                self.logger.info(f"测试: {test_name}")
                
                # 创建虚拟图像数据
                with self.profiler.profile():
                    # 创建图像张量
                    images = torch.randn(batch_size, 3, img_size[0], img_size[1])
                    
                    # 模拟图像预处理
                    normalized = (images - images.mean()) / images.std()
                    resized = torch.nn.functional.interpolate(normalized, size=(256, 256), mode='bilinear')
                    
                    # 模拟数据增强
                    flipped = torch.flip(resized, dims=[3])
                    
                    del images, normalized, resized, flipped
                
                metrics = self.profiler.get_last_metrics()
                results[test_name] = metrics
        
        return results
    
    def benchmark_data_loading(self) -> Dict[str, Any]:
        """基准测试数据加载"""
        self.logger.info("开始数据加载基准测试...")
        
        results = {}
        
        # 测试不同的数据加载策略
        strategies = {
            "sequential": self._test_sequential_loading,
            "parallel": self._test_parallel_loading,
            "cached": self._test_cached_loading
        }
        
        for strategy_name, strategy_func in strategies.items():
            self.logger.info(f"测试策略: {strategy_name}")
            
            with self.profiler.profile():
                strategy_func()
            
            metrics = self.profiler.get_last_metrics()
            results[strategy_name] = metrics
        
        return results
    
    def _test_sequential_loading(self):
        """测试顺序加载"""
        # 模拟顺序加载100个样本
        for i in range(100):
            # 创建虚拟图像
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            # 转换为张量
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            del image, tensor
    
    def _test_parallel_loading(self):
        """测试并行加载"""
        
        def load_sample(idx):
            """加载单个样本数据"""
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return tensor
        
        # 使用线程池并行加载
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_sample, i) for i in range(100)]
            results = [future.result() for future in futures]
        
        del results
    
    def _test_cached_loading(self):
        """测试缓存加载策略"""
        # 模拟缓存策略
        cache = {}
        
        for i in range(100):
            cache_key = i % 10  # 模拟10个不同的样本
            
            if cache_key not in cache:
                # 创建并缓存
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                cache[cache_key] = tensor
            else:
                # 从缓存获取
                tensor = cache[cache_key]
        
        del cache
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """基准测试内存使用"""
        self.logger.info("开始内存使用基准测试...")
        
        results = {}
        
        # 测试内存分配和释放
        allocation_sizes = [10, 50, 100, 500]  # MB
        
        for size_mb in allocation_sizes:
            test_name = f"allocation_{size_mb}MB"
            self.logger.info(f"测试: {test_name}")
            
            with self.profiler.profile():
                # 分配内存
                size_elements = size_mb * 1024 * 1024 // 4  # float32
                tensor = torch.randn(size_elements)
                
                # 模拟使用
                result = tensor.sum()
                
                # 释放内存
                del tensor, result
                gc.collect()
            
            metrics = self.profiler.get_last_metrics()
            results[test_name] = metrics
        
        return results
    
    def optimize_torch_settings(self) -> Dict[str, Any]:
        """优化PyTorch设置"""
        self.logger.info("开始PyTorch设置优化...")
        
        optimizations = {}
        
        # 测试不同的线程数设置
        original_threads = torch.get_num_threads()
        thread_counts = [1, 2, 4, 8]
        
        for num_threads in thread_counts:
            if num_threads <= psutil.cpu_count():
                torch.set_num_threads(num_threads)
                
                with self.profiler.profile():
                    # 执行计算密集型任务
                    a = torch.randn(1000, 1000)
                    b = torch.randn(1000, 1000)
                    c = torch.mm(a, b)
                    del a, b, c
                
                metrics = self.profiler.get_last_metrics()
                optimizations[f"threads_{num_threads}"] = metrics
        
        # 恢复原始设置
        torch.set_num_threads(original_threads)
        
        # 测试内存管理策略
        if torch.cuda.is_available():
            # 测试不同的CUDA内存管理
            strategies = {
                "default": lambda: None,
                "empty_cache": lambda: torch.cuda.empty_cache(),
                "memory_fraction": lambda: torch.cuda.set_per_process_memory_fraction(0.8)
            }
            
            for strategy_name, strategy_func in strategies.items():
                strategy_func()
                
                with self.profiler.profile():
                    if torch.cuda.is_available():
                        a = torch.randn(1000, 1000, device='cuda')
                        b = torch.randn(1000, 1000, device='cuda')
                        c = torch.mm(a, b)
                        torch.cuda.synchronize()
                        del a, b, c
                
                metrics = self.profiler.get_last_metrics()
                optimizations[f"cuda_{strategy_name}"] = metrics
        
        return optimizations
    
    def analyze_bottlenecks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """分析性能瓶颈"""
        self.logger.info("分析性能瓶颈...")
        
        bottlenecks = []
        
        # 分析内存使用
        high_memory_tests = []
        for test_name, result in benchmark_results.items():
            if isinstance(result, dict) and 'memory_usage' in result:
                if result['memory_usage'] > 100:  # MB
                    high_memory_tests.append((test_name, result['memory_usage']))
        
        if high_memory_tests:
            bottlenecks.append(f"高内存使用测试: {', '.join([f'{name}({mem:.1f}MB)' for name, mem in high_memory_tests])}")
        
        # 分析执行时间
        slow_tests = []
        for test_name, result in benchmark_results.items():
            if isinstance(result, dict) and 'execution_time' in result:
                if result['execution_time'] > 1.0:  # 秒
                    slow_tests.append((test_name, result['execution_time']))
        
        if slow_tests:
            bottlenecks.append(f"慢速测试: {', '.join([f'{name}({time:.2f}s)' for name, time in slow_tests])}")
        
        # 分析CPU使用
        high_cpu_tests = []
        for test_name, result in benchmark_results.items():
            if isinstance(result, dict) and 'cpu_usage' in result:
                if result['cpu_usage'] > 80:  # 百分比
                    high_cpu_tests.append((test_name, result['cpu_usage']))
        
        if high_cpu_tests:
            bottlenecks.append(f"高CPU使用测试: {', '.join([f'{name}({cpu:.1f}%)' for name, cpu in high_cpu_tests])}")
        
        return bottlenecks
    
    def generate_recommendations(
        self,
        benchmark_results: Dict[str,
        Any],
        bottlenecks: List[str]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于瓶颈生成建议
        for bottleneck in bottlenecks:
            if "高内存使用" in bottleneck:
                recommendations.extend([
                    "考虑使用梯度检查点减少内存使用",
                    "实施批次大小自适应调整",
                    "使用混合精度训练减少内存占用"
                ])
            
            if "慢速测试" in bottleneck:
                recommendations.extend([
                    "考虑使用数据并行加速计算",
                    "优化数据加载管道",
                    "使用编译优化（如TorchScript）"
                ])
            
            if "高CPU使用" in bottleneck:
                recommendations.extend([
                    "调整PyTorch线程数设置",
                    "使用GPU加速计算密集型操作",
                    "优化数据预处理流程"
                ])
        
        # 通用优化建议
        if torch.cuda.is_available():
            recommendations.extend([
                "启用CUDA加速以提高性能",
                "使用适当的CUDA内存管理策略"
            ])
        
        recommendations.extend([
            "定期清理未使用的张量和变量",
            "使用适当的批次大小平衡内存和性能",
            "考虑使用数据加载器的多进程功能"
        ])
        
        # 去重
        return list(set(recommendations))
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """运行完整的性能优化分析"""
        self.logger.info("开始完整的性能优化分析...")
        start_time = time.time()
        
        # 运行各种基准测试
        benchmark_results = {}
        
        try:
            benchmark_results["tensor_operations"] = self.benchmark_tensor_operations()
        except Exception as e:
            self.logger.error(f"张量操作基准测试失败: {e}")
            benchmark_results["tensor_operations"] = {"error": str(e)}
        
        try:
            benchmark_results["image_processing"] = self.benchmark_image_processing()
        except Exception as e:
            self.logger.error(f"图像处理基准测试失败: {e}")
            benchmark_results["image_processing"] = {"error": str(e)}
        
        try:
            benchmark_results["data_loading"] = self.benchmark_data_loading()
        except Exception as e:
            self.logger.error(f"数据加载基准测试失败: {e}")
            benchmark_results["data_loading"] = {"error": str(e)}
        
        try:
            benchmark_results["memory_usage"] = self.benchmark_memory_usage()
        except Exception as e:
            self.logger.error(f"内存使用基准测试失败: {e}")
            benchmark_results["memory_usage"] = {"error": str(e)}
        
        try:
            benchmark_results["torch_optimizations"] = self.optimize_torch_settings()
        except Exception as e:
            self.logger.error(f"PyTorch优化失败: {e}")
            benchmark_results["torch_optimizations"] = {"error": str(e)}
        
        # 分析结果
        bottlenecks = self.analyze_bottlenecks(benchmark_results)
        recommendations = self.generate_recommendations(benchmark_results, bottlenecks)
        
        total_duration = time.time() - start_time
        
        # 生成报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "system_info": self._get_system_info(),
            "benchmark_results": benchmark_results,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "summary": {
                "total_tests": len(benchmark_results),
                "successful_tests": len([r for r in benchmark_results.values() if "error" not in r]),
                "failed_tests": len([r for r in benchmark_results.values() if "error" in r]),
                "bottlenecks_found": len(bottlenecks),
                "recommendations_generated": len(recommendations)
            }
        }
        
        # 保存报告
        self._save_report(report)
        
        self.logger.info(f"性能优化分析完成，耗时: {total_duration:.2f}秒")
        self.logger.info(f"发现 {len(bottlenecks)} 个瓶颈，生成 {len(recommendations)} 条建议")
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            info = {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                info.update({
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                })
            
            return info
        except Exception as e:
            return {"error": f"无法获取系统信息: {str(e)}"}
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """保存性能报告"""
        # 保存JSON报告
        json_file = self.output_dir / "performance_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可读的文本报告
        text_file = self.output_dir / "performance_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("FlorenceForge 性能优化报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"执行时间: {report['timestamp']}\n")
            f.write(f"总耗时: {report['total_duration']:.2f}秒\n\n")
            
            # 系统信息
            f.write("系统信息:\n")
            f.write("-" * 20 + "\n")
            for key, value in report['system_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 测试摘要
            f.write("测试摘要:\n")
            f.write("-" * 20 + "\n")
            summary = report['summary']
            f.write(f"总测试数: {summary['total_tests']}\n")
            f.write(f"成功: {summary['successful_tests']}\n")
            f.write(f"失败: {summary['failed_tests']}\n")
            f.write(f"发现瓶颈: {summary['bottlenecks_found']}\n")
            f.write(f"生成建议: {summary['recommendations_generated']}\n\n")
            
            # 瓶颈分析
            if report['bottlenecks']:
                f.write("性能瓶颈:\n")
                f.write("-" * 20 + "\n")
                for i, bottleneck in enumerate(report['bottlenecks'], 1):
                    f.write(f"{i}. {bottleneck}\n")
                f.write("\n")
            
            # 优化建议
            f.write("优化建议:\n")
            f.write("-" * 20 + "\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
        
        self.logger.info(f"性能报告已保存到: {json_file}")
        self.logger.info(f"可读报告已保存到: {text_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FlorenceForge性能优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
这个工具会分析FlorenceForge框架的性能并提供优化建议，包括：
- 张量操作性能测试
- 图像处理性能测试
- 数据加载性能测试
- 内存使用分析
- PyTorch设置优化
- 瓶颈识别和优化建议

示例用法:
  python performance_optimizer.py                    # 运行完整优化分析
  python performance_optimizer.py --output-dir ./perf_results  # 指定输出目录
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./performance_results", 
        help="结果输出目录"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行性能优化
    optimizer = PerformanceOptimizer(output_dir=args.output_dir)
    report = optimizer.run_full_optimization()
    
    # 输出简要结果
    print("\n" + "="*60)
    print("FlorenceForge 性能优化结果")
    print("="*60)
    
    summary = report['summary']
    print(f"总测试数: {summary['total_tests']}")
    print(f"成功测试: {summary['successful_tests']}")
    print(f"失败测试: {summary['failed_tests']}")
    print(f"发现瓶颈: {summary['bottlenecks_found']}")
    print(f"生成建议: {summary['recommendations_generated']}")
    print(f"总耗时: {report['total_duration']:.2f}秒")
    
    if report['bottlenecks']:
        print("\n主要瓶颈:")
        for i, bottleneck in enumerate(report['bottlenecks'][:3], 1):
            print(f"  {i}. {bottleneck}")
    
    print("\n主要建议:")
    for i, recommendation in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n详细报告已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()