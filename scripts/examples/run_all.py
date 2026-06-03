#!/usr/bin/env python3
"""
统一脚本入口 - 提供完整的测试、验证和示例运行功能
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from florence_forge.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

class FlorenceForgeRunner:
    """FlorenceForge统一运行器
    
    提供完整的测试、验证、基准测试和示例运行功能
    """
    
    def __init__(self, output_dir: str = "./florence_forge_results"):
        """初始化统一运行器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.test_dir = self.output_dir / "tests"
        self.validation_dir = self.output_dir / "validation"
        self.benchmark_dir = self.output_dir / "benchmarks"
        self.example_dir = self.output_dir / "examples"
        
        for dir_path in [self.test_dir, self.validation_dir, self.benchmark_dir, self.example_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging(
            level=logging.INFO,
            log_file=self.output_dir / "florence_forge_runner.log"
        )
        
        logger.info("FlorenceForge统一运行器初始化完成")
    
    def run_validation(self) -> Dict[str, Any]:
        """运行验证套件"""
        logger.info("开始运行验证套件...")
        start_time = time.time()
        
        try:
            from scripts.testing.validation_suite import ValidationSuite

            validator = ValidationSuite(output_dir=str(self.validation_dir))
            results = validator.run_all_validations()
            
            duration = time.time() - start_time
            logger.info(f"验证套件完成，耗时: {duration:.2f}秒")
            
            return {
                "status": "completed",
                "duration": duration,
                "results": results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"验证套件执行失败: {e}")
            return {
                "status": "error",
                "duration": duration,
                "error": str(e)
            }
    
    def run_tests(self) -> Dict[str, Any]:
        """运行测试套件"""
        logger.info("开始运行测试套件...")
        start_time = time.time()
        
        try:
            from scripts.testing.quick_test import QuickTester

            test_runner = QuickTester()
            results = test_runner.run_all_tests()
            
            duration = time.time() - start_time
            logger.info(f"测试套件完成，耗时: {duration:.2f}秒")
            
            return {
                "status": "completed",
                "duration": duration,
                "results": results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"测试套件执行失败: {e}")
            return {
                "status": "error",
                "duration": duration,
                "error": str(e)
            }
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """运行基准测试"""
        logger.info("开始运行基准测试...")
        start_time = time.time()
        
        try:
            from scripts.performance.benchmark_tools import BenchmarkTools

            benchmark_tools = BenchmarkTools(output_dir=str(self.benchmark_dir))
            results = benchmark_tools.run_all_benchmarks()
            
            duration = time.time() - start_time
            logger.info(f"基准测试完成，耗时: {duration:.2f}秒")
            
            return {
                "status": "completed",
                "duration": duration,
                "results": results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"基准测试执行失败: {e}")
            return {
                "status": "error",
                "duration": duration,
                "error": str(e)
            }
    
    def run_examples(self) -> Dict[str, Any]:
        """运行示例"""
        logger.info("开始运行示例...")
        start_time = time.time()
        
        try:
            from scripts.examples.example_runner import ExampleRunner

            example_runner = ExampleRunner(output_dir=str(self.example_dir))
            results = example_runner.run_all_examples()
            
            duration = time.time() - start_time
            logger.info(f"示例运行完成，耗时: {duration:.2f}秒")
            
            return {
                "status": "completed",
                "duration": duration,
                "results": results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"示例运行失败: {e}")
            return {
                "status": "error",
                "duration": duration,
                "error": str(e)
            }
    
    def run_all(self, skip_on_error: bool = False) -> Dict[str, Any]:
        """运行所有功能
        
        Args:
            skip_on_error: 是否在出错时跳过后续步骤
            
        Returns:
            完整的运行结果
        """
        logger.info("开始运行完整的FlorenceForge测试和验证套件")
        total_start_time = time.time()
        
        # 运行顺序：验证 -> 示例 -> 测试 -> 基准测试
        components = [
            ("validation", self.run_validation),
            ("examples", self.run_examples),
            ("tests", self.run_tests),
            ("benchmarks", self.run_benchmarks)
        ]
        
        results = {}
        overall_status = "completed"
        
        for component_name, component_func in components:
            logger.info(f"\n{'='*50}")
            logger.info(f"运行 {component_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                result = component_func()
                results[component_name] = result
                
                if result["status"] == "error":
                    overall_status = "partial_failure"
                    if skip_on_error:
                        logger.warning(f"{component_name} 失败，跳过后续组件")
                        break
                
                logger.info(f"{component_name} 完成: {result['status']} ({result['duration']:.2f}s)")
                
            except Exception as e:
                logger.error(f"{component_name} 执行异常: {e}")
                results[component_name] = {
                    "status": "error",
                    "duration": 0.0,
                    "error": str(e)
                }
                overall_status = "partial_failure"
                
                if skip_on_error:
                    logger.warning(f"{component_name} 异常，跳过后续组件")
                    break
        
        total_duration = time.time() - total_start_time
        
        # 生成总结报告
        summary = self._generate_summary(results, total_duration, overall_status)
        
        # 保存总结报告
        self._save_summary(summary)
        
        logger.info(f"\n{'='*50}")
        logger.info("FlorenceForge完整测试套件执行完成")
        logger.info(f"总耗时: {total_duration:.2f}秒")
        logger.info(f"整体状态: {overall_status}")
        logger.info(f"{'='*50}")
        
        return summary
    
    def _generate_summary(
        self,
        results: Dict[str,
        Any],
        total_duration: float,
        overall_status: str
    ) -> Dict[str, Any]:
        """生成总结报告"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "overall_status": overall_status,
            "output_directory": str(self.output_dir),
            "components": {}
        }
        
        # 统计各组件状态
        for component_name, result in results.items():
            summary["components"][component_name] = {
                "status": result["status"],
                "duration": result["duration"],
                "has_error": "error" in result
            }
            
            # 添加组件特定的摘要信息
            if component_name == "validation" and "results" in result:
                validation_results = result["results"]
                summary["components"][component_name]["summary"] = validation_results.get(
                    "summary",
                    {}
                )
            
            elif component_name == "tests" and "results" in result:
                test_results = result["results"]
                summary["components"][component_name]["summary"] = {
                    "overall_status": test_results.get("overall_status", "unknown")
                }
            
            elif component_name == "benchmarks" and "results" in result:
                benchmark_results = result["results"]
                summary["components"][component_name]["summary"] = benchmark_results.get(
                    "summary",
                    {}
                )
            
            elif component_name == "examples" and "results" in result:
                example_results = result["results"]
                summary["components"][component_name]["summary"] = {
                    "total_examples": example_results.get("total_examples", 0),
                    "successful": example_results.get("successful", 0),
                    "failed": example_results.get("failed", 0)
                }
        
        # 添加系统信息
        summary["system_info"] = self._get_system_info()
        
        # 添加建议和下一步
        summary["recommendations"] = self._generate_recommendations(results)
        
        return summary
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import psutil
            import torch
            
            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "pytorch_version": torch.__version__
            }
        except Exception as e:
            return {"error": f"无法获取系统信息: {str(e)}"}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成建议和下一步"""
        recommendations = []
        
        # 检查验证结果
        if "validation" in results:
            validation_result = results["validation"]
            if validation_result["status"] == "error":
                recommendations.append("验证失败，请检查环境配置和依赖安装")
            elif "results" in validation_result:
                validation_summary = validation_result["results"].get("summary", {})
                if validation_summary.get("failed", 0) > 0:
                    recommendations.append("部分验证失败，请查看详细日志并修复相关问题")
        
        # 检查测试结果
        if "tests" in results:
            test_result = results["tests"]
            if test_result["status"] == "error":
                recommendations.append("测试执行失败，请检查测试环境")
        
        # 检查基准测试结果
        if "benchmarks" in results:
            benchmark_result = results["benchmarks"]
            if benchmark_result["status"] == "completed" and "results" in benchmark_result:
                benchmark_summary = benchmark_result["results"].get("summary", {})
                if benchmark_summary.get("errors", 0) > 0:
                    recommendations.append("部分基准测试失败，可能存在性能问题")
        
        # 通用建议
        if not recommendations:
            recommendations.extend([
                "所有测试通过，框架运行正常",
                "可以开始使用FlorenceForge进行模型训练和评估",
                "建议定期运行测试套件以确保代码质量"
            ])
        else:
            recommendations.append("请查看详细日志文件以获取更多信息")
            recommendations.append("修复问题后重新运行测试")
        
        return recommendations
    
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """保存总结报告"""
        summary_file = self.output_dir / "florence_forge_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可读的文本报告
        report_file = self.output_dir / "florence_forge_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FlorenceForge 测试和验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"执行时间: {summary['timestamp']}\n")
            f.write(f"总耗时: {summary['total_duration']:.2f}秒\n")
            f.write(f"整体状态: {summary['overall_status']}\n")
            f.write(f"输出目录: {summary['output_directory']}\n\n")
            
            f.write("组件执行结果:\n")
            f.write("-" * 30 + "\n")
            for component_name, component_info in summary['components'].items():
                status_symbol = "✓" if component_info['status'] == "completed" else "✗"
                f.write(f"{status_symbol} {component_name}: {component_info['status']} ({component_info['duration']:.2f}s)\n")
            
            f.write("\n建议和下一步:\n")
            f.write("-" * 30 + "\n")
            for i, recommendation in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
        
        logger.info(f"总结报告已保存到: {summary_file}")
        logger.info(f"可读报告已保存到: {report_file}")

def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(
        description="FlorenceForge统一测试和验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_all.py --mode all                    # 运行所有功能
  python run_all.py --mode validation             # 只运行验证
  python run_all.py --mode tests                  # 只运行测试
  python run_all.py --mode benchmarks             # 只运行基准测试
  python run_all.py --mode examples               # 只运行示例
  python run_all.py --mode all --skip-on-error    # 出错时跳过后续步骤
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["all", "validation", "tests", "benchmarks", "examples"], 
        default="all", 
        help="运行模式"
    )
    parser.add_argument(
        "--output-dir", 
        default="./florence_forge_results", 
        help="结果输出目录"
    )
    parser.add_argument(
        "--skip-on-error", 
        action="store_true", 
        help="出错时跳过后续步骤"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # 创建统一运行器
    runner = FlorenceForgeRunner(output_dir=args.output_dir)
    
    # 运行指定模式
    if args.mode == "validation":
        results = runner.run_validation()
    elif args.mode == "tests":
        results = runner.run_tests()
    elif args.mode == "benchmarks":
        results = runner.run_benchmarks()
    elif args.mode == "examples":
        results = runner.run_examples()
    else:  # all
        results = runner.run_all(skip_on_error=args.skip_on_error)
    
    # 输出简要结果
    print("\n" + "="*60)
    print("FlorenceForge 执行结果")
    print("="*60)
    
    if args.mode == "all":
        print(f"整体状态: {results['overall_status']}")
        print(f"总耗时: {results['total_duration']:.2f}秒")
        print(f"输出目录: {results['output_directory']}")
        
        print("\n组件状态:")
        for component_name, component_info in results['components'].items():
            status_symbol = "✓" if component_info['status'] == "completed" else "✗"
            print(f"  {status_symbol} {component_name}: {component_info['status']}")
        
        print("\n建议:")
        for i, recommendation in enumerate(results['recommendations'], 1):
            print(f"  {i}. {recommendation}")
    else:
        status_symbol = "✓" if results['status'] == "completed" else "✗"
        print(f"{status_symbol} {args.mode}: {results['status']} ({results['duration']:.2f}s)")
    
    # 根据结果设置退出码
    if args.mode == "all":
        if results['overall_status'] in ["partial_failure", "error"]:
            sys.exit(1)
    else:
        if results['status'] == "error":
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
