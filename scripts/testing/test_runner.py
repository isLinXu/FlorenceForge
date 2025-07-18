#!/usr/bin/env python3
"""
测试运行器 - 提供完整的测试执行和管理功能
"""

import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(level=logging.INFO, log_file=None):
    """设置日志配置"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果数据结构"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class TestRunner:
    """测试运行器
    
    提供统一的测试执行接口，支持多种测试框架
    """
    
    def __init__(self, output_dir: str = "./test_results"):
        """初始化测试运行器
        
        Args:
            output_dir: 测试结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # 设置日志
        setup_logging(
            level=logging.INFO,
            log_file=self.output_dir / "test_runner.log"
        )
        
        logger.info("测试运行器初始化完成")
    
    def run_unit_tests(self, test_dir: str = "tests/unit") -> Dict[str, Any]:
        """运行单元测试
        
        Args:
            test_dir: 单元测试目录
            
        Returns:
            测试结果摘要
        """
        logger.info("开始运行单元测试")
        start_time = time.time()
        
        test_dir_path = project_root / test_dir
        if not test_dir_path.exists():
            logger.warning(f"单元测试目录不存在: {test_dir_path}")
            return {"status": "skipped", "reason": "测试目录不存在"}
        
        try:
            # 使用pytest运行测试
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_dir_path),
                "-v", 
                "--tb=short",
                f"--junitxml={self.output_dir}/unit_tests.xml"
            ], capture_output=True, text=True, cwd=project_root)
            
            duration = time.time() - start_time
            
            test_result = TestResult(
                test_name="unit_tests",
                status="passed" if result.returncode == 0 else "failed",
                duration=duration,
                error_message=result.stderr if result.returncode != 0 else None,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            )
            
            self.test_results.append(test_result)
            
            logger.info(f"单元测试完成，耗时: {duration:.2f}秒")
            return {
                "status": test_result.status,
                "duration": duration,
                "details": test_result.details
            }
            
        except Exception as e:
            logger.error(f"单元测试执行失败: {e}")
            test_result = TestResult(
                test_name="unit_tests",
                status="error",
                duration=time.time() - start_time,
                error_message=str(e)
            )
            self.test_results.append(test_result)
            return {"status": "error", "error": str(e)}
    
    def run_integration_tests(self, test_dir: str = "tests/integration") -> Dict[str, Any]:
        """运行集成测试
        
        Args:
            test_dir: 集成测试目录
            
        Returns:
            测试结果摘要
        """
        logger.info("开始运行集成测试")
        start_time = time.time()
        
        test_dir_path = project_root / test_dir
        if not test_dir_path.exists():
            logger.warning(f"集成测试目录不存在: {test_dir_path}")
            return {"status": "skipped", "reason": "测试目录不存在"}
        
        try:
            # 运行集成测试
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_dir_path),
                "-v", 
                "--tb=short",
                "-s",  # 不捕获输出，便于调试
                f"--junitxml={self.output_dir}/integration_tests.xml"
            ], capture_output=True, text=True, cwd=project_root)
            
            duration = time.time() - start_time
            
            test_result = TestResult(
                test_name="integration_tests",
                status="passed" if result.returncode == 0 else "failed",
                duration=duration,
                error_message=result.stderr if result.returncode != 0 else None,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            )
            
            self.test_results.append(test_result)
            
            logger.info(f"集成测试完成，耗时: {duration:.2f}秒")
            return {
                "status": test_result.status,
                "duration": duration,
                "details": test_result.details
            }
            
        except Exception as e:
            logger.error(f"集成测试执行失败: {e}")
            test_result = TestResult(
                test_name="integration_tests",
                status="error",
                duration=time.time() - start_time,
                error_message=str(e)
            )
            self.test_results.append(test_result)
            return {"status": "error", "error": str(e)}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试
        
        Returns:
            性能测试结果
        """
        logger.info("开始运行性能测试")
        start_time = time.time()
        
        try:
            from .benchmark_tools import BenchmarkTools
            
            benchmark = BenchmarkTools()
            results = benchmark.run_all_benchmarks()
            
            duration = time.time() - start_time
            
            test_result = TestResult(
                test_name="performance_tests",
                status="passed",
                duration=duration,
                details=results
            )
            
            self.test_results.append(test_result)
            
            logger.info(f"性能测试完成，耗时: {duration:.2f}秒")
            return {
                "status": "passed",
                "duration": duration,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"性能测试执行失败: {e}")
            test_result = TestResult(
                test_name="performance_tests",
                status="error",
                duration=time.time() - start_time,
                error_message=str(e)
            )
            self.test_results.append(test_result)
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试
        
        Returns:
            完整的测试结果摘要
        """
        logger.info("开始运行完整测试套件")
        self.start_time = time.time()
        
        # 运行各类测试
        unit_results = self.run_unit_tests()
        integration_results = self.run_integration_tests()
        performance_results = self.run_performance_tests()
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # 汇总结果
        summary = {
            "total_duration": total_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "unit_tests": unit_results,
            "integration_tests": integration_results,
            "performance_tests": performance_results,
            "overall_status": self._get_overall_status()
        }
        
        # 保存结果
        self._save_results(summary)
        
        logger.info(f"完整测试套件执行完成，总耗时: {total_duration:.2f}秒")
        return summary
    
    def _get_overall_status(self) -> str:
        """获取整体测试状态"""
        if not self.test_results:
            return "no_tests"
        
        statuses = [result.status for result in self.test_results]
        
        if "error" in statuses:
            return "error"
        elif "failed" in statuses:
            return "failed"
        elif all(status in ["passed", "skipped"] for status in statuses):
            return "passed"
        else:
            return "unknown"
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """保存测试结果"""
        # 保存JSON格式结果
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细的测试结果
        detailed_results = {
            "summary": summary,
            "detailed_results": [result.__dict__ for result in self.test_results]
        }
        
        detailed_file = self.output_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试结果已保存到: {results_file}")

def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlorenceForge测试运行器")
    parser.add_argument("--test-type", choices=["unit", "integration", "performance", "all"], 
                       default="all", help="要运行的测试类型")
    parser.add_argument("--output-dir", default="./test_results", help="测试结果输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # 创建测试运行器
    runner = TestRunner(output_dir=args.output_dir)
    
    # 运行指定的测试
    if args.test_type == "unit":
        results = runner.run_unit_tests()
    elif args.test_type == "integration":
        results = runner.run_integration_tests()
    elif args.test_type == "performance":
        results = runner.run_performance_tests()
    else:  # all
        results = runner.run_all_tests()
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("测试结果摘要")
    print("="*50)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    # 根据测试结果设置退出码
    if isinstance(results, dict):
        overall_status = results.get("overall_status", results.get("status", "unknown"))
        if overall_status in ["failed", "error"]:
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()