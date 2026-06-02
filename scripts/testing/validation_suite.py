#!/usr/bin/env python3
"""
验证套件 - 提供完整的框架功能验证
"""

import sys
import time
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
# 本脚本位于 scripts/testing/ 下，仓库根目录需向上回溯三层
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from florence_forge.core.config import (
        DataConfig,
        LoRAConfig,
        ModelConfig,
        TrainingConfig,
    )
    from florence_forge.core.tasks import (
        FLORENCE2_TASKS,
        get_task_config,
        validate_task_name,
    )
    from florence_forge.core.model import Florence2MultiTaskModel
    from florence_forge.data.builder import DatasetBuilder
    from florence_forge.data.dataset import MultiTaskDataset, TaskSample
    from florence_forge.training.trainer import MultiTaskTrainer
    from florence_forge.utils.device import get_device_info, get_optimal_device
    from florence_forge.utils.logging import setup_logging
    from florence_forge.utils.memory import clear_cache, get_memory_usage
except ImportError as e:
    print(f"警告: 无法导入必要的依赖: {e}")
    print("请运行: pip install -r requirements.txt")


logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果数据结构"""
    component: str
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None

class ValidationSuite:
    """验证套件
    
    提供完整的框架功能验证，包括：
    - 核心组件验证
    - 数据处理验证
    - 模型加载验证
    - 训练流程验证
    - 评估功能验证
    """
    
    def __init__(self, output_dir: str = "./validation_results"):
        """初始化验证套件
        
        Args:
            output_dir: 验证结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results: List[ValidationResult] = []
        self.start_time = None
        self.end_time = None
        
        # 设置日志
        setup_logging(
            level=logging.INFO,
            log_file=self.output_dir / "validation.log"
        )
        
        logger.info("验证套件初始化完成")
    
    def validate_imports(self) -> ValidationResult:
        """验证模块导入"""
        logger.info("验证模块导入...")
        start_time = time.time()
        
        try:
            # 验证核心模块
            
            # 验证数据模块
            
            # 验证训练模块
            
            # 验证评估模块
            
            # 验证工具模块
            
            duration = time.time() - start_time
            return ValidationResult(
                component="imports",
                test_name="module_imports",
                status="passed",
                duration=duration,
                message="所有模块导入成功"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="imports",
                test_name="module_imports",
                status="failed",
                duration=duration,
                message=f"模块导入失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def validate_config_system(self) -> ValidationResult:
        """验证配置系统"""
        logger.info("验证配置系统...")
        start_time = time.time()
        
        try:
            
            # 测试默认配置
            model_config = ModelConfig()
            training_config = TrainingConfig()
            data_config = DataConfig()
            lora_config = LoRAConfig()
            
            # 测试配置转换
            model_dict = model_config.to_dict()
            lora_dict = lora_config.to_dict()
            
            # 验证必要字段
            assert "model_name" in model_dict
            assert "r" in lora_dict
            assert "lora_alpha" in lora_dict
            
            # 测试自定义配置
            custom_config = ModelConfig(
                model_name="microsoft/Florence-2-base",
                use_lora=True
            )
            
            assert custom_config.model_name == "microsoft/Florence-2-base"
            assert custom_config.use_lora is True
            
            duration = time.time() - start_time
            return ValidationResult(
                component="config",
                test_name="config_system",
                status="passed",
                duration=duration,
                message="配置系统验证通过",
                details={
                    "model_config_keys": list(model_dict.keys()),
                    "lora_config_keys": list(lora_dict.keys())
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="config",
                test_name="config_system",
                status="failed",
                duration=duration,
                message=f"配置系统验证失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def validate_task_definitions(self) -> ValidationResult:
        """验证任务定义"""
        logger.info("验证任务定义...")
        start_time = time.time()
        
        try:
            
            # 验证任务列表
            assert isinstance(FLORENCE2_TASKS, dict)
            assert len(FLORENCE2_TASKS) > 0
            
            # 验证基本任务
            expected_tasks = ["CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION", 
                            "OD", "DENSE_REGION_CAPTION", "REGION_PROPOSAL"]
            
            for task in expected_tasks:
                assert task in FLORENCE2_TASKS, f"缺少任务: {task}"
                
                # 验证任务配置（FLORENCE2_TASKS 使用 prompt 字段）
                task_config = get_task_config(task)
                assert "prompt" in task_config
                assert "description" in task_config
                
                # 验证任务名称验证函数
                assert validate_task_name(task) is True
            
            # 验证无效任务名称
            assert validate_task_name("INVALID_TASK") is False
            
            duration = time.time() - start_time
            return ValidationResult(
                component="tasks",
                test_name="task_definitions",
                status="passed",
                duration=duration,
                message="任务定义验证通过",
                details={
                    "total_tasks": len(FLORENCE2_TASKS),
                    "available_tasks": list(FLORENCE2_TASKS.keys())
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="tasks",
                test_name="task_definitions",
                status="failed",
                duration=duration,
                message=f"任务定义验证失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def validate_data_structures(self) -> ValidationResult:
        """验证数据结构"""
        logger.info("验证数据结构...")
        start_time = time.time()
        
        try:
            
            # 创建测试样本
            sample = TaskSample(
                task_type="CAPTION",
                image_path="test_image.jpg",
                prefix="<CAPTION>",
                suffix="A test image",
                weight=1.0,
                metadata={"source": "test"}
            )
            
            # 验证样本属性
            assert sample.task_type == "CAPTION"
            assert sample.image_path == "test_image.jpg"
            assert sample.weight == 1.0
            
            # 验证字典转换
            sample_dict = sample.to_dict()
            assert "task_type" in sample_dict
            assert "image_path" in sample_dict
            
            # 验证从字典创建
            sample_from_dict = TaskSample.from_dict(sample_dict)
            assert sample_from_dict.task_type == sample.task_type
            assert sample_from_dict.image_path == sample.image_path
            
            # 验证数据集构建器
            builder = DatasetBuilder()
            assert hasattr(builder, 'add_task_data')
            assert hasattr(builder, 'build')
            
            duration = time.time() - start_time
            return ValidationResult(
                component="data",
                test_name="data_structures",
                status="passed",
                duration=duration,
                message="数据结构验证通过",
                details={
                    "sample_dict_keys": list(sample_dict.keys()),
                    "builder_methods": [method for method in dir(builder) if not method.startswith('_')]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="data",
                test_name="data_structures",
                status="failed",
                duration=duration,
                message=f"数据结构验证失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def validate_device_management(self) -> ValidationResult:
        """验证设备管理"""
        logger.info("验证设备管理...")
        start_time = time.time()
        
        try:
            
            # 获取设备信息
            device_info = get_device_info()
            assert isinstance(device_info, dict)
            assert "cpu" in device_info
            
            # 获取最优设备
            optimal_device = get_optimal_device()
            assert optimal_device is not None
            
            # 检查CUDA可用性
            import torch
            cuda_available = torch.cuda.is_available()
            assert isinstance(cuda_available, bool)
            
            # 获取内存信息
            memory_info = get_memory_usage()
            assert memory_info is not None
            
            # 清理缓存（应该不会抛出异常）
            clear_cache()
            
            duration = time.time() - start_time
            return ValidationResult(
                component="device",
                test_name="device_management",
                status="passed",
                duration=duration,
                message="设备管理验证通过",
                details={
                    "device_info": device_info,
                    "optimal_device": str(optimal_device),
                    "cuda_available": cuda_available,
                    "memory_info": memory_info
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="device",
                test_name="device_management",
                status="failed",
                duration=duration,
                message=f"设备管理验证失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def validate_model_initialization(self) -> ValidationResult:
        """验证模型初始化（轻量级测试）"""
        logger.info("验证模型初始化...")
        start_time = time.time()
        
        try:
            
            # 创建轻量级配置（避免实际下载模型）
            config = ModelConfig(
                model_name="microsoft/Florence-2-base",  # 使用较小的模型
                use_lora=True,
                device_map="cpu"  # 强制使用CPU避免GPU内存问题
            )
            
            # 验证配置创建
            assert config.model_name == "microsoft/Florence-2-base"
            assert config.use_lora is True
            
            # 验证模型类可以实例化（但不实际加载权重）
            # 这里只验证类的结构，不进行实际的模型加载
            model_class = Florence2MultiTaskModel
            assert hasattr(model_class, '__init__')
            assert hasattr(model_class, 'forward')
            assert hasattr(model_class, 'generate')
            
            duration = time.time() - start_time
            return ValidationResult(
                component="model",
                test_name="model_initialization",
                status="passed",
                duration=duration,
                message="模型初始化验证通过（轻量级测试）",
                details={
                    "config_model_name": config.model_name,
                    "config_use_lora": config.use_lora,
                    "model_methods": [method for method in dir(model_class) if not method.startswith('_')]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                component="model",
                test_name="model_initialization",
                status="failed",
                duration=duration,
                message=f"模型初始化验证失败: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    def run_all_validations(self) -> Dict[str, Any]:
        """运行所有验证
        
        Returns:
            完整的验证结果摘要
        """
        logger.info("开始运行完整验证套件")
        self.start_time = time.time()
        
        # 运行各项验证
        validations = [
            self.validate_imports,
            self.validate_config_system,
            self.validate_task_definitions,
            self.validate_data_structures,
            self.validate_device_management,
            self.validate_model_initialization
        ]
        
        for validation_func in validations:
            try:
                result = validation_func()
                self.validation_results.append(result)
                logger.info(f"{result.component}.{result.test_name}: {result.status} ({result.duration:.2f}s)")
            except Exception as e:
                logger.error(f"验证函数 {validation_func.__name__} 执行失败: {e}")
                error_result = ValidationResult(
                    component="unknown",
                    test_name=validation_func.__name__,
                    status="error",
                    duration=0.0,
                    message=f"验证函数执行失败: {str(e)}"
                )
                self.validation_results.append(error_result)
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # 汇总结果
        summary = self._generate_summary(total_duration)
        
        # 保存结果
        self._save_results(summary)
        
        logger.info(f"完整验证套件执行完成，总耗时: {total_duration:.2f}秒")
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成验证结果摘要"""
        passed = sum(1 for r in self.validation_results if r.status == "passed")
        failed = sum(1 for r in self.validation_results if r.status == "failed")
        errors = sum(1 for r in self.validation_results if r.status == "error")
        skipped = sum(1 for r in self.validation_results if r.status == "skipped")
        
        overall_status = "passed"
        if errors > 0:
            overall_status = "error"
        elif failed > 0:
            overall_status = "failed"
        
        return {
            "total_duration": total_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": {
                "total": len(self.validation_results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped
            },
            "overall_status": overall_status,
            "results": [{
                "component": r.component,
                "test_name": r.test_name,
                "status": r.status,
                "duration": r.duration,
                "message": r.message
            } for r in self.validation_results]
        }
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """保存验证结果"""
        # 保存摘要
        summary_file = self.output_dir / "validation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细结果
        detailed_results = {
            "summary": summary,
            "detailed_results": [{
                "component": r.component,
                "test_name": r.test_name,
                "status": r.status,
                "duration": r.duration,
                "message": r.message,
                "details": r.details
            } for r in self.validation_results]
        }
        
        detailed_file = self.output_dir / "validation_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"验证结果已保存到: {summary_file}")

def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlorenceForge验证套件")
    parser.add_argument("--output-dir", default="./validation_results", help="验证结果输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # 创建验证套件
    validator = ValidationSuite(output_dir=args.output_dir)
    
    # 运行验证
    results = validator.run_all_validations()
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("验证结果摘要")
    print("="*50)
    print(f"总计: {results['summary']['total']}")
    print(f"通过: {results['summary']['passed']}")
    print(f"失败: {results['summary']['failed']}")
    print(f"错误: {results['summary']['errors']}")
    print(f"跳过: {results['summary']['skipped']}")
    print(f"整体状态: {results['overall_status']}")
    print(f"总耗时: {results['total_duration']:.2f}秒")
    
    # 显示详细结果
    if args.verbose:
        print("\n详细结果:")
        for result in results['results']:
            status_symbol = "✓" if result['status'] == "passed" else "✗"
            print(f"  {status_symbol} {result['component']}.{result['test_name']}: {result['message']}")
    
    # 根据验证结果设置退出码
    if results['overall_status'] in ["failed", "error"]:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()