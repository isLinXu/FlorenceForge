#!/usr/bin/env python3
"""
Florence-2 配置和CLI工具快速测试脚本

本脚本用于快速验证所有配置文件和CLI工具是否正常工作
包括配置文件语法检查、CLI工具功能测试、硬件检测等

使用方法:
    python scripts/quick_test.py                    # 运行所有测试
    python scripts/quick_test.py --config-only      # 仅测试配置文件
    python scripts/quick_test.py --cli-only         # 仅测试CLI工具
    python scripts/quick_test.py --hardware-only    # 仅测试硬件检测
"""

import argparse
import sys
import subprocess
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# 添加项目根目录到Python路径
# 本脚本位于 scripts/testing/ 下，仓库根目录需向上回溯三层
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from florence_forge.utils.logging import setup_logging
except ImportError:
    setup_logging = None

# 简化导入，避免循环依赖
try:
    # 尝试导入核心配置模块
    sys.path.insert(0, str(project_root / "core"))
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入核心配置模块: {e}")
    TrainingConfig = None
    CORE_AVAILABLE = False

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class QuickTester:
    """快速测试器"""
    
    def __init__(self):
        """TODO: Add documentation for __init__"""
        self.project_root = project_root
        self.configs_dir = self.project_root / "configs"
        self.examples_dir = self.configs_dir / "examples"
        self.scripts_dir = self.project_root / "scripts"
        
        # 测试结果
        self.test_results = {
            'config_tests': {},
            'cli_tests': {},
            'hardware_tests': {},
            'integration_tests': {}
        }
        
        # 预期的配置文件
        self.expected_configs = [
            'caption_training.yaml',
            'object_detection_training.yaml',
            'ocr_training.yaml',
            'segmentation_training.yaml',
            'multitask_training.yaml'
        ]
        
        # 预期的CLI脚本
        self.expected_scripts = [
            'florence_cli.py',
            'advanced_config_manager.py',
            'usage_examples.py'
        ]
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("\n🚀 开始Florence-2快速测试...")
        
        overall_results = {}
        
        # 配置文件测试
        logger.info("\n=== 配置文件测试 ===")
        config_success = self.test_configurations()
        overall_results['configurations'] = config_success
        
        # CLI工具测试
        logger.info("\n=== CLI工具测试 ===")
        cli_success = self.test_cli_tools()
        overall_results['cli_tools'] = cli_success
        
        # 硬件检测测试
        logger.info("\n=== 硬件检测测试 ===")
        hardware_success = self.test_hardware_detection()
        overall_results['hardware_detection'] = hardware_success
        
        # 集成测试
        logger.info("\n=== 集成测试 ===")
        integration_success = self.test_integration()
        overall_results['integration'] = integration_success
        
        # 总结结果
        self.print_test_summary(overall_results)
        
        return overall_results
    
    def test_configurations(self) -> bool:
        """测试配置文件"""
        all_passed = True
        
        # 检查配置文件是否存在
        logger.info("检查配置文件存在性...")
        for config_name in self.expected_configs:
            config_path = self.examples_dir / config_name
            exists = config_path.exists()
            self.test_results['config_tests'][f'{config_name}_exists'] = exists
            
            if exists:
                logger.info(f"  ✅ {config_name}")
            else:
                logger.error(f"  ❌ {config_name} - 文件不存在")
                all_passed = False
        
        # 检查配置文件语法
        logger.info("\n检查配置文件语法...")
        for config_name in self.expected_configs:
            config_path = self.examples_dir / config_name
            if config_path.exists():
                syntax_ok, error_msg = self._check_yaml_syntax(config_path)
                self.test_results['config_tests'][f'{config_name}_syntax'] = syntax_ok
                
                if syntax_ok:
                    logger.info(f"  ✅ {config_name} - 语法正确")
                else:
                    logger.error(f"  ❌ {config_name} - 语法错误: {error_msg}")
                    all_passed = False
        
        # 检查配置文件内容
        logger.info("\n检查配置文件内容...")
        for config_name in self.expected_configs:
            config_path = self.examples_dir / config_name
            if config_path.exists():
                content_ok, error_msg = self._validate_config_content(config_path)
                self.test_results['config_tests'][f'{config_name}_content'] = content_ok
                
                if content_ok:
                    logger.info(f"  ✅ {config_name} - 内容有效")
                else:
                    logger.error(f"  ❌ {config_name} - 内容错误: {error_msg}")
                    all_passed = False
        
        return all_passed
    
    def test_cli_tools(self) -> bool:
        """测试CLI工具"""
        all_passed = True
        
        # 检查CLI脚本是否存在
        logger.info("检查CLI脚本存在性...")
        for script_name in self.expected_scripts:
            script_path = self.scripts_dir / script_name
            exists = script_path.exists()
            self.test_results['cli_tests'][f'{script_name}_exists'] = exists
            
            if exists:
                logger.info(f"  ✅ {script_name}")
            else:
                logger.error(f"  ❌ {script_name} - 文件不存在")
                all_passed = False
        
        # 测试florence_cli.py功能
        logger.info("\n测试florence_cli.py功能...")
        cli_tests = [
            {
                'name': 'list_tasks',
                'command': ['python', str(self.scripts_dir / 'florence_cli.py'), 'list-tasks'],
                'description': '列出任务'
            },
            {
                'name': 'help',
                'command': ['python', str(self.scripts_dir / 'florence_cli.py'), '--help'],
                'description': '显示帮助'
            }
        ]
        
        for test in cli_tests:
            try:
                result = subprocess.run(
                    test['command'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                success = result.returncode == 0
                self.test_results['cli_tests'][f"florence_cli_{test['name']}"] = success
                
                if success:
                    logger.info(f"  ✅ {test['description']}")
                else:
                    logger.error(f"  ❌ {test['description']} - 返回码: {result.returncode}")
                    logger.error(f"     错误: {result.stderr}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"  ❌ {test['description']} - 超时")
                self.test_results['cli_tests'][f"florence_cli_{test['name']}"] = False
                all_passed = False
            except Exception as e:
                logger.error(f"  ❌ {test['description']} - 异常: {e}")
                self.test_results['cli_tests'][f"florence_cli_{test['name']}"] = False
                all_passed = False
        
        # 测试advanced_config_manager.py功能
        logger.info("\n测试advanced_config_manager.py功能...")
        advanced_tests = [
            {
                'name': 'validate_all',
                'command': ['python', str(self.scripts_dir / 'advanced_config_manager.py'), 'validate-all'],
                'description': '验证所有配置'
            },
            {
                'name': 'detect_hardware',
                'command': ['python', str(self.scripts_dir / 'advanced_config_manager.py'), 'detect-hardware'],
                'description': '检测硬件'
            }
        ]
        
        for test in advanced_tests:
            try:
                result = subprocess.run(
                    test['command'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                success = result.returncode == 0
                self.test_results['cli_tests'][f"advanced_manager_{test['name']}"] = success
                
                if success:
                    logger.info(f"  ✅ {test['description']}")
                else:
                    logger.error(f"  ❌ {test['description']} - 返回码: {result.returncode}")
                    logger.error(f"     错误: {result.stderr}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"  ❌ {test['description']} - 超时")
                self.test_results['cli_tests'][f"advanced_manager_{test['name']}"] = False
                all_passed = False
            except Exception as e:
                logger.error(f"  ❌ {test['description']} - 异常: {e}")
                self.test_results['cli_tests'][f"advanced_manager_{test['name']}"] = False
                all_passed = False
        
        return all_passed
    
    def test_hardware_detection(self) -> bool:
        """测试硬件检测"""
        all_passed = True
        
        logger.info("检测系统硬件信息...")
        
        try:
            # 检测CPU信息
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            logger.info(f"  CPU核心数: {cpu_count}")
            logger.info(f"  系统内存: {memory_gb:.1f} GB")
            
            self.test_results['hardware_tests']['cpu_detection'] = True
            self.test_results['hardware_tests']['memory_detection'] = True
            
        except Exception as e:
            logger.error(f"  ❌ CPU/内存检测失败: {e}")
            self.test_results['hardware_tests']['cpu_detection'] = False
            self.test_results['hardware_tests']['memory_detection'] = False
            all_passed = False
        
        # 检测GPU信息
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                logger.info(f"  GPU可用: 是 ({gpu_count} 个)")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_properties(i).name
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                logger.info("  GPU可用: 否")
            
            self.test_results['hardware_tests']['gpu_detection'] = True
            
        except ImportError:
            logger.warning("  ⚠️  PyTorch未安装，无法检测GPU")
            self.test_results['hardware_tests']['gpu_detection'] = False
        except Exception as e:
            logger.error(f"  ❌ GPU检测失败: {e}")
            self.test_results['hardware_tests']['gpu_detection'] = False
            all_passed = False
        
        return all_passed
    
    def test_integration(self) -> bool:
        """测试集成功能"""
        all_passed = True
        
        logger.info("测试配置文件与CLI工具集成...")
        
        # 测试配置验证功能
        for config_name in self.expected_configs[:2]:  # 只测试前两个配置
            config_path = self.examples_dir / config_name
            if config_path.exists():
                try:
                    # 使用CLI工具验证配置
                    result = subprocess.run([
                        'python', str(self.scripts_dir / 'florence_cli.py'),
                        'validate', '--config', str(config_path)
                    ], 
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                    )
                    
                    success = result.returncode == 0
                    test_name = f'validate_{config_name.replace(".yaml", "")}'
                    self.test_results['integration_tests'][test_name] = success
                    
                    if success:
                        logger.info(f"  ✅ 验证 {config_name}")
                    else:
                        logger.error(f"  ❌ 验证 {config_name} 失败")
                        logger.error(f"     错误: {result.stderr}")
                        all_passed = False
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"  ❌ 验证 {config_name} 超时")
                    all_passed = False
                except Exception as e:
                    logger.error(f"  ❌ 验证 {config_name} 异常: {e}")
                    all_passed = False
        
        return all_passed

    def _check_yaml_syntax(self, config_path: Path) -> Tuple[bool, str]:
        """检查YAML语法"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True, ""
        except yaml.YAMLError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _validate_config_content(self, config_path: Path) -> Tuple[bool, str]:
        """验证配置文件内容"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 检查必要字段
            required_fields = [
                'num_epochs', 'model_config', 'data_config', 'optimization_config'
            ]
            
            for field in required_fields:
                if field not in config_data:
                    return False, f"缺少必要字段: {field}"
            
            # 检查数值合理性
            if config_data.get('num_epochs', 0) <= 0:
                return False, "num_epochs 必须大于 0"
            
            if 'batch_size' in config_data.get('data_config', {}):
                if config_data['data_config']['batch_size'] <= 0:
                    return False, "batch_size 必须大于 0"
            
            if 'learning_rate' in config_data.get('optimization_config', {}):
                if config_data['optimization_config']['learning_rate'] <= 0:
                    return False, "learning_rate 必须大于 0"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def print_test_summary(self, overall_results: Dict[str, bool]) -> None:
        """打印测试总结"""
        logger.info("\n" + "="*60)
        logger.info("🎯 测试结果总结")
        logger.info("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, success in overall_results.items():
            status = "✅ 通过" if success else "❌ 失败"
            logger.info(f"{category:20s}: {status}")
            total_tests += 1
            if success:
                passed_tests += 1
        
        logger.info(f"\n总计: {passed_tests}/{total_tests} 项测试通过")
        
        if passed_tests == total_tests:
            logger.info("🎉 所有测试通过！Florence-2配置和CLI工具运行正常")
        else:
            logger.warning(f"⚠️  有 {total_tests - passed_tests} 项测试失败，请检查相关配置")
        
        # 详细结果
        logger.info("\n详细测试结果:")
        for category, tests in self.test_results.items():
            if tests:
                logger.info(f"\n{category}:")
                for test_name, result in tests.items():
                    status = "✅" if result else "❌"
                    logger.info(f"  {status} {test_name}")
    
    def save_test_report(self, output_path: Optional[Path] = None) -> None:
        """保存测试报告"""
        if output_path is None:
            output_path = self.project_root / "test_report.json"
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
            'summary': {
                'total_categories': len(self.test_results),
                'passed_categories': sum(1 for tests in self.test_results.values() if all(tests.values())),
                'total_tests': sum(len(tests) for tests in self.test_results.values()),
                'passed_tests': sum(sum(tests.values()) for tests in self.test_results.values())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 测试报告已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Florence-2 配置和CLI工具快速测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='仅测试配置文件'
    )
    
    parser.add_argument(
        '--cli-only',
        action='store_true',
        help='仅测试CLI工具'
    )
    
    parser.add_argument(
        '--hardware-only',
        action='store_true',
        help='仅测试硬件检测'
    )
    
    parser.add_argument(
        '--integration-only',
        action='store_true',
        help='仅测试集成功能'
    )
    
    parser.add_argument(
        '--save-report',
        help='保存测试报告到指定文件'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    try:
        setup_logging(level=level)
    except Exception:
        logging.basicConfig(level=level)
    
    tester = QuickTester()
    
    # 根据参数运行特定测试
    if args.config_only:
        success = tester.test_configurations()
    elif args.cli_only:
        success = tester.test_cli_tools()
    elif args.hardware_only:
        success = tester.test_hardware_detection()
    elif args.integration_only:
        success = tester.test_integration()
    else:
        # 运行所有测试
        overall_results = tester.run_all_tests()
        success = all(overall_results.values())
    
    # 保存测试报告
    if args.save_report:
        tester.save_test_report(Path(args.save_report))
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()