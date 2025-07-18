#!/usr/bin/env python3
"""
简化的Florence-2配置测试脚本
避免复杂依赖，专注于基本功能验证
"""

import sys
import yaml
import argparse
import subprocess
import logging

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_yaml_files():
    """测试YAML配置文件"""
    logger.info("=== 测试YAML配置文件 ===")
    
    config_dir = project_root / "configs" / "examples"
    if not config_dir.exists():
        logger.error(f"配置目录不存在: {config_dir}")
        return False
    
    yaml_files = list(config_dir.glob("*.yaml"))
    if not yaml_files:
        logger.warning("未找到YAML配置文件")
        return False
    
    success_count = 0
    total_count = len(yaml_files)
    
    for yaml_file in yaml_files:
        try:
            logger.info(f"测试: {yaml_file.name}")
            
            # 检查文件大小
            file_size = yaml_file.stat().st_size
            if file_size == 0:
                logger.error(f"  ❌ 文件为空")
                continue
            
            # 测试YAML语法
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                logger.error(f"  ❌ 配置数据为空")
                continue
            
            # 检查基本结构
            required_sections = ['num_epochs', 'model_config', 'data_config']
            missing_sections = []
            for section in required_sections:
                if section not in config_data:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.warning(f"  ⚠️  缺少部分: {missing_sections}")
            else:
                logger.info(f"  ✅ 结构完整")
            
            # 检查数值合理性
            if 'num_epochs' in config_data:
                epochs = config_data['num_epochs']
                if isinstance(epochs, int) and epochs > 0:
                    logger.info(f"  ✅ epochs: {epochs}")
                else:
                    logger.warning(f"  ⚠️  epochs值异常: {epochs}")
            
            success_count += 1
            logger.info(f"  ✅ {yaml_file.name} 测试通过")
            
        except yaml.YAMLError as e:
            logger.error(f"  ❌ YAML语法错误: {e}")
        except Exception as e:
            logger.error(f"  ❌ 测试失败: {e}")
    
    logger.info(f"\n配置文件测试结果: {success_count}/{total_count} 通过")
    return success_count == total_count

def test_cli_scripts():
    """测试CLI脚本"""
    logger.info("\n=== 测试CLI脚本 ===")
    
    scripts_dir = project_root / "scripts"
    expected_scripts = [
        'florence_cli.py',
        'advanced_config_manager.py',
        'usage_examples.py'
    ]
    
    success_count = 0
    total_count = len(expected_scripts)
    
    for script_name in expected_scripts:
        script_path = scripts_dir / script_name
        
        if script_path.exists():
            file_size = script_path.stat().st_size
            if file_size > 0:
                logger.info(f"  ✅ {script_name} (大小: {file_size} 字节)")
                success_count += 1
            else:
                logger.error(f"  ❌ {script_name} 文件为空")
        else:
            logger.error(f"  ❌ {script_name} 不存在")
    
    # 测试florence_cli.py的基本功能
    florence_cli = scripts_dir / 'florence_cli.py'
    if florence_cli.exists():
        try:
            logger.info("\n测试florence_cli.py基本功能...")
            
            # 测试帮助命令
            result = subprocess.run(
                ['python', str(florence_cli), '--help'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("  ✅ --help 命令正常")
            else:
                logger.warning(f"  ⚠️  --help 命令异常: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("  ⚠️  CLI测试超时")
        except Exception as e:
            logger.warning(f"  ⚠️  CLI测试异常: {e}")
    
    logger.info(f"\nCLI脚本测试结果: {success_count}/{total_count} 通过")
    return success_count == total_count

def test_directory_structure():
    """测试目录结构"""
    logger.info("\n=== 测试目录结构 ===")
    
    expected_dirs = [
        'configs',
        'configs/examples',
        'scripts',
        'core',
        'utils'
    ]
    
    success_count = 0
    total_count = len(expected_dirs)
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            logger.info(f"  ✅ {dir_path}/")
            success_count += 1
        else:
            logger.error(f"  ❌ {dir_path}/ 不存在")
    
    logger.info(f"\n目录结构测试结果: {success_count}/{total_count} 通过")
    return success_count == total_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Florence-2简化测试工具')
    parser.add_argument('--config-only', action='store_true', help='仅测试配置文件')
    parser.add_argument('--cli-only', action='store_true', help='仅测试CLI工具')
    parser.add_argument('--structure-only', action='store_true', help='仅测试目录结构')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Florence-2简化测试开始...")
    logger.info(f"项目根目录: {project_root}")
    
    results = []
    
    if args.structure_only:
        results.append(test_directory_structure())
    elif args.config_only:
        results.append(test_yaml_files())
    elif args.cli_only:
        results.append(test_cli_scripts())
    else:
        # 运行所有测试
        results.append(test_directory_structure())
        results.append(test_yaml_files())
        results.append(test_cli_scripts())
    
    # 总结
    passed_tests = sum(results)
    total_tests = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试总结: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过!")
        return 0
    else:
        logger.error("❌ 部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())