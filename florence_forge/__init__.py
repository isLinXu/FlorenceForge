#!/usr/bin/env python3
"""
FlorenceForge - Florence-2多任务微调库

一个专为Florence-2模型设计的多任务微调框架，支持图像描述、目标检测、OCR、分割等多种视觉任务。

主要特性:
- 支持所有Florence-2原生任务
- 灵活的配置系统
- LoRA高效微调
- 多任务混合训练
- 完整的评估体系
- 便捷的CLI工具

使用示例:
    from florence_forge import TrainingConfig, Trainer
    
    # 加载配置
    config = TrainingConfig.from_yaml('config.yaml')
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()

命令行工具:
    florence_forge_cli train --task caption
    florence_forge_cli list-tasks
    florence_forge_cli validate --config config.yaml
"""

__version__ = "0.1.0"
__author__ = "FlorenceForge Team"
__email__ = "contact@florenceforge.ai"
__license__ = "MIT"
__url__ = "https://github.com/florenceforge/florenceforge"

# 版本信息
VERSION = __version__
AUTHOR = __author__
EMAIL = __email__
LICENSE = __license__
URL = __url__

# 导入核心组件（可选，避免循环导入）
try:
    from .core.config import TrainingConfig
    from .core.trainer import Trainer
    from .core.tasks import FLORENCE2_TASKS, TaskCategory
    
    # 导入多数据集功能
    from .data.multi_dataset_manager import MultiDatasetManager, DatasetInfo, TaskDatasetMapping
    from .training.multi_dataset_trainer import MultiDatasetTrainer
    
    # 导入CLI功能
    from .cli import cli_main, ConfigManager
    
    CORE_AVAILABLE = True
    MULTI_DATASET_AVAILABLE = True
    CLI_AVAILABLE = True
except ImportError:
    # 核心模块不可用时的优雅降级
    TrainingConfig = None
    Trainer = None
    FLORENCE2_TASKS = {}
    TaskCategory = None
    MultiDatasetManager = None
    DatasetInfo = None
    TaskDatasetMapping = None
    MultiDatasetTrainer = None
    cli_main = None
    ConfigManager = None
    CORE_AVAILABLE = False
    MULTI_DATASET_AVAILABLE = False
    CLI_AVAILABLE = False

# 公开API
__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__url__',
    'VERSION',
    'AUTHOR',
    'EMAIL',
    'LICENSE',
    'URL',
    'TrainingConfig',
    'Trainer',
    'FLORENCE2_TASKS',
    'TaskCategory',
    'MultiDatasetManager',
    'DatasetInfo',
    'TaskDatasetMapping',
    'MultiDatasetTrainer',
    'cli_main',
    'ConfigManager',
    'CORE_AVAILABLE',
    'MULTI_DATASET_AVAILABLE',
    'CLI_AVAILABLE',
    'print_info',
    'print_examples',
    'check_dependencies',
]

def print_info():
    """打印库信息"""
    print(f"\n{'='*60}")
    print(f"FlorenceForge v{__version__}")
    print(f"{'='*60}")
    print(f"描述: Florence-2多任务微调库")
    print(f"作者: {__author__}")
    print(f"邮箱: {__email__}")
    print(f"许可: {__license__}")
    print(f"主页: {__url__}")
    print(f"核心模块: {'可用' if CORE_AVAILABLE else '不可用'}")
    print(f"多数据集功能: {'可用' if MULTI_DATASET_AVAILABLE else '不可用'}")
    print(f"CLI工具: {'可用' if CLI_AVAILABLE else '不可用'}")
    print(f"{'='*60}\n")

def print_examples():
    """打印使用示例"""
    print("\n🎯 FlorenceForge 使用示例")
    print("\n1. 命令行工具:")
    print("   florence_forge_cli train --task caption")
    print("   florence_forge_cli train --task detection --epochs 10")
    print("   florence_forge_cli list-tasks")
    print("   florence_forge_cli validate --config config.yaml")
    
    print("\n2. Python API:")
    print("   from florence_forge import TrainingConfig, Trainer")
    print("   config = TrainingConfig.from_yaml('config.yaml')")
    print("   trainer = Trainer(config)")
    print("   trainer.train()")
    
    print("\n3. 多数据集训练:")
    print("   from florence_forge import MultiDatasetManager, MultiDatasetTrainer")
    print("   manager = MultiDatasetManager()")
    print("   trainer = MultiDatasetTrainer(model, manager, config)")
    print("   trainer.train()")
    
    print("\n4. 配置文件:")
    print("   florence_forge_cli generate-config --task caption --output my_config.yaml")
    
    print("\n📚 更多信息: https://florenceforge.readthedocs.io\n")

def check_dependencies():
    """检查依赖项"""
    import sys
    import importlib
    
    required_packages = [
        'torch',
        'torchvision', 
        'transformers',
        'peft',
        'accelerate',
        'datasets',
        'tokenizers',
        'pillow',
        'numpy',
        'yaml'
    ]
    
    results = {}
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = {'status': '✅', 'version': version}
        except ImportError:
            results[package] = {'status': '❌', 'version': 'not installed'}
    
    print("\n📦 依赖检查结果:")
    print("-" * 40)
    for package, info in results.items():
        print(f"{info['status']} {package:<15} {info['version']}")
    
    missing = [pkg for pkg, info in results.items() if info['status'] == '❌']
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
    else:
        print("\n🎉 所有依赖都已安装!")
    
    print(f"\nPython版本: {sys.version}")
    print(f"核心模块: {'可用' if CORE_AVAILABLE else '不可用'}")
    print(f"CLI工具: {'可用' if CLI_AVAILABLE else '不可用'}")
    
    return results

# 包初始化时的信息
if __name__ != '__main__':
    # 静默导入，避免在导入时打印信息
    pass