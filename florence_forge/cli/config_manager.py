#!/usr/bin/env python3
"""
配置管理脚本 - 用于创建、验证和转换Florence-2训练配置

功能:
- 创建默认配置文件
- 验证配置文件格式和内容
- 在JSON和YAML格式之间转换
- 合并多个配置文件
- 生成配置模板
"""

import argparse
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..core.config import (
    TrainingConfig, ModelConfig, DataConfig, 
    OptimizationConfig, LoRAConfig
)
from ..core.yaml_config import (
    FlorenceForgeYAMLConfig, create_yaml_config_template, validate_yaml_config
)
from ..core.tasks import list_all_tasks, get_tasks_by_category

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        """初始化配置管理器
        
        设置支持的配置文件格式列表，包括JSON和YAML格式。
        """
        self.supported_formats = ['.json', '.yaml', '.yml']
    
    def create_default_config(self, output_path: str, format_type: str = 'yaml') -> None:
        """创建默认配置文件
        
        Args:
            output_path: 输出文件路径
            format_type: 文件格式 ('json' 或 'yaml')
        """
        config = TrainingConfig()
        
        # 设置一些合理的默认值
        config.experiment_name = "florence2_default_experiment"
        config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config.tags = ["florence2", "multitask", "default"]
        
        output_path = Path(output_path)
        if format_type.lower() == 'yaml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.yaml')
            config.save_to_yaml(output_path)
        else:
            if not output_path.suffix:
                output_path = output_path.with_suffix('.json')
            config.save_to_json(output_path)
        
        print(f"✓ 默认配置已创建: {output_path}")
    
    def validate_config(self, config_path: str) -> bool:
        """验证配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            验证是否通过
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                print(f"✗ 配置文件不存在: {config_path}")
                return False
            
            if config_path.suffix.lower() not in self.supported_formats:
                print(f"✗ 不支持的文件格式: {config_path.suffix}")
                return False
            
            # 尝试加载配置
            config = TrainingConfig.load_from_file(config_path)
            
            # 基本验证
            errors = []
            
            if config.num_epochs <= 0:
                errors.append("num_epochs 必须大于 0")
            
            if config.data_settings.batch_size <= 0:
                errors.append("batch_size 必须大于 0")
            
            if config.optimization_settings.learning_rate <= 0:
                errors.append("learning_rate 必须大于 0")
            
            if not config.model_settings.model_name:
                errors.append("model_name 不能为空")
            
            if config.model_settings.use_lora:
                if config.model_settings.lora_config.r <= 0:
                    errors.append("LoRA rank (r) 必须大于 0")
            
            if errors:
                print(f"✗ 配置验证失败:")
                for error in errors:
                    print(f"  - {error}")
                return False
            
            print(f"✓ 配置验证通过: {config_path}")
            return True
            
        except Exception as e:
            print(f"✗ 配置验证失败: {e}")
            return False
    
    def convert_config(self, input_path: str, output_path: str) -> None:
        """转换配置文件格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # 加载配置
            config = TrainingConfig.load_from_file(input_path)
            
            # 保存为新格式
            config.save_to_file(output_path)
            
            print(f"✓ 配置转换完成: {input_path} -> {output_path}")
            
        except Exception as e:
            print(f"✗ 配置转换失败: {e}")
    
    def merge_configs(self, base_config_path: str, override_config_path: str, 
                     output_path: str) -> None:
        """合并配置文件
        
        Args:
            base_config_path: 基础配置文件路径
            override_config_path: 覆盖配置文件路径
            output_path: 输出文件路径
        """
        try:
            # 加载基础配置
            base_config = TrainingConfig.load_from_file(base_config_path)
            
            # 加载覆盖配置的字典
            override_path = Path(override_config_path)
            if override_path.suffix.lower() in ['.yaml', '.yml']:
                with open(override_path, 'r', encoding='utf-8') as f:
                    override_dict = yaml.safe_load(f)
            else:
                with open(override_path, 'r', encoding='utf-8') as f:
                    override_dict = json.load(f)
            
            # 移除元数据
            override_dict.pop('_metadata', None)
            
            # 合并配置
            base_dict = base_config.to_dict()
            merged_dict = self._deep_merge(base_dict, override_dict)
            
            # 创建新配置并保存
            merged_config = TrainingConfig.from_dict(merged_dict)
            merged_config.save_to_file(output_path)
            
            print(f"✓ 配置合并完成: {output_path}")
            
        except Exception as e:
            print(f"✗ 配置合并失败: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def show_config_info(self, config_path: str) -> None:
        """显示配置文件信息
        
        Args:
            config_path: 配置文件路径
        """
        try:
            config_path = Path(config_path)
            config = TrainingConfig.load_from_file(config_path)
            
            print(f"\n配置文件信息: {config_path}")
            print("=" * 50)
            print(f"实验名称: {config.experiment_name or 'N/A'}")
            print(f"运行名称: {config.run_name or 'N/A'}")
            print(f"模型: {config.model_settings.model_name}")
            print(f"使用LoRA: {config.model_settings.use_lora}")
            if config.model_settings.use_lora:
                print(f"LoRA Rank: {config.model_settings.lora_config.r}")
            print(f"训练轮数: {config.num_epochs}")
            print(f"批次大小: {config.data_settings.batch_size}")
            print(f"学习率: {config.optimization_settings.learning_rate}")
            print(f"输出目录: {config.output_dir}")
            print(f"标签: {', '.join(config.tags) if config.tags else 'N/A'}")
            
        except Exception as e:
            print(f"✗ 无法读取配置信息: {e}")
    
    def create_template(self, template_type: str, output_path: str) -> None:
        """创建配置模板
        
        Args:
            template_type: 模板类型 ('minimal', 'full', 'lora', 'production', 'yaml_multitask')
            output_path: 输出文件路径
        """
        if template_type == 'yaml_multitask':
            # 创建YAML多任务配置模板
            create_yaml_config_template(output_path)
            print(f"✓ YAML多任务模板已创建: {output_path}")
            return
            
        if template_type == 'minimal':
            config = TrainingConfig(
                num_epochs=3,
                output_dir="./outputs/minimal_test",
                model_config=ModelConfig(
                    model_name="microsoft/Florence-2-base",
                    use_lora=True,
                    lora_config=LoRAConfig(r=8, lora_alpha=16)
                ),
                data_config=DataConfig(batch_size=2),
                optimization_config=OptimizationConfig(learning_rate=2e-5)
            )
        elif template_type == 'lora':
            config = TrainingConfig(
                model_config=ModelConfig(
                    use_lora=True,
                    lora_config=LoRAConfig(
                        r=64,
                        lora_alpha=128,
                        lora_dropout=0.1
                    )
                )
            )
        elif template_type == 'production':
            config = TrainingConfig(
                num_epochs=20,
                eval_steps=200,
                save_steps=500,
                early_stopping_patience=3,
                model_config=ModelConfig(
                    model_name="microsoft/Florence-2-large",
                    use_lora=True
                ),
                data_config=DataConfig(
                    batch_size=8,
                    use_balanced_sampling=True
                ),
                optimization_config=OptimizationConfig(
                    learning_rate=1e-5,
                    lr_scheduler_type="cosine",
                    warmup_ratio=0.1
                )
            )
        else:  # full
            config = TrainingConfig()
        
        config.save_to_file(output_path)
        print(f"✓ {template_type} 模板已创建: {output_path}")
    
    def validate_yaml_config(self, config_path: str) -> bool:
        """验证YAML配置文件
        
        Args:
            config_path: YAML配置文件路径
            
        Returns:
            验证是否通过
        """
        try:
            result = validate_yaml_config(config_path)
            if result:
                print(f"✓ YAML配置验证通过: {config_path}")
            else:
                print(f"✗ YAML配置验证失败: {config_path}")
            return result
        except Exception as e:
            print(f"✗ YAML配置验证失败: {e}")
            return False
    
    def show_yaml_config_info(self, config_path: str) -> None:
        """显示YAML配置文件信息
        
        Args:
            config_path: YAML配置文件路径
        """
        try:
            yaml_config = FlorenceForgeYAMLConfig.load_from_file(config_path)
            
            print(f"\nYAML配置文件信息: {config_path}")
            print("=" * 50)
            print(f"项目名称: {yaml_config.project_name}")
            print(f"项目描述: {yaml_config.description}")
            print(f"实验名称: {yaml_config.experiment_name or 'N/A'}")
            print(f"启用的任务类型: {', '.join(yaml_config.enabled_tasks)}")
            print(f"数据集数量: {len(yaml_config.datasets)}")
            print(f"任务映射数量: {len(yaml_config.task_mappings)}")
            print(f"输出目录: {yaml_config.output_dir}")
            print(f"图像基础路径: {yaml_config.image_base_path or 'N/A'}")
            
            # 显示训练配置信息
            if yaml_config.training:
                print(f"\n训练配置:")
                training = yaml_config.training
                print(f"  训练轮数: {training.get('num_epochs', 'N/A')}")
                print(f"  批次大小: {training.get('batch_size', 'N/A')}")
                print(f"  学习率: {training.get('learning_rate', 'N/A')}")
                print(f"  模型名称: {training.get('model_name', 'N/A')}")
                print(f"  使用LoRA: {training.get('use_lora', 'N/A')}")
            
            print("\n数据集详情:")
            for dataset in yaml_config.datasets:
                print(f"  - {dataset.name}: {dataset.path} ({', '.join(dataset.task_types)})")
            
            print("\n任务映射详情:")
            for mapping in yaml_config.task_mappings:
                print(f"  - {mapping.task_type}: {len(mapping.datasets)} 个数据集 ({', '.join(mapping.datasets)})")
                
        except Exception as e:
            print(f"✗ 无法读取YAML配置信息: {e}")
    
    def list_supported_tasks(self) -> None:
        """列出所有支持的任务类型"""
        from florence_forge.core.tasks import FLORENCE2_TASKS, TaskCategory
        
        print("\n支持的Florence-2任务类型:")
        print("=" * 50)
        
        categories = {}
        
        for task_name, task_info in FLORENCE2_TASKS.items():
            category = task_info.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append((task_name, task_info.description))
        
        for category, task_list in categories.items():
            print(f"\n{category.upper()}:")
            for task_name, description in task_list:
                print(f"  - {task_name}: {description}")
    
    def convert_to_yaml_config(self, training_config_path: str, output_path: str) -> None:
        """将传统训练配置转换为YAML配置
        
        Args:
            training_config_path: 传统训练配置文件路径
            output_path: 输出YAML配置文件路径
        """
        try:
            # 加载传统配置
            training_config = TrainingConfig.load_from_file(training_config_path)
            
            # 创建YAML配置
            yaml_config = FlorenceForgeYAMLConfig.create_example_config()
            
            # Preserve the nested schema consumed by FlorenceForgeYAMLConfig.
            yaml_config.training = training_config.to_dict()
            yaml_config.output_dir = training_config.output_dir
            yaml_config.experiment_name = training_config.experiment_name
            
            # 保存YAML配置
            yaml_config.save_to_file(output_path)
            
            print(f"✓ 配置转换完成: {training_config_path} -> {output_path}")
            print("注意: 请手动添加数据集和任务映射配置")
            
        except Exception as e:
            print(f"✗ 配置转换失败: {e}")

def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(description="Florence-2 配置管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建默认配置
    create_parser = subparsers.add_parser('create', help='创建默认配置文件')
    create_parser.add_argument('output', help='输出文件路径')
    create_parser.add_argument('--format', choices=['json', 'yaml'], default='yaml',
                              help='文件格式 (默认: yaml)')
    
    # 验证配置
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config', help='配置文件路径')
    
    # 转换格式
    convert_parser = subparsers.add_parser('convert', help='转换配置文件格式')
    convert_parser.add_argument('input', help='输入文件路径')
    convert_parser.add_argument('output', help='输出文件路径')
    
    # 合并配置
    merge_parser = subparsers.add_parser('merge', help='合并配置文件')
    merge_parser.add_argument('base', help='基础配置文件路径')
    merge_parser.add_argument('override', help='覆盖配置文件路径')
    merge_parser.add_argument('output', help='输出文件路径')
    
    # 显示信息
    info_parser = subparsers.add_parser('info', help='显示配置文件信息')
    info_parser.add_argument('config', help='配置文件路径')
    
    # 创建模板
    template_parser = subparsers.add_parser('template', help='创建配置模板')
    template_parser.add_argument('template_type', choices=['minimal', 'full', 'lora', 'production', 'yaml_multitask'],
                                help='模板类型')
    template_parser.add_argument('output', help='输出文件路径')
    
    # YAML配置验证
    yaml_validate_parser = subparsers.add_parser('yaml-validate', help='验证YAML配置文件')
    yaml_validate_parser.add_argument('config', help='YAML配置文件路径')
    
    # YAML配置信息
    yaml_info_parser = subparsers.add_parser('yaml-info', help='显示YAML配置文件信息')
    yaml_info_parser.add_argument('config', help='YAML配置文件路径')
    
    # 列出支持的任务
    subparsers.add_parser('list-tasks', help='列出所有支持的任务类型')
    
    # 转换为YAML配置
    convert_yaml_parser = subparsers.add_parser('convert-to-yaml', help='将传统配置转换为YAML配置')
    convert_yaml_parser.add_argument('input', help='输入训练配置文件路径')
    convert_yaml_parser.add_argument('output', help='输出YAML配置文件路径')
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    # 解析参数
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    try:
        # 执行命令
        if args.command == 'create':
            manager.create_default_config(
                args.output,
                args.format
            )
        
        elif args.command == 'validate':
            success = manager.validate_config(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'convert':
            manager.convert_config(
                args.input,
                args.output
            )
        
        elif args.command == 'merge':
            manager.merge_configs(
                args.base,
                args.override,
                args.output
            )
        
        elif args.command == 'info':
            manager.show_config_info(args.config)
        
        elif args.command == 'template':
            manager.create_template(
                args.template_type,
                args.output
            )
        
        elif args.command == 'yaml-validate':
            success = manager.validate_yaml_config(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'yaml-info':
            manager.show_yaml_config_info(args.config)
        
        elif args.command == 'list-tasks':
            manager.list_supported_tasks()
        
        elif args.command == 'convert-to-yaml':
            manager.convert_to_yaml_config(
                args.input,
                args.output
            )
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
