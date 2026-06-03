#!/usr/bin/env python3
"""
Florence-2 训练配置使用示例

本脚本展示了如何使用不同的配置文件进行Florence-2模型训练
包括单任务训练、多任务训练、硬件适配等各种场景

使用方法:
    python scripts/usage_examples.py --example caption
    python scripts/usage_examples.py --example detection
    python scripts/usage_examples.py --example all
"""

import argparse
import sys
import subprocess
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from florence_forge.utils import setup_logging
except ImportError as e:
    print(f"警告: 无法导入日志模块: {e}")
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class UsageExamples:
    """使用示例管理器"""
    
    def __init__(self):
        """TODO: Add documentation for __init__"""
        self.project_root = project_root
        self.configs_dir = self.project_root / "configs" / "examples"
        self.scripts_dir = self.project_root / "scripts"
        
        # 示例配置映射
        self.examples = {
            'caption': {
                'name': '图像描述任务',
                'config': 'caption_training.yaml',
                'description': '训练Florence-2进行图像描述生成',
                'data_requirements': ['图像文件', '描述文本'],
                'estimated_time': '2-4小时 (10 epochs)',
                'gpu_memory': '8GB+'
            },
            'detection': {
                'name': '目标检测任务',
                'config': 'object_detection_training.yaml',
                'description': '训练Florence-2进行目标检测',
                'data_requirements': ['图像文件', '边界框标注'],
                'estimated_time': '4-8小时 (20 epochs)',
                'gpu_memory': '12GB+'
            },
            'ocr': {
                'name': 'OCR文字识别',
                'config': 'ocr_training.yaml',
                'description': '训练Florence-2进行文字识别',
                'data_requirements': ['图像文件', '文字内容'],
                'estimated_time': '3-6小时 (15 epochs)',
                'gpu_memory': '10GB+'
            },
            'segmentation': {
                'name': '图像分割任务',
                'config': 'segmentation_training.yaml',
                'description': '训练Florence-2进行图像分割',
                'data_requirements': ['图像文件', '分割掩码'],
                'estimated_time': '5-10小时 (25 epochs)',
                'gpu_memory': '16GB+'
            },
            'multitask': {
                'name': '多任务混合训练',
                'config': 'multitask_training.yaml',
                'description': '同时训练多个任务，提升模型泛化能力',
                'data_requirements': ['多种任务的数据'],
                'estimated_time': '8-16小时 (30 epochs)',
                'gpu_memory': '20GB+'
            }
        }
    
    def list_examples(self) -> None:
        """列出所有可用示例"""
        logger.info("\n=== Florence-2 训练示例 ===")
        
        for key, example in self.examples.items():
            logger.info(f"\n📋 {key}: {example['name']}")
            logger.info(f"   描述: {example['description']}")
            logger.info(f"   配置文件: {example['config']}")
            logger.info(f"   数据需求: {', '.join(example['data_requirements'])}")
            logger.info(f"   预估时间: {example['estimated_time']}")
            logger.info(f"   GPU内存: {example['gpu_memory']}")
        
        logger.info("\n使用方法:")
        logger.info("  python scripts/usage_examples.py --example <示例名称>")
        logger.info("  python scripts/usage_examples.py --example all  # 运行所有示例")
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """检查运行前提条件"""
        checks = {
            'config_files': True,
            'cli_tool': True,
            'data_directories': True,
            'output_directories': True
        }
        
        # 检查配置文件
        for example in self.examples.values():
            config_path = self.configs_dir / example['config']
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path}")
                checks['config_files'] = False
        
        # 检查CLI工具
        cli_path = self.scripts_dir / "florence_cli.py"
        if not cli_path.exists():
            logger.error(f"CLI工具不存在: {cli_path}")
            checks['cli_tool'] = False
        
        # 检查数据目录
        data_dir = self.project_root / "data"
        if not data_dir.exists():
            logger.warning(f"数据目录不存在: {data_dir}")
            logger.info("将创建示例数据目录结构")
            checks['data_directories'] = False
        
        # 检查输出目录
        output_dir = self.project_root / "outputs"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        
        return checks
    
    def create_sample_data_structure(self) -> None:
        """创建示例数据目录结构"""
        logger.info("创建示例数据目录结构...")
        
        data_structure = {
            'caption': ['train.jsonl', 'val.jsonl', 'test.jsonl'],
            'detection': ['train.jsonl', 'val.jsonl', 'test.jsonl'],
            'ocr': ['train.jsonl', 'val.jsonl', 'test.jsonl'],
            'segmentation': ['train.jsonl', 'val.jsonl', 'test.jsonl'],
            'multitask': ['train.jsonl', 'val.jsonl', 'test.jsonl']
        }
        
        base_data_dir = self.project_root / "data"
        
        for task, files in data_structure.items():
            task_dir = base_data_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            
            for file_name in files:
                file_path = task_dir / file_name
                if not file_path.exists():
                    # 创建示例数据文件
                    sample_data = self._generate_sample_data(task, file_name)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for item in sample_data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    logger.info(f"创建示例数据文件: {file_path}")
        
        # 创建图像目录
        images_dir = base_data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建README文件
        readme_content = self._generate_data_readme()
        readme_path = base_data_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"数据目录结构创建完成: {base_data_dir}")
    
    def _generate_sample_data(self, task: str, file_name: str) -> List[Dict]:
        """生成示例数据"""
        num_samples = 10 if 'train' in file_name else 5
        
        if task == 'caption':
            return [
                {
                    "image_path": f"images/sample_{i:03d}.jpg",
                    "caption": f"这是第{i+1}张示例图像的描述文本",
                    "task_type": "CAPTION"
                }
                for i in range(num_samples)
            ]
        
        elif task == 'detection':
            return [
                {
                    "image_path": f"images/sample_{i:03d}.jpg",
                    "objects": [
                        {
                            "bbox": [100, 100, 200, 200],
                            "label": "person"
                        },
                        {
                            "bbox": [250, 150, 350, 250],
                            "label": "car"
                        }
                    ],
                    "task_type": "OD"
                }
                for i in range(num_samples)
            ]
        
        elif task == 'ocr':
            return [
                {
                    "image_path": f"images/sample_{i:03d}.jpg",
                    "text": f"示例文字内容 {i+1}",
                    "regions": [
                        {
                            "bbox": [50, 50, 300, 100],
                            "text": f"文字区域 {i+1}"
                        }
                    ],
                    "task_type": "OCR"
                }
                for i in range(num_samples)
            ]
        
        elif task == 'segmentation':
            return [
                {
                    "image_path": f"images/sample_{i:03d}.jpg",
                    "mask_path": f"masks/sample_{i:03d}.png",
                    "expression": f"分割目标描述 {i+1}",
                    "task_type": "REGION_TO_SEGMENTATION"
                }
                for i in range(num_samples)
            ]
        
        elif task == 'multitask':
            tasks = ['CAPTION', 'OD', 'OCR', 'REGION_TO_SEGMENTATION']
            return [
                {
                    "image_path": f"images/sample_{i:03d}.jpg",
                    "task_type": tasks[i % len(tasks)],
                    "caption": f"图像描述 {i+1}" if tasks[i % len(tasks)] == 'CAPTION' else None,
                    "objects": [{"bbox": [100, 100, 200, 200], "label": "object"}] if tasks[i % len(tasks)] == 'OD' else None,
                    "text": f"文字内容 {i+1}" if tasks[i % len(tasks)] == 'OCR' else None,
                    "mask_path": f"masks/sample_{i:03d}.png" if tasks[i % len(tasks)] == 'REGION_TO_SEGMENTATION' else None
                }
                for i in range(num_samples)
            ]
        
        return []
    
    def _generate_data_readme(self) -> str:
        """生成数据目录README"""
        return """
# Florence-2 训练数据

本目录包含Florence-2模型训练所需的示例数据。

## 目录结构

```
data/
├── README.md                 # 本文档
├── images/                   # 图像文件目录
├── caption/                  # 图像描述任务数据
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── detection/                # 目标检测任务数据
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── ocr/                      # OCR任务数据
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── segmentation/             # 分割任务数据
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── multitask/                # 多任务数据
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## 数据格式

### 图像描述 (Caption)
```json
{
    "image_path": "images/sample_001.jpg",
    "caption": "图像的描述文本",
    "task_type": "CAPTION"
}
```

### 目标检测 (Detection)
```json
{
    "image_path": "images/sample_001.jpg",
    "objects": [
        {
            "bbox": [x1, y1, x2, y2],
            "label": "object_class"
        }
    ],
    "task_type": "OD"
}
```

### OCR文字识别
```json
{
    "image_path": "images/sample_001.jpg",
    "text": "图像中的文字内容",
    "regions": [
        {
            "bbox": [x1, y1, x2, y2],
            "text": "区域文字"
        }
    ],
    "task_type": "OCR"
}
```

### 图像分割 (Segmentation)
```json
{
    "image_path": "images/sample_001.jpg",
    "mask_path": "masks/sample_001.png",
    "expression": "分割目标的描述",
    "task_type": "REGION_TO_SEGMENTATION"
}
```

## 注意事项

1. **图像路径**: 所有图像路径都是相对于项目根目录的相对路径
2. **数据质量**: 实际训练时请使用高质量的标注数据
3. **数据量**: 示例数据仅用于测试，实际训练需要更多数据
4. **格式验证**: 使用配置验证工具检查数据格式正确性

## 数据准备建议

1. **图像预处理**: 确保图像清晰，尺寸适中
2. **标注质量**: 保证标注准确性和一致性
3. **数据平衡**: 各类别样本数量尽量均衡
4. **数据增强**: 适当使用数据增强技术
"""
    
    def run_example(self, example_name: str, dry_run: bool = False) -> bool:
        """运行指定示例"""
        if example_name not in self.examples:
            logger.error(f"未知示例: {example_name}")
            logger.info(f"可用示例: {', '.join(self.examples.keys())}")
            return False
        
        example = self.examples[example_name]
        config_path = self.configs_dir / example['config']
        
        logger.info(f"\n=== 运行示例: {example['name']} ===")
        logger.info(f"配置文件: {config_path}")
        logger.info(f"描述: {example['description']}")
        logger.info(f"预估时间: {example['estimated_time']}")
        logger.info(f"GPU内存需求: {example['gpu_memory']}")
        
        if dry_run:
            logger.info("[DRY RUN] 仅显示命令，不实际执行")
            command = [
                "python", str(self.scripts_dir / "florence_cli.py"),
                "train", "--config", str(config_path)
            ]
            logger.info(f"执行命令: {' '.join(command)}")
            return True
        
        # 检查前提条件
        checks = self.check_prerequisites()
        if not all(checks.values()):
            logger.error("前提条件检查失败")
            if not checks['data_directories']:
                logger.info("正在创建示例数据结构...")
                self.create_sample_data_structure()
            else:
                return False
        
        # 执行训练命令
        try:
            command = [
                "python", str(self.scripts_dir / "florence_cli.py"),
                "train", "--config", str(config_path)
            ]
            
            logger.info(f"执行命令: {' '.join(command)}")
            logger.info("开始训练...")
            
            start_time = time.time()
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            end_time = time.time()
            
            if result.returncode == 0:
                logger.info(f"✅ 训练完成! 耗时: {end_time - start_time:.2f}秒")
                logger.info("训练输出:")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ 训练失败! 返回码: {result.returncode}")
                logger.error("错误输出:")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"执行训练时发生错误: {e}")
            return False
    
    def run_all_examples(self, dry_run: bool = False) -> Dict[str, bool]:
        """运行所有示例"""
        results = {}
        
        logger.info("\n=== 运行所有训练示例 ===")
        
        for example_name in self.examples.keys():
            logger.info(f"\n--- 开始示例: {example_name} ---")
            success = self.run_example(example_name, dry_run)
            results[example_name] = success
            
            if success:
                logger.info(f"✅ {example_name} 完成")
            else:
                logger.error(f"❌ {example_name} 失败")
        
        # 总结结果
        logger.info("\n=== 运行结果总结 ===")
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"总计: {total} 个示例")
        logger.info(f"成功: {successful} 个")
        logger.info(f"失败: {total - successful} 个")
        
        for name, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {name}")
        
        return results
    
    def demonstrate_cli_usage(self) -> None:
        """演示CLI工具的各种用法"""
        logger.info("\n=== Florence-2 CLI工具使用演示 ===")
        
        cli_examples = [
            {
                'description': '列出所有可用任务',
                'command': 'python scripts/florence_cli.py list-tasks'
            },
            {
                'description': '运行图像描述任务',
                'command': 'python scripts/florence_cli.py train caption'
            },
            {
                'description': '使用自定义配置',
                'command': 'python scripts/florence_cli.py train --config configs/examples/caption_training.yaml'
            },
            {
                'description': '验证配置文件',
                'command': 'python scripts/florence_cli.py validate --config configs/examples/caption_training.yaml'
            },
            {
                'description': '生成配置模板',
                'command': 'python scripts/florence_cli.py generate-template --task caption --output my_config.yaml'
            }
        ]
        
        for example in cli_examples:
            logger.info(f"\n📋 {example['description']}:")
            logger.info(f"   {example['command']}")
        
        logger.info("\n=== 高级配置管理工具演示 ===")
        
        advanced_examples = [
            {
                'description': '验证所有配置文件',
                'command': 'python scripts/advanced_config_manager.py validate-all'
            },
            {
                'description': '比较两个配置文件',
                'command': 'python scripts/advanced_config_manager.py compare config1.yaml config2.yaml'
            },
            {
                'description': '检测硬件配置',
                'command': 'python scripts/advanced_config_manager.py detect-hardware'
            },
            {
                'description': '硬件优化建议',
                'command': 'python scripts/advanced_config_manager.py optimize --config config.yaml'
            },
            {
                'description': '生成硬件适配配置',
                'command': 'python scripts/advanced_config_manager.py hardware-adapt --base-config base.yaml --output optimized.yaml --gpu-memory 24'
            }
        ]
        
        for example in advanced_examples:
            logger.info(f"\n🔧 {example['description']}:")
            logger.info(f"   {example['command']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Florence-2 训练配置使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--example',
        choices=['caption', 'detection', 'ocr', 'segmentation', 'multitask', 'all'],
        help='要运行的示例'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用示例'
    )
    
    parser.add_argument(
        '--demo-cli',
        action='store_true',
        help='演示CLI工具用法'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅显示命令，不实际执行'
    )
    
    parser.add_argument(
        '--setup-data',
        action='store_true',
        help='创建示例数据结构'
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
    
    examples = UsageExamples()
    
    if args.list:
        examples.list_examples()
    
    elif args.demo_cli:
        examples.demonstrate_cli_usage()
    
    elif args.setup_data:
        examples.create_sample_data_structure()
    
    elif args.example:
        if args.example == 'all':
            results = examples.run_all_examples(args.dry_run)
            success_count = sum(results.values())
            total_count = len(results)
            sys.exit(0 if success_count == total_count else 1)
        else:
            success = examples.run_example(args.example, args.dry_run)
            sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        logger.info("\n💡 提示:")
        logger.info("  使用 --list 查看所有可用示例")
        logger.info("  使用 --demo-cli 查看CLI工具用法")
        logger.info("  使用 --setup-data 创建示例数据结构")

if __name__ == '__main__':
    main()
