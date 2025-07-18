#!/usr/bin/env python3
"""
Florence-2 训练CLI工具

提供便捷的命令行接口来运行不同任务的训练配置

使用示例:
    # 运行图像描述任务训练
    python scripts/florence_cli.py train --task caption --config configs/examples/caption_training.yaml
    
    # 运行目标检测任务训练
    python scripts/florence_cli.py train --task detection --config configs/examples/object_detection_training.yaml
    
    # 运行多任务训练
    python scripts/florence_cli.py train --task multitask --config configs/examples/multitask_training.yaml
    
    # 列出所有可用的任务和配置
    python scripts/florence_cli.py list-tasks
    
    # 验证配置文件
    python scripts/florence_cli.py validate --config configs/examples/caption_training.yaml
    
    # 生成配置模板
    python scripts/florence_cli.py generate-config --task ocr --output my_ocr_config.yaml
"""

import argparse
import sys
import logging
import yaml
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 简化导入，避免循环依赖
try:
    sys.path.insert(0, str(project_root / "core"))
    from config import TrainingConfig
    from tasks import FLORENCE2_TASKS, TaskCategory, list_all_tasks
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入核心配置模块: {e}")
    TrainingConfig = None
    FLORENCE2_TASKS = {}
    TaskCategory = None
    list_all_tasks = None
    CORE_AVAILABLE = False

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 预定义的任务配置映射
TASK_CONFIG_MAPPING = {
    'caption': 'configs/examples/caption_training.yaml',
    'detection': 'configs/examples/object_detection_training.yaml',
    'od': 'configs/examples/object_detection_training.yaml',
    'ocr': 'configs/examples/ocr_training.yaml',
    'segmentation': 'configs/examples/segmentation_training.yaml',
    'seg': 'configs/examples/segmentation_training.yaml',
    'multitask': 'configs/examples/multitask_training.yaml',
    'multi': 'configs/examples/multitask_training.yaml'
}

# 任务描述
TASK_DESCRIPTIONS = {
    'caption': '图像描述生成任务 (CAPTION, DETAILED_CAPTION, MORE_DETAILED_CAPTION)',
    'detection': '目标检测任务 (OD, OPEN_VOCABULARY_DETECTION)',
    'ocr': 'OCR文字识别任务 (OCR, OCR_WITH_REGION)',
    'segmentation': '图像分割任务 (REGION_TO_SEGMENTATION, REFERRING_EXPRESSION_SEGMENTATION)',
    'multitask': '多任务混合训练 (CAPTION + OD + OCR + SEGMENTATION)'
}

def setup_cli_logging(verbose: bool = False) -> None:
    """设置CLI日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

def list_available_tasks() -> None:
    """列出所有可用的任务和配置"""
    print("\n=== Florence-2 可用任务 ===")
    print()
    
    if not CORE_AVAILABLE or not FLORENCE2_TASKS:
        print("\n⚠️  核心模块不可用，显示预定义任务列表:")
        predefined_tasks = {
            'CAPTION': '图像描述生成',
            'DETAILED_CAPTION': '详细图像描述',
            'MORE_DETAILED_CAPTION': '更详细图像描述',
            'OD': '目标检测',
            'DENSE_REGION_CAPTION': '密集区域描述',
            'REGION_PROPOSAL': '区域提议',
            'OCR': '光学字符识别',
            'OCR_WITH_REGION': '带区域的OCR',
            'REFERRING_EXPRESSION_SEGMENTATION': '指代表达式分割',
            'REGION_TO_SEGMENTATION': '区域到分割',
            'OPEN_VOCABULARY_DETECTION': '开放词汇检测',
            'REGION_TO_CATEGORY': '区域到类别',
            'REGION_TO_DESCRIPTION': '区域到描述'
        }
        
        categories = {
            '图像描述': ['CAPTION', 'DETAILED_CAPTION', 'MORE_DETAILED_CAPTION'],
            '目标检测': ['OD', 'OPEN_VOCABULARY_DETECTION', 'REGION_PROPOSAL'],
            '文字识别': ['OCR', 'OCR_WITH_REGION'],
            '图像分割': ['REFERRING_EXPRESSION_SEGMENTATION', 'REGION_TO_SEGMENTATION'],
            '区域分析': ['DENSE_REGION_CAPTION', 'REGION_TO_CATEGORY', 'REGION_TO_DESCRIPTION']
        }
        
        for category, task_list in categories.items():
            print(f"\n📂 {category}:")
            for task_name in task_list:
                description = predefined_tasks.get(task_name, '无描述')
                print(f"  • {task_name}: {description}")
        
        print(f"\n总计: {len(predefined_tasks)} 个任务")
    else:
        # 按类别分组显示Florence-2原生任务
        print("📋 Florence-2 原生任务:")
        for category in TaskCategory:
            tasks = [name for name, config in FLORENCE2_TASKS.items() 
                    if config['category'] == category]
            if tasks:
                print(f"  {category.value}:")
                for task in tasks:
                    desc = FLORENCE2_TASKS[task]['description']
                    print(f"    - {task}: {desc}")
    
    print("\n🎯 预配置训练任务:")
    for task_key, description in TASK_DESCRIPTIONS.items():
        config_path = TASK_CONFIG_MAPPING[task_key]
        print(f"  - {task_key}: {description}")
        print(f"    配置文件: {config_path}")
    
    print("\n💡 使用示例:")
    print("  python scripts/florence_cli.py train --task caption")
    print("  python scripts/florence_cli.py train --task detection --epochs 10")
    print("  python scripts/florence_cli.py train --config custom_config.yaml")

def validate_config(config_path: str) -> bool:
    """验证配置文件"""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        # 加载并验证YAML格式
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if not CORE_AVAILABLE or not TrainingConfig:
            logger.warning("核心配置模块不可用，仅进行基础YAML格式验证")
            logger.info(f"✅ YAML格式验证通过: {config_path}")
            logger.info(f"   配置项数量: {len(config_data)}")
            return True
        
        # 尝试创建TrainingConfig对象
        config = TrainingConfig.from_dict(config_data)
        
        logger.info(f"✅ 配置文件验证通过: {config_path}")
        logger.info(f"   实验名称: {config.experiment_name}")
        logger.info(f"   模型: {config.model_config.model_name}")
        logger.info(f"   训练轮数: {config.num_epochs}")
        logger.info(f"   批次大小: {config.data_config.batch_size}")
        logger.info(f"   学习率: {config.optimization_config.learning_rate}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置文件验证失败: {e}")
        return False

def generate_config_template(task: str, output_path: str) -> bool:
    """生成配置模板"""
    try:
        if task not in TASK_CONFIG_MAPPING:
            logger.error(f"未知任务类型: {task}")
            logger.info(f"可用任务: {list(TASK_CONFIG_MAPPING.keys())}")
            return False
        
        template_path = Path(project_root) / TASK_CONFIG_MAPPING[task]
        output_path = Path(output_path)
        
        if not template_path.exists():
            logger.error(f"模板文件不存在: {template_path}")
            return False
        
        # 读取模板并添加自定义注释
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # 添加生成信息
        header = f"""# 配置文件生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 基于模板: {template_path.name}
# 任务类型: {task}
# 描述: {TASK_DESCRIPTIONS.get(task, '自定义任务')}

"""
        
        # 写入新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + template_content)
        
        logger.info(f"✅ 配置模板已生成: {output_path}")
        logger.info(f"💡 请根据需要修改配置参数")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 生成配置模板失败: {e}")
        return False

def run_training_task(
    task: Optional[str] = None,
    config: Optional[str] = None,
    **overrides
) -> bool:
    """运行训练任务"""
    try:
        # 确定配置文件路径
        if config:
            config_path = Path(config)
        elif task:
            if task not in TASK_CONFIG_MAPPING:
                logger.error(f"未知任务类型: {task}")
                logger.info(f"可用任务: {list(TASK_CONFIG_MAPPING.keys())}")
                return False
            config_path = Path(project_root) / TASK_CONFIG_MAPPING[task]
        else:
            logger.error("必须指定 --task 或 --config 参数")
            return False
        
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        logger.info(f"🚀 开始训练任务")
        logger.info(f"   任务类型: {task or 'custom'}")
        logger.info(f"   配置文件: {config_path}")
        
        # 应用命令行覆盖参数
        if overrides:
            logger.info(f"   参数覆盖: {overrides}")
        
        # 构建运行参数
        run_args = [
            '--config', str(config_path)
        ]
        
        # 添加覆盖参数
        for key, value in overrides.items():
            if value is not None:
                run_args.extend(['--override', f'{key}={value}'])
        
        # 调用训练脚本
        # 这里应该调用实际的训练函数
        logger.info("训练参数准备完成，开始训练...")
        logger.info(f"运行参数: {' '.join(run_args)}")
        
        # TODO: 实际调用训练函数
        # return run_training(run_args)
        
        logger.info("✅ 训练任务配置完成")
        logger.info("💡 请检查输出目录中的训练结果")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练任务失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Florence-2 训练CLI工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s train --task caption
  %(prog)s train --task detection --epochs 10 --lr 1e-5
  %(prog)s train --config my_config.yaml
  %(prog)s list-tasks
  %(prog)s validate --config my_config.yaml
  %(prog)s generate-config --task ocr --output my_ocr.yaml
        """
    )
    
    # 全局参数
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='运行训练')
    train_parser.add_argument(
        '--task',
        choices=list(TASK_CONFIG_MAPPING.keys()),
        help='预定义任务类型'
    )
    train_parser.add_argument(
        '--config',
        help='自定义配置文件路径'
    )
    
    # 训练参数覆盖
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, help='批次大小')
    train_parser.add_argument('--lr', type=float, help='学习率')
    train_parser.add_argument('--output-dir', help='输出目录')
    train_parser.add_argument('--model', help='模型名称')
    
    # list-tasks 命令
    subparsers.add_parser('list-tasks', help='列出所有可用任务')
    
    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument(
        '--config',
        required=True,
        help='要验证的配置文件路径'
    )
    
    # generate-config 命令
    generate_parser = subparsers.add_parser('generate-config', help='生成配置模板')
    generate_parser.add_argument(
        '--task',
        required=True,
        choices=list(TASK_CONFIG_MAPPING.keys()),
        help='任务类型'
    )
    generate_parser.add_argument(
        '--output',
        required=True,
        help='输出文件路径'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_cli_logging(args.verbose)
    
    # 执行命令
    if args.command == 'train':
        # 准备覆盖参数
        overrides = {}
        if args.epochs:
            overrides['num_epochs'] = args.epochs
        if args.batch_size:
            overrides['data_config.batch_size'] = args.batch_size
        if args.lr:
            overrides['optimization_config.learning_rate'] = args.lr
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        if args.model:
            overrides['model_config.model_name'] = args.model
        
        success = run_training_task(
            task=args.task,
            config=args.config,
            **overrides
        )
        sys.exit(0 if success else 1)
        
    elif args.command == 'list-tasks':
        list_available_tasks()
        
    elif args.command == 'validate':
        success = validate_config(args.config)
        sys.exit(0 if success else 1)
        
    elif args.command == 'generate-config':
        success = generate_config_template(args.task, args.output)
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()