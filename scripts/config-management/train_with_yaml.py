#!/usr/bin/env python3
"""
使用YAML配置文件进行Florence-2多任务训练

这个脚本允许用户使用统一的YAML配置文件来配置和启动多数据集、多任务的Florence-2训练。

使用方法:
    python train_with_yaml.py config.yaml
    python train_with_yaml.py config.yaml --dry-run  # 仅验证配置，不开始训练
    python train_with_yaml.py config.yaml --resume   # 从检查点恢复训练
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from florence_forge.core.yaml_config import FlorenceForgeYAMLConfig, validate_yaml_config
from florence_forge.training.trainer import MultiTaskTrainer
from florence_forge.data.multi_dataset_manager import MultiDatasetManager
from florence_forge.core.config import TrainingConfig
from florence_forge.utils.logging import setup_logging

def setup_training_logger(output_dir: str, experiment_name: str) -> logging.Logger:
    """设置训练日志记录器
    
    Args:
        output_dir: 输出目录
        experiment_name: 实验名称
        
    Returns:
        配置好的日志记录器
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}.log"
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 配置根日志记录器
    logger = logging.getLogger("florence_forge")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_and_load_config(config_path: str) -> FlorenceForgeYAMLConfig:
    """验证并加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        加载的配置对象
        
    Raises:
        ValueError: 配置验证失败
        FileNotFoundError: 配置文件不存在
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"正在验证配置文件: {config_path}")
    if not validate_yaml_config(str(config_path)):
        raise ValueError("配置文件验证失败")
    
    print("✓ 配置文件验证通过")
    
    # 加载配置
    yaml_config = FlorenceForgeYAMLConfig.load_from_file(str(config_path))
    print(f"✓ 配置文件加载成功: {yaml_config.project_name}")
    
    return yaml_config

def prepare_training_components(yaml_config: FlorenceForgeYAMLConfig) -> tuple[TrainingConfig, MultiDatasetManager]:
    """准备训练组件
    
    Args:
        yaml_config: YAML配置对象
        
    Returns:
        训练配置和多数据集管理器的元组
    """
    print("正在准备训练组件...")
    
    # 转换为训练配置
    training_config = yaml_config.to_training_config()
    print(f"✓ 训练配置已准备: {training_config.experiment_name}")
    
    # 创建多数据集管理器
    dataset_manager = yaml_config.to_multi_dataset_manager()
    print(f"✓ 数据集管理器已准备: {len(dataset_manager.datasets)} 个数据集")
    
    # 验证数据集路径
    print("正在验证数据集路径...")
    for dataset_info in dataset_manager.datasets.values():
        dataset_path = Path(dataset_info.path)
        if not dataset_path.exists():
            print(f"⚠️  警告: 数据集路径不存在: {dataset_path}")
        else:
            print(f"✓ 数据集路径有效: {dataset_info.name} -> {dataset_path}")
    
    return training_config, dataset_manager

def run_training(yaml_config: FlorenceForgeYAMLConfig, resume: bool = False, dry_run: bool = False) -> None:
    """运行训练
    
    Args:
        yaml_config: YAML配置对象
        resume: 是否从检查点恢复训练
        dry_run: 是否仅进行配置验证而不实际训练
    """
    # 准备训练组件
    training_config, dataset_manager = prepare_training_components(yaml_config)
    
    # 设置日志记录
    logger = setup_training_logger(
        training_config.output_dir, 
        training_config.experiment_name or "florence2_yaml_training"
    )
    
    if dry_run:
        print("\n=== 配置验证完成 (干运行模式) ===")
        print(f"实验名称: {training_config.experiment_name}")
        print(f"模型: {training_config.model_config.model_name}")
        print(f"训练轮数: {training_config.num_epochs}")
        print(f"批次大小: {training_config.data_config.batch_size}")
        print(f"学习率: {training_config.optimization_config.learning_rate}")
        print(f"输出目录: {training_config.output_dir}")
        print(f"数据集数量: {len(dataset_manager.datasets)}")
        print(f"任务映射数量: {len(dataset_manager.task_mappings)}")
        print("\n配置验证成功！可以开始正式训练。")
        return
    
    try:
        print("\n=== 开始训练 ===")
        logger.info(f"开始训练实验: {training_config.experiment_name}")
        logger.info(f"使用配置: {yaml_config.project_name}")
        
        # 创建训练器
        trainer = MultiTaskTrainer(
            config=training_config,
            dataset_manager=dataset_manager
        )
        
        # 开始训练
        if resume:
            logger.info("从检查点恢复训练")
            trainer.resume_training()
        else:
            logger.info("开始新的训练")
            trainer.train()
        
        logger.info("训练完成")
        print("\n✓ 训练完成！")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"\n✗ 训练失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用YAML配置文件进行Florence-2多任务训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s config.yaml                    # 开始训练
  %(prog)s config.yaml --dry-run          # 仅验证配置
  %(prog)s config.yaml --resume           # 从检查点恢复训练
  %(prog)s config.yaml --dry-run --verbose # 详细验证信息
        """
    )
    
    parser.add_argument(
        'config', 
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='仅验证配置，不开始训练'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='从检查点恢复训练'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 验证并加载配置
        yaml_config = validate_and_load_config(args.config)
        
        # 运行训练
        run_training(
            yaml_config=yaml_config,
            resume=args.resume,
            dry_run=args.dry_run
        )
        
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()