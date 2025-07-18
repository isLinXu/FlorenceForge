#!/usr/bin/env python3
"""
使用YAML配置运行训练的示例脚本

这个脚本展示了如何:
1. 加载YAML配置文件
2. 根据配置初始化训练组件
3. 运行训练流程

使用方法:
    python scripts/run_with_yaml_config.py --config configs/quick_start.yaml
    python scripts/run_with_yaml_config.py --config configs/production.yaml --override num_epochs=5
"""

import argparse
import sys
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(config: TrainingConfig) -> logging.Logger:
    """设置日志系统
    
    Args:
        config: 训练配置
        
    Returns:
        配置好的logger
    """
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / 'training.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def validate_config(config: TrainingConfig, logger: logging.Logger) -> bool:
    """验证配置参数
    
    Args:
        config: 训练配置
        logger: 日志记录器
        
    Returns:
        验证是否通过
    """
    errors = []
    warnings = []
    
    # 基本参数检查
    if config.num_epochs <= 0:
        errors.append("num_epochs 必须大于 0")
    
    if config.data_config.batch_size <= 0:
        errors.append("batch_size 必须大于 0")
    
    if config.optimization_config.learning_rate <= 0:
        errors.append("learning_rate 必须大于 0")
    
    # LoRA配置检查
    if config.model_config.use_lora:
        if config.model_config.lora_config.r <= 0:
            errors.append("LoRA rank (r) 必须大于 0")
        
        if config.model_config.lora_config.r > 256:
            warnings.append(f"LoRA rank ({config.model_config.lora_config.r}) 较大，可能影响性能")
    
    # 学习率合理性检查
    lr = config.optimization_config.learning_rate
    if lr > 1e-3:
        warnings.append(f"学习率 ({lr}) 可能过高")
    elif lr < 1e-7:
        warnings.append(f"学习率 ({lr}) 可能过低")
    
    # 批次大小检查
    if config.data_config.batch_size > 32:
        warnings.append(f"批次大小 ({config.data_config.batch_size}) 较大，请确保有足够显存")
    
    # 输出验证结果
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False
    
    if warnings:
        logger.warning("配置警告:")
        for warning in warnings:
            logger.warning(f"  ⚠ {warning}")
    
    logger.info("✓ 配置验证通过")
    return True

def print_config_summary(config: TrainingConfig, logger: logging.Logger) -> None:
    """打印配置摘要
    
    Args:
        config: 训练配置
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("训练配置摘要")
    logger.info("=" * 60)
    
    # 实验信息
    logger.info(f"实验名称: {config.experiment_name or 'N/A'}")
    logger.info(f"运行名称: {config.run_name or 'N/A'}")
    logger.info(f"标签: {', '.join(config.tags) if config.tags else 'N/A'}")
    
    # 模型配置
    logger.info(f"模型: {config.model_config.model_name}")
    logger.info(f"使用LoRA: {config.model_config.use_lora}")
    if config.model_config.use_lora:
        lora = config.model_config.lora_config
        logger.info(f"  - LoRA Rank: {lora.r}")
        logger.info(f"  - LoRA Alpha: {lora.lora_alpha}")
        logger.info(f"  - LoRA Dropout: {lora.lora_dropout}")
        logger.info(f"  - 目标模块: {', '.join(lora.target_modules)}")
    
    # 训练参数
    logger.info(f"训练轮数: {config.num_epochs}")
    logger.info(f"批次大小: {config.data_config.batch_size}")
    logger.info(f"学习率: {config.optimization_config.learning_rate}")
    logger.info(f"权重衰减: {config.optimization_config.weight_decay}")
    logger.info(f"学习率调度: {config.optimization_config.lr_scheduler_type}")
    
    # 训练设置
    logger.info(f"混合精度: FP16={config.use_fp16}, BF16={config.use_bf16}")
    logger.info(f"梯度累积步数: {config.gradient_accumulation_steps}")
    logger.info(f"最大梯度范数: {config.optimization_config.max_grad_norm}")
    
    # 评估和保存
    logger.info(f"评估步数: {config.eval_steps}")
    logger.info(f"保存步数: {config.save_steps}")
    logger.info(f"早停耐心: {config.early_stopping_patience}")
    
    # 输出目录
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"日志目录: {config.logging_dir}")
    
    logger.info("=" * 60)

def apply_overrides(
    config: TrainingConfig,
    overrides: Dict[str,
    Any],
    logger: logging.Logger
) -> None:
    """应用命令行覆盖参数
    
    Args:
        config: 训练配置
        overrides: 覆盖参数字典
        logger: 日志记录器
    """
    if not overrides:
        return
    
    logger.info("应用命令行覆盖参数:")
    
    for key, value in overrides.items():
        # 支持嵌套参数，如 optimization_config.learning_rate
        if '.' in key:
            parts = key.split('.')
            obj = config
            
            # 导航到目标对象
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    logger.error(f"无效的配置路径: {key}")
                    continue
            
            # 设置最终属性
            final_attr = parts[-1]
            if hasattr(obj, final_attr):
                old_value = getattr(obj, final_attr)
                # 尝试保持原始类型
                if isinstance(old_value, bool):
                    new_value = str(value).lower() in ('true', '1', 'yes')
                elif isinstance(old_value, int):
                    new_value = int(value)
                elif isinstance(old_value, float):
                    new_value = float(value)
                else:
                    new_value = value
                
                setattr(obj, final_attr, new_value)
                logger.info(f"  {key}: {old_value} -> {new_value}")
            else:
                logger.error(f"无效的配置属性: {key}")
        else:
            # 顶级属性
            if hasattr(config, key):
                old_value = getattr(config, key)
                # 类型转换
                if isinstance(old_value, bool):
                    new_value = str(value).lower() in ('true', '1', 'yes')
                elif isinstance(old_value, int):
                    new_value = int(value)
                elif isinstance(old_value, float):
                    new_value = float(value)
                else:
                    new_value = value
                
                setattr(config, key, new_value)
                logger.info(f"  {key}: {old_value} -> {new_value}")
            else:
                logger.error(f"无效的配置属性: {key}")

def simulate_training(config: TrainingConfig, logger: logging.Logger) -> None:
    """模拟训练过程（实际项目中替换为真实训练代码）
    
    Args:
        config: 训练配置
        logger: 日志记录器
    """
    logger.info("开始模拟训练...")
    
    # 这里应该是实际的训练代码
    # 例如:
    # 1. 初始化模型
    # 2. 加载数据
    # 3. 设置优化器
    # 4. 训练循环
    
    import time
    
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        
        # 模拟训练步骤
        for step in range(1, 6):  # 模拟5个步骤
            time.sleep(0.1)  # 模拟计算时间
            
            if step % config.logging_steps == 0 or step == 1:
                # 模拟损失值
                loss = 2.5 - (epoch - 1) * 0.2 - step * 0.05
                logger.info(f"  Step {step}: loss = {loss:.4f}")
        
        # 模拟评估
        if epoch % 2 == 0:  # 每2个epoch评估一次
            eval_loss = 2.0 - (epoch - 1) * 0.15
            logger.info(f"  Evaluation: eval_loss = {eval_loss:.4f}")
        
        logger.info(f"  Epoch {epoch} 完成")
    
    logger.info("训练完成！")
    logger.info(f"模型和日志已保存到: {config.output_dir}")

def parse_overrides(override_strings: list) -> Dict[str, Any]:
    """解析命令行覆盖参数
    
    Args:
        override_strings: 覆盖参数字符串列表，格式为 "key=value"
        
    Returns:
        解析后的参数字典
    """
    overrides = {}
    
    for override_str in override_strings:
        if '=' not in override_str:
            print(f"警告: 忽略无效的覆盖参数格式: {override_str}")
            continue
        
        key, value = override_str.split('=', 1)
        overrides[key.strip()] = value.strip()
    
    return overrides

def main():
    """TODO: Add documentation for main"""
    parser = argparse.ArgumentParser(
        description="使用YAML配置运行Florence-2训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/run_with_yaml_config.py --config configs/quick_start.yaml
  python scripts/run_with_yaml_config.py --config configs/production.yaml --override num_epochs=5
  python scripts/run_with_yaml_config.py --config configs/quick_start.yaml --override optimization_config.learning_rate=1e-5 data_config.batch_size=8
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--override', '-o',
        nargs='*',
        default=[],
        help='覆盖配置参数，格式: key=value（支持嵌套，如 optimization_config.learning_rate=1e-5）'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证配置，不运行训练'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='保存最终配置到指定文件'
    )
    
    args = parser.parse_args()
    
    try:
        # 检查配置文件是否存在
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误: 配置文件不存在: {config_path}")
            sys.exit(1)
        
        # 加载配置
        print(f"加载配置文件: {config_path}")
        config = TrainingConfig.load_from_file(config_path)
        
        # 设置日志
        logger = setup_logging(config)
        logger.info(f"成功加载配置文件: {config_path}")
        
        # 应用命令行覆盖
        if args.override:
            overrides = parse_overrides(args.override)
            apply_overrides(config, overrides, logger)
        
        # 验证配置
        if not validate_config(config, logger):
            logger.error("配置验证失败，退出")
            sys.exit(1)
        
        # 打印配置摘要
        print_config_summary(config, logger)
        
        # 保存最终配置（如果指定）
        if args.save_config:
            save_path = Path(args.save_config)
            config.save_to_file(save_path)
            logger.info(f"最终配置已保存到: {save_path}")
        
        # 如果只是验证，则退出
        if args.validate_only:
            logger.info("配置验证完成，退出")
            return
        
        # 运行训练（这里是模拟）
        simulate_training(config, logger)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()