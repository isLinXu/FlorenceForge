#!/usr/bin/env python3
"""
配置使用示例 - 展示如何在代码中使用YAML配置文件

本脚本演示了:
1. 加载YAML配置文件
2. 修改配置参数
3. 保存配置文件
4. 配置验证
5. 配置合并
"""

import sys
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from florence_forge.core import (
    TrainingConfig, ModelConfig, DataConfig, 
)

def example_1_load_and_modify_config():
    """示例1: 加载并修改配置"""
    print("\n=== 示例1: 加载并修改配置 ===")
    
    # 加载配置文件
    config_path = project_root / "configs" / "quick_start.yaml"
    config = TrainingConfig.load_from_file(config_path)
    
    print(f"原始学习率: {config.optimization_config.learning_rate}")
    print(f"原始批次大小: {config.data_config.batch_size}")
    
    # 修改配置
    config.optimization_config.learning_rate = 1e-5
    config.data_config.batch_size = 4
    config.num_epochs = 5
    config.experiment_name = "modified_experiment"
    config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"修改后学习率: {config.optimization_config.learning_rate}")
    print(f"修改后批次大小: {config.data_config.batch_size}")
    
    # 保存修改后的配置
    output_path = project_root / "configs" / "modified_config.yaml"
    config.save_to_yaml(output_path)
    print(f"修改后的配置已保存到: {output_path}")

def example_2_create_custom_config():
    """示例2: 创建自定义配置"""
    print("\n=== 示例2: 创建自定义配置 ===")
    
    # 创建自定义LoRA配置
    custom_lora = LoRAConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.1
    )
    
    # 创建自定义模型配置
    custom_model = ModelConfig(
        model_name="microsoft/Florence-2-base",
        use_lora=True,
        lora_config=custom_lora
    )
    
    # 创建自定义数据配置
    custom_data = DataConfig(
        batch_size=6,
        num_workers=4,
        use_augmentation=True,
        augmentation_prob=0.3
    )
    
    # 创建自定义优化配置
    custom_optimization = OptimizationConfig(
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    # 创建完整的训练配置
    custom_config = TrainingConfig(
        num_epochs=8,
        eval_steps=150,
        save_steps=300,
        output_dir="./outputs/custom_experiment",
        experiment_name="custom_florence2_experiment",
        run_name="custom_run_v1",
        tags=["custom", "experiment", "florence2"],
        model_config=custom_model,
        data_config=custom_data,
        optimization_config=custom_optimization
    )
    
    # 保存自定义配置
    output_path = project_root / "configs" / "custom_config.yaml"
    custom_config.save_to_yaml(output_path)
    print(f"自定义配置已保存到: {output_path}")
    
    # 显示配置信息
    print(f"实验名称: {custom_config.experiment_name}")
    print(f"模型: {custom_config.model_config.model_name}")
    print(f"LoRA Rank: {custom_config.model_config.lora_config.r}")
    print(f"学习率: {custom_config.optimization_config.learning_rate}")

def example_3_config_validation():
    """示例3: 配置验证"""
    print("\n=== 示例3: 配置验证 ===")
    
    # 创建一个有问题的配置
    invalid_config = TrainingConfig(
        num_epochs=0,  # 无效值
        data_config=DataConfig(batch_size=-1),  # 无效值
        optimization_config=OptimizationConfig(learning_rate=0)  # 无效值
    )
    
    # 保存无效配置
    invalid_path = project_root / "configs" / "invalid_config.yaml"
    invalid_config.save_to_yaml(invalid_path)
    
    # 验证配置
    def validate_config(config: TrainingConfig) -> bool:
        """简单的配置验证函数"""
        errors = []
        
        if config.num_epochs <= 0:
            errors.append("num_epochs 必须大于 0")
        
        if config.data_config.batch_size <= 0:
            errors.append("batch_size 必须大于 0")
        
        if config.optimization_config.learning_rate <= 0:
            errors.append("learning_rate 必须大于 0")
        
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("配置验证通过")
        return True
    
    # 验证无效配置
    print("验证无效配置:")
    validate_config(invalid_config)
    
    # 验证有效配置
    print("\n验证有效配置:")
    valid_config = TrainingConfig()
    validate_config(valid_config)

def example_4_config_conversion():
    """示例4: 配置格式转换"""
    print("\n=== 示例4: 配置格式转换 ===")
    
    # 加载YAML配置
    yaml_path = project_root / "configs" / "quick_start.yaml"
    config = TrainingConfig.load_from_yaml(yaml_path)
    
    # 保存为JSON格式
    json_path = project_root / "configs" / "quick_start.json"
    config.save_to_json(json_path)
    print(f"YAML配置已转换为JSON: {json_path}")
    
    # 重新加载JSON配置验证
    config_from_json = TrainingConfig.load_from_json(json_path)
    print(f"从JSON加载的配置实验名称: {config_from_json.experiment_name}")
    
    # 验证两个配置是否相同
    yaml_dict = config.to_dict()
    json_dict = config_from_json.to_dict()
    
    # 移除时间戳进行比较
    yaml_dict.pop('_metadata', None)
    json_dict.pop('_metadata', None)
    
    if yaml_dict == json_dict:
        print("✓ YAML和JSON配置内容一致")
    else:
        print("✗ YAML和JSON配置内容不一致")

def example_5_config_inheritance():
    """示例5: 配置继承和覆盖"""
    print("\n=== 示例5: 配置继承和覆盖 ===")
    
    # 加载基础配置
    base_config = TrainingConfig.load_from_file(
        project_root / "configs" / "quick_start.yaml"
    )
    
    # 创建一个覆盖配置（只包含需要修改的部分）
    override_dict = {
        "num_epochs": 10,
        "experiment_name": "inherited_experiment",
        "optimization_config": {
            "learning_rate": 5e-6,
            "lr_scheduler_type": "cosine"
        },
        "model_config": {
            "model_name": "microsoft/Florence-2-large",
            "lora_config": {
                "r": 64
            }
        }
    }
    
    # 手动合并配置
    base_dict = base_config.to_dict()
    
    def deep_merge(base, override):
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_dict)
    merged_config = TrainingConfig.from_dict(merged_dict)
    
    # 保存合并后的配置
    output_path = project_root / "configs" / "inherited_config.yaml"
    merged_config.save_to_yaml(output_path)
    
    print(f"基础配置学习率: {base_config.optimization_config.learning_rate}")
    print(f"合并后学习率: {merged_config.optimization_config.learning_rate}")
    print(f"基础配置LoRA rank: {base_config.model_config.lora_config.r}")
    print(f"合并后LoRA rank: {merged_config.model_config.lora_config.r}")
    print(f"合并后的配置已保存到: {output_path}")

def main():
    """主函数 - 运行所有示例"""
    print("Florence-2 配置使用示例")
    print("=" * 50)
    
    try:
        example_1_load_and_modify_config()
        example_2_create_custom_config()
        example_3_config_validation()
        example_4_config_conversion()
        example_5_config_inheritance()
        
        print("\n=== 所有示例运行完成 ===")
        print("\n生成的配置文件:")
        configs_dir = project_root / "configs"
        for config_file in configs_dir.glob("*.yaml"):
            if config_file.name not in ["training_config_sample.yaml", "quick_start.yaml", "production.yaml"]:
                print(f"  - {config_file}")
        
        for config_file in configs_dir.glob("*.json"):
            print(f"  - {config_file}")
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()