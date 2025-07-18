# 配置管理脚本

本目录包含用于配置文件管理、YAML 配置处理和训练配置的脚本工具。

## 脚本列表

### advanced_config_manager.py
高级配置管理工具，提供复杂配置场景的管理功能。

**功能特性：**
- 多环境配置管理
- 配置继承和覆盖
- 配置模板生成
- 配置验证和校验
- 动态配置更新
- 配置版本控制

**使用示例：**
```bash
# 生成配置模板
python advanced_config_manager.py --generate-template --task object_detection

# 验证配置文件
python advanced_config_manager.py --validate config.yaml

# 合并多个配置
python advanced_config_manager.py --merge base.yaml custom.yaml --output merged.yaml

# 环境特定配置
python advanced_config_manager.py --environment production --config base.yaml
```

### run_with_yaml_config.py
使用 YAML 配置文件运行训练和推理的脚本。

**功能特性：**
- YAML 配置解析
- 参数验证和类型检查
- 配置覆盖支持
- 运行时配置修改
- 配置日志记录

**使用示例：**
```bash
# 使用配置文件运行
python run_with_yaml_config.py --config training_config.yaml

# 覆盖特定参数
python run_with_yaml_config.py --config config.yaml --override learning_rate=0.001

# 指定输出目录
python run_with_yaml_config.py --config config.yaml --output-dir ./outputs/experiment1

# 干运行模式（验证配置）
python run_with_yaml_config.py --config config.yaml --dry-run
```

### train_with_yaml.py
专门用于训练任务的 YAML 配置脚本。

**功能特性：**
- 训练配置管理
- 多任务训练支持
- 分布式训练配置
- 检查点管理
- 训练监控集成

**使用示例：**
```bash
# 开始训练
python train_with_yaml.py --config train_config.yaml

# 从检查点恢复训练
python train_with_yaml.py --config train_config.yaml --resume checkpoint.pth

# 分布式训练
python train_with_yaml.py --config train_config.yaml --distributed --world-size 4

# 多任务训练
python train_with_yaml.py --config multitask_config.yaml --tasks detection,caption
```

## 配置文件类型

### 基础训练配置

#### 单任务配置示例
```yaml
# single_task_config.yaml
model:
  name: "florence-2-base"
  pretrained: true
  checkpoint: null

data:
  dataset_name: "coco"
  data_dir: "./data/coco"
  batch_size: 16
  num_workers: 4
  
training:
  epochs: 10
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler: "cosine"
  
output:
  save_dir: "./outputs"
  experiment_name: "florence2_coco"
```

#### 多任务配置示例
```yaml
# multitask_config.yaml
model:
  name: "florence-2-large"
  tasks:
    - "object_detection"
    - "image_captioning"
    - "visual_grounding"

data:
  datasets:
    object_detection:
      name: "coco"
      data_dir: "./data/coco"
      weight: 1.0
    image_captioning:
      name: "coco_captions"
      data_dir: "./data/coco_captions"
      weight: 0.5
  batch_size: 8
  
training:
  epochs: 20
  task_sampling: "proportional"
```

### 高级配置

#### 分布式训练配置
```yaml
# distributed_config.yaml
distributed:
  enabled: true
  backend: "nccl"
  world_size: 4
  rank: 0
  master_addr: "localhost"
  master_port: "12355"
  
model:
  sync_bn: true
  find_unused_parameters: false
  
training:
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
```

#### 优化配置
```yaml
# optimization_config.yaml
optimization:
  mixed_precision:
    enabled: true
    opt_level: "O1"
  
  gradient_checkpointing: true
  
  dataloader:
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true
```

## 配置管理功能

### 配置验证

#### 基本验证
```bash
# 验证配置文件语法
python advanced_config_manager.py --validate config.yaml

# 深度验证（包括数据路径等）
python advanced_config_manager.py --validate config.yaml --deep
```

#### 自定义验证规则
```python
# 示例：自定义验证器
from florence_forge.config import ConfigValidator

class CustomValidator(ConfigValidator):
    def validate_model_config(self, config):
        # 实现自定义验证逻辑
        pass
```

### 配置模板

#### 生成标准模板
```bash
# 生成目标检测模板
python advanced_config_manager.py --generate-template \
  --task object_detection \
  --output od_template.yaml

# 生成图像描述模板
python advanced_config_manager.py --generate-template \
  --task image_captioning \
  --output caption_template.yaml
```

#### 自定义模板
```bash
# 基于现有配置生成模板
python advanced_config_manager.py --create-template \
  --from-config existing_config.yaml \
  --output custom_template.yaml
```

### 配置继承

#### 基础配置继承
```yaml
# base_config.yaml
base:
  model:
    name: "florence-2-base"
  training:
    optimizer: "adamw"
    learning_rate: 1e-4

# child_config.yaml
inherits: "base_config.yaml"
overrides:
  training:
    learning_rate: 5e-5  # 覆盖学习率
  data:
    batch_size: 32       # 添加新配置
```

```bash
# 处理配置继承
python advanced_config_manager.py --resolve-inheritance child_config.yaml
```

### 环境配置

#### 多环境支持
```yaml
# config.yaml
default: &default
  model:
    name: "florence-2-base"
  training:
    epochs: 10

development:
  <<: *default
  training:
    epochs: 2  # 开发环境快速训练
    
production:
  <<: *default
  model:
    name: "florence-2-large"  # 生产环境使用大模型
  training:
    epochs: 50
```

```bash
# 指定环境
python run_with_yaml_config.py --config config.yaml --env production
```

## 配置覆盖

### 命令行覆盖

#### 简单参数覆盖
```bash
# 覆盖学习率
python run_with_yaml_config.py --config config.yaml \
  --override learning_rate=0.001

# 覆盖多个参数
python run_with_yaml_config.py --config config.yaml \
  --override learning_rate=0.001,batch_size=32,epochs=20
```

#### 嵌套参数覆盖
```bash
# 覆盖嵌套配置
python run_with_yaml_config.py --config config.yaml \
  --override training.learning_rate=0.001,model.name=florence-2-large
```

### 配置文件覆盖

#### 部分覆盖文件
```yaml
# override.yaml
training:
  learning_rate: 0.001
  batch_size: 32
model:
  checkpoint: "./checkpoints/best.pth"
```

```bash
# 使用覆盖文件
python run_with_yaml_config.py --config base_config.yaml \
  --override-file override.yaml
```

## 配置版本控制

### 配置快照

#### 保存配置快照
```bash
# 保存当前配置状态
python advanced_config_manager.py --snapshot \
  --config config.yaml \
  --output snapshots/config_v1.0.yaml
```

#### 配置差异对比
```bash
# 对比两个配置文件
python advanced_config_manager.py --diff \
  config_v1.yaml config_v2.yaml
```

### 配置历史

#### 配置变更追踪
```bash
# 启用配置变更追踪
python run_with_yaml_config.py --config config.yaml \
  --track-changes \
  --change-log changes.log
```

## 动态配置

### 运行时配置更新

#### 热更新配置
```python
# 示例：运行时配置更新
from florence_forge.config import DynamicConfig

config = DynamicConfig("config.yaml")
config.update("training.learning_rate", 0.0005)
config.save("updated_config.yaml")
```

#### 配置监听
```bash
# 监听配置文件变化
python advanced_config_manager.py --watch config.yaml \
  --on-change "restart_training"
```

## 配置优化

### 自动配置调优

#### 超参数搜索
```yaml
# hyperparameter_search.yaml
search_space:
  learning_rate:
    type: "log_uniform"
    low: 1e-5
    high: 1e-2
  batch_size:
    type: "choice"
    choices: [8, 16, 32, 64]
  
search:
  algorithm: "optuna"
  trials: 100
  objective: "validation_loss"
```

```bash
# 运行超参数搜索
python advanced_config_manager.py --hyperparameter-search \
  --search-config hyperparameter_search.yaml \
  --base-config base_config.yaml
```

### 配置推荐

#### 基于硬件的配置推荐
```bash
# 根据硬件推荐配置
python advanced_config_manager.py --recommend-config \
  --hardware-info \
  --task object_detection
```

#### 基于数据集的配置推荐
```bash
# 根据数据集特征推荐配置
python advanced_config_manager.py --recommend-config \
  --dataset-path ./data/custom_dataset \
  --analyze-dataset
```

## 配置安全

### 敏感信息管理

#### 环境变量集成
```yaml
# secure_config.yaml
api:
  key: "${API_KEY}"  # 从环境变量读取
  secret: "${API_SECRET}"
  
database:
  password: "${DB_PASSWORD}"
```

#### 配置加密
```bash
# 加密敏感配置
python advanced_config_manager.py --encrypt \
  --config sensitive_config.yaml \
  --key-file encryption.key

# 解密配置
python advanced_config_manager.py --decrypt \
  --config encrypted_config.yaml \
  --key-file encryption.key
```

## 配置文档

### 自动文档生成

#### 配置模式文档
```bash
# 生成配置文档
python advanced_config_manager.py --generate-docs \
  --config-schema schema.yaml \
  --output config_docs.md
```

#### 配置示例生成
```bash
# 生成配置示例
python advanced_config_manager.py --generate-examples \
  --task all \
  --output examples/
```

## 故障排除

### 常见配置问题

1. **YAML 语法错误**
   ```bash
   # 检查 YAML 语法
   python advanced_config_manager.py --check-syntax config.yaml
   ```

2. **配置路径错误**
   ```bash
   # 验证文件路径
   python advanced_config_manager.py --check-paths config.yaml
   ```

3. **配置类型错误**
   ```bash
   # 类型检查
   python advanced_config_manager.py --type-check config.yaml
   ```

### 调试工具

#### 配置调试模式
```bash
# 启用详细调试
python run_with_yaml_config.py --config config.yaml \
  --debug-config \
  --verbose
```

#### 配置可视化
```bash
# 可视化配置结构
python advanced_config_manager.py --visualize config.yaml \
  --output config_structure.png
```

## 最佳实践

### 配置组织
1. **模块化配置**：将配置分解为可重用的模块
2. **环境分离**：为不同环境维护独立配置
3. **版本控制**：跟踪配置变更历史
4. **文档化**：为配置选项提供清晰文档

### 配置安全
1. **敏感信息**：使用环境变量或加密存储
2. **访问控制**：限制配置文件访问权限
3. **审计日志**：记录配置变更操作
4. **备份恢复**：定期备份重要配置

### 配置维护
1. **定期审查**：定期检查和清理过时配置
2. **自动化测试**：为配置变更添加测试
3. **监控告警**：监控配置相关的运行时错误
4. **文档同步**：保持配置文档与代码同步

## 相关文档

- [YAML 配置指南](../../docs/configuration/YAML_CONFIG_GUIDE.md)
- [配置参数说明](../../docs/configuration/config_guide.md)
- [用户指南](../../docs/user-guides/)
- [API 参考](../../docs/reference/)

## 扩展开发

### 自定义配置处理器
```python
# 示例：自定义配置处理器
from florence_forge.config import BaseConfigProcessor

class CustomConfigProcessor(BaseConfigProcessor):
    def process(self, config):
        # 实现自定义处理逻辑
        return processed_config
```

### 配置插件
```python
# 示例：配置验证插件
from florence_forge.config.plugins import ValidationPlugin

class CustomValidationPlugin(ValidationPlugin):
    def validate(self, config):
        # 实现自定义验证逻辑
        return validation_result
```

## 注意事项

- 配置文件应使用版本控制管理
- 敏感信息不应直接写入配置文件
- 大型配置文件可能影响启动性能
- 配置继承层次不宜过深
- 定期验证配置文件的有效性
- 保持配置文档与实际配置同步