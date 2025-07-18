# Florence-2 训练配置指南

本目录包含了Florence-2模型各种任务的训练配置文件和管理工具。

## 📁 目录结构

```
configs/
├── README.md                          # 本文档
├── training_config_sample.yaml         # 原始示例配置
├── quick_start.yaml                    # 快速开始配置
├── examples/                           # 任务示例配置
│   ├── caption_training.yaml           # 图像描述任务
│   ├── object_detection_training.yaml  # 目标检测任务
│   ├── ocr_training.yaml              # OCR文字识别任务
│   ├── segmentation_training.yaml     # 图像分割任务
│   └── multitask_training.yaml        # 多任务混合训练
└── templates/                          # 配置模板（可选）
```

## 🚀 快速开始

### 1. 使用CLI工具运行训练

```bash
# 运行图像描述任务训练
python scripts/florence_cli.py train caption

# 运行目标检测任务训练
python scripts/florence_cli.py train detection

# 运行OCR任务训练
python scripts/florence_cli.py train ocr

# 运行图像分割任务训练
python scripts/florence_cli.py train segmentation

# 运行多任务混合训练
python scripts/florence_cli.py train multitask

# 使用自定义配置文件
python scripts/florence_cli.py train --config path/to/your/config.yaml
```

### 2. 列出可用任务

```bash
python scripts/florence_cli.py list-tasks
```

### 3. 验证配置文件

```bash
python scripts/florence_cli.py validate --config configs/examples/caption_training.yaml
```

## 📋 任务配置详解

### 图像描述任务 (Caption)

**配置文件**: `examples/caption_training.yaml`

**支持的任务类型**:
- `CAPTION`: 基础图像描述
- `DETAILED_CAPTION`: 详细图像描述
- `MORE_DETAILED_CAPTION`: 更详细的图像描述

**数据格式要求**:
```json
{
    "image_path": "path/to/image.jpg",
    "caption": "图像的描述文本"
}
```

**关键配置参数**:
- `batch_size`: 4 (推荐)
- `learning_rate`: 1e-5
- `num_epochs`: 10
- `lora_r`: 32

### 目标检测任务 (Object Detection)

**配置文件**: `examples/object_detection_training.yaml`

**支持的任务类型**:
- `OD`: 标准目标检测
- `OPEN_VOCABULARY_DETECTION`: 开放词汇检测

**数据格式要求**:
```json
{
    "image_path": "path/to/image.jpg",
    "objects": [
        {
            "bbox": [x1, y1, x2, y2],
            "label": "object_class"
        }
    ]
}
```

**关键配置参数**:
- `batch_size`: 2 (检测任务内存需求较大)
- `learning_rate`: 5e-6
- `num_epochs`: 20
- `lora_r`: 64

### OCR文字识别任务

**配置文件**: `examples/ocr_training.yaml`

**支持的任务类型**:
- `OCR`: 基础OCR识别
- `OCR_WITH_REGION`: 带区域的OCR识别

**数据格式要求**:
```json
{
    "image_path": "path/to/image.jpg",
    "text": "图像中的文字内容",
    "regions": [
        {
            "bbox": [x1, y1, x2, y2],
            "text": "区域文字"
        }
    ]
}
```

**关键配置参数**:
- `batch_size`: 6
- `learning_rate`: 1e-5
- `num_epochs`: 15
- `lora_r`: 32

### 图像分割任务

**配置文件**: `examples/segmentation_training.yaml`

**支持的任务类型**:
- `REGION_TO_SEGMENTATION`: 区域分割
- `REFERRING_EXPRESSION_SEGMENTATION`: 指代表达式分割

**数据格式要求**:
```json
{
    "image_path": "path/to/image.jpg",
    "mask_path": "path/to/mask.png",
    "expression": "分割目标的描述"
}
```

**关键配置参数**:
- `batch_size`: 2
- `learning_rate`: 5e-6
- `num_epochs`: 25
- `lora_r`: 64

### 多任务混合训练

**配置文件**: `examples/multitask_training.yaml`

**特点**:
- 同时训练多个任务
- 任务间知识共享
- 动态任务调度
- 梯度平衡机制

**关键配置参数**:
- `batch_size`: 4
- `learning_rate`: 1e-5
- `num_epochs`: 30
- `task_weights`: 动态调整

## 🔧 高级配置管理

### 使用高级配置管理工具

```bash
# 验证所有配置文件
python scripts/advanced_config_manager.py validate-all

# 比较两个配置文件
python scripts/advanced_config_manager.py compare config1.yaml config2.yaml

# 检测硬件配置
python scripts/advanced_config_manager.py detect-hardware

# 基于硬件优化配置
python scripts/advanced_config_manager.py optimize --config config.yaml

# 生成硬件适配配置
python scripts/advanced_config_manager.py hardware-adapt \
    --base-config examples/caption_training.yaml \
    --output optimized_caption.yaml \
    --gpu-memory 24

# 格式化配置文件
python scripts/advanced_config_manager.py format --config config.yaml
```

### 硬件适配建议

| GPU内存 | 推荐批次大小 | LoRA Rank | 梯度累积 |
|---------|-------------|-----------|----------|
| 4GB     | 1           | 16        | 8        |
| 8GB     | 2           | 32        | 4        |
| 12GB    | 4           | 32        | 2        |
| 16GB    | 6           | 64        | 2        |
| 24GB    | 8           | 64        | 1        |
| 32GB+   | 12+         | 128       | 1        |

## 📊 配置参数详解

### 基础训练参数

```yaml
# 训练轮数
num_epochs: 10

# 批次大小（根据GPU内存调整）
data_config:
  batch_size: 4
  num_workers: 4

# 梯度累积步数
gradient_accumulation_steps: 2

# 混合精度训练
use_bf16: true  # 推荐用于A100等现代GPU
use_fp16: false
```

### 模型配置

```yaml
model_config:
  model_name: "microsoft/Florence-2-base"  # 或 Florence-2-large
  use_lora: true
  lora_config:
    r: 32              # LoRA rank，影响参数量
    lora_alpha: 64     # 通常设为 r 的 2倍
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### 优化器配置

```yaml
optimization_config:
  optimizer_type: "adamw"
  learning_rate: 1e-5    # LoRA微调推荐 1e-5 到 5e-5
  weight_decay: 0.01
  warmup_steps: 500
  lr_scheduler_type: "cosine"
```

### 数据配置

```yaml
data_config:
  train_data_path: "data/train.jsonl"
  val_data_path: "data/val.jsonl"
  max_length: 512
  image_size: [384, 384]  # Florence-2 推荐尺寸
  
  # 数据增强
  augmentation:
    enabled: true
    rotation_range: 10
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
```

### 任务调度配置

```yaml
task_scheduling_config:
  strategy: "weighted"     # round_robin, weighted, curriculum, adaptive
  task_weights:
    CAPTION: 1.0
    OD: 1.5
    OCR: 1.2
  
  # 课程学习
  curriculum_learning:
    enabled: true
    difficulty_metric: "loss"
    stages: 3
```

## 🎯 最佳实践

### 1. 数据准备

- **图像格式**: 支持 JPG, PNG, WebP
- **图像尺寸**: 推荐 384x384 或保持宽高比缩放
- **数据质量**: 确保标注准确，图像清晰
- **数据平衡**: 各类别样本数量尽量均衡

### 2. 训练策略

- **学习率**: 从小开始，逐步调整
- **批次大小**: 根据GPU内存和任务复杂度调整
- **早停**: 设置合理的早停策略避免过拟合
- **检查点**: 定期保存模型检查点

### 3. 性能优化

- **混合精度**: 使用 bf16 或 fp16 加速训练
- **梯度累积**: 在内存受限时使用
- **数据加载**: 合理设置 num_workers
- **LoRA**: 大多数情况下推荐使用 LoRA 微调

### 4. 监控和调试

- **日志记录**: 启用详细日志记录
- **可视化**: 使用 TensorBoard 或 WandB 监控训练
- **验证**: 定期在验证集上评估模型
- **错误分析**: 分析失败案例，改进数据和配置

## 🔍 故障排除

### 常见问题

1. **内存不足 (OOM)**
   - 减小 batch_size
   - 增加 gradient_accumulation_steps
   - 使用混合精度训练
   - 减小 LoRA rank

2. **训练速度慢**
   - 增加 num_workers
   - 使用混合精度训练
   - 检查数据加载瓶颈
   - 优化数据预处理

3. **收敛困难**
   - 调整学习率
   - 检查数据质量
   - 增加 warmup_steps
   - 尝试不同的优化器

4. **过拟合**
   - 增加数据增强
   - 调整 weight_decay
   - 使用早停策略
   - 增加 dropout

### 配置验证

```bash
# 验证单个配置文件
python scripts/florence_cli.py validate --config your_config.yaml

# 验证所有配置文件
python scripts/advanced_config_manager.py validate-all

# 检查配置兼容性
python scripts/advanced_config_manager.py optimize --config your_config.yaml
```

## 📚 参考资源

- [Florence-2 论文](https://arxiv.org/abs/2311.06242)
- [Hugging Face 模型页面](https://huggingface.co/microsoft/Florence-2-base)
- [LoRA 微调指南](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [PyTorch 混合精度训练](https://pytorch.org/docs/stable/amp.html)

## 🤝 贡献指南

如果您有新的配置示例或改进建议，欢迎提交 Pull Request：

1. 确保配置文件格式正确
2. 添加详细的注释说明
3. 提供使用示例和预期结果
4. 通过配置验证测试

## 📄 许可证

本项目遵循 MIT 许可证。详见 LICENSE 文件。