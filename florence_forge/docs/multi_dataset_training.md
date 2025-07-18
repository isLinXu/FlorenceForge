# Florence Forge - 多数据集多任务训练指南

本指南详细介绍如何使用 Florence Forge 框架进行多数据集多任务训练。

## 概述

Florence Forge 现在支持同时使用多个数据集进行多任务训练，这使得模型能够：

- 从不同来源的数据中学习
- 处理多种任务类型
- 实现更好的泛化性能
- 支持跨数据集的知识迁移

## 核心组件

### 1. MultiDatasetManager

多数据集管理器负责：
- 注册和管理多个数据集
- 配置任务到数据集的映射
- 处理数据集间的平衡和采样
- 提供统一的数据访问接口

### 2. MultiDatasetTrainer

多数据集训练器提供：
- 继承自 `MultiTaskTrainer` 的所有功能
- 数据集感知的训练循环
- 动态数据集权重调整
- 详细的数据集性能监控

### 3. 配置系统

支持通过 JSON 配置文件定义：
- 数据集信息和路径
- 任务到数据集的映射关系
- 采样策略和权重
- 预处理参数

## 快速开始

### 1. 基本使用

```python
from florence_forge.data.multi_dataset_manager import MultiDatasetManager, DatasetInfo
from florence_forge.training.multi_dataset_trainer import MultiDatasetTrainer
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.config import TrainingConfig

# 创建数据集管理器
manager = MultiDatasetManager()

# 注册数据集
coco_info = DatasetInfo(
    name="coco_captions",
    path="/path/to/coco",
    format="coco",
    task_types=["image_captioning"],
    priority=1.0
)
manager.register_dataset(coco_info)

vqa_info = DatasetInfo(
    name="vqa_v2",
    path="/path/to/vqa",
    format="vqa",
    task_types=["visual_question_answering"],
    priority=1.2
)
manager.register_dataset(vqa_info)

# 配置任务映射
from florence_forge.data.multi_dataset_manager import TaskDatasetMapping

captioning_mapping = TaskDatasetMapping(
    task_type="image_captioning",
    datasets=["coco_captions"],
    weights=[1.0],
    sampling_strategy="sequential"
)
manager.add_task_mapping(captioning_mapping)

# 创建模型和训练配置
model = Florence2MultiTaskModel.from_pretrained("microsoft/Florence-2-base")
config = TrainingConfig(
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-5,
    output_dir="./outputs"
)

# 创建训练器
trainer = MultiDatasetTrainer(
    model=model,
    dataset_manager=manager,
    config=config
)

# 开始训练
results = trainer.train()
```

### 2. 使用配置文件

创建配置文件 `dataset_config.json`：

```json
{
  "datasets": {
    "coco_captions": {
      "name": "coco_captions",
      "path": "/path/to/coco/captions",
      "format": "coco",
      "task_types": ["image_captioning"],
      "priority": 1.0,
      "max_samples": 50000,
      "preprocessing": {
        "image_size": [384, 384],
        "normalize": true
      }
    },
    "vqa_v2": {
      "name": "vqa_v2",
      "path": "/path/to/vqa/v2",
      "format": "vqa",
      "task_types": ["visual_question_answering"],
      "priority": 1.2,
      "max_samples": 30000
    }
  },
  "task_mappings": {
    "image_captioning": {
      "datasets": ["coco_captions"],
      "weights": [1.0],
      "sampling_strategy": "sequential"
    },
    "visual_question_answering": {
      "datasets": ["vqa_v2"],
      "weights": [1.0],
      "sampling_strategy": "sequential"
    }
  },
  "global_settings": {
    "enable_balanced_sampling": true,
    "adaptive_dataset_weighting": true,
    "max_total_samples": 100000
  }
}
```

然后使用配置文件：

```python
# 从配置文件创建训练器
trainer = MultiDatasetTrainer.from_config(
    model=model,
    dataset_config_path="dataset_config.json",
    training_config=config
)

results = trainer.train()
```

## 高级功能

### 1. 多数据集任务映射

一个任务可以使用多个数据集：

```python
# 图像描述任务使用多个数据集
captioning_mapping = TaskDatasetMapping(
    task_type="image_captioning",
    datasets=["coco_captions", "flickr30k", "chinese_captions"],
    weights=[0.5, 0.3, 0.2],  # 数据集权重
    sampling_strategy="weighted_random"  # 加权随机采样
)
manager.add_task_mapping(captioning_mapping)
```

### 2. 动态权重调整

启用自适应数据集权重调整：

```python
config = TrainingConfig(
    # ... 其他参数
    adaptive_dataset_weighting=True  # 启用动态权重调整
)
```

训练过程中，系统会根据各数据集的性能自动调整权重。

### 3. 平衡采样

确保各数据集的样本被均匀采样：

```python
manager.enable_balanced_sampling()
```

### 4. 样本数量限制

限制总样本数量或单个数据集的样本数量：

```python
# 限制总样本数
manager.limit_total_samples(100000)

# 在数据集注册时限制单个数据集
dataset_info = DatasetInfo(
    name="large_dataset",
    path="/path/to/large/dataset",
    format="custom",
    task_types=["task1"],
    max_samples=10000  # 最多使用10000个样本
)
```

### 5. 跨数据集验证

启用跨数据集验证以评估模型的泛化能力：

```python
config = TrainingConfig(
    # ... 其他参数
    cross_dataset_validation=True
)
```

## 采样策略

支持多种采样策略：

### 1. Sequential（顺序）
按顺序遍历数据集中的样本。

### 2. Weighted Random（加权随机）
根据指定权重随机选择数据集，然后从选中的数据集中随机采样。

### 3. Round Robin（轮询）
轮流从各个数据集中采样。

### 4. Balanced（平衡）
确保每个epoch中各数据集的样本数量相对平衡。

## 监控和分析

### 1. 数据集性能监控

```python
# 获取数据集性能摘要
performance = trainer.get_dataset_performance_summary()
print(performance)

# 保存详细的性能分析
trainer.save_dataset_performance("performance_analysis.json")
```

### 2. 训练日志

系统会自动记录：
- 每个步骤的数据集信息
- 各数据集的损失变化
- 数据集权重调整历史

日志文件：
- `step_metrics.csv` - 基础训练指标
- `dataset_step_metrics.csv` - 数据集特定指标
- `dataset_performance.json` - 详细性能分析

## 最佳实践

### 1. 数据集优先级设置

- 高质量数据集设置较高优先级
- 任务相关性强的数据集设置较高优先级
- 考虑数据集大小，避免小数据集被忽略

### 2. 权重配置

- 初始权重可以基于数据集大小和质量设置
- 启用自适应权重调整以优化训练效果
- 定期检查权重变化，确保合理性

### 3. 采样策略选择

- 数据集大小相近时使用 `round_robin`
- 数据集大小差异较大时使用 `weighted_random`
- 需要严格控制比例时使用 `balanced`

### 4. 性能监控

- 定期检查各数据集的损失变化
- 关注数据集间的性能差异
- 使用跨数据集验证评估泛化能力

## 故障排除

### 1. 内存不足

- 减少 `batch_size`
- 限制 `max_samples`
- 使用梯度累积

### 2. 训练不稳定

- 检查数据集质量和一致性
- 调整学习率和权重衰减
- 考虑使用课程学习

### 3. 某个数据集性能差

- 检查数据预处理是否正确
- 调整该数据集的权重
- 考虑数据集特定的预处理

## 示例项目

完整的示例代码请参考：
- `examples/multi_dataset_training_example.py` - 基础示例
- `examples/advanced_multi_dataset_training.py` - 高级功能示例
- `configs/` - 各种配置文件示例

## API 参考

详细的 API 文档请参考：
- `MultiDatasetManager` - 数据集管理
- `MultiDatasetTrainer` - 训练器
- `DatasetInfo` - 数据集信息
- `TaskDatasetMapping` - 任务映射

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多数据集注册和管理
- 实现多种采样策略
- 添加动态权重调整
- 提供详细的性能监控

---

如有问题或建议，请提交 Issue 或联系开发团队。