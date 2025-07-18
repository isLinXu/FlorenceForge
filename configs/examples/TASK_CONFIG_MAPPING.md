# Florence-2 任务配置文件映射

本文档提供了 Florence-2 所有支持任务与对应配置文件的完整映射关系。

## 任务分类概览

### 图像描述类任务 (IMAGE_CAPTIONING)

| 任务名称 | 配置文件 | 描述 | 模型推荐 |
|---------|---------|------|----------|
| `CAPTION` | `caption_training.yaml` | 基础图像描述生成 | Florence-2-base |
| `DETAILED_CAPTION` | `detailed_caption_training.yaml` | 详细图像描述生成 | Florence-2-base |
| `MORE_DETAILED_CAPTION` | `more_detailed_caption_training.yaml` | 更详细的图像描述生成 | Florence-2-large |

### 目标检测类任务 (OBJECT_DETECTION)

| 任务名称 | 配置文件 | 描述 | 模型推荐 |
|---------|---------|------|----------|
| `OD` | `object_detection_training.yaml` | 标准目标检测 | Florence-2-base |
| `OPEN_VOCABULARY_DETECTION` | `open_vocabulary_detection_training.yaml` | 开放词汇目标检测 | Florence-2-large |
| `CAPTION_TO_PHRASE_GROUNDING` | `phrase_grounding_training.yaml` | 短语定位 | Florence-2-base |

### 区域分析类任务 (REGION_ANALYSIS)

| 任务名称 | 配置文件 | 描述 | 模型推荐 |
|---------|---------|------|----------|
| `REGION_PROPOSAL` | `region_proposal_training.yaml` | 区域提议生成 | Florence-2-base |
| `DENSE_REGION_CAPTION` | `dense_region_caption_training.yaml` | 密集区域描述 | Florence-2-large |
| `REGION_TO_CATEGORY` | `region_to_category_training.yaml` | 区域到类别分类 | Florence-2-base |
| `REGION_TO_DESCRIPTION` | `region_to_description_training.yaml` | 区域到描述生成 | Florence-2-base |

### 文字识别类任务 (TEXT_RECOGNITION)

| 任务名称 | 配置文件 | 描述 | 模型推荐 |
|---------|---------|------|----------|
| `OCR` | `ocr_training.yaml` | 光学字符识别 | Florence-2-base |
| `OCR_WITH_REGION` | `ocr_with_region_training.yaml` | 带区域的光学字符识别 | Florence-2-base |

### 图像分割类任务 (IMAGE_SEGMENTATION)

| 任务名称 | 配置文件 | 描述 | 模型推荐 |
|---------|---------|------|----------|
| `REGION_TO_SEGMENTATION` | `region_to_segmentation_training.yaml` | 区域到分割 | Florence-2-base |
| `REFERRING_EXPRESSION_SEGMENTATION` | `referring_expression_segmentation_training.yaml` | 参考表达式分割 | Florence-2-large |

### 多任务训练

| 配置文件 | 描述 | 包含任务 |
|---------|------|----------|
| `multitask_training.yaml` | 多任务联合训练 | 可配置多个任务 |

## 配置文件特点

### 按任务复杂度分类

**简单任务** (较小模型，较高学习率):
- `caption_training.yaml`
- `region_to_category_training.yaml`
- `region_proposal_training.yaml`

**中等复杂度任务** (中等配置):
- `detailed_caption_training.yaml`
- `object_detection_training.yaml`
- `ocr_training.yaml`
- `phrase_grounding_training.yaml`
- `region_to_description_training.yaml`
- `region_to_segmentation_training.yaml`
- `ocr_with_region_training.yaml`

**复杂任务** (大模型，较低学习率，更多训练轮数):
- `more_detailed_caption_training.yaml`
- `open_vocabulary_detection_training.yaml`
- `dense_region_caption_training.yaml`
- `referring_expression_segmentation_training.yaml`

### 关键配置差异

| 配置项 | 简单任务 | 中等任务 | 复杂任务 |
|-------|---------|---------|----------|
| 模型 | Florence-2-base | Florence-2-base | Florence-2-large |
| LoRA rank | 16-32 | 32-48 | 48-64 |
| 学习率 | 2.5e-05 - 3.0e-05 | 1.5e-05 - 2.5e-05 | 1.0e-05 - 1.5e-05 |
| 批次大小 | 8-12 | 6-8 | 4-6 |
| 训练轮数 | 5-10 | 8-12 | 12-15 |
| 梯度累积 | 4 | 4-6 | 6 |

## 使用示例

### 单任务训练
```bash
# 图像描述任务
python florence_forge/scripts/florence_cli.py validate --config florence_forge/configs/examples/caption_training.yaml

# 目标检测任务
python florence_forge/scripts/florence_cli.py validate --config florence_forge/configs/examples/object_detection_training.yaml

# OCR任务
python florence_forge/scripts/florence_cli.py validate --config florence_forge/configs/examples/ocr_training.yaml
```

### 多任务训练
```bash
# 多任务联合训练
python florence_forge/scripts/florence_cli.py validate --config florence_forge/configs/examples/multitask_training.yaml
```

## 自定义配置建议

1. **选择基础配置**: 根据目标任务选择最接近的配置文件作为模板
2. **调整模型大小**: 根据计算资源和任务复杂度选择 base 或 large 模型
3. **优化超参数**: 根据数据集大小调整学习率、批次大小和训练轮数
4. **配置LoRA**: 复杂任务使用更高的 rank 值
5. **数据增强**: 根据任务特点调整增强策略和概率

## 注意事项

- 所有配置文件都已通过验证，可以直接使用
- 建议在实际训练前先进行小规模测试
- 根据GPU内存调整批次大小和梯度累积步数
- 复杂任务可能需要更长的训练时间和更多的计算资源