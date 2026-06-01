# 模型合并和Benchmark评估功能

本文档介绍FlorenceForge框架中新增的模型合并和benchmark评估功能。

## 功能概述

### 1. 模型合并功能 (ModelMerger)

`ModelMerger`类提供了将LoRA权重合并到基础模型的功能，支持：

- **LoRA权重合并**: 将训练好的LoRA适配器权重合并到基础模型中
- **多适配器合并**: 支持合并多个LoRA适配器，可设置不同权重
- **模型导出**: 支持导出为PyTorch、ONNX、TorchScript格式
- **模型验证**: 验证合并后模型的正确性

### 2. Benchmark评估功能 (BenchmarkEvaluator)

`BenchmarkEvaluator`类提供了标准化的模型评估功能，支持：

- **多任务评估**: 同时评估多个任务的性能
- **标准指标计算**: 自动计算各任务的标准评估指标
- **基线比较**: 与基线模型结果进行对比分析
- **报告生成**: 生成Markdown、HTML、JSON格式的评估报告

## 使用方法

### 1. 基本配置

在训练配置中启用模型合并功能：

```yaml
# training_config.yaml
training:
  use_lora: true
  save_merged_model: true  # 启用模型合并
  merge_strategy: "linear"  # 合并策略: linear, weighted
  export_formats: ["pytorch", "onnx"]  # 导出格式
```

### 2. LoRA权重合并

#### 2.1 训练时自动合并

在训练脚本中，当`save_merged_model: true`时，训练完成后会自动保存合并后的模型：

```bash
# 使用LoRA训练脚本
./scripts/training/lora/training_caption_lora.sh

# 训练完成后，合并后的模型会保存在:
# outputs/caption_lora/merged_model/
```

#### 2.2 手动合并

```python
from florence_forge.training.model_merger import ModelMerger
from florence_forge.training.lora_manager import LoRAManager
from florence_forge.core.model import Florence2MultiTaskModel

# 1. 加载基础模型和LoRA管理器
base_model = Florence2MultiTaskModel("microsoft/Florence-2-base")
lora_manager = LoRAManager()
model_merger = ModelMerger(lora_manager)

# 2. 加载LoRA模型
lora_model = lora_manager.load_adapter(
    base_model, 
    "./outputs/caption_lora/checkpoints/final"
)

# 3. 合并并保存
model_merger.merge_and_unload(
    lora_model,
    "./merged_models/caption_merged",
    save_tokenizer=True,
    save_processor=True
)
```

#### 2.3 多适配器合并

```python
# 合并多个LoRA适配器
adapter_paths = {
    "caption": "./outputs/caption_lora/checkpoints/final",
    "detection": "./outputs/detection_lora/checkpoints/final",
    "segmentation": "./outputs/segmentation_lora/checkpoints/final"
}

adapter_weights = {
    "caption": 1.0,
    "detection": 0.8,
    "segmentation": 0.6
}

merged_model = model_merger.merge_multiple_adapters(
    base_model,
    adapter_paths,
    adapter_weights,
    "./merged_models/multi_task_merged"
)
```

### 3. Benchmark评估

#### 3.1 完整Benchmark评估

```python
from florence_forge.evaluation.benchmark import BenchmarkEvaluator
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.data.dataset import MultiTaskDataset

# 1. 加载模型
model = Florence2MultiTaskModel("./merged_models/caption_merged")

# 2. 创建评估器
benchmark_config = {
    'batch_size': 8,
    'num_workers': 4,
    'max_samples_per_task': 1000,
    'save_predictions': True,
    'compute_detailed_metrics': True
}

evaluator = BenchmarkEvaluator(
    model=model,
    device="cuda:0",
    benchmark_config=benchmark_config
)

# 3. 准备数据集
datasets = {
    "coco_caption": MultiTaskDataset(
        data_configs=[{
            "task_type": "CAPTION",
            "data_path": "./data/coco_caption_val.jsonl",
            "weight": 1.0
        }]
    ),
    "voc_detection": MultiTaskDataset(
        data_configs=[{
            "task_type": "DETECTION", 
            "data_path": "./data/voc_detection_val.jsonl",
            "weight": 1.0
        }]
    )
}

# 4. 运行评估
results = evaluator.run_benchmark(
    datasets=datasets,
    output_dir="./benchmark_results",
    save_detailed=True
)

# 5. 生成报告
evaluator.generate_benchmark_report(
    results,
    "./benchmark_results/report.md",
    format="markdown"
)
```

#### 3.2 单任务评估

```python
# 评估单个任务
task_results = evaluator.evaluate_single_task(
    dataset=datasets["coco_caption"],
    task_name="caption",
    output_dir="./single_task_results"
)

print(f"Caption任务BLEU-4分数: {task_results['bleu_4']}")
```

#### 3.3 标准指标计算

```python
# 计算图像描述任务指标
predictions = ["a cat sitting on a table", "a dog running"]
references = ["a cat is on the table", "a dog is running"]

metrics = evaluator.compute_standard_metrics(
    predictions=predictions,
    references=references,
    task_type="caption"
)

print(f"BLEU-4: {metrics['bleu_4']}")
print(f"ROUGE-L: {metrics['rouge_l']}")
print(f"CIDEr: {metrics['cider']}")
```

### 4. 基线比较

```python
# 加载基线结果
with open("./baseline_results.json", 'r') as f:
    baseline_results = json.load(f)

# 比较当前结果与基线
comparison = evaluator.compare_with_baseline(
    current_results=results,
    baseline_results=baseline_results
)

# 查看改进情况
for metric, improvement in comparison['overall_improvement'].items():
    print(f"{metric}: {improvement['relative']*100:+.2f}% 改进")
```

## 支持的任务和指标

### 任务类型

- **图像描述 (Caption)**: BLEU, ROUGE, CIDEr, METEOR
- **目标检测 (Detection)**: mAP, Precision, Recall
- **OCR**: 字符准确率, 单词准确率, 编辑距离
- **分割 (Segmentation)**: IoU, Dice系数, 像素准确率
- **短语定位 (Phrase Grounding)**: 定位准确率, IoU
- **指代表达分割**: 分割IoU, 定位准确率

### 评估指标

#### 图像描述任务
- **BLEU-1/2/3/4**: 基于n-gram的相似度
- **ROUGE-L**: 最长公共子序列
- **CIDEr**: 共识图像描述评估
- **METEOR**: 基于词干和同义词的评估

#### 检测任务
- **mAP**: 平均精度均值
- **AP@IoU**: 不同IoU阈值下的平均精度
- **Precision/Recall**: 精确率和召回率

#### OCR任务
- **字符准确率**: 字符级别的准确率
- **单词准确率**: 单词级别的准确率
- **编辑距离**: Levenshtein距离

#### 分割任务
- **IoU**: 交并比
- **Dice系数**: Dice相似系数
- **像素准确率**: 像素级别准确率

## 配置选项

### ModelMerger配置

```python
# 在TrainingConfig中配置
training_config = {
    "save_merged_model": True,      # 是否保存合并模型
    "merge_strategy": "linear",     # 合并策略: linear, weighted
    "export_formats": ["pytorch"]   # 导出格式: pytorch, onnx, torchscript
}
```

### BenchmarkEvaluator配置

```python
benchmark_config = {
    'batch_size': 8,                    # 批处理大小
    'num_workers': 4,                   # 数据加载器工作进程数
    'max_samples_per_task': 1000,       # 每个任务最大样本数
    'save_predictions': True,           # 是否保存预测结果
    'compute_detailed_metrics': True,   # 是否计算详细指标
    'confidence_threshold': 0.5,        # 检测任务置信度阈值
    'iou_threshold': 0.5,              # IoU阈值
    'max_detections': 100              # 最大检测数量
}
```

## 示例脚本

完整的使用示例请参考：
- `scripts/examples/model_merge_and_benchmark.py` - 完整功能演示
- `scripts/training/lora/` - LoRA训练脚本（已配置自动合并）

## 输出文件结构

```
outputs/
├── task_lora/                    # LoRA训练输出
│   ├── checkpoints/              # 检查点
│   ├── logs/                     # 训练日志
│   └── merged_model/             # 合并后的模型
│       ├── pytorch_model.bin
│       ├── config.json
│       ├── tokenizer.json
│       └── preprocessor_config.json
├── benchmark_results/            # Benchmark评估结果
│   ├── predictions/              # 预测结果
│   ├── metrics/                  # 指标计算结果
│   └── reports/                  # 评估报告
│       ├── benchmark_report.md
│       ├── benchmark_report.html
│       └── benchmark_report.json
└── exported_models/              # 导出的模型
    ├── pytorch/
    ├── onnx/
    └── torchscript/
```

## 注意事项

1. **内存使用**: 模型合并过程需要额外内存，建议在GPU内存充足时进行
2. **ONNX导出**: 需要安装`onnx`和`onnxruntime`包
3. **数据格式**: 确保评估数据集格式与训练数据格式一致
4. **基线比较**: 基线结果文件格式需要与当前评估结果格式匹配
5. **多GPU**: 目前仅支持单GPU评估，多GPU支持正在开发中

## 故障排除

### 常见问题

1. **合并失败**: 检查LoRA适配器路径是否正确
2. **ONNX导出失败**: 确保安装了正确版本的ONNX相关包
3. **评估内存不足**: 减少batch_size或max_samples_per_task
4. **指标计算错误**: 检查预测和参考数据格式是否正确

### 调试建议

1. 启用详细日志: `logging.basicConfig(level=logging.DEBUG)`
2. 使用小数据集测试: 设置`max_samples_per_task=10`
3. 检查模型输出格式: 确保与任务要求匹配
4. 验证数据路径: 确保所有数据文件存在且可访问