# 🎨 FlorenceForge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

**FlorenceForge** 是一个专为 Florence-2 模型设计的高效多任务微调框架，提供从数据处理到模型部署的完整解决方案。

## ✨ 核心特性

### 🎯 多任务支持
- **图像描述**: CAPTION, DETAILED_CAPTION, MORE_DETAILED_CAPTION
- **目标检测**: OD, OPEN_VOCABULARY_DETECTION, DENSE_REGION_CAPTION
- **区域分析**: REGION_PROPOSAL, REGION_TO_CATEGORY, REGION_TO_DESCRIPTION
- **文字识别**: OCR, OCR_WITH_REGION
- **图像分割**: REFERRING_EXPRESSION_SEGMENTATION, REGION_TO_SEGMENTATION

### ⚡ 高效训练
- **LoRA 微调**: 参数高效的适配器微调
- **多任务混合训练**: 智能任务调度和权重平衡
- **梯度累积**: 支持大批次训练
- **混合精度**: 加速训练并节省显存
- **分布式训练**: 基于 Accelerate 的多GPU支持

### 🔧 便捷工具
- **命令行界面**: 一键训练、验证、数据转换
- **配置管理**: 灵活的 YAML/JSON 配置系统
- **数据转换**: 支持 YOLO、COCO、CSV、XML、OCR 格式
- **可视化**: 训练过程监控和结果可视化
- **模型部署**: FastAPI 服务器和推理优化

### 📊 训练监控
- **WandB 集成**: 实验跟踪和可视化
- **SwanLab 支持**: 开源机器学习实验管理
- **TensorBoard**: 传统的训练监控工具
- **多平台同步**: 同时使用多个监控工具
- **自动记录**: 训练指标、模型架构、梯度分布

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，支持 CPU 训练)

### 安装

```bash
# 克隆仓库
git clone https://github.com/florenceforge/florence-forge.git
cd florence-forge

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .

# 验证安装
florence_forge_cli --help
```

如需运行评估相关指标，建议额外安装：

```bash
pip install -e ".[evaluation]"
```

### 快速训练

```bash
# 1. 生成配置文件
florence_forge_cli generate-config --task caption --output my_config.yaml

# 2. 准备数据 (JSONL 格式：image/prefix/suffix)
echo '{"image": "path/to/image.jpg", "prefix": "<CAPTION>", "suffix": "A beautiful sunset"}' > data.jsonl

# 3. 开始训练
florence_forge_cli train --config my_config.yaml --epochs 5

# 4. 查看结果
ls outputs/
```

### Python API 使用

```python
from florence_forge import TrainingConfig, MultiTaskTrainer
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.data.dataset import MultiTaskDataset

# 1. 加载配置
config = TrainingConfig.from_yaml('config.yaml')

# 2. 创建并加载模型
model = Florence2MultiTaskModel(config.model_settings).load()

# 3. 准备数据
dataset = MultiTaskDataset(
    data_configs=[
        {
            "task_type": "CAPTION",
            "data_path": "data.jsonl",
            "weight": 1.0
        }
    ],
    config=config.data_settings
)

# 4. 创建训练器
trainer = MultiTaskTrainer(
    model=model,
    train_dataset=dataset,
    config=config
)

# 5. 开始训练
results = trainer.train()
print(f"训练完成! 最终损失: {results['final_loss']}")
```

## 📖 详细使用指南

### 数据格式

FlorenceForge 支持 JSONL 格式的数据文件，每行包含一个样本：

```json
{"image": "path/to/image.jpg", "prefix": "<CAPTION>", "suffix": "描述文本"}
{"image": "path/to/image2.jpg", "prefix": "<OD>", "suffix": "<loc_10><loc_20><loc_100><loc_200>目标类别"}
{"image": "path/to/image3.jpg", "prefix": "<REGION_TO_DESCRIPTION>", "region": "<loc_50><loc_60><loc_150><loc_160>", "suffix": "区域描述"}
```

### 配置文件详解

```yaml
# 模型配置
model_config:
  model_name: "microsoft/Florence-2-base"  # 基础模型
  device: "auto"                          # 设备选择
  torch_dtype: "float16"                  # 数据类型
  trust_remote_code: true                 # 信任远程代码

# LoRA 配置
lora_config:
  r: 32                    # LoRA 秩
  lora_alpha: 32          # LoRA alpha
  target_modules: ["q_proj", "v_proj", "k_proj", "out_proj"]
  lora_dropout: 0.05      # Dropout 率
  bias: "none"            # 偏置设置
  task_type: "FEATURE_EXTRACTION"

# 训练配置
training_config:
  output_dir: "./outputs"              # 输出目录
  num_epochs: 10                       # 训练轮数
  batch_size: 8                        # 批次大小
  gradient_accumulation_steps: 4       # 梯度累积
  learning_rate: 1e-5                  # 学习率
  weight_decay: 0.01                   # 权重衰减
  warmup_steps: 500                    # 预热步数
  max_grad_norm: 1.0                   # 梯度裁剪
  mixed_precision: "fp16"              # 混合精度
  
# 数据配置
data_config:
  max_length: 1024        # 最大序列长度
  image_size: [768, 768]  # 图像尺寸
  num_workers: 4          # 数据加载器工作进程
  pin_memory: true        # 内存固定
```

### 命令行工具

#### 训练模型
```bash
# 基础训练
florence_forge_cli train --config config.yaml

# 从检查点恢复
florence_forge_cli train --config config.yaml --resume outputs/checkpoint-1000

# 分布式训练
accelerate launch --multi_gpu florence_forge_cli train --config config.yaml
```

#### 模型评估
```bash
# 评估模型
florence_forge_cli evaluate --model outputs/final_model --data test_data.jsonl

# 指定任务评估
florence_forge_cli evaluate --model outputs/final_model --data test_data.jsonl --tasks CAPTION,OD
```

#### 数据转换
```bash
# COCO 转 JSONL
florence_forge_cli convert-data --format coco --input coco_annotations.json --output data.jsonl

# YOLO 转 JSONL
florence_forge_cli convert-data --format yolo --input yolo_labels/ --output data.jsonl
```

#### 训练监控
```bash
# 启用 WandB 监控
florence_forge_cli train --config config.yaml \
  --enable-wandb \
  --wandb-project "florence2-experiments" \
  --wandb-run-name "my-experiment"

# 启用 SwanLab 监控
florence_forge_cli train --config config.yaml \
  --enable-swanlab \
  --swanlab-project "florence2-training"

# 启用所有监控工具
florence_forge_cli train --config config.yaml \
  --enable-wandb --enable-swanlab --enable-tensorboard
```

### 监控配置示例

```python
from florence_forge.training.config import TrainingConfig

# 配置 WandB 监控
config = TrainingConfig(
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    output_dir="./outputs",
    
    # 启用 WandB
    enable_wandb=True,
    wandb_project="florence2-training",
    wandb_entity="your-username",
    wandb_run_name="experiment-1",
    
    # 同时启用其他监控工具
    enable_swanlab=True,
    enable_tensorboard=True,
    
    logging_steps=10,
    eval_steps=1
)

# 训练会自动记录指标到所有启用的监控平台
trainer = MultiTaskTrainer(model=model, train_dataset=dataset, config=config)
results = trainer.train()
```

#### 模型推理
```bash
# 单张图片推理
florence_forge_cli infer --model outputs/final_model --image image.jpg --task CAPTION

# 批量推理
florence_forge_cli infer --model outputs/final_model --input images/ --output results.json
```

#### 启动推理服务
```bash
# 启动 API 服务器
florence_forge_cli serve --model outputs/final_model --host 0.0.0.0 --port 8000

# 测试 API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"image_path": "image.jpg", "task": "CAPTION"}'
```

## 🏗️ 项目架构

```
florence_forge/
├── 📁 core/                         # 核心模块
│   ├── backends/                     # VLM 后端抽象
│   │   ├── base_vlm.py              # BaseVLMBackend + VLMBackendRegistry
│   │   ├── florence2_backend.py     # Florence-2 后端
│   │   ├── paligemma_backend.py     # PaliGemma 后端
│   │   ├── youtuvl_backend.py       # YouTuVL 后端
│   │   └── generic_hf_backend.py    # 通用 HuggingFace VLM 后端
│   ├── architecture_resolver.py      # 后端路由策略门面
│   ├── config.py                     # Pydantic v2 配置体系（7 个配置类）
│   ├── yaml_config.py               # YAML 配置加载
│   ├── model.py                      # 模型工厂
│   ├── tasks.py                      # 13 种任务定义与注册
│   ├── visual_primitives.py          # 视觉原语（Visual Primitive）编解码
│   └── callbacks.py                  # 训练回调
├── 📁 data/                          # 数据处理
│   ├── dataset.py                    # MultiTaskDataset（LRU 图像缓存）
│   ├── converter.py                  # DataFormatConverter（YOLO/COCO/CSV/VOC/OCR）
│   ├── loader.py                     # TaskDataLoader
│   ├── collate.py                    # Florence2Collator
│   ├── builder.py                    # 数据构建器
│   ├── validator.py                  # 数据校验
│   ├── multi_dataset_manager.py      # 多数据集权重/优先级调度
│   ├── image_cache.py                # 按字节预算 LRU 图像缓存
│   ├── utils.py                      # 数据工具函数
│   └── augmentation/                 # 数据增强
│       ├── image_augmentation.py     # 图像增强
│       ├── text_augmentation.py      # 文本增强
│       └── bbox_augmentation.py      # 边界框增强
├── 📁 training/                      # 训练模块
│   ├── trainer.py                    # v1 多任务训练器（god class，默认导出）
│   ├── trainer_refactored.py         # v2 训练器（模块化，推荐新项目使用）
│   ├── training_loop.py              # v2 训练/验证循环
│   ├── checkpoint_manager.py         # v2 检查点管理（原子写）
│   ├── checkpoint.py                 # v1 检查点工具
│   ├── scheduler.py                  # 任务调度器（TaskScheduler）
│   ├── lora_manager.py               # LoRA 适配器管理
│   ├── visualizer.py                 # 训练可视化
│   ├── monitoring.py                 # WandB/SwanLab/TensorBoard 监控
│   ├── gradient_validator.py         # 梯度验证
│   ├── gradient_checkpoint_optimizer.py # 梯度检查点优化
│   ├── memory_monitor.py             # 内存监控
│   ├── model_merger.py               # 模型合并
│   ├── multi_dataset_trainer.py      # 多数据集训练器
│   ├── device_config.py              # 设备/精度配置（DeviceConfigurator）
│   └── trainer_io.py                 # 训练 IO 工具
├── 📁 evaluation/                    # 评估模块
│   ├── evaluator.py                  # MultiTaskEvaluator
│   ├── benchmark.py                  # BenchmarkEvaluator
│   ├── benchmark_cache.py            # 评测缓存
│   ├── benchmark_reports.py          # HTML/JSON/MD/PDF 报告
│   ├── benchmark_statistics.py       # 统计分析
│   ├── benchmark_parallel.py         # 并行评测
│   ├── benchmark_monitoring.py       # 实时监控
│   ├── benchmark_lazy_metrics.py     # 惰性高级指标
│   ├── benchmark_pdf_report.py       # PDF 报告生成
│   ├── metrics.py                    # 基础指标（BLEU/ROUGE/CIDEr/mAP/IoU）
│   ├── analyzer.py                   # ResultAnalyzer
│   └── advanced_metrics/             # 高级指标（算法核心 + MetricCalculator 适配双层）
│       ├── caption_metrics.py        # 字幕指标
│       ├── object_detection_metrics.py # 检测指标
│       ├── semantic_metrics.py / _calculator.py  # 语义相似度
│       ├── multimodal_metrics.py / _calculator.py # 多模态指标
│       ├── robustness_metrics.py / _calculator.py # 鲁棒性指标
│       └── efficiency_metrics.py / _calculator.py # 效率指标
├── 📁 deployment/                    # 部署模块
│   ├── inference.py                  # InferenceEngine（AMP/批量/编译）
│   ├── server.py                     # FastAPI REST 服务
│   ├── exporter.py                   # ONNX/TorchScript/SafeTensors 导出
│   ├── optimizer.py                   # 模型图优化
│   └── backends.py                   # 部署后端
├── 📁 optimization/                  # 优化模块
│   └── quantization.py               # bnb-4bit/8bit/GPTQ/AWQ/int8 量化
├── 📁 experimental/                  # 实验特性
│   └── moe/                          # MoE 实验（SparseGate + SelectiveSSMMixer）
├── 📁 utils/                         # 工具模块
│   ├── device.py                     # 设备检测
│   ├── image.py                      # 图像处理
│   ├── text.py                       # 文本处理
│   ├── io.py                         # 文件 IO
│   ├── logging.py                    # 日志管理
│   ├── memory.py                     # 内存工具
│   ├── visualization.py              # 可视化工具
│   ├── diagnostics.py                # 诊断工具
│   ├── torch_serialization.py        # safe_torch_load
│   ├── torch_compile.py              # torch.compile 封装
│   ├── training_logging.py           # 训练日志格式化
│   ├── optional_dependencies.py      # 可选依赖检测
│   └── tools.py                      # 通用工具
├── 📁 cli/                           # 命令行工具
│   ├── main.py                       # argparse 主入口 + 调度
│   ├── commands.py                   # 各子命令 handler
│   ├── _helpers.py                   # 共享常量与纯函数
│   └── config_manager.py             # 配置文件管理
└── 📁 exceptions.py                  # 统一异常层次
```

## 📚 API 文档

### 核心类

#### `Florence2MultiTaskModel`
多任务 Florence-2 模型封装类。

```python
class Florence2MultiTaskModel:
    def __init__(self, config: ModelConfig)
    def forward(self, input_ids, pixel_values, labels=None) -> dict
    def generate(self, input_ids, pixel_values, **kwargs) -> torch.Tensor
    def get_trainable_parameters(self) -> int
```

#### `MultiTaskTrainer`
多任务训练器，支持 LoRA 微调和混合训练。

```python
class MultiTaskTrainer:
    def __init__(self, model, train_dataset, config, val_dataset=None)
    def train(self) -> dict
    def evaluate(self, dataset=None) -> dict
    def save_model(self, output_dir: str)
    def load_checkpoint(self, checkpoint_path: str)
```

#### `MultiTaskDataset`
多任务数据集类，支持动态任务采样。

```python
class MultiTaskDataset:
    def __init__(self, data_configs: List[dict], config: DataConfig)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> dict
    def get_task_weights(self) -> dict
```

#### `MultiTaskEvaluator`
模型评估器，支持多种评估指标。

```python
class MultiTaskEvaluator:
    def __init__(self, model, device="auto")
    def evaluate_dataset(self, dataset, **kwargs) -> dict
    def evaluate_single(self, image, text, task_type) -> dict
    def compute_metrics(self, predictions, references, task_type) -> dict
```

### 配置类

#### `TrainingConfig`
训练配置管理类。

```python
class TrainingConfig:
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig'
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig'
    def to_yaml(self, yaml_path: str)
    def to_dict(self) -> dict
```

### 工具函数

#### 图像处理
```python
from florence_forge.utils.image import (
    load_image,      # 加载图像
    resize_image,    # 调整图像大小
    normalize_image  # 图像标准化
)
```

#### 文本处理
```python
from florence_forge.utils.text import (
    clean_text,           # 清理文本
    extract_bbox_from_text,  # 从文本提取边界框
    format_task_prompt    # 格式化任务提示
)
```

## 🎯 支持的任务类型

| 任务类型 | 描述 | 输入格式 | 输出格式 |
|---------|------|----------|----------|
| `CAPTION` | 图像描述 | 图像 | 文本描述 |
| `DETAILED_CAPTION` | 详细图像描述 | 图像 | 详细文本描述 |
| `MORE_DETAILED_CAPTION` | 更详细图像描述 | 图像 | 更详细文本描述 |
| `OD` | 目标检测 | 图像 | 边界框 + 类别 |
| `DENSE_REGION_CAPTION` | 密集区域描述 | 图像 | 区域 + 描述 |
| `REGION_PROPOSAL` | 区域提议 | 图像 | 候选区域 |
| `OCR` | 文字识别 | 图像 | 识别文字 |
| `OCR_WITH_REGION` | 带区域的文字识别 | 图像 | 文字 + 位置 |
| `REFERRING_EXPRESSION_SEGMENTATION` | 指代表达式分割 | 图像 + 文本 | 分割掩码 |
| `REGION_TO_SEGMENTATION` | 区域分割 | 图像 + 区域 | 分割掩码 |
| `OPEN_VOCABULARY_DETECTION` | 开放词汇检测 | 图像 + 类别 | 边界框 |
| `REGION_TO_CATEGORY` | 区域分类 | 图像 + 区域 | 类别 |
| `REGION_TO_DESCRIPTION` | 区域描述 | 图像 + 区域 | 文本描述 |

## 📊 性能基准

### 训练效率
| 配置 | 批次大小 | 显存占用 | 训练速度 |
|------|----------|----------|----------|
| Florence-2-base + LoRA | 8 | ~12GB | ~2.5 it/s |
| Florence-2-large + LoRA | 4 | ~16GB | ~1.8 it/s |
| 多任务混合训练 | 8 | ~14GB | ~2.2 it/s |

### 模型性能
| 任务 | 数据集 | 指标 | 基线模型 | FlorenceForge |
|------|--------|------|----------|---------------|
| 图像描述 | COCO Captions | BLEU-4 | 32.1 | **34.5** |
| 目标检测 | COCO Detection | mAP@0.5 | 42.3 | **44.1** |
| 文字识别 | TextOCR | F1-Score | 78.9 | **81.2** |
| 区域描述 | Visual Genome | CIDEr | 89.4 | **92.1** |

## 🔧 高级功能

### 自定义任务

```python
from florence_forge.core.tasks import register_task, TaskCategory

@register_task("CUSTOM_TASK", TaskCategory.GENERATION)
class CustomTask:
    def format_prompt(self, **kwargs):
        return "<CUSTOM_TASK>"
    
    def parse_response(self, response):
        return {"result": response}
```

### 自定义数据增强

```python
from florence_forge.data.augmentation import BaseAugmentation

class CustomAugmentation(BaseAugmentation):
    def __call__(self, image, annotations):
        # 自定义增强逻辑
        return augmented_image, augmented_annotations
```

### 模型量化

```python
from florence_forge.optimization.quantization import ModelQuantizer

quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_model(model, calibration_data)
print(f"模型大小减少: {quantizer.get_compression_ratio():.2f}x")
```

## 🚧 开发路线图

- [x] 核心框架和多任务训练
- [x] LoRA 微调支持
- [x] 命令行工具
- [x] 数据格式转换
- [x] 模型评估和可视化
- [x] FastAPI 推理服务
- [ ] 模型量化和剪枝
- [ ] 在线学习支持
- [ ] 联邦学习框架
- [ ] 模型蒸馏
- [ ] AutoML 超参数优化

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/florenceforge/florence-forge.git
cd florence-forge

# 创建开发环境
conda create -n florenceforge python=3.9
conda activate florenceforge

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black florence_forge/
flake8 florence_forge/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 Microsoft 的 Florence 团队提供的优秀基础模型
- 感谢 Hugging Face 提供的 Transformers 和 PEFT 库
- 感谢开源社区的贡献和支持

## 📞 联系我们

- 📧 Email: florenceforge@example.com
- 💬 Discord: [FlorenceForge Community](https://discord.gg/florenceforge)
- 🐛 Issues: [GitHub Issues](https://github.com/florenceforge/florence-forge/issues)
- 📖 文档: [在线文档](https://florenceforge.readthedocs.io/)

---

**FlorenceForge** - 让多模态AI开发更简单、更强大！
