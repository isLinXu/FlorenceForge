# 🎨 FlorenceForge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

**FlorenceForge** 是一个面向视觉语言模型的多任务微调、评估与部署框架。它以 Florence-2 为主路径，同时通过统一的 VLM 后端注册表支持 PaliGemma、YouTuVL 和通用 Hugging Face VLM 后端。

框架覆盖数据转换、JSONL 数据集、多任务调度、LoRA 微调、量化加载、评估指标、推理脚本和 FastAPI 服务，并使用 v2 模块化训练栈作为唯一训练入口。

## ✨ 核心特性

### 多任务 VLM 微调
- 支持 Florence-2 原生任务注册表中的图像描述、目标检测、短语定位、OCR、区域分析和分割任务。
- 支持单任务与多任务混合训练，任务权重可配置。
- 训练数据统一使用 JSONL：`image`、`prefix`、`suffix`。

### 后端与配置收敛
- `VLMBackendRegistry` 是 VLM 后端的单一事实源。
- `ArchitectureResolver` 保留为兼容门面，主要承接非 VLM 扩展和 builder 包装。
- `model_config.revision` 和部署侧 `--model-revision` 可 pin Hugging Face 模型/处理器版本。
- Pydantic v2 配置体系提供字段校验、兼容 alias 和 YAML/JSON 序列化。

### 训练与评估
- v2 训练栈提供清晰的模块边界与统一的训练入口。
- 支持 Accelerate、FSDP/DeepSpeed 配置、混合精度、梯度累积、梯度检查点和检查点管理。
- 评估模块包含多任务评估器、基础指标、高级指标、benchmark 缓存和可选 PDF 报告。

### 工具链
- CLI 覆盖 `doctor`、`train`、`eval`、`infer`、`serve`、`convert`、`validate`、`generate-config`。
- 数据转换支持 YOLO、COCO Detection、COCO Caption、CSV、VOC XML、OCR 目录和 OCR TXT。
- FastAPI 服务默认只监听本机，显式传入 `--host 0.0.0.0` 才对外暴露。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ 可选；CPU/MPS 也可用于轻量验证

### 安装

```bash
git clone https://github.com/florenceforge/florence-forge.git
cd florence-forge

# 全功能入口：核心依赖 + 可选扩展
pip install -r requirements.txt

# 开发环境
pip install -r requirements-core.txt -r requirements-dev.txt
pip install -e .

# 验证 CLI
florence_forge_cli --help
florence_forge_cli doctor --device auto
```

按需安装也可以更轻：

```bash
# 最小核心
pip install -r requirements-core.txt

# 评估指标、监控、PDF 报告、增强等扩展
pip install -r requirements-optional.txt

# pyproject extras
pip install -e ".[evaluation]"
pip install -e ".[dev]"
```

核心依赖已经显式约束主要兼容边界，例如 `accelerate<2`、`datasets<5`、`numpy<3`、`rich<16`。

### CLI 最小训练流程

```bash
# 1. 生成配置文件
florence_forge_cli generate-config --task caption --output my_config.yaml

# 2. 准备一行 JSONL 样例
echo '{"image": "path/to/image.jpg", "prefix": "<CAPTION>", "suffix": "A beautiful sunset"}' > data.jsonl

# 3. 启动训练。也可以在配置里写 train_data_path
florence_forge_cli train \
  --config my_config.yaml \
  --train-data data.jsonl \
  --epochs 5

# 4. v2 是唯一训练栈；--trainer-version 可省略
florence_forge_cli train \
  --config my_config.yaml \
  --train-data data.jsonl \
  --trainer-version v2
```

### coco128 LoRA 训练

仓库内提供了两份轻量模板，便于先用 coco128 跑通 OD LoRA 闭环：
模板默认设置 `save_full_model_on_end: false`，训练结束保存 LoRA adapter，
避免把基础大模型权重重复落盘；需要完整权重时可在 YAML 中改回 `true`。

```bash
# Florence-2：直接使用 Florence OD JSONL
florence_forge_cli train \
  --config configs/examples/coco128_florence_od_lora.yaml \
  --train-data /path/to/coco128/coco128_od.jsonl \
  --val-data /path/to/coco128/coco8_od.jsonl

# PaliGemma：先把 Florence OD suffix 转成 PaliGemma <loc0000> 格式
python scripts/data-conversion/convert_florence_od_to_paligemma.py \
  --input-jsonl /path/to/coco128/coco128_od.jsonl \
  --output-jsonl /path/to/coco128/coco128_paligemma_od.jsonl

python scripts/data-conversion/convert_florence_od_to_paligemma.py \
  --input-jsonl /path/to/coco128/coco8_od.jsonl \
  --output-jsonl /path/to/coco128/coco8_paligemma_od.jsonl

florence_forge_cli train \
  --config configs/examples/coco128_paligemma_od_lora.yaml \
  --train-data /path/to/coco128/coco128_paligemma_od.jsonl \
  --val-data /path/to/coco128/coco8_paligemma_od.jsonl
```

如果 PaliGemma 权重来自 ModelScope 或其他本地缓存，可直接覆盖模型路径：

```bash
florence_forge_cli train \
  --config configs/examples/coco128_paligemma_od_lora.yaml \
  --model /path/to/paligemma-3b-pt-224 \
  --train-data /path/to/coco128/coco128_paligemma_od.jsonl
```

### Python API

```python
from florence_forge import TrainingConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.training import MultiTaskTrainer

config = TrainingConfig.from_yaml("my_config.yaml")

model = Florence2MultiTaskModel(config.model_settings).load()

train_dataset = MultiTaskDataset(
    data_configs=[
        {
            "task_type": "CAPTION",
            "data_path": "data.jsonl",
            "weight": 1.0,
        }
    ],
    config=config.data_settings,
    processor=model.processor,
)

trainer = MultiTaskTrainer(
    model=model,
    train_dataset=train_dataset,
    config=config,
)

results = trainer.train()
print(results)
```

## 📖 数据与配置

### JSONL 数据格式

```json
{"image": "images/001.jpg", "prefix": "<CAPTION>", "suffix": "A small boat on the lake."}
{"image": "images/002.jpg", "prefix": "<OD>", "suffix": "<loc_10><loc_20><loc_100><loc_200>person"}
{"image": "images/003.jpg", "prefix": "<REGION_TO_DESCRIPTION><loc_50><loc_60><loc_150><loc_160>", "suffix": "a red sign"}
```

`image` 可以配合 `MultiTaskDataset(image_base_path=...)` 使用相对路径；额外字段会进入样本 metadata。

### 关键配置片段

```yaml
num_epochs: 5
max_steps: null
output_dir: "./outputs/florence2_caption"
gradient_accumulation_steps: 4
use_fp16: false
use_bf16: true

model_config:
  model_name: "microsoft/Florence-2-base"
  revision: null              # 生产环境建议写具体 commit hash
  backend_name: "florence-2"  # florence-2 / paligemma / youtuvl / generic-hf
  trust_remote_code: true
  torch_dtype: "auto"
  device: "auto"
  device_map: "auto"
  attn_implementation: "sdpa"
  use_lora: true
  lora_config:
    r: 32
    lora_alpha: 64
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

data_config:
  batch_size: 1
  num_workers: 0
  pin_memory: false
  shuffle: false
  use_augmentation: true
  augmentation_prob: 0.3

optimization_config:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1

task_scheduling_config:
  strategy: "weighted"
  temperature: 1.0
```

`max_steps` 一旦设置为正数，会优先于 `num_epochs` 成为训练硬上限。

## 🧰 命令行工具

### 训练

```bash
# 任务模板训练
florence_forge_cli train --task caption --epochs 5

# 自定义配置训练
florence_forge_cli train --config config.yaml --train-data train.jsonl --val-data val.jsonl

# 从检查点恢复
florence_forge_cli train --config config.yaml --resume outputs/checkpoint-1000

# 分布式训练
accelerate launch --multi_gpu florence_forge_cli train --config config.yaml

# 显式选择 v2 训练栈
florence_forge_cli train --config config.yaml --trainer-version v2
```

### 评估

```bash
florence_forge_cli eval --model outputs/final_model --data test_data.jsonl
florence_forge_cli eval --model outputs/final_model --data test_data.jsonl --output results.json
```

### 推理

```bash
# 单张图片
florence_forge_cli infer \
  --model outputs/final_model \
  --input image.jpg \
  --output results \
  --task-prompt "<CAPTION>"

# 目录批量推理
florence_forge_cli infer \
  --model outputs/final_model \
  --input images/ \
  --output results \
  --batch-size 4 \
  --use-amp
```

### 数据转换

```bash
# YOLO
florence_forge_cli convert yolo \
  --labels-dir labels \
  --images-dir images \
  --classes-file classes.txt \
  --output data.jsonl

# COCO Detection
florence_forge_cli convert coco \
  --json-file annotations.json \
  --images-dir images \
  --output data.jsonl

# COCO Caption
florence_forge_cli convert coco-caption \
  --json-file captions.json \
  --images-dir images \
  --output captions.jsonl

# CSV / VOC XML / OCR
florence_forge_cli convert csv --csv-file captions.csv --output data.jsonl
florence_forge_cli convert xml --xml-dir annotations --images-dir images --output data.jsonl
florence_forge_cli convert ocr --images-dir images --texts-dir texts --output data.jsonl
florence_forge_cli convert ocr-txt --txt-file ocr.tsv --images-dir images --output data.jsonl
```

### 推理服务

```bash
# 默认仅本机访问：127.0.0.1:8000
florence_forge_cli serve --model outputs/final_model --port 8000

# 对外暴露时必须显式指定 host，并自行配置鉴权和网络边界
florence_forge_cli serve --model outputs/final_model --host 0.0.0.0 --port 8000

# 生产环境建议 pin 模型/处理器 revision
florence_forge_cli serve \
  --model microsoft/Florence-2-base \
  --model-revision <commit-hash> \
  --port 8000

# 健康检查
curl http://127.0.0.1:8000/health
```

## 🏗️ 项目架构

```text
florence_forge/
├── cli/
│   ├── main.py                 # argparse 入口与子命令定义
│   ├── commands.py             # train / infer / serve / eval / convert handlers
│   ├── config_manager.py       # 配置管理
│   └── _helpers.py             # CLI 共享常量与工具
├── core/
│   ├── config.py               # Pydantic v2 配置体系
│   ├── model.py                # Florence2MultiTaskModel 与后端接入
│   ├── tasks.py                # Florence-2 任务注册表
│   ├── callbacks.py            # 回调管理
│   ├── yaml_config.py          # YAML 配置加载
│   ├── architecture_resolver.py # 兼容门面与非 VLM builder
│   └── backends/
│       ├── base_vlm.py         # BaseVLMBackend / VLMBackendRegistry
│       ├── florence2_backend.py
│       ├── paligemma_backend.py
│       ├── youtuvl_backend.py
│       └── generic_hf_backend.py
├── data/
│   ├── dataset.py              # MultiTaskDataset
│   ├── converter.py            # YOLO / COCO / CSV / XML / OCR 转换
│   ├── collate.py              # Florence2Collator
│   ├── image_cache.py          # 图像 payload LRU 缓存
│   ├── validator.py            # 数据校验
│   ├── builder.py              # 数据集构建工具
│   └── augmentation/           # 图像 / 文本 / bbox 增强
├── training/
│   ├── trainer_refactored.py   # v2 模块化训练器
│   ├── training_loop.py        # v2 训练循环
│   ├── checkpoint.py           # 目录式 checkpoint 兼容 API
│   ├── checkpoint_manager.py   # v2 checkpoint manager
│   ├── _checkpoint_io.py       # 共享安全序列化原语
│   ├── lora_manager.py
│   ├── scheduler.py
│   ├── monitoring.py
│   ├── visualizer.py
│   └── multi_dataset_trainer.py
├── evaluation/
│   ├── evaluator.py
│   ├── metrics.py
│   ├── analyzer.py
│   ├── benchmark*.py
│   └── advanced_metrics/
├── deployment/
│   ├── inference.py
│   ├── server.py
│   ├── backends.py
│   ├── exporter.py
│   └── optimizer.py
├── optimization/
│   └── quantization.py
├── experimental/
│   └── moe/
└── utils/
```

## 📚 API 速览

### 配置与模型

```python
from florence_forge import TrainingConfig
from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel

config = TrainingConfig.from_yaml("config.yaml")
model = Florence2MultiTaskModel(config.model_settings).load()
```

### 后端注册表

```python
from florence_forge.core.backends import VLMBackendRegistry, create_backend

print(VLMBackendRegistry.list_backends())
backend = create_backend("florence-2", config.model_settings)
```

当前内置后端名称包括 `florence-2`/`florence2`、`paligemma`、`paligemma-3b`、`youtuvl`/`youtu-vl`/`tencent-youtuvl`、`generic-hf`/`auto`/`hf`。

### 训练器

```python
from florence_forge.training import MultiTaskTrainer

trainer = MultiTaskTrainer(model=model, train_dataset=train_dataset, config=config)
```

### 评估器

```python
from florence_forge.evaluation import MultiTaskEvaluator

evaluator = MultiTaskEvaluator(model)
results = evaluator.evaluate_dataset(eval_dataset)
```

### 量化加载

```python
from florence_forge.optimization.quantization import ModelQuantizer, QuantizationConfig

quantizer = ModelQuantizer(QuantizationConfig(method="bnb-4bit"))
model, processor = quantizer.load_quantized_model("microsoft/Florence-2-base")
```

## 🎯 支持的 Florence-2 任务

| 任务类型 | 描述 | 是否需要文本输入 |
|---------|------|------------------|
| `CAPTION` | 基础图像描述 | 否 |
| `DETAILED_CAPTION` | 详细图像描述 | 否 |
| `MORE_DETAILED_CAPTION` | 更详细图像描述 | 否 |
| `CAPTION_TO_PHRASE_GROUNDING` | 标题到短语定位 | 是 |
| `DENSE_REGION_CAPTION` | 密集区域描述 | 否 |
| `OD` | 通用目标检测 | 否 |
| `OPEN_VOCABULARY_DETECTION` | 开放词汇检测 | 是 |
| `REGION_PROPOSAL` | 区域提议 | 否 |
| `REGION_TO_CATEGORY` | 区域到类别 | 是 |
| `REGION_TO_DESCRIPTION` | 区域到描述 | 是 |
| `OCR` | 光学字符识别 | 否 |
| `OCR_WITH_REGION` | 带区域 OCR | 否 |
| `REGION_TO_SEGMENTATION` | 区域到分割 | 是 |
| `REFERRING_EXPRESSION_SEGMENTATION` | 指代表达式分割 | 是 |

## 🧪 开发与验证

```bash
# 安装开发依赖
pip install -r requirements-core.txt -r requirements-dev.txt
pip install -e .

# 单元测试
pytest -q

# 覆盖率门禁示例
pytest tests --cov=florence_forge --cov-report=term-missing --cov-fail-under=35

# 可选 PDF 报告测试需要 reportlab
pip install "reportlab>=4.0.0,<5.0.0"
pytest tests/test_benchmark_reports.py -q
```

CI 和本地测试中，缺失可选依赖的能力会通过 `pytest.importorskip` 或框架内的 optional dependency guard 优雅跳过。

## 🗺️ 路线图

- [x] Pydantic v2 配置体系
- [x] VLM 后端抽象与注册表
- [x] Florence-2 / PaliGemma / YouTuVL / Generic HF 后端
- [x] 数据转换、校验和缓存增强
- [x] FastAPI 推理服务安全默认 host
- [x] v2 训练栈显式导出与 CLI 选择
- [ ] 继续补齐 v2 高级训练能力测试覆盖
- [ ] 进一步收敛 checkpoint 对外 API
- [ ] 扩充每个 CLI 子命令的端到端集成冒烟
- [ ] 增强公开 API 文档与示例覆盖

## 🤝 贡献指南

欢迎提交 issue、测试样例、后端适配、数据转换器和训练/评估改进。开发前建议先运行：

```bash
florence_forge_cli doctor --device auto
pytest -q
```

## 📄 许可证

本项目采用 MIT 许可证，详情见 [LICENSE](LICENSE)。

## 🙏 致谢

- Microsoft Florence 团队提供的 Florence-2 基础模型
- Hugging Face Transformers、PEFT、Accelerate 与 Datasets 生态
- PyTorch、FastAPI、Pydantic 和开源社区

---

**FlorenceForge** - 面向多任务 VLM 微调的可扩展工具链。
