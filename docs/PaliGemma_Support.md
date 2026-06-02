# FlorenceForge PaliGemma 支持文档

> **版本**: v1.0  
> **日期**: 2026-04-25  
> **状态**: ✅ 已实现并测试通过

---

## 目录

1. [概述](#概述)
2. [PaliGemma 模型特点](#paligemma-模型特点)
3. [使用方法](#使用方法)
4. [任务支持](#任务支持)
5. [与 Florence-2 的差异](#与-florence-2-的差异)
6. [配置示例](#配置示例)
7. [训练最佳实践](#训练最佳实践)
8. [故障排除](#故障排除)

---

## 概述

FlorenceForge 现已支持 **PaliGemma** 系列模型，包括：

- `google/paligemma-3b-pt-224`
- `google/paligemma-3b-pt-448`
- `google/paligemma-3b-mix-224`
- `google/paligemma-3b-ft-vqav2-448`
- 以及所有 transformers 兼容的 PaliGemma 变体

通过 VLM 后端抽象层，PaliGemma 与 Florence-2 共享统一的训练和推理接口，无需修改上层代码即可切换模型。

---

## PaliGemma 模型特点

| 特性 | PaliGemma | Florence-2 |
|------|-----------|------------|
| **模型大小** | 3B | 0.77B (base) / 0.77B (large) |
| **视觉编码器** | SigLIP (400M) | DaViT (232M) |
| **语言模型** | Gemma 2B | 自定义 Transformer |
| **任务格式** | 自然语言 prompt | 特殊 token (如 `<OD>`) |
| **图像尺寸** | 224x224 或 448x448 | 固定尺寸 |
| **生成方式** | Greedy / Beam Search | Beam Search |

### 架构优势

- **更强大的视觉理解**: SigLIP 视觉编码器在多个视觉基准上表现优异
- **更大的语言模型**: 2B 参数的 Gemma 语言模型提供更强的文本生成能力
- **自然语言接口**: 使用自然语言描述任务，更直观易懂

---

## 使用方法

### 1. 基本推理

```python
from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from PIL import Image

# 创建 PaliGemma 配置
config = ModelConfig(
    model_name="google/paligemma-3b-pt-224",
    backend_name="paligemma",
    use_lora=False,
)

# 加载模型
model = Florence2MultiTaskModel(config)

# 加载图像
image = Image.open("image.jpg").convert("RGB")

# 执行推理（自然语言任务提示）
result = model.predict_task(
    images=image,
    task_name="CAPTION",
    max_new_tokens=256,
)
print(result)
```

### 2. 使用 Registry 直接创建后端

```python
from florence_forge.core.backends import VLMBackendRegistry
from florence_forge.core.config import ModelConfig

config = ModelConfig(
    model_name="google/paligemma-3b-pt-224",
    backend_name="paligemma",
)

backend = VLMBackendRegistry.create("paligemma", config)

# 查看支持的后端
print(VLMBackendRegistry.list_backends())
# ['florence-2', 'florence2', 'paligemma', 'paligemma-3b']
```

### 3. 命令行示例

```bash
# 使用 PaliGemma 进行图像描述
python examples/multi_backend_example.py \
    --backend paligemma \
    --task CAPTION \
    --image image.jpg \
    --mode inference

# 对比 Florence-2 和 PaliGemma
python examples/multi_backend_example.py \
    --image image.jpg \
    --mode compare
```

---

## 任务支持

PaliGemma 使用自然语言任务提示，与 Florence-2 的特殊 token 格式不同：

| 任务 | Florence-2 Prompt | PaliGemma Prompt |
|------|-------------------|------------------|
| 图像描述 | `<CAPTION>` | `caption` |
| 详细描述 | `<DETAILED_CAPTION>` | `caption` |
| 目标检测 | `<OD>` | `detect` |
| 区域提议 | `<REGION_PROPOSAL>` | `detect` |
| OCR | `<OCR>` | `ocr` |
| 视觉问答 | `<VQA>` | `answer` |

### 自定义任务提示

```python
from florence_forge.core.backends import VLMBackendRegistry

backend = VLMBackendRegistry.create("paligemma", config)

# 获取默认任务提示
print(backend.get_task_prompt("CAPTION"))  # "caption"

# PaliGemma 支持任意自然语言提示
# 例如:
# "Describe this image in detail"
# "What objects are in this image?"
# "Translate the text in this image to English"
```

---

## 与 Florence-2 的差异

### 1. 数据预处理

PaliGemma 的 processor 处理图像和文本的方式与 Florence-2 略有不同：

- **图像尺寸**: PaliGemma 支持 224x224 和 448x448
- **文本格式**: 使用自然语言而非特殊 token
- **Tokenizer**: 使用 Gemma 的 SentencePiece tokenizer

**框架已自动处理这些差异**，用户无需手动调整数据预处理代码。

### 2. 训练参数建议

| 参数 | PaliGemma 建议 | Florence-2 建议 |
|------|---------------|----------------|
| Batch Size | 2-4 (16GB VRAM) | 4-8 (16GB VRAM) |
| Learning Rate | 1e-4 ~ 5e-5 | 1e-4 ~ 5e-5 |
| LoRA r | 16-32 | 16-32 |
| Max Grad Norm | 1.0 | 1.0 |
| Precision | bfloat16 | float16 |

### 3. 生成参数差异

```python
# PaliGemma 通常使用 greedy decoding 即可
result = model.predict_task(
    images=image,
    task_name="CAPTION",
    max_new_tokens=256,
    num_beams=1,        # PaliGemma 推荐 1
    do_sample=False,    # PaliGemma 推荐 False
)

# Florence-2 通常使用 beam search
result = model.predict_task(
    images=image,
    task_name="CAPTION",
    max_new_tokens=256,
    num_beams=3,        # Florence-2 推荐 3
    do_sample=False,
)
```

---

## 配置示例

### YAML 配置

```yaml
# configs/paligemma_caption.yaml
model:
  model_name: "google/paligemma-3b-pt-224"
  backend_name: "paligemma"
  trust_remote_code: true
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules:
      - "q_proj"
      - "v_proj"
      - "k_proj"
      - "o_proj"
    lora_dropout: 0.05

data:
  batch_size: 4
  num_workers: 2
  use_cache: true
  cache_dir: "./cache/paligemma"

optimization:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1

training:
  num_epochs: 10
  output_dir: "./output/paligemma_caption"
  device: "auto"
  use_bf16: true
```

### Python 配置

```python
from florence_forge.core.config import (
    ModelConfig, DataConfig, TrainingConfig,
    OptimizationConfig, LoRAConfig
)

model_config = ModelConfig(
    model_name="google/paligemma-3b-pt-224",
    backend_name="paligemma",
    use_lora=True,
    lora_config=LoRAConfig(r=16, lora_alpha=32),
)

data_config = DataConfig(
    batch_size=4,
    num_workers=2,
    use_cache=True,
    cache_dir="./cache/paligemma",
)

training_config = TrainingConfig(
    num_epochs=10,
    model_config=model_config,
    data_config=data_config,
    optimization_config=OptimizationConfig(
        learning_rate=1e-4,
        use_bf16=True,
    ),
    output_dir="./output/paligemma_caption",
)
```

---

## 训练最佳实践

### 1. 显存优化

PaliGemma 3B 比 Florence-2 大，需要更多显存：

```python
# 使用 QLoRA 进一步降低显存占用
from florence_forge.core.config import ModelConfig, LoRAConfig

config = ModelConfig(
    model_name="google/paligemma-3b-pt-224",
    backend_name="paligemma",
    use_lora=True,
    lora_config=LoRAConfig(
        r=8,              # 更小的 rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 更少的 target 模块
    ),
    torch_dtype="auto",
)
```

### 2. 数据缓存

强烈建议启用预编码缓存，因为 PaliGemma 的 processor 较重：

```python
data_config = DataConfig(
    batch_size=4,
    num_workers=2,
    use_cache=True,              # 启用缓存
    cache_dir="./cache/paligemma",  # 磁盘缓存目录
)
```

### 3. 混合精度

PaliGemma 推荐使用 bfloat16（如果硬件支持）：

```python
# 自动检测
optimization_config = OptimizationConfig(
    learning_rate=1e-4,
    # 训练器会自动选择最佳精度:
    # CUDA + Ampere+: bfloat16
    # CUDA + 其他: float16
    # MPS/CPU: float32
)
```

### 4. 多任务训练

PaliGemma 支持多任务训练，但任务提示使用自然语言：

```python
data_configs = [
    {"task_type": "CAPTION", "data_path": "data/captions.json"},
    {"task_type": "OD", "data_path": "data/detection.json"},
    {"task_type": "VQA", "data_path": "data/vqa.json"},
]
```

---

## 故障排除

### 问题 1: `ImportError: cannot import name 'PaliGemmaForConditionalGeneration'`

**原因**: transformers 版本过低

**解决**:
```bash
pip install transformers>=4.40.0
```

### 问题 2: `RuntimeError: CUDA out of memory`

**原因**: PaliGemma 3B 模型较大

**解决**:
```python
# 1. 减小 batch size
DataConfig(batch_size=2)

# 2. 使用更小的 LoRA rank
LoRAConfig(r=8, target_modules=["q_proj", "v_proj"])

# 3. 启用梯度累积
TrainingConfig(gradient_accumulation_steps=4)

# 4. 使用更短的序列长度
# 在数据预处理中限制最大长度
```

### 问题 3: 生成结果质量差

**原因**: PaliGemma 使用自然语言提示，需要精确的 prompt 格式

**解决**:
```python
# 确保使用正确的任务提示
backend = VLMBackendRegistry.create("paligemma", config)
print(backend.get_task_prompt("CAPTION"))  # "caption"

# 对于自定义任务，使用清晰的自然语言描述
custom_prompt = "Describe this image in one sentence"
```

### 问题 4: 处理器加载失败

**原因**: PaliGemma 的 processor 需要特定文件

**解决**:
```python
# 确保模型文件完整下载
from huggingface_hub import snapshot_download
snapshot_download("google/paligemma-3b-pt-224")
```

---

## 参考资源

- **PaliGemma 论文**: [arXiv:2407.07726](https://arxiv.org/abs/2407.07726)
- **Hugging Face 文档**: [PaliGemma](https://huggingface.co/docs/transformers/model_doc/paligemma)
- **参考实现**:
  - [roboflow/maestro](https://github.com/roboflow/maestro) - 多 VLM 统一训练接口
  - [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) - 轻量 VLM 极简实现

---

## 附录: 新增/修改文件清单

### 新增文件

```
florence_forge/core/backends/paligemma_backend.py    # PaliGemmaBackend 实现
tests/test_paligemma_backend.py                       # PaliGemma 单元测试
examples/multi_backend_example.py                     # 多后端统一示例
configs/paligemma_caption.yaml                        # PaliGemma 配置模板
docs/PaliGemma_Support.md                             # 本文档
```

### 修改文件

```
florence_forge/core/backends/__init__.py              # 导出 PaliGemmaBackend
```
