# YouTu-VL 支持文档

FlorenceForge 现已支持腾讯优图实验室开源的 **Youtu-VL** / **Youtu-VL-4B-Instruct** 视觉语言模型。

---

## 模型概述

| 属性 | 说明 |
|------|------|
| **发布机构** | 腾讯优图实验室 (Tencent YouTu Lab) |
| **参数量** | 4B（40亿参数） |
| **架构** | Encoder-Decoder（视觉编码器 + Youtu-LLM 解码器） |
| **核心技术** | VLUAS（视觉-语言统一自回归监督） |
| **模型主页** | [ModelScope](https://modelscope.cn/models/tencent-YouTu/Youtu-VL) |
| **HuggingFace** | `tencent-YouTu/Youtu-VL-4B-Instruct` |

### 核心特点

- **统一自回归框架**：所有任务（检测、描述、VQA、OCR 等）都建模为序列生成问题，无需任务专用解码模块
- **轻量高效**：4B 参数即可实现多任务统一处理
- **结构化输出**：检测结果使用 `<ref>...</ref><box>...</box>` 标记格式
- **多平台同步**：HuggingFace / ModelScope / GitHub 三端同步更新

---

## 快速开始

### 1. 基础推理

```python
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.config import ModelConfig

config = ModelConfig(
    model_name="tencent-YouTu/Youtu-VL-4B-Instruct",
    backend_name="youtuvl",
)

model = Florence2MultiTaskModel(config)

from PIL import Image
image = Image.open("example.jpg")

# VQA
answer = model.predict_task(image, "VQA", text_input="What is in the image?")
print(answer)

# Caption
caption = model.predict_task(image, "CAPTION")
print(caption)

# OCR
text = model.predict_task(image, "OCR")
print(text)
```

### 2. 直接使用后端

```python
from florence_forge.core.backends import VLMBackendRegistry

backend = VLMBackendRegistry.create("youtuvl", config)

# 便捷推理方法
result = backend.generate_with_task(
    image="example.jpg",
    task_name="OD",
    max_new_tokens=512,
)

# 解析检测结果
detections = backend.parse_detection_output(result)
for det in detections:
    print(f"Label: {det['label']}, BBox: {det['bbox']}")
```

### 3. 使用 Auto 自动选择

```python
from florence_forge.core.backends import auto_select_backend

config = ModelConfig(model_name="tencent-YouTu/Youtu-VL-4B-Instruct")
backend = auto_select_backend(config)  # 自动识别并加载 YouTuVLBackend
```

---

## 配置示例

### YAML 配置

```yaml
# configs/youtuvl_vqa.yaml
model_config:
  model_name: "tencent-YouTu/Youtu-VL-4B-Instruct"
  backend_name: "youtuvl"
  trust_remote_code: true
  device: "auto"

data_config:
  batch_size: 4
  num_workers: 2
  use_cache: true
  cache_dir: "./cache_youtuvl"

training_config:
  num_epochs: 5
  learning_rate: 2e-5
  tasks:
    - "VQA"
    - "CAPTION"
    - "OCR"
  output_dir: "./outputs/youtuvl"
```

### 自定义任务 Prompt

```python
config = ModelConfig(
    model_name="tencent-YouTu/Youtu-VL-4B-Instruct",
    backend_name="youtuvl",
    task_prompts={
        "MY_CUSTOM_TASK": "Perform my custom analysis on this image.",
    },
    supported_tasks=["VQA", "CAPTION", "MY_CUSTOM_TASK"],
)
```

---

## 支持的任务

| 任务名称 | Prompt | 输出格式 |
|---------|--------|---------|
| `CAPTION` | Describe the image in detail. | 自然语言描述 |
| `DETAILED_CAPTION` | Provide a comprehensive description of the image. | 详细描述 |
| `OD` | Detect all objects in the image and provide their locations. | `<ref>label</ref><box>...</box>` |
| `DENSE_REGION_CAPTION` | Detect all objects and describe each region in detail. | 结构化文本 |
| `VISUAL_GROUNDING` | Locate the object described by the text in the image. | 边界框坐标 |
| `VQA` | Answer the question based on the image. | 自然语言回答 |
| `OCR` | Read all text present in the image. | 文本列表 |
| `OCR_WITH_REGION` | Read all text in the image and provide their locations. | 带位置的文本 |
| `SEGMENTATION` | Segment all objects in the image. | 分割掩码描述 |
| `POSE_ESTIMATION` | Estimate the pose of all persons in the image. | 关键点坐标 |
| `GUI` | Understand the GUI interface and answer the question. | 界面分析结果 |
| `DOCUMENT` | Extract information from the document image. | 结构化文档信息 |

---

## 输出解析

### 检测结果解析

YouTu-VL 的检测输出使用结构化标记：

```text
<ref>cat</ref><box>
  <x_min>100</x_min>
  <y_min>200</y_min>
  <x_max>300</x_max>
  <y_max>400</y_max>
</box>
```

使用内置解析器：

```python
detections = backend.parse_detection_output(raw_text)
# [
#     {"label": "cat", "bbox": [100, 200, 300, 400]},
# ]
```

---

## 训练最佳实践

### 1. 输入分辨率

YouTu-VL 推荐输入分辨率为 **512×512**，但支持动态尺寸：

```python
# 在数据预处理中调整图像大小
image = image.resize((512, 512))
```

### 2. 批大小建议

| 设备 | 推荐 batch_size |
|------|----------------|
| RTX 4090 (24GB) | 4-8 |
| RTX 4080 (16GB) | 2-4 |
| A100 (80GB) | 16-32 |

### 3. 学习率

YouTu-VL 作为 4B 参数模型，建议使用稍高的学习率：

```yaml
optimization_config:
  learning_rate: 2e-5  # 比 Florence-2 的 1e-5 稍高
  warmup_ratio: 0.1
```

### 4. LoRA 配置

```python
from florence_forge.core.config import LoRAConfig

lora_config = LoRAConfig(
    r=16,           # YouTu-VL 较小，r=16 通常足够
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
)
```

---

## 与其他模型对比

| 特性 | Florence-2 | PaliGemma | YouTu-VL |
|------|-----------|-----------|----------|
| **参数规模** | 0.23B-0.77B | 3B | 4B |
| **架构** | Encoder-Decoder | Decoder-only | Encoder-Decoder |
| **任务统一** | 特殊 token (`<OD>`) | 自然语言 prompt | VLUAS 自回归统一 |
| **检测输出** | 文本坐标 | 文本坐标 | XML 结构化标记 |
| **中文支持** | 有限 | 有限 | **原生优秀** |
| **文档理解** | 一般 | 一般 | **优秀** |
| **GUI 理解** | 不支持 | 不支持 | **支持** |

---

## 故障排除

### 问题：模型加载失败，提示 trust_remote_code

**解决**：确保配置中设置 `trust_remote_code=True`：

```python
config = ModelConfig(
    model_name="tencent-YouTu/Youtu-VL-4B-Instruct",
    trust_remote_code=True,
)
```

### 问题：AutoModelForVision2Seq 弃用警告

**说明**：transformers v5 将移除 `AutoModelForVision2Seq`，FlorenceForge 已自动适配为 `AutoModelForImageTextToText`。如果看到警告，请升级 transformers：

```bash
pip install transformers -U
```

### 问题：中文输出乱码

**解决**：YouTu-VL 原生支持中文，如果出现乱码，请检查 tokenizer 是否正确加载：

```python
# 验证 tokenizer 词表
print(backend._processor.vocab_size)
```

### 问题：检测输出无法解析

**解决**：确保使用 `parse_detection_output` 方法，或检查模型输出格式是否符合预期：

```python
raw_output = backend.generate_with_task(image, "OD")
print(raw_output)  # 先打印原始输出确认格式
```

---

## 参考链接

- [Youtu-VL ModelScope 主页](https://modelscope.cn/models/tencent-YouTu/Youtu-VL)
- [Youtu-VL-4B-Instruct HuggingFace](https://huggingface.co/tencent-YouTu/Youtu-VL-4B-Instruct)
- [腾讯优图实验室](https://github.com/Tencent/YouTu)
