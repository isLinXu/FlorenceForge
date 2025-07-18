# Florence Forge 数据模块

本模块提供了 Florence-2 模型的数据处理、转换和验证功能。

## 模块结构

```
florence_forge/data/
├── __init__.py              # 模块初始化文件
├── builder.py               # 数据集构建器
├── dataset.py               # 多任务数据集类
├── loader.py                # 数据加载器和采样器
├── converter.py             # 数据格式转换器（新增）
├── converter_example.py     # 转换器使用示例（新增）
└── README.md               # 本文档（新增）
```

## 核心功能

### 1. 数据集构建 (builder.py)

`DatasetBuilder` 类用于构建多任务数据集：

```python
from florence_forge.data import DatasetBuilder

builder = DatasetBuilder(image_base_path="/path/to/images")
builder.add_task_data("task1", "/path/to/task1.jsonl", weight=1.0)
builder.add_task_data("task2", "/path/to/task2.jsonl", weight=0.5)
dataset = builder.build(processor)
```

### 2. 多任务数据集 (dataset.py)

`MultiTaskDataset` 类支持多种 Florence-2 任务：

- 目标检测 (`<OD>`)
- 图像标题 (`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`)
- OCR (`<OCR>`, `<OCR_WITH_REGION>`)
- 视觉定位 (`<CAPTION_TO_PHRASE_GROUNDING>`)
- 区域分割 (`<REGION_TO_SEGMENTATION>`)
- 引用表达分割 (`<REFERRING_EXPRESSION_SEGMENTATION>`)
- 区域描述 (`<REGION_TO_DESCRIPTION>`)
- 区域分类 (`<REGION_TO_CATEGORY>`)
- 区域提议 (`<REGION_PROPOSAL>`)
- 密集区域描述 (`<DENSE_REGION_CAPTION>`)
- 开放词汇检测 (`<OPEN_VOCABULARY_DETECTION>`)

### 3. 数据加载 (loader.py)

`TaskBalancedSampler` 和 `TaskDataLoader` 提供任务平衡的数据加载：

```python
from florence_forge.data import TaskDataLoader

loader = TaskDataLoader(
    dataset=dataset,
    batch_size=8,
    task_balanced=True,
    num_workers=4
)
```

### 4. 数据格式转换 (converter.py) 🆕

`DataFormatConverter` 类提供多种数据格式到 Florence-2 格式的转换：

#### 支持的转换格式

| 源格式 | 目标任务 | 方法名 |
|--------|----------|--------|
| YOLO | 目标检测 | `yolo_to_florence2_od` |
| COCO | 目标检测 | `coco_to_florence2_od` |
| COCO | 区域分割 | `coco_to_florence2_region_segmentation` |
| CSV | 图像标题 | `csv_to_florence2_caption` |
| CSV | 区域分类 | `csv_to_florence2_region_category` |
| CSV | 区域描述 | `csv_to_florence2_region_description` |
| TXT | OCR | `txt_ocr_to_florence2` |
| VOC XML | 目标检测 | `voc_xml_to_florence2_od` |
| JSON | 视觉定位 | `json_to_florence2_grounding` |
| JSON | 区域提议 | `json_to_florence2_region_proposal` |
| JSON | 区域OCR | `json_to_florence2_ocr_with_region` |
| JSON | 引用表达分割 | `json_to_florence2_referring_expression_segmentation` |
| JSON | 区域分割 | `json_to_florence2_region_segmentation` |
| JSON | 密集区域描述 | `json_to_florence2_dense_region_caption` |
| JSON | 开放词汇检测 | `json_to_florence2_open_vocabulary_detection` |
| JSON | 带置信度检测 | `json_to_florence2_detection_with_confidence` |
| JSON | 带置信度定位 | `json_to_florence2_grounding_with_confidence` |

#### 使用示例

```python
from florence_forge.data import DataFormatConverter

# YOLO 转 Florence-2
DataFormatConverter.yolo_to_florence2_od(
    image_dir="/path/to/images",
    label_dir="/path/to/labels",
    output_path="/path/to/output.jsonl",
    class_names=["person", "car", "bicycle"]
)

# COCO 转 Florence-2
DataFormatConverter.coco_to_florence2_od(
    coco_json_path="/path/to/annotations.json",
    image_dir="/path/to/images",
    output_path="/path/to/output.jsonl"
)

# CSV标题 转 Florence-2
DataFormatConverter.csv_to_florence2_caption(
    csv_path="/path/to/captions.csv",
    output_path="/path/to/output.jsonl",
    task_type="CAPTION"
)
```

### 5. 数据验证 (converter.py) 🆕

`DataValidator` 类提供数据验证和报告生成功能：

```python
from florence_forge.data import DataValidator

# 验证数据
report = DataValidator.validate_florence2_jsonl(
    jsonl_path="/path/to/data.jsonl",
    image_base_path="/path/to/images"
)

# 生成报告
DataValidator.generate_validation_report(
    report=report,
    output_path="/path/to/report.md"
)
```

## 数据格式说明

### Florence-2 JSONL 格式

每行是一个 JSON 对象，包含以下字段：

```json
{
    "image": "/path/to/image.jpg",
    "prefix": "<TASK_TYPE>",
    "suffix": "task_specific_output",
    "text_input": "optional_text_input",
    "region": "optional_region_info"
}
```

### 任务特定格式

#### 目标检测
```json
{
    "image": "/path/to/image.jpg",
    "prefix": "<OD>",
    "suffix": "{\"<OD>\": {\"bboxes\": [[x1,y1,x2,y2]], \"bboxes_labels\": [\"person\"]}}"
}
```

#### 图像标题
```json
{
    "image": "/path/to/image.jpg",
    "prefix": "<CAPTION>",
    "suffix": "A person riding a bicycle on the street."
}
```

#### 视觉定位
```json
{
    "image": "/path/to/image.jpg",
    "prefix": "<CAPTION_TO_PHRASE_GROUNDING>",
    "suffix": "{\"<CAPTION_TO_PHRASE_GROUNDING>\": {\"bboxes\": [[x1,y1,x2,y2]], \"labels\": [\"person\"]}}",
    "text_input": "A person riding a bicycle"
}
```

## 安装和依赖

确保安装了以下依赖：

```bash
pip install torch torchvision
pip install transformers
pip install pillow
pip install pandas
pip install tqdm
```

## 快速开始

1. **准备数据**：将您的数据转换为 Florence-2 格式
2. **构建数据集**：使用 `DatasetBuilder` 创建多任务数据集
3. **训练模型**：使用 `florence_forge.training` 模块进行训练

详细示例请参考 `converter_example.py` 文件。

## 注意事项

1. **路径处理**：所有路径都会自动转换为绝对路径
2. **图像格式**：支持 JPG、JPEG、PNG、BMP 格式
3. **内存使用**：大型数据集建议使用数据流处理
4. **并行处理**：转换过程支持进度条显示
5. **错误处理**：转换过程中的错误会被记录和报告

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本模块！