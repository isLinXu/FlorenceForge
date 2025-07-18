# 示例和教程脚本

本目录包含各种使用示例、教程脚本和演示程序，帮助用户快速上手 Florence Forge。

## 脚本列表

### config_usage_example.py
配置使用示例脚本，展示各种配置场景的具体用法。

**功能特性：**
- 基础配置示例
- 高级配置模式
- 配置继承演示
- 动态配置更新
- 多环境配置

**使用示例：**
```bash
# 运行基础配置示例
python config_usage_example.py --example basic

# 高级配置示例
python config_usage_example.py --example advanced

# 多任务配置示例
python config_usage_example.py --example multitask

# 所有示例
python config_usage_example.py --all
```

### example_runner.py
示例运行器，提供统一的示例执行和管理界面。

**功能特性：**
- 示例目录管理
- 批量示例执行
- 示例结果对比
- 交互式示例选择
- 示例性能测试

**使用示例：**
```bash
# 列出所有可用示例
python example_runner.py --list

# 运行特定示例
python example_runner.py --run training_example

# 交互式选择示例
python example_runner.py --interactive

# 批量运行示例
python example_runner.py --batch --category training
```

### usage_examples.py
使用示例集合，包含各种常见使用场景的代码示例。

**功能特性：**
- 训练示例
- 推理示例
- 数据处理示例
- 模型评估示例
- 自定义任务示例

**使用示例：**
```bash
# 训练示例
python usage_examples.py --example training --task object_detection

# 推理示例
python usage_examples.py --example inference --model florence-2-base

# 数据处理示例
python usage_examples.py --example data_processing --format coco

# 查看所有示例
python usage_examples.py --list-examples
```

### run_all.py
批量运行脚本，用于执行多个脚本或示例的自动化工具。

**功能特性：**
- 批量脚本执行
- 依赖关系管理
- 执行顺序控制
- 错误处理和恢复
- 执行报告生成

**使用示例：**
```bash
# 运行所有示例
python run_all.py --examples

# 运行特定类别
python run_all.py --category training --parallel

# 生成执行报告
python run_all.py --examples --report --output report.html

# 错误时继续执行
python run_all.py --examples --continue-on-error
```

## 示例分类

### 基础示例

#### 快速开始示例
```python
# quick_start_example.py
from florence_forge import Florence2Model, Florence2Trainer

# 加载预训练模型
model = Florence2Model.from_pretrained("florence-2-base")

# 准备数据
data_config = {
    "dataset_name": "coco",
    "data_dir": "./data/coco",
    "batch_size": 16
}

# 创建训练器
trainer = Florence2Trainer(
    model=model,
    data_config=data_config,
    output_dir="./outputs/quick_start"
)

# 开始训练
trainer.train(epochs=5)
```

#### 简单推理示例
```python
# simple_inference_example.py
from florence_forge import Florence2Model
from PIL import Image

# 加载模型
model = Florence2Model.from_pretrained("florence-2-base")

# 加载图像
image = Image.open("example.jpg")

# 运行推理
results = model.predict(
    image=image,
    task="object_detection",
    prompt="Detect all objects"
)

print(f"检测结果: {results}")
```

### 训练示例

#### 目标检测训练
```python
# object_detection_training.py
from florence_forge import Florence2Trainer
from florence_forge.data import COCODataset

# 数据集配置
dataset = COCODataset(
    data_dir="./data/coco",
    split="train",
    transform_config={
        "resize": (384, 384),
        "normalize": True
    }
)

# 训练配置
training_config = {
    "epochs": 20,
    "learning_rate": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_steps": 1000
}

# 创建训练器
trainer = Florence2Trainer(
    model_name="florence-2-base",
    dataset=dataset,
    training_config=training_config,
    output_dir="./outputs/object_detection"
)

# 开始训练
trainer.train()
```

#### 多任务训练
```python
# multitask_training.py
from florence_forge import MultiTaskTrainer
from florence_forge.data import MultiTaskDataset

# 多任务数据集
datasets = {
    "object_detection": COCODataset("./data/coco"),
    "image_captioning": CaptionDataset("./data/captions"),
    "visual_grounding": GroundingDataset("./data/grounding")
}

# 任务权重
task_weights = {
    "object_detection": 1.0,
    "image_captioning": 0.5,
    "visual_grounding": 0.3
}

# 多任务训练器
trainer = MultiTaskTrainer(
    model_name="florence-2-large",
    datasets=datasets,
    task_weights=task_weights,
    output_dir="./outputs/multitask"
)

trainer.train(epochs=30)
```

### 数据处理示例

#### 数据格式转换
```python
# data_conversion_example.py
from florence_forge.data.converters import COCOToFlorence

# COCO 到 Florence 格式转换
converter = COCOToFlorence(
    input_dir="./data/coco",
    output_dir="./data/florence_format"
)

# 执行转换
converter.convert(
    split="train",
    include_images=True,
    validate_output=True
)

print("数据转换完成")
```

#### 自定义数据集
```python
# custom_dataset_example.py
from florence_forge.data import BaseDataset
from torch.utils.data import DataLoader

class CustomDataset(BaseDataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # 实现自定义数据加载逻辑
        pass
    
    def __getitem__(self, idx):
        # 实现数据获取逻辑
        pass

# 使用自定义数据集
dataset = CustomDataset("./data/custom")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### 推理示例

#### 批量推理
```python
# batch_inference_example.py
from florence_forge import Florence2Model
from florence_forge.inference import BatchInference
import glob

# 加载模型
model = Florence2Model.from_pretrained("./models/fine_tuned")

# 批量推理器
inference = BatchInference(
    model=model,
    batch_size=8,
    device="cuda"
)

# 获取图像列表
image_paths = glob.glob("./test_images/*.jpg")

# 执行批量推理
results = inference.predict_batch(
    image_paths=image_paths,
    task="object_detection",
    output_dir="./inference_results"
)

print(f"处理了 {len(results)} 张图像")
```

#### 实时推理
```python
# realtime_inference_example.py
from florence_forge import Florence2Model
from florence_forge.inference import RealtimeInference
import cv2

# 实时推理器
inference = RealtimeInference(
    model_name="florence-2-base",
    task="object_detection"
)

# 摄像头捕获
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 实时推理
    results = inference.predict(frame)
    
    # 绘制结果
    annotated_frame = inference.draw_results(frame, results)
    
    # 显示结果
    cv2.imshow('Florence-2 Real-time Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 评估示例

#### 模型评估
```python
# model_evaluation_example.py
from florence_forge.evaluation import ModelEvaluator
from florence_forge.data import COCODataset

# 准备测试数据
test_dataset = COCODataset(
    data_dir="./data/coco",
    split="val"
)

# 创建评估器
evaluator = ModelEvaluator(
    model_path="./models/fine_tuned/best.pth",
    dataset=test_dataset,
    metrics=["mAP", "precision", "recall"]
)

# 运行评估
results = evaluator.evaluate(
    output_dir="./evaluation_results",
    save_predictions=True,
    generate_report=True
)

print(f"评估结果: {results}")
```

#### 性能基准测试
```python
# benchmark_example.py
from florence_forge.benchmark import PerformanceBenchmark

# 性能基准测试
benchmark = PerformanceBenchmark(
    model_name="florence-2-base",
    test_images="./test_images",
    batch_sizes=[1, 4, 8, 16],
    input_sizes=[224, 384, 512]
)

# 运行基准测试
results = benchmark.run(
    warmup_iterations=10,
    measurement_iterations=100
)

# 生成报告
benchmark.generate_report(
    results=results,
    output_file="benchmark_report.html"
)
```

## 高级示例

### 自定义训练循环
```python
# custom_training_loop.py
from florence_forge import Florence2Model
from florence_forge.training import BaseTrainer
import torch

class CustomTrainer(BaseTrainer):
    def __init__(self, model, dataloader, optimizer):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
    
    def training_step(self, batch, batch_idx):
        # 自定义训练步骤
        images, targets = batch
        
        # 前向传播
        outputs = self.model(images, targets)
        
        # 计算损失
        loss = self.compute_loss(outputs, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def compute_loss(self, outputs, targets):
        # 自定义损失计算
        pass

# 使用自定义训练器
trainer = CustomTrainer(model, dataloader, optimizer)
trainer.fit(epochs=10)
```

### 模型微调
```python
# fine_tuning_example.py
from florence_forge import Florence2Model
from florence_forge.training import FineTuner

# 加载预训练模型
model = Florence2Model.from_pretrained("florence-2-base")

# 冻结部分层
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

# 微调器
fine_tuner = FineTuner(
    model=model,
    dataset=custom_dataset,
    learning_rate=1e-5,  # 较小的学习率
    epochs=10
)

# 开始微调
fine_tuner.fine_tune(
    output_dir="./outputs/fine_tuned",
    save_best_only=True
)
```

### 分布式训练
```python
# distributed_training_example.py
from florence_forge.training import DistributedTrainer
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend="nccl")

# 分布式训练器
trainer = DistributedTrainer(
    model_name="florence-2-large",
    dataset=dataset,
    world_size=4,
    rank=dist.get_rank()
)

# 开始分布式训练
trainer.train(
    epochs=50,
    sync_bn=True,
    find_unused_parameters=False
)
```

## 运行示例

### 快速运行

#### 运行单个示例
```bash
# 运行配置示例
python config_usage_example.py --example basic

# 运行训练示例
python usage_examples.py --example training --task object_detection
```

#### 批量运行示例
```bash
# 运行所有基础示例
python run_all.py --category basic

# 运行所有训练示例
python run_all.py --category training --parallel
```

### 交互式运行

#### 示例选择器
```bash
# 启动交互式示例选择器
python example_runner.py --interactive
```

这将显示一个菜单，让您选择要运行的示例：
```
=== Florence Forge 示例选择器 ===
1. 基础配置示例
2. 训练示例
3. 推理示例
4. 数据处理示例
5. 评估示例
6. 高级示例

请选择要运行的示例 (1-6): 
```

### 自定义运行

#### 参数化示例
```bash
# 使用自定义参数运行示例
python usage_examples.py --example training \
  --model florence-2-large \
  --dataset ./data/custom \
  --epochs 20 \
  --batch-size 16
```

## 示例配置

### 示例配置文件
```yaml
# examples_config.yaml
examples:
  basic:
    - name: "quick_start"
      script: "quick_start_example.py"
      description: "快速开始示例"
    
  training:
    - name: "object_detection"
      script: "object_detection_training.py"
      description: "目标检测训练"
      requirements: ["coco_dataset"]
    
  inference:
    - name: "batch_inference"
      script: "batch_inference_example.py"
      description: "批量推理示例"
```

### 环境配置
```bash
# 设置示例环境变量
export FLORENCE_EXAMPLES_DIR=/path/to/examples
export FLORENCE_DATA_DIR=/path/to/data
export FLORENCE_OUTPUT_DIR=/path/to/outputs
```

## 故障排除

### 常见问题

1. **依赖缺失**
   ```bash
   # 检查示例依赖
   python example_runner.py --check-dependencies
   ```

2. **数据路径错误**
   ```bash
   # 验证数据路径
   python example_runner.py --validate-data
   ```

3. **内存不足**
   ```bash
   # 使用较小的批处理大小
   python usage_examples.py --example training --batch-size 8
   ```

### 调试模式
```bash
# 启用详细调试输出
python example_runner.py --run training_example --debug --verbose
```

## 贡献示例

### 添加新示例

1. **创建示例脚本**
   ```python
   # new_example.py
   def main():
       """新示例的主函数"""
       pass
   
   if __name__ == "__main__":
       main()
   ```

2. **更新示例配置**
   ```yaml
   # 在 examples_config.yaml 中添加
   new_category:
     - name: "new_example"
       script: "new_example.py"
       description: "新示例描述"
   ```

3. **添加文档**
   - 在脚本中添加详细的文档字符串
   - 更新 README.md
   - 添加使用说明

### 示例最佳实践

1. **代码质量**
   - 遵循 PEP 8 代码风格
   - 添加类型注解
   - 包含错误处理

2. **文档化**
   - 清晰的函数和类文档
   - 使用示例和参数说明
   - 预期输出描述

3. **可重用性**
   - 参数化配置
   - 模块化设计
   - 易于扩展

## 相关文档

- [用户指南](../../docs/user-guides/)
- [API 参考](../../docs/reference/)
- [配置指南](../../docs/configuration/)
- [开发文档](../../docs/development/)

## 获取帮助

- **示例问题**：查看示例脚本中的注释和文档
- **技术支持**：参考主项目文档
- **社区讨论**：加入项目讨论区
- **问题报告**：在项目 Issues 页面提交问题

## 注意事项

- 运行示例前请确保已安装所有依赖
- 某些示例需要特定的数据集或模型文件
- 大型示例可能需要较长的运行时间
- 建议在虚拟环境中运行示例
- 注意示例的硬件要求（GPU、内存等）