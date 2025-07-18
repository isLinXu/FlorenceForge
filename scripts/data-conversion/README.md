# 数据转换脚本

本目录包含用于数据格式转换和批量数据处理的脚本工具。

## 脚本列表

### batch_data_conversion.py
批量数据转换工具，支持多种数据格式的批量处理。

**功能特性：**
- 批量转换多个数据文件
- 支持多种输入输出格式
- 进度跟踪和错误处理
- 并行处理支持

**使用示例：**
```bash
# 批量转换 COCO 格式到 Florence 格式
python batch_data_conversion.py --input coco_data/ --output florence_data/ --format florence

# 指定并行处理数量
python batch_data_conversion.py --input data/ --output converted/ --workers 4
```

### data_conversion_examples.py
数据转换示例脚本，展示各种转换场景的具体实现。

**包含示例：**
- COCO 到 Florence 格式转换
- YOLO 到 Florence 格式转换
- 自定义格式转换
- 数据验证和清理

**使用示例：**
```bash
# 运行 COCO 转换示例
python data_conversion_examples.py --example coco

# 运行所有示例
python data_conversion_examples.py --all
```

### convert/ 目录
包含特定格式的转换脚本。

#### convert_yolo.sh
YOLO 格式数据转换的 Shell 脚本。

**功能：**
- YOLO 标注格式转换
- 批量处理 YOLO 数据集
- 自动化转换流程

**使用示例：**
```bash
# 转换 YOLO 数据集
./convert/convert_yolo.sh input_dir output_dir

# 指定配置文件
./convert/convert_yolo.sh input_dir output_dir config.yaml
```

## 支持的数据格式

### 输入格式
- **COCO**: Microsoft COCO 数据集格式
- **YOLO**: YOLO 标注格式
- **Pascal VOC**: Pascal VOC XML 格式
- **自定义 JSON**: 用户自定义的 JSON 格式

### 输出格式
- **Florence**: Florence-2 训练格式
- **JSONL**: 行分隔的 JSON 格式
- **CSV**: 逗号分隔值格式

## 快速开始

1. **准备数据：**
   ```bash
   # 确保数据目录结构正确
   mkdir -p input_data output_data
   ```

2. **运行转换：**
   ```bash
   # 基本转换
   python batch_data_conversion.py --input input_data/ --output output_data/
   ```

3. **验证结果：**
   ```bash
   # 检查转换结果
   python data_conversion_examples.py --validate output_data/
   ```

## 配置选项

### 通用参数
- `--input`: 输入数据目录
- `--output`: 输出数据目录
- `--format`: 目标格式 (florence, jsonl, csv)
- `--workers`: 并行处理数量
- `--batch-size`: 批处理大小
- `--validate`: 转换后验证数据

### 高级选项
- `--filter`: 数据过滤条件
- `--transform`: 数据变换配置
- `--resume`: 从中断点恢复转换
- `--dry-run`: 预览转换操作

## 性能优化

### 并行处理
```bash
# 使用多进程加速转换
python batch_data_conversion.py --workers 8 --batch-size 100
```

### 内存优化
```bash
# 处理大型数据集时的内存优化
python batch_data_conversion.py --chunk-size 1000 --low-memory
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少 `--batch-size` 参数
   - 使用 `--low-memory` 选项
   - 增加系统交换空间

2. **转换失败**
   - 检查输入数据格式
   - 验证文件权限
   - 查看详细错误日志

3. **性能问题**
   - 调整 `--workers` 参数
   - 使用 SSD 存储
   - 优化数据预处理

### 调试模式
```bash
# 启用详细日志
python batch_data_conversion.py --verbose --log-level DEBUG

# 保存转换日志
python batch_data_conversion.py --log-file conversion.log
```

## 相关文档

- [数据转换指南](../../docs/user-guides/DATA_CONVERSION_GUIDE.md)
- [配置文件说明](../../docs/configuration/)
- [API 参考](../../docs/reference/)

## 扩展开发

### 添加新格式支持
1. 在 `florence_forge/data/converters/` 中添加转换器
2. 更新 `batch_data_conversion.py` 中的格式注册
3. 添加相应的测试用例

### 自定义转换逻辑
```python
# 示例：自定义转换器
from florence_forge.data.converters import BaseConverter

class CustomConverter(BaseConverter):
    def convert(self, data):
        # 实现转换逻辑
        return converted_data
```

## 注意事项

- 转换前请备份原始数据
- 大型数据集转换可能需要较长时间
- 确保有足够的磁盘空间存储转换结果
- 建议在转换前验证数据格式的正确性