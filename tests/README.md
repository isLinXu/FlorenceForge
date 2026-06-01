# FlorenceForge 测试套件

本目录包含 FlorenceForge 项目的自动化测试。

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # Pytest 配置和共享 fixtures
├── test_core_model.py       # 核心模型和后端测试
├── test_data_pipeline.py    # 数据管线测试
└── README.md                # 本文件
```

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定文件
```bash
pytest tests/test_core_model.py -v
```

### 运行特定测试类或方法
```bash
pytest tests/test_core_model.py::TestFlorence2MultiTaskModel::test_model_to_device -v
```

### 排除慢速测试
```bash
pytest tests/ -m "not slow"
```

### 运行集成测试
```bash
pytest tests/ -m integration
```

### 生成覆盖率报告
```bash
pytest tests/ --cov=florence_forge --cov-report=html
```

## 测试标记

- `@pytest.mark.slow`: 标记为慢速测试
- `@pytest.mark.gpu`: 需要 GPU 的测试
- `@pytest.mark.integration`: 集成测试

## 测试覆盖范围

### ✅ 已实现

1. **核心模型测试** (`test_core_model.py`)
   - 模型初始化
   - 模型加载
   - 设备转移（包括后端设备同步验证）
   - 模型信息获取
   - VLM 后端注册表
   - 后端创建和错误处理

2. **数据管线测试** (`test_data_pipeline.py`)
   - TaskSample 数据结构
   - MultiTaskDataset 初始化
   - 样本获取
   - 延迟加载模式
   - OrderedDict LRU 缓存
   - 任务统计
   - 图像 LRU 缓存

### 🔄 待扩展

1. **训练流程测试**
   - TaskScheduler 任务调度
   - LoRAManager LoRA 管理
   - MultiTaskTrainer 训练循环
   - Callback 系统

2. **评估测试**
   - MultiTaskEvaluator
   - 各类评估指标

3. **部署测试**
   - InferenceEngine
   - FastAPI 服务器
   - 模型导出

## 添加新测试

创建新测试文件时：

1. 文件名以 `test_` 开头
2. 测试类以 `Test` 开头
3. 测试方法以 `test_` 开头
4. 使用描述性的测试名称

示例：
```python
class TestNewFeature:
    def test_feature_basic_functionality(self):
        # 测试代码
        pass
```

## CI/CD 集成

测试应在以下情况自动运行：
- 每次提交到主分支
- Pull Request 创建时
- 定期（每日/每周）

## 依赖

测试依赖已在 `pyproject.toml` 中定义：
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-asyncio >= 0.21.0

安装测试依赖：
```bash
pip install -e ".[dev]"
```
