# 测试和验证脚本

本目录包含用于测试、验证和质量检查的脚本工具。

## 脚本列表

### quick_test.py
快速测试脚本，用于基本功能的快速验证。

**功能特性：**
- 核心功能快速测试
- 基本配置验证
- 依赖检查
- 环境兼容性测试

**使用示例：**
```bash
# 运行快速测试
python quick_test.py

# 指定测试模块
python quick_test.py --module core

# 详细输出
python quick_test.py --verbose
```

### simple_test.py
简单测试脚本，提供基础的单元测试功能。

**功能特性：**
- 单元测试执行
- 基本断言检查
- 测试结果报告
- 错误日志记录

**使用示例：**
```bash
# 运行简单测试
python simple_test.py

# 指定测试文件
python simple_test.py --test-file test_data.py
```

### test_runner.py
测试运行器，提供完整的测试套件管理。

**功能特性：**
- 测试套件管理
- 并行测试执行
- 测试报告生成
- 覆盖率统计
- 持续集成支持

**使用示例：**
```bash
# 运行所有测试
python test_runner.py --all

# 运行特定测试套件
python test_runner.py --suite integration

# 生成覆盖率报告
python test_runner.py --coverage --report-html
```

### quick_check.py
快速检查脚本，用于系统状态和配置的快速诊断。

**功能特性：**
- 系统环境检查
- 配置文件验证
- 依赖版本检查
- 数据完整性验证
- 性能基准测试

**使用示例：**
```bash
# 系统检查
python quick_check.py --system

# 配置检查
python quick_check.py --config config.yaml

# 数据检查
python quick_check.py --data data_dir/
```

### validation_suite.py
验证套件，提供全面的数据和模型验证功能。

**功能特性：**
- 数据质量验证
- 模型性能验证
- 训练结果验证
- 回归测试
- 基准对比

**使用示例：**
```bash
# 数据验证
python validation_suite.py --validate-data dataset/

# 模型验证
python validation_suite.py --validate-model model.pth

# 完整验证
python validation_suite.py --full-validation
```

## 测试类型

### 单元测试
测试单个函数或类的功能。

```bash
# 运行单元测试
python test_runner.py --type unit
```

### 集成测试
测试模块间的交互和集成。

```bash
# 运行集成测试
python test_runner.py --type integration
```

### 端到端测试
测试完整的工作流程。

```bash
# 运行端到端测试
python test_runner.py --type e2e
```

### 性能测试
测试系统性能和资源使用。

```bash
# 运行性能测试
python test_runner.py --type performance
```

## 快速开始

### 1. 环境检查
```bash
# 检查测试环境
python quick_check.py --system
```

### 2. 快速测试
```bash
# 运行快速测试
python quick_test.py
```

### 3. 完整测试
```bash
# 运行完整测试套件
python test_runner.py --all
```

### 4. 验证结果
```bash
# 验证测试结果
python validation_suite.py --validate-results
```

## 测试配置

### 配置文件
测试配置通常存储在 `test_config.yaml` 中：

```yaml
testing:
  timeout: 300
  parallel: true
  workers: 4
  coverage:
    enabled: true
    threshold: 80
  reports:
    format: ["html", "xml", "json"]
    output_dir: "test_reports/"
```

### 环境变量
```bash
# 设置测试环境
export FLORENCE_TEST_MODE=true
export FLORENCE_TEST_DATA_DIR=/path/to/test/data
export FLORENCE_LOG_LEVEL=DEBUG
```

## 测试数据管理

### 测试数据准备
```bash
# 准备测试数据
python quick_check.py --prepare-test-data
```

### 数据清理
```bash
# 清理测试数据
python quick_check.py --cleanup-test-data
```

### 数据验证
```bash
# 验证测试数据
python validation_suite.py --validate-test-data
```

## 持续集成

### GitHub Actions 示例
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          python test_runner.py --ci
          python validation_suite.py --ci
```

### 本地 CI 模拟
```bash
# 模拟 CI 环境测试
python test_runner.py --ci-mode
```

## 测试报告

### HTML 报告
```bash
# 生成 HTML 测试报告
python test_runner.py --report-html --output reports/
```

### JSON 报告
```bash
# 生成 JSON 测试报告
python test_runner.py --report-json --output reports/test_results.json
```

### 覆盖率报告
```bash
# 生成覆盖率报告
python test_runner.py --coverage --coverage-report
```

## 性能基准

### 基准测试
```bash
# 运行性能基准测试
python validation_suite.py --benchmark
```

### 性能对比
```bash
# 对比不同版本性能
python validation_suite.py --compare-performance baseline.json current.json
```

## 故障排除

### 常见问题

1. **测试超时**
   ```bash
   # 增加超时时间
   python test_runner.py --timeout 600
   ```

2. **内存不足**
   ```bash
   # 减少并行测试数量
   python test_runner.py --workers 2
   ```

3. **依赖问题**
   ```bash
   # 检查依赖
   python quick_check.py --check-dependencies
   ```

### 调试模式
```bash
# 启用调试模式
python test_runner.py --debug --verbose
```

### 日志分析
```bash
# 分析测试日志
python validation_suite.py --analyze-logs test.log
```

## 最佳实践

### 测试编写
1. **遵循 AAA 模式**：Arrange, Act, Assert
2. **使用描述性测试名称**
3. **保持测试独立性**
4. **使用适当的断言**

### 测试维护
1. **定期更新测试数据**
2. **清理过时的测试**
3. **监控测试性能**
4. **维护测试文档**

### 测试策略
1. **金字塔测试策略**：更多单元测试，适量集成测试，少量端到端测试
2. **快速反馈**：优先运行快速测试
3. **并行执行**：利用并行测试提高效率
4. **持续监控**：集成到 CI/CD 流程

## 相关文档

- [开发指南](../../docs/development/)
- [故障排除](../../docs/development/troubleshooting.md)
- [API 参考](../../docs/reference/)
- [配置指南](../../docs/configuration/)

## 扩展开发

### 添加新测试
1. 在相应目录创建测试文件
2. 继承基础测试类
3. 实现测试方法
4. 更新测试配置

### 自定义验证器
```python
# 示例：自定义验证器
from florence_forge.testing import BaseValidator

class CustomValidator(BaseValidator):
    def validate(self, data):
        # 实现验证逻辑
        return validation_result
```

## 注意事项

- 测试前确保环境配置正确
- 大型测试套件可能需要较长时间
- 定期清理测试生成的临时文件
- 保持测试数据的版本控制
- 注意测试环境与生产环境的差异