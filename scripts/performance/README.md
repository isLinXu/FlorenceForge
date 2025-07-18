# 性能优化和质量检查脚本

本目录包含用于性能优化、基准测试和代码质量检查的脚本工具。

## 脚本列表

### performance_optimizer.py
性能优化工具，提供系统性能分析和优化建议。

**功能特性：**
- 训练性能分析
- 内存使用优化
- GPU 利用率监控
- 数据加载优化
- 模型推理加速
- 性能瓶颈识别

**使用示例：**
```bash
# 分析训练性能
python performance_optimizer.py --analyze-training --config config.yaml

# 优化数据加载
python performance_optimizer.py --optimize-dataloader --batch-size 32

# GPU 性能监控
python performance_optimizer.py --monitor-gpu --duration 300

# 生成优化报告
python performance_optimizer.py --generate-report --output perf_report.html
```

### benchmark_tools.py
基准测试工具，提供标准化的性能基准测试。

**功能特性：**
- 模型训练基准
- 推理速度基准
- 内存使用基准
- 数据处理基准
- 多设备对比
- 历史性能对比

**使用示例：**
```bash
# 运行完整基准测试
python benchmark_tools.py --full-benchmark

# 训练基准测试
python benchmark_tools.py --benchmark-training --model florence-2-base

# 推理基准测试
python benchmark_tools.py --benchmark-inference --batch-sizes 1,4,8,16

# 对比不同配置
python benchmark_tools.py --compare-configs config1.yaml config2.yaml
```

### code_quality_checker.py
代码质量检查工具，提供代码质量分析和改进建议。

**功能特性：**
- 代码风格检查
- 复杂度分析
- 安全漏洞扫描
- 依赖分析
- 文档完整性检查
- 测试覆盖率分析

**使用示例：**
```bash
# 完整代码质量检查
python code_quality_checker.py --full-check

# 代码风格检查
python code_quality_checker.py --style-check --fix

# 安全扫描
python code_quality_checker.py --security-scan

# 生成质量报告
python code_quality_checker.py --generate-report --format html
```

## 性能优化类型

### 训练性能优化

#### GPU 利用率优化
```bash
# 分析 GPU 利用率
python performance_optimizer.py --gpu-analysis

# 优化 GPU 内存使用
python performance_optimizer.py --optimize-gpu-memory
```

#### 数据加载优化
```bash
# 优化数据加载器
python performance_optimizer.py --optimize-dataloader \
  --num-workers 8 \
  --prefetch-factor 2 \
  --pin-memory
```

#### 混合精度训练
```bash
# 启用混合精度优化
python performance_optimizer.py --mixed-precision \
  --amp-level O1
```

### 推理性能优化

#### 模型量化
```bash
# 模型量化优化
python performance_optimizer.py --quantize-model \
  --model model.pth \
  --precision int8
```

#### 批处理优化
```bash
# 优化批处理大小
python performance_optimizer.py --optimize-batch-size \
  --target-latency 100ms
```

## 基准测试套件

### 标准基准测试

#### 训练基准
```bash
# Florence-2 训练基准
python benchmark_tools.py --benchmark florence2-training \
  --dataset coco \
  --epochs 1 \
  --batch-sizes 8,16,32
```

#### 推理基准
```bash
# 推理速度基准
python benchmark_tools.py --benchmark inference \
  --model florence-2-base \
  --input-sizes 224,384,512
```

### 自定义基准

#### 创建基准配置
```yaml
# benchmark_config.yaml
benchmark:
  name: "custom_benchmark"
  model: "florence-2-large"
  dataset: "custom_dataset"
  metrics:
    - "throughput"
    - "latency"
    - "memory_usage"
  iterations: 100
```

```bash
# 运行自定义基准
python benchmark_tools.py --config benchmark_config.yaml
```

## 代码质量检查

### 静态分析

#### 代码风格
```bash
# PEP 8 风格检查
python code_quality_checker.py --pep8-check

# 自动修复风格问题
python code_quality_checker.py --pep8-fix
```

#### 复杂度分析
```bash
# 圈复杂度分析
python code_quality_checker.py --complexity-analysis

# 函数长度检查
python code_quality_checker.py --function-length-check
```

### 安全检查

#### 漏洞扫描
```bash
# 安全漏洞扫描
python code_quality_checker.py --security-scan \
  --severity high,medium
```

#### 依赖安全检查
```bash
# 检查依赖安全性
python code_quality_checker.py --dependency-security
```

## 性能监控

### 实时监控

#### 系统资源监控
```bash
# 启动性能监控
python performance_optimizer.py --monitor \
  --metrics cpu,memory,gpu \
  --interval 5 \
  --duration 3600
```

#### 训练监控
```bash
# 训练过程监控
python performance_optimizer.py --monitor-training \
  --log-file training.log \
  --alert-thresholds memory:80%,gpu:90%
```

### 性能分析

#### Profiling 分析
```bash
# CPU Profiling
python performance_optimizer.py --profile-cpu \
  --script train.py \
  --output cpu_profile.prof

# 内存 Profiling
python performance_optimizer.py --profile-memory \
  --script train.py \
  --output memory_profile.prof
```

#### 性能热点分析
```bash
# 识别性能热点
python performance_optimizer.py --hotspot-analysis \
  --profile cpu_profile.prof
```

## 报告生成

### 性能报告

#### HTML 报告
```bash
# 生成详细性能报告
python performance_optimizer.py --generate-report \
  --format html \
  --include-charts \
  --output performance_report.html
```

#### PDF 报告
```bash
# 生成 PDF 报告
python performance_optimizer.py --generate-report \
  --format pdf \
  --template executive \
  --output performance_summary.pdf
```

### 基准报告

#### 对比报告
```bash
# 生成基准对比报告
python benchmark_tools.py --compare-results \
  --baseline baseline_results.json \
  --current current_results.json \
  --output comparison_report.html
```

### 质量报告

#### 代码质量仪表板
```bash
# 生成质量仪表板
python code_quality_checker.py --dashboard \
  --output quality_dashboard.html
```

## 优化建议

### 自动优化

#### 配置优化
```bash
# 自动优化训练配置
python performance_optimizer.py --auto-optimize \
  --config config.yaml \
  --target throughput \
  --output optimized_config.yaml
```

#### 超参数优化
```bash
# 性能导向的超参数优化
python performance_optimizer.py --hyperparameter-optimization \
  --objective speed \
  --trials 50
```

### 手动优化指南

#### 内存优化
1. **减少批处理大小**
2. **使用梯度累积**
3. **启用梯度检查点**
4. **优化数据类型**

#### 速度优化
1. **增加数据加载器工作进程**
2. **使用混合精度训练**
3. **优化模型架构**
4. **使用编译优化**

## 配置文件

### 性能配置
```yaml
# performance_config.yaml
performance:
  monitoring:
    enabled: true
    interval: 10
    metrics: ["cpu", "memory", "gpu"]
  
  optimization:
    auto_tune: true
    mixed_precision: true
    gradient_checkpointing: false
  
  benchmarking:
    warmup_iterations: 10
    measurement_iterations: 100
    repeat_count: 3
```

### 质量配置
```yaml
# quality_config.yaml
quality:
  style:
    line_length: 88
    enforce_pep8: true
  
  complexity:
    max_complexity: 10
    max_function_length: 50
  
  security:
    scan_dependencies: true
    check_hardcoded_secrets: true
```

## 集成和自动化

### CI/CD 集成

#### GitHub Actions
```yaml
name: Performance Check
on: [push, pull_request]
jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - name: Run performance tests
        run: |
          python benchmark_tools.py --ci-mode
          python code_quality_checker.py --ci-mode
```

#### 性能回归检测
```bash
# 检测性能回归
python benchmark_tools.py --regression-check \
  --baseline-branch main \
  --threshold 5%
```

### 自动化工作流

#### 定期性能检查
```bash
# 设置定期性能检查
crontab -e
# 添加：0 2 * * * /path/to/performance_optimizer.py --daily-check
```

## 故障排除

### 常见性能问题

1. **GPU 利用率低**
   ```bash
   # 诊断 GPU 利用率问题
   python performance_optimizer.py --diagnose-gpu
   ```

2. **内存泄漏**
   ```bash
   # 检测内存泄漏
   python performance_optimizer.py --memory-leak-detection
   ```

3. **数据加载瓶颈**
   ```bash
   # 分析数据加载性能
   python performance_optimizer.py --dataloader-analysis
   ```

### 调试工具

#### 性能调试
```bash
# 启用详细性能日志
python performance_optimizer.py --debug-performance
```

#### 质量调试
```bash
# 调试代码质量问题
python code_quality_checker.py --debug --verbose
```

## 最佳实践

### 性能优化
1. **建立基准**：优化前先建立性能基准
2. **逐步优化**：一次优化一个方面
3. **测量验证**：每次优化后测量效果
4. **文档记录**：记录优化过程和结果

### 质量保证
1. **自动化检查**：集成到开发流程
2. **定期审查**：定期进行代码质量审查
3. **持续改进**：根据反馈持续改进
4. **团队标准**：建立团队代码标准

## 相关文档

- [性能优化指南](../../docs/development/)
- [基准测试文档](../../docs/development/)
- [代码质量标准](../../docs/development/)
- [故障排除指南](../../docs/development/troubleshooting.md)

## 扩展开发

### 添加新的性能指标
```python
# 示例：自定义性能指标
from florence_forge.performance import BaseMetric

class CustomMetric(BaseMetric):
    def measure(self):
        # 实现测量逻辑
        return measurement_result
```

### 自定义优化策略
```python
# 示例：自定义优化器
from florence_forge.optimization import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    def optimize(self, config):
        # 实现优化逻辑
        return optimized_config
```

## 注意事项

- 性能优化可能影响模型精度，需要平衡考虑
- 基准测试结果受硬件环境影响，需要标准化环境
- 代码质量检查可能产生误报，需要人工审查
- 大型项目的质量检查可能需要较长时间
- 定期更新性能基准以反映代码变化