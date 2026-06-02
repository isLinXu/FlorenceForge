## 背景

`tests/test_advanced_metrics.py` 当前承担了高级评估指标模块的主要回归保护，但测试风格偏向“轻集成试跑”，存在以下问题：

- 部分用例依赖可选的重型三方库，导致在不同开发机或 CI 环境中的可执行性不稳定。
- 部分断言与实现公开契约不一致，例如返回类型、参数顺序和键名假设不够明确。
- 多个测试依赖 `torch.rand()`、真实时间流逝、真实进程内存波动等非确定性因素，容易出现偶发失败。
- 文件中存在未使用导入、未使用局部变量和重复样板，增加了噪音，也降低了测试意图的可读性。

本次工作只优化测试稳定性与测试契约表达，不修改 `florence_forge.evaluation.advanced_metrics` 的生产实现逻辑。

## 目标

- 将 `tests/test_advanced_metrics.py` 收敛为稳定、可重复、可在普通开发环境运行的测试集合。
- 优先验证 FlorenceForge 自身封装层的公开行为，而不是第三方依赖内部实现。
- 保留对关键公开入口的基本覆盖，包括：
  - `SemanticMetricsCalculator`
  - `MultiModalMetricsCalculator`
  - `RobustnessMetricsCalculator`
  - `EfficiencyMetricsCalculator`
  - `BenchmarkEvaluator` 对高级指标计算器的挂载行为
- 降低这组测试对机器性能、可选依赖和随机性的敏感度。

## 非目标

- 不重构高级指标生产代码。
- 不新增真实 BERTScore、CLIP、对抗攻击库的端到端集成测试。
- 不修改 benchmark 懒加载设计与现有生产模块导出结构。
- 不扩大到其他测试文件，除非为了保持当前文件可运行必须做极小范围联动。

## 方案对比

### 方案 A：契约测试优先（推荐）

做法：

- 使用 `unittest.mock.patch` 替代底层重型方法返回值。
- 对时间、内存和随机输出使用可控替身。
- 断言公开接口返回结构、关键字段、调用路径和边界行为。

优点：

- 稳定、快速、适合 CI。
- 能精准保护我们自己的封装层契约。
- 与本次“只优化测试稳定性”的目标最一致。

缺点：

- 不覆盖第三方依赖的真实行为。

### 方案 B：保留半集成测试，仅做轻量收敛

做法：

- 保留真实调用路径。
- 仅缩小输入规模、固定随机种子、减少耗时操作。

优点：

- 改动少。

缺点：

- 仍依赖环境、性能和可选库，稳定性提升有限。

### 方案 C：双层测试

做法：

- 当前文件改为稳定契约测试。
- 另起可选集成测试文件，只有依赖齐全时才运行。

优点：

- 覆盖面更完整。

缺点：

- 超出本轮最小改造范围。

结论：

本次采用方案 A。若后续确实需要验证真实三方依赖集成，再单独补方案 C。

## 设计

### 1. 语义指标测试

目标：

- 验证 `compute_bert_score()` 会透传到底层语义实现。
- 验证 `compute_clip_score()` 的调用和返回契约与当前实现保持一致。

调整：

- patch `calculator.semantic_metrics.calculate_bert_score`，返回稳定的伪结果列表。
- patch `calculator.semantic_metrics.calculate_clip_score`，返回稳定的分数列表。
- 断言：
  - 被调用次数与入参正确。
  - 返回值类型与内容符合当前实现。
  - 不再依赖是否安装真实 `bert_score` 或 CLIP 相关库。

特别说明：

当前 `compute_clip_score()` 的实现签名是 `(predictions, images)`，而现有测试以 `(images, predictions)` 调用，并将结果当成 `dict` 断言。此次会将测试修正为与实现一致的契约。

### 2. 多模态指标测试

目标：

- 验证 `compute_image_text_matching()` 的公开返回结构。

调整：

- patch `calculator.multimodal_metrics.calculate_image_text_matching` 返回固定分数列表。
- 断言：
  - 返回对象是 `dict`。
  - 包含 `matching_score` 与 `individual_scores`。
  - 均值计算符合预期。

这样测试保护的是 FlorenceForge 在公开接口层做的包装逻辑，而非底层特征提取模型。

### 3. 鲁棒性指标测试

目标：

- 验证对抗鲁棒性和噪声鲁棒性接口的返回键和值域。

调整：

- 用确定性 `mock_model()` 代替 `torch.rand()` 输出，例如构造稳定 logits。
- 固定输入形状与标签。
- 对 `evaluate_adversarial_robustness()` 和 `evaluate_noise_robustness()`：
  - 只断言公开承诺的关键字段存在。
  - 对数值做宽松但确定的范围约束。

理由：

这些方法本身已经是“简化评估逻辑”，继续依赖随机张量只会引入波动，不增加有效保护。

### 4. 效率指标测试

目标：

- 去除对真实时间流逝和真实内存波动的依赖。

调整：

- `test_inference_speed`：
  - patch `time.time` 返回可控时间序列。
  - 模型函数只返回固定张量，不再 `sleep`。
  - 断言平均时间、吞吐量和标准差字段存在且为正值。
- `test_memory_usage`：
  - patch `psutil.Process().memory_info().rss` 返回固定的起始值和峰值。
  - patch `gc.collect` 避免副作用。
  - 模型调用使用轻量对象，不再分配超大临时张量。
  - `parameters()` 返回稳定的假参数张量列表。

这样可以把测试从“依赖机器状态”改为“验证内存测量逻辑的输出结构和基本计算关系”。

### 5. 集成入口测试

目标：

- 保持 `florence_forge.evaluation` 导出入口和 `BenchmarkEvaluator` 的高级指标挂载测试。

调整：

- 保留导入与实例化测试。
- 保留 `BenchmarkEvaluator` 初始化后四个计算器属性存在的断言。
- 不在此文件重复覆盖懒加载细节，因为该行为已由 `tests/test_benchmark_lazy_metrics.py` 和 `tests/test_config.py` 单独保护。

## 变更范围

主要修改文件：

- `tests/test_advanced_metrics.py`

允许的极小范围辅助改动：

- 无，除非当前文件修改后发现必须同步修复导入路径或测试工具引用。

## 验证策略

完成后执行以下验证：

- 运行 `tests/test_advanced_metrics.py`。
- 检查该文件诊断信息，清理新增的 linter 问题。
- 确认测试不再依赖真实可选重型库安装。

不要求：

- 跑完整测试集。
- 对高级指标真实模型效果做数值回归。

## 风险与缓解

风险 1：

过度 mock 导致测试过于贴近实现细节。

缓解：

- 只 mock 外部不稳定依赖与时间/资源来源。
- 重点断言公开返回契约，而非内部临时变量。

风险 2：

修正测试契约后暴露出当前实现与原测试假设不一致。

缓解：

- 以当前生产代码的公开方法签名和返回值为准。
- 若发现实现本身存在明显对外契约问题，单独记录，不在本轮顺带修改。

风险 3：

现有测试覆盖面下降。

缓解：

- 保留入口级初始化测试。
- 将“真实依赖集成覆盖”明确标记为后续可选工作，而不是混入本轮。

## 验收标准

- `tests/test_advanced_metrics.py` 在普通环境中可稳定运行。
- 不再依赖真实 `sleep`、超大内存分配或随机输出波动。
- 语义、多模态、鲁棒性、效率四类测试都明确表达各自保护的公开契约。
- 文件内未使用导入和明显测试噪音被清理。
