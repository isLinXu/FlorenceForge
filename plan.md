# FlorenceForge 后续实现计划

## 目标
继续推进 MoE 实验模块从 Tier-3 实验 → Tier-2 候选，核心实现：
1. **稀疏前向传播**（仅激活 top-k 专家，真正节省 FLOPs）
2. **在复杂任务上验证**（CIFAR-10 图像分类，体现 MoE 容量优势）
3. **集成验证**（将稀疏 MoE 与 aux loss 串联，端到端验证）

## Stage 1 — 稀疏前向实现

### 目标
重构 `MoELayer.forward()`，使其仅计算 top-k 专家，而非全部专家。

### 关键变更点
- `SparseGate.forward()` 已返回 top-k 权重（mask 其余为 0）
- `MoELayer.forward()` 当前仍循环所有专家：`[expert(x) for expert in self.experts]`
- 需要改为：仅计算被 gate 选中的专家

### 实现策略
1. 保持 `SparseGate` 输出不变（全维度 softmax，但仅 top-k 非零）
2. 在 `MoELayer` 中识别非零权重的专家索引
3. 仅对这些索引调用 `expert(x)`
4. 将稀疏计算结果聚合回输出张量

### 验证指标
- 前向传播时间对比（dense vs sparse）
- 输出数值一致性（sparse 结果与 dense 结果在 top-k 等价假设下一致）

## Stage 2 — 复杂任务验证（CIFAR-10）✅ 已完成（2026-08-24）

### 结果摘要

- 四配置对比完成（subset=10240, 10 epochs, seed=42, MPS）：
  Dense 66.59% / MoE-dense 68.44% / MoE-sparse 37.89% / MoE-sparse+aux 45.50%
- MoE-dense 以 +9% 参数换 +1.85pp 精度，容量优势成立
- aux loss（梯度修复后）带来 +7.61pp 精度、Gini 0.625→0.500、活跃专家 3→4
- 发现：逐专家 Python 循环在 MPS 上慢 13.9×，bmm 化列为 P1 优化
- 详见 `florence_forge/training/moe/MOE_PROGRESS_REPORT.md` §八

## Stage 3 — 报告与集成

### 输出
1. 稀疏前向实现代码
2. CIFAR-10 实验结果（含图表）
3. 更新 MoE 成熟度评估
4. 将实验结果合并到主报告

## 语言
- 中文报告，英文技术术语保留原词
