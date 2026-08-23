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

## Stage 2 — 复杂任务验证（CIFAR-10）

### 目标
在 CIFAR-10 上对比 Dense / MoE-dense / MoE-sparse / MoE-sparse+aux 四种配置，验证：
- MoE 在复杂任务上是否优于 Dense
- aux loss 是否改善专家利用率
- sparse 计算是否减少训练时间

### 模型架构
- 输入：CIFAR-10 图像 (32×32×3)
- 特征提取：轻量 CNN (3 层 Conv)
- 分类头：Dense 或 MoE 层 → 10 类

### 实验设置
- 训练 Epochs: 50
- 批次大小: 128
- 学习率: 1e-3 (Adam)
- 专家数: 8, top-k: 2
- aux_loss_weight: 0.05, z_loss_weight: 0.001

## Stage 3 — 报告与集成

### 输出
1. 稀疏前向实现代码
2. CIFAR-10 实验结果（含图表）
3. 更新 MoE 成熟度评估
4. 将实验结果合并到主报告

## 语言
- 中文报告，英文技术术语保留原词
