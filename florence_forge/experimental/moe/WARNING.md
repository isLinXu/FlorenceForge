# MoE / SelectiveSSM 实验模块说明

**状态**: Tier-2 候选 — 核心算法已实现（稀疏前向、负载均衡损失、专家并行 simulation），但缺少复杂任务端到端验证。
**迁移日期**: 2026-05-21（从 `florence_forge/core/backends/` 迁入）
**最新更新**: 2026-07-02（稀疏前向 + 真实损失函数落地）

## 已修复问题（历史）

### 1. `selective_ssm_mixer.py` 随机常量参数（2026-05-22 修复）

旧实现使用完全随机的张量，无任何可训练参数。当前已改为 `nn.Linear` + `SparseGate` 的可训练残差 mixer。

### 2. `moe_layer.py` einsum 维度错误（2026-05-22 修复）

旧实现中 einsum 字符串错误导致 shape mismatch。当前已修正为 `"bse,bsed->bsd"`。

### 3. `SparseGate` 输出契约不明确（2026-05-22 修复）

当前 `SparseGate` 明确返回 `(batch, seq, num_experts)`，并在最后一维归一化。

## 当前已落地特性（2026-07-02）

| 特性 | 文件 | 状态 |
|------|------|------|
| 真正稀疏前向（仅 top-k 专家计算） | `moe_layer.py` | ✅ 验证通过（sparse==dense 数值一致） |
| Auxiliary Load-Balancing Loss | `moe_adapter.py` | ✅ 真实实现（非桩） |
| Router Z-Loss | `moe_adapter.py` | ✅ 真实实现（非桩） |
| Capacity Factor + Overflow Tracking | `moe_layer.py` | ✅ 实现并测试 |
| Hard Routing (straight-through) | `sparse_gate.py` | ✅ 验证通过 |
| Expert Parallelism (simulation) | `expert_parallel.py` | ✅ 单 GPU 模拟完整 |
| MoETrainingAdapter（注入/卸载/loss hook） | `moe_adapter.py` | ✅ 实现 |
| Pydantic v2 配置 | `moe_config.py` | ✅ 实现 |

## 仍需注意

主训练通路（`trainer.py`、`training_loop.py`）中 **尚未默认启用 MoE**，需通过 `MoETrainingAdapter` 显式注入。MoE 模块目前为可选扩展，不应在不了解其限制的情况下用于生产训练。

## 当前阻塞项

1. **🔴 MoECallback 未定义**：`callbacks.py` 中缺少 `MoECallback` 类，但 `test_moe_callback.py` 引用它。需要补全实现或更新测试。
2. **🔴 CIFAR-10 验证未执行**：plan.md Stage 2 尚未执行，无法证明 MoE 在真实任务上的优势。
3. **🟡 真实多 GPU EP 未验证**：`ExpertParallelMoE` 的 distributed 路径仅为骨架，未在 NCCL 环境测试。
4. **🟡 MoETrainer 骨架**：`_train_step()` 返回硬编码 0.0，不具备独立训练能力。

## 处置策略

- ✅ 已迁移到 `florence_forge/training/moe/`（生产目录）
- ✅ 已从 `florence_forge/core/backends/__init__.py` 移除旧导出
- ✅ 已在 `training/moe/__init__.py` 实现 lazy re-export
- ✅ 已补单元测试覆盖（17+ tests，4 个测试文件）
- ✅ 稀疏前向和负载均衡损失已非桩实现
- ⏳ 需 CIFAR-10 端到端验证后考虑默认启用
- ⏳ 需 MoECallback 补全后才能与 Callback 系统完全集成

## 修复清单（进入 Tier-1 生产级前必须完成）

1. [x] 将 dense-all-experts 替换为真正 top-k 稀疏计算
2. [x] 实现 router load-balancing loss（aux + z-loss）
3. [x] 支持专家容量和溢出 token 处理
4. [x] 专家并行 simulation 模式
5. [ ] CIFAR-10 端到端对比实验
6. [ ] 补全 MoECallback 定义
7. [ ] 真实多 GPU EP 验证
8. [ ] 提供 `examples/moe_training_example.py` 跑通最小训练步
9. [ ] 接入 `MultiTaskTrainer` 作为一等公民（非 adapter 注入模式）
