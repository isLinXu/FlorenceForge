# MoE / SelectiveSSM 实验模块说明

**状态**：仍未投入主训练通路，但早期草稿中的直接崩溃问题已修复。
**迁移日期**：2026-05-21（从 `florence_forge/core/backends/` 迁入）
**最小可运行修复**：2026-05-22

## 已修复问题

### 1. `selective_ssm_mixer.py` 随机常量参数

旧实现中 `_compute_selective_params(d_model, d_state)` 使用：

```python
x = torch.randn(d_model, d_state)
theta = torch.sigmoid(x @ torch.randn(d_state, d_model))
mask = theta > 0.5
selective_attention = torch.randn_like(x) * mask
return selective_attention
```

返回值是**完全随机的张量** × **随机掩码**，没有任何可训练参数、没有输入依赖。
当前实现已改为 `nn.Linear` + `SparseGate` 的可训练残差 mixer。

### 2. `moe_layer.py` einsum 维度错误

旧实现中：

```python
expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
# expert_outputs.shape = (num_experts, B, S, d_state)
gate_weights = F.softmax(gate_output, dim=0)
output = torch.einsum("bsnd,bsne->bsnd", gate_weights, expert_outputs)
```

当前实现改为：

```python
gate_weights.shape == (batch, seq, num_experts)
expert_outputs.shape == (batch, seq, num_experts, d_state)
output = torch.einsum("bse,bsed->bsd", gate_weights, expert_outputs)
```

### 3. `SparseGate` 输出契约不明确

当前 `SparseGate` 明确返回 `(batch, seq, num_experts)`，并在最后一维归一化。

## 仍需注意

主训练通路（`trainer.py`、`dataset.py`、`evaluation/`、`cli/`、`examples/`）中
**没有任何文件 import 这些 MoE 模块**。它们仍然是实验 API，不应在生产训练中依赖。

## 处置策略

- ✅ 已迁移到 `florence_forge/experimental/moe/`
- ✅ 已从 `florence_forge/core/backends/__init__.py` 移除导出
- ✅ 已在 `experimental/__init__.py` 与 `experimental/moe/__init__.py` 标注稳定性
- ✅ 已补最小单元测试覆盖 forward shape、gradient flow 和 gate 归一化
- ⏳ 若要生产化，仍需真实稀疏专家调度、负载均衡和端到端训练验证

## 修复清单（未来真正引入 MoE 时遵循）

1. 将当前 dense-all-experts 计算替换为真正 top-k 稀疏计算；
2. 增加 router load-balancing loss；
3. 支持专家容量、溢出 token 处理和多设备专家并行；
4. 提供 `examples/moe_example.py` 跑通最小训练步；
5. 全部完成后才考虑迁回 `core/backends/` 并暴露公共 API。
