# MoE 实验模块

> ⚠️ **稳定性等级：实验性 (Experimental, Tier-3)**
> 本模块不属于主训练通路，API 可能随时变更。导入时会触发 `ExperimentalMoEWarning`。

## 设计动机

FlorenceForge 的 MoE 实验模块旨在探索稀疏门控混合专家（Mixture of Experts）在多任务 VLM 微调中的应用潜力。核心假设是：不同视觉任务（如 OD、Caption、OCR）可能受益于不同的专家子网络，而稀疏门控可以在推理时只激活相关专家，降低计算成本。

## 模块组成

| 文件 | 类 | 说明 |
|------|-----|------|
| `sparse_gate.py` | `SparseGate` | 可训练稀疏门控：投影隐藏状态到专家权重，支持 top-k 和阈值裁剪 |
| `moe_layer.py` | `MoELayer` | 最小 MoE 层：dense-all-experts 计算 + 门控加权混合 |
| `moe_encoder.py` | `MoEEncoder` | 编码器薄壳，包裹 `MoELayer` |
| `moe_decoder.py` | `MoEDecoder` | 解码器薄壳，包裹 `MoELayer` |
| `moe_model.py` | `MoEModel` | 完整 token 模型：Embedding + MoELayer + 输出投影 |
| `moe_language_model.py` | `MoELanguageModel` | `MoEModel` 的向后兼容别名 |
| `moe_trainer.py` | `MoETrainer` | MoE 训练器（⚠️ 当前为骨架实现） |
| `moe_validator.py` | `MoEValidator` | 验证 MoE 输出的形状、有限性和路由不变量 |
| `moe_utils.py` | `create_moe_layer()` | 工厂函数 |
| `selective_ssm_mixer.py` | `SelectiveSSMMixer` | 残差混合器：状态投影 + 稀疏门控 + sigmoid 混合 |

## 使用示例

```python
from florence_forge.experimental.moe.moe_layer import MoELayer
from florence_forge.experimental.moe.moe_validator import MoEValidator
import torch

# 创建 MoE 层
layer = MoELayer(num_experts=8, d_model=768, d_state=256, top_k=2)

# 验证不变量
validator = MoEValidator(layer)
x = torch.randn(2, 10, 768)
assert validator.validate(x), "MoE invariant check failed"

# 前向传播
output = layer(x)  # shape: (2, 10, 256)
```

## 已知限制

1. **Dense-all-experts 计算**：当前 `MoELayer` 对每个 token 都运行所有专家，然后通过门控加权混合。这不是真正的稀疏计算——没有节省 FLOPs。生产化需要替换为真正的稀疏专家调度（如 Megablocks 或 ScatterMoE）。

2. **MoETrainer 为骨架**：`_train_step()` 返回硬编码 `0.0`，没有实际损失计算或反向传播。需要接入 FlorenceForge 主训练管线的优化器和调度器。

3. **无负载均衡损失**：缺少辅助负载均衡损失（auxiliary load-balancing loss），可能导致路由崩塌（所有 token 被分配到同一专家）。

4. **无专家并行**：不支持跨 GPU 的专家并行（Expert Parallelism），限制了大规模扩展。

5. **无溢出处理**：当 top-k 专家容量不足时，没有 token drop/overflow 机制。

## 生产化路线图

如果要将其纳入主训练通路，需完成以下步骤：

1. 替换 dense-all-experts 为真正的稀疏内核
2. 实现辅助负载均衡损失
3. 补充端到端训练/推理示例
4. 对路由负载均衡、溢出处理和专家并行做系统测试
5. 上述全部完成后才考虑迁回 `core/backends/`

## 历史说明

本模块原先存放在 `florence_forge/core/backends/`，于 2026-05-21 迁移到实验性目录。早期版本存在递归实例化、随机非参数张量和 einsum shape 错误，已在迁移时一并修复。详见 `WARNING.md`。
