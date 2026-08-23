# MoE 实验模块

> **稳定性等级：Tier-2 候选（Beta）**
> 核心算法已实现（稀疏前向、负载均衡损失、专家并行骨架），但尚未经过复杂任务端到端验证。
> 生产训练请通过 `florence_forge.training.moe` 导入（lazy re-export）。

## 设计动机

FlorenceForge 的 MoE 模块旨在探索稀疏门控混合专家（Mixture of Experts）在多任务 VLM 微调中的应用潜力。核心假设是：不同视觉任务（如 OD、Caption、OCR）可能受益于不同的专家子网络，而稀疏门控可以在推理时只激活相关专家，降低计算成本。

## 模块组成

| 文件 | 类 | 说明 |
|------|-----|------|
| `sparse_gate.py` | `SparseGate` | 可训练稀疏门控：投影 + top-k masking + softmax + threshold 裁剪 + hard routing (straight-through estimator) |
| `moe_layer.py` | `MoELayer` | 稀疏前向 MoE 层：仅计算 top-k 专家，支持 capacity factor 和 overflow tracking |
| `moe_encoder.py` | `MoEEncoder` | 编码器薄壳 |
| `moe_decoder.py` | `MoEDecoder` | 解码器薄壳 |
| `moe_model.py` | `MoEModel` | 完整 token 模型：Embedding + MoELayer + 输出投影 |
| `moe_language_model.py` | `MoELanguageModel` | 向后兼容别名 |
| `moe_trainer.py` | `MoETrainer` | ⚠️ 骨架实现，待接入主训练循环 |
| `moe_validator.py` | `MoEValidator` | 验证输出形状、有限性和路由权重归一化 |
| `moe_utils.py` | `create_moe_layer()` | 工厂函数 |
| `moe_config.py` | `MoEConfig` | Pydantic v2 配置模型 |
| `moe_adapter.py` | `MoETrainingAdapter` | **核心集成接口**：注入/卸载 MoE 层、aux loss、z-loss、路由统计 |
| `expert_parallel.py` | `ExpertParallelMoE` | 专家并行：simulation 模式 + 真实 distributed 骨架 |
| `selective_ssm_mixer.py` | `SelectiveSSMMixer` | 残差混合器（实验性子模块） |

## 使用示例

### 基础稀疏前向

```python
from florence_forge.training.moe import MoELayer
import torch

layer = MoELayer(num_experts=8, d_model=768, d_state=256, top_k=2)
x = torch.randn(2, 10, 768)
output = layer(x)  # shape: (2, 10, 256)
# 每个 token 仅激活 2 个专家（节省 ~75% expert FLOPs）
```

### 接入训练管线

```python
from florence_forge.training.moe import MoETrainingAdapter, MoEConfig

config = MoEConfig(num_experts=8, d_model=768, d_state=256, top_k=2,
                   aux_loss_weight=0.05, z_loss_weight=0.001)
adapter = MoETrainingAdapter(config)

# 将模型中的 encoder layers 替换为 MoE
adapter.inject_moe_into_model(model, target_layer_pattern=r"encoder\.layer\.([0-9]+)")

# 训练时自动叠加 aux + z-loss
total_loss = adapter.loss_hook(base_loss)

# 查看路由统计
print(adapter.summarize_routing())
```

### Expert Parallelism（单 GPU 模拟）

```python
from florence_forge.training.moe import ExpertParallelMoE

ep_moe = ExpertParallelMoE(
    num_experts=8, d_model=768, d_state=256, top_k=2,
    world_size=4, simulate=True,  # 在单 GPU 上模拟 4 路 EP
)
output = ep_moe(x)
```

## 已实现特性

1. ✅ **真正稀疏前向**：仅计算 top-k 专家，跳过零权重专家
2. ✅ **Auxiliary Load-Balancing Loss**：基于 Switch Transformer 设计
3. ✅ **Router Z-Loss**：基于 ST-MoE / PaLM，防止 logits 过大
4. ✅ **Capacity Factor**：专家容量限制 + overflow token 统计
5. ✅ **Hard Routing**：straight-through estimator，每个 token 精确路由到 1 个专家
6. ✅ **Expert Parallelism**：simulation 模式完整，真实 distributed 骨架就绪
7. ✅ **MoETrainingAdapter**：支持注入/卸载、loss hook、路由诊断
8. ✅ **Pydantic v2 配置**：字段校验 + 交叉校验

## 已知限制

1. **⚠️ CIFAR-10 / 复杂任务验证缺失**：尚未在真实视觉任务上验证 MoE 是否优于 Dense baseline
2. **⚠️ MoETrainer 为骨架**：`_train_step()` 返回硬编码 0.0，需通过 `MoETrainingAdapter` + `MultiTaskTrainer` 接入主训练循环
3. **⚠️ 真实多 GPU EP 未验证**：`ExpertParallelMoE` 的 distributed 路径依赖 `torch.distributed`，尚未在 2+ GPU 环境验证
4. **⚠️ MoECallback 未定义**：`test_moe_callback.py` 引用了 `MoECallback`，但 `callbacks.py` 中无该类定义
5. **专家类型单一**：当前专家为 `nn.Linear`，暂不支持 MLP 或更复杂的专家架构

## 生产化路线图

| 阶段 | 任务 | 状态 |
|------|------|------|
| **Stage 1** | 稀疏前向实现 | ✅ 完成 |
| **Stage 2** | CIFAR-10 端到端验证 | ⏳ 待执行 |
| **Stage 3** | 报告与集成 | ✅ 本报告完成 |
| **下一步** | 补全 MoECallback / 接入真实训练循环 | 🔜 短期 |
| **下一步** | 多 GPU EP 验证 | 🔜 中期 |
| **下一步** | 接入 FlorenceForge 主训练管线并暴露公共 API | 🔜 需 Stage 2 完成后 |

## 历史说明

- **2026-05-21**：从 `florence_forge/core/backends/` 迁移到实验性目录
- **2026-05-22**：修复递归实例化、随机非参数张量和 einsum shape 错误
- **2026-06-24**：ruff 全部通过，代码质量跃升
- **2026-07-02**：稀疏前向、aux loss、z-loss、capacity factor、hard routing、EP simulation 全部落地，成熟度从 Tier-3 提升至 Tier-2 候选

---

> **导入路径**: 推荐使用 `from florence_forge.training.moe import ...`
> `florence_forge/experimental/moe/` 中的文件为历史兼容保留，实际代码与 `training/moe/` 同步。
