# FlorenceForge 分布式训练与内存优化指南

> 版本: v2.0  
> 更新日期: 2026-05-01  
> 适用版本: FlorenceForge >= 0.5.0

---

## 一、分布式训练

### 1.1 支持的分布式策略

FlorenceForge 通过 `accelerate` 库集成，支持三种分布式训练策略：

| 策略 | 说明 | 适用场景 | 显存效率 |
|------|------|---------|---------|
| **DDP** | 分布式数据并行 | 单节点多卡 / 多节点 | ★★☆ |
| **FSDP** | 全分片数据并行 | 大模型训练（>7B） | ★★★ |
| **DeepSpeed ZeRO** | 零冗余优化器 | 超大模型 / 显存极度紧张 | ★★★★ |

### 1.2 快速启动

#### 单节点多卡 DDP（最常用）

```bash
# 4卡训练
torchrun --nproc_per_node=4 \
  -m florence_forge.cli train \
  --config configs/distributed_training.yaml
```

#### 多节点训练

```bash
# Node 0 (主节点)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr="192.168.1.1" --master_port=29500 \
  -m florence_forge.cli train \
  --config configs/distributed_training.yaml

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr="192.168.1.1" --master_port=29500 \
  -m florence_forge.cli train \
  --config configs/distributed_training.yaml
```

### 1.3 配置详解

#### 基础分布式配置

```yaml
distributed_config:
  enabled: true
  strategy: "ddp"           # ddp / fsdp / deepspeed
  backend: "nccl"           # nccl (GPU) / gloo (CPU)
```

#### FSDP 配置

```yaml
distributed_config:
  enabled: true
  strategy: "fsdp"
  backend: "nccl"

  # FSDP 分片策略
  fsdp_sharding_strategy: "FULL_SHARD"     # FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD
  fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
  fsdp_cpu_offload: false                   # 显存不足时设为 true
  fsdp_activation_checkpointing: true       # FSDP 内部激活值检查点
```

#### DeepSpeed ZeRO 配置

```yaml
distributed_config:
  enabled: true
  strategy: "deepspeed"

  # ZeRO 阶段
  deepspeed_stage: 2                        # 0(禁用) / 1 / 2 / 3
  deepspeed_offload_optimizer: true         # 优化器状态卸载到 CPU
  deepspeed_offload_param: false            # 参数卸载（仅 ZeRO-3）
```

### 1.4 数据加载器分布式适配

FlorenceForge 的 `TaskDataLoader` 会自动检测分布式环境并启用 `DistributedTaskSampler`：

```python
from florence_forge.data.loader import TaskDataLoader

loader = TaskDataLoader(
    dataset=dataset,
    config=data_config,
    sampling_strategy="balanced"
)

# 自动检测分布式环境（三级检测）:
# 1. 配置显式启用
# 2. torch.distributed.is_initialized()
# 3. 环境变量 RANK / WORLD_SIZE
```

**关键特性**:
- 每个 epoch 自动调用 `sampler.set_epoch(epoch)` 确保不同 epoch 数据顺序不同
- 所有 rank 共享确定性随机种子，保证打乱顺序一致
- 支持 `balanced` / `round_robin` / `random` 三种采样策略的分布式版本

---

## 二、内存优化（激活值重计算）

### 2.1 四级策略

| 策略 | 显存节省 | 速度损失 | 适用场景 |
|------|---------|---------|---------|
| **none** | - | - | 小模型 (<1B) 或显存充足 |
| **full** | ~30-40% | ~20-30% | 中等模型 (1B-7B) |
| **selective** | ~15-25% | ~10-15% | 大模型 (>=7B)，平衡显存与速度 |
| **auto** | 自动选择 | 自动选择 | 不确定时的最佳选择 |

### 2.2 配置方法

#### 方法1: YAML 配置

```yaml
model_config:
  # 向后兼容开关（已映射到 activation_checkpointing_strategy='full'）
  gradient_checkpointing: true

  # 推荐：使用 auto 策略自动选择
  activation_checkpointing_strategy: "auto"

  # 或显式指定策略
  # activation_checkpointing_strategy: "selective"
  # checkpoint_every_n_layers: 2          # 每隔 2 层启用
  # checkpoint_target_layers: ["encoder.layers.0", "encoder.layers.3"]  # 指定层
```

#### 方法2: 代码配置

```python
from florence_forge.core.config import ModelConfig

config = ModelConfig(
    gradient_checkpointing=True,                    # 向后兼容
    activation_checkpointing_strategy="selective",  # 选择性重计算
    checkpoint_every_n_layers=2,                    # 每隔 2 层
)
```

### 2.3 Auto 策略决策逻辑

```
模型参数量 < 1B      → none
模型参数量 1B-7B     → full
模型参数量 >= 7B     → selective (自动 checkpoint_every_n_layers=2)
可用显存 < 10GB      → selective
```

### 2.4 使用建议

1. **训练 770M-2B 模型**: 使用 `auto` 或 `full`
2. **训练 4B-8B 模型**: 使用 `auto` 或 `selective` + `checkpoint_every_n_layers=2`
3. **训练 10B+ 模型**: 使用 `selective` + 结合 FSDP/DeepSpeed
4. **推理时**: 不需要启用（推理不计算梯度）

---

## 三、组合使用：分布式 + 内存优化

### 3.1 大模型训练最佳实践

```yaml
# configs/large_model_training.yaml

model_config:
  model_name: "microsoft/Florence-2-large"
  activation_checkpointing_strategy: "selective"
  checkpoint_every_n_layers: 2

data_config:
  batch_size: 2           # 每卡 batch size
  prefetch_factor: 4
  persistent_workers: true

distributed_config:
  enabled: true
  strategy: "fsdp"
  fsdp_sharding_strategy: "FULL_SHARD"
  fsdp_cpu_offload: true   # 大模型建议开启 CPU 卸载
```

启动命令:
```bash
torchrun --nproc_per_node=8 \
  -m florence_forge.cli train \
  --config configs/large_model_training.yaml
```

### 3.2 显存预算参考

| 模型大小 | 单卡显存需求 | 推荐配置 |
|---------|-------------|---------|
| 770M (Florence-2-base) | ~8GB | DDP + auto checkpoint |
| 2B (Florence-2-large) | ~16GB | DDP + selective checkpoint |
| 4B (YouTu-VL) | ~24GB | FSDP + selective + cpu_offload |
| 7B+ | ~40GB+ | FSDP/DeepSpeed + selective + offload |

---

## 四、故障排除

### 4.1 分布式训练问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: Address already in use` | master_port 被占用 | 更换 `--master_port` |
| `NCCL error` | NCCL 版本不兼容 | 检查 PyTorch 与 CUDA 版本匹配 |
| 各卡显存不均 | 数据分配不均 | 检查 `drop_last` 设置 |
| 训练卡住 | 分布式初始化失败 | 检查 `MASTER_ADDR` / `MASTER_PORT` |

### 4.2 显存问题

| 问题 | 原因 | 解决 |
|------|------|------|
| OOM 即使启用 checkpoint | batch_size 过大 | 减小 batch_size 或增加梯度累积 |
| checkpoint 后速度下降明显 | 频繁的重计算 | 改用 selective 替代 full |
| 验证时 OOM | 验证未启用 inference_mode | 检查验证代码 |

---

## 五、API 参考

### 5.1 DistributedConfig 字段

```python
class DistributedConfig:
    enabled: bool                    # 是否启用
    strategy: str                    # ddp / fsdp / deepspeed / none
    backend: str                     # nccl / gloo / mpi

    # FSDP
    fsdp_sharding_strategy: str      # FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD
    fsdp_auto_wrap_policy: str       # TRANSFORMER_BASED_WRAP / SIZE_BASED_WRAP
    fsdp_cpu_offload: bool           # CPU 卸载
    fsdp_activation_checkpointing: bool

    # DeepSpeed
    deepspeed_stage: int             # 0-3
    deepspeed_offload_optimizer: bool
    deepspeed_offload_param: bool
```

### 5.2 ModelConfig 内存优化字段

```python
class ModelConfig:
    gradient_checkpointing: bool                    # 向后兼容
    activation_checkpointing_strategy: str          # none / full / selective / auto
    checkpoint_target_layers: Optional[List[str]]   # 目标层名列表
    checkpoint_every_n_layers: Optional[int]        # 每隔 N 层
```

---

## 六、参考

- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Gradient Checkpointing Paper](https://arxiv.org/abs/1604.06174)
