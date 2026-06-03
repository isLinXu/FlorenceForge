# v1 / v2 训练栈迁移时间线

> 状态基准：2026-06-03 工作树源码核验（见 `docs/Deep_Analysis_2026-06-03.md`）。
> 本文件给出 v1 → v2 训练栈的**正式弃用与迁移计划**，回应增量核验报告 P0/P1 项。

## 背景

仓库当前并行维护两套训练栈，这是目前唯一的结构性技术债：

| 维度 | v1 (`trainer.py`) | v2 (`trainer_refactored.py` + `training_loop.py` + `checkpoint_manager.py` + `device_config.py`) |
| --- | --- | --- |
| 形态 | 单文件 god class（~1369 行） | 组合式模块化（职责分层） |
| 默认导出 | `MultiTaskTrainerV1` 指向 v1 | **`MultiTaskTrainer` 指向 v2**（v1.2.0，2026-06-03） |
| 测试 | `tests/test_trainer.py` | `tests/test_training_integration.py` |
| CheckpointManager | `checkpoint.py`（函数式工具集） | `checkpoint_manager.py`（OO 生命周期版） |

## 当前功能对齐状态（2026-06-01）

| 能力 | v1 | v2 | 说明 |
| --- | --- | --- | --- |
| FSDP / DeepSpeed | ✅ | ✅ | v2 由 `device_config.build_distributed_plugin` 提供，`trainer_refactored._create_accelerator` 接线 |
| 异步 checkpoint | ✅ | ✅ | v2 `checkpoint_manager` 使用单线程 `ThreadPoolExecutor` 后台保存 |
| `max_steps` 硬上限 | ✅ | ✅（2026-06-01 补齐） | v2 `TrainingLoop._max_steps_reached` + 内/外层循环终止 |
| `load_best_model_at_end` | ✅ | ✅（2026-06-01 补齐） | v2 `_maybe_restore_best_model` 收尾前恢复最佳权重 |
| 梯度验证 / 内存监控 | ✅ | ✅ | v2 在 `train_epoch` 中按间隔触发 |
| 激活值重计算多档策略 | ✅ | ✅（2026-06-03） | v1/v2 共用 `training/activation_checkpointing.py`（full/selective/auto/none + KV cache 禁用） |
| 双 `CheckpointManager` 合并 | — | ✅ 命名收敛 | v2 为默认 `CheckpointManager`；v1 目录式 API 更名为 `DirectoryCheckpointManager` |

> 结论：v2 的核心训练能力已与 v1 对齐；剩余差距集中在**双 CheckpointManager 合并**（v1.2.0）与**默认导出切换**。

## 迁移里程碑

### v1.1.0 —「对齐与软弃用」
- v2 补齐 `max_steps`、`load_best_model_at_end`（**已完成**，2026-06-01）。
- v2 补齐激活值重计算的剩余档位，达到与 v1 功能对等（**已完成**，2026-06-03，见 `activation_checkpointing.py`）。
- 在 v1 `trainer.py` 顶部与 `MultiTaskTrainer.__init__` 中加入**非强制** `DeprecationWarning`（可通过环境变量关闭），引导新代码迁移到 v2。
- 文档与示例统一推荐 v2 入口。

### v1.2.0 —「默认切换」（**进行中 → 核心项已完成**，2026-06-03）
- ✅ 顶层默认导出 `florence_forge.training.MultiTaskTrainer` 与 CLI `--trainer-version` 默认均为 v2。
- ✅ 遗留 v1 训练器导出为 `MultiTaskTrainerV1` / `TrainerV1`。
- ✅ 目录式检查点类更名为 `DirectoryCheckpointManager`；`checkpoint.py::CheckpointManager` 保留别名。
- ✅ `save_model_only` / `load_model_only` 已使用 `atomic_torch_save` / `safe_torch_load`。
- ⏳ 补充真实多 GPU CUDA 集成测试覆盖 v2 的 FSDP/DeepSpeed 路径。

### v2.0.0 —「移除 v1」（**已完成核心项**，2026-06-03）
- ✅ 删除 `trainer.py`（v1）；`MultiTaskTrainerV1` / `TrainerV1` 导出移除；CLI `--trainer-version v1` 报错。
- ✅ `MultiDatasetTrainer` 迁移至 v2 父类 + 独立 `train()` / `trainer_step_metrics.py`。
- ✅ v2 训练步 CPU 集成测试（`test_training_distributed_v2.py`）。
- ✅ `checkpoint.py` 与 `checkpoint_manager.py` 分工文档化；底层共用 `_checkpoint_io`（目录式 API 保留为 `DirectoryCheckpointManager`）。

## 迁移指引（面向使用者）

```python
# 默认（v2，v1.2.0+）
from florence_forge.training import MultiTaskTrainer

# 遗留 v1
from florence_forge.training import MultiTaskTrainerV1

# 细粒度 v2 组件
from florence_forge.training.training_loop import TrainingLoop
from florence_forge.training.checkpoint_manager import CheckpointManager
```

二者构造签名一致（`model, train_dataset, val_dataset, config, accelerator`），
`train()` 返回的摘要字段保持兼容，常规训练脚本可直接切换。

## `max_steps` 与 `num_epochs` 语义（v1/v2 一致）

- `max_steps`（> 0）一旦设定即为训练终止的**硬上限**，优先于 `num_epochs`：
  达到 `max_steps` 立即停止，即使 epoch 未跑满。
- `num_epochs` 仅作为 epoch 维度的上界。
- 同时设定时，`TrainingConfig` 会发出优先级告警（见 `core/config.py::_check_max_steps_epochs`）。
