# v1 / v2 训练栈迁移时间线

> 状态基准：2026-06-01 工作树源码核验（见 `docs/Deep_Analysis_2026-05-30_Verified.md` 与同日增量核验报告）。
> 本文件给出 v1 → v2 训练栈的**正式弃用与迁移计划**，回应增量核验报告 P0/P1 项。

## 背景

仓库当前并行维护两套训练栈，这是目前唯一的结构性技术债：

| 维度 | v1 (`trainer.py`) | v2 (`trainer_refactored.py` + `training_loop.py` + `checkpoint_manager.py` + `device_config.py`) |
| --- | --- | --- |
| 形态 | 单文件 god class（~1369 行） | 组合式模块化（职责分层） |
| 默认导出 | `MultiTaskTrainer` 指向 v1 | 需显式 `from florence_forge.training.trainer_refactored import MultiTaskTrainer` |
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
| 激活值重计算多档策略 | ✅ | ⚠️ 部分 | v2 `GradientCheckpointOptimizer` 提供 full/selective 自动选择，尚未覆盖 v1 的全部档位 |
| 双 `CheckpointManager` 合并 | — | ⏳ 待办 | 见下方里程碑 |

> 结论：v2 的核心训练能力已基本达到与 v1 对齐；剩余差距集中在**激活值重计算高级档位**与**双 CheckpointManager 合并**。

## 迁移里程碑

### v1.1.0 —「对齐与软弃用」
- v2 补齐 `max_steps`、`load_best_model_at_end`（**已完成**，2026-06-01）。
- v2 补齐激活值重计算的剩余档位，达到与 v1 功能对等。
- 在 v1 `trainer.py` 顶部与 `MultiTaskTrainer.__init__` 中加入**非强制** `DeprecationWarning`（可通过环境变量关闭），引导新代码迁移到 v2。
- 文档与示例统一推荐 v2 入口。

### v1.2.0 —「默认切换」
- 顶层默认导出 `florence_forge.training.MultiTaskTrainer` 切换为 v2。
- 合并双 `CheckpointManager`：保留 `checkpoint_manager.py` 的 OO 生命周期版为唯一实现，
  将 `checkpoint.py` 中仍被外部脚本依赖的 `save_model_only / load_model_only / create_checkpoint_manager`
  改为对前者的薄封装（thin shim），保持向后兼容。
- 补充真实多 GPU CUDA 集成测试覆盖 v2 的 FSDP/DeepSpeed 路径（对应增量报告 P1）。

### v2.0.0 —「移除 v1」
- 删除 `trainer.py`（v1）及 `checkpoint.py` 中已被 shim 取代的实现。
- 文档归档 v1 资料，移除并存说明。

## 迁移指引（面向使用者）

```python
# 旧（v1，默认导出）
from florence_forge.training import MultiTaskTrainer

# 新（v2，推荐）
from florence_forge.training.trainer_refactored import MultiTaskTrainer
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
