# v1 / v2 训练栈迁移时间线

> 状态基准：2026-06-01 工作树源码核验（见 `docs/Deep_Analysis_2026-05-30_Verified.md` 与同日增量核验报告）。
> 本文件给出 v1 → v2 训练栈的**正式弃用与迁移计划**，回应增量核验报告 P0/P1 项。

## 背景

仓库当前并行维护两套训练栈，这是目前唯一的结构性技术债：

| 维度 | v1 (`trainer.py`) | v2 (`trainer_refactored.py` + `training_loop.py` + `checkpoint_manager.py` + `device_config.py`) |
| --- | --- | --- |
| 形态 | 单文件 god class（~1369 行） | 组合式模块化（职责分层） |
| 默认导出 | `MultiTaskTrainer` 指向 **v2**（v1.2.0） | `MultiTaskTrainerV1` 指向 v1 |
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
| 激活值重计算多档策略 | ✅ | ✅ | v2 `ActivationRecomputePolicy`（off/low/medium/high）与 v1 策略映射 |
| 双 `CheckpointManager` 合并 | — | ✅（v1.2.0） | v2 `CheckpointManager` 为默认导出；`checkpoint.py` 保留 `LegacyCheckpointManager` shim |

> 结论：v2 的核心训练能力已基本达到与 v1 对齐；剩余差距集中在**激活值重计算高级档位**与**双 CheckpointManager 合并**。

## 迁移里程碑

### v1.1.0 —「对齐与软弃用」
- v2 补齐 `max_steps`、`load_best_model_at_end`（**已完成**，2026-06-01）。
- v2 补齐激活值重计算的剩余档位，达到与 v1 功能对等。
- 在 v1 `trainer.py` 顶部与 `MultiTaskTrainer.__init__` 中加入**非强制** `DeprecationWarning`（可通过环境变量关闭），引导新代码迁移到 v2。
- 文档与示例统一推荐 v2 入口。

### v1.2.0 —「默认切换」（**已完成**，2026-06-17）
- 顶层默认导出 `florence_forge.training.MultiTaskTrainer` 切换为 v2。
- 新增 `MultiTaskTrainerV1` / `LegacyCheckpointManager` 别名保留 v1 兼容。
- CLI `--trainer-version` 默认改为 v2。
- `checkpoint.py` 类重命名为 `LegacyCheckpointManager`（`CheckpointManager` 为模块内别名）。
- 补充 v2 模块单元测试（`tests/test_v2_training_modules.py`）。
- ⏳ 真实多 GPU CUDA 集成测试仍待 CUDA 环境补齐。

# v2.0.0 —「移除 v1」（**已完成**，2026-06-17）
- 删除 v1 `trainer.py`（god class）与 `trainer_io.py`。
- 模块化训练栈合并为唯一 `trainer.py`。
- 删除 `checkpoint.py`；`save_model_only` / `load_model_only` 迁入 `checkpoint_manager.py`。
- CLI 移除 `--trainer-version`；v1 请求显式报错。
- `MultiDatasetTrainer` 迁移至新训练栈基类。

## 迁移指引（面向使用者）

```python
# 默认（v2，推荐）
from florence_forge.training import MultiTaskTrainer

# 遗留 v1（已弃用）
from florence_forge.training import MultiTaskTrainerV1

# 显式 v2 别名
from florence_forge.training import MultiTaskTrainerV2
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
