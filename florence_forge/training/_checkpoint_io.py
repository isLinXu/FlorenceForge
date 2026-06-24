"""检查点底层序列化原语（v1/v2 共享）。

本模块抽出 ``checkpoint.py``（v1 函数式）与 ``checkpoint_manager.py``（v2 OO）
两套 CheckpointManager 中**重复且易出错**的底层逻辑：

- ``atomic_torch_save``：原子写入，避免训练崩溃时残留半截损坏的检查点文件。
- ``load_checkpoint_file``：统一的 fail-closed 安全加载（``weights_only=True``），
  在运行时不支持时给出一致、清晰的报错而非回退到 unsafe pickle。
- ``prune_checkpoints``：通用的「保留最近 N 个、保护受保护项」保留策略，
  对文件或目录条目都适用。

两套 CheckpointManager 的**对外 API 与磁盘布局保持不变**，仅共用这些原语；
完整合并计划见 ``docs/v1_v2_Migration_Timeline.md``（v1.2.0 里程碑）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

import torch

from ..utils.torch_serialization import safe_torch_load

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
T = TypeVar("T")


def atomic_torch_save(payload: Any, path: PathLike) -> Path:
    """原子地保存 torch 检查点。

    先写入同目录下的临时文件，再 ``os.replace`` 原子改名到目标路径，
    确保读到的检查点要么是上一份完整文件、要么是这一份完整文件，
    不会出现写入中途崩溃导致的截断/损坏。

    Args:
        payload: 任意可被 ``torch.save`` 序列化的对象。
        path: 目标检查点路径。

    Returns:
        最终写入的目标路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        # 失败时清理临时文件，避免污染检查点目录
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def load_checkpoint_file(
    path: PathLike,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    context: str = "Training checkpoint",
) -> Any:
    """以 fail-closed 方式安全加载检查点。

    统一 v1/v2 的加载行为：始终走 ``weights_only=True``；当运行时不支持或
    检查点需要 unsafe pickle 时，抛出清晰一致的 ``RuntimeError`` 而非静默回退。
    """
    try:
        return safe_torch_load(path, map_location=map_location, context=context)
    except Exception as exc:  # noqa: BLE001 - 需统一转译多种底层异常
        message = str(exc).lower()
        if "weights_only" in message or "weightsunpickler" in message or "unsupported" in message:
            raise RuntimeError(
                f"检查点 {path} 需要 unsafe pickle 加载，已拒绝。"
                "请重新导出为 weights_only 兼容格式，或仅在完全信任来源时使用独立迁移脚本。"
            ) from exc
        raise


def prune_checkpoints(
    entries: Sequence[T],
    keep: int,
    *,
    sort_key: Callable[[T], Any],
    is_protected: Callable[[T], bool],
    remove: Callable[[T], None],
    reverse: bool = True,
) -> List[T]:
    """按保留策略清理旧检查点条目（文件或目录通用）。

    保留 ``sort_key`` 排序后的前 ``keep`` 个条目；其余条目中，未被
    ``is_protected`` 标记保护的将通过 ``remove`` 删除。

    Args:
        entries: 候选检查点条目集合。
        keep: 保留数量；<= 0 表示不清理。
        sort_key: 排序键（如 mtime、step、timestamp）。
        is_protected: 判断条目是否受保护（如最佳模型）不应删除。
        remove: 实际执行删除的回调。
        reverse: True 表示 sort_key 越大越「新」（默认保留最大的 keep 个）。

    Returns:
        实际被删除的条目列表。
    """
    if keep <= 0:
        return []

    ordered = sorted(entries, key=sort_key, reverse=reverse)
    removed: List[T] = []
    for entry in ordered[keep:]:
        if is_protected(entry):
            continue
        try:
            remove(entry)
            removed.append(entry)
        except Exception as exc:  # noqa: BLE001 - 删除失败不应中断训练
            logger.warning("清理旧检查点失败 %r: %s", entry, exc)
    return removed
