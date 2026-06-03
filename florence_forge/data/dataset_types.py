"""多任务数据集基础类型定义。"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TaskSample:
    """任务样本数据结构。"""

    def __init__(
        self,
        task_type: str,
        image_path: str,
        prefix: str,
        suffix: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_type = task_type
        self.image_path = image_path
        self.prefix = prefix
        self.suffix = suffix
        self.weight = weight
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "image_path": self.image_path,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSample":
        return cls(
            task_type=data["task_type"],
            image_path=data["image_path"],
            prefix=data["prefix"],
            suffix=data["suffix"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )
