"""CLI 命令处理函数（回导门面）。

所有重型子命令实现已拆分到子模块：
  - ``commands_infer.py``   — 推理（infer）
  - ``commands_serve.py``   — 服务（serve）
  - ``commands_eval.py``    — 评估（eval）
  - ``commands_convert.py`` — 数据转换（convert）
  - ``commands_train.py``   — 训练（train）+ TVP 训练 + 辅助工具

``main.py`` 仍从本模块回导这些 handler，因此
``from florence_forge.cli.main import run_inference_task`` 等历史导入路径完全兼容。
"""

from __future__ import annotations

from .commands_infer import (
    _apply_structured_vp_decode,
    run_inference_task,
)
from .commands_serve import run_serve_task
from .commands_eval import (
    _build_eval_dataset_from_jsonl,
    _run_tvp_eval_task,
    run_eval_task,
)
from .commands_convert import (
    _resolve_vp_box_format,
    _resolve_vp_marker_style,
    _run_vp_coco_conversion,
    _run_vp_yolo_conversion,
    run_data_conversion,
)
from .commands_train import (
    _apply_config_overrides,
    _coerce_override_value,
    _prepare_datasets,
    _resolve_image_base_path,
    _select_trainer_class,
    _set_nested_attr,
    run_training_task,
    run_tvp_training_task,
)
from .commands_agentic import (
    InferenceEngineAdapter,
    run_agentic_task,
)

__all__ = [
    "_apply_config_overrides",
    "_apply_structured_vp_decode",
    "_build_eval_dataset_from_jsonl",
    "_coerce_override_value",
    "_prepare_datasets",
    "_resolve_image_base_path",
    "_resolve_vp_box_format",
    "_resolve_vp_marker_style",
    "_run_tvp_eval_task",
    "_run_vp_coco_conversion",
    "_run_vp_yolo_conversion",
    "_select_trainer_class",
    "_set_nested_attr",
    "InferenceEngineAdapter",
    "run_agentic_task",
    "run_data_conversion",
    "run_eval_task",
    "run_inference_task",
    "run_serve_task",
    "run_training_task",
    "run_tvp_training_task",
]
