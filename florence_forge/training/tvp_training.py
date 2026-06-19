"""Bridge TVP YAML configs to FlorenceForge MultiTaskTrainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..core.config import DataConfig, ModelConfig, OptimizationConfig, TrainingConfig
from ..core.tasks import validate_task_name

logger = logging.getLogger(__name__)

TVP_TASK_ALIASES: Dict[str, str] = {
    "od": "OD",
    "od_vp": "OD_VP",
    "counting": "COUNT_VP_COT",
    "count_vp": "COUNT_VP",
    "count_vp_cot": "COUNT_VP_COT",
    "grounding": "PHRASE_GROUNDING_VP",
    "phrase_grounding_vp": "PHRASE_GROUNDING_VP",
    "spatial": "SPATIAL_VP",
    "spatial_vp": "SPATIAL_VP",
    "maze": "MAZE_VP",
    "maze_vp": "MAZE_VP",
    "path": "PATH_VP",
    "path_vp": "PATH_VP",
    "caption": "CAPTION",
    "detailed_caption": "DETAILED_CAPTION",
    "ocr": "OCR",
    # Agentic meta-cognitive tasks
    "agentic_count": "AGENTIC_COUNT",
    "agentic_spatial": "AGENTIC_SPATIAL",
    "agentic_maze": "AGENTIC_MAZE",
    "agentic_grounding": "AGENTIC_GROUNDING",
}


def normalize_tvp_task_type(task_type: str) -> str:
    """Map TVP YAML task aliases to registered FlorenceForge task names."""
    key = str(task_type or "").strip().lower()
    if not key:
        raise ValueError("task_type must not be empty")
    resolved = TVP_TASK_ALIASES.get(key, task_type.strip().upper())
    if not validate_task_name(resolved):
        raise ValueError(f"Unknown TVP task_type after normalization: {task_type!r} -> {resolved!r}")
    return resolved


def load_tvp_yaml(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_tvp_data_configs(
    cfg: Dict[str, Any],
    *,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Build MultiTaskDataset configs from a TVP stage YAML."""
    datasets = cfg.get("datasets") or []
    if not datasets:
        raise ValueError("TVP config must define at least one entry under 'datasets'")

    data_configs: List[Dict[str, Any]] = []
    for entry in datasets:
        if split == "val" and entry.get("val_path"):
            data_path = entry["val_path"]
        else:
            data_path = entry.get("path") or entry.get("data_path")
        if not data_path:
            continue

        task_type = normalize_tvp_task_type(entry.get("task_type", "OD_VP"))
        image_root = entry.get("image_root") or entry.get("images_dir") or "."
        data_configs.append({
            "task_type": task_type,
            "data_path": str(Path(data_path).expanduser()),
            "weight": float(entry.get("weight", 1.0)),
            "image_root": str(Path(image_root).expanduser()),
        })

    if not data_configs:
        raise ValueError(f"No dataset paths found for split={split!r}")
    return data_configs


def apply_mixed_training_weights(
    data_configs: List[Dict[str, Any]],
    *,
    tvp_ratio: float = 0.3,
    tvp_task_prefixes: Tuple[str, ...] = ("OD_VP", "COUNT_VP", "PHRASE_GROUNDING_VP", "COUNT_VP_COT", "SPATIAL_VP", "MAZE_VP", "PATH_VP", "AGENTIC_COUNT", "AGENTIC_SPATIAL", "AGENTIC_MAZE", "AGENTIC_GROUNDING"),
) -> List[Dict[str, Any]]:
    """Rescale dataset weights to approximate TVP paper 70/30 general/VP mix."""
    if not data_configs:
        return data_configs

    tvp_ratio = max(0.0, min(float(tvp_ratio), 1.0))
    general_ratio = 1.0 - tvp_ratio
    if tvp_ratio == 0.0 or general_ratio == 0.0:
        return data_configs

    tvp_items = [item for item in data_configs if item["task_type"] in tvp_task_prefixes]
    general_items = [item for item in data_configs if item not in tvp_items]
    if not tvp_items or not general_items:
        return data_configs

    tvp_scale = tvp_ratio / max(len(tvp_items), 1)
    general_scale = general_ratio / max(len(general_items), 1)
    scaled: List[Dict[str, Any]] = []
    for item in data_configs:
        updated = dict(item)
        if item in tvp_items:
            updated["weight"] = float(item.get("weight", 1.0)) * tvp_scale
        else:
            updated["weight"] = float(item.get("weight", 1.0)) * general_scale
        scaled.append(updated)
    return scaled


def build_training_config_from_tvp(
    cfg: Dict[str, Any],
    *,
    checkpoint_dir: Optional[str] = None,
) -> TrainingConfig:
    """Convert a TVP stage YAML dict into a FlorenceForge TrainingConfig."""
    output_dir = checkpoint_dir or cfg.get("output_dir") or cfg.get("save_dir") or "outputs/tvp/sft"
    model_name = cfg.get("model_name_or_path") or cfg.get("model_name") or "microsoft/Florence-2-base"

    mixed = cfg.get("mixed_training") or {}
    tvp_ratio = mixed.get("tvp_ratio", 0.3)
    train_data_configs = build_tvp_data_configs(cfg, split="train")
    if mixed.get("enabled", True):
        train_data_configs = apply_mixed_training_weights(
            train_data_configs,
            tvp_ratio=float(tvp_ratio),
        )

    tasks = sorted({item["task_type"] for item in train_data_configs})
    task_weights = {item["task_type"]: float(item.get("weight", 1.0)) for item in train_data_configs}

    # Auto-enable agentic tokens when any agentic task is present
    has_agentic_tasks = any(t.startswith("AGENTIC_") for t in tasks)

    optimization = OptimizationConfig(
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
    )

    training_config = TrainingConfig(
        num_epochs=int(cfg.get("epochs", cfg.get("num_epochs", 3))),
        max_steps=cfg.get("max_steps"),
        output_dir=str(output_dir),
        device=str(cfg.get("device", "auto")),
        use_bf16=str(cfg.get("torch_dtype", "bfloat16")).lower() in {"bf16", "bfloat16"},
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        batch_size=int(cfg.get("batch_size", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        tasks=tasks,
        task_weights=task_weights,
        experiment_name=cfg.get("experiment_name", "florenceforge_tvp_sft"),
        tags=list(cfg.get("tags", ["tvp", "sft"])),
        model_settings=ModelConfig(
            model_name=model_name,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            use_lora=bool(cfg.get("use_lora", True)),
            device=str(cfg.get("device", "auto")),
            enable_agentic_tokens=has_agentic_tasks,
        ),
        data_settings=DataConfig(
            batch_size=int(cfg.get("batch_size", 4)),
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=bool(cfg.get("pin_memory", False)),
        ),
        optimization_settings=optimization,
    )
    training_config._tvp_data_configs = train_data_configs  # type: ignore[attr-defined]
    return training_config


def build_tvp_datasets(
    training_config: TrainingConfig,
    *,
    model: Any,
) -> Tuple[Any, Optional[Any]]:
    """Create train/val MultiTaskDataset instances for TVP training."""
    from ..data.dataset import MultiTaskDataset

    data_configs = getattr(training_config, "_tvp_data_configs", None)
    if not data_configs:
        raise ValueError("TrainingConfig is missing _tvp_data_configs; use build_training_config_from_tvp()")

    processor = getattr(model, "processor", None)
    backend = getattr(model, "_backend", None)
    image_base = Path(data_configs[0].get("image_root", "."))

    train_dataset = MultiTaskDataset(
        data_configs=data_configs,
        image_base_path=str(image_base),
        config=training_config.data_settings,
        processor=processor,
        backend=backend,
    )

    val_dataset = None
    val_configs = getattr(training_config, "_tvp_val_data_configs", None)
    if val_configs:
        val_dataset = MultiTaskDataset(
            data_configs=val_configs,
            image_base_path=str(Path(val_configs[0].get("image_root", image_base))),
            config=training_config.data_settings,
            processor=processor,
            backend=backend,
        )
    return train_dataset, val_dataset


def run_tvp_sft_with_multitask_trainer(
    config_path: str | Path,
    *,
    checkpoint_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run TVP stage-1 SFT using the stable MultiTaskTrainer stack."""
    from ..core.model import Florence2MultiTaskModel
    from .trainer import MultiTaskTrainer

    cfg = load_tvp_yaml(config_path)
    if overrides:
        cfg.update(overrides)

    training_config = build_training_config_from_tvp(cfg, checkpoint_dir=checkpoint_dir)
    try:
        val_configs = build_tvp_data_configs(cfg, split="val")
        training_config._tvp_val_data_configs = val_configs  # type: ignore[attr-defined]
    except ValueError:
        pass

    logger.info("Loading Florence model for TVP SFT: %s", training_config.model_settings.model_name)
    model = Florence2MultiTaskModel(training_config.model_settings)
    model.load()

    train_dataset, val_dataset = build_tvp_datasets(training_config, model=model)
    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
    )
    summary = trainer.train()
    return {
        "status": "completed",
        "checkpoint_dir": training_config.output_dir,
        "summary": summary,
    }


def _resolve_device(cfg: Dict[str, Any]) -> str:
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _checkpoint_is_peft(checkpoint_path: Path) -> bool:
    return (checkpoint_path / "adapter_config.json").exists()


def load_tvp_checkpoint(
    checkpoint_path: str | Path,
    cfg: Dict[str, Any],
    *,
    trainable: bool = True,
) -> Any:
    """Load a FlorenceForge wrapper from a TVP stage checkpoint."""
    from ..core.model import Florence2MultiTaskModel

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    is_peft = _checkpoint_is_peft(path)
    model_config = ModelConfig(
        model_name=str(path),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        use_lora=bool(cfg.get("use_lora", False)) and not is_peft,
        device=str(cfg.get("device", "auto")),
    )
    wrapper = Florence2MultiTaskModel.load_pretrained(
        str(path),
        config=model_config,
        is_peft_model=is_peft,
    )
    if trainable:
        wrapper.train()
    else:
        wrapper.eval()
    return wrapper


def save_tvp_checkpoint(model_wrapper: Any, output_dir: str | Path) -> Path:
    """Persist a TVP stage checkpoint under ``output_dir/final``."""
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model_wrapper.save_pretrained(str(final_dir))
    return final_dir


def build_tvp_dataloader(
    training_config: TrainingConfig,
    *,
    model: Any,
    batch_size: Optional[int] = None,
) -> Any:
    """Build a TaskDataLoader for TVP OPD/GRPO stages."""
    from ..data.loader import TaskDataLoader

    data_settings = training_config.data_settings.model_copy(deep=True)
    if batch_size is not None:
        data_settings.batch_size = batch_size

    train_dataset, _ = build_tvp_datasets(training_config, model=model)
    return TaskDataLoader(
        dataset=train_dataset,
        config=data_settings,
    ).get_dataloader()


def _resolve_teacher_paths(cfg: Dict[str, Any], student_path: str) -> List[str]:
    teacher_paths = list(cfg.get("teacher_models") or cfg.get("teachers") or [])
    if not teacher_paths:
        return []
    resolved: List[str] = []
    allow_missing = bool(cfg.get("allow_missing_teachers", True))
    for path in teacher_paths:
        expanded = Path(path).expanduser()
        if expanded.exists():
            resolved.append(str(expanded))
            continue
        if allow_missing:
            logger.warning(
                "Teacher checkpoint missing at %s; falling back to student checkpoint %s",
                expanded,
                student_path,
            )
            resolved.append(student_path)
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found: {expanded}")
    return resolved


def run_tvp_opd(
    config_path: str | Path,
    *,
    checkpoint_dir: Optional[str] = None,
    student_checkpoint: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run TVP stage-2 OPD with FlorenceForge models and dataloaders."""
    from .opd_trainer import OPDTrainer

    cfg = load_tvp_yaml(config_path)
    if overrides:
        cfg.update(overrides)

    output_dir = checkpoint_dir or cfg.get("output_dir") or "outputs/tvp/opd"
    student_path = student_checkpoint or cfg.get("student_model") or cfg.get("student_model_path")
    if not student_path:
        raise ValueError("OPD config must define student_model or pass student_checkpoint")

    training_config = build_training_config_from_tvp(cfg, checkpoint_dir=output_dir)
    device = _resolve_device(cfg)

    logger.info("Loading OPD student from %s", student_path)
    student_wrapper = load_tvp_checkpoint(student_path, cfg, trainable=True)
    student_module = student_wrapper.model

    teacher_paths = _resolve_teacher_paths(cfg, str(student_path))
    if not teacher_paths:
        logger.warning("No teacher_models configured; using student for both expert slots")
        teacher_paths = [str(student_path), str(student_path)]

    teachers = [
        load_tvp_checkpoint(path, cfg, trainable=False).model
        for path in teacher_paths
    ]

    dataloader = build_tvp_dataloader(
        training_config,
        model=student_wrapper,
        batch_size=int(cfg.get("batch_size", 1)),
    )

    trainer = OPDTrainer(
        student=student_module,
        teachers=teachers,
        teacher_weights=list(cfg.get("teacher_weights") or [1.0 / len(teachers)] * len(teachers)),
        temperature=float(cfg.get("temperature", 2.0)),
        ce_coeff=float(cfg.get("ce_coeff", 0.1)),
        lr=float(cfg.get("learning_rate", 5e-7)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        device=device,
    )

    epoch_results: Dict[str, Any] = {}
    num_epochs = int(cfg.get("epochs", 2))
    save_every = int(cfg.get("save_every", 1))
    for epoch in range(num_epochs):
        epoch_results[f"epoch_{epoch}"] = trainer.train_epoch(
            dataloader=dataloader,
            epoch=epoch,
            save_dir=Path(output_dir),
            save_every=save_every,
        )

    final_dir = save_tvp_checkpoint(student_wrapper, output_dir)
    return {
        "status": "completed",
        "checkpoint_dir": str(final_dir),
        "epochs": epoch_results,
    }


def prepare_grpo_prompt_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Trim SFT-encoded batches down to prompt tokens for GRPO rollouts."""
    labels = batch.get("labels")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if labels is None or input_ids is None or attention_mask is None:
        return batch

    prompt_lengths: List[int] = []
    for row_labels in labels:
        non_masked = (row_labels != -100).nonzero(as_tuple=False)
        prompt_lengths.append(int(non_masked[0].item()) if len(non_masked) else input_ids.shape[1])

    max_prompt_len = max(prompt_lengths)
    trimmed = dict(batch)
    trimmed["input_ids"] = input_ids[:, :max_prompt_len]
    trimmed["attention_mask"] = attention_mask[:, :max_prompt_len]
    trimmed.pop("labels", None)
    return trimmed


def _build_grpo_reward_models(cfg: Dict[str, Any]) -> Tuple[List[Any], Optional[List[float]]]:
    from .reward_models import (
        FormatRewardModel, QualityRewardModel, build_reward_models,
        build_agentic_reward_models,
    )

    reward_cfg = cfg.get("reward_models") or {}
    accuracy_cfg = reward_cfg.get("accuracy_rm") or {}
    task_type = str(accuracy_cfg.get("task_type", "mixed"))

    # Auto-detect agentic tasks in the dataset config
    datasets = cfg.get("datasets") or []
    has_agentic = any(
        str(entry.get("task_type", "")).lower().startswith("agentic")
        for entry in datasets
    ) or task_type.lower().startswith("agentic")

    reward_fns: List[Any] = []
    weights: List[float] = []

    if has_agentic:
        # Use agentic reward models (format + quality + self-correction + accuracy)
        agentic_fns = build_agentic_reward_models()
        agentic_weights = [0.25, 0.20, 0.25, 0.30]
        return agentic_fns, agentic_weights

    format_cfg = reward_cfg.get("format_rm") or {}
    if format_cfg.get("enabled", True):
        reward_fns.append(FormatRewardModel())
        weights.append(float(format_cfg.get("weight", 0.3)))

    quality_cfg = reward_cfg.get("quality_rm") or {}
    if quality_cfg.get("enabled", True):
        reward_fns.append(QualityRewardModel())
        weights.append(float(quality_cfg.get("weight", 0.2)))

    if accuracy_cfg.get("enabled", True):
        reward_fns.append(build_reward_models(task_type=task_type)[2])
        weights.append(float(accuracy_cfg.get("weight", 0.5)))

    if not reward_fns:
        return build_reward_models(task_type=task_type), None

    total = sum(weights)
    return reward_fns, [weight / max(total, 1e-8) for weight in weights]


class _GRPODataLoaderAdapter:
    """Wrap a dataloader so GRPO receives prompt-only batches."""

    def __init__(self, dataloader: Any):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield prepare_grpo_prompt_batch(batch)

    def __len__(self) -> int:
        return len(self.dataloader)


def run_tvp_grpo(
    config_path: str | Path,
    *,
    checkpoint_dir: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
    ref_checkpoint: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run TVP stage-3 GRPO with FlorenceForge models and reward models."""
    from .grpo_trainer import GRPOTrainer

    cfg = load_tvp_yaml(config_path)
    if overrides:
        cfg.update(overrides)

    output_dir = checkpoint_dir or cfg.get("output_dir") or "outputs/tvp/grpo"
    policy_path = model_checkpoint or cfg.get("model_name_or_path") or cfg.get("model_path")
    ref_path = ref_checkpoint or cfg.get("ref_model") or cfg.get("reference_model")
    if not policy_path:
        raise ValueError("GRPO config must define model_name_or_path")
    if not ref_path:
        raise ValueError("GRPO config must define ref_model (typically the SFT checkpoint)")

    training_config = build_training_config_from_tvp(cfg, checkpoint_dir=output_dir)
    device = _resolve_device(cfg)

    logger.info("Loading GRPO policy from %s", policy_path)
    policy_wrapper = load_tvp_checkpoint(policy_path, cfg, trainable=True)
    logger.info("Loading GRPO reference model from %s", ref_path)
    ref_wrapper = load_tvp_checkpoint(ref_path, cfg, trainable=False)

    tokenizer = getattr(policy_wrapper.processor, "tokenizer", policy_wrapper.processor)
    reward_fns, reward_weights = _build_grpo_reward_models(cfg)

    dataloader = _GRPODataLoaderAdapter(
        build_tvp_dataloader(
            training_config,
            model=policy_wrapper,
            batch_size=int(cfg.get("batch_size", 1)),
        )
    )

    trainer = GRPOTrainer(
        model=policy_wrapper.model,
        ref_model=ref_wrapper.model,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        group_size=int(cfg.get("group_size", 4)),
        kl_coeff=float(cfg.get("kl_coeff", 0.04)),
        clip_eps=float(cfg.get("clip_eps", 0.2)),
        lr=float(cfg.get("learning_rate", 1e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        device=device,
        reward_weights=reward_weights,
    )

    epoch_results: Dict[str, Any] = {}
    num_epochs = int(cfg.get("epochs", 2))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    for epoch in range(num_epochs):
        epoch_results[f"epoch_{epoch}"] = trainer.train_epoch(
            dataloader=dataloader,
            max_new_tokens=max_new_tokens,
            epoch=epoch,
        )

    final_dir = save_tvp_checkpoint(policy_wrapper, output_dir)
    return {
        "status": "completed",
        "checkpoint_dir": str(final_dir),
        "epochs": epoch_results,
    }
