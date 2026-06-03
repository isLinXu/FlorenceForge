"""
FlorenceForge 配置模块 (Pydantic v2 重构版)

所有配置类均继承自 pydantic.BaseModel，获得：
  1. 字段类型自动校验
  2. 字段值约束校验 (Field(ge=0, le=1, ...))
  3. 交叉字段校验 (model_validator)
  4. 自动序列化/反序列化 (model_dump / model_validate / model_dump_json)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WarnOnUnknownFieldsModel(BaseModel):
    """Pydantic 基类：保留兼容性，但对未知配置字段发出警告。"""

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        allowed_keys = set(cls.model_fields.keys())
        for field in cls.model_fields.values():
            if field.alias:
                allowed_keys.add(field.alias)

        unknown_keys = sorted(set(data.keys()) - allowed_keys)
        if unknown_keys:
            warnings.warn(
                f"{cls.__name__} 收到未知配置字段，将按兼容模式处理: "
                f"{', '.join(unknown_keys)}",
                stacklevel=2,
            )
        return data

# ---------------------------------------------------------------------------
# LoRA 配置
# ---------------------------------------------------------------------------


class LoRAConfig(WarnOnUnknownFieldsModel):
    """LoRA 配置 — 字段级约束直接写在 Field 中"""

    r: int = Field(default=32, ge=1, description="LoRA rank，必须大于 0")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        description="注入 LoRA 的目标模块名称列表",
    )
    lora_dropout: float = Field(
        default=0.05, ge=0.0, lt=1.0,
        description="LoRA dropout 概率，范围 [0, 1)",
    )
    bias: str = Field(default="none", description="偏置训练策略")
    task_type: str = Field(default="CAUSAL_LM", description="任务类型")

    # ----------------------------------
    # 兼容性：旧代码调用 to_dict()
    # ----------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    # ----------------------------------
    # Pydantic v2 兼容：从字典创建
    # ----------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoRAConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------


class ModelConfig(WarnOnUnknownFieldsModel):
    """模型配置"""

    model_name: str = Field(
        default="microsoft/Florence-2-large",
        description="HuggingFace 模型标识符",
    )
    revision: Optional[str] = Field(
        default=None,
        description="HuggingFace 模型/处理器的版本（分支名、tag 或 commit hash）。"
        "强烈建议在生产环境 pin 到具体 commit，以避免上游变更带来的供应链风险。",
    )
    trust_remote_code: bool = Field(default=True)
    torch_dtype: str = Field(
        default="auto",
        description="权重数据类型：auto / float16 / bfloat16 / float32",
    )
    device: str = Field(
        default="auto",
        description="目标设备：auto / cpu / cuda / cuda:N / mps",
    )
    device_map: str = Field(default="auto")
    attn_implementation: str = Field(
        default="sdpa",
        description="注意力实现：flash_attention_2 / sdpa / eager",
    )

    # VLM 后端
    backend_name: str = Field(
        default="florence-2",
        description="后端名称：florence-2, paligemma, youtuvl, generic-hf ...",
    )

    # LoRA
    use_lora: bool = Field(default=True)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

    # Gradient Checkpointing
    gradient_checkpointing: bool = Field(
        default=False,
        description="启用梯度检查点以减少显存占用（增加计算时间约 20-30%）",
    )

    # 高级激活值重计算策略
    activation_checkpointing_strategy: str = Field(
        default="none",
        description="激活值重计算策略: none / full / selective / auto"
    )
    checkpoint_target_layers: Optional[Union[List[str], str]] = Field(
        default=None,
        description="选择性重计算的目标层。可以是列表（如 ['encoder.layers.0', 'encoder.layers.1']）"
                    "或通配符模式（如 'encoder.layers.*'）"
    )
    checkpoint_every_n_layers: Optional[int] = Field(
        default=None, ge=1,
        description="每隔 N 层启用一次重计算（用于均匀分布策略）"
    )
    use_fp16: bool = Field(default=False, description="向后兼容：是否优先使用 FP16")
    use_bf16: bool = Field(default=False, description="向后兼容：是否优先使用 BF16")

    # ----------------------------------
    # 字段级校验
    # ----------------------------------
    @field_validator("attn_implementation")
    @classmethod
    def _check_attn_impl(cls, v: str) -> str:
        allowed = {"flash_attention_2", "sdpa", "eager"}
        if v not in allowed:
            raise ValueError(
                f"attn_implementation 必须是 {allowed} 之一，收到 '{v}'"
            )
        # 如果指定了 flash_attention_2，检查是否实际安装
        if v == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                warnings.warn(
                    "flash_attention_2 被指定但 flash-attn 未安装，"
                    "已自动降级为 sdpa。请执行: pip install flash-attn",
                    stacklevel=2,
                )
                return "sdpa"
        return v

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            warnings.warn(
                f"device='{v}' 不是标准值，标准值为 {allowed} 或 'cuda:N' 格式",
                stacklevel=2,
            )
        return v

    @field_validator("activation_checkpointing_strategy")
    @classmethod
    def _check_checkpoint_strategy(cls, v: str) -> str:
        allowed = {"none", "full", "selective", "auto"}
        if v not in allowed:
            raise ValueError(
                f"activation_checkpointing_strategy 必须是 {allowed} 之一，收到 '{v}'"
            )
        return v

    @field_validator("backend_name")
    @classmethod
    def _check_backend_name(cls, v: str) -> str:
        allowed_prefixes = {
            "florence-2", "florence2",
            "paligemma", "generic-hf", "auto", "hf",
            "youtuvl", "youtu-vl",
        }
        # 允许任意值，但警告未识别的后端
        return v

    # ----------------------------------
    # 兼容性
    # ----------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 数据配置
# ---------------------------------------------------------------------------


class DataConfig(WarnOnUnknownFieldsModel):
    """数据配置"""

    batch_size: int = Field(default=4, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)
    shuffle: bool = Field(default=True)
    drop_last: bool = Field(default=True)

    # DataLoader 性能调优
    prefetch_factor: Optional[int] = Field(
        default=None, ge=1,
        description="每个 worker 预取的 batch 数，默认 2（PyTorch 默认）。"
                    "GPU 训练建议设为 2-4 以提高数据管线吞吐量。"
    )
    persistent_workers: bool = Field(
        default=False,
        description="是否保持 worker 进程存活。多 epoch 训练时设为 True 可避免每个 epoch "
                    "重启 worker 的开销，需 num_workers > 0。"
    )

    # 数据增强
    use_augmentation: bool = Field(default=False)
    augmentation_prob: float = Field(default=0.5, ge=0.0, le=1.0)

    # 数据平衡
    use_balanced_sampling: bool = Field(default=True)
    max_samples_per_task: Optional[int] = Field(default=None, ge=1)

    # 预编码缓存
    use_cache: bool = Field(default=False, description="是否启用预编码缓存")
    cache_dir: Optional[str] = Field(default=None, description="磁盘缓存目录路径")
    cache_max_size: int = Field(
        default=10000, ge=1,
        description="内存 LRU 缓存允许保留的最大样本数"
    )

    # 分布式训练
    distributed: bool = Field(
        default=False,
        description="是否启用分布式训练（自动检测分布式环境，如 torchrun/ddp）"
    )
    world_size: Optional[int] = Field(
        default=None, ge=1,
        description="分布式世界大小，默认自动检测（通过 torch.distributed）"
    )
    rank: Optional[int] = Field(
        default=None, ge=0,
        description="当前进程 rank，默认自动检测"
    )
    local_rank: Optional[int] = Field(
        default=None, ge=0,
        description="当前进程 local rank，默认自动检测"
    )
    distributed_seed: Optional[int] = Field(
        default=None,
        description="分布式采样器的随机种子，默认使用全局随机种子"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 优化器配置
# ---------------------------------------------------------------------------


class OptimizationConfig(WarnOnUnknownFieldsModel):
    """优化器配置"""

    learning_rate: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, le=1.0)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)

    # 学习率调度
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    warmup_steps: Optional[int] = Field(default=None, ge=0)

    @field_validator("lr_scheduler_type")
    @classmethod
    def _check_scheduler(cls, v: str) -> str:
        allowed = {
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup",
        }
        if v not in allowed:
            raise ValueError(f"lr_scheduler_type 必须是 {allowed} 之一，收到 '{v}'")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizationConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 分布式训练配置
# ---------------------------------------------------------------------------


class DistributedConfig(WarnOnUnknownFieldsModel):
    """分布式训练配置

    支持 DDP、FSDP、DeepSpeed ZeRO 等多种分布式策略。
    通过 accelerate 库集成，兼容单卡/多卡/多节点训练。
    """

    # 基础分布式设置
    enabled: bool = Field(default=False, description="是否启用分布式训练")
    backend: str = Field(default="nccl", description="分布式后端: nccl / gloo / mpi")
    init_method: Optional[str] = Field(default=None, description="初始化方法，默认使用环境变量")
    timeout_seconds: int = Field(default=1800, ge=1, description="分布式操作超时时间（秒）")

    # 数据并行策略
    strategy: str = Field(
        default="ddp",
        description="分布式策略: ddp / fsdp / deepspeed / none"
    )

    # FSDP (Fully Sharded Data Parallel) 配置
    fsdp_sharding_strategy: str = Field(
        default="FULL_SHARD",
        description="FSDP 分片策略: FULL_SHARD / SHARD_GRAD_OP / NO_SHARD / HYBRID_SHARD"
    )
    fsdp_auto_wrap_policy: str = Field(
        default="TRANSFORMER_BASED_WRAP",
        description="FSDP 自动包装策略: TRANSFORMER_BASED_WRAP / SIZE_BASED_WRAP / NO_WRAP"
    )
    fsdp_min_num_params: float = Field(
        default=1e7, ge=0,
        description="SIZE_BASED_WRAP 策略下的最小参数数量（默认 10M）"
    )
    fsdp_backward_prefetch: str = Field(
        default="BACKWARD_PRE",
        description="FSDP 反向传播预取: BACKWARD_PRE / BACKWARD_POST / NO_PREFETCH"
    )
    fsdp_cpu_offload: bool = Field(
        default=False,
        description="FSDP CPU 卸载：将不使用的参数卸载到 CPU 内存（可训练更大模型）"
    )
    fsdp_activation_checkpointing: bool = Field(
        default=False,
        description="FSDP 内部激活值检查点"
    )

    # DeepSpeed ZeRO 配置
    deepspeed_stage: int = Field(
        default=0, ge=0, le=3,
        description="DeepSpeed ZeRO 阶段: 0(禁用) / 1(优化器状态分片) / 2(优化器+梯度分片) / 3(全参数分片)"
    )
    deepspeed_config_file: Optional[str] = Field(
        default=None,
        description="DeepSpeed JSON 配置文件路径（如提供则优先使用）"
    )
    deepspeed_offload_optimizer: bool = Field(
        default=False,
        description="ZeRO-Offload：将优化器状态卸载到 CPU/NVMe"
    )
    deepspeed_offload_param: bool = Field(
        default=False,
        description="ZeRO-Offload：将模型参数卸载到 CPU/NVMe（仅 ZeRO-3）"
    )

    # 流水线并行（实验性）
    pipeline_parallel_size: int = Field(
        default=1, ge=1,
        description="流水线并行度（1 表示禁用）"
    )
    pipeline_num_microbatches: int = Field(
        default=1, ge=1,
        description="流水线微批次数量"
    )

    # 张量并行（实验性，需 Megatron/Colossal-AI）
    tensor_parallel_size: int = Field(
        default=1, ge=1,
        description="张量并行度（1 表示禁用）"
    )

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        allowed = {"nccl", "gloo", "mpi"}
        if v not in allowed:
            raise ValueError(f"分布式后端必须是 {allowed} 之一，收到 '{v}'")
        return v

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        allowed = {"ddp", "fsdp", "deepspeed", "none"}
        if v not in allowed:
            raise ValueError(f"分布式策略必须是 {allowed} 之一，收到 '{v}'")
        return v

    @field_validator("fsdp_sharding_strategy")
    @classmethod
    def _check_fsdp_sharding(cls, v: str) -> str:
        allowed = {"FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"}
        if v not in allowed:
            raise ValueError(f"FSDP 分片策略必须是 {allowed} 之一，收到 '{v}'")
        return v

    @field_validator("fsdp_auto_wrap_policy")
    @classmethod
    def _check_fsdp_wrap(cls, v: str) -> str:
        allowed = {"TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP", "NO_WRAP"}
        if v not in allowed:
            raise ValueError(f"FSDP 包装策略必须是 {allowed} 之一，收到 '{v}'")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DistributedConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 任务调度配置
# ---------------------------------------------------------------------------


class TaskSchedulingConfig(WarnOnUnknownFieldsModel):
    """任务调度配置"""

    strategy: str = Field(default="round_robin")
    temperature: float = Field(default=1.0, gt=0.0)
    update_frequency: int = Field(default=100, ge=1)

    # 课程学习
    curriculum_start_epoch: int = Field(default=0, ge=0)
    curriculum_end_epoch: int = Field(default=10, ge=0)

    # 任务复杂度（可选，如果未提供则使用后端默认值）
    task_complexity: Optional[Dict[str, int]] = Field(default=None)

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        allowed = {"round_robin", "weighted", "curriculum", "adaptive"}
        if v not in allowed:
            raise ValueError(f"strategy 必须是 {allowed} 之一，收到 '{v}'")
        return v

    # 交叉字段校验：结束 epoch >= 开始 epoch
    @model_validator(mode="after")
    def _check_curriculum_order(self) -> TaskSchedulingConfig:
        if self.curriculum_end_epoch < self.curriculum_start_epoch:
            raise ValueError(
                f"curriculum_end_epoch ({self.curriculum_end_epoch}) "
                f"必须 >= curriculum_start_epoch ({self.curriculum_start_epoch})"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskSchedulingConfig:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# 训练配置（主配置）
# ---------------------------------------------------------------------------


class TrainingConfig(WarnOnUnknownFieldsModel):
    """完整训练配置

    Pydantic v2 保留 ``model_config`` 作为 ConfigDict 属性名，
    因此字段名改为 ``model_settings``，并通过 ``alias="model_config"``
    保持 YAML/JSON 文件的键名兼容。
    """

    model_settings: ModelConfig = Field(
        default_factory=ModelConfig,
        alias="model_config",
        description="模型配置（YAML/JSON 键名：model_config）",
    )
    data_settings: DataConfig = Field(
        default_factory=DataConfig,
        alias="data_config",
        description="数据配置（YAML/JSON 键名：data_config）",
    )
    optimization_settings: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        alias="optimization_config",
        description="优化器配置（YAML/JSON 键名：optimization_config）",
    )
    task_scheduling_settings: TaskSchedulingConfig = Field(
        default_factory=TaskSchedulingConfig,
        alias="task_scheduling_config",
        description="任务调度配置（YAML/JSON 键名：task_scheduling_config）",
    )
    distributed_settings: DistributedConfig = Field(
        default_factory=DistributedConfig,
        alias="distributed_config",
        description="分布式训练配置（YAML/JSON 键名：distributed_config）",
    )

    # ---------- 基础训练参数 ----------
    num_epochs: int = Field(default=10, ge=1)
    max_steps: Optional[int] = Field(default=None, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    save_steps: int = Field(default=1000, ge=1)
    logging_steps: int = Field(default=100, ge=1)

    # ---------- 输出目录 ----------
    output_dir: str = Field(default="./outputs")
    logging_dir: Optional[str] = Field(default=None)
    generate_training_report_on_end: bool = Field(
        default=True,
        description="训练结束时是否生成 HTML 可视化报告",
    )
    async_training_report: bool = Field(
        default=True,
        description="训练结束报告是否在后台线程异步生成",
    )

    # ---------- 设备 ----------
    device: str = Field(default="auto")

    # ---------- 混合精度 ----------
    use_fp16: bool = Field(default=False)
    use_bf16: bool = Field(default=False)

    # ---------- 梯度累积 ----------
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    gradient_checkpointing: bool = Field(default=False)

    # ---------- 检查点 ----------
    save_total_limit: int = Field(default=3, ge=1)
    keep_checkpoints: int = Field(default=3, ge=0)
    save_best_only: bool = Field(default=False)
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    greater_is_better: bool = Field(default=False)

    # ---------- 早停 ----------
    early_stopping_patience: int = Field(default=5, ge=0)
    early_stopping_threshold: float = Field(default=0.001, ge=0.0)

    # ---------- 任务 ----------
    tasks: List[str] = Field(default_factory=lambda: ["CAPTION"])
    task_weights: Dict[str, float] = Field(default_factory=dict)
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None

    # ---------- 实验 ----------
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Pydantic v2 ConfigDict：
    #   populate_by_name=True → 初始化时允许用字段名或 alias
    #   （序列化输出 alias 键名需通过 model_dump(by_alias=True) 实现）
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # ===============================================================
    # 校验器
    # ===============================================================
    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            warnings.warn(
                f"device='{v}' 不是标准值，标准值为 {allowed} 或 'cuda:N' 格式",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def _check_logging_dir(self) -> TrainingConfig:
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
        return self

    @model_validator(mode="after")
    def _check_fp16_exclusive(self) -> TrainingConfig:
        if self.use_fp16 and self.use_bf16:
            raise ValueError("use_fp16 和 use_bf16 不能同时为 True")
        if not self.use_fp16 and not self.use_bf16:
            try:
                import torch
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 8:
                        object.__setattr__(self, 'use_bf16', True)
                    else:
                        object.__setattr__(self, 'use_fp16', True)
            except Exception:
                pass

        # 检测 GPU 架构：BF16 需要 Ampere (SM80+) 或更新
        if self.use_bf16:
            try:
                import torch
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability[0] < 8:
                        warnings.warn(
                            f"当前 GPU 计算能力 {capability[0]}.{capability[1]} 不支持 BF16，"
                            f"已自动切换为 FP16。BF16 需要 SM80+ (Ampere) 架构。",
                            stacklevel=2,
                        )
                        object.__setattr__(self, 'use_bf16', False)
                        object.__setattr__(self, 'use_fp16', True)
            except Exception:
                pass  # 非 CUDA 环境（CPU/MPS）不做检测
        return self

    @model_validator(mode="after")
    def _check_max_steps_epochs(self) -> TrainingConfig:
        """校验 max_steps 并明确其与 num_epochs 的优先级关系。

        语义约定（v1/v2 训练栈一致）：
        - ``max_steps`` 一旦设定（> 0），即作为训练终止的**硬上限**，
          优先于 ``num_epochs``：达到 ``max_steps`` 后立即停止，
          即使尚未跑满 ``num_epochs``。
        - ``num_epochs`` 仅作为 epoch 维度的上界；当 ``max_steps`` 在
          某个 epoch 中途触发时，该 epoch 不会跑完。
        - 二者都未生效时，按 ``num_epochs`` 跑满。
        """
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps 必须 > 0")

        # 同时设定 max_steps 与非默认 num_epochs 时，显式告警优先级，
        # 避免用户误以为 num_epochs 会被跑满。
        if self.max_steps is not None and self.max_steps > 0 and self.num_epochs != 1:
            warnings.warn(
                f"已同时设定 max_steps={self.max_steps} 与 num_epochs={self.num_epochs}；"
                f"max_steps 优先生效，训练将在达到 {self.max_steps} 步时终止"
                f"（num_epochs 仅作为 epoch 上界，可能不会跑满）。",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _sync_device_to_model_settings(self) -> TrainingConfig:
        """CLI ``--device`` 写入顶层 ``device`` 时同步到 ``model_settings``。"""
        if self.device and self.device != "auto":
            self.model_settings.device = self.device
        return self

    # --- 嵌套配置便捷访问（v2 训练器 / 旧代码兼容） ---
    @property
    def num_workers(self) -> int:
        return self.data_settings.num_workers

    @property
    def weight_decay(self) -> float:
        return self.optimization_settings.weight_decay

    @property
    def adam_beta1(self) -> float:
        return self.optimization_settings.adam_beta1

    @property
    def adam_beta2(self) -> float:
        return self.optimization_settings.adam_beta2

    @property
    def adam_epsilon(self) -> float:
        return self.optimization_settings.adam_epsilon

    @property
    def warmup_ratio(self) -> float:
        return self.optimization_settings.warmup_ratio

    @property
    def lr_scheduler_type(self) -> str:
        return self.optimization_settings.lr_scheduler_type

    @property
    def use_lora(self) -> bool:
        return self.model_settings.use_lora

    @property
    def lora(self) -> LoRAConfig:
        return self.model_settings.lora_config

    @property
    def eval_batch_size(self) -> int:
        return self.data_settings.batch_size

    # ===============================================================
    # 序列化
    # ===============================================================
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，YAML/JSON 键名使用 alias（如 model_config）"""
        return self.model_dump(by_alias=True, exclude_none=True)

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            self.model_dump_json(by_alias=True, indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        import yaml
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["_metadata"] = {
            "created_at": datetime.now().isoformat(),
            "config_version": "2.0",
            "description": "FlorenceForge Training Configuration (Pydantic v2)",
        }
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            self.save_to_yaml(file_path)
        else:
            self.save_to_json(file_path)

    # ===============================================================
    # 反序列化
    # ===============================================================
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """从字典创建配置（支持字段名或 alias 键名）

        Pydantic v2 的 model_validate 会自动递归构造嵌套模型，
        并同时接受字段名（model_settings）或 alias（model_config）作为键。
        """
        return cls.model_validate(data)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> TrainingConfig:
        file_path = Path(file_path)
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            return cls.load_from_yaml(file_path)
        return cls.load_from_json(file_path)

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> TrainingConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]) -> TrainingConfig:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.pop("_metadata", None)
        return cls.from_dict(data)

    # 向后兼容别名：README 与示例代码使用 from_yaml / to_yaml
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> TrainingConfig:
        """从 YAML 加载配置（load_from_yaml 的别名，兼容 README 示例）"""
        return cls.load_from_yaml(file_path)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """保存为 YAML（save_to_yaml 的别名，如不存在则使用通用保存）"""
        if hasattr(self, "save_to_yaml"):
            return self.save_to_yaml(file_path)
        import yaml
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(by_alias=True), f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# 评估配置
# ---------------------------------------------------------------------------


class EvaluationConfig(WarnOnUnknownFieldsModel):
    """评估配置"""

    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    device: str = Field(default="auto")

    eval_split: str = Field(default="test")
    max_samples: Optional[int] = Field(default=None, ge=1)
    shuffle: bool = Field(default=False)

    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    save_predictions: bool = Field(default=True)
    save_detailed_results: bool = Field(default=False)

    output_dir: str = Field(default="./evaluation_results")
    save_format: str = Field(default="json")

    generate_plots: bool = Field(default=True)
    plot_format: str = Field(default="png")
    max_visualization_samples: int = Field(default=100, ge=0)

    task_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    use_amp: bool = Field(default=False)
    compile_model: bool = Field(default=False)

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            warnings.warn(f"device='{v}' 不是标准值", stacklevel=2)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return self.model_dump(by_alias=True, exclude_none=True)

    def save_to_json(self, json_path: Union[str, Path]) -> None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            self.model_dump_json(by_alias=True, indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def save_to_yaml(self, yaml_path: Union[str, Path]) -> None:
        import yaml
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> EvaluationConfig:
        with open(json_path, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> EvaluationConfig:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
