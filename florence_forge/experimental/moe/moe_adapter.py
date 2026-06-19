"""MoE 训练适配器 — 将实验性 MoE 能力接入主训练管线。

提供 ``MoETrainingAdapter``，将 ``MoEModel`` 包装为与 FlorenceForge
主训练栈（``MultiTaskTrainer``）兼容的接口。当前阶段为**接口层设计**：
实际稀疏内核和负载均衡损失仍在实验目录中迭代，本模块只负责统一入口和配置校验。

使用示例
--------

    from florence_forge.experimental.moe import MoETrainingAdapter
    from florence_forge.experimental.moe.moe_config import MoEConfig

    config = MoEConfig(num_experts=8, d_model=768, d_state=256, top_k=2)
    adapter = MoETrainingAdapter(config)

    # 替换主干中的特定层为 MoE 层
    model = Florence2MultiTaskModel(training_config.model_settings)
    model.load()
    adapter.inject_moe_into_model(model, target_layer_pattern="encoder\\.layer\\.([0-9]+)")

    # 之后 model 可直接传入 MultiTaskTrainer
    trainer = MultiTaskTrainer(model=model, train_dataset=train_dataset, config=config)
    trainer.train()
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

import torch
import torch.nn as nn

from .moe_config import MoEConfig
from .moe_layer import MoELayer
from .moe_validator import MoEValidator

logger = logging.getLogger(__name__)

_EXPERIMENTAL_MOE_WARNING = (
    "MoE 训练适配器处于实验阶段 (Tier-3)。API 可能变更；"
    "dense-all-experts 计算未节省 FLOPs，且负载均衡损失尚未实现。"
)


class MoETrainingAdapter:
    """将实验性 MoE 层接入 FlorenceForge 主训练管线。

    职责：
      1. 根据配置创建 ``MoELayer`` 并注入到已有模型中。
      2. 提供 MoE 专用的损失项（负载均衡损失、路由器 z-loss）接口。
      3. 与 ``MultiTaskTrainer`` 兼容：适配器本身不持有训练循环，
         只提供 ``forward_hook`` 和 ``loss_hook`` 供训练器调用。
    """

    def __init__(self, config: MoEConfig):
        warnings.warn(_EXPERIMENTAL_MOE_WARNING, UserWarning, stacklevel=2)
        self.config = config
        self._moe_layers: List[MoELayer] = []
        self._validators: List[MoEValidator] = []
        self._routing_stats: Dict[str, List[torch.Tensor]] = {}

    # ── 注入 / 卸载 ───────────────────────────────────────────────────

    def inject_moe_into_model(
        self,
        model: nn.Module,
        target_layer_pattern: Union[str, Pattern] = r"encoder\.layer\.([0-9]+)",
        create_moe_fn: Optional[Callable[[int, int], MoELayer]] = None,
    ) -> None:
        """将 MoE 层注入到匹配正则的模型子模块中。

        Args:
            model: 目标模型（如 Florence2MultiTaskModel）
            target_layer_pattern: 正则表达式，匹配要替换的子模块全名
            create_moe_fn: 可选工厂函数，签名 ``(d_model, d_state) -> MoELayer``
        """
        if isinstance(target_layer_pattern, str):
            target_layer_pattern = re.compile(target_layer_pattern)

        if create_moe_fn is None:
            create_moe_fn = self._default_create_moe

        matched = 0
        for name, module in model.named_modules():
            if target_layer_pattern.search(name):
                # 获取输入/输出维度
                d_model = self._infer_d_model(module)
                d_state = self._infer_d_state(module) or d_model
                moe_layer = create_moe_fn(d_model, d_state)
                self._replace_module(model, name, moe_layer)
                self._moe_layers.append(moe_layer)
                self._validators.append(MoEValidator(moe_layer))
                matched += 1
                logger.info(f"注入 MoE 层到 {name}: {moe_layer}")

        if matched == 0:
            logger.warning(
                f"未匹配到任何层 (pattern={target_layer_pattern.pattern})，"
                f"MoE 未注入。"
            )
        else:
            logger.info(f"共注入 {matched} 个 MoE 层")

    def remove_moe_from_model(self, model: nn.Module) -> None:
        """将模型中所有由本适配器注入的 MoE 层回退为原始模块。"""
        # TODO: 需要保存原始模块的备份才能回退。当前阶段仅做日志记录。
        logger.warning("MoE 卸载尚未实现，需提前保存原始层备份。")

    # ── 损失钩子 ───────────────────────────────────────────────────────

    def get_auxiliary_loss(self, *, loss_weight: float = 0.01) -> torch.Tensor:
        """计算辅助负载均衡损失（Auxiliary Load-Balancing Loss）。

        当前为**桩实现**，返回 0.0 张量。生产化需要：
          1. 在 ``MoELayer`` 中记录每批次的专家路由分布
          2. 计算 fraction-of-tokens-per-expert 与均匀分布的 KL 散度
        """
        if not self._moe_layers:
            return torch.tensor(0.0)
        # TODO: 接入真实路由统计后实现负载均衡损失
        return torch.tensor(0.0, requires_grad=True)

    def get_router_z_loss(self, *, loss_weight: float = 0.001) -> torch.Tensor:
        """路由器 z-loss（鼓励门控 logits 不要过大）。

        当前为**桩实现**，返回 0.0 张量。生产化需要：
          1. 在 ``SparseGate`` 中保存门控 logits
          2. 计算 log(sum(exp(logits)))² 的均值
        """
        if not self._moe_layers:
            return torch.tensor(0.0)
        # TODO: 接入真实门控 logits 后实现 z-loss
        return torch.tensor(0.0, requires_grad=True)

    def forward_hook(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """供 ``MultiTaskTrainer`` 在 forward 中调用的钩子。

        如果模型已注入 MoE 层，此钩子无需显式调用（MoE 层已嵌入模型前向图）。
        保留该接口用于未来外挂式 MoE（如 MoE adapter 旁路）。
        """
        return hidden_states

    def loss_hook(self, base_loss: torch.Tensor) -> torch.Tensor:
        """将 MoE 辅助损失叠加到基础损失上。

        Args:
            base_loss: 主训练损失（如 caption loss / OD loss）

        Returns:
            总损失 = base_loss + aux_loss + z_loss
        """
        if not self._moe_layers:
            return base_loss
        total_loss = base_loss + self.get_auxiliary_loss() + self.get_router_z_loss()
        return total_loss

    # ── 验证 / 诊断 ───────────────────────────────────────────────────

    def validate_all(self, sample_input: torch.Tensor) -> bool:
        """对所有注入的 MoE 层运行不变量验证。"""
        if not self._validators:
            logger.warning("没有注入的 MoE 层，跳过验证")
            return True
        results = [v.validate(sample_input) for v in self._validators]
        if not all(results):
            failed = [i for i, ok in enumerate(results) if not ok]
            logger.error(f"MoE 不变量验证失败: layer indices {failed}")
        return all(results)

    def summarize_routing(self) -> Dict[str, Any]:
        """返回路由统计摘要（专家激活频率、平均负载等）。"""
        if not self._routing_stats:
            return {"status": "no routing stats collected yet"}
        # TODO: 接入真实路由统计
        return {
            "num_moe_layers": len(self._moe_layers),
            "num_experts": self.config.num_experts,
            "top_k": self.config.top_k,
        }

    # ── 内部 ────────────────────────────────────────────────────────────

    def _default_create_moe(self, d_model: int, d_state: int) -> MoELayer:
        return MoELayer(
            num_experts=self.config.num_experts,
            d_model=d_model,
            d_state=d_state,
            top_k=self.config.top_k,
        )

    @staticmethod
    def _infer_d_model(module: nn.Module) -> int:
        """从模块属性推断 d_model。"""
        if hasattr(module, "d_model"):
            return int(module.d_model)
        if hasattr(module, "in_features"):
            return int(module.in_features)
        if hasattr(module, "hidden_size"):
            return int(module.hidden_size)
        return 768  # fallback

    @staticmethod
    def _infer_d_state(module: nn.Module) -> Optional[int]:
        """从模块属性推断 d_state。"""
        if hasattr(module, "d_state"):
            return int(module.d_state)
        if hasattr(module, "out_features"):
            return int(module.out_features)
        return None

    @staticmethod
    def _replace_module(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
        """递归替换子模块。"""
        parts = full_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
