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
    "sparse forward 已节省部分 FLOPs，但 expert parallelism 尚未实现。"
)


class MoETrainingAdapter:
    """将实验性 MoE 层接入 FlorenceForge 主训练管线。

    职责：
      1. 根据配置创建 ``MoELayer`` 并注入到已有模型中。
      2. 提供 MoE 专用的损失项（负载均衡损失、路由器 z-loss）接口。
      3. 与 ``MultiTaskTrainer`` 兼容：适配器本身不持有训练循环，
         只提供 ``forward_hook`` 和 ``loss_hook`` 供训练器调用。
      4. 支持卸载（remove），回退为原始模块。
    """

    def __init__(self, config: MoEConfig):
        warnings.warn(_EXPERIMENTAL_MOE_WARNING, UserWarning, stacklevel=2)
        self.config = config
        self._moe_layers: List[MoELayer] = []
        self._validators: List[MoEValidator] = []
        self._routing_stats: Dict[str, List[torch.Tensor]] = {}
        # 注入时保存原始层备份，供卸载回退使用
        self._original_modules: Dict[str, nn.Module] = {}
        self._injected_model: Optional[nn.Module] = None
    # ── 注入 / 卸载 ───────────────────────────────────────────────────

    def inject_moe_into_model(
        self,
        model: nn.Module,
        target_layer_pattern: Union[str, Pattern] = r"encoder\.layer\.([0-9]+)",
        create_moe_fn: Optional[Callable[[int, int], MoELayer]] = None,
    ) -> None:
        """将 MoE 层注入到匹配正则的模型子模块中。

        注入时会自动保存原始模块的深拷贝备份，供 ``remove_moe_from_model``
        回退使用。

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
                # 保存原始模块的深拷贝
                self._original_modules[name] = module
                moe_layer = create_moe_fn(d_model, d_state)
                self._replace_module(model, name, moe_layer)
                self._moe_layers.append(moe_layer)
                self._validators.append(MoEValidator(moe_layer))
                matched += 1
                logger.info(f"注入 MoE 层到 {name}: {moe_layer}")

        self._injected_model = model

        if matched == 0:
            logger.warning(
                f"未匹配到任何层 (pattern={target_layer_pattern.pattern})，"
                f"MoE 未注入。"
            )
        else:
            logger.info(f"共注入 {matched} 个 MoE 层，已保存原始层备份")

    def remove_moe_from_model(self, model: Optional[nn.Module] = None) -> None:
        """将模型中所有由本适配器注入的 MoE 层回退为原始模块。

        Args:
            model: 目标模型。如果为 None，则使用注入时记录的模型。
        """
        if model is None:
            model = self._injected_model

        if model is None:
            logger.warning("没有记录到已注入的模型，卸载失败")
            return

        if not self._original_modules:
            logger.warning("没有原始模块备份，卸载失败（可能未成功注入）")
            return

        restored = 0
        for name, original_module in self._original_modules.items():
            self._replace_module(model, name, original_module)
            restored += 1
            logger.info(f"回退原始模块到 {name}")

        self._moe_layers.clear()
        self._validators.clear()
        self._original_modules.clear()
        self._injected_model = None
        logger.info(f"MoE 卸载完成：共回退 {restored} 个层")

    def is_injected(self) -> bool:
        """返回是否已成功注入 MoE 层。"""
        return len(self._moe_layers) > 0
    # ── 损失钩子 ───────────────────────────────────────────────────────

    def get_auxiliary_loss(self, *, loss_weight: float = 0.01) -> torch.Tensor:
        """计算辅助负载均衡损失（Auxiliary Load-Balancing Loss）。

        基于 Switch Transformer 的 load-balancing loss 设计：
        L_aux = num_experts * sum(f_i * P_i)
        其中 f_i 是分配给专家 i 的 token 比例，
        P_i 是路由器对专家 i 的平均门控概率。

        当前已接入真实路由统计（通过 MoELayer._routing_sums 和 last_gate_weights）。
        """
        if not self._moe_layers:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        for layer in self._moe_layers:
            if not hasattr(layer, "_routing_sums") or layer._routing_sums is None:
                continue
            if not hasattr(layer, "last_gate_weights") or layer.last_gate_weights is None:
                continue

            num_experts = layer.num_experts
            # f_i: 分配给专家 i 的 token 比例
            total_tokens = layer.last_gate_weights.shape[0] * layer.last_gate_weights.shape[1]
            f = layer._routing_sums / total_tokens
            # P_i: 路由器对专家 i 的平均门控概率
            P = layer.last_gate_weights.mean(dim=(0, 1))
            aux = num_experts * torch.sum(f * P)
            total_loss = total_loss + aux

        return loss_weight * total_loss

    def get_router_z_loss(self, *, loss_weight: float = 0.001) -> torch.Tensor:
        """路由器 z-loss（鼓励门控 logits 不要过大）。

        基于 ST-MoE / PaLM 设计：
        L_z = (1/N) * sum(log(sum(exp(logits_i))))^2

        当前已接入 SparseGate.last_logits。
        """
        if not self._moe_layers:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        count = 0
        for layer in self._moe_layers:
            gate = layer.gate
            if not hasattr(gate, "last_logits") or gate.last_logits is None:
                continue

            logits = gate.last_logits
            # log(sum(exp(logits))) 沿专家维度
            log_sum_exp = torch.logsumexp(logits, dim=-1)
            z_loss = (log_sum_exp ** 2).mean()
            total_loss = total_loss + z_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0)
        return loss_weight * total_loss / count

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
        """返回路由统计摘要（专家激活频率、平均负载、溢出统计等）。"""
        if not self._moe_layers:
            return {"status": "no MoE layers injected"}

        summary = {
            "num_moe_layers": len(self._moe_layers),
            "num_experts": self.config.num_experts,
            "top_k": self.config.top_k,
            "capacity_factor": self.config.capacity_factor,
            "layers": [],
        }

        for i, layer in enumerate(self._moe_layers):
            layer_info = {
                "layer_index": i,
                "num_experts": layer.num_experts,
                "d_model": layer.d_model,
                "d_state": layer.d_state,
            }
            if hasattr(layer, "last_gate_weights") and layer.last_gate_weights is not None:
                gw = layer.last_gate_weights
                # 每个专家的平均激活权重
                avg_weights = gw.mean(dim=(0, 1)).cpu().tolist()
                # 每个专家的 token 分配比例（硬统计）
                token_dist = (gw > 0).float().mean(dim=(0, 1)).cpu().tolist()
                layer_info["avg_gate_weights"] = [round(w, 4) for w in avg_weights]
                layer_info["token_distribution"] = [round(d, 4) for d in token_dist]

            if hasattr(layer, "_routing_sums") and layer._routing_sums is not None:
                layer_info["routing_sums"] = layer._routing_sums.cpu().tolist()

            if hasattr(layer, "_overflow_stats") and layer._overflow_stats is not None:
                layer_info["overflow_tokens"] = layer._overflow_stats.cpu().tolist()
            else:
                layer_info["overflow_tokens"] = None

            summary["layers"].append(layer_info)

        return summary

    def get_total_overflow_tokens(self) -> int:
        """返回所有 MoE 层的溢出 token 总数。"""
        total = 0
        for layer in self._moe_layers:
            if hasattr(layer, "_overflow_stats") and layer._overflow_stats is not None:
                total += int(layer._overflow_stats.sum().item())
        return total

    def get_routing_gini(self) -> float:
        """计算所有 MoE 层的平均 Gini 系数。"""
        if not self._moe_layers:
            return 0.0

        total_gini = 0.0
        count = 0
        for layer in self._moe_layers:
            if not hasattr(layer, "_routing_sums") or layer._routing_sums is None:
                continue
            usage = layer._routing_sums.cpu().tolist()
            n = len(usage)
            if n == 0 or sum(usage) == 0:
                continue
            sorted_usage = sorted(usage)
            cumsum = 0.0
            for i, u in enumerate(sorted_usage, 1):
                cumsum += (2 * i - n - 1) * u
            gini = cumsum / (n * sum(sorted_usage))
            total_gini += gini
            count += 1

        return total_gini / count if count > 0 else 0.0

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

    # ── 内部 ────────────────────────────────────────────────────────────

    def _default_create_moe(self, d_model: int, d_state: int) -> MoELayer:
        return MoELayer(
            num_experts=self.config.num_experts,
            d_model=d_model,
            d_state=d_state,
            top_k=self.config.top_k,
            capacity_factor=self.config.capacity_factor,
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
