"""MoE 配置模型 — Pydantic v2 校验。

定义 ``MoEConfig``，供 ``MoETrainingAdapter`` 和 ``MoETrainer`` 统一使用。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MoEConfig(BaseModel):
    """MoE 配置"""

    num_experts: int = Field(default=8, ge=2, description="专家数量")
    d_model: int = Field(default=768, ge=1, description="模型隐藏维度")
    d_state: int = Field(default=256, ge=1, description="专家状态维度")
    top_k: int = Field(default=2, ge=1, description="每个 token 激活的专家数")
    aux_loss_weight: float = Field(default=0.01, ge=0.0, description="负载均衡损失权重")
    z_loss_weight: float = Field(default=0.001, ge=0.0, description="路由器 z-loss 权重")
    capacity_factor: float = Field(default=1.25, ge=1.0, description="专家容量溢出因子")
    enable_expert_parallelism: bool = Field(default=False, description="是否启用专家并行")
    device_map: str = Field(default="auto", description="设备映射策略")

    class Config:
        extra = "ignore"
