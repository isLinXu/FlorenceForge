"""
MoE 工具函数

稀疏门控 MoE 相关工具函数。
"""

from .moe_layer import MoELayer


def create_moe_layer(
    num_experts: int,
    d_model: int,
    d_state: int,
    sparse_gate_threshold: float = 0.5,
) -> MoELayer:
    """创建 MoE 层

    Args:
        num_experts: 专家数量
        d_model: 模型维度
        d_state: 状态维度
        sparse_gate_threshold: 稀疏门控阈值

    Returns:
        MoE 层实例
    """
    return MoELayer(
        num_experts=num_experts,
        d_model=d_model,
        d_state=d_state,
        gate_threshold=sparse_gate_threshold,
    )
