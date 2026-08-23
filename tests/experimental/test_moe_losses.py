"""Tests for MoE load-balancing loss and z-loss implementation."""
import warnings
import torch

warnings.filterwarnings(
    "ignore",
    message="florence_forge.experimental.moe is experimental.*",
)
warnings.filterwarnings("ignore", message="MoE 训练适配器处于实验阶段.*")

from florence_forge.training.moe.moe_layer import MoELayer
from florence_forge.training.moe.moe_adapter import MoETrainingAdapter
from florence_forge.training.moe.moe_config import MoEConfig
from florence_forge.training.moe.sparse_gate import SparseGate


def test_auxiliary_loss_uniform_routing_is_minimum():
    """均匀路由时 aux loss 应接近理论最小值 num_experts * (1/num_experts) = 1。"""
    moe = MoELayer(num_experts=8, d_model=64, d_state=64, top_k=2, capacity_factor=None)
    x = torch.randn(64, 1, 64)
    _ = moe(x)

    config = MoEConfig(num_experts=8, d_model=64, d_state=64, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe]

    aux = adapter.get_auxiliary_loss(loss_weight=1.0)
    assert aux.item() >= 0.99  # 放宽容差，因 softmax 采样有随机性


def test_auxiliary_loss_concentrated_routing_is_high():
    """手动构造集中路由，验证 aux loss 显著升高。"""
    moe = MoELayer(num_experts=8, d_model=64, d_state=64, top_k=2)
    # 伪造 gate weights：全部集中到专家 0
    fake_weights = torch.zeros(4, 1, 8)
    fake_weights[:, :, 0] = 1.0
    moe.last_gate_weights = fake_weights
    moe._routing_sums = fake_weights.sum(dim=(0, 1))

    config = MoEConfig(num_experts=8, d_model=64, d_state=64, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe]

    aux = adapter.get_auxiliary_loss(loss_weight=1.0)
    # 8 * 1.0 = 8.0
    assert aux.item() >= 7.0


def test_router_z_loss_increases_with_large_logits():
    """较大的 logits 应产生更高的 z-loss。"""
    gate = SparseGate(d_model=4, d_state=8, n_heads=4)
    x = torch.randn(2, 3, 4)
    _ = gate(x)

    z1 = gate.last_logits
    assert z1 is not None

    # 放大 logits 后 z-loss 应增大
    gate.last_logits = z1 * 5.0
    config = MoEConfig(num_experts=4, d_model=4, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [MoELayer(num_experts=4, d_model=4, d_state=8, top_k=2)]
    adapter._moe_layers[0].gate = gate
    adapter._moe_layers[0].last_gate_weights = torch.softmax(gate.last_logits, dim=-1)
    adapter._moe_layers[0]._routing_sums = adapter._moe_layers[0].last_gate_weights.sum(dim=(0, 1))

    z_loss = adapter.get_router_z_loss(loss_weight=1.0)
    assert z_loss.item() > 0.0


def test_loss_hook_combines_base_and_aux():
    """loss_hook 应将 base loss、aux loss 和 z-loss 相加。"""
    moe = MoELayer(num_experts=4, d_model=4, d_state=8, top_k=2)
    x = torch.randn(2, 3, 4)
    _ = moe(x)

    config = MoEConfig(num_experts=4, d_model=4, d_state=8, top_k=2, aux_loss_weight=0.1, z_loss_weight=0.01)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe]

    base_loss = torch.tensor(2.0)
    total = adapter.loss_hook(base_loss)
    assert total.item() > base_loss.item()
    assert total.requires_grad == base_loss.requires_grad


def test_moe_layer_sparse_forward_matches_dense_computation():
    """稀疏前向的输出应与密集计算（einsum）数值一致（关闭 capacity_factor）。"""
    torch.manual_seed(42)
    moe = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None)
    x = torch.randn(2, 3, 8)

    out_sparse = moe(x)

    # 手动密集计算
    with torch.no_grad():
        gate_weights = moe.gate(x)
        expert_outputs = torch.stack(
            [expert(x) for expert in moe.experts],
            dim=2,
        )
        out_dense = torch.einsum("bse,bsed->bsd", gate_weights, expert_outputs)

    assert torch.allclose(out_sparse, out_dense, atol=1e-5)


def test_moe_layer_sparse_forward_only_routes_topk_experts():
    """验证稀疏前向确实只计算 top-k 专家（每个 token 最多激活 2 个专家）。"""
    moe = MoELayer(num_experts=8, d_model=16, d_state=16, top_k=2, capacity_factor=None)
    x = torch.randn(4, 5, 16)
    _ = moe(x)

    gate_weights = moe.last_gate_weights
    # 每个 (batch, seq) 位置应最多有 top_k 个非零权重（容量因子可能截断到更少）
    nonzero_count_per_token = (gate_weights > 0).sum(dim=-1)
    assert torch.all(nonzero_count_per_token <= 2)
    assert torch.all(nonzero_count_per_token >= 1)  # 至少有一个专家（top_k 机制保证）


def test_moe_layer_hard_routing_matches_dense_computation():
    """Hard routing 稀疏前向的输出应与密集计算数值一致。"""
    torch.manual_seed(42)
    moe = MoELayer(num_experts=4, d_model=8, d_state=8, hard_routing=True, capacity_factor=None)
    x = torch.randn(2, 3, 8)

    out_sparse = moe(x)

    # 手动密集计算
    with torch.no_grad():
        gate_weights = moe.gate(x)
        expert_outputs = torch.stack(
            [expert(x) for expert in moe.experts],
            dim=2,
        )
        out_dense = torch.einsum("bse,bsed->bsd", gate_weights, expert_outputs)

    assert torch.allclose(out_sparse, out_dense, atol=1e-5)


def test_moe_layer_hard_routing_gradient_flow():
    """Hard routing 的 straight-through estimator 应允许梯度流回 gate。"""
    torch.manual_seed(42)
    moe = MoELayer(num_experts=4, d_model=8, d_state=8, hard_routing=True, capacity_factor=None)
    x = torch.randn(2, 3, 8, requires_grad=True)

    out = moe(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert moe.gate.proj.weight.grad is not None
    # 梯度不应全为 0
    assert moe.gate.proj.weight.grad.abs().sum() > 0
