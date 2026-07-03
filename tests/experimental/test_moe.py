"""Regression tests for experimental MoE components."""

import warnings

import torch

warnings.filterwarnings(
    "ignore",
    message="florence_forge.experimental.moe is experimental.*",
)

from florence_forge.training.moe.moe_decoder import MoEDecoder
from florence_forge.training.moe.moe_encoder import MoEEncoder
from florence_forge.training.moe.moe_layer import MoELayer
from florence_forge.training.moe.moe_model import MoEModel
from florence_forge.training.moe.moe_validator import MoEValidator
from florence_forge.training.moe.selective_ssm_mixer import SelectiveSSMMixer
from florence_forge.training.moe.sparse_gate import SparseGate


def test_sparse_gate_returns_normalized_expert_weights():
    gate = SparseGate(d_model=4, d_state=8, n_heads=3)
    x = torch.randn(2, 5, 4)

    weights = gate(x)

    assert weights.shape == (2, 5, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_moe_layer_forward_shape_and_gradient_flow():
    layer = MoELayer(num_experts=3, d_model=4, d_state=6)
    x = torch.randn(2, 5, 4, requires_grad=True)

    output = layer(x)
    output.sum().backward()

    assert output.shape == (2, 5, 6)
    assert x.grad is not None
    assert layer.gate.proj.weight.grad is not None
    assert layer.experts[0].weight.grad is not None


def test_moe_validator_accepts_valid_layer_output():
    layer = MoELayer(num_experts=3, d_model=4, d_state=6)
    validator = MoEValidator(layer)

    assert validator.validate(torch.randn(2, 5, 4)) is True


def test_selective_ssm_mixer_shape_and_gradient_flow():
    mixer = SelectiveSSMMixer(d_model=4, d_state=8, n_heads=2)
    x = torch.randn(2, 5, 4, requires_grad=True)

    output = mixer(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert mixer.state_in.weight.grad is not None
    assert mixer.sparse_gate.proj.weight.grad is not None


def test_moe_model_token_output_shape():
    x = torch.randn(2, 5, 4)

    assert MoEEncoder(num_experts=3, d_model=4, d_state=6)(x).shape == (2, 5, 6)
    assert MoEDecoder(num_experts=3, d_model=4, d_state=6)(x).shape == (2, 5, 6)

    model = MoEModel(
        vocab_size=11,
        max_position_embeddings=8,
        num_experts=2,
        d_model=4,
        d_state=6,
    )
    input_ids = torch.ones(2, 5, dtype=torch.long)

    assert model(input_ids).shape == (2, 5, 11)


def test_sparse_gate_hard_routing_returns_one_hot():
    """Hard routing 模式下每个 token 恰好路由到 1 个专家。"""
    gate = SparseGate(d_model=4, d_state=8, n_heads=3, hard_routing=True)
    x = torch.randn(2, 5, 4)

    weights = gate(x)

    assert weights.shape == (2, 5, 3)
    # 每个 token 只有一个专家的权重为 1.0
    nonzero_count_per_token = (weights > 0).sum(dim=-1)
    assert torch.all(nonzero_count_per_token == 1)
    # 权重值恰好为 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_moe_layer_hard_routing_forward_shape_and_gradient():
    """Hard routing 模式下 MoELayer 前向传播和梯度回传正常。"""
    torch.manual_seed(42)
    layer = MoELayer(num_experts=3, d_model=4, d_state=6, hard_routing=True)
    x = torch.randn(2, 5, 4, requires_grad=True)

    output = layer(x)
    output.sum().backward()

    assert output.shape == (2, 5, 6)
    assert x.grad is not None
    assert layer.gate.proj.weight.grad is not None
    # hard routing 下每个 token 只路由到 1 个专家，
    # 因此可能并非所有专家都有梯度，但被选中的专家应有梯度
    expert_grads = [e.weight.grad for e in layer.experts if e.weight.grad is not None]
    assert len(expert_grads) > 0, "至少应有一个专家接收到梯度"


def test_moe_layer_hard_routing_exactly_one_expert_per_token():
    """验证 hard routing 下每个 token 恰好由 1 个专家处理。"""
    moe = MoELayer(num_experts=8, d_model=16, d_state=16, hard_routing=True, capacity_factor=None)
    x = torch.randn(4, 5, 16)
    _ = moe(x)

    gate_weights = moe.last_gate_weights
    nonzero_count_per_token = (gate_weights > 0).sum(dim=-1)
    assert torch.all(nonzero_count_per_token == 1)
