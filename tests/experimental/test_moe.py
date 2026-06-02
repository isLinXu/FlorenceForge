"""Regression tests for experimental MoE components."""

import warnings

import torch

warnings.filterwarnings(
    "ignore",
    message="florence_forge.experimental.moe is experimental.*",
)

from florence_forge.experimental.moe.moe_decoder import MoEDecoder
from florence_forge.experimental.moe.moe_encoder import MoEEncoder
from florence_forge.experimental.moe.moe_layer import MoELayer
from florence_forge.experimental.moe.moe_model import MoEModel
from florence_forge.experimental.moe.moe_validator import MoEValidator
from florence_forge.experimental.moe.selective_ssm_mixer import SelectiveSSMMixer
from florence_forge.experimental.moe.sparse_gate import SparseGate


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


def test_moe_encoder_decoder_and_token_model_are_runnable():
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
