"""Tests for Expert Parallelism MoE implementation."""

import warnings

import pytest
import torch

warnings.filterwarnings(
    "ignore",
    message="florence_forge.experimental.moe is experimental.*",
)

from florence_forge.training.moe.expert_parallel import ExpertParallelMoE, ExpertParallelMoELayer
from florence_forge.training.moe.moe_layer import MoELayer


# ── Basic construction tests ─────────────────────────────────────────


def test_ep_moe_construction_validates_args():
    """Constructor should validate hyperparameters."""
    with pytest.raises(ValueError, match="num_experts must be positive"):
        ExpertParallelMoE(num_experts=0, d_model=4, d_state=6)

    with pytest.raises(ValueError, match="d_model and d_state must be positive"):
        ExpertParallelMoE(num_experts=4, d_model=0, d_state=6)

    with pytest.raises(ValueError, match="world_size must be positive"):
        ExpertParallelMoE(num_experts=4, d_model=4, d_state=6, world_size=0)

    with pytest.raises(ValueError, match="rank must be in"):
        ExpertParallelMoE(num_experts=4, d_model=4, d_state=6, world_size=2, rank=2)

    with pytest.raises(ValueError, match="must be divisible by world_size"):
        ExpertParallelMoE(num_experts=5, d_model=4, d_state=6, world_size=2)


def test_ep_moe_expert_to_device_mapping():
    """Expert-to-device mapping should be deterministic and round-trip."""
    ep = ExpertParallelMoE(num_experts=8, d_model=4, d_state=6, world_size=4, simulate=True)

    assert ep.expert_to_device(0) == 0
    assert ep.expert_to_device(1) == 0
    assert ep.expert_to_device(2) == 1
    assert ep.expert_to_device(7) == 3

    assert ep.device_to_experts(0) == [0, 1]
    assert ep.device_to_experts(3) == [6, 7]

    assert ep.get_local_expert_indices() == [0, 1]  # rank 0 by default


def test_ep_moe_simulation_mode_single_gpu():
    """simulate=True should place all experts on the local device for full testing."""
    ep = ExpertParallelMoE(num_experts=8, d_model=4, d_state=6, world_size=4, simulate=True)
    assert ep.simulate is True
    assert len(ep.local_experts) == 8  # all experts available in simulation mode
    assert ep._logical_expert_indices == [0, 1]  # but logically rank 0 owns first 2


# ── Forward pass tests ───────────────────────────────────────────────


def test_ep_moe_forward_shape_and_gradient():
    """Forward should produce correct output shape and support backprop."""
    ep = ExpertParallelMoE(num_experts=4, d_model=8, d_state=8, top_k=2, simulate=True)
    x = torch.randn(2, 5, 8, requires_grad=True)

    out = ep(x)
    out.sum().backward()

    assert out.shape == (2, 5, 8)
    assert x.grad is not None
    assert ep.gate.proj.weight.grad is not None
    assert ep.local_experts[0].weight.grad is not None


def test_ep_moe_forward_matches_dense_computation():
    """EP simulation output should match dense einsum computation."""
    torch.manual_seed(42)
    ep = ExpertParallelMoE(
        num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None, simulate=True
    )
    x = torch.randn(2, 3, 8)

    out_ep = ep(x)

    # Dense reference: compute all experts then einsum with gate weights
    with torch.no_grad():
        gate_weights = ep.gate(x)
        # In simulation mode, all experts live on the same device
        all_experts = ep.local_experts if ep.world_size == 1 else None
        if all_experts is not None and len(all_experts) == ep.num_experts:
            expert_outputs = torch.stack([expert(x) for expert in all_experts], dim=2)
            out_dense = torch.einsum("bse,bsed->bsd", gate_weights, expert_outputs)
            assert torch.allclose(out_ep, out_dense, atol=1e-5)


def test_ep_moe_forward_with_capacity_factor():
    """Capacity factor should truncate overflow tokens and record stats."""
    ep = ExpertParallelMoE(
        num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=1.0, simulate=True
    )
    x = torch.randn(8, 8, 8)  # large batch to trigger overflow
    _ = ep(x)

    assert ep._overflow_stats is not None
    assert ep._overflow_stats.shape == (4,)
    assert ep._dispatch_counts is not None
    assert ep._dispatch_counts.shape == (4,)


def test_ep_moe_forward_only_routes_topk():
    """Each token should activate at most top_k experts."""
    ep = ExpertParallelMoE(
        num_experts=8, d_model=16, d_state=16, top_k=2, capacity_factor=None, simulate=True
    )
    x = torch.randn(4, 5, 16)
    _ = ep(x)

    gate_weights = ep.last_gate_weights
    nonzero_per_token = (gate_weights > 0).sum(dim=-1)
    assert torch.all(nonzero_per_token <= 2)
    assert torch.all(nonzero_per_token >= 1)


def test_ep_moe_forward_different_world_sizes():
    """Simulation should work for various world sizes."""
    for world_size in [1, 2, 4, 8]:
        ep = ExpertParallelMoE(
            num_experts=8,
            d_model=8,
            d_state=8,
            top_k=2,
            world_size=world_size,
            simulate=True,
        )
        x = torch.randn(2, 3, 8)
        out = ep(x)
        assert out.shape == (2, 3, 8)


# ── ExpertParallelMoELayer compatibility tests ───────────────────────


def test_ep_layer_compatible_with_moe_adapter():
    """ExpertParallelMoELayer should expose MoELayer-compatible attributes."""
    from florence_forge.training.moe.moe_adapter import MoETrainingAdapter
    from florence_forge.training.moe.moe_config import MoEConfig

    ep_layer = ExpertParallelMoELayer(
        num_experts=4, d_model=8, d_state=8, top_k=2, simulate=True
    )

    # Must expose compatibility attributes
    assert ep_layer.num_experts == 4
    assert ep_layer.d_model == 8
    assert ep_layer.d_state == 8
    assert hasattr(ep_layer, "experts")
    assert hasattr(ep_layer, "gate")
    assert hasattr(ep_layer, "selective_params")
    assert hasattr(ep_layer, "last_gate_weights")
    assert hasattr(ep_layer, "_overflow_stats")

    x = torch.randn(2, 3, 8)
    _ = ep_layer(x)
    assert ep_layer.last_gate_weights is not None
    assert ep_layer._routing_sums is not None

    # Adapter should be able to compute aux loss on EP layer
    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [ep_layer]
    aux = adapter.get_auxiliary_loss(loss_weight=1.0)
    assert isinstance(aux, torch.Tensor)
    assert aux.item() >= 0.0


# ── All-to-all simulation tests ──────────────────────────────────────


def test_all_to_all_dispatch_simulation_roundtrip():
    """Dispatch then combine should be a no-op in simulation mode."""
    ep = ExpertParallelMoE(num_experts=4, d_model=8, d_state=8, simulate=True)
    tokens = torch.randn(12, 8)
    send_counts = [3, 3, 3, 3]

    received, recv_counts = ep._all_to_all_dispatch(tokens, send_counts)
    assert received.shape == tokens.shape
    assert recv_counts == send_counts

    combined = ep._all_to_all_combine(received, recv_counts)
    assert torch.equal(combined, tokens)


def test_expert_device_map_summary():
    """summarize should return a complete diagnostic dict."""
    ep = ExpertParallelMoE(num_experts=8, d_model=4, d_state=6, world_size=4, simulate=True)
    x = torch.randn(2, 3, 4)
    _ = ep(x)

    summary = ep.summarize()
    assert summary["num_experts"] == 8
    assert summary["world_size"] == 4
    assert summary["experts_per_device"] == 2
    assert "dispatch_counts" in summary
    assert "overflow_tokens" in summary or "routing_sums" in summary

    device_map = ep.get_expert_device_map()
    assert len(device_map) == 8
    assert set(device_map.values()) == {0, 1, 2, 3}


# ── Integration with MoETrainingAdapter ─────────────────────────────


def test_ep_layer_can_be_injected_and_removed():
    """ExpertParallelMoELayer should be injectable via MoETrainingAdapter."""
    from florence_forge.training.moe.moe_adapter import MoETrainingAdapter
    from florence_forge.training.moe.moe_config import MoEConfig

    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)

    # Create a simple dummy model with a Linear layer that matches d_model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.ModuleDict()
            self.encoder["layer_0"] = torch.nn.Linear(8, 8)
            self.encoder["layer_1"] = torch.nn.Linear(8, 8)

    model = DummyModel()

    # Inject EP layer using custom factory
    def make_ep(d_model, d_state):
        return ExpertParallelMoELayer(
            num_experts=config.num_experts,
            d_model=d_model,
            d_state=d_state,
            top_k=config.top_k,
            simulate=True,
        )

    adapter.inject_moe_into_model(model, target_layer_pattern=r"encoder\.layer_\d+", create_moe_fn=make_ep)
    assert adapter.is_injected()
    assert len(adapter._moe_layers) == 2

    # Verify forward works
    x = torch.randn(2, 3, 8)
    for layer in adapter._moe_layers:
        out = layer(x)
        assert out.shape == (2, 3, 8)

    # Verify loss hooks work
    base_loss = torch.tensor(2.0)
    total_loss = adapter.loss_hook(base_loss)
    assert total_loss.item() > base_loss.item()

    # Unload
    adapter.remove_moe_from_model(model)
    assert not adapter.is_injected()


# ── Regression: ensure existing MoE tests still pass ─────────────────


def test_existing_moe_layer_unchanged():
    """Existing MoELayer should be unaffected by EP additions."""
    moe = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2)
    x = torch.randn(2, 3, 8)
    out = moe(x)
    assert out.shape == (2, 3, 8)
    assert moe.last_gate_weights is not None
    assert moe._routing_sums is not None
