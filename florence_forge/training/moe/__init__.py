"""Production-grade Mixture-of-Experts (MoE) components for FlorenceForge training.

This module provides sparse-gated MoE layers, load-balancing losses, and
training-loop adapters for integrating MoE into the FlorenceForge training pipeline.

Key Components:
- ``MoELayer``: Sparse forward MoE layer with capacity factor and top-k routing.
- ``MoETrainingAdapter``: Inject/remove MoE layers into existing models.
- ``MoEConfig``: Pydantic configuration model for MoE hyperparameters.
- ``SparseGate``: Trainable gate for per-token expert routing.

Usage Example:
    from florence_forge.training.moe import MoETrainingAdapter, MoEConfig

    config = MoEConfig(num_experts=8, d_model=768, d_state=256, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter.inject_moe_into_model(model, target_layer_pattern="encoder\\.layer\\.([0-9]+)")
"""

from __future__ import annotations

__all__ = [
    "MoETrainingAdapter",
    "MoEConfig",
    "MoELayer",
    "SparseGate",
    "MoEValidator",
    "ExpertParallelMoE",
    "ExpertParallelMoELayer",
    "MoEEncoder",
    "MoEDecoder",
    "MoEModel",
]

# Lazy re-exports to avoid heavy import at package level
_LAZY_EXPORTS = {
    "MoETrainingAdapter": (
        "florence_forge.training.moe.moe_adapter",
        "MoETrainingAdapter",
    ),
    "MoEConfig": ("florence_forge.training.moe.moe_config", "MoEConfig"),
    "MoELayer": ("florence_forge.training.moe.moe_layer", "MoELayer"),
    "SparseGate": ("florence_forge.training.moe.sparse_gate", "SparseGate"),
    "MoEValidator": ("florence_forge.training.moe.moe_validator", "MoEValidator"),
    "ExpertParallelMoE": (
        "florence_forge.training.moe.expert_parallel",
        "ExpertParallelMoE",
    ),
    "ExpertParallelMoELayer": (
        "florence_forge.training.moe.expert_parallel",
        "ExpertParallelMoELayer",
    ),
    "SelectiveSSMMixer": (
        "florence_forge.training.moe.selective_ssm_mixer",
        "SelectiveSSMMixer",
    ),
    "MoEEncoder": ("florence_forge.training.moe.moe_encoder", "MoEEncoder"),
    "MoEDecoder": ("florence_forge.training.moe.moe_decoder", "MoEDecoder"),
    "MoEModel": ("florence_forge.training.moe.moe_model", "MoEModel"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
