"""Backward compatibility shim for experimental MoE.

All MoE components have been promoted to ``florence_forge.training.moe``.
This module re-exports from the production location for backward compatibility.

⚠️ **Deprecation Notice**: Importing from ``florence_forge.experimental.moe`` is
deprecated. Please migrate to ``florence_forge.training.moe``.
"""

from __future__ import annotations

import warnings

class ExperimentalMoEWarning(UserWarning):
    """Warning emitted when importing deprecated experimental MoE modules."""


warnings.warn(
    "florence_forge.experimental.moe is deprecated. "
    "Please use florence_forge.training.moe instead.",
    ExperimentalMoEWarning,
    stacklevel=2,
)

# Re-export all public symbols from the production module
from florence_forge.training.moe import (  # noqa: F401
    MoETrainingAdapter,
    MoEConfig,
    MoELayer,
    SparseGate,
    MoEValidator,
    SelectiveSSMMixer,
    MoEEncoder,
    MoEDecoder,
    MoEModel,
)

__all__ = [
    "MoETrainingAdapter",
    "MoEConfig",
    "MoELayer",
    "SparseGate",
    "MoEValidator",
    "SelectiveSSMMixer",
    "MoEEncoder",
    "MoEDecoder",
    "MoEModel",
    "ExperimentalMoEWarning",
]
