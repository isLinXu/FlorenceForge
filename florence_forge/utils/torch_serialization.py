"""Safe helpers for loading torch artifacts."""

from pathlib import Path
from typing import Any, Optional, Union

import torch


PathLike = Union[str, Path]


def safe_torch_load(
    path: PathLike,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    context: str = "Torch artifact",
) -> Any:
    """Load a torch artifact with ``weights_only=True``.

    FlorenceForge requires PyTorch versions that support ``weights_only``.
    Falling back to the legacy default would re-enable pickle execution for
    checkpoints and cache files, so unsupported runtimes fail closed.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            f"{context} {path} cannot be loaded safely because this PyTorch "
            "runtime does not support torch.load(weights_only=True). "
            "Upgrade PyTorch instead of using unsafe pickle loading."
        ) from exc


def safe_torch_load_cpu(path: PathLike, *, context: str = "Torch artifact") -> Any:
    """Load a torch artifact safely onto CPU."""
    return safe_torch_load(path, map_location="cpu", context=context)
