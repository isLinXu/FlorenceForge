#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference backend abstractions for deployment services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.optional_dependencies import missing_dependency_message


class InferenceBackend(ABC):
    """Common serving interface used by :class:`ModelServer`."""

    name = "base"

    @abstractmethod
    def predict(self, inputs: Any, return_raw: bool = False, **kwargs: Any) -> Any:
        """Run a single prediction."""

    @abstractmethod
    def predict_batch(
        self,
        inputs_list: List[Any],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Run a batch prediction."""

    def benchmark(
        self,
        input_shape: tuple,
        num_runs: int = 100,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run a backend benchmark if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support benchmark()")

    def get_stats(self) -> Dict[str, Any]:
        """Return backend runtime statistics."""
        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Return serializable model/backend metadata."""
        return {"backend": self.name}


class NativeInferenceBackend(InferenceBackend):
    """Adapter for the existing PyTorch :class:`InferenceEngine`."""

    name = "native"

    def __init__(self, engine: Any):
        self.engine = engine

    def predict(self, inputs: Any, return_raw: bool = False, **kwargs: Any) -> Any:
        return self.engine.predict(inputs, return_raw=return_raw, **kwargs)

    def predict_batch(
        self,
        inputs_list: List[Any],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return self.engine.predict_batch(
            inputs_list,
            batch_size=batch_size,
            **kwargs,
        )

    def benchmark(
        self,
        input_shape: tuple,
        num_runs: int = 100,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return self.engine.benchmark(input_shape, num_runs=num_runs, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        if hasattr(self.engine, "get_stats"):
            return self.engine.get_stats()
        return {}

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "backend": self.name,
            "engine": type(self.engine).__name__,
            "engine_stats": self.get_stats(),
        }
        for attr in ("device", "batch_size", "use_amp", "compile_model"):
            if hasattr(self.engine, attr):
                value = getattr(self.engine, attr)
                info[attr] = str(value) if attr == "device" else value
        return info


class VLLMInferenceBackend(InferenceBackend):
    """Placeholder for future vLLM serving support.

    vLLM support for multimodal Florence-style inputs needs a dedicated request
    adapter. This class exists so server wiring can depend on a stable backend
    contract while failing loudly if the optional backend is selected too early.
    """

    name = "vllm"

    def __init__(self, model: Union[str, Path], **kwargs: Any):
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                missing_dependency_message("vLLM推理后端", "vllm")
            ) from exc

        raise NotImplementedError(
            "VLLMInferenceBackend is not yet implemented for FlorenceForge "
            "multimodal requests. Use NativeInferenceBackend for serving."
        )

    def predict(self, inputs: Any, return_raw: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("VLLMInferenceBackend.predict is not implemented")

    def predict_batch(
        self,
        inputs_list: List[Any],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]:
        raise NotImplementedError("VLLMInferenceBackend.predict_batch is not implemented")
