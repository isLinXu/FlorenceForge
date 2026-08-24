#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlorenceForge exception hierarchy.

The framework still raises standard Python exceptions in many leaf modules for
backward compatibility. These base classes provide a stable target for new code
and for callers that want to handle FlorenceForge failures explicitly.
"""


class FlorenceForgeError(Exception):
    """Base class for all FlorenceForge-specific errors."""


class ConfigError(FlorenceForgeError):
    """Raised when configuration validation or loading fails."""


class DataError(FlorenceForgeError):
    """Raised when dataset parsing, loading, or collation fails."""


class TrainingError(FlorenceForgeError):
    """Raised for training loop failures that belong to FlorenceForge."""


class BackendError(FlorenceForgeError):
    """Raised when a model backend cannot load, encode, or infer correctly."""


class DeploymentError(FlorenceForgeError):
    """Raised by deployment and serving components."""


class EvaluationError(FlorenceForgeError):
    """Raised when evaluation or benchmarking fails."""


class AgenticError(FlorenceForgeError):
    """Raised by the Agentic orchestrator (planning, tool execution, timeouts)."""


class AgenticTimeoutError(AgenticError):
    """Raised when an Agentic run exceeds its configured time budget."""


class MoEError(FlorenceForgeError):
    """Raised by Mixture-of-Experts components (routing, injection, capacity)."""


class QuantizationError(FlorenceForgeError):
    """Raised when model quantization fails or is misconfigured."""


class SecurityWarning(UserWarning):
    """Warning for compatibility paths that weaken default security posture."""
