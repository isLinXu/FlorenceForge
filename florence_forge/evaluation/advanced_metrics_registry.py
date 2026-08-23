"""Advanced metric calculator availability registry.

Replaces silent ``ImportError`` degradation with explicit availability checks
and structured warnings so benchmark/eval pipelines never report fake scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancedMetricSpec:
    """Descriptor for one advanced metric calculator."""

    name: str
    module_path: str
    class_name: str
    required_packages: tuple[str, ...] = ()
    description: str = ""


@dataclass
class AdvancedMetricAvailability:
    """Availability status for a single advanced metric family."""

    name: str
    available: bool
    reason: str = ""
    calculator_cls: Optional[Type[Any]] = None


ADVANCED_METRIC_SPECS: tuple[AdvancedMetricSpec, ...] = (
    AdvancedMetricSpec(
        name="semantic",
        module_path="florence_forge.evaluation.advanced_metrics.semantic_metrics_calculator",
        class_name="SemanticMetricsCalculator",
        required_packages=("transformers",),
        description="BERTScore / sentence similarity / CLIP semantic metrics",
    ),
    AdvancedMetricSpec(
        name="efficiency",
        module_path="florence_forge.evaluation.advanced_metrics.efficiency_metrics_calculator",
        class_name="EfficiencyMetricsCalculator",
        required_packages=("psutil",),
        description="Inference latency, memory, throughput efficiency metrics",
    ),
    AdvancedMetricSpec(
        name="robustness",
        module_path="florence_forge.evaluation.advanced_metrics.robustness_metrics_calculator",
        class_name="RobustnessMetricsCalculator",
        required_packages=("torch",),
        description="Adversarial / noise / transform robustness metrics",
    ),
    AdvancedMetricSpec(
        name="multimodal",
        module_path="florence_forge.evaluation.advanced_metrics.multimodal_metrics_calculator",
        class_name="MultiModalMetricsCalculator",
        required_packages=("torch",),
        description="Cross-modal image-text matching metrics",
    ),
)


def _import_check(package: str) -> Optional[str]:
    """Return error message if *package* cannot be imported."""
    try:
        import_module(package)
        return None
    except ImportError as exc:
        return str(exc)


def probe_advanced_metric(spec: AdvancedMetricSpec) -> AdvancedMetricAvailability:
    """Check whether a single advanced metric calculator can be loaded."""
    missing: List[str] = []
    for pkg in spec.required_packages:
        err = _import_check(pkg)
        if err:
            missing.append(f"{pkg} ({err})")

    if missing:
        reason = f"missing dependencies: {', '.join(missing)}"
        logger.warning("Advanced metric '%s' unavailable — %s", spec.name, reason)
        return AdvancedMetricAvailability(spec.name, False, reason)

    try:
        module = import_module(spec.module_path)
        calculator_cls = getattr(module, spec.class_name)
    except (ImportError, AttributeError) as exc:
        reason = f"import failed: {exc}"
        logger.warning("Advanced metric '%s' unavailable — %s", spec.name, reason)
        return AdvancedMetricAvailability(spec.name, False, reason)

    return AdvancedMetricAvailability(spec.name, True, calculator_cls=calculator_cls)


def probe_all_advanced_metrics() -> Dict[str, AdvancedMetricAvailability]:
    """Probe every registered advanced metric calculator."""
    return {spec.name: probe_advanced_metric(spec) for spec in ADVANCED_METRIC_SPECS}


def get_available_advanced_calculators() -> Dict[str, Type[Any]]:
    """Return only calculators that loaded successfully."""
    available: Dict[str, Type[Any]] = {}
    for name, status in probe_all_advanced_metrics().items():
        if status.available and status.calculator_cls is not None:
            available[name] = status.calculator_cls
    return available


def unavailable_advanced_metrics() -> List[str]:
    """Names of advanced metrics that failed availability checks."""
    return [
        name
        for name, status in probe_all_advanced_metrics().items()
        if not status.available
    ]


def advanced_metrics_status_report() -> Dict[str, Any]:
    """Structured report suitable for CLI doctor / benchmark logs."""
    statuses = probe_all_advanced_metrics()
    return {
        "available": [n for n, s in statuses.items() if s.available],
        "unavailable": {
            n: s.reason for n, s in statuses.items() if not s.available
        },
        "total": len(statuses),
    }
