#!/usr/bin/env python3
"""FlorenceForge testing and validation script package."""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "FlorenceForge Team"

__all__ = [
    "TestRunner",
    "ValidationSuite",
    "BenchmarkTools",
    "ExampleRunner",
]

_LAZY_EXPORTS = {
    "TestRunner": ("scripts.testing.quick_test", "QuickTester"),
    "ValidationSuite": ("scripts.testing.validation_suite", "ValidationSuite"),
    "BenchmarkTools": ("scripts.performance.benchmark_tools", "BenchmarkTools"),
    "ExampleRunner": ("scripts.examples.example_runner", "ExampleRunner"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
