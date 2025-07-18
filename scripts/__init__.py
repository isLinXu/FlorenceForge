#!/usr/bin/env python3
"""
FlorenceForge测试和验证脚本包

提供完整的测试、验证和示例脚本，包括：
- 单元测试
- 集成测试
- 性能测试
- 示例脚本
- 验证工具
"""

__version__ = "0.1.0"
__author__ = "FlorenceForge Team"

# 导入主要的测试工具
from .test_runner import TestRunner
from .validation_suite import ValidationSuite
from .benchmark_tools import BenchmarkTools
from .example_runner import ExampleRunner

__all__ = [
    'TestRunner',
    'ValidationSuite', 
    'BenchmarkTools',
    'ExampleRunner'
]