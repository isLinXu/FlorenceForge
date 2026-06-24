"""可视化依赖检测与 Mock 适配层。

将 matplotlib / seaborn / pandas / numpy 的可用性检测和降级 mock 集中管理，
避免在 visualizer.py 中重复冗长的 try/except 块。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 分别检测各个可视化库的可用性
MATLOTLIB_AVAILABLE = False  # 拼写保留以兼容旧代码
MATPLOTLIB_AVAILABLE = False
SEABORN_AVAILABLE = False
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False

# 检测 matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
    MATLOTLIB_AVAILABLE = True
except ImportError as e:
    logger.debug(f"matplotlib导入失败: {e}")
    MATPLOTLIB_AVAILABLE = False
    MATLOTLIB_AVAILABLE = False

    class MockRcParams:
        def __setitem__(self, key, value):
            pass

        def __getitem__(self, key):
            return None

    class _MockPlt:
        rcParams = MockRcParams()

        @staticmethod
        def subplots(*args, **kwargs):
            raise ImportError("matplotlib not available")

        @staticmethod
        def tight_layout():
            pass

        @staticmethod
        def savefig(*args, **kwargs):
            pass

        @staticmethod
        def close():
            pass

        @staticmethod
        def figure(*args, **kwargs):
            raise ImportError("matplotlib not available")

    plt = _MockPlt()  # type: ignore[misc]

# 检测 seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"seaborn导入失败: {e}")
    SEABORN_AVAILABLE = False

    class _MockSns:
        @staticmethod
        def set_style(*args, **kwargs):
            pass

        @staticmethod
        def set_palette(*args, **kwargs):
            pass

    sns = _MockSns()  # type: ignore[misc]

# 检测 pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"pandas导入失败: {e}")
    PANDAS_AVAILABLE = False

    class _MockPd:
        @staticmethod
        def read_csv(*args, **kwargs):
            raise ImportError("pandas not available")

        @staticmethod
        def DataFrame(*args, **kwargs):
            raise ImportError("pandas not available")

        @staticmethod
        def Timestamp(*args, **kwargs):
            raise ImportError("pandas not available")

    pd = _MockPd()  # type: ignore[misc]

# 检测 numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    logger.debug(f"numpy导入失败: {e}")
    NUMPY_AVAILABLE = False

    class _MockNp:
        @staticmethod
        def array(*args, **kwargs):
            raise ImportError("numpy not available")

        @staticmethod
        def linspace(*args, **kwargs):
            raise ImportError("numpy not available")

    np = _MockNp()  # type: ignore[misc]

VISUALIZATION_AVAILABLE = MATPLOTLIB_AVAILABLE and PANDAS_AVAILABLE


def check_visualization_dependencies() -> dict[str, bool]:
    """返回当前可视化依赖状态字典。"""
    return {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "pandas": PANDAS_AVAILABLE,
        "seaborn": SEABORN_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
    }
