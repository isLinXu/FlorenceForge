"""Matplotlib 显示控制（CI 无头环境默认不弹窗）。"""

from __future__ import annotations

import os
from typing import Optional


def should_show_plots(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("FLORENCE_FORGE_SHOW_PLOTS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def finalize_matplotlib_figure(show: Optional[bool] = None) -> None:
    """保存后关闭或显示图形；默认关闭以避免阻塞 CI。"""
    import matplotlib.pyplot as plt

    if should_show_plots(show):
        plt.show()
    else:
        plt.close("all")
