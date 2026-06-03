"""向后兼容：实现已迁至 ``florence_forge.utils.plot_backend``。"""

from florence_forge.utils.plot_backend import (  # noqa: F401
    finalize_matplotlib_figure,
    should_show_plots,
)

__all__ = ["finalize_matplotlib_figure", "should_show_plots"]
