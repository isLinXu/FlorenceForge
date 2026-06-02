"""Optional dependency messaging helpers.

Centralizes user-facing install hints so optional feature degradation is
consistent across modules.
"""

from __future__ import annotations

from typing import Optional


def format_install_hint(
    package_spec: str,
    extra_name: Optional[str] = None,
) -> str:
    """Return a consistent install hint for an optional dependency."""
    if extra_name:
        return f"请安装 `{package_spec}` 或执行 `pip install -e \".[{extra_name}]\"`"
    return f"请安装 `{package_spec}`"


def missing_dependency_message(
    feature: str,
    package_spec: str,
    extra_name: Optional[str] = None,
) -> str:
    """Build a consistent missing-dependency message."""
    return f"{feature}需要 {package_spec}，{format_install_hint(package_spec, extra_name)}"
