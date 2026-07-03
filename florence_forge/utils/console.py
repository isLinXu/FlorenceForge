"""Unified CLI output — Rich console with logging fallback.

Use ``get_console()`` in CLI modules instead of bare ``print()`` so output
is styled consistently and can be captured by log aggregators.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

_console: Optional[Any] = None
_rich_available: Optional[bool] = None


def _detect_rich() -> bool:
    global _rich_available
    if _rich_available is None:
        try:
            import rich  # noqa: F401
            _rich_available = True
        except ImportError:
            _rich_available = False
    return _rich_available


def get_console(force_plain: bool = False) -> Any:
    """Return a Rich ``Console`` or a lightweight logging-backed proxy."""
    global _console
    if force_plain or not _detect_rich():
        return _LoggingConsole()
    if _console is None:
        from rich.console import Console
        _console = Console(stderr=False, highlight=False)
    return _console


class _LoggingConsole:
    """Drop-in replacement for ``rich.console.Console`` using stdlib logging."""

    def print(self, *objects: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        style = kwargs.get("style")
        message = sep.join(str(o) for o in objects)
        if style and style.startswith("bold"):
            message = message  # keep plain for log pipelines
        stream = kwargs.get("file", sys.stdout)
        if stream is sys.stderr:
            logger.info(message.rstrip("\n"))
        else:
            # CLI user-facing output goes to stdout via print for compatibility
            print(message, end=end, file=stream)

    def rule(self, title: str = "", **kwargs: Any) -> None:
        line = f"{'─' * 40} {title} {'─' * 40}".strip()
        self.print(line)

    def status(self, *args: Any, **kwargs: Any) -> "_NoOpContext":
        return _NoOpContext()


class _NoOpContext:
    def __enter__(self) -> "_NoOpContext":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def cli_print(*objects: Any, **kwargs: Any) -> None:
    """Convenience wrapper: ``get_console().print(...)``."""
    get_console().print(*objects, **kwargs)
