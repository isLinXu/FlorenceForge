"""Florence Forge CLI 模块导出入口。"""

from importlib import import_module

__all__ = [
    "cli_main",
    "ConfigManager",
]

_LAZY_EXPORTS = {
    "cli_main": ("florence_forge.cli.main", "main"),
    "ConfigManager": ("florence_forge.cli.config_manager", "ConfigManager"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
