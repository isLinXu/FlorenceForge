"""稀疏门控 MoE / SelectiveSSM 实验性组件。

⚠️ **稳定性等级：实验性 (Experimental, Tier-3)**

本目录原先存放在 `florence_forge/core/backends/`，于 2026-05-21 迁移到本实验性
模块。当前实现已修复早期草稿中的递归实例化、随机非参数张量和 einsum shape
错误，但仍然只是最小可运行实验模块，不属于主训练通路。

如果未来要真正生产化 MoE：
- 替换当前 dense-all-experts 计算为真正稀疏专家调度；
- 补充端到端训练/推理示例；
- 对路由负载均衡、溢出处理和专家并行做系统测试；
- 上述全部完成后才考虑迁回 `core/backends/`。

历史审查见 `WARNING.md`。
"""

import warnings


class ExperimentalMoEWarning(UserWarning):
    """Warning emitted when importing experimental MoE modules."""


warnings.warn(
    "florence_forge.experimental.moe is experimental and not part of the production training path.",
    ExperimentalMoEWarning,
    stacklevel=2,
)

# 不在此处主动 re-export 子模块，避免误用。
# 如确需使用：from florence_forge.experimental.moe.moe_layer import MoELayer

__all__ = ["MoETrainingAdapter", "MoEConfig", "ExperimentalMoEWarning"]

# 懒加载 re-export，避免包级导入时触发重依赖拉起
_LAZY_EXPORTS = {
    "MoETrainingAdapter": ("florence_forge.experimental.moe.moe_adapter", "MoETrainingAdapter"),
    "MoEConfig": ("florence_forge.experimental.moe.moe_config", "MoEConfig"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
