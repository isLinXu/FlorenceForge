"""测试配置模块

覆盖 ModelConfig, DataConfig, OptimizationConfig, TrainingConfig 的创建、
序列化、反序列化和验证逻辑。
"""

import pytest
import json
import tempfile
from pathlib import Path
import importlib
import ast
import sys

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover - py310 and below
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback for existing repo deps
        import toml as tomllib  # type: ignore

from florence_forge.core.config import (
    ModelConfig,
    DataConfig,
    OptimizationConfig,
    TaskSchedulingConfig,
    TrainingConfig,
    EvaluationConfig,
    LoRAConfig,
)


def _load_pyproject() -> dict:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


class TestLoRAConfig:
    def test_default_values(self):
        cfg = LoRAConfig()
        assert cfg.r == 32
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.bias == "none"

    def test_to_dict(self):
        cfg = LoRAConfig(r=16, lora_alpha=32)
        d = cfg.to_dict()
        assert d["r"] == 16
        assert d["lora_alpha"] == 32


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.model_name == "microsoft/Florence-2-large"
        assert cfg.use_lora is True
        assert cfg.trust_remote_code is True
        assert cfg.attn_implementation == "sdpa"

    def test_to_dict(self):
        cfg = ModelConfig(model_name="test-model", use_lora=False)
        d = cfg.to_dict()
        assert d["model_name"] == "test-model"
        assert d["use_lora"] is False


class TestDataConfig:
    def test_default_values(self):
        cfg = DataConfig()
        assert cfg.batch_size == 4
        assert cfg.num_workers == 4
        assert cfg.use_balanced_sampling is True

    def test_from_dict(self):
        cfg = DataConfig.from_dict({"batch_size": 8, "num_workers": 2})
        assert cfg.batch_size == 8
        assert cfg.num_workers == 2

    def test_unknown_field_warns_instead_of_silent_ignore(self):
        with pytest.warns(UserWarning, match="未知配置字段.*batc_size"):
            cfg = DataConfig.from_dict({"batc_size": 8})

        assert cfg.batch_size == 4


class TestOptimizationConfig:
    def test_default_scheduler(self):
        cfg = OptimizationConfig()
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.warmup_ratio == 0.1


class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.num_epochs == 10
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.early_stopping_patience == 5
        assert cfg.logging_dir == "./outputs/logs"

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig(num_epochs=5)
        cfg.data_settings.batch_size = 16
        d = cfg.to_dict()
        assert d["num_epochs"] == 5
        # to_dict 使用 alias，所以键名是 data_config
        assert d["data_config"]["batch_size"] == 16
        assert "model_config" in d
        assert "data_config" in d

    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(num_epochs=3, output_dir=tmpdir)
            path = Path(tmpdir) / "config.json"
            cfg.save_to_json(path)
            assert path.exists()
            loaded = TrainingConfig.load_from_json(path)
            assert loaded.num_epochs == 3

    def test_save_load_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(num_epochs=3, output_dir=tmpdir)
            path = Path(tmpdir) / "config.yaml"
            cfg.save_to_yaml(path)
            assert path.exists()
            loaded = TrainingConfig.load_from_yaml(path)
            assert loaded.num_epochs == 3

    def test_from_dict_nested(self):
        d = {
            "num_epochs": 5,
            "model_config": {"model_name": "custom-model", "use_lora": False},
            "data_config": {"batch_size": 8},
            "optimization_config": {"learning_rate": 2e-5},
            "task_scheduling_config": {"strategy": "weighted"},
        }
        cfg = TrainingConfig.from_dict(d)
        assert cfg.num_epochs == 5
        # Pydantic v2: 字段名用 Python 属性名（xxx_settings），而非 alias
        assert cfg.model_settings.model_name == "custom-model"
        assert cfg.data_settings.batch_size == 8
        assert cfg.optimization_settings.learning_rate == 2e-5
        assert cfg.task_scheduling_settings.strategy == "weighted"

    def test_unknown_top_level_field_warns_but_is_preserved_for_compatibility(self):
        with pytest.warns(UserWarning, match="未知配置字段.*legacy_flag"):
            cfg = TrainingConfig.from_dict({"legacy_flag": True})

        assert cfg.legacy_flag is True

    def test_max_steps_with_epochs_warns_priority(self):
        """同时设定 max_steps 与 num_epochs 时应告警 max_steps 优先。"""
        with pytest.warns(UserWarning, match="max_steps.*优先生效"):
            cfg = TrainingConfig(num_epochs=10, max_steps=50)

        assert cfg.max_steps == 50
        assert cfg.num_epochs == 10

    def test_max_steps_alone_does_not_warn_when_epochs_is_one(self):
        """num_epochs=1（单轮）+ max_steps 不触发优先级告警。"""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            cfg = TrainingConfig(num_epochs=1, max_steps=50)

        assert cfg.max_steps == 50


class TestEvaluationConfig:
    def test_default_metrics(self):
        cfg = EvaluationConfig()
        assert "accuracy" in cfg.metrics

    def test_to_dict(self):
        cfg = EvaluationConfig(batch_size=16)
        d = cfg.to_dict()
        assert d["batch_size"] == 16


class TestPackagingMetadata:
    def test_package_import_is_lightweight(self):
        package = importlib.import_module("florence_forge")
        assert hasattr(package, "TrainingConfig")
        assert package.__version__ == "1.0.0"

    def test_subpackage_imports_are_lightweight(self):
        for module_name in [
            "florence_forge.core",
            "florence_forge.utils",
            "florence_forge.cli",
            "florence_forge.training",
            "florence_forge.evaluation",
        ]:
            module = importlib.import_module(module_name)
            assert module is not None

    def test_optional_heavy_modules_import_cleanly(self):
        for module_name in [
            "florence_forge.utils.visualization",
            "florence_forge.evaluation.benchmark",
            "florence_forge.evaluation.analyzer",
            "florence_forge.evaluation.advanced_metrics",
            "florence_forge.deployment.exporter",
        ]:
            module = importlib.import_module(module_name)
            assert module is not None

    def test_advanced_metrics_package_is_lazy(self):
        module_name = "florence_forge.evaluation.advanced_metrics"
        child_module = f"{module_name}.semantic_metrics_calculator"

        sys.modules.pop(module_name, None)
        sys.modules.pop(child_module, None)

        module = importlib.import_module(module_name)

        assert module is not None
        assert child_module not in sys.modules

        exported = module.SemanticMetricsCalculator
        assert exported.__name__ == "SemanticMetricsCalculator"
        assert child_module in sys.modules

    def test_package_version_matches_init(self):
        data = _load_pyproject()
        init_text = (Path(__file__).resolve().parents[1] / "florence_forge" / "__init__.py").read_text(encoding="utf-8")
        version_line = next(line for line in init_text.splitlines() if line.startswith("__version__"))
        init_version = version_line.split("=", 1)[1].strip().strip('"').strip("'")

        assert data["project"]["version"] == init_version

    def test_evaluation_extra_declared(self):
        data = _load_pyproject()

        evaluation_extra = data["project"]["optional-dependencies"]["evaluation"]
        assert any(dep.startswith("nltk") for dep in evaluation_extra)
        assert any(dep.startswith("rouge-score") for dep in evaluation_extra)
        assert any(dep.startswith("pycocotools") for dep in evaluation_extra)

    def test_console_script_targets_exist(self):
        data = _load_pyproject()

        scripts = data["project"]["scripts"]
        repo_root = Path(__file__).resolve().parents[1]
        for _, target in scripts.items():
            module_name, func_name = target.split(":")
            module_path = repo_root / (module_name.replace(".", "/") + ".py")
            assert module_path.exists()
            module_ast = ast.parse(module_path.read_text(encoding="utf-8"))
            exported_funcs = {
                node.name for node in module_ast.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert func_name in exported_funcs

    def test_setup_py_is_metadata_shim(self):
        setup_py = (Path(__file__).resolve().parents[1] / "setup.py").read_text(encoding="utf-8")
        assert "florence-train" not in setup_py
        assert "florence-evaluate" not in setup_py
        assert "setup()" in setup_py
