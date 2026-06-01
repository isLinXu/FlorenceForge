"""Tests for lightweight routing and training configuration facades."""

import pytest

from florence_forge.core.architecture_resolver import ArchitectureResolver
from florence_forge.core.config import TrainingConfig
from florence_forge.training.config import (
    create_default_config,
    load_config_from_file,
    validate_config_file,
)


@pytest.fixture(autouse=True)
def clear_architecture_registry():
    ArchitectureResolver.clear()
    yield
    ArchitectureResolver.clear()


def test_architecture_resolver_supports_registered_and_builder_only_backends():
    class DemoBackend:
        def __init__(self, model_name=None):
            self.model_name = model_name

    ArchitectureResolver.register("demo", DemoBackend)
    direct = ArchitectureResolver.resolve("demo", model_name="direct")
    assert direct.model_name == "direct"

    ArchitectureResolver.register_builder(
        "demo",
        lambda backend_class, **kwargs: backend_class(model_name=f"wrapped-{kwargs['model_name']}"),
    )
    wrapped = ArchitectureResolver.resolve("demo", model_name="value")
    assert wrapped.model_name == "wrapped-value"
    assert ArchitectureResolver.get_builder("demo") is not None

    ArchitectureResolver.register_builder("factory", lambda **kwargs: kwargs["identifier"])
    assert ArchitectureResolver.resolve("factory", identifier="created") == "created"

    with pytest.raises(ValueError, match="未知的后端名称"):
        ArchitectureResolver.resolve("missing")


def test_training_config_facade_loads_valid_files_and_rejects_invalid(tmp_path):
    default = create_default_config()
    config_path = tmp_path / "config.yaml"
    default.num_epochs = 3
    default.save_to_yaml(config_path)

    loaded = load_config_from_file(config_path)

    assert isinstance(loaded, TrainingConfig)
    assert loaded.num_epochs == 3
    assert validate_config_file(config_path) is True
    assert validate_config_file(tmp_path / "missing.yaml") is False
