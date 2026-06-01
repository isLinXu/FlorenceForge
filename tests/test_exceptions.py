"""Framework exception hierarchy tests."""

import florence_forge
from florence_forge.exceptions import (
    BackendError,
    ConfigError,
    DataError,
    DeploymentError,
    FlorenceForgeError,
    SecurityWarning,
    TrainingError,
)


def test_exception_hierarchy_is_exported():
    assert florence_forge.FlorenceForgeError is FlorenceForgeError
    assert florence_forge.SecurityWarning is SecurityWarning

    for exc_type in (
        ConfigError,
        DataError,
        TrainingError,
        BackendError,
        DeploymentError,
    ):
        assert issubclass(exc_type, FlorenceForgeError)
