"""Regression tests for safe torch artifact loading."""

import pytest

from florence_forge.utils.torch_serialization import safe_torch_load_cpu


def test_safe_torch_load_cpu_fails_closed_when_weights_only_is_unsupported(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("weights_only is not supported")
        raise AssertionError("unsafe torch.load fallback should not be called")

    monkeypatch.setattr("florence_forge.utils.torch_serialization.torch.load", fake_load)

    with pytest.raises(RuntimeError, match="weights_only=True"):
        safe_torch_load_cpu(tmp_path / "checkpoint.pt")

    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"
