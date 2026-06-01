"""Deployment backend abstraction tests."""

import pytest
import torch

from florence_forge.deployment.backends import (
    InferenceBackend,
    NativeInferenceBackend,
    VLLMInferenceBackend,
)


class DummyEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.use_amp = False
        self.compile_model = False

    def predict(self, inputs, return_raw=False, **kwargs):
        return {"inputs": inputs, "return_raw": return_raw, "kwargs": kwargs}

    def predict_batch(self, inputs_list, batch_size=None, **kwargs):
        return [
            {"inputs": inputs, "batch_size": batch_size, "kwargs": kwargs}
            for inputs in inputs_list
        ]

    def benchmark(self, input_shape, num_runs=100, **kwargs):
        return {"num_runs": num_runs, "rank": len(input_shape)}

    def get_stats(self):
        return {"total_inferences": 3}


class DummyBackend(InferenceBackend):
    name = "dummy"

    def predict(self, inputs, return_raw=False, **kwargs):
        return inputs

    def predict_batch(self, inputs_list, batch_size=None, **kwargs):
        return list(inputs_list)


def test_native_backend_adapts_existing_inference_engine():
    backend = NativeInferenceBackend(DummyEngine())

    assert backend.predict("image", return_raw=True)["return_raw"] is True
    assert backend.predict_batch(["a", "b"], batch_size=4)[0]["batch_size"] == 4
    assert backend.benchmark((1, 3, 8, 8), num_runs=5) == {"num_runs": 5, "rank": 4}

    model_info = backend.get_model_info()
    assert model_info["backend"] == "native"
    assert model_info["device"] == "cpu"
    assert model_info["batch_size"] == 2
    assert model_info["engine_stats"] == {"total_inferences": 3}


def test_model_server_wraps_legacy_engine():
    pytest.importorskip("fastapi")

    from florence_forge.deployment.server import ModelServer

    server = ModelServer(DummyEngine())

    assert isinstance(server.inference_backend, NativeInferenceBackend)
    assert server._get_model_info()["backend"] == "native"


def test_model_server_accepts_backend_directly():
    pytest.importorskip("fastapi")

    from florence_forge.deployment.server import ModelServer

    backend = DummyBackend()
    server = ModelServer(backend)

    assert server.inference_backend is backend
    assert server.inference_engine is backend
    assert server._get_model_info() == {"backend": "dummy"}


def test_vllm_backend_fails_with_actionable_error():
    with pytest.raises((ImportError, NotImplementedError)):
        VLLMInferenceBackend("dummy-model")


def test_create_server_rejects_unknown_backend_before_model_load():
    from florence_forge.deployment.server import create_server

    with pytest.raises(ValueError, match="不支持的推理后端"):
        create_server("dummy-model", backend="unknown")
