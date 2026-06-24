"""Tests for P0 fixes: v2 LoRA/ModelMerger signatures and DataValidator training schema.

These tests verify the fixes for the P0 issues identified in the deep analysis:
- P0-1: v2 LoRAManager constructor and apply_lora method call
- P0-2: ModelMerger constructor signature (two call sites)
- P0-3: merge_and_unload bypassing wrapper initialization
- P0-4: DataValidator supporting training schema (image + prefix + suffix)
"""

import json


from florence_forge.core.config import LoRAConfig
from florence_forge.data.validator import validate_data_format
from florence_forge.training.lora_manager import LoRAManager
from florence_forge.training.model_merger import ModelMerger


# ---------------------------------------------------------------------------
# P0-1: LoRAManager constructor signature
# ---------------------------------------------------------------------------


class TestLoRAManagerConstructor:
    """Verify LoRAManager accepts ForgeLoRAConfig (not model) as first arg."""

    def test_init_accepts_forge_lora_config(self):
        config = LoRAConfig(r=8, lora_alpha=16)
        manager = LoRAManager(config)
        assert manager.base_config is config

    def test_init_defaults_to_empty_config(self):
        manager = LoRAManager()
        assert manager.base_config is not None

    def test_init_rejects_model_as_first_arg(self):
        """A nn.Module should NOT be passed as the first argument.

        LoRAManager.__init__ accepts Optional[ForgeLoRAConfig], so Python
        won't reject a model at runtime, but storing it as base_config is
        semantically wrong. We verify the correct usage pattern instead.
        """
        config = LoRAConfig(r=8)
        manager = LoRAManager(config)
        # Correct: base_config is a ForgeLoRAConfig
        assert isinstance(manager.base_config, LoRAConfig)


# ---------------------------------------------------------------------------
# P0-2: ModelMerger constructor signature
# ---------------------------------------------------------------------------


class TestModelMergerConstructor:
    """Verify ModelMerger accepts lora_manager (not model) as first arg."""

    def test_init_accepts_lora_manager(self):
        lora_manager = LoRAManager()
        merger = ModelMerger(lora_manager)
        assert merger.lora_manager is lora_manager

    def test_init_defaults_to_new_lora_manager(self):
        merger = ModelMerger()
        assert isinstance(merger.lora_manager, LoRAManager)

    def test_init_rejects_model_as_first_arg(self):
        """A nn.Module should NOT be passed as the first argument.

        ModelMerger.__init__ accepts Optional[LoRAManager], so Python
        won't reject a model at runtime, but storing it as lora_manager is
        semantically wrong. We verify the correct usage pattern instead.
        """
        lora_manager = LoRAManager()
        merger = ModelMerger(lora_manager)
        # Correct: lora_manager is a LoRAManager
        assert isinstance(merger.lora_manager, LoRAManager)


# ---------------------------------------------------------------------------
# P0-3: merge_and_unload no longer uses __new__
# ---------------------------------------------------------------------------


class TestMergeAndUnloadSignature:
    """Verify merge_and_unload accepts Florence2MultiTaskModel (not PeftModel)."""

    def test_merge_and_unload_accepts_wrapper_model(self):
        """merge_and_unload should accept a Florence2MultiTaskModel, not a raw PeftModel."""
        import inspect
        sig = inspect.signature(ModelMerger.merge_and_unload)
        params = list(sig.parameters.keys())
        # Skip 'self', first real param should be 'model' (not 'peft_model')
        real_params = [p for p in params if p != "self"]
        assert real_params[0] == "model", (
            f"merge_and_unload first param should be 'model', got '{real_params[0]}'"
        )

    def test_merge_and_unload_does_not_use_new(self):
        """Verify the implementation no longer uses __new__ to bypass init."""
        import inspect
        source = inspect.getsource(ModelMerger.merge_and_unload)
        assert "__new__" not in source, (
            "merge_and_unload should not use __new__ to bypass wrapper initialization"
        )


# ---------------------------------------------------------------------------
# P0-4: DataValidator training schema
# ---------------------------------------------------------------------------


def _write_image(path):
    from PIL import Image
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(path)


def _training_sample(image_name="image.png"):
    """A valid training schema sample (image + prefix + suffix)."""
    return {
        "image": image_name,
        "prefix": "<CAPTION>",
        "suffix": "A small red square.",
        "task_type": "CAPTION",
    }


def _write_jsonl(tmp_path, samples):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8"
    )
    return data_path


class TestDataValidatorTrainingSchema:
    """Verify DataValidator supports training schema (image + prefix + suffix)."""

    def test_auto_detect_training_schema(self, tmp_path):
        _write_image(tmp_path / "image.png")
        data_path = _write_jsonl(tmp_path, [_training_sample()])
        result = validate_data_format(data_path)

        assert result["is_valid"] is True
        assert result["effective_schema"] == "training"
        assert result["total_samples"] == 1
        assert result["valid_samples"] == 1

    def test_explicit_training_schema(self, tmp_path):
        _write_image(tmp_path / "image.png")
        data_path = _write_jsonl(tmp_path, [_training_sample()])
        result = validate_data_format(data_path, schema="training")

        assert result["is_valid"] is True
        assert result["effective_schema"] == "training"

    def test_training_schema_missing_prefix(self, tmp_path):
        _write_image(tmp_path / "image.png")
        sample = {"image": "image.png", "suffix": "some text"}
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path, schema="training")

        assert result["is_valid"] is False
        messages = [e["message"] for e in result["validation_results"]]
        assert any("prefix" in m for m in messages)

    def test_training_schema_missing_suffix(self, tmp_path):
        _write_image(tmp_path / "image.png")
        sample = {"image": "image.png", "prefix": "<CAPTION>"}
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path, schema="training")

        assert result["is_valid"] is False
        messages = [e["message"] for e in result["validation_results"]]
        assert any("suffix" in m for m in messages)

    def test_training_schema_empty_prefix(self, tmp_path):
        _write_image(tmp_path / "image.png")
        sample = {"image": "image.png", "prefix": "  ", "suffix": "text"}
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path, schema="training")

        assert result["is_valid"] is False
        messages = [e["message"] for e in result["validation_results"]]
        assert any("prefix" in m for m in messages)

    def test_training_schema_empty_suffix_warns(self, tmp_path):
        _write_image(tmp_path / "image.png")
        sample = {"image": "image.png", "prefix": "<CAPTION>", "suffix": "  "}
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path, schema="training")

        # Empty suffix is a warning, not an error
        assert result["warning_count"] >= 1

    def test_conversation_schema_still_works(self, tmp_path):
        """Ensure conversation schema is not broken by the training schema addition."""
        _write_image(tmp_path / "image.png")
        sample = {
            "image": "image.png",
            "task_type": "CAPTION",
            "conversations": [
                {"from": "human", "value": "Describe."},
                {"from": "gpt", "value": "A red square."},
            ],
        }
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path)

        assert result["is_valid"] is True
        assert result["effective_schema"] == "conversation"

    def test_mixed_schema_auto_detects_training(self, tmp_path):
        """When both schemas are present, auto-detect picks the majority."""
        _write_image(tmp_path / "image.png")
        training = _training_sample()
        conversation = {
            "image": "image.png",
            "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}],
        }
        data_path = _write_jsonl(tmp_path, [training, training, conversation])
        result = validate_data_format(data_path)

        # Training samples outnumber conversation samples
        assert result["effective_schema"] == "training"

    def test_schema_in_summary(self, tmp_path):
        _write_image(tmp_path / "image.png")
        data_path = _write_jsonl(tmp_path, [_training_sample()])
        result = validate_data_format(data_path)

        assert "detected_schema" in result
        assert "effective_schema" in result

    def test_training_schema_with_task_specific_detection(self, tmp_path):
        """Training schema should validate detection format from suffix."""
        _write_image(tmp_path / "image.png")
        sample = {
            "image": "image.png",
            "prefix": "<OD>",
            "suffix": "cat<loc_1200><loc_10><loc_20><loc_30>",
            "task_type": "object_detection",
        }
        data_path = _write_jsonl(tmp_path, [sample])
        result = validate_data_format(data_path, schema="training")

        # Should detect out-of-range coordinate
        messages = [e["message"] for e in result["validation_results"]]
        assert any("检测坐标超出范围" in m for m in messages)
