"""测试 Florence2Collator

验证动态 padding、批次组装、空批次处理等核心逻辑。
"""

import pytest
import torch

from florence_forge.data.collate import Florence2Collator, collate_fn


class TestFlorence2Collator:
    def test_pad_sequence_right(self):
        collator = Florence2Collator(pad_token_id=0, padding_side="right")
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded = collator._pad_sequence(sequences)
        assert padded.shape == (2, 3)
        assert padded[0].tolist() == [1, 2, 3]
        assert padded[1].tolist() == [4, 5, 0]

    def test_pad_sequence_left(self):
        collator = Florence2Collator(pad_token_id=0, padding_side="left")
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded = collator._pad_sequence(sequences)
        assert padded.shape == (2, 3)
        assert padded[0].tolist() == [1, 2, 3]
        assert padded[1].tolist() == [0, 4, 5]

    def test_pad_sequence_labels(self):
        collator = Florence2Collator(pad_token_id=-100, padding_side="right")
        sequences = [torch.tensor([1, 2]), torch.tensor([3])]
        padded = collator._pad_sequence(sequences, pad_value=-100)
        assert padded[1].tolist() == [3, -100]

    def test_pad_sequence_preserves_tensor_device(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS is not available")

        collator = Florence2Collator(pad_token_id=0, padding_side="right")
        sequences = [
            torch.tensor([1, 2, 3], device="mps"),
            torch.tensor([4, 5], device="mps"),
        ]

        padded = collator._pad_sequence(sequences)

        assert padded.device.type == "mps"
        assert padded.cpu().tolist() == [[1, 2, 3], [4, 5, 0]]

    def test_collate_batch_basic(self):
        collator = Florence2Collator()
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.tensor([1, 2, 3]),
                "task_type": "CAPTION",
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.tensor([4, 5]),
                "task_type": "CAPTION",
            },
        ]
        result = collator(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["pixel_values"].shape == (2, 3, 224, 224)
        assert result["labels"].shape == (2, 3)
        assert result["task_type"] == "CAPTION"
        assert result["is_empty"] is False

    def test_collate_mixed_tasks(self):
        collator = Florence2Collator()
        batch = [
            {"input_ids": torch.tensor([1]), "pixel_values": torch.randn(3, 224, 224), "task_type": "OD"},
            {"input_ids": torch.tensor([2]), "pixel_values": torch.randn(3, 224, 224), "task_type": "CAPTION"},
        ]
        result = collator(batch)
        assert result["task_types"] == ["OD", "CAPTION"]
        assert result["task_type"] == "OD"

    def test_collate_skips_missing_optional_attention_mask(self):
        collator = Florence2Collator()
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": None,
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.tensor([1, 2]),
                "task_type": "CAPTION",
            },
            {
                "input_ids": torch.tensor([3]),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.tensor([3]),
                "task_type": "CAPTION",
            },
        ]

        result = collator(batch)

        assert "attention_mask" not in result
        assert result["input_ids"].shape == (2, 2)
        assert result["labels"].shape == (2, 2)

    def test_collate_rejects_unencoded_samples(self):
        collator = Florence2Collator()
        batch = [
            {
                "image_path": "/tmp/image.jpg",
                "prompt": "<CAPTION>",
                "answer": "caption",
                "task_type": "CAPTION",
                "_needs_encoding": True,
            }
        ]

        with pytest.raises(RuntimeError, match="未编码样本"):
            collator(batch)

    def test_collate_rejects_partial_encoded_batch(self):
        collator = Florence2Collator()
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "pixel_values": torch.randn(3, 224, 224),
                "task_type": "CAPTION",
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "task_type": "CAPTION",
            },
        ]

        with pytest.raises(ValueError, match="pixel_values"):
            collator(batch)

    def test_collate_empty_batch(self):
        with pytest.raises(ValueError, match="Cannot collate an empty batch"):
            Florence2Collator()([])

    def test_collate_fn_entry(self):
        """测试向后兼容的 collate_fn 入口"""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "pixel_values": torch.randn(3, 224, 224),
                "task_type": "CAPTION",
            }
        ]
        result = collate_fn(batch)
        assert result["is_empty"] is False
        assert result["input_ids"].shape == (1, 2)

    def test_collate_fn_empty(self):
        result = collate_fn([])
        assert result["is_empty"] is True
