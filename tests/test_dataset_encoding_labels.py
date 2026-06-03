"""dataset_encoding 标签监督辅助函数测试。"""

import torch

from florence_forge.data.dataset_encoding import (
    default_prepare_labels,
    supervised_label_count,
)


def test_supervised_label_count_ignores_masked_tokens():
    labels = torch.tensor([-100, -100, 42, 7])
    assert supervised_label_count(labels) == 2


def test_supervised_label_count_zero_when_all_masked():
    labels = torch.full((5,), -100)
    assert supervised_label_count(labels) == 0


def test_default_prepare_labels_warns_on_empty_answer(caplog):
    prompt = {"input_ids": torch.tensor([1, 2, 3])}
    full = {"input_ids": torch.tensor([1, 2, 3])}
    labels = default_prepare_labels(prompt, full)
    assert supervised_label_count(labels) == 0
    assert "无有效监督 token" in caplog.text
