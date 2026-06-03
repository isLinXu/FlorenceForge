"""dataset_encoding 模块单元测试。"""

import torch

from florence_forge.data.dataset_encoding import (
    build_prompt_and_answer,
    default_prepare_labels,
    get_task_prompt,
    unencoded_sample_dict,
)
from florence_forge.data.dataset_types import TaskSample


def test_get_task_prompt_falls_back_to_florence2_tasks():
    assert get_task_prompt("OD", backend=None).startswith("<")


def test_build_prompt_and_answer_merges_prefix_and_metadata():
    sample = TaskSample(
        image_path="/tmp/x.jpg",
        prefix="<OD>",
        suffix="cat",
        task_type="OD",
        weight=1.0,
        metadata={"text_input": "a cat"},
    )
    prompt, answer = build_prompt_and_answer(sample, backend=None)
    assert answer == "cat"
    assert "a cat" in prompt


def test_default_prepare_labels_masks_prompt_region():
    full_ids = torch.tensor([1, 2, 3, 4, 5])
    prompt_ids = torch.tensor([1, 2])
    labels = default_prepare_labels(
        {"input_ids": prompt_ids},
        {"input_ids": full_ids},
    )
    assert (labels[:2] == -100).all()
    assert labels[2:].tolist() == [3, 4, 5]


def test_unencoded_sample_dict_marks_needs_encoding():
    sample = TaskSample(
        image_path="/p.jpg",
        prefix="p",
        suffix="s",
        task_type="CAPTION",
        weight=0.5,
        metadata={},
    )
    d = unencoded_sample_dict(sample)
    assert d["_needs_encoding"] is True
    assert d["prompt"] == "p"
