import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_visualize_module():
    path = Path("scripts/infer/visualize_florence_vp_adapter.py")
    spec = importlib.util.spec_from_file_location("visualize_florence_vp_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_VISUALIZE = _load_visualize_module()


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()


class _LocTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {
            "<loc_0>": 10,
            "<loc_1>": 11,
            "<loc_999>": 9990,
            "other": 12,
        }

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


def test_filter_rows_can_select_dense_query_rows():
    rows = [
        {"query_box_count": 1, "text_input": "cat"},
        {"query_box_count": 3, "text_input": "dog"},
        {"query_box_count": 4, "text_input": "person"},
        {"gt_box_count": 9, "text_input": "umbrella"},
    ]
    args = SimpleNamespace(min_query_boxes=4, max_query_boxes=None)

    filtered = _VISUALIZE._filter_rows(rows, args)

    assert [row["text_input"] for row in filtered] == ["person", "umbrella"]


def test_filter_rows_can_cap_query_rows():
    rows = [
        {"query_box_count": 1, "text_input": "cat"},
        {"query_box_count": 3, "text_input": "dog"},
        {"query_box_count": 4, "text_input": "person"},
    ]
    args = SimpleNamespace(min_query_boxes=None, max_query_boxes=3)

    filtered = _VISUALIZE._filter_rows(rows, args)

    assert [row["text_input"] for row in filtered] == ["cat", "dog"]


def test_prediction_length_diagnostics_counts_loc_tokens_and_budget_hits():
    text = (
        "dog <loc_0> <loc_0> <loc_10> <loc_10> "
        "<loc_20> <loc_20> <loc_30> <loc_30>"
    )

    diagnostics = _VISUALIZE._prediction_length_diagnostics(
        text,
        tokenizer=_WhitespaceTokenizer(),
        max_new_tokens=10,
    )

    assert diagnostics["raw_prediction_token_count"] == 9
    assert diagnostics["raw_loc_token_count"] == 8
    assert diagnostics["raw_loc_box_count"] == 2
    assert diagnostics["generation_budget_near_hit"] is True
    assert diagnostics["generation_budget_hit"] is False


def test_format_text_input_template_uses_query_fields():
    formatted = _VISUALIZE._format_text_input(
        "all instances of {label} from {text_input}: {query_box_count}",
        "people",
        [
            {"label": "person", "bbox": [0, 0, 1, 1]},
            {"label": "person", "bbox": [2, 2, 3, 3]},
        ],
    )

    assert formatted == "all instances of person from people: 2"


def test_generation_kwargs_only_includes_explicit_search_settings():
    args = SimpleNamespace(
        length_penalty=1.2,
        repetition_penalty=None,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    kwargs = _VISUALIZE._generation_kwargs(args)

    assert kwargs == {
        "length_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }


def test_resolve_loc_token_ids_ignores_unknown_tokens():
    token_ids = _VISUALIZE._resolve_loc_token_ids(_LocTokenizer())

    assert token_ids == [10, 11, 9990]


def test_vp_box_count_stopping_criteria_counts_four_loc_tokens_per_box():
    criterion = _VISUALIZE.VPBoxCountStoppingCriteria(
        loc_token_ids=[10, 11, 12, 13],
        max_total_boxes=2,
    )

    assert criterion(torch.tensor([[10, 11, 12, 13, 99]])) is False
    assert criterion.triggered is False
    assert criterion(torch.tensor([[10, 11, 12, 13, 10, 11, 12, 13]])) is True
    assert criterion.triggered is True
    assert criterion.last_loc_token_count == 8


def test_vp_count_stopping_runtime_info_reports_triggered_state():
    criterion = _VISUALIZE.VPBoxCountStoppingCriteria(
        loc_token_ids=[10, 11, 12, 13],
        max_total_boxes=1,
    )
    criterion(torch.tensor([[10, 11, 12, 13]]))

    info = _VISUALIZE._vp_count_stopping_runtime_info([criterion])

    assert info == {
        "vp_count_stopping_triggered": True,
        "vp_count_stopping_last_loc_token_count": 4,
    }


def test_build_vp_count_stopping_criteria_reports_unavailable_without_loc_vocab():
    criteria, info = _VISUALIZE._build_vp_count_stopping_criteria(
        tokenizer=_WhitespaceTokenizer(),
        max_total_boxes=2,
    )

    assert criteria is None
    assert info["vp_count_stopping_available"] is False
    assert info["vp_count_stopping_target_boxes"] == 2


def test_build_continuation_decoder_prefix_strips_eos_and_closing_box():
    raw = "</s><ref>person</ref> <box><loc_1><loc_2><loc_3><loc_4></box></s>"

    prefix = _VISUALIZE._build_continuation_decoder_prefix(
        raw,
        fallback_prefix="<ref>person</ref> <box>",
    )

    assert prefix == "<ref>person</ref> <box><loc_1><loc_2><loc_3><loc_4>"


def test_build_continuation_decoder_prefix_truncates_malformed_tail_after_box():
    raw = (
        "</s><ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840></box> "
        "<<loc_282><loc_517><loc_446><loc_840>person</s>"
    )

    prefix = _VISUALIZE._build_continuation_decoder_prefix(
        raw,
        fallback_prefix="<ref>person</ref> <box>",
    )

    assert prefix == "<ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840>"


def test_build_continuation_decoder_prefix_strips_trailing_label_noise_in_box():
    raw = (
        "</s><ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840>"
        "person</box> <<loc_282><loc_517><loc_446><loc_840></s>"
    )

    prefix = _VISUALIZE._build_continuation_decoder_prefix(
        raw,
        fallback_prefix="<ref>person</ref> <box>",
    )

    assert prefix == "<ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840>"


def test_build_continuation_decoder_prefix_uses_fallback_for_empty_generation():
    prefix = _VISUALIZE._build_continuation_decoder_prefix(
        "</s><pad>",
        fallback_prefix="<ref>person</ref> <box>",
    )

    assert prefix == "<ref>person</ref> <box>"


def test_vp_continuation_box_count_uses_parseable_structured_count_not_raw_tail():
    raw = (
        "<ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840></box> "
        "<<loc_282><loc_517><loc_446><loc_840>person"
        "<loc_327><loc_487><loc_366><loc_661>"
    )
    decoder = _VISUALIZE.StructuredVisualPrimitiveDecoder(
        box_format="loc_tokens",
        marker_style="plain",
    )

    count, source = _VISUALIZE._vp_continuation_box_count(
        raw,
        structured_decoder=decoder,
        structured_filter_caps={
            "max_boxes_per_label": None,
            "max_total_boxes": 6,
            "nms_iou_threshold": None,
            "allowed_labels": ["person"],
        },
    )

    assert _VISUALIZE._raw_loc_box_count(raw) == 3
    assert count == 1
    assert source == "structured_visual_primitive"


def test_vp_continuation_box_count_can_use_repaired_tail_count():
    raw = (
        "<ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840></box> "
        "<<loc_282><loc_517><loc_446><loc_840>person"
        "<loc_327><loc_487><loc_366><loc_661>"
    )
    decoder = _VISUALIZE.StructuredVisualPrimitiveDecoder(
        box_format="loc_tokens",
        marker_style="plain",
    )

    count, source = _VISUALIZE._vp_continuation_box_count(
        raw,
        structured_decoder=decoder,
        structured_filter_caps={
            "max_boxes_per_label": None,
            "max_total_boxes": 6,
            "nms_iou_threshold": 0.5,
            "allowed_labels": ["person"],
        },
        repair_malformed_tail=True,
    )

    assert count == 2
    assert source == "structured_visual_primitive_repaired_tail"


def test_vp_continuation_box_count_can_count_open_box_via_structured_native_parse():
    raw = (
        "<ref>person</ref> <box><loc_1><loc_1><loc_4><loc_4>"
        "person<loc_5><loc_5><loc_8><loc_8>"
    )
    decoder = _VISUALIZE.StructuredVisualPrimitiveDecoder(
        box_format="loc_tokens",
        marker_style="plain",
    )

    count, source = _VISUALIZE._vp_continuation_box_count(
        raw,
        structured_decoder=decoder,
        structured_filter_caps={
            "max_boxes_per_label": None,
            "max_total_boxes": None,
            "nms_iou_threshold": None,
            "allowed_labels": ["person"],
        },
    )

    assert count == 2
    assert source == "structured_florence_native"


def test_resolve_row_positive_int_field_uses_first_valid_candidate():
    row = {
        "query_box_count": "6",
        "nested": {"count": "bad"},
    }

    assert _VISUALIZE._resolve_row_positive_int_field(
        row,
        "nested.count,query_box_count",
        fallback=3,
    ) == 6
    assert _VISUALIZE._resolve_row_positive_int_field(
        row,
        "missing",
        fallback=3,
    ) == 3
