import importlib.util
from pathlib import Path

import torch


_PROBE_PATH = Path("scripts/experiments/probe_florence_vp_tokens.py")
_SPEC = importlib.util.spec_from_file_location("probe_florence_vp_tokens", _PROBE_PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<|ref|>": 10,
            "<|/ref|>": 11,
            "<|box|>": 12,
            "<|/box|>": 13,
            "<|point|>": 14,
            "<|/point|>": 15,
            "<loc_0>": 20,
            "<loc_1>": 21,
            "<loc_999>": 22,
            "cat": 30,
            "<": 40,
            "ref": 41,
            ">": 42,
            "/": 43,
            "box": 44,
        }
        self.reverse = {value: key for key, value in self.vocab.items()}
        self.sequences = {
            "<ref>": [40, 41, 42],
            "</ref>": [40, 43, 41, 42],
            "<box>": [40, 44, 42],
            "</box>": [40, 43, 44, 42],
            "<point>": [40, 0, 42],
            "</point>": [40, 43, 0, 42],
        }
        self.additional_special_tokens = [
            "<|ref|>",
            "<|/ref|>",
            "<|box|>",
            "<|/box|>",
            "<|point|>",
            "<|/point|>",
        ]
        self.unk_token_id = 0

    def __len__(self):
        return 64

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        if text in self.sequences:
            return {"input_ids": list(self.sequences[text])}
        return {"input_ids": [self.vocab.get(text, self.unk_token_id)]}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.reverse.get(int(token_id), f"<unk_{token_id}>") for token_id in token_ids)


def test_build_token_inventory_marks_vp_tokens_as_single_tokens():
    inventory = probe.build_token_inventory(FakeTokenizer())

    assert inventory["all_vp_tokens_single"] is True
    assert inventory["vp_tokens"][0]["token"] == "<|ref|>"
    assert inventory["vp_tokens"][0]["token_id"] == 10
    assert inventory["loc_tokens"][0]["single_token"] is True


def test_summarize_label_tokens_counts_supervised_vp_and_loc_tokens():
    tokenizer = FakeTokenizer()
    labels = torch.tensor([-100, 10, 30, 11, 12, 20, 21, 22, 13])

    summary = probe.summarize_label_tokens(
        labels,
        tokenizer,
        {
            "<|ref|>": 10,
            "<|/ref|>": 11,
            "<|box|>": 12,
            "<|/box|>": 13,
        },
    )

    assert summary["supervised_token_count"] == 8
    assert summary["vp_marker_token_count"] == 4
    assert summary["loc_token_count"] == 3
    assert summary["first_supervised_tokens"][0]["is_vp_marker"] is True


def test_summarize_label_tokens_counts_plain_marker_sequences():
    tokenizer = FakeTokenizer()
    labels = torch.tensor([-100, 40, 41, 42, 30, 40, 43, 41, 42])

    summary = probe.summarize_label_tokens(
        labels,
        tokenizer,
        {"<ref>": None, "</ref>": None},
        vp_tokenizations={"<ref>": [40, 41, 42], "</ref>": [40, 43, 41, 42]},
    )

    assert summary["vp_marker_token_count"] == 0
    assert summary["vp_marker_sequence_count"] == 2
    assert summary["vp_marker_sequence_counts"]["<ref>"] == 1
    assert summary["vp_marker_sequence_counts"]["</ref>"] == 1


def test_summarize_label_tokens_counts_spaced_plain_marker_sequences():
    tokenizer = FakeTokenizer()
    labels = torch.tensor([28696, 44, 42])

    summary = probe.summarize_label_tokens(
        labels,
        tokenizer,
        {"<box>": None},
        vp_tokenizations={"<box>": [[40, 44, 42], [28696, 44, 42]]},
    )

    assert summary["vp_marker_sequence_count"] == 1
    assert summary["vp_marker_sequence_counts"]["<box>"] == 1


def test_rank_tokens_from_scores_reports_rank_and_top_tokens():
    tokenizer = FakeTokenizer()
    score = torch.zeros(64)
    score[30] = 4.0
    score[12] = 3.0
    score[10] = 2.0

    summary = probe.rank_tokens_from_scores(
        score,
        tokenizer,
        {"<|ref|>": 10, "<|box|>": 12},
        top_k=2,
    )

    assert summary["selected_tokens"]["<|box|>"]["rank"] == 2
    assert summary["selected_tokens"]["<|ref|>"]["rank"] == 3
    assert summary["top_tokens"][0]["token"] == "cat"


def test_probe_status_uses_first_content_step_not_forced_bos():
    summary = {
        "token_inventory": {"all_vp_tokens_single": True, "marker_style": "special"},
        "label_probe": {"total_vp_marker_token_count": 4, "total_vp_marker_sequence_count": 4},
        "generation_probe": {
            "records": [{
                "steps": [
                    {
                        "generated_token": "<s>",
                        "selected_tokens": {"<|ref|>": {"rank": None}},
                    },
                    {
                        "generated_token": "cat",
                        "selected_tokens": {"<|ref|>": {"rank": 23000}},
                    },
                ],
            }],
        },
    }

    assert probe._classify_probe(summary) == "generation_prior_blocks_wrapper"


def test_probe_status_allows_plain_multitoken_markers():
    summary = {
        "token_inventory": {"all_vp_tokens_single": False, "marker_style": "plain"},
        "label_probe": {"total_vp_marker_token_count": 0, "total_vp_marker_sequence_count": 4},
        "ref_open_probe_key": "<ref>[0]",
        "generation_probe": {
            "records": [{
                "steps": [{
                    "generated_token": "cat",
                    "selected_tokens": {"<ref>[0]": {"rank": 8}},
                }],
            }],
        },
    }

    assert probe._classify_probe(summary) == "vp_token_path_probe_ready"


def test_probe_parser_accepts_data_path_and_plain_marker_style():
    args = probe.parse_args([
        "--data-path",
        "plain.jsonl",
        "--task-type",
        "OPEN_VOCABULARY_DETECTION",
        "--marker-style",
        "plain",
        "--max-samples",
        "2",
    ])

    assert args.data_path == "plain.jsonl"
    assert args.task_type == "OPEN_VOCABULARY_DETECTION"
    assert args.marker_style == "plain"
    assert args.max_samples == 2


def test_resolve_data_path_accepts_manifest_key_fallbacks():
    manifest = {
        "train_od_path": "od.jsonl",
        "train_grounding_path": "grounding.jsonl",
    }

    path = probe._resolve_data_path(
        manifest,
        split="train",
        data_key="train_grounding_effective_path,train_grounding_path",
    )

    assert str(path) == "grounding.jsonl"
