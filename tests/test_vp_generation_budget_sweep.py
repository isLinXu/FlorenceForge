import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_sweep_module():
    path = Path("scripts/experiments/sweep_vp_generation_budgets.py")
    spec = importlib.util.spec_from_file_location("sweep_vp_generation_budgets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_SWEEP = _load_sweep_module()


def test_parse_int_list_ignores_empty_items():
    assert _SWEEP._parse_int_list("64, 128,,192") == [64, 128, 192]


def test_parse_optional_lists_allow_default_sentinel():
    assert _SWEEP._parse_optional_float_list("none,0.8,1.2") == [None, 0.8, 1.2]
    assert _SWEEP._parse_optional_int_list("default,3") == [None, 3]


def test_score_row_prefers_focus_f1_when_focus_bucket_is_set():
    row = {"f1": 0.9, "focus_f1": 0.4}

    assert _SWEEP._score_row(row, "dense") == 0.4
    assert _SWEEP._score_row(row, "") == 0.9


def test_label_part_is_filesystem_friendly():
    assert _SWEEP._label_part("lp", 1.2) == "lp1p2"
    assert _SWEEP._label_part("ngram", None) == "ngramdefault"


def test_policy_sweep_cmd_can_include_phrase_policy(tmp_path):
    args = SimpleNamespace(
        structured_vp_mode="auto",
        structured_vp_box_format="loc_tokens",
        structured_vp_marker_style="plain",
        structured_vp_filter_policy="nms",
        focus_bucket="dense",
        max_bad_cases=3,
        structured_vp_max_boxes_per_label=None,
        structured_vp_max_total_boxes=None,
        structured_vp_max_total_boxes_field="query_box_count",
        structured_vp_nms_iou_threshold=0.5,
        structured_vp_allowed_labels=None,
        structured_vp_allowed_labels_field="text_input",
        include_phrase_label_policy=True,
        include_target_label_oracle=True,
        include_repair_policy=True,
        structured_vp_allowed_label_match_mode="strict",
    )

    cmd = _SWEEP._build_policy_sweep_cmd(
        args,
        summary_path=tmp_path / "summary.json",
        output_dir=tmp_path / "policy",
    )

    assert "--include-phrase-label-policy" in cmd
    assert "--include-target-label-oracle" in cmd
    assert "--include-repair-policy" in cmd
    assert "--structured-vp-max-total-boxes-field" in cmd
    assert "query_box_count" in cmd


def test_render_markdown_includes_budget_and_focus_metrics():
    report = {
        "focus_bucket": "dense",
        "recommended_label": "tokens128_beams1",
        "ranked_runs": [
            {
                "label": "tokens128_beams1",
                "num_samples": 5,
                "avg_pred_boxes": 3.0,
                "avg_gt_boxes": 7.0,
                "generation_budget_near_hit_ratio": 0.2,
                "dense_generation_budget_near_hit_ratio": 0.4,
                "best_policy_row": {
                    "policy": "nms",
                    "focus_recall": 0.5,
                    "focus_f1": 0.6,
                    "focus_avg_pred_boxes": 4.0,
                    "focus_false_negatives": 3,
                },
            }
        ],
    }

    markdown = _SWEEP._render_markdown(report)

    assert "`tokens128_beams1`" in markdown
    assert "`nms`" in markdown
    assert "0.6000" in markdown
