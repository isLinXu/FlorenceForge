import json
import subprocess
import sys
from pathlib import Path

from florence_forge.evaluation.vp_detection_quality import (
    VPDetectionQualityConfig,
    analyze_vp_target_count_gap,
    compare_vp_quality_record_reports,
    compare_vp_quality_reports,
    compute_bbox_iou,
    evaluate_vp_detection_quality,
    evaluate_vp_summary,
    match_vp_detections,
    render_vp_detection_quality_markdown,
    render_vp_policy_comparison_markdown,
    render_vp_record_comparison_markdown,
    render_vp_target_count_gap_markdown,
)
from florence_forge.evaluation.vp_report_card import (
    VPReportCardThresholds,
    build_vp_report_card,
    render_vp_report_card_markdown,
)


def test_vp_detection_quality_matches_by_label_and_iou():
    predictions = [
        {"label": "cat", "bbox": [0, 0, 100, 100]},
        {"label": "dog", "bbox": [0, 0, 50, 50]},
    ]
    references = [
        {"label": "cat", "bbox": [0, 0, 100, 100]},
        {"label": "cat", "bbox": [200, 200, 300, 300]},
    ]

    result = match_vp_detections(predictions, references, iou_threshold=0.5)

    assert abs(compute_bbox_iou([0, 0, 100, 100], [50, 50, 150, 150]) - (1 / 7)) < 1e-9
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["mean_matched_iou"] == 1.0


def test_evaluate_vp_detection_quality_reports_overgeneration_and_single_target_hits():
    predictions = [
        (
            "cat<loc_0><loc_0><loc_100><loc_100>"
            "footwear<loc_200><loc_200><loc_300><loc_300>"
        ),
        "zebra<loc_10><loc_10><loc_110><loc_110>",
    ]
    references = [
        "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
        "<ref>zebra</ref> <box><loc_10><loc_10><loc_110><loc_110></box>",
    ]

    report = evaluate_vp_detection_quality(
        predictions,
        references,
        config=VPDetectionQualityConfig(marker_style="plain"),
    )

    assert report["num_samples"] == 2
    assert report["precision"] == 2 / 3
    assert report["recall"] == 1.0
    assert report["box_count_overgeneration_ratio"] == 0.5
    assert report["single_target_hit_ratio"] == 1.0
    assert report["single_target_exact_hit_ratio"] == 0.5
    assert report["bad_cases"][0]["reasons"] == ["false_positive", "overgenerated"]


def test_evaluate_vp_detection_quality_reports_box_count_buckets():
    predictions = [
        "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
        "<ref>dog</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
        "",
    ]
    references = [
        "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
        (
            "<ref>dog</ref> <box>"
            "<loc_0><loc_0><loc_100><loc_100>"
            "<loc_200><loc_200><loc_300><loc_300>"
            "</box>"
        ),
        (
            "<ref>person</ref> <box>"
            "<loc_0><loc_0><loc_100><loc_100>"
            "<loc_200><loc_200><loc_300><loc_300>"
            "<loc_400><loc_400><loc_500><loc_500>"
            "<loc_600><loc_600><loc_700><loc_700>"
            "</box>"
        ),
    ]

    report = evaluate_vp_detection_quality(
        predictions,
        references,
        config=VPDetectionQualityConfig(marker_style="plain"),
    )
    buckets = report["box_count_bucket_summary"]

    assert buckets["single"]["num_samples"] == 1
    assert buckets["single"]["recall"] == 1.0
    assert buckets["medium"]["num_samples"] == 1
    assert buckets["medium"]["true_positives"] == 1
    assert buckets["medium"]["false_negatives"] == 1
    assert buckets["medium"]["recall"] == 0.5
    assert buckets["dense"]["num_samples"] == 1
    assert buckets["dense"]["false_negatives"] == 4
    assert buckets["dense"]["recall"] == 0.0


def test_evaluate_vp_summary_can_apply_single_target_policy():
    summary = {
        "records": [
            {
                "index": 0,
                "image": "cat.jpg",
                "raw_prediction": (
                    "cat<loc_0><loc_0><loc_100><loc_100>"
                    "footwear<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                "prefix": "<OD_VP>",
            }
        ]
    }

    unfiltered = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(marker_style="plain"),
    )
    filtered = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            filter_policy="auto",
        ),
    )

    assert unfiltered["avg_pred_boxes"] == 2.0
    assert unfiltered["single_target_exact_hit_ratio"] == 0.0
    assert filtered["avg_pred_boxes"] == 1.0
    assert filtered["single_target_exact_hit_ratio"] == 1.0
    assert filtered["records"][0]["filtered_detection_count"] == 1


def test_evaluate_vp_summary_can_apply_nms_policy():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "cat<loc_0><loc_0><loc_100><loc_100>"
                    "<loc_5><loc_5><loc_105><loc_105>"
                    "dog<loc_5><loc_5><loc_105><loc_105>"
                ),
                "target": (
                    "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>\n"
                    "<ref>dog</ref> <box><loc_5><loc_5><loc_105><loc_105></box>"
                ),
            }
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            filter_policy="nms",
            nms_iou_threshold=0.5,
        ),
    )

    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["avg_pred_boxes"] == 2.0
    assert report["records"][0]["raw_detection_count"] == 3
    assert report["records"][0]["filtered_detection_count"] == 1


def test_evaluate_vp_summary_reports_repaired_tail_counts():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "<ref>person</ref> <box><loc_0><loc_0><loc_100><loc_100></box> "
                    "<<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": (
                    "<ref>person</ref> <box>"
                    "<loc_0><loc_0><loc_100><loc_100>"
                    "<loc_200><loc_200><loc_300><loc_300>"
                    "</box>"
                ),
            }
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            repair_malformed_tail=True,
        ),
    )

    assert report["f1"] == 1.0
    assert report["repaired_tail_detection_count"] == 1
    assert report["repaired_tail_record_ratio"] == 1.0
    assert report["records"][0]["repaired_tail_detection_count"] == 1
    assert report["records"][0]["prediction_source"] == "visual_primitive_repaired_tail"


def test_evaluate_vp_summary_can_apply_dynamic_total_box_cap_field():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "cat<loc_0><loc_0><loc_100><loc_100>"
                    "cat<loc_200><loc_200><loc_300><loc_300>"
                    "cat<loc_400><loc_400><loc_500><loc_500>"
                ),
                "target": (
                    "<ref>cat</ref> <box>"
                    "<loc_0><loc_0><loc_100><loc_100>"
                    "<loc_200><loc_200><loc_300><loc_300>"
                    "</box>"
                ),
                "query_box_count": 2,
            }
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            max_total_boxes_field="query_box_count",
        ),
    )

    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["avg_pred_boxes"] == 2.0
    assert report["config"]["max_total_boxes_field"] == "query_box_count"
    assert report["records"][0]["filtered_detection_count"] == 1


def test_evaluate_vp_summary_can_apply_allowed_label_filter():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "cat<loc_0><loc_0><loc_100><loc_100>"
                    "footwear<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
            }
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            allowed_labels="cat",
        ),
    )

    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["avg_pred_boxes"] == 1.0
    assert report["records"][0]["raw_detection_count"] == 2
    assert report["records"][0]["filtered_detection_count"] == 1


def test_evaluate_vp_summary_can_apply_per_record_allowed_label_field():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "cat<loc_0><loc_0><loc_100><loc_100>"
                    "footwear<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                "text_input": "cat",
            },
            {
                "index": 1,
                "raw_prediction": (
                    "dog<loc_10><loc_10><loc_110><loc_110>"
                    "cat<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>dog</ref> <box><loc_10><loc_10><loc_110><loc_110></box>",
                "text_input": "dog",
            },
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            allowed_labels_field="text_input",
        ),
    )

    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["avg_pred_boxes"] == 1.0
    assert report["config"]["allowed_labels_field"] == "text_input"
    assert report["records"][0]["allowed_labels"] == "cat"
    assert report["records"][1]["allowed_labels"] == "dog"
    assert report["records"][0]["filtered_detection_count"] == 1


def test_evaluate_vp_summary_can_apply_phrase_contained_label_matching():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "coffee cup<loc_0><loc_0><loc_100><loc_100>"
                    "business sign<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>cup</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                "text_input": "cup,bus",
            }
        ]
    }

    strict = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            allowed_labels_field="text_input",
        ),
    )
    phrase = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            allowed_labels_field="text_input",
            label_match_mode="contains",
            allowed_label_match_mode="contains",
        ),
    )

    assert strict["f1"] == 0.0
    assert strict["records"][0]["filtered_detection_count"] == 2
    assert phrase["precision"] == 1.0
    assert phrase["recall"] == 1.0
    assert phrase["records"][0]["filtered_detection_count"] == 1
    assert phrase["config"]["label_match_mode"] == "contains"
    assert phrase["config"]["allowed_label_match_mode"] == "contains"


def test_evaluate_vp_summary_can_apply_target_label_oracle_field():
    summary = {
        "records": [
            {
                "index": 0,
                "raw_prediction": (
                    "zebra<loc_0><loc_0><loc_100><loc_100>"
                    "footwear<loc_200><loc_200><loc_300><loc_300>"
                ),
                "target": "<ref>zebra</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
            }
        ]
    }

    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            marker_style="plain",
            allowed_labels_field="target_labels",
        ),
    )

    assert report["precision"] == 1.0
    assert report["records"][0]["allowed_labels"] == ["zebra"]


def test_render_vp_detection_quality_markdown_includes_bad_cases():
    report = evaluate_vp_detection_quality(
        ["cat<loc_0><loc_0><loc_100><loc_100>dog<loc_0><loc_0><loc_50><loc_50>"],
        ["<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"],
        config=VPDetectionQualityConfig(marker_style="plain"),
    )

    markdown = render_vp_detection_quality_markdown(report)

    assert "VP Detection Quality" in markdown
    assert "Box Count Buckets" in markdown
    assert "false_positive" in markdown


def test_compare_vp_quality_reports_recommends_best_policy_with_caveat():
    predictions = [
        (
            "cat<loc_0><loc_0><loc_100><loc_100>"
            "footwear<loc_200><loc_200><loc_300><loc_300>"
        ),
        "dog<loc_10><loc_10><loc_110><loc_110>",
    ]
    references = [
        "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
        "<ref>dog</ref> <box><loc_10><loc_10><loc_110><loc_110></box>",
    ]
    unfiltered = evaluate_vp_detection_quality(
        predictions,
        references,
        config=VPDetectionQualityConfig(marker_style="plain"),
    )
    allowed = evaluate_vp_detection_quality(
        predictions,
        references,
        config=VPDetectionQualityConfig(marker_style="plain", allowed_labels="cat,dog"),
    )

    comparison = compare_vp_quality_reports({
        "none": unfiltered,
        "allowed": allowed,
    })
    markdown = render_vp_policy_comparison_markdown(comparison)

    assert comparison["recommended_policy"] == "allowed"
    assert comparison["ranked_rows"][0]["policy"] == "allowed"
    assert comparison["ranked_rows"][0]["policy_kind"] == "allowed_labels"
    assert comparison["recommendation"]["confidence"] == "exploratory"
    assert "explicit allowed-label" in " ".join(comparison["recommendation"]["caveats"])
    assert "| `allowed` " in markdown


def test_compare_vp_quality_reports_can_focus_dense_bucket():
    overall_strong_dense_weak = {
        "num_samples": 20,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "avg_pred_boxes": 1.5,
        "avg_gt_boxes": 1.5,
        "box_count_exact_match_ratio": 0.9,
        "box_count_overgeneration_ratio": 0.0,
        "mean_matched_iou": 0.9,
        "box_count_bucket_summary": {
            "dense": {
                "num_samples": 10,
                "precision": 0.8,
                "recall": 0.3,
                "f1": 0.4364,
                "avg_pred_boxes": 3.0,
                "avg_gt_boxes": 8.0,
                "box_count_exact_match_ratio": 0.0,
                "box_count_overgeneration_ratio": 0.0,
                "false_positives": 2,
                "false_negatives": 35,
            }
        },
    }
    overall_weaker_dense_better = {
        "num_samples": 20,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.7,
        "avg_pred_boxes": 2.0,
        "avg_gt_boxes": 2.0,
        "box_count_exact_match_ratio": 0.7,
        "box_count_overgeneration_ratio": 0.0,
        "mean_matched_iou": 0.8,
        "box_count_bucket_summary": {
            "dense": {
                "num_samples": 10,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 0.6667,
                "avg_pred_boxes": 5.0,
                "avg_gt_boxes": 8.0,
                "box_count_exact_match_ratio": 0.2,
                "box_count_overgeneration_ratio": 0.0,
                "false_positives": 4,
                "false_negatives": 20,
            }
        },
    }

    comparison = compare_vp_quality_reports(
        {
            "overall": overall_strong_dense_weak,
            "dense": overall_weaker_dense_better,
        },
        focus_bucket="dense",
    )
    markdown = render_vp_policy_comparison_markdown(comparison)

    assert comparison["focus_bucket"] == "dense"
    assert comparison["recommended_policy"] == "dense"
    assert comparison["ranked_rows"][0]["focus_f1"] == 0.6667
    assert "Focus bucket: `dense`" in markdown
    assert "focus recall" in markdown


def test_compare_vp_quality_record_reports_summarizes_per_sample_deltas():
    baseline = {
        "num_samples": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "true_positives": 2,
        "false_positives": 0,
        "false_negatives": 2,
        "avg_pred_boxes": 1.0,
        "avg_gt_boxes": 2.0,
        "records": [
            {
                "index": 0,
                "image": "/tmp/a.jpg",
                "allowed_labels": "cat",
                "box_count_bucket": "medium",
                "pred_box_count": 1,
                "gt_box_count": 2,
                "query_box_count": 2,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "undergenerated": True,
            },
            {
                "index": 1,
                "image": "/tmp/b.jpg",
                "allowed_labels": "dog",
                "box_count_bucket": "single",
                "pred_box_count": 1,
                "gt_box_count": 1,
                "query_box_count": 1,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
            },
        ],
    }
    candidate = {
        "num_samples": 2,
        "precision": 3 / 4,
        "recall": 1.0,
        "f1": 6 / 7,
        "true_positives": 3,
        "false_positives": 1,
        "false_negatives": 0,
        "avg_pred_boxes": 2.0,
        "avg_gt_boxes": 2.0,
        "records": [
            {
                "index": 0,
                "image": "/tmp/a.jpg",
                "allowed_labels": "cat",
                "box_count_bucket": "medium",
                "pred_box_count": 2,
                "gt_box_count": 2,
                "query_box_count": 2,
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 0,
                "undergenerated": False,
            },
            {
                "index": 1,
                "image": "/tmp/b.jpg",
                "allowed_labels": "dog",
                "box_count_bucket": "single",
                "pred_box_count": 2,
                "gt_box_count": 1,
                "query_box_count": 1,
                "true_positives": 1,
                "false_positives": 1,
                "false_negatives": 0,
            },
        ],
    }

    comparison = compare_vp_quality_record_reports(
        candidate,
        baseline,
        candidate_name="adapter",
        baseline_name="base",
    )
    markdown = render_vp_record_comparison_markdown(comparison)

    assert comparison["compared_records"] == 2
    assert comparison["delta"]["true_positives"] == 1
    assert comparison["delta"]["false_positives"] == 1
    assert comparison["delta"]["false_negatives"] == -1
    assert comparison["tp_improved_records"] == 1
    assert comparison["fp_increased_records"] == 1
    assert comparison["undergeneration_fixed_records"] == 1
    assert comparison["outcome_counts"] == {
        "strict_improvement": 1,
        "precision_regression": 1,
    }
    assert comparison["bucket_summary"]["medium"]["delta_true_positives"] == 1
    assert "VP Record Comparison" in markdown
    assert "`strict_improvement`" in markdown


def test_vp_detection_quality_script_writes_json_and_markdown(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "raw_prediction": "cat<loc_0><loc_0><loc_100><loc_100>",
                    "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                    "prefix": "<OD_VP>",
                }
            ]
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "quality"
    script = Path("scripts/experiments/evaluate_vp_detection_quality.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["precision"] == 1.0
    assert (output_dir / "vp_detection_quality.json").exists()
    assert (output_dir / "vp_detection_quality.md").exists()


def test_vp_policy_comparison_script_writes_json_and_markdown(tmp_path):
    unfiltered = evaluate_vp_detection_quality(
        ["cat<loc_0><loc_0><loc_100><loc_100>dog<loc_0><loc_0><loc_50><loc_50>"],
        ["<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"],
        config=VPDetectionQualityConfig(marker_style="plain"),
    )
    single = evaluate_vp_detection_quality(
        ["cat<loc_0><loc_0><loc_100><loc_100>dog<loc_0><loc_0><loc_50><loc_50>"],
        ["<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"],
        config=VPDetectionQualityConfig(marker_style="plain", filter_policy="single-target"),
    )
    unfiltered_path = tmp_path / "none.json"
    single_path = tmp_path / "single.json"
    unfiltered_path.write_text(json.dumps(unfiltered), encoding="utf-8")
    single_path.write_text(json.dumps(single), encoding="utf-8")
    output_dir = tmp_path / "comparison"
    script = Path("scripts/experiments/compare_vp_quality_policies.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            f"none={unfiltered_path}",
            "--report",
            f"single={single_path}",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads(result.stdout)
    assert comparison["recommended_policy"] == "single"
    assert (output_dir / "vp_policy_comparison.json").exists()
    assert (output_dir / "vp_policy_comparison.md").exists()


def test_vp_record_comparison_script_writes_json_and_markdown(tmp_path):
    baseline = {
        "records": [
            {
                "index": 0,
                "image": "/tmp/a.jpg",
                "allowed_labels": "cat",
                "box_count_bucket": "dense",
                "pred_box_count": 1,
                "gt_box_count": 4,
                "query_box_count": 4,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 3,
                "undergenerated": True,
            }
        ]
    }
    candidate = {
        "records": [
            {
                "index": 0,
                "image": "/tmp/a.jpg",
                "allowed_labels": "cat",
                "box_count_bucket": "dense",
                "pred_box_count": 2,
                "gt_box_count": 4,
                "query_box_count": 4,
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 2,
                "undergenerated": True,
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    output_dir = tmp_path / "records"
    script = Path("scripts/experiments/compare_vp_quality_records.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidate-report",
            str(candidate_path),
            "--baseline-report",
            str(baseline_path),
            "--candidate-name",
            "adapter",
            "--baseline-name",
            "baseline",
            "--output-dir",
            str(output_dir),
            "--focus-bucket",
            "dense",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads(result.stdout)
    assert comparison["focus_bucket"] == "dense"
    assert comparison["delta"]["true_positives"] == 1
    assert (output_dir / "vp_record_comparison.json").exists()
    assert (output_dir / "vp_record_comparison.md").exists()


def test_target_count_gap_analysis_estimates_oracle_count_fill():
    report = {
        "records": [
            {
                "index": 0,
                "allowed_labels": "cat",
                "box_count_bucket": "dense",
                "pred_box_count": 1,
                "gt_box_count": 4,
                "query_box_count": 4,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 3,
            },
            {
                "index": 1,
                "allowed_labels": "dog",
                "box_count_bucket": "dense",
                "pred_box_count": 5,
                "gt_box_count": 4,
                "query_box_count": 4,
                "true_positives": 3,
                "false_positives": 2,
                "false_negatives": 1,
            },
        ]
    }

    analysis = analyze_vp_target_count_gap(report, focus_bucket="dense")

    assert analysis["num_records"] == 2
    assert analysis["count_gap"]["target_box_deficit"] == 3
    assert analysis["count_gap"]["target_box_overage"] == 1
    assert analysis["count_gap"]["recoverable_false_negatives"] == 3
    assert analysis["count_gap"]["unrecoverable_false_negatives"] == 1
    assert analysis["current"]["true_positives"] == 4
    assert analysis["oracle_count_fill"]["true_positives"] == 7
    assert analysis["oracle_count_fill"]["false_negatives"] == 1
    assert analysis["oracle_count_fill"]["f1"] > analysis["current"]["f1"]
    assert analysis["bucket_summary"]["dense"]["records_blocked_by_no_count_slots"] == 1
    markdown = render_vp_target_count_gap_markdown(analysis)
    assert "Recoverable FN" in markdown
    assert "Top Gap Records" in markdown


def test_target_count_gap_script_writes_json_and_markdown(tmp_path):
    report = {
        "records": [
            {
                "index": 0,
                "allowed_labels": "cat",
                "box_count_bucket": "dense",
                "pred_box_count": 1,
                "gt_box_count": 4,
                "query_box_count": 4,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 3,
            }
        ]
    }
    report_path = tmp_path / "quality.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "gap"
    script = Path("scripts/experiments/analyze_vp_target_count_gap.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--focus-bucket",
            "dense",
            "--max-rows",
            "5",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    analysis = json.loads(result.stdout)
    assert analysis["focus_bucket"] == "dense"
    assert analysis["oracle_count_fill"]["recovered_true_positives"] == 3
    assert (output_dir / "vp_target_count_gap.json").exists()
    assert (output_dir / "vp_target_count_gap.md").exists()


def test_vp_quality_policy_sweep_script_writes_reports_and_comparison(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "raw_prediction": (
                        "cat<loc_0><loc_0><loc_100><loc_100>"
                        "footwear<loc_200><loc_200><loc_300><loc_300>"
                    ),
                    "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                    "text_input": "cat",
                    "query_box_count": 1,
                }
            ]
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sweep"
    script = Path("scripts/experiments/sweep_vp_quality_policies.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "text_input",
            "--structured-vp-max-total-boxes-field",
            "query_box_count",
            "--include-phrase-label-policy",
            "--include-repair-policy",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    sweep = json.loads(result.stdout)
    assert sweep["recommended_policy"] in {
        "none",
        "single",
        "text_input_allowed",
        "none_repair",
        "single_repair",
        "text_input_allowed_repair",
    }
    assert (output_dir / "none" / "vp_detection_quality.json").exists()
    assert (output_dir / "nms" / "vp_detection_quality.json").exists()
    assert (output_dir / "single" / "vp_detection_quality.json").exists()
    assert (output_dir / "none_repair" / "vp_detection_quality.json").exists()
    assert (output_dir / "nms_repair" / "vp_detection_quality.json").exists()
    assert (output_dir / "single_repair" / "vp_detection_quality.json").exists()
    assert (output_dir / "text_input_allowed" / "vp_detection_quality.json").exists()
    assert (output_dir / "text_input_allowed_repair" / "vp_detection_quality.json").exists()
    assert (output_dir / "text_input_phrase_allowed" / "vp_detection_quality.json").exists()
    assert (output_dir / "text_input_phrase_allowed_repair" / "vp_detection_quality.json").exists()
    assert (output_dir / "vp_policy_comparison.json").exists()
    none_report = json.loads((output_dir / "none" / "vp_detection_quality.json").read_text())
    assert none_report["config"]["max_total_boxes_field"] == "query_box_count"
    repair_report = json.loads(
        (output_dir / "nms_repair" / "vp_detection_quality.json").read_text()
    )
    assert repair_report["config"]["repair_malformed_tail"] is True


def test_build_vp_report_card_flags_underfit_and_repair_dependency():
    quality_report = {
        "num_samples": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "true_positives": 2,
        "false_positives": 0,
        "false_negatives": 2,
        "avg_pred_boxes": 1.0,
        "avg_gt_boxes": 2.0,
        "box_count_exact_match_ratio": 0.0,
        "box_count_undergeneration_ratio": 1.0,
        "box_count_overgeneration_ratio": 0.0,
        "repaired_tail_detection_count": 1,
        "repaired_tail_record_ratio": 0.5,
        "avg_repaired_tail_detection_count": 0.5,
        "records": [
            {
                "index": 0,
                "pred_box_count": 1,
                "gt_box_count": 2,
                "query_box_count": 2,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "box_count_bucket": "medium",
            },
            {
                "index": 1,
                "pred_box_count": 1,
                "gt_box_count": 2,
                "query_box_count": 2,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "box_count_bucket": "medium",
            },
        ],
    }
    policy_sweep = {
        "recommended_policy": "none_repair",
        "comparison": {
            "recommendation": {
                "policy": "none_repair",
                "confidence": "exploratory",
                "general_detection_policy": "none",
            },
            "ranked_rows": [
                {
                    "rank": 1,
                    "policy": "none_repair",
                    "policy_kind": "none",
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1": 2 / 3,
                    "constraints": ["repair_malformed_tail"],
                },
                {
                    "rank": 2,
                    "policy": "none",
                    "policy_kind": "none",
                    "precision": 1.0,
                    "recall": 0.25,
                    "f1": 0.4,
                    "constraints": [],
                },
            ],
        },
    }
    gap = analyze_vp_target_count_gap(quality_report)

    card = build_vp_report_card(
        quality_report,
        policy_sweep=policy_sweep,
        target_count_gap=gap,
        thresholds=VPReportCardThresholds(min_samples=5, max_repair_record_ratio=0.25),
    )
    checks = {check["name"]: check for check in card["checks"]}

    assert card["status"] == "fail"
    assert card["readiness"] == "needs_work"
    assert checks["sample_size"]["status"] == "warn"
    assert checks["recall"]["status"] == "fail"
    assert checks["undergeneration_ratio"]["status"] == "fail"
    assert checks["repair_dependency"]["status"] == "warn"
    assert checks["policy_confidence"]["status"] == "warn"
    assert checks["target_count_gap"]["status"] == "warn"
    assert card["policy_summary"]["repair_lift"]["delta_f1"] > 0
    assert any("count-conditioned" in action for action in card["next_actions"])

    markdown = render_vp_report_card_markdown(card)
    assert "# Florence-VP Report Card" in markdown
    assert "Target-Count Gap" in markdown


def test_build_vp_report_card_flags_raw_wrapper_decoder_dependency():
    quality_report = {
        "num_samples": 4,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "true_positives": 4,
        "false_positives": 0,
        "false_negatives": 0,
        "avg_pred_boxes": 1.0,
        "avg_gt_boxes": 1.0,
        "box_count_exact_match_ratio": 1.0,
        "box_count_undergeneration_ratio": 0.0,
        "box_count_overgeneration_ratio": 0.0,
        "repaired_tail_record_ratio": 0.0,
        "prediction_source_counts": {"florence_native": 4},
    }

    card = build_vp_report_card(
        quality_report,
        thresholds=VPReportCardThresholds(min_samples=1),
    )
    checks = {check["name"]: check for check in card["checks"]}

    assert card["status"] == "fail"
    assert card["quality_summary"]["raw_vp_format_ratio"] == 0.0
    assert card["quality_summary"]["structured_decoder_ratio"] == 1.0
    assert checks["raw_vp_internalization"]["status"] == "fail"
    assert checks["structured_decoder_dependency"]["status"] == "fail"
    assert any("wrapper-focused SFT" in action for action in card["next_actions"])

    markdown = render_vp_report_card_markdown(card)
    assert "Raw VP format ratio" in markdown
    assert "Structured decoder dependency" in markdown


def test_vp_report_card_script_writes_json_and_markdown(tmp_path):
    quality_report = {
        "num_samples": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 1,
        "avg_pred_boxes": 1.0,
        "avg_gt_boxes": 2.0,
        "box_count_exact_match_ratio": 0.0,
        "box_count_undergeneration_ratio": 1.0,
        "box_count_overgeneration_ratio": 0.0,
        "records": [
            {
                "index": 0,
                "pred_box_count": 1,
                "gt_box_count": 2,
                "query_box_count": 2,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "box_count_bucket": "medium",
            }
        ],
    }
    policy_sweep = {
        "recommended_policy": "none",
        "comparison": {
            "recommendation": {
                "policy": "none",
                "confidence": "moderate",
                "general_detection_policy": "none",
            },
            "ranked_rows": [
                {
                    "rank": 1,
                    "policy": "none",
                    "policy_kind": "none",
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1": 2 / 3,
                    "constraints": [],
                },
            ],
        },
    }
    quality_path = tmp_path / "vp_detection_quality.json"
    sweep_path = tmp_path / "vp_quality_policy_sweep.json"
    output_dir = tmp_path / "card"
    quality_path.write_text(json.dumps(quality_report), encoding="utf-8")
    sweep_path.write_text(json.dumps(policy_sweep), encoding="utf-8")
    script = Path("scripts/experiments/build_vp_report_card.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--quality-report",
            str(quality_path),
            "--policy-sweep",
            str(sweep_path),
            "--output-dir",
            str(output_dir),
            "--min-samples",
            "2",
            "--min-raw-vp-format-ratio",
            "0.8",
            "--max-structured-decoder-ratio",
            "0.4",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    card = json.loads(result.stdout)
    assert card["status"] == "fail"
    assert card["thresholds"]["min_raw_vp_format_ratio"] == 0.8
    assert card["thresholds"]["max_structured_decoder_ratio"] == 0.4
    assert card["target_count_gap_summary"]["recoverable_false_negatives"] == 1
    assert (output_dir / "vp_report_card.json").exists()
    assert (output_dir / "vp_report_card.md").exists()
