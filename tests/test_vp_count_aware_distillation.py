import json
import subprocess
import sys
from pathlib import Path


def test_build_vp_count_aware_distillation_writes_dense_gap_rows(tmp_path):
    summary_path = tmp_path / "vp_inference_summary.json"
    target = (
        "<ref>cat</ref> <box>"
        "<loc_0><loc_0><loc_10><loc_10>"
        "<loc_50><loc_50><loc_60><loc_60>"
        "<loc_80><loc_80><loc_90><loc_90>"
        "<loc_100><loc_100><loc_110><loc_110>"
        "</box>"
    )
    summary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "cat",
                    "text_input": "cat",
                    "query_box_count": 4,
                    "gt_box_count": 4,
                    "raw_prediction": "cat<loc_0><loc_0><loc_10><loc_10>",
                    "target": target,
                },
                {
                    "index": 1,
                    "image": "dog.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "dog",
                    "text_input": "dog",
                    "query_box_count": 1,
                    "gt_box_count": 1,
                    "raw_prediction": "dog<loc_0><loc_0><loc_10><loc_10>",
                    "target": "<ref>dog</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
                },
            ]
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "count_aware.jsonl"
    script = Path("scripts/data-conversion/build_vp_count_aware_distillation.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--inference-summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--text-input-template",
            "{label} count={query_box_count} missing={missing_box_count}",
            "--focus-bucket",
            "dense",
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-filter-policy",
            "nms",
            "--structured-vp-allowed-labels-field",
            "query_label",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["output_rows"] == 1
    assert summary["skip_counts"]["focus_bucket"] == 1
    assert summary["total_missing_boxes_in_output"] == 3
    assert summary["total_recoverable_fn_in_output"] == 3
    assert rows[0]["text_input"] == "cat count=4 missing=3"
    assert rows[0]["query_label"] == "cat"
    assert rows[0]["query_box_count"] == 4
    assert rows[0]["suffix"] == target
    assert rows[0]["distillation_target_mode"] == "reference"
    assert rows[0]["distillation_delta_tp"] == 3
    assert rows[0]["distillation_delta_fn"] == -3
    assert rows[0]["count_aware_missing_box_count"] == 3
    assert (tmp_path / "count_aware_summary.json").exists()
    assert (tmp_path / "count_aware_summary.md").exists()


def test_build_vp_count_aware_distillation_respects_min_missing_boxes(tmp_path):
    summary_path = tmp_path / "vp_inference_summary.json"
    summary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "cat",
                    "text_input": "cat",
                    "query_box_count": 4,
                    "gt_box_count": 4,
                    "raw_prediction": (
                        "cat<loc_0><loc_0><loc_10><loc_10>"
                        "<loc_20><loc_20><loc_30><loc_30>"
                        "<loc_40><loc_40><loc_50><loc_50>"
                    ),
                    "target": (
                        "<ref>cat</ref> <box>"
                        "<loc_0><loc_0><loc_10><loc_10>"
                        "<loc_20><loc_20><loc_30><loc_30>"
                        "<loc_40><loc_40><loc_50><loc_50>"
                        "<loc_60><loc_60><loc_70><loc_70>"
                        "</box>"
                    ),
                },
            ]
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "count_aware.jsonl"
    script = Path("scripts/data-conversion/build_vp_count_aware_distillation.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--inference-summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--min-missing-boxes",
            "2",
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "query_label",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["output_rows"] == 0
    assert summary["skip_counts"]["min_missing_boxes"] == 1
    assert output_path.read_text(encoding="utf-8") == ""
