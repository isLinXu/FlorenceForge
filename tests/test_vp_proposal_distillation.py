import json
import subprocess
import sys
from pathlib import Path


def test_build_vp_proposal_distillation_writes_training_jsonl(tmp_path):
    primary_path = tmp_path / "primary_summary.json"
    proposal_path = tmp_path / "proposal_summary.json"
    target = (
        "<ref>cat</ref> <box>"
        "<loc_0><loc_0><loc_10><loc_10>"
        "<loc_50><loc_50><loc_60><loc_60>"
        "</box>"
    )
    primary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "cat",
                    "text_input": "cat",
                    "query_box_count": 2,
                    "gt_box_count": 2,
                    "raw_prediction": "cat<loc_0><loc_0><loc_10><loc_10>",
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    proposal_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "query_label": "cat",
                    "query_box_count": 2,
                    "gt_box_count": 2,
                    "proposal_candidates": [
                        {
                            "label": "cat",
                            "bbox": [50, 50, 60, 60],
                            "confidence": 0.9,
                            "proposal_source": "slic",
                            "proposal_edge_density": 0.8,
                        }
                    ],
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "distill.jsonl"
    script = Path("scripts/data-conversion/build_vp_proposal_distillation.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
            "--output",
            str(output_path),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "query_label",
            "--proposal-selection-policy",
            "confidence",
            "--proposal-allowed-sources",
            "slic",
            "--quality-filter",
            "improvement",
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
    assert summary["delta_tp_total"] == 1
    assert rows[0]["prefix"] == "<OPEN_VOCABULARY_DETECTION>"
    assert rows[0]["text_input"] == "cat"
    assert rows[0]["query_box_count"] == 2
    assert rows[0]["distillation_added_box_count"] == 1
    assert rows[0]["distillation_teacher_tp"] == 2
    assert rows[0]["distillation_primary_tp"] == 1
    assert rows[0]["suffix"] == target
    assert (tmp_path / "distill_summary.json").exists()
    assert (tmp_path / "distill_summary.md").exists()


def test_build_vp_proposal_distillation_quality_filter_drops_regression(tmp_path):
    primary_path = tmp_path / "primary_summary.json"
    proposal_path = tmp_path / "proposal_summary.json"
    target = "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>"
    primary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "cat",
                    "text_input": "cat",
                    "query_box_count": 2,
                    "gt_box_count": 1,
                    "raw_prediction": "cat<loc_0><loc_0><loc_10><loc_10>",
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    proposal_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "query_label": "cat",
                    "query_box_count": 2,
                    "gt_box_count": 1,
                    "proposal_candidates": [
                        {
                            "label": "cat",
                            "bbox": [800, 800, 900, 900],
                            "confidence": 0.9,
                            "proposal_source": "slic",
                        }
                    ],
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "distill.jsonl"
    script = Path("scripts/data-conversion/build_vp_proposal_distillation.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
            "--output",
            str(output_path),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "query_label",
            "--proposal-selection-policy",
            "confidence",
            "--proposal-allowed-sources",
            "slic",
            "--quality-filter",
            "non_regression",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["output_rows"] == 0
    assert summary["skip_counts"]["quality_filter"] == 1
    assert output_path.read_text(encoding="utf-8") == ""


def test_build_vp_proposal_distillation_can_train_on_reference_targets(tmp_path):
    primary_path = tmp_path / "primary_summary.json"
    proposal_path = tmp_path / "proposal_summary.json"
    target = (
        "<ref>cat</ref> <box>"
        "<loc_0><loc_0><loc_10><loc_10>"
        "<loc_50><loc_50><loc_60><loc_60>"
        "<loc_90><loc_90><loc_100><loc_100>"
        "</box>"
    )
    primary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "prefix": "<OPEN_VOCABULARY_DETECTION>",
                    "query_label": "cat",
                    "text_input": "cat",
                    "query_box_count": 3,
                    "gt_box_count": 3,
                    "raw_prediction": "cat<loc_0><loc_0><loc_10><loc_10>",
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    proposal_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "query_label": "cat",
                    "query_box_count": 3,
                    "gt_box_count": 3,
                    "proposal_candidates": [
                        {
                            "label": "cat",
                            "bbox": [50, 50, 60, 60],
                            "confidence": 0.9,
                            "proposal_source": "slic",
                        }
                    ],
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "distill_reference.jsonl"
    script = Path("scripts/data-conversion/build_vp_proposal_distillation.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
            "--output",
            str(output_path),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "query_label",
            "--proposal-selection-policy",
            "confidence",
            "--proposal-allowed-sources",
            "slic",
            "--quality-filter",
            "improvement",
            "--max-proposal-additions-per-record",
            "1",
            "--distillation-target-mode",
            "reference",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["distillation_target_mode"] == "reference"
    assert rows[0]["distillation_target_mode"] == "reference"
    assert rows[0]["distillation_teacher_tp"] == 2
    assert rows[0]["suffix"] == target
    assert rows[0]["suffix"].count("<loc_") == 12
