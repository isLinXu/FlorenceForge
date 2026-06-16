import json
import subprocess
import sys
from pathlib import Path


def test_structured_vp_filter_replay_recomputes_box_metrics(tmp_path):
    summary_path = tmp_path / "vp_inference_visualization_summary.json"
    summary_path.write_text(
        json.dumps({
            "model_path": "model",
            "adapter_dir": "adapter",
            "data_path": "data.jsonl",
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "raw_prediction": (
                        "cat<loc_0><loc_0><loc_10><loc_10>"
                        "footwear<loc_20><loc_20><loc_30><loc_30>"
                    ),
                    "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
                    "gt_box_count": 1,
                },
                {
                    "index": 1,
                    "image": "zebra.jpg",
                    "raw_prediction": (
                        "zebra<loc_1><loc_1><loc_11><loc_11>"
                        "<loc_2><loc_2><loc_12><loc_12>"
                    ),
                    "target": "<ref>zebra</ref> <box><loc_1><loc_1><loc_11><loc_11></box>",
                    "gt_box_count": 1,
                },
            ],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "replay"
    script = Path("scripts/experiments/replay_structured_vp_filters.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--inference-summary",
            f"adapter={summary_path}",
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--filter-config",
            "total1:max_total_boxes=1",
            "--filter-config",
            "nms:nms_iou_threshold=0.5",
            "--filter-config",
            "cat_only:allowed_labels=cat",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    total1 = summary["runs"]["adapter"]["policies"]["total1"]
    assert total1["num_samples"] == 2
    assert total1["avg_pred_boxes"] == 1.0
    assert total1["avg_gt_boxes"] == 1.0
    assert total1["box_count_overgeneration_ratio"] == 0.0
    assert total1["structured_filtered_detection_count"] == 2
    assert total1["records"][0]["structured_prediction"] == (
        "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>"
    )

    policy_summary = output_dir / "adapter_total1_vp_inference_visualization_summary.json"
    assert policy_summary.exists()
    written = json.loads(policy_summary.read_text())
    assert written["structured_vp_max_total_boxes"] == 1

    nms = summary["runs"]["adapter"]["policies"]["nms"]
    assert nms["structured_vp_nms_iou_threshold"] == 0.5
    assert nms["records"][1]["structured_filtered_detection_count"] == 1

    cat_only = summary["runs"]["adapter"]["policies"]["cat_only"]
    assert cat_only["structured_vp_allowed_labels"] == "cat"
    assert cat_only["records"][0]["pred_box_count"] == 1
    assert cat_only["records"][1]["pred_box_count"] == 0


def test_target_count_proposal_replay_fills_missing_boxes(tmp_path):
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
                    "query_label": "cat",
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
                    "raw_prediction": (
                        "cat<loc_0><loc_0><loc_10><loc_10>"
                        "<loc_50><loc_50><loc_60><loc_60>"
                    ),
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "proposal_replay"
    script = Path("scripts/experiments/replay_vp_target_count_proposals.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
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
    assert report["fill_summary"]["added_proposal_boxes"] == 1
    assert report["fill_summary"]["target_deficit_after"] == 0
    assert report["quality"]["true_positives"] == 2
    assert report["quality"]["false_negatives"] == 0
    assert report["quality"]["f1"] == 1.0
    assert (output_dir / "vp_target_count_proposal_summary.json").exists()
    assert (output_dir / "vp_target_count_proposal_quality.json").exists()
    summary = json.loads((output_dir / "vp_target_count_proposal_summary.json").read_text())
    assert summary["records"][0]["target_count_added_box_count"] == 1
    assert summary["records"][0]["pred_box_count"] == 2


def test_target_count_proposal_replay_uses_ranked_candidates(tmp_path):
    primary_path = tmp_path / "primary_summary.json"
    proposal_path = tmp_path / "proposal_summary.json"
    target = "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>"
    primary_path.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": "cat.jpg",
                    "query_label": "cat",
                    "query_box_count": 1,
                    "gt_box_count": 1,
                    "raw_prediction": "",
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
                    "query_box_count": 1,
                    "gt_box_count": 1,
                    "raw_prediction": "cat<loc_800><loc_800><loc_900><loc_900>",
                    "proposal_candidates": [
                        {
                            "label": "cat",
                            "bbox": [800, 800, 900, 900],
                            "confidence": 0.1,
                            "proposal_source": "grid",
                            "proposal_edge_density": 0.1,
                        },
                        {
                            "label": "cat",
                            "bbox": [0, 0, 10, 10],
                            "confidence": 0.9,
                            "proposal_source": "slic",
                            "proposal_edge_density": 0.8,
                        },
                    ],
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ranked_proposal_replay"
    script = Path("scripts/experiments/replay_vp_target_count_proposals.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--proposal-selection-policy",
            "confidence",
            "--proposal-min-confidence",
            "0.5",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["fill_summary"]["added_proposal_boxes"] == 1
    assert report["quality"]["true_positives"] == 1
    assert report["quality"]["false_positives"] == 0
    assert report["quality"]["f1"] == 1.0
    summary = json.loads((output_dir / "vp_target_count_proposal_summary.json").read_text())
    record = summary["records"][0]
    assert record["proposal_raw_detection_count"] == 2
    assert record["proposal_pred_box_count"] == 1
    assert record["structured_prediction"] == (
        "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>"
    )
    assert summary["target_count_proposal_config"]["proposal_selection_policy"] == "confidence"
    assert summary["target_count_proposal_config"]["proposal_min_confidence"] == 0.5


def test_target_count_proposal_replay_caps_added_boxes(tmp_path):
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
                    "query_label": "cat",
                    "query_box_count": 2,
                    "gt_box_count": 2,
                    "raw_prediction": "",
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
                    "raw_prediction": (
                        "cat<loc_0><loc_0><loc_10><loc_10>"
                        "<loc_50><loc_50><loc_60><loc_60>"
                    ),
                    "target": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "capped_proposal_replay"
    script = Path("scripts/experiments/replay_vp_target_count_proposals.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--primary-summary",
            str(primary_path),
            "--proposal-summary",
            str(proposal_path),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--max-proposal-additions-per-record",
            "1",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["fill_summary"]["added_proposal_boxes"] == 1
    assert report["fill_summary"]["target_deficit_after"] == 1
    assert report["quality"]["true_positives"] == 1
    assert report["quality"]["false_negatives"] == 1
    summary = json.loads((output_dir / "vp_target_count_proposal_summary.json").read_text())
    assert summary["target_count_proposal_config"]["max_proposal_additions_per_record"] == 1


def test_vp_experiment_runner_can_plan_filter_replay(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-filter-replay",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--filter-replay-config",
            "total1:max_total_boxes=1",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["infer_adapter", "infer_baseline", "audit", "filter_replay"]
    filter_cmd = summary["commands"][-1]["cmd"]
    assert "scripts/experiments/replay_structured_vp_filters.py" in filter_cmd
    assert "--inference-summary" in filter_cmd
    assert f"adapter={output_dir / 'adapter_inference' / 'vp_inference_visualization_summary.json'}" in filter_cmd
    assert f"baseline={output_dir / 'baseline_inference' / 'vp_inference_visualization_summary.json'}" in filter_cmd
    assert "--filter-config" in filter_cmd
    assert "total1:max_total_boxes=1" in filter_cmd
    assert summary["filter_replay_summary"] == str(output_dir / "postfilter_replay" / "filtered_replay_summary.json")
