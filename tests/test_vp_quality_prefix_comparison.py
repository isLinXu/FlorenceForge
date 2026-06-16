import json
import subprocess
import sys
from pathlib import Path


def test_compare_vp_quality_prefix_aggregates_same_prefix(tmp_path):
    baseline_path = tmp_path / "baseline_quality.json"
    candidate_path = tmp_path / "candidate_quality.json"
    output_dir = tmp_path / "prefix_comparison"
    _write_report(
        baseline_path,
        [
            _record(0, tp=1, fp=0, fn=0, pred=1, gt=1),
            _record(1, tp=1, fp=0, fn=1, pred=1, gt=2),
            _record(2, tp=0, fp=4, fn=4, pred=4, gt=4),
        ],
    )
    _write_report(
        candidate_path,
        [
            _record(0, tp=1, fp=0, fn=0, pred=1, gt=1),
            _record(1, tp=2, fp=0, fn=0, pred=2, gt=2),
            _record(2, tp=0, fp=4, fn=4, pred=4, gt=4),
        ],
    )
    script = Path("scripts/experiments/compare_vp_quality_prefix.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            f"baseline={baseline_path}",
            "--report",
            f"candidate={candidate_path}",
            "--max-records",
            "2",
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
    assert comparison["max_records"] == 2
    assert [row["name"] for row in comparison["reports"]] == ["baseline", "candidate"]
    baseline, candidate = comparison["reports"]
    assert baseline["num_samples"] == 2
    assert baseline["true_positives"] == 2
    assert baseline["false_negatives"] == 1
    assert baseline["f1"] == 0.8
    assert candidate["num_samples"] == 2
    assert candidate["true_positives"] == 3
    assert candidate["false_negatives"] == 0
    assert candidate["f1"] == 1.0
    assert candidate["delta_vs_first"]["true_positives"] == 1
    assert candidate["delta_vs_first"]["false_negatives"] == -1
    assert candidate["delta_vs_first"]["f1"] == 0.19999999999999996
    assert (output_dir / "vp_quality_prefix_comparison.json").exists()
    assert (output_dir / "vp_quality_prefix_comparison.md").exists()


def test_compare_vp_quality_prefix_can_filter_bucket_before_prefix(tmp_path):
    report_path = tmp_path / "quality.json"
    output_dir = tmp_path / "prefix_comparison"
    _write_report(
        report_path,
        [
            _record(0, tp=1, fp=0, fn=0, pred=1, gt=1, bucket="single"),
            _record(1, tp=1, fp=0, fn=3, pred=1, gt=4, bucket="dense"),
            _record(2, tp=2, fp=0, fn=2, pred=2, gt=4, bucket="dense"),
        ],
    )
    script = Path("scripts/experiments/compare_vp_quality_prefix.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            f"adapter={report_path}",
            "--max-records",
            "1",
            "--focus-bucket",
            "dense",
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
    row = comparison["reports"][0]
    assert comparison["focus_bucket"] == "dense"
    assert row["num_samples"] == 1
    assert row["selected_record_keys"][0].startswith("1|")
    assert row["true_positives"] == 1
    assert row["false_negatives"] == 3


def _record(index, *, tp, fp, fn, pred, gt, bucket=None):
    bucket = bucket or ("single" if gt <= 1 else "medium" if gt <= 3 else "dense")
    return {
        "index": index,
        "image": f"image_{index}.jpg",
        "prediction_source": "visual_primitive_structured",
        "pred_box_count": pred,
        "gt_box_count": gt,
        "query_box_count": gt,
        "box_count_bucket": bucket,
        "box_count_exact_match": pred == gt,
        "overgenerated": pred > gt,
        "undergenerated": pred < gt,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "matched_ious": [1.0] * tp,
        "bad_case_reasons": [],
    }


def _write_report(path, records):
    path.write_text(
        json.dumps({"num_samples": len(records), "records": records}),
        encoding="utf-8",
    )
