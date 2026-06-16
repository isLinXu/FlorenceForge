import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_image_proposal_summary_generates_parseable_vp(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), color=(240, 240, 240)).save(image_path)
    source_summary = tmp_path / "source_summary.json"
    source_summary.write_text(
        json.dumps({
            "records": [
                {
                    "index": 0,
                    "image": str(image_path),
                    "query_label": "cat",
                    "query_box_count": 1,
                    "gt_box_count": 1,
                    "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_500><loc_500></box>",
                }
            ],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "proposals"
    script = Path("scripts/experiments/generate_vp_image_proposal_summary.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-summary",
            str(source_summary),
            "--output-dir",
            str(output_dir),
            "--methods",
            "grid",
            "--grid-size-fractions",
            "0.5",
            "--grid-stride-fraction",
            "1.0",
            "--max-proposals-per-record",
            "10",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["num_samples"] == 1
    summary = json.loads((output_dir / "vp_image_proposal_summary.json").read_text())
    record = summary["records"][0]
    assert record["structured_vp_format_valid"] is True
    assert record["pred_box_count"] > 0
    assert record["proposal_candidates"]
    candidate = record["proposal_candidates"][0]
    assert {"label", "bbox", "confidence", "proposal_source", "proposal_area_ratio"} <= set(candidate)
    assert candidate["label"] == "cat"
    assert summary["proposal_gt_recall_iou50"] == 1.0
    assert (output_dir / "vp_image_proposal_summary.json").exists()
    assert (output_dir / "vp_image_proposal_summary.md").exists()
