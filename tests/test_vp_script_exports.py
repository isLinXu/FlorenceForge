"""VP 脚本入口所需的 re-export 回归测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_load_vp_quality_summary_reexported():
    from florence_forge.evaluation.vp_detection_quality import load_vp_quality_summary

    assert callable(load_vp_quality_summary)


def test_evaluate_vp_detection_quality_script_imports(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "index": 0,
                        "raw_prediction": "cat<loc_0><loc_0><loc_100><loc_100>",
                        "target": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
                        "prefix": "<OD_VP>",
                    }
                ]
            }
        ),
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
        timeout=120,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout)
    assert report["precision"] == 1.0
    assert (output_dir / "vp_detection_quality.json").exists()
