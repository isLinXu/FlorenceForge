import importlib.util
import json
from pathlib import Path


def _load_ensemble_module():
    path = Path("scripts/experiments/ensemble_vp_inference_summaries.py")
    spec = importlib.util.spec_from_file_location("ensemble_vp_inference_summaries", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ENSEMBLE = _load_ensemble_module()


def test_ensemble_vp_summaries_unions_candidate_boxes(tmp_path):
    base_record = {
        "index": 0,
        "image": str(tmp_path / "sample.jpg"),
        "prefix": "<OPEN_VOCABULARY_DETECTION>",
        "text_input": "cat",
        "query_box_count": 2,
        "gt_box_count": 2,
        "target": (
            "<ref>cat</ref> <box>"
            "<loc_0><loc_0><loc_100><loc_100>"
            "<loc_200><loc_200><loc_300><loc_300>"
            "</box>"
        ),
    }
    summary_a = tmp_path / "a.json"
    summary_b = tmp_path / "b.json"
    summary_a.write_text(
        json.dumps({
            "records": [
                dict(base_record, raw_prediction="cat<loc_0><loc_0><loc_100><loc_100>")
            ]
        }),
        encoding="utf-8",
    )
    summary_b.write_text(
        json.dumps({
            "records": [
                dict(base_record, raw_prediction="cat<loc_200><loc_200><loc_300><loc_300>")
            ]
        }),
        encoding="utf-8",
    )

    args = _ENSEMBLE.parse_args([
        "--summary",
        f"a={summary_a}",
        "--summary",
        f"b={summary_b}",
        "--output-dir",
        str(tmp_path / "out"),
        "--structured-vp-marker-style",
        "plain",
        "--structured-vp-allowed-labels-field",
        "text_input",
    ])
    result = _ENSEMBLE.run(args)

    output = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    record = output["records"][0]
    assert output["num_samples"] == 1
    assert record["pred_box_count"] == 2
    assert record["ensemble_member_detection_counts"] == {"a": 1, "b": 1}
    assert "<loc_0><loc_0><loc_100><loc_100>" in record["raw_prediction"]
    assert "<loc_200><loc_200><loc_300><loc_300>" in record["raw_prediction"]
