import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from florence_forge.data import VisualPrimitiveConverter
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


def test_coco_to_vp_od_outputs_parseable_jsonl(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
    }
    coco_path = tmp_path / "coco.json"
    output_path = tmp_path / "vp_od.jsonl"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    VisualPrimitiveConverter.coco_to_vp_od(
        str(coco_path),
        str(output_path),
        str(image_dir),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prefix"] == "<OD>"
    assert rows[0]["task_family"] == "visual_primitive"
    assert rows[0]["source_format"] == "coco"
    assert VisualPrimitiveParser().parse_detections(rows[0]["suffix"]) == [
        {"label": "cat", "bbox": [100, 200, 400, 599], "confidence": 1.0}
    ]


def test_coco_to_vp_counting_outputs_one_sample_per_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10]},
        ],
    }
    coco_path = tmp_path / "coco.json"
    output_path = tmp_path / "vp_count.jsonl"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    VisualPrimitiveConverter.coco_to_vp_counting(
        str(coco_path),
        str(output_path),
        str(image_dir),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prefix"] == "<COUNT>"
    assert rows[0]["count_label"] == "cat"
    assert rows[0]["count"] == 2
    assert "There are 2 cat in this image." in rows[0]["suffix"]
    assert len(VisualPrimitiveParser().parse_detections(rows[0]["suffix"])) == 2


def test_coco_to_vp_grounding_outputs_one_query_sample_per_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10]},
        ],
    }
    coco_path = tmp_path / "coco.json"
    output_path = tmp_path / "vp_grounding.jsonl"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    VisualPrimitiveConverter.coco_to_vp_grounding(
        str(coco_path),
        str(output_path),
        str(image_dir),
        box_format="loc_tokens",
        marker_style="plain",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["prefix"] == "<CAPTION_TO_PHRASE_GROUNDING>"
    assert rows[0]["text_input"] == "cat"
    assert rows[0]["query_label"] == "cat"
    assert rows[0]["query_box_count"] == 1
    assert rows[0]["suffix"] == "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"
    assert rows[1]["text_input"] == "dog"


def test_yolo_to_vp_od_outputs_parseable_jsonl(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")
    output_path = tmp_path / "vp_yolo.jsonl"

    VisualPrimitiveConverter.yolo_to_vp_od(
        str(label_dir),
        str(output_path),
        str(image_dir),
        str(classes_file),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert VisualPrimitiveParser().parse_detections(rows[0]["suffix"]) == [
        {"label": "cat", "bbox": [400, 300, 599, 699], "confidence": 1.0}
    ]


def test_yolo_to_vp_od_can_emit_loc_token_boxes(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")
    output_path = tmp_path / "vp_yolo_loc.jsonl"

    VisualPrimitiveConverter.yolo_to_vp_od(
        str(label_dir),
        str(output_path),
        str(image_dir),
        str(classes_file),
        box_format="loc_tokens",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert "<loc_400><loc_300><loc_599><loc_699>" in rows[0]["suffix"]
    assert VisualPrimitiveParser().parse_detections(rows[0]["suffix"]) == [
        {"label": "cat", "bbox": [400, 300, 599, 699], "confidence": 1.0}
    ]


def test_yolo_to_vp_od_can_emit_plain_marker_boxes(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")
    output_path = tmp_path / "vp_yolo_plain.jsonl"

    VisualPrimitiveConverter.yolo_to_vp_od(
        str(label_dir),
        str(output_path),
        str(image_dir),
        str(classes_file),
        box_format="loc_tokens",
        marker_style="plain",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["vp_marker_style"] == "plain"
    assert rows[0]["suffix"] == "<ref>cat</ref> <box><loc_400><loc_300><loc_599><loc_699></box>"
    assert VisualPrimitiveParser().parse_detections(rows[0]["suffix"]) == [
        {"label": "cat", "bbox": [400, 300, 599, 699], "confidence": 1.0}
    ]


def test_yolo_to_vp_counting_outputs_one_sample_per_label(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")
    output_path = tmp_path / "vp_yolo_count.jsonl"

    VisualPrimitiveConverter.yolo_to_vp_counting(
        str(label_dir),
        str(output_path),
        str(image_dir),
        str(classes_file),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prefix"] == "<COUNT>"
    assert rows[0]["source_format"] == "yolo"
    assert rows[0]["count_label"] == "cat"
    assert rows[0]["count"] == 2
    assert "There are 2 cat in this image." in rows[0]["suffix"]
    assert len(VisualPrimitiveParser().parse_detections(rows[0]["suffix"])) == 2


def test_yolo_to_vp_grounding_outputs_query_samples(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\n", encoding="utf-8")
    output_path = tmp_path / "vp_yolo_grounding.jsonl"

    VisualPrimitiveConverter.yolo_to_vp_grounding(
        str(label_dir),
        str(output_path),
        str(image_dir),
        str(classes_file),
        box_format="loc_tokens",
        marker_style="plain",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["text_input"] for row in rows] == ["cat", "dog"]
    assert rows[0]["prefix"] == "<CAPTION_TO_PHRASE_GROUNDING>"
    assert VisualPrimitiveParser().parse_detections(rows[0]["suffix"]) == [
        {"label": "cat", "bbox": [400, 300, 599, 699], "confidence": 1.0}
    ]


def test_vp_od_jsonl_to_query_grounding_derives_one_sample_per_label(tmp_path):
    input_path = tmp_path / "od_vp.jsonl"
    input_path.write_text(
        json.dumps({
            "image": str(tmp_path / "sample.jpg"),
            "prefix": "<OD>",
            "suffix": (
                "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>\n"
                "<ref>dog</ref> <box><loc_200><loc_200><loc_300><loc_300></box>"
            ),
            "source_format": "unit",
        }) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "query_vp.jsonl"

    VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
        str(input_path),
        str(output_path),
        marker_style="plain",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["source_task"] == "<OD>"
    assert rows[0]["text_input"] == "cat"
    assert rows[0]["suffix"] == "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"
    assert rows[1]["text_input"] == "dog"


def test_build_vp_query_curriculum_oversamples_multi_instance_queries(tmp_path):
    input_path = tmp_path / "query.jsonl"
    input_rows = [
        {
            "image": str(tmp_path / "a.jpg"),
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>",
            "query_label": "cat",
            "text_input": "cat",
            "query_box_count": 1,
        },
        {
            "image": str(tmp_path / "b.jpg"),
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": (
                "<ref>dog</ref> <box><loc_0><loc_0><loc_100><loc_100>"
                "<loc_200><loc_200><loc_300><loc_300></box>"
            ),
            "query_label": "dog",
            "text_input": "dog",
            "query_box_count": 2,
        },
        {
            "image": str(tmp_path / "c.jpg"),
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": (
                "<ref>person</ref> <box><loc_0><loc_0><loc_100><loc_100>"
                "<loc_100><loc_100><loc_200><loc_200>"
                "<loc_200><loc_200><loc_300><loc_300>"
                "<loc_300><loc_300><loc_400><loc_400></box>"
            ),
            "query_label": "person",
            "text_input": "person",
            "query_box_count": 4,
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in input_rows) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "curriculum.jsonl"
    script = Path("scripts/data-conversion/build_vp_query_curriculum.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--single-weight",
            "1",
            "--medium-weight",
            "2",
            "--dense-weight",
            "3",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert summary["bucket_counts"] == {"single": 1, "medium": 1, "dense": 1}
    assert summary["bucket_output_counts"] == {"single": 1, "medium": 2, "dense": 3}
    assert summary["avg_query_box_count_input"] == (1 + 2 + 4) / 3
    assert summary["avg_query_box_count_output"] == (1 + 2 + 2 + 4 + 4 + 4) / 6
    assert [row["curriculum_bucket"] for row in rows] == [
        "single",
        "medium",
        "medium",
        "dense",
        "dense",
        "dense",
    ]
    assert rows[-1]["curriculum_repeat_total"] == 3
    assert (tmp_path / "curriculum_summary.json").exists()
    assert (tmp_path / "curriculum_summary.md").exists()
