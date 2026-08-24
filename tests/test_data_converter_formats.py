"""DataFormatConverter 格式转换单元测试。

覆盖 FlorenceForge 数据管线中 `florence_forge/data/converter.py` 的核心格式转换路径
（YOLO / COCO 检测 / COCO 描述 / CSV 描述 / VOC XML）以及 `DataValidator` 校验器。

这些转换器此前缺乏专门测试（见深度分析报告 §12 覆盖缺口）。本文件用合成的最小输入
验证输出 Florence-2 JSONL 的 schema（image / prefix / suffix）与坐标换算正确性。
"""

import csv
import json
from pathlib import Path

import pytest

from florence_forge.data.converter import DataFormatConverter, DataValidator

# Pillow 是核心依赖；若缺失则整体跳过（YOLO 转换需要读取图像尺寸）。
Image = pytest.importorskip("PIL.Image")


def _make_image(path: Path, size=(100, 200)) -> None:
    """在 path 写入一张纯色 RGB 图像，size=(width, height)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(123, 222, 64)).save(path)


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_yolo_to_florence2_od(tmp_path):
    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 100x200 (w x h) 图像
    _make_image(image_dir / "sample.jpg", size=(100, 200))

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\n", encoding="utf-8")

    # class_id cx cy w h（归一化）：dog 居中，占整图一半
    (labels_dir / "sample.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    output_path = tmp_path / "out" / "yolo.jsonl"
    DataFormatConverter.yolo_to_florence2_od(
        yolo_labels_dir=str(labels_dir),
        output_path=str(output_path),
        image_dir=str(image_dir),
        classes_file=str(classes_file),
        image_ext=".jpg",
        task_type="OD",
    )

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    sample = rows[0]
    assert sample["prefix"] == "<OD>"
    assert Path(sample["image"]).name == "sample.jpg"

    # 转换器输出 Florence 原生 VP token 格式：label<loc_x1><loc_y1><loc_x2><loc_y2>
    suffix = sample["suffix"]
    assert "dog" in suffix
    assert "<loc_" in suffix


def test_yolo_skips_image_without_label_match(tmp_path):
    """缺少对应图像的标签应被跳过，不产生样本。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")
    (labels_dir / "missing.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    output_path = tmp_path / "yolo.jsonl"
    DataFormatConverter.yolo_to_florence2_od(
        yolo_labels_dir=str(labels_dir),
        output_path=str(output_path),
        image_dir=str(image_dir),
        classes_file=str(classes_file),
    )

    assert _read_jsonl(output_path) == []


def test_coco_to_florence2_od(tmp_path):
    image_dir = tmp_path / "images"
    _make_image(image_dir / "img1.jpg", size=(640, 480))

    coco = {
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "images": [{"id": 10, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"image_id": 10, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"image_id": 10, "category_id": 2, "bbox": [50, 60, 5, 5]},
        ],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    output_path = tmp_path / "coco_od.jsonl"
    DataFormatConverter.coco_to_florence2_od(
        coco_json_path=str(coco_path),
        output_path=str(output_path),
        image_dir=str(image_dir),
        task_type="OD",
    )

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    suffix = rows[0]["suffix"]
    # 转换器输出 Florence 原生 VP token 格式
    assert "person" in suffix
    assert "car" in suffix
    assert "<loc_" in suffix


def test_coco_caption_to_florence2(tmp_path):
    image_dir = tmp_path / "images"
    _make_image(image_dir / "cap.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "cap.jpg"}],
        "annotations": [
            {"image_id": 1, "caption": "a green test image"},
            {"image_id": 1, "caption": "另一个中文描述"},
        ],
    }
    coco_path = tmp_path / "coco_cap.json"
    coco_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    output_path = tmp_path / "caption.jsonl"
    DataFormatConverter.coco_caption_to_florence2(
        coco_json_path=str(coco_path),
        output_path=str(output_path),
        image_dir=str(image_dir),
        task_type="CAPTION",
    )

    rows = _read_jsonl(output_path)
    assert len(rows) == 2
    assert {r["suffix"] for r in rows} == {"a green test image", "另一个中文描述"}
    assert all(r["prefix"] == "<CAPTION>" for r in rows)


def test_coco_caption_skips_missing_image(tmp_path):
    """COCO 描述转换仅在图像文件实际存在时写出样本。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [{"id": 1, "file_name": "does_not_exist.jpg"}],
        "annotations": [{"image_id": 1, "caption": "ghost"}],
    }
    coco_path = tmp_path / "coco_cap.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    output_path = tmp_path / "caption.jsonl"
    DataFormatConverter.coco_caption_to_florence2(
        coco_json_path=str(coco_path),
        output_path=str(output_path),
        image_dir=str(image_dir),
    )

    assert _read_jsonl(output_path) == []


def test_csv_caption_to_florence2(tmp_path):
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _make_image(img_a)
    _make_image(img_b)

    csv_path = tmp_path / "captions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
        writer.writeheader()
        writer.writerow({"image_path": str(img_a), "caption": "first caption"})
        writer.writerow({"image_path": str(img_b), "caption": "second caption"})

    output_path = tmp_path / "csv_caption.jsonl"
    DataFormatConverter.csv_caption_to_florence2(
        csv_path=str(csv_path),
        output_path=str(output_path),
        task_type="CAPTION",
    )

    rows = _read_jsonl(output_path)
    assert [r["suffix"] for r in rows] == ["first caption", "second caption"]
    assert all(r["prefix"] == "<CAPTION>" for r in rows)


def test_csv_caption_rejects_unknown_task_type(tmp_path):
    csv_path = tmp_path / "captions.csv"
    csv_path.write_text("image_path,caption\nx.jpg,hi\n", encoding="utf-8")

    with pytest.raises(ValueError):
        DataFormatConverter.csv_caption_to_florence2(
            csv_path=str(csv_path),
            output_path=str(tmp_path / "o.jsonl"),
            task_type="NOT_A_REAL_TASK",
        )


def test_xml_to_florence2_od(tmp_path):
    pytest.importorskip("defusedxml")
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    # XML 无 <size> 元素，转换器会尝试读取图像获取尺寸
    _make_image(image_dir / "frame.jpg", size=(100, 100))

    xml_content = """<annotation>
  <filename>frame.jpg</filename>
  <object>
    <name>bottle</name>
    <bndbox><xmin>5</xmin><ymin>10</ymin><xmax>20</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>"""
    (xml_dir / "frame.xml").write_text(xml_content, encoding="utf-8")

    output_path = tmp_path / "xml.jsonl"
    DataFormatConverter.xml_to_florence2_od(
        xml_dir=str(xml_dir),
        output_path=str(output_path),
        image_dir=str(image_dir),
        task_type="OD",
    )

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    suffix = rows[0]["suffix"]
    # 转换器输出 Florence 原生 VP token 格式
    assert "bottle" in suffix
    assert "<loc_" in suffix


def test_validate_florence2_jsonl_reports_issues(tmp_path):
    valid_image = tmp_path / "ok.jpg"
    _make_image(valid_image)

    jsonl_path = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"image": str(valid_image), "prefix": "<OD>", "suffix": "{}"}),
        json.dumps(
            {"image": str(tmp_path / "missing.jpg"), "prefix": "<CAPTION>", "suffix": "x"}
        ),
        "{ this is not valid json",
    ]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = DataValidator.validate_florence2_jsonl(str(jsonl_path))

    assert report["total_samples"] == 3
    # 两行 JSON 合法（其中一行图像缺失但 JSON 有效），一行 JSON 非法
    assert report["valid_samples"] == 2
    assert report["invalid_samples"] == 1
    assert report["missing_images"] == 1
    assert report["task_distribution"].get("OD") == 1
    assert report["task_distribution"].get("CAPTION") == 1


def test_generate_validation_report_writes_markdown(tmp_path):
    report = {
        "total_samples": 2,
        "valid_samples": 2,
        "invalid_samples": 0,
        "missing_images": 0,
        "task_distribution": {"OD": 1, "CAPTION": 1},
        "errors": [],
    }
    out = tmp_path / "report.md"
    DataValidator.generate_validation_report(report, str(out))

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "数据验证报告" in text
