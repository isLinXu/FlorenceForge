import importlib.util
import yaml
from pathlib import Path
from types import SimpleNamespace

from florence_forge.cli._helpers import TASK_CONFIG_MAPPING, TASK_DESCRIPTIONS
from florence_forge.cli.commands import run_data_conversion
from florence_forge.cli.main import create_parser
from florence_forge.core.config import TrainingConfig


def test_visual_primitive_training_config_is_registered_and_valid():
    config_path = Path(TASK_CONFIG_MAPPING["visual_primitive"])
    assert TASK_CONFIG_MAPPING["vp"] == str(config_path)
    assert "视觉原语" in TASK_DESCRIPTIONS["visual_primitive"]
    assert config_path.exists()

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = TrainingConfig.from_dict(config_data)

    assert config.experiment_name == "florence2_visual_primitive_training"
    assert config.model_settings.enable_visual_primitives is True
    assert config.model_settings.use_lora is True
    assert config.model_settings.lora_config.r == 32
    assert config.model_settings.lora_config.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]
    assert config.model_settings.lora_config.modules_to_save == [
        "lm_head",
        "model.shared",
    ]
    assert config.data_settings.use_augmentation is False


def test_cli_parser_accepts_visual_primitive_training_alias():
    parser = create_parser()

    args = parser.parse_args(["train", "--task", "visual_primitive"])

    assert args.command == "train"
    assert args.task == "visual_primitive"


def test_cli_parser_accepts_structured_vp_auto_inference_mode():
    parser = create_parser()

    args = parser.parse_args([
        "infer",
        "--model",
        "model",
        "--input",
        "image.png",
        "--output",
        "out",
        "--structured-vp-mode",
        "auto",
        "--structured-vp-marker-style",
        "plain",
        "--structured-vp-max-boxes-per-label",
        "1",
        "--structured-vp-max-total-boxes",
        "2",
        "--structured-vp-filter-policy",
        "nms",
        "--structured-vp-nms-iou-threshold",
        "0.6",
        "--structured-vp-allowed-labels",
        "cat,dog",
    ])

    assert args.command == "infer"
    assert args.structured_vp_mode == "auto"
    assert args.structured_vp_decode is False
    assert args.structured_vp_marker_style == "plain"
    assert args.structured_vp_filter_policy == "nms"
    assert args.structured_vp_max_boxes_per_label == 1
    assert args.structured_vp_max_total_boxes == 2
    assert args.structured_vp_nms_iou_threshold == 0.6
    assert args.structured_vp_allowed_labels == "cat,dog"


def test_cli_parser_accepts_visual_primitive_convert_commands():
    parser = create_parser()

    args = parser.parse_args([
        "convert",
        "vp-coco-od",
        "--json-file",
        "annotations.json",
        "--images-dir",
        "images",
        "--output",
        "od_vp.jsonl",
    ])

    assert args.command == "convert"
    assert args.convert_type == "vp-coco-od"
    assert args.task_type == "OD_VP"
    assert args.marker_style == "special"

    args = parser.parse_args([
        "convert",
        "vp-yolo-count",
        "--labels-dir",
        "labels",
        "--images-dir",
        "images",
        "--classes-file",
        "classes.txt",
        "--output",
        "count_vp.jsonl",
        "--box-format",
        "loc_tokens",
        "--marker-style",
        "plain",
    ])

    assert args.command == "convert"
    assert args.convert_type == "vp-yolo-count"
    assert args.task_type == "COUNT_VP"
    assert args.box_format == "loc_tokens"
    assert args.marker_style == "plain"

    args = parser.parse_args([
        "convert",
        "vp-jsonl-grounding",
        "--input",
        "od_vp.jsonl",
        "--output",
        "query_vp.jsonl",
        "--task-type",
        "OPEN_VOCABULARY_DETECTION",
    ])

    assert args.command == "convert"
    assert args.convert_type == "vp-jsonl-grounding"
    assert args.task_type == "OPEN_VOCABULARY_DETECTION"
    assert args.box_format == "loc_tokens"
    assert args.marker_style == "plain"


def test_run_data_conversion_dispatches_visual_primitive_converter(monkeypatch):
    calls = {}

    def fake_coco_to_vp_od(**kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "florence_forge.data.vp_converter.VisualPrimitiveConverter.coco_to_vp_od",
        fake_coco_to_vp_od,
    )

    result = run_data_conversion(SimpleNamespace(
        convert_type="vp-coco-od",
        json_file="annotations.json",
        output="od_vp.jsonl",
        images_dir="images",
        task_type="OD_VP",
        marker_style="plain",
    ))

    assert result is True
    assert calls["kwargs"] == {
        "coco_json_path": "annotations.json",
        "output_path": "od_vp.jsonl",
        "image_dir": "images",
        "task_type": "OD_VP",
        "box_format": "json",
        "marker_style": "plain",
    }


def test_run_data_conversion_dispatches_yolo_count_converter(monkeypatch):
    calls = {}

    def fake_yolo_to_vp_counting(**kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "florence_forge.data.vp_converter.VisualPrimitiveConverter.yolo_to_vp_counting",
        fake_yolo_to_vp_counting,
    )

    result = run_data_conversion(SimpleNamespace(
        convert_type="vp-yolo-count",
        labels_dir="labels",
        output="count_vp.jsonl",
        images_dir="images",
        classes_file="classes.txt",
        image_ext=".jpg",
        task_type="COUNT_VP",
        box_format="loc_tokens",
        marker_style="plain",
    ))

    assert result is True
    assert calls["kwargs"] == {
        "yolo_labels_dir": "labels",
        "output_path": "count_vp.jsonl",
        "image_dir": "images",
        "classes_file": "classes.txt",
        "image_ext": ".jpg",
        "task_type": "COUNT_VP",
        "box_format": "loc_tokens",
        "marker_style": "plain",
    }


def test_run_data_conversion_dispatches_jsonl_grounding_converter(monkeypatch):
    calls = {}

    def fake_jsonl_to_grounding(**kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "florence_forge.data.vp_converter.VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding",
        fake_jsonl_to_grounding,
    )

    result = run_data_conversion(SimpleNamespace(
        convert_type="vp-jsonl-grounding",
        input="od_vp.jsonl",
        output="query_vp.jsonl",
        task_type="PHRASE_GROUNDING_VP",
        box_format="loc_tokens",
        marker_style="plain",
    ))

    assert result is True
    assert calls["kwargs"] == {
        "input_path": "od_vp.jsonl",
        "output_path": "query_vp.jsonl",
        "task_type": "PHRASE_GROUNDING_VP",
        "box_format": "loc_tokens",
        "marker_style": "plain",
    }


def test_convert_visual_primitives_script_parser_accepts_modes():
    script_path = Path("scripts/data-conversion/convert_visual_primitives.py")
    spec = importlib.util.spec_from_file_location("convert_visual_primitives", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.create_parser()
    args = parser.parse_args([
        "yolo-od",
        "--labels-dir",
        "labels",
        "--images-dir",
        "images",
        "--classes-file",
        "classes.txt",
        "--output",
        "od_vp.jsonl",
    ])

    assert args.mode == "yolo-od"
    assert args.task_type == "OD_VP"
    assert args.marker_style == "special"

    args = parser.parse_args([
        "jsonl-grounding",
        "--input",
        "od_vp.jsonl",
        "--output",
        "query_vp.jsonl",
    ])

    assert args.mode == "jsonl-grounding"
    assert args.task_type == "PHRASE_GROUNDING_VP"
    assert args.box_format == "loc_tokens"

    args = parser.parse_args([
        "yolo-count",
        "--labels-dir",
        "labels",
        "--images-dir",
        "images",
        "--classes-file",
        "classes.txt",
        "--output",
        "count_vp.jsonl",
        "--box-format",
        "loc_tokens",
        "--marker-style",
        "plain",
    ])

    assert args.mode == "yolo-count"
    assert args.task_type == "COUNT_VP"
    assert args.box_format == "loc_tokens"
    assert args.marker_style == "plain"


def test_real_florence_vp_training_smoke_parser_and_coco80(tmp_path):
    script_path = Path("scripts/smoke/real_florence_vp_training_smoke.py")
    spec = importlib.util.spec_from_file_location("real_florence_vp_training_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.parse_args([
        "--dataset-root",
        "dataset",
        "--model-path",
        "model",
        "--max-train-samples",
        "2",
        "--max-val-samples",
        "1",
        "--training-data-order",
        "grounding-first",
        "--shuffle-train-data",
        "--shuffle-seed",
        "7",
        "--training-mode",
        "lora",
        "--lora-target-modules",
        "q_proj",
        "v_proj",
        "--save-adapter",
        "--train-vp-head",
        "--include-count",
        "--include-grounding",
        "--grounding-task-type",
        "OPEN_VOCABULARY_DETECTION",
        "--grounding-curriculum",
        "multi-instance",
        "--grounding-min-query-boxes",
        "4",
        "--grounding-selection",
        "multi-instance",
        "--grounding-train-path",
        str(tmp_path / "train_grounding.jsonl"),
        "--grounding-val-path",
        str(tmp_path / "val_grounding.jsonl"),
        "--grounding-count-hint-template",
        "{label} | count={query_box_count}",
        "--grounding-count-hint-splits",
        "train",
        "--vp-marker-style",
        "plain",
    ])

    assert args.dataset_root == "dataset"
    assert args.model_path == "model"
    assert args.vp_marker_style == "plain"
    assert args.max_train_samples == 2
    assert args.max_val_samples == 1
    assert args.training_data_order == "grounding-first"
    assert args.shuffle_train_data is True
    assert args.shuffle_seed == 7
    assert args.training_mode == "lora"
    assert args.lora_target_modules == ["q_proj", "v_proj"]
    assert args.lora_modules_to_save == ["lm_head", "model.shared"]
    assert args.save_adapter is True
    assert args.include_count is True
    assert args.include_grounding is True
    assert args.grounding_train_path == str(tmp_path / "train_grounding.jsonl")
    assert args.grounding_val_path == str(tmp_path / "val_grounding.jsonl")
    assert args.grounding_selection == "multi-instance"
    assert args.grounding_task_type == "OPEN_VOCABULARY_DETECTION"
    assert args.grounding_curriculum == "multi-instance"
    assert args.grounding_single_weight == 1
    assert args.grounding_medium_weight == 2
    assert args.grounding_dense_weight == 3
    assert args.grounding_min_query_boxes == 4
    assert args.grounding_max_query_boxes is None
    assert args.grounding_count_hint_template == "{label} | count={query_box_count}"
    assert args.grounding_count_hint_splits == "train"
    assert "q_proj" in module.DEFAULT_LORA_TARGET_MODULES
    curriculum_rows, curriculum_summary = module._build_query_curriculum_rows(
        [
            {"query_box_count": 1, "query_label": "cat"},
            {"query_box_count": 2, "query_label": "dog"},
            {"query_box_count": 4, "query_label": "person"},
        ],
        single_weight=1,
        medium_weight=2,
        dense_weight=3,
    )
    assert len(curriculum_rows) == 6
    assert curriculum_summary["bucket_counts"] == {"single": 1, "medium": 1, "dense": 1}
    assert curriculum_summary["bucket_output_counts"] == {"single": 1, "medium": 2, "dense": 3}
    selected_rows = module._select_query_grounding_rows(
        [
            {"query_box_count": 1, "query_label": "cat", "image": "a.jpg", "suffix": "one"},
            {"query_box_count": 5, "query_label": "person", "image": "b.jpg", "suffix": "dense"},
            {"query_box_count": 2, "query_label": "dog", "image": "c.jpg", "suffix": "two"},
        ],
        total=2,
        selection="multi-instance",
    )
    assert [row["query_label"] for row in selected_rows] == ["person", "dog"]
    ordered_configs = module._order_data_configs(
        [
            {"task_type": "OD_VP", "data_path": "od.jsonl"},
            {"task_type": "OPEN_VOCABULARY_DETECTION", "data_path": "grounding.jsonl"},
        ],
        training_data_order="grounding-first",
        grounding_task_type="OPEN_VOCABULARY_DETECTION",
    )
    assert [config["task_type"] for config in ordered_configs] == ["OPEN_VOCABULARY_DETECTION", "OD_VP"]
    filtered_rows, filter_summary = module._filter_query_grounding_rows(
        [
            {"query_box_count": 1, "query_label": "cat"},
            {"query_box_count": 4, "query_label": "person"},
            {"query_box_count": 7, "query_label": "car"},
        ],
        min_query_boxes=4,
        max_query_boxes=None,
    )
    assert [row["query_label"] for row in filtered_rows] == ["person", "car"]
    assert filter_summary["input_rows"] == 3
    assert filter_summary["output_rows"] == 2
    hinted_rows, hinted_summary = module._apply_query_count_hints(
        [
            {
                "query_label": "person",
                "text_input": "person",
                "query_box_count": 4,
            },
            {
                "query_label": "traffic light",
                "text_input": "traffic light",
                "query_box_count": 7,
            },
        ],
        template="{label} | count={query_box_count}",
    )
    assert hinted_rows[0]["text_input"] == "person | count=4"
    assert hinted_rows[0]["query_label"] == "person"
    assert hinted_rows[0]["count_hint_original_text_input"] == "person"
    assert hinted_rows[1]["text_input"] == "traffic light | count=7"
    assert hinted_summary["changed_text_input_rows"] == 2
    assert hinted_summary["stats"]["bucket_counts"]["dense"] == 2

    train_grounding_path = tmp_path / "train_grounding_rows.jsonl"
    val_grounding_path = tmp_path / "val_grounding_rows.jsonl"
    module._write_jsonl(
        train_grounding_path,
        [{"query_label": "person", "text_input": "person", "query_box_count": 4}],
    )
    module._write_jsonl(
        val_grounding_path,
        [{"query_label": "dog", "text_input": "dog", "query_box_count": 3}],
    )
    effective_train_path, effective_val_path, split_hint_summary = module._apply_query_count_hints_for_splits(
        train_path=train_grounding_path,
        val_path=val_grounding_path,
        output_dir=tmp_path,
        template="{label} | count={query_box_count}",
        splits="train",
    )
    assert effective_train_path.name == "train_grounding_count_hint_vp.jsonl"
    assert effective_val_path == val_grounding_path
    assert module._read_jsonl(effective_train_path)[0]["text_input"] == "person | count=4"
    assert module._read_jsonl(effective_val_path)[0]["text_input"] == "dog"
    assert split_hint_summary["splits"] == "train"
    assert split_hint_summary["train"]["changed_text_input_rows"] == 1
    assert split_hint_summary["val"] is None

    assert "fc2" in module.DEFAULT_LORA_TARGET_MODULES
    assert module.DEFAULT_LORA_MODULES_TO_SAVE == ("lm_head", "model.shared")

    classes_path = module._write_coco80_classes(tmp_path / "coco80.names")
    classes = classes_path.read_text(encoding="utf-8").splitlines()
    assert classes[0] == "person"
    assert classes[79] == "toothbrush"


def test_visual_primitive_smoke_script_runs_end_to_end(tmp_path):
    script_path = Path("scripts/smoke/visual_primitive_mvp_smoke.py")
    spec = importlib.util.spec_from_file_location("visual_primitive_mvp_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = module.run_smoke(tmp_path / "vp_smoke")

    assert report["ok"] is True
    assert report["checks"]["od_rows"] == 1
    assert report["checks"]["count_rows"] == 1
    assert report["checks"]["od_detections"] == 2
    assert report["checks"]["vp_format_valid_ratio"] == 1.0


def test_visual_primitive_training_shell_exists():
    script_path = Path("scripts/training/lora/training_visual_primitive_lora.sh")

    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    assert "--task visual_primitive" in content
    assert "visual_primitive_training.yaml" in content
