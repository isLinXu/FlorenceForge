"""Tests for the YAML-based multi-dataset configuration entry points."""

from pathlib import Path

import pytest

from florence_forge.core.yaml_config import (
    FlorenceForgeYAMLConfig,
    YAMLDatasetConfig,
    YAMLTaskMapping,
    create_yaml_config_template,
    validate_yaml_config,
)


def _valid_config(tmp_path: Path) -> FlorenceForgeYAMLConfig:
    caption_path = tmp_path / "captions"
    detection_path = tmp_path / "detection"
    caption_path.mkdir()
    detection_path.mkdir()
    return FlorenceForgeYAMLConfig.from_dict(
        {
            "project_name": "yaml-regression",
            "experiment_name": "caption-od",
            "output_dir": str(tmp_path / "outputs"),
            "image_base_path": str(tmp_path),
            "training": {
                "num_epochs": 2,
                "data_config": {"batch_size": 2, "num_workers": 0},
            },
            "datasets": [
                {
                    "name": "caption_set",
                    "path": str(caption_path),
                    "task_types": ["CAPTION"],
                    "format": "jsonl",
                    "max_samples": 12,
                },
                {
                    "name": "detection_set",
                    "path": str(detection_path),
                    "task_types": ["OD"],
                },
            ],
            "task_mappings": [
                {"task_type": "CAPTION", "datasets": ["caption_set"]},
                {
                    "task_type": "OD",
                    "datasets": ["detection_set"],
                    "weights": {"detection_set": 0.7},
                },
            ],
        }
    )


def test_config_infers_tasks_validates_and_builds_training_config(tmp_path):
    config = _valid_config(tmp_path)

    assert config.enabled_tasks == ["CAPTION", "OD"]
    assert config.validate() == {"errors": [], "warnings": []}

    training = config.to_training_config()
    assert training.num_epochs == 2
    assert training.data_settings.batch_size == 2
    assert training.output_dir == str(tmp_path / "outputs")
    assert training.experiment_name == "caption-od"

    dataset_info = config.datasets[0].to_dataset_info()
    assert dataset_info.name == "caption_set"
    assert dataset_info.max_samples == 12


def test_validation_reports_cross_reference_and_task_errors(tmp_path):
    missing_path = tmp_path / "missing"
    config = FlorenceForgeYAMLConfig(
        enabled_tasks=["CAPTION", "NOT_A_TASK"],
        datasets=[
            YAMLDatasetConfig("set_a", str(missing_path), ["OD"]),
            YAMLDatasetConfig("set_a", str(missing_path), ["INVALID"]),
        ],
        task_mappings=[
            YAMLTaskMapping("CAPTION", ["set_a"]),
            YAMLTaskMapping("CAPTION", ["unknown"]),
        ],
    )

    result = config.validate()

    assert any("无效的任务类型: NOT_A_TASK" in error for error in result["errors"])
    assert any("重复的数据集名称: set_a" in error for error in result["errors"])
    assert any("包含无效的任务类型: INVALID" in error for error in result["errors"])
    assert any("重复的任务映射: CAPTION" in error for error in result["errors"])
    assert any("引用了不存在的数据集: unknown" in error for error in result["errors"])
    assert any("不支持任务类型 CAPTION" in error for error in result["errors"])
    assert any("数据集路径不存在" in warning for warning in result["warnings"])
    assert any("NOT_A_TASK 没有对应的数据集映射" in warning for warning in result["warnings"])


def test_multi_dataset_manager_receives_registrations_and_default_weights(tmp_path):
    config = _valid_config(tmp_path)

    manager = config.to_multi_dataset_manager()

    assert set(manager.datasets) == {"caption_set", "detection_set"}
    assert manager.config.batch_size == 2
    assert manager.task_mappings["CAPTION"].weights == {"caption_set": 1.0}
    assert manager.task_mappings["OD"].weights == {"detection_set": 0.7}
    assert config.task_mappings[0].weights == {"caption_set": 1.0}


def test_yaml_json_and_default_file_roundtrips(tmp_path):
    config = _valid_config(tmp_path)
    yaml_path = tmp_path / "nested" / "config.yaml"
    json_path = tmp_path / "config.json"
    default_path = tmp_path / "default_config"

    config.save_to_yaml(yaml_path)
    config.save_to_json(json_path)
    config.save_to_file(default_path)

    loaded_yaml = FlorenceForgeYAMLConfig.load_from_file(yaml_path)
    loaded_json = FlorenceForgeYAMLConfig.load_from_file(json_path)
    loaded_default = FlorenceForgeYAMLConfig.load_from_yaml(default_path.with_suffix(".yaml"))

    assert loaded_yaml.to_dict() == config.to_dict()
    assert loaded_json.to_dict() == config.to_dict()
    assert loaded_default.project_name == "yaml-regression"

    with pytest.raises(ValueError, match="不支持的文件格式"):
        FlorenceForgeYAMLConfig.load_from_file(tmp_path / "config.toml")


def test_template_creation_and_validation_cli_helpers(tmp_path, capsys):
    template_path = tmp_path / "template.yaml"
    create_yaml_config_template(template_path)

    assert template_path.exists()
    assert validate_yaml_config(template_path) is True
    output = capsys.readouterr().out
    assert "YAML配置模板已创建" in output
    assert "配置验证警告" in output
    assert "配置验证通过" in output

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("enabled_tasks:\n  - INVALID\n", encoding="utf-8")
    assert validate_yaml_config(invalid_path) is False
    assert "配置验证失败" in capsys.readouterr().out


def test_default_training_config_and_validation_load_failure(tmp_path):
    config = FlorenceForgeYAMLConfig()

    assert config.to_training_config().output_dir == "./outputs"
    assert validate_yaml_config(tmp_path / "does_not_exist.yaml") is False
