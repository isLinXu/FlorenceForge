"""Regression tests for the standalone configuration management CLI."""

import json

from florence_forge.cli.config_manager import ConfigManager, create_parser
from florence_forge.core.config import TrainingConfig
from florence_forge.core.yaml_config import FlorenceForgeYAMLConfig


def test_create_validate_convert_merge_and_show_training_config(tmp_path, capsys):
    manager = ConfigManager()
    base_path = tmp_path / "base"
    converted_path = tmp_path / "converted.json"
    merged_path = tmp_path / "merged.yaml"
    override_path = tmp_path / "override.json"

    manager.create_default_config(str(base_path), format_type="yaml")
    base_yaml = base_path.with_suffix(".yaml")
    assert manager.validate_config(str(base_yaml)) is True

    manager.convert_config(str(base_yaml), str(converted_path))
    loaded = TrainingConfig.load_from_json(converted_path)
    assert loaded.experiment_name == "florence2_default_experiment"

    override_path.write_text(
        json.dumps(
            {
                "_metadata": {"ignored": True},
                "num_epochs": 4,
                "data_config": {"batch_size": 3},
            }
        ),
        encoding="utf-8",
    )
    manager.merge_configs(str(base_yaml), str(override_path), str(merged_path))
    merged = TrainingConfig.load_from_yaml(merged_path)
    assert merged.num_epochs == 4
    assert merged.data_settings.batch_size == 3

    manager.show_config_info(str(merged_path))
    output = capsys.readouterr().out
    assert "默认配置已创建" in output
    assert "配置转换完成" in output
    assert "配置合并完成" in output
    assert "训练轮数: 4" in output


def test_validate_training_config_failure_paths(tmp_path, capsys):
    manager = ConfigManager()
    unsupported = tmp_path / "config.txt"
    unsupported.write_text("irrelevant", encoding="utf-8")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("num_epochs: not-an-integer\n", encoding="utf-8")

    assert manager.validate_config(str(tmp_path / "missing.yaml")) is False
    assert manager.validate_config(str(unsupported)) is False
    assert manager.validate_config(str(malformed)) is False

    output = capsys.readouterr().out
    assert "配置文件不存在" in output
    assert "不支持的文件格式" in output
    assert "配置验证失败" in output


def test_templates_yaml_info_and_supported_task_listing(tmp_path, capsys):
    manager = ConfigManager()
    for template_type in ("minimal", "lora", "production", "full"):
        path = tmp_path / f"{template_type}.yaml"
        manager.create_template(template_type, str(path))
        assert TrainingConfig.load_from_yaml(path)

    multitask_path = tmp_path / "multitask.yaml"
    manager.create_template("yaml_multitask", str(multitask_path))
    assert manager.validate_yaml_config(str(multitask_path)) is True
    manager.show_yaml_config_info(str(multitask_path))
    manager.list_supported_tasks()

    output = capsys.readouterr().out
    assert "lora 模板已创建" in output
    assert "YAML多任务模板已创建" in output
    assert "YAML配置文件信息" in output
    assert "CAPTION" in output
    assert "OD" in output


def test_convert_legacy_training_config_to_nested_yaml_schema(tmp_path, capsys):
    manager = ConfigManager()
    source_path = tmp_path / "training.json"
    output_path = tmp_path / "training_multitask.yaml"
    source = TrainingConfig(
        num_epochs=7,
        experiment_name="converted",
        output_dir=str(tmp_path / "out"),
    )
    source.data_settings.batch_size = 6
    source.optimization_settings.learning_rate = 3e-5
    source.save_to_json(source_path)

    manager.convert_to_yaml_config(str(source_path), str(output_path))

    converted = FlorenceForgeYAMLConfig.load_from_yaml(output_path)
    training = converted.to_training_config()
    assert converted.experiment_name == "converted"
    assert training.num_epochs == 7
    assert training.data_settings.batch_size == 6
    assert training.optimization_settings.learning_rate == 3e-5
    assert "请手动添加数据集和任务映射配置" in capsys.readouterr().out


def test_parser_recognizes_yaml_commands():
    parser = create_parser()

    args = parser.parse_args(["template", "yaml_multitask", "config.yaml"])

    assert args.command == "template"
    assert args.template_type == "yaml_multitask"
