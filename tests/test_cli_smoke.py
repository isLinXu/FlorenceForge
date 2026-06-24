"""CLI 子命令端到端冒烟：解析器注册、--help 与轻量子进程。"""

import subprocess
import sys

import pytest

from florence_forge.cli.config_manager import create_parser as create_config_parser
from florence_forge.cli.main import create_parser as create_main_parser


@pytest.mark.parametrize(
    "argv,expected_command",
    [
        (["train", "--help"], "train"),
        (["convert", "--help"], "convert"),
        (["eval", "--help"], "eval"),
        (["infer", "--help"], "infer"),
        (["serve", "--help"], "serve"),
        (["doctor", "--help"], "doctor"),
        (["validate", "--help"], "validate"),
        (["generate-config", "--help"], "generate-config"),
        (["list-tasks"], "list-tasks"),
    ],
)
def test_main_cli_parser_smoke(argv, expected_command):
    parser = create_main_parser()
    if argv[-1] == "--help":
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 0
        return

    args = parser.parse_args(argv)
    assert args.command == expected_command


@pytest.mark.parametrize(
    "argv",
    [
        ["convert", "yolo", "--help"],
        ["convert", "coco", "--help"],
        ["convert", "coco-caption", "--help"],
        ["convert", "csv", "--help"],
        ["convert", "xml", "--help"],
        ["convert", "ocr", "--help"],
        ["convert", "ocr-txt", "--help"],
    ],
)
def test_convert_subcommand_help(argv):
    parser = create_main_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["create", "--help"],
        ["validate", "--help"],
        ["convert", "--help"],
        ["merge", "--help"],
        ["info", "--help"],
    ],
)
def test_config_manager_cli_parser_smoke(argv):
    parser = create_config_parser()
    if argv[-1] == "--help":
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 0
        return
    parser.parse_args(argv)


def test_main_module_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "florence_forge.cli.main", "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "Florence" in result.stdout


def test_list_tasks_subprocess_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "florence_forge.cli.main", "list-tasks"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0


def test_training_config_exposes_nested_fields():
    from florence_forge.core.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.num_workers == cfg.data_settings.num_workers
    assert cfg.weight_decay == cfg.optimization_settings.weight_decay
    assert cfg.use_lora == cfg.model_settings.use_lora


def test_doctor_help_subprocess_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "florence_forge.cli.main", "doctor", "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0


def test_resolve_image_base_path_from_jsonl_parent(tmp_path):
    from florence_forge.cli.commands import _resolve_image_base_path
    from florence_forge.core.config import TrainingConfig

    images = tmp_path / "images"
    images.mkdir()
    jsonl = tmp_path / "caption.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")

    config = TrainingConfig(train_data_path=str(jsonl))
    assert _resolve_image_base_path(config) == str(images)
