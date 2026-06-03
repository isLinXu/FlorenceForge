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


def test_train_defaults_trainer_version_v2():
    parser = create_main_parser()
    args = parser.parse_args(["train", "--task", "caption"])
    assert args.trainer_version == "v2"


def test_train_rejects_v1_trainer_version():
    parser = create_main_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--task", "caption", "--trainer-version", "v1"])


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
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "Florence" in result.stdout


def test_list_tasks_subprocess_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "florence_forge.cli.main", "list-tasks"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0


def test_doctor_help_subprocess_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "florence_forge.cli.main", "doctor", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
