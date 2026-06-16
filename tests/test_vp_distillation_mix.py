import json
import subprocess
import sys
from pathlib import Path


def test_build_vp_distillation_mix_repeats_and_replaces(tmp_path):
    base_path = tmp_path / "base.jsonl"
    distill_path = tmp_path / "distill.jsonl"
    base_rows = [
        {
            "image": "cat.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
            "query_label": "cat",
            "text_input": "cat",
            "query_box_count": 1,
        },
        {
            "image": "dog.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>dog</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
            "query_label": "dog",
            "text_input": "dog",
            "query_box_count": 1,
        },
    ]
    distill_rows = [
        {
            "image": "cat.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": (
                "<ref>cat</ref> <box><loc_0><loc_0><loc_10><loc_10>"
                "<loc_50><loc_50><loc_60><loc_60></box>"
            ),
            "query_label": "cat",
            "text_input": "cat",
            "query_box_count": 2,
            "distillation_delta_tp": 1,
            "distillation_target_mode": "reference",
        }
    ]
    base_path.write_text("\n".join(json.dumps(row) for row in base_rows) + "\n", encoding="utf-8")
    distill_path.write_text("\n".join(json.dumps(row) for row in distill_rows) + "\n", encoding="utf-8")
    output_path = tmp_path / "mixed.jsonl"
    script = Path("scripts/data-conversion/build_vp_distillation_mix.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-input",
            str(base_path),
            "--distillation-input",
            str(distill_path),
            "--output",
            str(output_path),
            "--base-repeat",
            "1",
            "--distillation-repeat",
            "3",
            "--distillation-min-delta-tp",
            "1",
            "--distillation-target-mode",
            "reference",
            "--replace-base-on-distillation-key",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["base_input_rows"] == 2
    assert summary["distillation_input_rows"] == 1
    assert summary["skipped_base_replaced_rows"] == 1
    assert summary["base_output_rows"] == 1
    assert summary["distillation_output_rows"] == 3
    assert summary["distillation_output_ratio"] == 0.75
    assert len(rows) == 4
    assert [row["mix_group"] for row in rows] == ["base", "distillation", "distillation", "distillation"]
    assert rows[1]["mix_repeat_total"] == 3
    assert rows[1]["query_box_count"] == 2
    assert (tmp_path / "mixed_summary.json").exists()
    assert (tmp_path / "mixed_summary.md").exists()


def test_build_vp_distillation_mix_placement_controls_order(tmp_path):
    base_path = tmp_path / "base.jsonl"
    distill_path = tmp_path / "distill.jsonl"
    base_rows = [
        {
            "image": f"base_{index}.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>item</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
            "query_label": "item",
            "text_input": "item",
            "query_box_count": 1,
        }
        for index in range(3)
    ]
    distill_rows = [
        {
            "image": "hard.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": (
                "<ref>item</ref> <box><loc_0><loc_0><loc_10><loc_10>"
                "<loc_50><loc_50><loc_60><loc_60></box>"
            ),
            "query_label": "item",
            "text_input": "item",
            "query_box_count": 2,
            "distillation_delta_tp": 1,
            "distillation_target_mode": "reference",
        }
    ]
    base_path.write_text("\n".join(json.dumps(row) for row in base_rows) + "\n", encoding="utf-8")
    distill_path.write_text("\n".join(json.dumps(row) for row in distill_rows) + "\n", encoding="utf-8")
    script = Path("scripts/data-conversion/build_vp_distillation_mix.py")

    prepend_path = tmp_path / "prepend.jsonl"
    prepend = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-input",
            str(base_path),
            "--distillation-input",
            str(distill_path),
            "--output",
            str(prepend_path),
            "--distillation-repeat",
            "2",
            "--placement",
            "prepend",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert prepend.returncode == 0, prepend.stderr
    prepend_rows = [json.loads(line) for line in prepend_path.read_text(encoding="utf-8").splitlines()]
    assert [row["mix_group"] for row in prepend_rows] == [
        "distillation",
        "distillation",
        "base",
        "base",
        "base",
    ]
    assert json.loads(prepend.stdout)["placement"] == "prepend"

    interleave_path = tmp_path / "interleave.jsonl"
    interleave = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-input",
            str(base_path),
            "--distillation-input",
            str(distill_path),
            "--output",
            str(interleave_path),
            "--distillation-repeat",
            "2",
            "--placement",
            "interleave",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert interleave.returncode == 0, interleave.stderr
    interleave_rows = [json.loads(line) for line in interleave_path.read_text(encoding="utf-8").splitlines()]
    assert [row["mix_group"] for row in interleave_rows] == [
        "distillation",
        "base",
        "distillation",
        "base",
        "base",
    ]
    assert json.loads(interleave.stdout)["placement"] == "interleave"


def test_build_vp_distillation_mix_round_robin_repeats_distillation_rows(tmp_path):
    base_path = tmp_path / "base.jsonl"
    distill_path = tmp_path / "distill.jsonl"
    base_rows = [
        {
            "image": "base.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>item</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
            "query_label": "item",
            "text_input": "item",
            "query_box_count": 1,
        }
    ]
    distill_rows = [
        {
            "image": f"hard_{index}.jpg",
            "prefix": "<OPEN_VOCABULARY_DETECTION>",
            "suffix": "<ref>item</ref> <box><loc_0><loc_0><loc_10><loc_10></box>",
            "query_label": f"hard-{index}",
            "text_input": f"hard-{index}",
            "query_box_count": 2,
            "distillation_delta_tp": 1,
            "distillation_target_mode": "reference",
        }
        for index in range(2)
    ]
    base_path.write_text("\n".join(json.dumps(row) for row in base_rows) + "\n", encoding="utf-8")
    distill_path.write_text("\n".join(json.dumps(row) for row in distill_rows) + "\n", encoding="utf-8")
    output_path = tmp_path / "round_robin.jsonl"
    script = Path("scripts/data-conversion/build_vp_distillation_mix.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-input",
            str(base_path),
            "--distillation-input",
            str(distill_path),
            "--output",
            str(output_path),
            "--distillation-repeat",
            "3",
            "--distillation-repeat-order",
            "round_robin",
            "--placement",
            "prepend",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["mix_group"] for row in rows] == [
        "distillation",
        "distillation",
        "distillation",
        "distillation",
        "distillation",
        "distillation",
        "base",
    ]
    assert [row["mix_key"] for row in rows[:6]] == [
        "hard_0.jpg|hard-0",
        "hard_1.jpg|hard-1",
        "hard_0.jpg|hard-0",
        "hard_1.jpg|hard-1",
        "hard_0.jpg|hard-0",
        "hard_1.jpg|hard-1",
    ]
    summary = json.loads(result.stdout)
    assert summary["distillation_repeat_order"] == "round_robin"
    assert summary["placement"] == "prepend"
