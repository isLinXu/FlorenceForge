import json
import subprocess
import sys
from pathlib import Path


def test_vp_experiment_runner_dry_run_builds_adapter_baseline_and_audit(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--max-samples",
            "3",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["infer_adapter", "infer_baseline", "audit"]
    assert summary["adapter_dir"] == str(tmp_path / "adapter")
    assert summary["ok"] is True

    adapter_cmd = summary["commands"][0]["cmd"]
    baseline_cmd = summary["commands"][1]["cmd"]
    audit_cmd = summary["commands"][2]["cmd"]
    assert "--adapter-dir" in adapter_cmd
    assert "--adapter-dir" not in baseline_cmd
    assert "--baseline-summary" in audit_cmd
    assert "--structured-vp-decode" not in adapter_cmd
    assert "--structured-vp-mode" in adapter_cmd
    assert "on" in adapter_cmd
    assert "--max-samples" in adapter_cmd
    assert "3" in adapter_cmd

    written = json.loads((output_dir / "experiment_summary.json").read_text())
    assert written["dry_run"] is True
    assert written["executed"][0]["skipped"] is True


def test_vp_experiment_runner_dry_run_can_plan_training(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--training-steps",
            "8",
            "--vp-marker-style",
            "plain",
            "--structured-vp-marker-style",
            "plain",
            "--device",
            "cpu",
            "--skip-baseline",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["train_adapter", "infer_adapter", "audit"]
    train_cmd = summary["commands"][0]["cmd"]
    assert "--max-steps" in train_cmd
    assert "8" in train_cmd
    assert "--lora-task-type" in train_cmd
    assert "SEQ_2_SEQ_LM" in train_cmd
    assert "--train-vp-head" in train_cmd
    assert "--vp-box-format" in train_cmd
    assert "loc_tokens" in train_cmd
    assert "--vp-marker-style" in train_cmd
    assert "plain" in train_cmd
    assert "--include-count" not in train_cmd
    assert "--baseline-summary" not in summary["commands"][-1]["cmd"]
    assert summary["vp_marker_style"] == "plain"
    assert summary["structured_vp_marker_style"] == "plain"


def test_vp_experiment_runner_dry_run_can_override_lora_task_type(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--lora-task-type",
            "CAUSAL_LM",
            "--device",
            "cpu",
            "--skip-baseline",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    train_cmd = summary["commands"][0]["cmd"]
    assert "--lora-task-type" in train_cmd
    assert "CAUSAL_LM" in train_cmd


def test_vp_experiment_runner_dry_run_can_plan_count_curriculum_and_cleanup(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--include-count",
            "--cleanup-adapter-after-audit",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    train_cmd = summary["commands"][0]["cmd"]
    assert "--include-count" in train_cmd
    assert summary["cleanup_adapter_after_audit"] is True
    assert summary["adapter_dir"] == str(output_dir / "training" / "adapter")


def test_vp_experiment_runner_dry_run_can_plan_distillation_mix_before_training(tmp_path):
    output_dir = tmp_path / "experiment"
    base_path = tmp_path / "base_grounding.jsonl"
    distill_path = tmp_path / "hard_positive.jsonl"
    base_path.write_text("", encoding="utf-8")
    distill_path.write_text("", encoding="utf-8")
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-distillation-mix",
            "--run-training",
            "--skip-baseline",
            "--model-path",
            str(tmp_path / "model"),
            "--grounding-train-path",
            str(base_path),
            "--distillation-mix-input",
            str(distill_path),
            "--distillation-mix-repeat",
            "2",
            "--distillation-mix-repeat-order",
            "round_robin",
            "--distillation-mix-placement",
            "prepend",
            "--distillation-mix-min-delta-tp",
            "1",
            "--distillation-mix-target-mode",
            "reference",
            "--distillation-mix-replace-base-on-key",
            "--distillation-mix-shuffle",
            "--distillation-mix-seed",
            "7",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["build_distillation_mix", "train_adapter", "infer_adapter", "audit"]

    mix_output = output_dir / "distillation_mix" / "query_ovd_distill_mix.jsonl"
    mix_cmd = summary["commands"][0]["cmd"]
    train_cmd = summary["commands"][1]["cmd"]
    assert "scripts/data-conversion/build_vp_distillation_mix.py" in mix_cmd
    assert "--base-input" in mix_cmd
    assert str(base_path) in mix_cmd
    assert "--distillation-input" in mix_cmd
    assert str(distill_path) in mix_cmd
    assert "--output" in mix_cmd
    assert str(mix_output) in mix_cmd
    assert "--summary-output" in mix_cmd
    assert str(output_dir / "distillation_mix" / "query_ovd_distill_mix_summary.json") in mix_cmd
    assert "--distillation-repeat" in mix_cmd
    assert "2" in mix_cmd
    assert "--distillation-repeat-order" in mix_cmd
    assert "round_robin" in mix_cmd
    assert "--placement" in mix_cmd
    assert "prepend" in mix_cmd
    assert "--distillation-min-delta-tp" in mix_cmd
    assert "--distillation-target-mode" in mix_cmd
    assert "reference" in mix_cmd
    assert "--replace-base-on-distillation-key" in mix_cmd
    assert "--shuffle" in mix_cmd
    assert "--seed" in mix_cmd
    assert "7" in mix_cmd

    assert "--include-grounding" in train_cmd
    assert "--grounding-train-path" in train_cmd
    assert str(mix_output) in train_cmd
    assert str(base_path) not in train_cmd
    assert summary["include_grounding"] is True
    assert summary["grounding_train_path"] == str(mix_output)
    assert summary["distillation_mix_output"] == str(mix_output)
    assert summary["distillation_mix_base_inputs"] == [str(base_path)]
    assert summary["distillation_mix_inputs"] == [str(distill_path)]
    assert summary["distillation_mix_repeat"] == 2
    assert summary["distillation_mix_repeat_order"] == "round_robin"
    assert summary["distillation_mix_placement"] == "prepend"


def test_vp_experiment_runner_dry_run_can_plan_query_grounding_data_key(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--include-grounding",
            "--grounding-task-type",
            "OPEN_VOCABULARY_DETECTION",
            "--manifest-data-key",
            "val_grounding_path",
            "--inference-min-query-boxes",
            "4",
            "--inference-max-query-boxes",
            "8",
            "--visualization-limit",
            "5",
            "--structured-vp-allowed-labels-field",
            "text_input",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    train_cmd = summary["commands"][0]["cmd"]
    adapter_cmd = summary["commands"][1]["cmd"]
    assert "--include-grounding" in train_cmd
    assert "--grounding-task-type" in train_cmd
    assert "OPEN_VOCABULARY_DETECTION" in train_cmd
    assert "--data-key" in adapter_cmd
    assert "val_grounding_path" in adapter_cmd
    assert "--min-query-boxes" in adapter_cmd
    assert "4" in adapter_cmd
    assert "--max-query-boxes" in adapter_cmd
    assert "8" in adapter_cmd
    assert "--visualization-limit" in adapter_cmd
    assert "5" in adapter_cmd
    assert "--structured-vp-allowed-labels-field" in adapter_cmd
    assert "text_input" in adapter_cmd
    assert summary["manifest_data_key"] == "val_grounding_path"
    assert summary["inference_min_query_boxes"] == 4
    assert summary["inference_max_query_boxes"] == 8


def test_vp_experiment_runner_dry_run_can_apply_ovd_nms_query_preset(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--run-policy-sweep",
            "--query-grounding-preset",
            "ovd-nms",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--skip-baseline",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["train_adapter", "infer_adapter", "audit", "policy_sweep_adapter"]
    train_cmd = summary["commands"][0]["cmd"]
    adapter_cmd = summary["commands"][1]["cmd"]
    sweep_cmd = summary["commands"][3]["cmd"]
    assert "--include-grounding" in train_cmd
    assert "--grounding-task-type" in train_cmd
    assert "OPEN_VOCABULARY_DETECTION" in train_cmd
    assert "--grounding-selection" in train_cmd
    assert "multi-instance" in train_cmd
    assert "--training-data-order" in train_cmd
    assert "grounding-first" in train_cmd
    assert "--grounding-curriculum" in train_cmd
    assert "multi-instance" in train_cmd
    assert "--grounding-single-weight" in train_cmd
    assert "1" in train_cmd
    assert "--grounding-medium-weight" in train_cmd
    assert "2" in train_cmd
    assert "--grounding-dense-weight" in train_cmd
    assert "3" in train_cmd
    assert "--data-key" in adapter_cmd
    assert "val_grounding_path" in adapter_cmd
    assert "--structured-vp-filter-policy" in adapter_cmd
    assert "nms" in adapter_cmd
    assert "--structured-vp-nms-iou-threshold" in adapter_cmd
    assert "0.5" in adapter_cmd
    assert "--structured-vp-allowed-labels-field" in adapter_cmd
    assert "text_input" in adapter_cmd
    assert "--structured-vp-marker-style" in adapter_cmd
    assert "plain" in adapter_cmd
    assert "--visualization-limit" in adapter_cmd
    assert "5" in adapter_cmd
    assert "--max-new-tokens" in adapter_cmd
    assert "64" in adapter_cmd
    assert "--structured-vp-filter-policy" in sweep_cmd
    assert "nms" in sweep_cmd
    assert summary["query_grounding_preset"] == "ovd-nms"
    assert summary["include_grounding"] is True
    assert summary["grounding_task_type"] == "OPEN_VOCABULARY_DETECTION"
    assert summary["grounding_selection"] == "multi-instance"
    assert summary["grounding_curriculum"] == "multi-instance"
    assert summary["training_data_order"] == "grounding-first"
    assert summary["manifest_data_key"] == "val_grounding_path"


def test_vp_experiment_runner_dry_run_can_plan_count_hinted_ovd_preset(tmp_path):
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--query-grounding-preset",
            "ovd-nms",
            "--grounding-count-hint-template",
            "{label} | count={query_box_count}",
            "--grounding-count-hint-splits",
            "train",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--skip-baseline",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    train_cmd = summary["commands"][0]["cmd"]
    adapter_cmd = summary["commands"][1]["cmd"]
    assert "--grounding-count-hint-template" in train_cmd
    assert "{label} | count={query_box_count}" in train_cmd
    assert "--grounding-count-hint-splits" in train_cmd
    assert "train" in train_cmd
    assert "--structured-vp-allowed-labels-field" in adapter_cmd
    assert "query_label" in adapter_cmd
    assert "text_input" not in adapter_cmd
    assert summary["grounding_count_hint_template"] == "{label} | count={query_box_count}"
    assert summary["grounding_count_hint_splits"] == "train"
    assert summary["structured_vp_allowed_labels_field"] == "query_label"


def test_vp_experiment_runner_dry_run_can_plan_external_grounding_paths(tmp_path):
    output_dir = tmp_path / "experiment"
    train_grounding = tmp_path / "query_ovd_curriculum.jsonl"
    val_grounding = tmp_path / "query_ovd_val.jsonl"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-training",
            "--grounding-train-path",
            str(train_grounding),
            "--grounding-val-path",
            str(val_grounding),
            "--grounding-min-query-boxes",
            "4",
            "--skip-od-training-data",
            "--grounding-task-type",
            "OPEN_VOCABULARY_DETECTION",
            "--manifest-data-key",
            "val_grounding_path",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--skip-baseline",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    train_cmd = summary["commands"][0]["cmd"]
    adapter_cmd = summary["commands"][1]["cmd"]
    assert "--include-grounding" in train_cmd
    assert "--grounding-train-path" in train_cmd
    assert str(train_grounding) in train_cmd
    assert "--grounding-val-path" in train_cmd
    assert str(val_grounding) in train_cmd
    assert "--grounding-min-query-boxes" in train_cmd
    assert "4" in train_cmd
    assert "--skip-od-training-data" in train_cmd
    assert "--training-data-order" in train_cmd
    assert "as-is" in train_cmd
    assert "--data-key" in adapter_cmd
    assert "val_grounding_path" in adapter_cmd
    assert summary["include_grounding"] is True
    assert summary["grounding_train_path"] == str(train_grounding)
    assert summary["grounding_val_path"] == str(val_grounding)
    assert summary["grounding_min_query_boxes"] == 4
    assert summary["grounding_max_query_boxes"] is None
    assert summary["skip_od_training_data"] is True
    assert summary["training_data_order"] == "as-is"


def test_vp_experiment_runner_dry_run_can_plan_token_probe(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-token-probe",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--token-probe-max-samples",
            "2",
            "--vp-marker-style",
            "plain",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["infer_adapter", "infer_baseline", "audit", "token_probe"]
    token_probe_cmd = summary["commands"][-1]["cmd"]
    assert "scripts/experiments/probe_florence_vp_tokens.py" in token_probe_cmd
    assert "--adapter-dir" in token_probe_cmd
    assert "--marker-style" in token_probe_cmd
    assert "plain" in token_probe_cmd
    assert "--task-type" in token_probe_cmd
    assert "OD_VP" in token_probe_cmd
    assert "--max-samples" in token_probe_cmd
    assert "2" in token_probe_cmd
    assert summary["token_probe_summary"] == str(output_dir / "token_probe" / "florence_vp_token_probe.json")


def test_vp_experiment_runner_token_probe_targets_grounding_data_with_query_preset(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-token-probe",
            "--query-grounding-preset",
            "ovd-nms",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    token_probe_cmd = summary["commands"][-1]["cmd"]
    assert "--task-type" in token_probe_cmd
    assert "OPEN_VOCABULARY_DETECTION" in token_probe_cmd
    assert "--manifest-data-key" in token_probe_cmd
    assert "train_grounding_effective_path,train_grounding_path" in token_probe_cmd


def test_vp_experiment_runner_dry_run_can_plan_decoder_prefix_diagnostic(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")
    prefix = "<ref>{label}</ref> <box>"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--decoder-prefix",
            prefix,
            "--vp-marker-style",
            "plain",
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-mode",
            "auto",
            "--structured-vp-filter-policy",
            "auto",
            "--structured-vp-max-boxes-per-label",
            "1",
            "--structured-vp-max-total-boxes",
            "2",
            "--structured-vp-max-total-boxes-field",
            "query_box_count",
            "--length-penalty",
            "1.2",
            "--repetition-penalty",
            "1.1",
            "--no-repeat-ngram-size",
            "3",
            "--early-stopping",
            "--stop-after-vp-max-total-boxes",
            "--continue-underfilled-vp-boxes",
            "--structured-vp-repair-malformed-tail",
            "--vp-continuation-max-rounds",
            "2",
            "--vp-continuation-max-new-tokens",
            "32",
            "--vp-continuation-min-missing-boxes",
            "1",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    adapter_cmd = summary["commands"][0]["cmd"]
    baseline_cmd = summary["commands"][1]["cmd"]
    assert "--decoder-prefix" in adapter_cmd
    assert prefix in adapter_cmd
    assert "--decoder-prefix" in baseline_cmd
    assert prefix in baseline_cmd
    assert "--structured-vp-max-boxes-per-label" in adapter_cmd
    assert "1" in adapter_cmd
    assert "--structured-vp-max-total-boxes" in adapter_cmd
    assert "2" in adapter_cmd
    assert "--structured-vp-max-total-boxes-field" in adapter_cmd
    assert "query_box_count" in adapter_cmd
    assert "--structured-vp-filter-policy" in adapter_cmd
    assert "auto" in adapter_cmd
    assert "--length-penalty" in adapter_cmd
    assert "1.2" in adapter_cmd
    assert "--repetition-penalty" in baseline_cmd
    assert "1.1" in baseline_cmd
    assert "--no-repeat-ngram-size" in adapter_cmd
    assert "3" in adapter_cmd
    assert "--early-stopping" in adapter_cmd
    assert "--stop-after-vp-max-total-boxes" in adapter_cmd
    assert "--stop-after-vp-max-total-boxes" in baseline_cmd
    assert "--continue-underfilled-vp-boxes" in adapter_cmd
    assert "--continue-underfilled-vp-boxes" in baseline_cmd
    assert "--structured-vp-repair-malformed-tail" in adapter_cmd
    assert "--structured-vp-repair-malformed-tail" in baseline_cmd
    assert "--vp-continuation-max-rounds" in adapter_cmd
    assert "2" in adapter_cmd
    assert "--vp-continuation-max-new-tokens" in adapter_cmd
    assert "32" in adapter_cmd
    assert summary["decoder_prefix"] == prefix
    assert summary["structured_vp_mode"] == "auto"
    assert summary["structured_vp_filter_policy"] == "auto"
    assert summary["structured_vp_max_boxes_per_label"] == 1
    assert summary["structured_vp_max_total_boxes"] == 2
    assert summary["structured_vp_max_total_boxes_field"] == "query_box_count"
    assert summary["length_penalty"] == 1.2
    assert summary["repetition_penalty"] == 1.1
    assert summary["no_repeat_ngram_size"] == 3
    assert summary["early_stopping"] is True
    assert summary["stop_after_vp_max_total_boxes"] is True
    assert summary["continue_underfilled_vp_boxes"] is True
    assert summary["structured_vp_repair_malformed_tail"] is True
    assert summary["vp_continuation_max_rounds"] == 2
    assert summary["vp_continuation_max_new_tokens"] == 32
    assert summary["vp_continuation_min_missing_boxes"] == 1


def test_vp_experiment_runner_dry_run_can_plan_quality_eval(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-quality-eval",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-filter-policy",
            "nms",
            "--structured-vp-nms-iou-threshold",
            "0.6",
            "--structured-vp-allowed-labels",
            "cat,dog",
            "--structured-vp-allowed-labels-field",
            "text_input",
            "--structured-vp-max-total-boxes-field",
            "query_box_count",
            "--quality-iou-threshold",
            "0.75",
            "--quality-max-bad-cases",
            "3",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == ["infer_adapter", "infer_baseline", "audit", "quality_adapter", "quality_baseline"]
    adapter_quality_cmd = summary["commands"][3]["cmd"]
    baseline_quality_cmd = summary["commands"][4]["cmd"]
    assert "scripts/experiments/evaluate_vp_detection_quality.py" in adapter_quality_cmd
    assert "--structured-vp-filter-policy" in adapter_quality_cmd
    assert "nms" in adapter_quality_cmd
    assert "--structured-vp-nms-iou-threshold" in adapter_quality_cmd
    assert "0.6" in adapter_quality_cmd
    assert "--structured-vp-allowed-labels" in adapter_quality_cmd
    assert "cat,dog" in adapter_quality_cmd
    assert "--structured-vp-allowed-labels-field" in adapter_quality_cmd
    assert "text_input" in adapter_quality_cmd
    assert "--structured-vp-max-total-boxes-field" in adapter_quality_cmd
    assert "query_box_count" in adapter_quality_cmd
    assert "--iou-threshold" in adapter_quality_cmd
    assert "0.75" in adapter_quality_cmd
    assert "--max-bad-cases" in baseline_quality_cmd
    assert "3" in baseline_quality_cmd
    assert summary["quality_summaries"] == {
        "adapter": str(output_dir / "quality" / "adapter" / "vp_detection_quality.json"),
        "baseline": str(output_dir / "quality" / "baseline" / "vp_detection_quality.json"),
    }
    assert summary["structured_vp_allowed_labels"] == "cat,dog"
    assert summary["structured_vp_allowed_labels_field"] == "text_input"
    assert summary["structured_vp_max_total_boxes_field"] == "query_box_count"


def test_vp_experiment_runner_dry_run_can_plan_policy_comparison(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-quality-eval",
            "--run-policy-comparison",
            "--policy-comparison-focus-bucket",
            "dense",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == [
        "infer_adapter",
        "infer_baseline",
        "audit",
        "quality_adapter",
        "quality_baseline",
        "policy_comparison",
    ]
    compare_cmd = summary["commands"][-1]["cmd"]
    assert "scripts/experiments/compare_vp_quality_policies.py" in compare_cmd
    assert "--report" in compare_cmd
    assert "--focus-bucket" in compare_cmd
    assert "dense" in compare_cmd
    assert f"adapter={output_dir / 'quality' / 'adapter' / 'vp_detection_quality.json'}" in compare_cmd
    assert f"baseline={output_dir / 'quality' / 'baseline' / 'vp_detection_quality.json'}" in compare_cmd
    assert summary["policy_comparison_summary"] == str(
        output_dir / "policy_comparison" / "vp_policy_comparison.json"
    )
    assert summary["policy_comparison_focus_bucket"] == "dense"


def test_vp_experiment_runner_dry_run_can_plan_target_count_gap_analysis(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-quality-eval",
            "--run-target-count-gap-analysis",
            "--target-count-gap-focus-bucket",
            "dense",
            "--target-count-gap-max-rows",
            "7",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == [
        "infer_adapter",
        "infer_baseline",
        "audit",
        "quality_adapter",
        "quality_baseline",
        "target_count_gap_adapter",
        "target_count_gap_baseline",
    ]
    gap_cmd = summary["commands"][-2]["cmd"]
    assert "scripts/experiments/analyze_vp_target_count_gap.py" in gap_cmd
    assert "--focus-bucket" in gap_cmd
    assert "dense" in gap_cmd
    assert "--max-rows" in gap_cmd
    assert "7" in gap_cmd
    assert str(output_dir / "quality" / "adapter" / "vp_detection_quality.json") in gap_cmd
    assert summary["target_count_gap_summaries"] == {
        "adapter": str(output_dir / "target_count_gap" / "adapter" / "vp_target_count_gap.json"),
        "baseline": str(output_dir / "target_count_gap" / "baseline" / "vp_target_count_gap.json"),
    }
    assert summary["target_count_gap_focus_bucket"] == "dense"


def test_vp_experiment_runner_dry_run_can_plan_record_comparison(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-quality-eval",
            "--run-record-comparison",
            "--record-comparison-focus-bucket",
            "dense",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == [
        "infer_adapter",
        "infer_baseline",
        "audit",
        "quality_adapter",
        "quality_baseline",
        "record_comparison",
    ]
    compare_cmd = summary["commands"][-1]["cmd"]
    assert "scripts/experiments/compare_vp_quality_records.py" in compare_cmd
    assert "--candidate-report" in compare_cmd
    assert str(output_dir / "quality" / "adapter" / "vp_detection_quality.json") in compare_cmd
    assert "--baseline-report" in compare_cmd
    assert str(output_dir / "quality" / "baseline" / "vp_detection_quality.json") in compare_cmd
    assert "--focus-bucket" in compare_cmd
    assert "dense" in compare_cmd
    assert summary["record_comparison_summary"] == str(
        output_dir / "record_comparison" / "vp_record_comparison.json"
    )
    assert summary["record_comparison_focus_bucket"] == "dense"


def test_vp_experiment_runner_dry_run_can_plan_policy_sweep(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-policy-sweep",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--structured-vp-marker-style",
            "plain",
            "--structured-vp-allowed-labels-field",
            "text_input",
            "--structured-vp-max-total-boxes-field",
            "query_box_count",
            "--policy-sweep-config",
            "nms07:filter_policy=nms,nms_iou_threshold=0.7",
            "--policy-sweep-include-target-label-oracle",
            "--policy-sweep-include-phrase-label-policy",
            "--policy-sweep-include-repair-policy",
            "--policy-sweep-focus-bucket",
            "dense",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == [
        "infer_adapter",
        "infer_baseline",
        "audit",
        "policy_sweep_adapter",
        "policy_sweep_baseline",
    ]
    sweep_cmd = summary["commands"][3]["cmd"]
    assert "scripts/experiments/sweep_vp_quality_policies.py" in sweep_cmd
    assert "--structured-vp-allowed-labels-field" in sweep_cmd
    assert "text_input" in sweep_cmd
    assert "--structured-vp-max-total-boxes-field" in sweep_cmd
    assert "query_box_count" in sweep_cmd
    assert "--policy-config" in sweep_cmd
    assert "nms07:filter_policy=nms,nms_iou_threshold=0.7" in sweep_cmd
    assert "--include-target-label-oracle" in sweep_cmd
    assert "--include-phrase-label-policy" in sweep_cmd
    assert "--include-repair-policy" in sweep_cmd
    assert "--focus-bucket" in sweep_cmd
    assert "dense" in sweep_cmd
    assert summary["policy_sweep_summaries"] == {
        "adapter": str(output_dir / "policy_sweep" / "adapter" / "vp_quality_policy_sweep.json"),
        "baseline": str(output_dir / "policy_sweep" / "baseline" / "vp_quality_policy_sweep.json"),
    }
    assert summary["policy_sweep_focus_bucket"] == "dense"
    assert summary["policy_sweep_include_repair_policy"] is True


def test_vp_experiment_runner_dry_run_can_plan_report_card(tmp_path):
    training_summary = tmp_path / "training_summary.json"
    training_summary.write_text(
        json.dumps({"adapter_dir": str(tmp_path / "adapter")}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    script = Path("scripts/experiments/run_florence_vp_training_experiment.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-quality-eval",
            "--run-target-count-gap-analysis",
            "--run-policy-sweep",
            "--run-report-card",
            "--report-card-focus-bucket",
            "dense",
            "--report-card-min-samples",
            "12",
            "--report-card-min-recall",
            "0.8",
            "--report-card-min-f1",
            "0.82",
            "--report-card-min-raw-vp-format-ratio",
            "0.9",
            "--report-card-max-structured-decoder-ratio",
            "0.3",
            "--report-card-min-policy-confidence",
            "strong",
            "--model-path",
            str(tmp_path / "model"),
            "--training-summary",
            str(training_summary),
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    names = [command["name"] for command in summary["commands"]]
    assert names == [
        "infer_adapter",
        "infer_baseline",
        "audit",
        "quality_adapter",
        "quality_baseline",
        "target_count_gap_adapter",
        "target_count_gap_baseline",
        "policy_sweep_adapter",
        "policy_sweep_baseline",
        "report_card_adapter",
        "report_card_baseline",
    ]
    card_cmd = summary["commands"][-2]["cmd"]
    assert "scripts/experiments/build_vp_report_card.py" in card_cmd
    assert "--quality-report" in card_cmd
    assert str(output_dir / "quality" / "adapter" / "vp_detection_quality.json") in card_cmd
    assert "--policy-sweep" in card_cmd
    assert str(output_dir / "policy_sweep" / "adapter" / "vp_quality_policy_sweep.json") in card_cmd
    assert "--target-count-gap" in card_cmd
    assert str(output_dir / "target_count_gap" / "adapter" / "vp_target_count_gap.json") in card_cmd
    assert "--focus-bucket" in card_cmd
    assert "dense" in card_cmd
    assert "--min-samples" in card_cmd
    assert "12" in card_cmd
    assert "--min-recall" in card_cmd
    assert "0.8" in card_cmd
    assert "--min-f1" in card_cmd
    assert "0.82" in card_cmd
    assert "--min-raw-vp-format-ratio" in card_cmd
    assert "0.9" in card_cmd
    assert "--max-structured-decoder-ratio" in card_cmd
    assert "0.3" in card_cmd
    assert "--min-policy-confidence" in card_cmd
    assert "strong" in card_cmd
    assert summary["report_card_summaries"] == {
        "adapter": str(output_dir / "report_card" / "adapter" / "vp_report_card.json"),
        "baseline": str(output_dir / "report_card" / "baseline" / "vp_report_card.json"),
    }
    assert summary["report_card_focus_bucket"] == "dense"
    assert summary["report_card_min_samples"] == 12
    assert summary["report_card_min_recall"] == 0.8
    assert summary["report_card_min_f1"] == 0.82
    assert summary["report_card_min_raw_vp_format_ratio"] == 0.9
    assert summary["report_card_max_structured_decoder_ratio"] == 0.3
