from florence_forge.evaluation.vp_training_audit import (
    VPTrainingAuditThresholds,
    build_vp_training_audit,
    render_vp_training_audit_markdown,
)


def _training_summary():
    return {
        "ok": True,
        "training_mode": "lora",
        "lora_task_type": "SEQ_2_SEQ_LM",
        "vp_box_format": "loc_tokens",
        "max_steps": 12,
        "steps_executed": 12,
        "final_loss": 2.7,
        "trainable_param_delta_norm": 2.5,
        "lora_modules_to_save": ["lm_head", "model.shared"],
        "data": {
            "train_od_rows": 2,
            "val_od_rows": 1,
            "train_count_rows": 2,
            "val_count_rows": 1,
        },
    }


def test_vp_training_audit_marks_structured_ready_but_not_internalized():
    audit = build_vp_training_audit(
        training_summary=_training_summary(),
        inference_summaries=[{
            "num_samples": 2,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 1.0,
            "structured_vp_decoder_ratio": 1.0,
            "structured_source_counts": {"florence_native": 2},
        }],
    )

    assert audit["status"] == "engineering_mvp_ready_needs_wrapper_training"
    assert audit["gates"]["training_smoke_passed"]["passed"] is True
    assert audit["gates"]["structured_vp_usable"]["passed"] is True
    assert audit["gates"]["raw_vp_internalized"]["passed"] is False
    assert audit["gates"]["decoder_dependency_low"]["passed"] is False
    assert audit["gates"]["baseline_present"]["passed"] is False
    assert audit["training"]["vp_head_trainable"] is True
    assert audit["training"]["lora_task_type"] == "SEQ_2_SEQ_LM"
    assert any("structured decoding is usable" in item for item in audit["recommendations"])


def test_vp_training_audit_accepts_grounding_only_data_ready():
    summary = _training_summary()
    summary["data"] = {
        "train_od_rows": 0,
        "val_od_rows": 0,
        "train_grounding_effective_rows": 2,
        "val_grounding_rows": 2,
        "skip_od_training_data": True,
    }

    audit = build_vp_training_audit(
        training_summary=summary,
        inference_summaries=[{
            "num_samples": 2,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 1.0,
            "structured_vp_decoder_ratio": 1.0,
        }],
    )

    assert audit["data"]["skip_od_training_data"] is True
    assert audit["gates"]["data_ready"]["passed"] is True
    assert "train_grounding_effective_rows=2" in audit["gates"]["data_ready"]["message"]


def test_vp_training_audit_marks_candidate_complete_with_raw_vp_and_baseline():
    audit = build_vp_training_audit(
        training_summary=_training_summary(),
        inference_summaries=[{
            "num_samples": 50,
            "vp_format_valid_ratio": 0.97,
            "structured_vp_format_valid_ratio": 0.97,
            "structured_vp_decoder_ratio": 0.0,
        }],
        baseline_summaries=[{
            "num_samples": 50,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 1.0,
            "structured_vp_decoder_ratio": 1.0,
        }],
        thresholds=VPTrainingAuditThresholds(min_inference_samples=10),
    )

    assert audit["status"] == "candidate_training_complete"
    assert audit["gates"]["raw_vp_internalized"]["passed"] is True
    assert audit["gates"]["baseline_present"]["passed"] is True
    assert audit["baseline"]["deltas"]["vp_format_valid_ratio"] == 0.97
    assert audit["baseline"]["deltas"]["structured_vp_decoder_ratio"] == -1.0


def test_vp_training_audit_markdown_renders_key_metrics():
    audit = build_vp_training_audit(
        training_summary=_training_summary(),
        inference_summaries=[{
            "num_samples": 2,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 1.0,
            "structured_vp_decoder_ratio": 1.0,
        }],
    )

    markdown = render_vp_training_audit_markdown(audit)

    assert "Florence-VP Training Audit" in markdown
    assert "LoRA task type" in markdown
    assert "structured_vp_format_valid_ratio" in markdown
    assert "raw_vp_internalized" in markdown


def test_vp_training_audit_computes_box_overgeneration_from_records():
    audit = build_vp_training_audit(
        training_summary=_training_summary(),
        inference_summaries=[{
            "num_samples": 2,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 1.0,
            "structured_vp_decoder_ratio": 1.0,
            "records": [
                {"pred_box_count": 1, "gt_box_count": 1},
                {"pred_box_count": 5, "gt_box_count": 1},
            ],
        }],
    )

    aggregate = audit["inference"]["aggregate"]
    assert aggregate["avg_pred_boxes"] == 3.0
    assert aggregate["avg_gt_boxes"] == 1.0
    assert aggregate["box_count_overgeneration_ratio"] == 0.5
    assert audit["gates"]["box_count_not_overgenerated"]["passed"] is False
    assert any("overgenerated native loc" in item for item in audit["recommendations"])
