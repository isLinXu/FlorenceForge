"""FlorenceForge 评估模块导出入口。"""

from importlib import import_module

__all__ = [
    "MultiTaskEvaluator",
    "BenchmarkEvaluator",
    "MetricCalculator",
    "CaptionMetrics",
    "DetectionMetrics",
    "OCRMetrics",
    "SegmentationMetrics",
    "ResultAnalyzer",
    "SemanticMetricsCalculator",
    "MultiModalMetricsCalculator",
    "RobustnessMetricsCalculator",
    "EfficiencyMetricsCalculator",
    "FlorenceNativeDetectionParser",
    "StructuredVisualPrimitiveDecoder",
    "labels_match",
    "normalize_allowed_labels",
    "native_detections_to_vp",
    "resolve_structured_vp_filter_caps",
    "filter_native_detections",
    "VisualPrimitiveDetectionMetrics",
    "TVPCompositeMetric",
    # VP quality (unified from vp_detection_quality + vp_report_card)
    "VPDetectionQualityConfig",
    "VPReportCardThresholds",
    "analyze_vp_target_count_gap",
    "build_vp_report_card",
    "compare_vp_quality_record_reports",
    "compare_vp_quality_reports",
    "compute_bbox_iou",
    "evaluate_vp_detection_quality",
    "evaluate_vp_summary",
    "match_vp_detections",
    "recommend_vp_policy",
    "render_vp_detection_quality_markdown",
    "render_vp_policy_comparison_markdown",
    "render_vp_record_comparison_markdown",
    "render_vp_report_card_markdown",
    "render_vp_target_count_gap_markdown",
    "summarize_vp_quality_records",
]

_LAZY_EXPORTS = {
    "MultiTaskEvaluator": ("florence_forge.evaluation.evaluator", "MultiTaskEvaluator"),
    "BenchmarkEvaluator": ("florence_forge.evaluation.benchmark", "BenchmarkEvaluator"),
    "MetricCalculator": ("florence_forge.evaluation.metrics", "MetricCalculator"),
    "CaptionMetrics": ("florence_forge.evaluation.metrics", "CaptionMetrics"),
    "DetectionMetrics": ("florence_forge.evaluation.metrics", "DetectionMetrics"),
    "OCRMetrics": ("florence_forge.evaluation.metrics", "OCRMetrics"),
    "SegmentationMetrics": ("florence_forge.evaluation.metrics", "SegmentationMetrics"),
    "ResultAnalyzer": ("florence_forge.evaluation.analyzer", "ResultAnalyzer"),
    "SemanticMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "SemanticMetricsCalculator"),
    "MultiModalMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "MultiModalMetricsCalculator"),
    "RobustnessMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "RobustnessMetricsCalculator"),
    "EfficiencyMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "EfficiencyMetricsCalculator"),
    "FlorenceNativeDetectionParser": ("florence_forge.evaluation.structured_vp_decoder", "FlorenceNativeDetectionParser"),
    "StructuredVisualPrimitiveDecoder": ("florence_forge.evaluation.structured_vp_decoder", "StructuredVisualPrimitiveDecoder"),
    "labels_match": ("florence_forge.evaluation.structured_vp_decoder", "labels_match"),
    "normalize_allowed_labels": ("florence_forge.evaluation.structured_vp_decoder", "normalize_allowed_labels"),
    "native_detections_to_vp": ("florence_forge.evaluation.structured_vp_decoder", "native_detections_to_vp"),
    "resolve_structured_vp_filter_caps": ("florence_forge.evaluation.structured_vp_decoder", "resolve_structured_vp_filter_caps"),
    "filter_native_detections": ("florence_forge.evaluation.structured_vp_decoder", "filter_native_detections"),
    "VisualPrimitiveDetectionMetrics": ("florence_forge.evaluation.metrics", "VisualPrimitiveDetectionMetrics"),
    "TVPCompositeMetric": ("florence_forge.evaluation.tvp_metrics", "TVPCompositeMetric"),
    # VP quality (unified)
    "VPDetectionQualityConfig": ("florence_forge.evaluation.vp_quality", "VPDetectionQualityConfig"),
    "VPReportCardThresholds": ("florence_forge.evaluation.vp_quality", "VPReportCardThresholds"),
    "analyze_vp_target_count_gap": ("florence_forge.evaluation.vp_quality", "analyze_vp_target_count_gap"),
    "build_vp_report_card": ("florence_forge.evaluation.vp_quality", "build_vp_report_card"),
    "compare_vp_quality_record_reports": ("florence_forge.evaluation.vp_quality", "compare_vp_quality_record_reports"),
    "compare_vp_quality_reports": ("florence_forge.evaluation.vp_quality", "compare_vp_quality_reports"),
    "compute_bbox_iou": ("florence_forge.evaluation.vp_quality", "compute_bbox_iou"),
    "evaluate_vp_detection_quality": ("florence_forge.evaluation.vp_quality", "evaluate_vp_detection_quality"),
    "evaluate_vp_summary": ("florence_forge.evaluation.vp_quality", "evaluate_vp_summary"),
    "match_vp_detections": ("florence_forge.evaluation.vp_quality", "match_vp_detections"),
    "recommend_vp_policy": ("florence_forge.evaluation.vp_quality", "recommend_vp_policy"),
    "render_vp_detection_quality_markdown": ("florence_forge.evaluation.vp_quality", "render_vp_detection_quality_markdown"),
    "render_vp_policy_comparison_markdown": ("florence_forge.evaluation.vp_quality", "render_vp_policy_comparison_markdown"),
    "render_vp_record_comparison_markdown": ("florence_forge.evaluation.vp_quality", "render_vp_record_comparison_markdown"),
    "render_vp_report_card_markdown": ("florence_forge.evaluation.vp_quality", "render_vp_report_card_markdown"),
    "render_vp_target_count_gap_markdown": ("florence_forge.evaluation.vp_quality", "render_vp_target_count_gap_markdown"),
    "summarize_vp_quality_records": ("florence_forge.evaluation.vp_quality", "summarize_vp_quality_records"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
