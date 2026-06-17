"""CLI 共享常量与纯辅助函数。

抽离自 ``cli/main.py``，供 ``main.py`` 与 ``commands.py`` 共同使用，
避免两者相互导入造成循环依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff',
    '.webp',
    '.gif',
}

# 预定义的任务 → 配置文件映射
TASK_CONFIG_MAPPING: Dict[str, str] = {
    'caption': 'configs/examples/caption_training.yaml',
    'detailed_caption': 'configs/examples/detailed_caption_training.yaml',
    'more_detailed_caption': 'configs/examples/more_detailed_caption_training.yaml',
    'detection': 'configs/examples/object_detection_training.yaml',
    'od': 'configs/examples/object_detection_training.yaml',
    'open_vocabulary_detection': 'configs/examples/open_vocabulary_detection_training.yaml',
    'phrase_grounding': 'configs/examples/phrase_grounding_training.yaml',
    'dense_region_caption': 'configs/examples/dense_region_caption_training.yaml',
    'region_proposal': 'configs/examples/region_proposal_training.yaml',
    'region_to_category': 'configs/examples/region_to_category_training.yaml',
    'region_to_description': 'configs/examples/region_to_description_training.yaml',
    'ocr': 'configs/examples/ocr_training.yaml',
    'ocr_with_region': 'configs/examples/ocr_with_region_training.yaml',
    'segmentation': 'configs/examples/segmentation_training.yaml',
    'seg': 'configs/examples/segmentation_training.yaml',
    'region_to_segmentation': 'configs/examples/region_to_segmentation_training.yaml',
    'referring_expression_segmentation': 'configs/examples/referring_expression_segmentation_training.yaml',
    'multitask': 'configs/examples/multitask_training.yaml',
    'multi': 'configs/examples/multitask_training.yaml',
    'visual_primitive': 'configs/examples/visual_primitive_training.yaml',
    'vp': 'configs/examples/visual_primitive_training.yaml',
}

# 任务描述（用于 list-tasks 展示）
TASK_DESCRIPTIONS: Dict[str, str] = {
    'caption': '基础图像描述生成任务 (CAPTION)',
    'detailed_caption': '详细图像描述生成任务 (DETAILED_CAPTION)',
    'more_detailed_caption': '更详细图像描述生成任务 (MORE_DETAILED_CAPTION)',
    'detection': '标准目标检测任务 (OD)',
    'open_vocabulary_detection': '开放词汇目标检测任务 (OPEN_VOCABULARY_DETECTION)',
    'phrase_grounding': '短语定位任务 (CAPTION_TO_PHRASE_GROUNDING)',
    'dense_region_caption': '密集区域描述任务 (DENSE_REGION_CAPTION)',
    'region_proposal': '区域提议任务 (REGION_PROPOSAL)',
    'region_to_category': '区域到类别分类任务 (REGION_TO_CATEGORY)',
    'region_to_description': '区域到描述生成任务 (REGION_TO_DESCRIPTION)',
    'ocr': 'OCR文字识别任务 (OCR)',
    'ocr_with_region': '带区域的OCR任务 (OCR_WITH_REGION)',
    'segmentation': '标准图像分割任务',
    'region_to_segmentation': '区域到分割任务 (REGION_TO_SEGMENTATION)',
    'referring_expression_segmentation': '参考表达式分割任务 (REFERRING_EXPRESSION_SEGMENTATION)',
    'multitask': '多任务混合训练 (CAPTION + OD + OCR + SEGMENTATION)',
    'visual_primitive': '视觉原语训练 (OD_VP + COUNT_VP + PHRASE_GROUNDING_VP)',
}


def _is_supported_image_file(path: Path) -> bool:
    """Return True when path is a supported image file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _iter_image_files(directory: Path) -> List[Path]:
    """Iterate supported image files with case-insensitive suffix matching."""
    return sorted(
        path
        for path in directory.iterdir()
        if _is_supported_image_file(path)
    )


def _normalize_inference_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return inference stats with stable keys for CLI summaries/logging."""
    stats = dict(stats or {})
    total_inferences = float(stats.get("total_inferences", 0) or 0)
    total_time = float(stats.get("total_time", 0.0) or 0.0)
    avg_inference_time = stats.get("avg_inference_time")
    if avg_inference_time is None:
        avg_inference_time = total_time / total_inferences if total_inferences > 0 else 0.0
    throughput = stats.get("throughput")
    if throughput is None:
        throughput = total_inferences / total_time if total_time > 0 else 0.0

    stats.update({
        "total_inferences": int(total_inferences),
        "total_time": total_time,
        "avg_inference_time": float(avg_inference_time or 0.0),
        "throughput": float(throughput or 0.0),
    })
    return stats
