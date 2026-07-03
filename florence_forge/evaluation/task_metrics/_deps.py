"""Shared optional dependencies for task metric calculators."""

import logging

from ...utils.optional_dependencies import missing_dependency_message

try:
    from pycocotools.coco import COCO  # noqa: F401
    from pycocotools.cocoeval import COCOeval  # noqa: F401
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    logging.warning(
        missing_dependency_message("部分检测指标", "pycocotools", "evaluation")
    )

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning(
        missing_dependency_message("部分分割指标", "opencv-python")
    )

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None
    logging.warning(
        missing_dependency_message("ROUGE指标", "rouge-score", "evaluation")
    )
