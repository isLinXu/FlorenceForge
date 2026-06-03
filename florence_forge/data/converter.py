"""FlorenceForge 数据格式转换器门面。

实现按格式分布在 ``converter_od`` / ``converter_caption`` / ``converter_ocr`` /
``converter_region`` 子模块；本模块保留 ``DataFormatConverter`` 与历史导入路径。
"""

from __future__ import annotations

from . import converter_caption as _caption
from . import converter_ocr as _ocr
from . import converter_od as _od
from . import converter_region as _region
from .converter_mask import generate_mask_from_polygon
from .validator import DataValidator  # noqa: F401 — 历史导入路径

__all__ = [
    "DataFormatConverter",
    "DataValidator",
    "generate_mask_from_polygon",
]


class DataFormatConverter:
    """数据格式转换器（各静态方法委托至子模块实现）。"""

    yolo_to_florence2_od = staticmethod(_od.yolo_to_florence2_od)
    coco_to_florence2_od = staticmethod(_od.coco_to_florence2_od)
    xml_to_florence2_od = staticmethod(_od.xml_to_florence2_od)

    coco_caption_to_florence2 = staticmethod(_caption.coco_caption_to_florence2)
    csv_caption_to_florence2 = staticmethod(_caption.csv_caption_to_florence2)

    txt_ocr_to_florence2 = staticmethod(_ocr.txt_ocr_to_florence2)
    txt_file_ocr_to_florence2 = staticmethod(_ocr.txt_file_ocr_to_florence2)

    json_to_florence2_grounding = staticmethod(_region.json_to_florence2_grounding)
    coco_to_florence2_region_segmentation = staticmethod(
        _region.coco_to_florence2_region_segmentation
    )
    csv_to_florence2_region_category = staticmethod(_region.csv_to_florence2_region_category)
    csv_to_florence2_region_description = staticmethod(
        _region.csv_to_florence2_region_description
    )
    json_to_florence2_region_proposal = staticmethod(_region.json_to_florence2_region_proposal)
    json_to_florence2_ocr_with_region = staticmethod(_region.json_to_florence2_ocr_with_region)
    json_to_florence2_referring_expression_segmentation = staticmethod(
        _region.json_to_florence2_referring_expression_segmentation
    )
    json_to_florence2_region_segmentation = staticmethod(
        _region.json_to_florence2_region_segmentation
    )
    json_to_florence2_dense_region_caption = staticmethod(
        _region.json_to_florence2_dense_region_caption
    )
    json_to_florence2_open_vocabulary_detection = staticmethod(
        _region.json_to_florence2_open_vocabulary_detection
    )
    json_to_florence2_detection_with_confidence = staticmethod(
        _region.json_to_florence2_detection_with_confidence
    )
    json_to_florence2_grounding_with_confidence = staticmethod(
        _region.json_to_florence2_grounding_with_confidence
    )
