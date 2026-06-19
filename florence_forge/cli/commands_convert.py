"""CLI 数据转换命令处理。

处理 ``convert`` 子命令：YOLO / COCO / CSV / VOC / OCR / VP / TVP 数据格式转换。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def _resolve_vp_box_format(box_format: str) -> str:
    normalized = str(box_format or "json").strip().lower()
    if normalized == "quad":
        return "json"
    if normalized in {"json", "loc_tokens", "loc"}:
        return "loc" if normalized == "loc" else normalized
    raise ValueError("box_format must be one of: json, loc_tokens, quad")


def _resolve_vp_marker_style(marker_style: str) -> str:
    from florence_forge.core.visual_primitives import resolve_marker_style
    return resolve_marker_style(marker_style)


def _run_vp_coco_conversion(args: "Namespace") -> None:
    from florence_forge.data.vp_converter import VisualPrimitiveConverter

    vp_task = getattr(args, "vp_task", "OD_VP")
    box_fmt = _resolve_vp_box_format(getattr(args, "box_format", "json"))
    marker = _resolve_vp_marker_style(getattr(args, "marker_style", "special"))

    if vp_task == "COUNT_VP_COT":
        from florence_forge.data.tvp_converter import TVPDataConverter
        TVPDataConverter.coco_to_tvp_counting(
            coco_json_path=args.json_file,
            output_path=args.output,
            image_dir=args.images_dir,
            task_type="COUNT_VP_COT",
            marker_style=marker,
            counting_mode=getattr(args, "counting_mode", "coarse"),
        )
        return

    converter_map = {
        "OD_VP": VisualPrimitiveConverter.coco_to_vp_od,
        "COUNT_VP": VisualPrimitiveConverter.coco_to_vp_counting,
        "PHRASE_GROUNDING_VP": VisualPrimitiveConverter.coco_to_vp_grounding,
    }
    converter = converter_map.get(vp_task)
    if converter is None:
        raise ValueError(f"Unsupported VP COCO task: {vp_task}")
    converter(
        coco_json_path=args.json_file,
        output_path=args.output,
        image_dir=args.images_dir,
        task_type=vp_task,
        box_format=box_fmt,
        marker_style=marker,
    )


def _run_vp_yolo_conversion(args: "Namespace") -> None:
    from florence_forge.data.vp_converter import VisualPrimitiveConverter

    vp_task = getattr(args, "vp_task", "OD_VP")
    box_fmt = _resolve_vp_box_format(getattr(args, "box_format", "json"))
    marker = _resolve_vp_marker_style(getattr(args, "marker_style", "special"))
    converter_map = {
        "OD_VP": VisualPrimitiveConverter.yolo_to_vp_od,
        "COUNT_VP": VisualPrimitiveConverter.yolo_to_vp_counting,
        "PHRASE_GROUNDING_VP": VisualPrimitiveConverter.yolo_to_vp_grounding,
    }
    converter = converter_map.get(vp_task)
    if converter is None:
        raise ValueError(f"Unsupported VP YOLO task: {vp_task}")
    converter(
        yolo_labels_dir=args.labels_dir,
        output_path=args.output,
        image_dir=args.images_dir,
        classes_file=args.classes_file,
        task_type=vp_task,
        box_format=box_fmt,
        marker_style=marker,
        image_ext=getattr(args, "image_ext", ".jpg"),
    )


def run_data_conversion(args: "Namespace") -> bool:
    """运行数据转换任务"""
    try:
        from florence_forge.data.converter import DataFormatConverter

        logger.info(f"开始数据转换: {args.convert_type}")
        ct = args.convert_type

        if ct == "yolo":
            DataFormatConverter.yolo_to_florence2_od(
                yolo_labels_dir=args.labels_dir,
                output_path=args.output,
                image_dir=args.images_dir,
                classes_file=args.classes_file,
                image_ext=args.image_ext,
                task_type=args.task_type,
            )
        elif ct == "coco":
            DataFormatConverter.coco_to_florence2_od(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir,
            )
        elif ct == "coco-caption":
            DataFormatConverter.coco_caption_to_florence2(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir,
            )
        elif ct == "csv":
            DataFormatConverter.csv_caption_to_florence2(
                csv_path=args.csv_file,
                output_path=args.output,
                image_column=args.image_column,
                caption_column=args.caption_column,
                task_type=args.task_type,
            )
        elif ct == "xml":
            DataFormatConverter.xml_to_florence2_od(
                xml_dir=args.xml_dir,
                output_path=args.output,
                image_dir=args.images_dir,
            )
        elif ct == "ocr":
            DataFormatConverter.txt_ocr_to_florence2(
                image_dir=args.images_dir,
                txt_dir=args.texts_dir,
                output_path=args.output,
                task_type=args.task_type,
            )
        elif ct == "ocr-txt":
            DataFormatConverter.txt_file_ocr_to_florence2(
                txt_file_path=args.txt_file,
                image_dir=args.images_dir,
                output_path=args.output,
                task_type=args.task_type,
            )
        elif ct in ("vp-coco", "vp-yolo"):
            if ct == "vp-coco":
                _run_vp_coco_conversion(args)
            else:
                _run_vp_yolo_conversion(args)
        elif ct == "vp-coco-od":
            from florence_forge.data.vp_converter import VisualPrimitiveConverter
            VisualPrimitiveConverter.coco_to_vp_od(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir,
                task_type=getattr(args, "task_type", "OD_VP"),
                box_format=getattr(args, "box_format", "json"),
                marker_style=getattr(args, "marker_style", "special"),
            )
        elif ct == "vp-yolo-count":
            from florence_forge.data.vp_converter import VisualPrimitiveConverter
            VisualPrimitiveConverter.yolo_to_vp_counting(
                yolo_labels_dir=args.labels_dir,
                output_path=args.output,
                image_dir=args.images_dir,
                classes_file=args.classes_file,
                image_ext=getattr(args, "image_ext", ".jpg"),
                task_type=getattr(args, "task_type", "COUNT_VP"),
                box_format=getattr(args, "box_format", "loc_tokens"),
                marker_style=getattr(args, "marker_style", "plain"),
            )
        elif ct == "vp-jsonl-grounding":
            from florence_forge.data.vp_converter import VisualPrimitiveConverter
            VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding(
                input_path=args.input,
                output_path=args.output,
                task_type=getattr(args, "task_type", "PHRASE_GROUNDING_VP"),
                box_format=getattr(args, "box_format", "loc_tokens"),
                marker_style=getattr(args, "marker_style", "plain"),
            )
        elif ct == "tvp-count-cot":
            from florence_forge.data.tvp_converter import TVPDataConverter
            TVPDataConverter.coco_to_tvp_counting(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir,
                task_type="COUNT_VP_COT",
                marker_style=_resolve_vp_marker_style(getattr(args, "marker_style", "special")),
                counting_mode=getattr(args, "counting_mode", "coarse"),
            )
        elif ct == "tvp-maze":
            from florence_forge.data.tvp_converter import TVPDataConverter
            TVPDataConverter.maze_jsonl_to_vp(
                input_path=args.input,
                output_path=args.output,
                image_dir=args.images_dir,
                task_type="MAZE_VP",
                marker_style=_resolve_vp_marker_style(getattr(args, "marker_style", "special")),
            )
        elif ct == "tvp-path":
            from florence_forge.data.tvp_converter import TVPDataConverter
            TVPDataConverter.path_jsonl_to_vp(
                input_path=args.input,
                output_path=args.output,
                image_dir=args.images_dir,
                task_type="PATH_VP",
                marker_style=_resolve_vp_marker_style(getattr(args, "marker_style", "special")),
            )
        elif ct == "tvp-spatial":
            from florence_forge.data.tvp_converter import TVPDataConverter
            TVPDataConverter.spatial_reasoning_jsonl_to_vp(
                input_path=args.input,
                output_path=args.output,
                image_dir=args.images_dir,
                task_type="SPATIAL_VP",
                marker_style=_resolve_vp_marker_style(getattr(args, "marker_style", "special")),
            )
        elif ct == "generate-tvp-maze":
            from florence_forge.data.tvp_synthetic import write_maze_jsonl
            jsonl_path = write_maze_jsonl(
                args.output_dir,
                num_samples=getattr(args, "num_samples", 100),
                rows=getattr(args, "rows", 8),
                cols=getattr(args, "cols", 8),
                seed=getattr(args, "seed", 42),
            )
            logger.info(f"✅ 迷宫合成数据已写入: {jsonl_path}")
        elif ct == "generate-tvp-path":
            from florence_forge.data.tvp_synthetic import write_path_jsonl
            jsonl_path = write_path_jsonl(
                args.output_dir,
                num_samples=getattr(args, "num_samples", 100),
                seed=getattr(args, "seed", 42),
            )
            logger.info(f"✅ 路径追踪合成数据已写入: {jsonl_path}")
        elif ct == "generate-tvp-spatial":
            from florence_forge.data.tvp_synthetic import write_spatial_jsonl
            jsonl_path = write_spatial_jsonl(
                args.output_dir,
                num_samples=getattr(args, "num_samples", 100),
                seed=getattr(args, "seed", 42),
            )
            logger.info(f"✅ 空间推理合成数据已写入: {jsonl_path}")
        elif ct == "generate-tvp-all":
            from florence_forge.data.tvp_synthetic import write_all_tvp_synthetic
            outputs = write_all_tvp_synthetic(
                args.output_dir,
                num_samples=getattr(args, "num_samples", 8),
                seed=getattr(args, "seed", 42),
            )
            logger.info(f"✅ TVP 全量合成数据已写入: {outputs}")
        else:
            logger.error(f"❌ 不支持的转换类型: {args.convert_type}")
            return False

        if hasattr(args, "output") and getattr(args, "output", None):
            logger.info(f"✅ 数据转换完成: {args.output}")
        else:
            logger.info("✅ 数据转换完成")
        return True

    except ImportError as e:
        logger.error(f"❌ 导入数据转换器失败: {e}")
        logger.error("请确保已正确安装florence_forge或数据转换器模块")
        return False
    except Exception as e:
        logger.error(f"❌ 数据转换失败: {e}")
        return False
