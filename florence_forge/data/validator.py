# -*- coding: utf-8 -*-
"""
数据验证器

提供数据质量检查、格式验证和完整性检测功能
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

from PIL import Image

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器

    检查数据集的质量、格式和完整性
    """

    def __init__(self, strict_mode: bool = False):
        """初始化验证器

        Args:
            strict_mode: 是否启用严格模式
        """
        self.strict_mode = strict_mode
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0

    def validate_dataset(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """验证数据集

        Args:
            data_path: 数据集路径

        Returns:
            验证结果字典
        """
        self.reset()
        data_path = Path(data_path)

        if not data_path.exists():
            self._add_error(f"数据集路径不存在: {data_path}")
            return self._get_validation_summary()

        # 验证JSON格式
        if data_path.suffix == ".json":
            return self._validate_json_dataset(data_path)
        elif data_path.suffix == ".jsonl":
            return self._validate_jsonl_dataset(data_path)
        else:
            self._add_error(f"不支持的数据格式: {data_path.suffix}")
            return self._get_validation_summary()

    def _validate_json_dataset(self, data_path: Path) -> Dict[str, Any]:
        """验证JSON格式数据集"""
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return self._validate_sample_list(data, data_path.parent)
            elif isinstance(data, dict):
                if "data" in data:
                    return self._validate_sample_list(data["data"], data_path.parent)
                else:
                    self._add_error("JSON数据格式错误：缺少'data'字段")
            else:
                self._add_error("JSON数据格式错误：根节点必须是列表或字典")

        except json.JSONDecodeError as e:
            self._add_error(f"JSON解析错误: {e}")
        except Exception as e:
            self._add_error(f"读取文件错误: {e}")

        return self._get_validation_summary()

    def _validate_jsonl_dataset(self, data_path: Path) -> Dict[str, Any]:
        """验证JSONL格式数据集"""
        samples = []
        line_num = 0

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        self._add_error(f"第{line_num}行JSON解析错误: {e}")

            return self._validate_sample_list(samples, data_path.parent)

        except Exception as e:
            self._add_error(f"读取JSONL文件错误: {e}")

        return self._get_validation_summary()

    def _validate_sample_list(
        self, samples: List[Dict], base_path: Path
    ) -> Dict[str, Any]:
        """验证样本列表"""
        if not samples:
            self._add_error("数据集为空")
            return self._get_validation_summary()

        # 统计信息
        task_counts = Counter()
        image_formats = Counter()
        missing_images = []
        invalid_samples = set()

        for i, sample in enumerate(samples):
            try:
                # 验证必需字段
                validation_result = self._validate_sample_fields(sample, i)
                if not validation_result:
                    invalid_samples.add(i)
                    continue

                # 统计任务类型
                task_type = sample.get("task_type", "unknown")
                task_counts[task_type] += 1

                # 验证图像文件
                image_path = sample.get("image")
                if image_path:
                    full_image_path = base_path / image_path
                    if not full_image_path.exists():
                        missing_images.append((i, image_path))
                        invalid_samples.add(i)
                    else:
                        # 检查图像格式
                        try:
                            with Image.open(full_image_path) as img:
                                image_formats[img.format] += 1

                                # 检查图像尺寸
                                if img.size[0] < 32 or img.size[1] < 32:
                                    self._add_warning(
                                        f"样本{i}: 图像尺寸过小 {img.size}"
                                    )

                        except Exception as e:
                            self._add_error(f"样本{i}: 无法读取图像 {image_path}: {e}")
                            invalid_samples.add(i)

                # 验证任务特定内容
                self._validate_task_specific_content(sample, i)

            except Exception as e:
                self._add_error(f"样本{i}: 验证过程出错: {e}")
                invalid_samples.add(i)

        # 记录缺失图像
        if missing_images:
            for sample_idx, image_path in missing_images[:10]:  # 只显示前10个
                self._add_error(f"样本{sample_idx}: 图像文件不存在: {image_path}")

            if len(missing_images) > 10:
                self._add_error(f"还有{len(missing_images) - 10}个样本的图像文件缺失")

        # 生成验证报告
        summary = self._get_validation_summary()
        summary.update(
            {
                "total_samples": len(samples),
                "valid_samples": len(samples) - len(invalid_samples),
                "invalid_samples": len(invalid_samples),
                "missing_images": len(missing_images),
                "task_distribution": dict(task_counts),
                "image_formats": dict(image_formats),
            }
        )

        return summary

    def _validate_sample_fields(self, sample: Dict[str, Any], index: int) -> bool:
        """验证样本字段"""
        required_fields = ["image", "conversations"]

        # 检查必需字段
        for field in required_fields:
            if field not in sample:
                self._add_error(f"样本{index}: 缺少必需字段 '{field}'")
                return False

        # 验证conversations格式
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) == 0:
            self._add_error(f"样本{index}: conversations字段必须是非空列表")
            return False

        # 验证对话格式
        for j, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                self._add_error(f"样本{index}: conversations[{j}]必须是字典")
                return False

            if "from" not in conv or "value" not in conv:
                self._add_error(
                    f"样本{index}: conversations[{j}]缺少'from'或'value'字段"
                )
                return False

            if conv["from"] not in ["human", "gpt"]:
                self._add_warning(
                    f"样本{index}: conversations[{j}]['from']值异常: {conv['from']}"
                )

        return True

    def _validate_task_specific_content(
        self, sample: Dict[str, Any], index: int
    ) -> None:
        """验证任务特定内容"""
        task_type = sample.get("task_type", "")
        conversations = sample.get("conversations", [])

        if not conversations:
            return

        # 获取最后一个回答（通常是模型的输出）
        last_response = None
        for conv in reversed(conversations):
            if conv.get("from") == "gpt":
                last_response = conv.get("value", "")
                break

        if not last_response:
            self._add_warning(f"样本{index}: 没有找到模型回答")
            return

        # 根据任务类型验证内容格式
        if "detection" in task_type.lower() or "object" in task_type.lower():
            self._validate_detection_format(last_response, index)
        elif "caption" in task_type.lower() or "description" in task_type.lower():
            self._validate_caption_format(last_response, index)
        elif "ocr" in task_type.lower():
            self._validate_ocr_format(last_response, index)
        elif "segmentation" in task_type.lower():
            self._validate_segmentation_format(last_response, index)

    def _validate_detection_format(self, response: str, index: int) -> None:
        """验证检测格式"""
        # 检查Florence-2检测格式
        florence_pattern = r"\w+<loc_\d+><loc_\d+><loc_\d+><loc_\d+>"
        if re.search(florence_pattern, response):
            # 验证坐标范围
            coord_pattern = r"<loc_(\d+)>"
            coords = re.findall(coord_pattern, response)
            for coord in coords:
                if int(coord) > 1000:
                    self._add_warning(f"样本{index}: 检测坐标超出范围: {coord}")
        else:
            # 尝试JSON格式
            try:
                data = json.loads(response)
                if not isinstance(data, (list, dict)):
                    self._add_warning(f"样本{index}: 检测结果格式可能不正确")
            except json.JSONDecodeError:
                self._add_warning(f"样本{index}: 检测结果格式无法识别")

    def _validate_caption_format(self, response: str, index: int) -> None:
        """验证描述格式"""
        if len(response.strip()) < 5:
            self._add_warning(f"样本{index}: 图像描述过短")
        elif len(response) > 500:
            self._add_warning(f"样本{index}: 图像描述过长")

    def _validate_ocr_format(self, response: str, index: int) -> None:
        """验证OCR格式"""
        # OCR结果通常是纯文本
        if not response.strip():
            self._add_warning(f"样本{index}: OCR结果为空")

    def _validate_segmentation_format(self, response: str, index: int) -> None:
        """验证分割格式"""
        # 检查多边形格式
        if "<poly>" not in response and "<seg>" not in response:
            self._add_warning(f"样本{index}: 分割结果格式可能不正确")

    def _add_error(self, message: str) -> None:
        """添加错误信息"""
        self.validation_results.append({"type": "error", "message": message})
        self.error_count += 1
        logger.error(message)

    def _add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.validation_results.append({"type": "warning", "message": message})
        self.warning_count += 1
        logger.warning(message)

    def _get_validation_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        is_valid = self.error_count == 0 and (
            not self.strict_mode or self.warning_count == 0
        )
        return {
            "is_valid": is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validation_results": self.validation_results,
            "status": "passed" if is_valid else "failed",
        }

    def reset(self) -> None:
        """重置验证状态"""
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0


def validate_data_format(
    data_path: Union[str, Path], strict_mode: bool = False
) -> Dict[str, Any]:
    """验证数据格式的便捷函数

    Args:
        data_path: 数据路径
        strict_mode: 是否启用严格模式

    Returns:
        验证结果
    """
    validator = DataValidator(strict_mode=strict_mode)
    return validator.validate_dataset(data_path)
