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
from tqdm.auto import tqdm

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
        self._detected_schema: str = ""

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
        """验证样本字段，支持 training schema (prefix/suffix) 和 conversations schema"""
        # 检测 schema 类型
        has_training_fields = "prefix" in sample or "suffix" in sample
        has_conversations = "conversations" in sample

        # 如果已明确指定 training schema，强制使用 training 验证
        if self._detected_schema == "training" or has_training_fields:
            if not self._detected_schema:
                self._detected_schema = "training"
            # Training schema: image + prefix + suffix
            if "image" not in sample:
                self._add_error(f"样本{index}: 缺少必需字段 'image'")
                return False
            if "prefix" not in sample:
                self._add_error(f"样本{index}: 缺少必需字段 'prefix'")
                return False
            if not sample.get("prefix", "").strip():
                self._add_error(f"样本{index}: prefix 字段不能为空")
                return False
            if "suffix" not in sample:
                self._add_error(f"样本{index}: 缺少必需字段 'suffix'")
                return False
            if not sample.get("suffix", "").strip():
                self._add_warning(f"样本{index}: suffix 字段为空")
            return True

        elif has_conversations or self._detected_schema == "conversations":
            # Conversations schema
            if not self._detected_schema:
                self._detected_schema = "conversation"
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
        else:
            # 未知 schema
            self._add_error(f"样本{index}: 缺少必需字段 'conversations'")
            return False

    def _validate_task_specific_content(
        self, sample: Dict[str, Any], index: int
    ) -> None:
        """验证任务特定内容"""
        task_type = sample.get("task_type", "")

        # Training schema: 检查 prefix/suffix 内容
        if "prefix" in sample or "suffix" in sample:
            prefix = sample.get("prefix", "")
            suffix = sample.get("suffix", "")
            if task_type and prefix:
                # 验证任务前缀是否匹配
                from ..core.tasks import FLORENCE2_TASKS
                task_config = FLORENCE2_TASKS.get(task_type, {})
                expected_prompt = task_config.get("prompt", "")
                if expected_prompt and expected_prompt not in prefix:
                    self._add_warning(
                        f"样本{index}: 任务类型 '{task_type}' 的前缀不包含预期提示 '{expected_prompt}'"
                    )

            # 检查 suffix 中的坐标是否超出范围
            import re
            loc_pattern = re.compile(r"<loc_(\d+)>")
            for match in loc_pattern.finditer(suffix):
                coord_value = int(match.group(1))
                if coord_value > 1000:
                    self._add_warning(
                        f"样本{index}: 检测坐标超出范围 <loc_{coord_value}>，有效范围 0-1000"
                    )
            return

        # Conversations schema
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
        result = {
            "is_valid": is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validation_results": self.validation_results,
            "status": "passed" if is_valid else "failed",
        }
        if self._detected_schema:
            result["effective_schema"] = self._detected_schema
            result["detected_schema"] = self._detected_schema
        return result

    def reset(self) -> None:
        """重置验证状态"""
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0
        self._detected_schema = ""

    @staticmethod
    def validate_florence2_jsonl(
        jsonl_path: str,
        image_base_path: str = "",
    ) -> Dict[str, Any]:
        """验证 Florence-2 JSONL 格式数据（静态入口，供转换管线使用）。"""
        logger.info("验证JSONL数据: %s", jsonl_path)
        jsonl_path = Path(jsonl_path).absolute()
        image_base_path = (
            Path(image_base_path).absolute() if image_base_path else Path()
        )

        report: Dict[str, Any] = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "missing_images": 0,
            "task_distribution": {},
            "errors": [],
        }

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(tqdm(handle, desc="验证JSONL进度"), 1):
                report["total_samples"] += 1
                try:
                    data = json.loads(line.strip())
                    for field in ("image", "prefix", "suffix"):
                        if field not in data:
                            report["errors"].append(
                                f"第{line_num}行: 缺少字段 '{field}'"
                            )
                            continue

                    image_path = Path(data["image"])
                    if not image_path.exists():
                        report["missing_images"] += 1
                        report["errors"].append(
                            f"第{line_num}行: 图像文件不存在 '{image_path}'"
                        )

                    for field in ("label_file", "txt_file", "xml_file", "mask_dir"):
                        if field in data and not Path(data[field]).exists():
                            report["errors"].append(
                                f"第{line_num}行: 标签文件不存在 '{data[field]}'"
                            )

                    task_type = data["prefix"].strip("<>")
                    report["task_distribution"][task_type] = (
                        report["task_distribution"].get(task_type, 0) + 1
                    )
                    report["valid_samples"] += 1
                except json.JSONDecodeError as exc:
                    report["invalid_samples"] += 1
                    report["errors"].append(f"第{line_num}行: JSON解析错误 - {exc}")
                except Exception as exc:
                    report["invalid_samples"] += 1
                    report["errors"].append(f"第{line_num}行: 验证错误 - {exc}")

        logger.info(
            "验证完成: 总计%s样本，有效%s，无效%s",
            report["total_samples"],
            report["valid_samples"],
            report["invalid_samples"],
        )
        return report

    @staticmethod
    def generate_validation_report(report: Dict[str, Any], output_path: str) -> None:
        """将 ``validate_florence2_jsonl`` 结果写入 Markdown 报告。"""
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("# 数据验证报告\n\n")
            handle.write(f"- **总样本数**: {report['total_samples']}\n")
            handle.write(f"- **有效样本数**: {report['valid_samples']}\n")
            handle.write(f"- **无效样本数**: {report['invalid_samples']}\n")
            handle.write(f"- **缺失图像数**: {report['missing_images']}\n\n")
            handle.write("## 任务分布\n\n")
            for task, count in report["task_distribution"].items():
                handle.write(f"- **{task}**: {count} 样本\n")
            if report["errors"]:
                handle.write("\n## 错误详情\n\n")
                for error in report["errors"][:100]:
                    handle.write(f"- {error}\n")
                if len(report["errors"]) > 100:
                    handle.write(
                        f"\n... 还有 {len(report['errors']) - 100} 个错误未显示\n"
                    )


def validate_data_format(
    data_path: Union[str, Path], strict_mode: bool = False, schema: str = None
) -> Dict[str, Any]:
    """验证数据格式的便捷函数

    Args:
        data_path: 数据路径
        strict_mode: 是否启用严格模式
        schema: 可选，指定数据 schema（"training" 或 "conversations"）

    Returns:
        验证结果
    """
    validator = DataValidator(strict_mode=strict_mode)
    if schema:
        validator._detected_schema = schema
    return validator.validate_dataset(data_path)
