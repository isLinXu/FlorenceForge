"""FlorenceForge指标计算模块

为不同任务类型提供专门的评估指标计算
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from ..utils.optional_dependencies import missing_dependency_message


try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
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

logger = logging.getLogger(__name__)

class MetricCalculator:
    """指标计算器基类"""
    
    def __init__(self, task_type: str):
        """初始化指标计算器
        
        Args:
            task_type: 任务类型
        """
        self.task_type = task_type
        self.predictions = []
        self.references = []
    
    def add_batch(self, predictions: List[str], references: List[str]) -> None:
        """添加一批预测和参考结果
        
        Args:
            predictions: 预测结果列表
            references: 参考结果列表
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> Dict[str, float]:
        """计算指标
        
        Returns:
            指标字典
        """
        if not self.predictions or not self.references:
            logger.warning(f"任务 {self.task_type} 没有预测或参考数据，返回空指标")
            return {}
        
        if len(self.predictions) != len(self.references):
            logger.error(f"预测数量({len(self.predictions)})与参考数量({len(self.references)})不匹配")
            return {}
        
        # 基类提供通用指标计算
        metrics = {}
        
        # 计算基本统计信息
        metrics['num_samples'] = len(self.predictions)
        
        # 计算预测和参考的长度统计
        pred_lengths = [len(str(pred).split()) for pred in self.predictions]
        ref_lengths = [len(str(ref).split()) for ref in self.references]
        
        metrics.update({
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'pred_length_std': np.std(pred_lengths),
            'ref_length_std': np.std(ref_lengths)
        })
        
        # 计算简单的字符串匹配指标
        exact_matches = sum(1 for p, r in zip(self.predictions, self.references) if str(p).strip() == str(r).strip())
        metrics['exact_match_ratio'] = exact_matches / len(self.predictions)
        
        return metrics
    
    def reset(self) -> None:
        """重置累积的预测和参考结果"""
        self.predictions = []
        self.references = []

class CaptionMetrics(MetricCalculator):
    """图像描述任务指标"""
    
    def __init__(self):
        """初始化图像描述任务指标计算器
        
        该类专门用于计算图像描述任务的评估指标，包括BLEU、ROUGE、CIDEr等
        自然语言生成任务的标准指标。
        """
        super().__init__("caption")
    
    def compute(self) -> Dict[str, float]:
        """计算图像描述指标
        
        Returns:
            包含BLEU、ROUGE、CIDEr等指标的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        metrics = {}
        
        # BLEU分数
        bleu_scores = self._compute_bleu()
        metrics.update(bleu_scores)
        
        # ROUGE分数
        rouge_scores = self._compute_rouge()
        metrics.update(rouge_scores)
        
        # 简单的词汇重叠指标
        metrics['word_overlap'] = self._compute_word_overlap()
        
        # 长度统计
        metrics.update(self._compute_length_stats())
        
        return metrics
    
    def _compute_bleu(self) -> Dict[str, float]:
        """计算BLEU分数"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            warned_tokenizer_fallback = False

            def tokenize(text: str) -> List[str]:
                nonlocal warned_tokenizer_fallback
                try:
                    return word_tokenize(text.lower())
                except LookupError:
                    if not warned_tokenizer_fallback:
                        logger.warning(
                            "NLTK punkt 数据未安装，BLEU 计算降级为正则分词；"
                            "如需标准分词，请安装 evaluation 额外依赖并预先下载 punkt。"
                        )
                        warned_tokenizer_fallback = True
                    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
            
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            
            for pred, ref in zip(self.predictions, self.references):
                pred_tokens = tokenize(pred)
                ref_tokens = [tokenize(ref)]
                
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            
            return {
                'bleu': np.mean(bleu_scores),
                'bleu_std': np.std(bleu_scores)
            }
        
        except ImportError:
            logger.warning(
                missing_dependency_message("BLEU计算", "nltk", "evaluation")
            )
            return {}
    
    def _compute_rouge(self) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not ROUGE_AVAILABLE:
            return {}

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = defaultdict(list)

            for pred, ref in zip(self.predictions, self.references):
                scores = scorer.score(ref, pred)
                for key, score in scores.items():
                    rouge_scores[f'{key}_f1'].append(score.fmeasure)
                    rouge_scores[f'{key}_precision'].append(score.precision)
                    rouge_scores[f'{key}_recall'].append(score.recall)

            return {
                key: np.mean(values)
                for key, values in rouge_scores.items()
            }

        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            return {}
    
    def _compute_word_overlap(self) -> float:
        """计算词汇重叠率"""
        overlaps = []
        
        for pred, ref in zip(self.predictions, self.references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0:
                overlap = 0.0
            else:
                overlap = len(pred_words & ref_words) / len(ref_words)
            
            overlaps.append(overlap)
        
        return np.mean(overlaps)
    
    def _compute_length_stats(self) -> Dict[str, float]:
        """计算长度统计"""
        pred_lengths = [len(pred.split()) for pred in self.predictions]
        ref_lengths = [len(ref.split()) for ref in self.references]
        
        return {
            'pred_avg_length': np.mean(pred_lengths),
            'ref_avg_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0
        }

class DetectionMetrics(MetricCalculator):
    """目标检测任务指标"""
    
    def __init__(self):
        """初始化目标检测指标计算器
        
        设置检测任务类型并配置默认的IoU阈值为0.5，用于计算mAP、精确率、召回率等检测指标。
        """
        super().__init__("detection")
        self.iou_threshold = 0.5
    
    def compute(self) -> Dict[str, float]:
        """计算检测指标
        
        Returns:
            包含mAP、精确率、召回率等指标的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        metrics = {}
        
        # 解析检测结果
        parsed_predictions = [self._parse_detection_result(pred) for pred in self.predictions]
        parsed_references = [self._parse_detection_result(ref) for ref in self.references]
        
        # 计算基本指标
        metrics.update(self._compute_basic_metrics(parsed_predictions, parsed_references))
        
        # 计算 mAP。_compute_map 使用 torchvision 实现轻量 AP，不应依赖 pycocotools。
        try:
            map_score = self._compute_map(parsed_predictions, parsed_references)
            metrics['mAP'] = map_score
        except Exception as e:
            logger.warning(f"mAP计算失败: {e}")
        
        return metrics
    
    def _parse_detection_result(self, result: str) -> List[Dict[str, Any]]:
        """解析检测结果字符串

        支持格式：
        - JSON: [{"label": "cat", "bbox": [...], "confidence": 1.0}]
        - VP ref+box: <|ref|>cat<|/ref|><|box|>[[0,0,100,100]]<|/box|>
        - VP loc tokens: cat<loc_0><loc_0><loc_100><loc_100>
        - Plain markers: <ref>cat</ref><box>[[0,0,100,100]]</box>

        Args:
            result: 检测结果字符串

        Returns:
            检测框列表
        """
        detections = []

        try:
            from .structured_vp_decoder import (
                FlorenceNativeDetectionParser,
                StructuredVisualPrimitiveDecoder,
            )

            text = str(result or "").strip()
            if not text:
                return detections

            # 优先使用 Florence 原生 / 结构化 VP 解析器，支持多词标签和 loc token。
            native_detections = FlorenceNativeDetectionParser().parse(text)
            if native_detections:
                return native_detections

            structured = StructuredVisualPrimitiveDecoder().decode(text)
            if structured.detections:
                return structured.detections

            # 尝试解析JSON格式
            if text.startswith('[') or text.startswith('{'):
                data = json.loads(text)
                if isinstance(data, list):
                    detections = data
                elif isinstance(data, dict) and 'objects' in data:
                    detections = data['objects']
                return detections

            # VP 格式: <|ref|>label<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>
            vp_pattern = r'<\|?ref\|?>([^<]+)<\|?/ref\|?>\s*<\|?box\|?>\s*\[\[([^\]]*)\]\]\s*<\|?/box\|?>'
            vp_matches = re.findall(vp_pattern, text)
            if vp_matches:
                for label, coords_str in vp_matches:
                    coords = [float(x.strip()) for x in coords_str.split(',')]
                    detections.append({
                        'label': label.strip(),
                        'bbox': coords,
                        'confidence': 1.0
                    })
                return detections

            # VP loc token 格式: label<loc_x><loc_y><loc_x2><loc_y2>
            pattern = r'([^<]+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
            matches = re.findall(pattern, text)

            for match in matches:
                label, x1, y1, x2, y2 = match
                detections.append({
                    'label': label.strip(),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': 1.0  # 默认置信度
                })

        except Exception as e:
            logger.warning(f"解析检测结果失败: {e}")

        return detections
    
    def _compute_basic_metrics(
        self,
        predictions: List[List[Dict]],
        references: List[List[Dict]]
    ) -> Dict[str, float]:
        """计算基本检测指标"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred_boxes, ref_boxes in zip(predictions, references):
            tp, fp, fn = self._compute_matches(pred_boxes, ref_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # 计算精确率、召回率、F1分数
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    
    def _compute_matches(
        self,
        pred_boxes: List[Dict],
        ref_boxes: List[Dict]
    ) -> Tuple[int, int, int]:
        """计算预测框和真实框的匹配"""
        if not pred_boxes and not ref_boxes:
            return 0, 0, 0
        
        if not pred_boxes:
            return 0, 0, len(ref_boxes)
        
        if not ref_boxes:
            return 0, len(pred_boxes), 0
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(pred_boxes), len(ref_boxes)))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, ref_box in enumerate(ref_boxes):
                if pred_box.get('label') == ref_box.get('label'):
                    iou = self._compute_iou(pred_box['bbox'], ref_box['bbox'])
                    iou_matrix[i, j] = iou
        
        # 贪心匹配
        matched_pred = set()
        matched_ref = set()
        
        # 按IoU降序排列
        matches = []
        for i in range(len(pred_boxes)):
            for j in range(len(ref_boxes)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True)
        
        tp = 0
        for iou, i, j in matches:
            if i not in matched_pred and j not in matched_ref:
                matched_pred.add(i)
                matched_ref.add(j)
                tp += 1
        
        fp = len(pred_boxes) - tp
        fn = len(ref_boxes) - tp
        
        return tp, fp, fn
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_map(self, predictions: List[List[Dict]], references: List[List[Dict]]) -> float:
        """计算 mAP (Mean Average Precision)

        使用 torchvision 的 box_iou 实现简单的 per-class AP 计算。
        当 pycocotools 不可用时作为轻量替代。
        """
        import torch
        from collections import defaultdict

        if not predictions or not references:
            return 0.0

        try:
            from torchvision.ops import box_iou
        except ImportError:
            logger.warning(
                missing_dependency_message("mAP计算", "torchvision")
            )
            return 0.0

        # 收集所有预测和真实框，按类别和图片分组，避免跨图片错误匹配。
        iou_threshold = 0.5
        all_aps = []

        pred_by_class: Dict[str, List[Tuple[int, torch.Tensor, float]]] = defaultdict(list)
        gt_by_class: Dict[str, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

        for image_idx, (pred_boxes, ref_boxes) in enumerate(zip(predictions, references)):
            for pred in pred_boxes:
                cat = pred.get('category') or pred.get('label') or 'default'
                bbox = pred.get('bbox', [0, 0, 0, 0])
                score = float(pred.get('score', pred.get('confidence', 1.0)))
                pred_by_class[cat].append(
                    (image_idx, torch.tensor(bbox, dtype=torch.float32).unsqueeze(0), score)
                )

            for ref in ref_boxes:
                cat = ref.get('category') or ref.get('label') or 'default'
                bbox = ref.get('bbox', [0, 0, 0, 0])
                gt_by_class[cat][image_idx].append(torch.tensor(bbox, dtype=torch.float32).unsqueeze(0))

        # 如果没有检测到任何类别，使用默认类别
        all_categories = set(pred_by_class.keys()) | set(gt_by_class.keys())
        if not all_categories:
            all_categories = {'default'}
            if not pred_by_class:
                pred_by_class['default'] = []
            if not gt_by_class:
                gt_by_class['default'] = {}

        for cat in all_categories:
            preds = pred_by_class.get(cat, [])
            gts = gt_by_class.get(cat, {})
            total_gts = sum(len(image_gts) for image_gts in gts.values())

            if total_gts == 0:
                continue
            if not preds:
                all_aps.append(0.0)
                continue

            # 按分数降序排列
            preds.sort(key=lambda x: x[2], reverse=True)

            # 简单 AP 计算：对每个预测找最佳匹配
            tp = torch.zeros(len(preds))
            fp = torch.zeros(len(preds))
            matched_gt_by_image: Dict[int, set[int]] = defaultdict(set)

            for i, (image_idx, pred_box, _score) in enumerate(preds):
                image_gt_list = gts.get(image_idx, [])
                if not image_gt_list:
                    fp[i] = 1
                    continue

                image_gt_boxes = torch.cat(image_gt_list)
                ious = box_iou(pred_box, image_gt_boxes).squeeze(0)
                if ious.numel() == 0:
                    fp[i] = 1
                    continue

                best_iou, best_j = ious.max(dim=0)
                best_j_int = int(best_j.item())
                if best_iou >= iou_threshold and best_j_int not in matched_gt_by_image[image_idx]:
                    tp[i] = 1
                    matched_gt_by_image[image_idx].add(best_j_int)
                else:
                    fp[i] = 1

            # 累积 precision/recall
            tp_cumsum = tp.cumsum(dim=0)
            fp_cumsum = fp.cumsum(dim=0)
            recalls = tp_cumsum / total_gts
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

            # 11-point interpolated AP
            ap = 0.0
            for t in torch.linspace(0, 1, 11):
                if (recalls >= t).any():
                    ap += precisions[recalls >= t].max().item()
            all_aps.append(ap / 11.0)

        return float(np.mean(all_aps)) if all_aps else 0.0

class OCRMetrics(MetricCalculator):
    """OCR任务指标"""
    
    def __init__(self):
        """初始化OCR任务指标计算器
        
        设置OCR任务类型，用于计算文本识别的准确率、编辑距离等指标。
        """
        super().__init__("ocr")
    
    def compute(self) -> Dict[str, float]:
        """计算OCR指标
        
        Returns:
            包含字符准确率、词准确率、编辑距离等指标的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        metrics = {}
        
        # 字符级别指标
        metrics.update(self._compute_character_metrics())
        
        # 词级别指标
        metrics.update(self._compute_word_metrics())
        
        # 编辑距离
        metrics['edit_distance'] = self._compute_edit_distance()
        
        return metrics
    
    def _compute_character_metrics(self) -> Dict[str, float]:
        """计算字符级别指标"""
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(self.predictions, self.references):
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)
            
            total_chars += len(ref_clean)
            
            # 计算正确字符数
            for i, char in enumerate(ref_clean):
                if i < len(pred_clean) and pred_clean[i] == char:
                    correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        return {
            'character_accuracy': char_accuracy,
            'total_characters': total_chars,
            'correct_characters': correct_chars
        }
    
    def _compute_word_metrics(self) -> Dict[str, float]:
        """计算词级别指标"""
        total_words = 0
        correct_words = 0
        
        for pred, ref in zip(self.predictions, self.references):
            pred_words = self._clean_text(pred).split()
            ref_words = self._clean_text(ref).split()
            
            total_words += len(ref_words)
            
            # 计算正确词数
            for i, word in enumerate(ref_words):
                if i < len(pred_words) and pred_words[i] == word:
                    correct_words += 1
        
        word_accuracy = correct_words / total_words if total_words > 0 else 0.0
        
        return {
            'word_accuracy': word_accuracy,
            'total_words': total_words,
            'correct_words': correct_words
        }
    
    def _compute_edit_distance(self) -> float:
        """计算平均编辑距离"""
        distances = []
        
        for pred, ref in zip(self.predictions, self.references):
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)
            
            distance = self._levenshtein_distance(pred_clean, ref_clean)
            distances.append(distance)
        
        return np.mean(distances)
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空格，转换为小写
        return ' '.join(text.lower().split())
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class SegmentationMetrics(MetricCalculator):
    """分割任务指标"""
    
    def __init__(self):
        """初始化分割任务指标计算器
        
        设置分割任务类型，用于计算IoU、Dice系数、像素准确率等分割指标。
        """
        super().__init__("segmentation")
    
    def compute(self) -> Dict[str, float]:
        """计算分割指标
        
        Returns:
            包含IoU、Dice系数等指标的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        metrics = {}
        
        # 解析分割结果
        parsed_predictions = [self._parse_segmentation_result(pred) for pred in self.predictions]
        parsed_references = [self._parse_segmentation_result(ref) for ref in self.references]
        
        # 计算IoU和Dice系数
        metrics.update(self._compute_segmentation_metrics(parsed_predictions, parsed_references))
        
        return metrics
    
    def _parse_segmentation_result(self, result: str) -> Optional[np.ndarray]:
        """解析分割结果
        
        Args:
            result: 分割结果字符串
            
        Returns:
            分割掩码数组
        """
        try:
            # 尝试解析多边形格式
            if '<poly>' in result:
                # 提取多边形坐标
                poly_pattern = r'<poly>(.*?)</poly>'
                matches = re.findall(poly_pattern, result)
                
                if matches and CV2_AVAILABLE:
                    # 这里需要图像尺寸信息，简化处理
                    # 实际应用中需要传入图像尺寸
                    return self._polygon_to_mask(matches[0], (224, 224))
            
            # 尝试解析其他格式
            # ...
            
        except Exception as e:
            logger.warning(f"解析分割结果失败: {e}")
        
        return None
    
    def _polygon_to_mask(self, polygon_str: str, image_size: Tuple[int, int]) -> np.ndarray:
        """将多边形转换为掩码"""
        if not CV2_AVAILABLE:
            return np.zeros(image_size, dtype=np.uint8)
        
        # 解析多边形坐标
        coords = []
        coord_pattern = r'(\d+),(\d+)'
        matches = re.findall(coord_pattern, polygon_str)
        
        for x, y in matches:
            coords.append([int(x), int(y)])
        
        if not coords:
            return np.zeros(image_size, dtype=np.uint8)
        
        # 创建掩码
        mask = np.zeros(image_size, dtype=np.uint8)
        coords_array = np.array(coords, dtype=np.int32)
        cv2.fillPoly(mask, [coords_array], 1)
        
        return mask
    
    def _compute_segmentation_metrics(
        self,
        predictions: List[Optional[np.ndarray]],
        references: List[Optional[np.ndarray]]
    ) -> Dict[str, float]:
        """计算分割指标"""
        ious = []
        dice_scores = []
        
        for pred_mask, ref_mask in zip(predictions, references):
            if pred_mask is None or ref_mask is None:
                continue
            
            # 确保掩码尺寸一致
            if pred_mask.shape != ref_mask.shape:
                continue
            
            # 计算IoU
            intersection = np.logical_and(pred_mask, ref_mask).sum()
            union = np.logical_or(pred_mask, ref_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                
                # 计算Dice系数
                dice = 2 * intersection / (pred_mask.sum() + ref_mask.sum())
                dice_scores.append(dice)
        
        return {
            'mean_iou': np.mean(ious) if ious else 0.0,
            'mean_dice': np.mean(dice_scores) if dice_scores else 0.0,
            'num_valid_samples': len(ious)
        }

def get_metric_calculator(task_type: str) -> MetricCalculator:
    """根据任务类型获取对应的指标计算器
    
    Args:
        task_type: 任务类型
        
    Returns:
        指标计算器实例
    """
    task_type_lower = task_type.lower()

    detection_aliases = {
        "od",
        "open_vocabulary_detection",
        "region_proposal",
        "phrase_grounding",
        "caption_to_phrase_grounding",
    }
    segmentation_aliases = {
        "region_to_segmentation",
        "referring_expression_segmentation",
        "seg",
        "segmentation",
    }

    if 'caption' in task_type_lower or 'description' in task_type_lower:
        return CaptionMetrics()
    elif '_vp' in task_type_lower or 'visual_primitive' in task_type_lower:
        return VisualPrimitiveDetectionMetrics()
    elif (
        'detection' in task_type_lower
        or 'object' in task_type_lower
        or task_type_lower in detection_aliases
    ):
        return DetectionMetrics()
    elif 'ocr' in task_type_lower or 'text' in task_type_lower:
        return OCRMetrics()
    elif (
        'segmentation' in task_type_lower
        or 'segment' in task_type_lower
        or task_type_lower in segmentation_aliases
    ):
        return SegmentationMetrics()
    else:
        # 默认使用基础指标计算器
        return MetricCalculator(task_type)


class VisualPrimitiveDetectionMetrics(DetectionMetrics):
    """视觉原语检测指标计算器

    扩展 DetectionMetrics，增加 VP 格式质量、坐标有效性、
    ref 覆盖率、结构化解码等指标。
    """

    def __init__(self, task_type: str = "OD_VP"):
        super().__init__()
        self.task_type = task_type
        self._vp_format_valid = 0
        self._vp_coordinate_valid = 0
        self._vp_ref_covered = 0
        self._vp_pred_box_counts: List[int] = []
        self._vp_ref_box_counts: List[int] = []
        self._vp_box_count_exact_match = 0
        self._structured_vp_format_valid = 0
        self._structured_vp_decoder_ok = 0
        self._structured_vp_source_florence_native = 0
        self._structured_box_count_exact_match = 0
        self._structured_pred_box_counts: List[int] = []
        self._structured_ref_box_counts: List[int] = []
        self._structured_parsed_preds: List[List[Dict]] = []
        self._structured_parsed_refs: List[List[Dict]] = []

    def add_batch(self, predictions: List[str], references: List[str]) -> None:
        """添加一批预测和参考结果，同时收集 VP 格式质量统计。"""
        super().add_batch(predictions, references)
        from .visual_primitive_parser import VisualPrimitiveParser
        from .structured_vp_decoder import (
            StructuredVisualPrimitiveDecoder,
            FlorenceNativeDetectionParser,
        )

        vp_parser = VisualPrimitiveParser()
        native_parser = FlorenceNativeDetectionParser()
        structured_decoder = StructuredVisualPrimitiveDecoder()

        for pred, ref in zip(predictions, references):
            # --- VP 格式质量 ---
            pred_dets = vp_parser.parse_detections(pred)
            ref_dets = vp_parser.parse_detections(ref)

            if pred_dets:
                self._vp_format_valid += 1
            if ref_dets:
                self._vp_ref_covered += 1

            # 坐标有效性
            pred_coords_valid = all(
                all(isinstance(v, (int, float)) and v >= 0 for v in d.get("bbox", []))
                for d in pred_dets
            )
            if pred_dets and pred_coords_valid:
                self._vp_coordinate_valid += 1

            self._vp_pred_box_counts.append(len(pred_dets))
            self._vp_ref_box_counts.append(len(ref_dets))
            if len(pred_dets) == len(ref_dets):
                self._vp_box_count_exact_match += 1

            # --- 结构化 VP 解码 ---
            native_dets = native_parser.parse(pred)
            structured = structured_decoder.decode(pred)

            if native_dets or structured:
                self._structured_vp_format_valid += 1
            if structured:
                self._structured_vp_decoder_ok += 1
            if native_dets:
                self._structured_vp_source_florence_native += 1

            # 提取结构化检测结果列表
            if native_dets:
                struct_pred_dets = native_dets
            elif hasattr(structured, 'detections'):
                struct_pred_dets = structured.detections
            else:
                struct_pred_dets = []
            struct_ref_dets = ref_dets

            self._structured_pred_box_counts.append(len(struct_pred_dets))
            self._structured_ref_box_counts.append(len(struct_ref_dets))
            if len(struct_pred_dets) == len(struct_ref_dets):
                self._structured_box_count_exact_match += 1

            self._structured_parsed_preds.append(struct_pred_dets)
            self._structured_parsed_refs.append(struct_ref_dets)

    def compute(self) -> Dict[str, float]:
        """计算检测指标 + VP 格式质量指标。"""
        metrics = super().compute()
        n = len(self.predictions)
        if n == 0:
            return metrics

        # VP 格式质量
        metrics["vp_format_valid_ratio"] = self._vp_format_valid / n
        metrics["vp_coordinate_valid_ratio"] = self._vp_coordinate_valid / n
        metrics["vp_ref_coverage_ratio"] = self._vp_ref_covered / n
        metrics["vp_avg_pred_boxes"] = (
            sum(self._vp_pred_box_counts) / n if n else 0.0
        )
        metrics["vp_box_count_exact_match"] = self._vp_box_count_exact_match / n

        # 结构化 VP 指标
        metrics["structured_vp_format_valid_ratio"] = self._structured_vp_format_valid / n
        metrics["structured_vp_decoder_ratio"] = self._structured_vp_decoder_ok / n
        metrics["structured_vp_source_florence_native_ratio"] = (
            self._structured_vp_source_florence_native / n
        )
        metrics["structured_vp_box_count_exact_match"] = (
            self._structured_box_count_exact_match / n
        )

        # 结构化 precision / recall
        if self._structured_parsed_preds and self._structured_parsed_refs:
            total_pred = sum(len(p) for p in self._structured_parsed_preds)
            total_ref = sum(len(r) for r in self._structured_parsed_refs)
            matched = 0
            for preds, refs in zip(
                self._structured_parsed_preds, self._structured_parsed_refs
            ):
                for p in preds:
                    for r in refs:
                        if (
                            p.get("label", "").lower() == r.get("label", "").lower()
                            and self._bbox_iou(p.get("bbox", []), r.get("bbox", [])) > 0.5
                        ):
                            matched += 1
                            break
            metrics["structured_precision"] = matched / total_pred if total_pred else 0.0
            metrics["structured_recall"] = matched / total_ref if total_ref else 0.0

        return metrics

    @staticmethod
    def _bbox_iou(box1: List, box2: List) -> float:
        """计算两个 bbox 的 IoU。"""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
