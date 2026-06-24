"""目标检测高级评估指标

提供完整的目标检测评估功能，包括IoU、AP、mAP等指标计算
"""

import logging
import warnings
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

from ...utils.optional_dependencies import missing_dependency_message

try:
    from sklearn.metrics import average_precision_score  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("部分目标检测高级指标", "scikit-learn")
    )

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    image_id: Optional[str] = None

@dataclass
class GroundTruthBox:
    """真实标注框数据类"""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    image_id: Optional[str] = None
    difficult: bool = False  # 是否为困难样本

class ObjectDetectionMetrics:
    """目标检测评估指标
    
    提供完整的目标检测评估功能，包括IoU、AP、mAP等指标计算
    """
    
    def __init__(
        self,
        iou_thresholds: Union[float, List[float]] = 0.5,
        confidence_threshold: float = 0.0,
        max_detections: int = 100,
        area_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """初始化目标检测指标计算器
        
        Args:
            iou_thresholds: IoU阈值，可以是单个值或列表
            confidence_threshold: 置信度阈值
            max_detections: 最大检测数量
            area_ranges: 不同尺度的面积范围
        """
        if isinstance(iou_thresholds, (int, float)):
            self.iou_thresholds = [float(iou_thresholds)]
        else:
            self.iou_thresholds = list(iou_thresholds)
        
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        
        # 默认面积范围（COCO标准）
        self.area_ranges = area_ranges or {
            "all": (0, float('inf')),
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float('inf'))
        }
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_box_area(self, bbox: List[float]) -> float:
        """计算边界框面积"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def filter_detections_by_confidence(
        self,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """根据置信度过滤检测结果"""
        filtered = [
            det for det in detections 
            if det.confidence >= self.confidence_threshold
        ]
        
        # 按置信度排序并限制数量
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        return filtered[:self.max_detections]
    
    def match_detections_to_ground_truth(
        self,
        detections: List[DetectionResult],
        ground_truths: List[GroundTruthBox],
        iou_threshold: float
    ) -> Tuple[List[bool], List[bool]]:
        """将检测结果与真实标注进行匹配
        
        Args:
            detections: 检测结果列表（已按置信度排序）
            ground_truths: 真实标注列表
            iou_threshold: IoU阈值
            
        Returns:
            (tp_flags, gt_matched): 检测结果的TP标记和GT匹配标记
        """
        tp_flags = [False] * len(detections)
        gt_matched = [False] * len(ground_truths)
        
        for det_idx, detection in enumerate(detections):
            best_iou = 0.0
            best_gt_idx = -1
            
            # 找到最佳匹配的GT
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx] or gt.class_id != detection.class_id:
                    continue
                
                iou = self.calculate_iou(detection.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 如果IoU超过阈值，标记为TP
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_flags[det_idx] = True
                gt_matched[best_gt_idx] = True
        
        return tp_flags, gt_matched
    
    def calculate_precision_recall_curve(
        self,
        detections: List[DetectionResult],
        ground_truths: List[GroundTruthBox],
        iou_threshold: float,
        class_id: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算精确率-召回率曲线
        
        Args:
            detections: 检测结果列表
            ground_truths: 真实标注列表
            iou_threshold: IoU阈值
            class_id: 类别ID（如果指定，只计算该类别）
            
        Returns:
            (precision, recall, confidence): 精确率、召回率、置信度数组
        """
        # 过滤特定类别
        if class_id is not None:
            detections = [det for det in detections if det.class_id == class_id]
            ground_truths = [gt for gt in ground_truths if gt.class_id == class_id]
        
        # 过滤置信度并排序
        detections = self.filter_detections_by_confidence(detections)
        
        # 计算总的GT数量（排除困难样本）
        total_gt = sum(1 for gt in ground_truths if not gt.difficult)
        
        if total_gt == 0:
            return np.array([1.0]), np.array([0.0]), np.array([1.0])
        
        # 匹配检测结果与GT
        tp_flags, _ = self.match_detections_to_ground_truth(
            detections, ground_truths, iou_threshold
        )
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp_flags)
        fp_cumsum = np.cumsum([not flag for flag in tp_flags])
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / total_gt
        
        # 添加起始点
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        confidence = np.array([det.confidence for det in detections])
        confidence = np.concatenate([[1.0], confidence])
        
        return precision, recall, confidence
    
    def calculate_ap(
        self,
        detections: List[DetectionResult],
        ground_truths: List[GroundTruthBox],
        iou_threshold: float = 0.5,
        class_id: Optional[int] = None,
        interpolation: str = "11point"
    ) -> Dict[str, float]:
        """计算平均精度(AP)
        
        Args:
            detections: 检测结果列表
            ground_truths: 真实标注列表
            iou_threshold: IoU阈值
            class_id: 类别ID
            interpolation: 插值方法 ('11point', 'all')
            
        Returns:
            AP结果字典
        """
        try:
            precision, recall, confidence = self.calculate_precision_recall_curve(
                detections, ground_truths, iou_threshold, class_id
            )
            
            if interpolation == "11point":
                # 11点插值法
                ap = 0.0
                for t in np.arange(0, 1.1, 0.1):
                    if np.sum(recall >= t) == 0:
                        p = 0
                    else:
                        p = np.max(precision[recall >= t])
                    ap += p / 11.0
            else:
                # 所有点插值法（COCO标准）
                # 单调递减插值
                for i in range(len(precision) - 2, -1, -1):
                    precision[i] = max(precision[i], precision[i + 1])
                
                # 计算面积
                indices = np.where(recall[1:] != recall[:-1])[0]
                ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
            
            return {
                "ap": float(ap),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "confidence": confidence.tolist(),
                "iou_threshold": iou_threshold,
                "interpolation": interpolation
            }
            
        except Exception as e:
            logger.error(f"AP计算失败: {e}")
            return {"ap": 0.0}
    
    def calculate_map(
        self,
        predictions: Dict[str, List[DetectionResult]],
        ground_truths: Dict[str, List[GroundTruthBox]],
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """计算平均AP(mAP)
        
        Args:
            predictions: 预测结果字典 {image_id: [DetectionResult, ...]}
            ground_truths: 真实标注字典 {image_id: [GroundTruthBox, ...]}
            class_names: 类别名称列表
            iou_thresholds: IoU阈值列表
            
        Returns:
            mAP结果字典
        """
        if iou_thresholds is None:
            iou_thresholds = self.iou_thresholds
        
        try:
            # 合并所有图像的检测结果和GT
            all_detections = []
            all_ground_truths = []
            
            for image_id in set(list(predictions.keys()) + list(ground_truths.keys())):
                if image_id in predictions:
                    for det in predictions[image_id]:
                        det.image_id = image_id
                        all_detections.append(det)
                
                if image_id in ground_truths:
                    for gt in ground_truths[image_id]:
                        gt.image_id = image_id
                        all_ground_truths.append(gt)
            
            # 获取所有类别
            all_classes = set()
            for det in all_detections:
                all_classes.add(det.class_id)
            for gt in all_ground_truths:
                all_classes.add(gt.class_id)
            all_classes = sorted(list(all_classes))
            
            # 计算每个类别在每个IoU阈值下的AP
            results = {
                "class_ap": {},
                "iou_ap": {},
                "overall_map": 0.0,
                "class_names": class_names or [f"class_{i}" for i in all_classes],
                "iou_thresholds": iou_thresholds
            }
            
            all_aps = []
            
            for iou_threshold in iou_thresholds:
                iou_aps = []
                
                for class_id in all_classes:
                    ap_result = self.calculate_ap(
                        all_detections,
                        all_ground_truths,
                        iou_threshold,
                        class_id
                    )
                    
                    ap_value = ap_result["ap"]
                    iou_aps.append(ap_value)
                    
                    # 存储类别AP
                    if class_id not in results["class_ap"]:
                        results["class_ap"][class_id] = []
                    results["class_ap"][class_id].append(ap_value)
                
                # 存储IoU阈值下的mAP
                iou_map = np.mean(iou_aps) if iou_aps else 0.0
                results["iou_ap"][iou_threshold] = iou_map
                all_aps.extend(iou_aps)
            
            # 计算总体mAP
            results["overall_map"] = np.mean(all_aps) if all_aps else 0.0
            
            # 计算每个类别的平均AP（跨所有IoU阈值）
            for class_id in all_classes:
                class_aps = results["class_ap"][class_id]
                results["class_ap"][class_id] = {
                    "individual_aps": class_aps,
                    "mean_ap": np.mean(class_aps) if class_aps else 0.0
                }
            
            logger.info(f"mAP计算完成，总体mAP: {results['overall_map']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"mAP计算失败: {e}")
            return {"overall_map": 0.0}
    
    def calculate_coco_metrics(
        self,
        predictions: Dict[str, List[DetectionResult]],
        ground_truths: Dict[str, List[GroundTruthBox]]
    ) -> Dict[str, float]:
        """计算COCO标准评估指标
        
        Returns:
            COCO评估指标字典
        """
        # COCO标准IoU阈值
        coco_iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
        
        # 计算mAP
        map_result = self.calculate_map(
            predictions, ground_truths, iou_thresholds=coco_iou_thresholds
        )
        
        # 计算不同IoU阈值下的mAP
        map_50 = self.calculate_map(
            predictions, ground_truths, iou_thresholds=[0.5]
        )["overall_map"]
        
        map_75 = self.calculate_map(
            predictions, ground_truths, iou_thresholds=[0.75]
        )["overall_map"]
        
        # 计算不同尺度下的mAP
        scale_maps = {}
        for scale_name, (min_area, max_area) in self.area_ranges.items():
            if scale_name == "all":
                continue
            
            # 过滤不同尺度的GT
            filtered_gt = {}
            for image_id, gts in ground_truths.items():
                filtered_gt[image_id] = [
                    gt for gt in gts
                    if min_area <= self.calculate_box_area(gt.bbox) < max_area
                ]
            
            if any(len(gts) > 0 for gts in filtered_gt.values()):
                scale_map = self.calculate_map(
                    predictions, filtered_gt, iou_thresholds=coco_iou_thresholds
                )["overall_map"]
                scale_maps[f"map_{scale_name}"] = scale_map
        
        return {
            "map": map_result["overall_map"],
            "map_50": map_50,
            "map_75": map_75,
            **scale_maps
        }
