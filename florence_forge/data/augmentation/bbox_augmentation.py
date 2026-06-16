"""边界框数据增强"""

import random
from typing import Dict, List, Tuple


class BBoxAugmentation:
    """边界框数据增强，适用于目标检测 / 区域分析任务的 bbox 坐标变换。"""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def random_jitter(
        self,
        bboxes: List[Dict[str, float]],
        jitter_range: float = 0.02,
    ) -> List[Dict[str, float]]:
        """对每个 bbox 施加微小随机抖动。"""
        if random.random() < self.probability:
            jittered = []
            for bb in bboxes:
                new_bb = dict(bb)
                for key in ("xmin", "ymin", "xmax", "ymax"):
                    if key in new_bb:
                        delta = random.uniform(-jitter_range, jitter_range)
                        new_bb[key] = max(0.0, min(1.0, new_bb[key] + delta))
                jittered.append(new_bb)
            return jittered
        return bboxes

    def random_scale(
        self,
        bboxes: List[Dict[str, float]],
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> List[Dict[str, float]]:
        """随机缩放 bbox 尺寸（保持中心不变）。"""
        if random.random() < self.probability:
            scaled = []
            for bb in bboxes:
                cx = (bb.get("xmin", 0) + bb.get("xmax", 1)) / 2
                cy = (bb.get("ymin", 0) + bb.get("ymax", 1)) / 2
                factor = random.uniform(*scale_range)
                hw = (bb.get("xmax", 1) - bb.get("xmin", 0)) * factor / 2
                hh = (bb.get("ymax", 1) - bb.get("ymin", 0)) * factor / 2
                scaled.append({
                    "xmin": max(0.0, cx - hw),
                    "ymin": max(0.0, cy - hh),
                    "xmax": min(1.0, cx + hw),
                    "ymax": min(1.0, cy + hh),
                })
            return scaled
        return bboxes

    def random_drop(
        self,
        bboxes: List[Dict[str, float]],
        drop_prob: float = 0.1,
    ) -> List[Dict[str, float]]:
        """随机丢弃部分 bbox。"""
        if random.random() < self.probability:
            kept = [bb for bb in bboxes if random.random() > drop_prob]
            return kept if kept else bboxes
        return bboxes

    def apply_augmentations(
        self,
        bboxes: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """应用所有 bbox 增强。"""
        bboxes = self.random_jitter(bboxes)
        bboxes = self.random_scale(bboxes)
        bboxes = self.random_drop(bboxes)
        return bboxes
