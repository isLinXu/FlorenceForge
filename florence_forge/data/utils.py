#!/usr/bin/env python3
"""
Florence Forge - 数据处理工具函数

提供数据处理相关的工具函数
"""

import torch
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw

def generate_mask_from_polygon(
    polygon: List[Tuple[float, float]], 
    image_size: Tuple[int, int]
) -> np.ndarray:
    """从多边形坐标生成掩码
    
    Args:
        polygon: 多边形顶点坐标列表 [(x1, y1), (x2, y2), ...]
        image_size: 图像尺寸 (width, height)
    
    Returns:
        二值掩码数组，形状为 (height, width)
    """
    width, height = image_size
    
    # 创建空白图像
    mask_image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    
    # 绘制多边形
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=255)
    
    # 转换为numpy数组
    mask = np.array(mask_image)
    
    return mask

def normalize_bbox(
    bbox: List[float], 
    image_size: Tuple[int, int]
) -> List[float]:
    """归一化边界框坐标
    
    Args:
        bbox: 边界框坐标 [x1, y1, x2, y2]
        image_size: 图像尺寸 (width, height)
    
    Returns:
        归一化后的边界框坐标
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox
    
    return [
        x1 / width,
        y1 / height, 
        x2 / width,
        y2 / height
    ]

def denormalize_bbox(
    bbox: List[float],
    image_size: Tuple[int, int]
) -> List[float]:
    """反归一化边界框坐标
    
    Args:
        bbox: 归一化的边界框坐标 [x1, y1, x2, y2]
        image_size: 图像尺寸 (width, height)
    
    Returns:
        实际像素坐标的边界框
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox
    
    return [
        x1 * width,
        y1 * height,
        x2 * width, 
        y2 * height
    ]

def resize_image_and_annotations(
    image: Image.Image,
    annotations: dict,
    target_size: Tuple[int, int]
) -> Tuple[Image.Image, dict]:
    """调整图像大小并相应调整标注
    
    Args:
        image: PIL图像
        annotations: 标注字典
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的图像和标注
    """
    original_size = image.size
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 计算缩放比例
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    # 调整标注
    updated_annotations = annotations.copy()
    
    # 调整边界框
    if 'bboxes' in annotations:
        updated_bboxes = []
        for bbox in annotations['bboxes']:
            x1, y1, x2, y2 = bbox
            updated_bboxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])
        updated_annotations['bboxes'] = updated_bboxes
    
    # 调整多边形
    if 'polygons' in annotations:
        updated_polygons = []
        for polygon in annotations['polygons']:
            updated_polygon = [
                (x * scale_x, y * scale_y) for x, y in polygon
            ]
            updated_polygons.append(updated_polygon)
        updated_annotations['polygons'] = updated_polygons
    
    return resized_image, updated_annotations

def collate_batch_data(batch_data: List[dict]) -> dict:
    """整理批次数据
    
    Args:
        batch_data: 批次数据列表
    
    Returns:
        整理后的批次数据字典
    """
    if not batch_data:
        return {}
    
    # 获取所有键
    keys = batch_data[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch_data]
        
        # 根据数据类型进行不同的处理
        if isinstance(values[0], torch.Tensor):
            # 对于张量，尝试堆叠
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                # 如果无法堆叠，保持列表形式
                collated[key] = values
        elif isinstance(values[0], (list, tuple)):
            # 对于列表或元组，保持原样
            collated[key] = values
        else:
            # 对于其他类型，尝试转换为张量
            try:
                collated[key] = torch.tensor(values)
            except (ValueError, TypeError):
                collated[key] = values
    
    return collated