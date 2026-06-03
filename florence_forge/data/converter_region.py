"""区域/定位/分割等 JSON·COCO·CSV 转换。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .converter_mask import generate_mask_from_polygon

logger = logging.getLogger(__name__)


def json_to_florence2_grounding(json_path: str, output_path: str, 
                               image_dir: str = "") -> None:
    """将JSON格式的视觉定位数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换视觉定位数据: {json_path} -> {output_path}")
    
    # 转换为绝对路径
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON grounding转Florence-2进度"):
            if 'phrases' in item and item['phrases']:
                bboxes = [phrase['bbox'] for phrase in item['phrases']]
                labels = [phrase['phrase'] for phrase in item['phrases']]
                
                answer = {
                    '<CAPTION_TO_PHRASE_GROUNDING>': {
                        'bboxes': bboxes,
                        'labels': labels
                    }
                }
                
                # 使用绝对路径
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<CAPTION_TO_PHRASE_GROUNDING>',
                    'suffix': json.dumps(answer, ensure_ascii=False),
                    'text_input': item['caption']
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("视觉定位数据转换完成")

def coco_to_florence2_region_segmentation(coco_json_path: str, output_path: str, 
                                         image_dir: str, mask_dir: str) -> None:
    """将COCO格式的分割数据转换为Florence-2区域分割格式
    
    Args:
        coco_json_path: COCO JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
        mask_dir: 掩码文件目录
    """
    logger.info(f"转换COCO分割数据: {coco_json_path} -> {output_path}")
    
    coco_json_path = Path(coco_json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute()
    mask_dir = Path(mask_dir).absolute()
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像映射
    images = {img['id']: img for img in coco_data['images']}
    
    # 按图像组织标注
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for image_id, annotations in tqdm(image_annotations.items(), desc="COCO转Florence-2区域分割进度"):
            image_info = images[image_id]
            image_path = image_dir / image_info['file_name']
            
            bboxes = []
            masks = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                # 转换为[x1, y1, x2, y2]格式
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                bboxes.append([x1, y1, x2, y2])
                
                # 处理 segmentation
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    # 假设每个 annotation 只有一个多边形
                    polygon = ann['segmentation'][0]
                    # 生成掩码文件名
                    mask_filename = f"mask_{ann['id']}.png"
                    # 生成掩码
                    mask_path = generate_mask_from_polygon(
                        polygon, 
                        (image_info['width'], image_info['height']), 
                        str(mask_dir), 
                        mask_filename
                    )
                    masks.append(mask_path)
                else:
                    masks.append(None)
            
            answer = {
                '<REGION_TO_SEGMENTATION>': {
                    'bboxes': bboxes,
                    'masks': masks
                }
            }
            
            sample = {
                'image': str(image_path.absolute()),
                'prefix': '<REGION_TO_SEGMENTATION>',
                'suffix': json.dumps(answer, ensure_ascii=False)
            }
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("COCO分割数据转换完成")

def csv_to_florence2_region_category(csv_path: str, output_path: str, 
                                    image_dir: str = "") -> None:
    """将CSV格式的区域分类数据转换为Florence-2格式
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换区域分类数据: {csv_path} -> {output_path}")
    
    csv_path = Path(csv_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    df = pd.read_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV区域分类转Florence-2进度"):
            image_path = image_dir / row['image']
            
            sample = {
                'image': str(image_path.absolute()),
                'prefix': '<REGION_TO_CATEGORY>',
                'suffix': row['category'],
                'region': row['region']
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("区域分类数据转换完成")

def csv_to_florence2_region_description(csv_path: str, output_path: str, 
                                       image_dir: str = "") -> None:
    """将CSV格式的区域描述数据转换为Florence-2格式
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换区域描述数据: {csv_path} -> {output_path}")
    
    csv_path = Path(csv_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    df = pd.read_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV区域描述转Florence-2进度"):
            image_path = image_dir / row['image']
            
            sample = {
                'image': str(image_path.absolute()),
                'prefix': '<REGION_TO_DESCRIPTION>',
                'suffix': row['description'],
                'region': row['region']
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("区域描述数据转换完成")

def json_to_florence2_region_proposal(json_path: str, output_path: str, 
                                     image_dir: str = "") -> None:
    """将JSON格式的区域提议数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换区域提议数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON区域提议转Florence-2进度"):
            if 'regions' in item and item['regions']:
                answer = {
                    '<REGION_PROPOSAL>': {
                        'bboxes': item['regions']
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<REGION_PROPOSAL>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("区域提议数据转换完成")

def json_to_florence2_ocr_with_region(json_path: str, output_path: str, 
                                     image_dir: str = "") -> None:
    """将JSON格式的区域OCR数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换区域OCR数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON区域OCR转Florence-2进度"):
            if 'ocr_results' in item and item['ocr_results']:
                bboxes = [result['bbox'] for result in item['ocr_results']]
                labels = [result['text'] for result in item['ocr_results']]
                
                answer = {
                    '<OCR_WITH_REGION>': {
                        'quad_boxes': bboxes,
                        'labels': labels
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<OCR_WITH_REGION>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("区域OCR数据转换完成")

def json_to_florence2_referring_expression_segmentation(json_path: str, output_path: str, 
                                                       image_dir: str = "", 
                                                       mask_dir: str = "") -> None:
    """将JSON格式的引用表达分割数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
        mask_dir: 掩码文件目录
    """
    logger.info(f"转换引用表达分割数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    mask_dir = Path(mask_dir).absolute() if mask_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON引用表达分割转Florence-2进度"):
            if 'polygons' in item and item['polygons']:
                # 生成掩码文件
                mask_filename = f"mask_{item.get('id', 'unknown')}.png"
                mask_path = None
                
                if mask_dir and 'image_size' in item:
                    mask_path = generate_mask_from_polygon(
                        item['polygons'][0],  # 假设只有一个多边形
                        item['image_size'],
                        str(mask_dir),
                        mask_filename
                    )
                
                answer = {
                    '<REFERRING_EXPRESSION_SEGMENTATION>': {
                        'polygons': item['polygons'],
                        'mask': mask_path
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<REFERRING_EXPRESSION_SEGMENTATION>',
                    'suffix': json.dumps(answer, ensure_ascii=False),
                    'text_input': item['expression']
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("引用表达分割数据转换完成")

def json_to_florence2_region_segmentation(json_path: str, output_path: str, 
                                         image_dir: str = "", 
                                         mask_dir: str = "") -> None:
    """将JSON格式的区域分割数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
        mask_dir: 掩码文件目录
    """
    logger.info(f"转换区域分割数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    mask_dir = Path(mask_dir).absolute() if mask_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON区域分割转Florence-2进度"):
            if 'regions' in item and item['regions']:
                bboxes = []
                masks = []
                
                for i, region in enumerate(item['regions']):
                    bboxes.append(region['bbox'])
                    
                    # 生成掩码文件
                    if 'polygon' in region and mask_dir and 'image_size' in item:
                        mask_filename = f"mask_{item.get('id', 'unknown')}_{i}.png"
                        mask_path = generate_mask_from_polygon(
                            region['polygon'],
                            item['image_size'],
                            str(mask_dir),
                            mask_filename
                        )
                        masks.append(mask_path)
                    else:
                        masks.append(None)
                
                answer = {
                    '<REGION_TO_SEGMENTATION>': {
                        'bboxes': bboxes,
                        'masks': masks
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<REGION_TO_SEGMENTATION>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("区域分割数据转换完成")

def json_to_florence2_dense_region_caption(json_path: str, output_path: str, 
                                          image_dir: str = "") -> None:
    """将JSON格式的密集区域描述数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换密集区域描述数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON密集区域描述转Florence-2进度"):
            if 'regions' in item and item['regions']:
                bboxes = [region['bbox'] for region in item['regions']]
                labels = [region['description'] for region in item['regions']]
                
                answer = {
                    '<DENSE_REGION_CAPTION>': {
                        'bboxes': bboxes,
                        'labels': labels
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<DENSE_REGION_CAPTION>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("密集区域描述数据转换完成")

def json_to_florence2_open_vocabulary_detection(json_path: str, output_path: str, 
                                               image_dir: str = "") -> None:
    """将JSON格式的开放词汇检测数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换开放词汇检测数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON开放词汇检测转Florence-2进度"):
            if 'detections' in item and item['detections']:
                bboxes = [det['bbox'] for det in item['detections']]
                labels = [det['label'] for det in item['detections']]
                
                answer = {
                    '<OPEN_VOCABULARY_DETECTION>': {
                        'bboxes': bboxes,
                        'bboxes_labels': labels
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<OPEN_VOCABULARY_DETECTION>',
                    'suffix': json.dumps(answer, ensure_ascii=False),
                    'text_input': item['text']
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("开放词汇检测数据转换完成")

def json_to_florence2_detection_with_confidence(json_path: str, output_path: str, 
                                               image_dir: str = "") -> None:
    """将JSON格式的带置信度检测数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换带置信度检测数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON带置信度检测转Florence-2进度"):
            if 'detections' in item and item['detections']:
                bboxes = [det['bbox'] for det in item['detections']]
                labels = [det['label'] for det in item['detections']]
                confidences = [det.get('confidence', 1.0) for det in item['detections']]
                
                answer = {
                    '<OD>': {
                        'bboxes': bboxes,
                        'bboxes_labels': labels,
                        'confidences': confidences
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<OD>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("带置信度检测数据转换完成")

def json_to_florence2_grounding_with_confidence(json_path: str, output_path: str, 
                                               image_dir: str = "") -> None:
    """将JSON格式的带置信度视觉定位数据转换为Florence-2格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
    """
    logger.info(f"转换带置信度视觉定位数据: {json_path} -> {output_path}")
    
    json_path = Path(json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="JSON带置信度视觉定位转Florence-2进度"):
            if 'phrases' in item and item['phrases']:
                bboxes = [phrase['bbox'] for phrase in item['phrases']]
                labels = [phrase['phrase'] for phrase in item['phrases']]
                confidences = [phrase.get('confidence', 1.0) for phrase in item['phrases']]
                
                answer = {
                    '<CAPTION_TO_PHRASE_GROUNDING>': {
                        'bboxes': bboxes,
                        'labels': labels,
                        'confidences': confidences
                    }
                }
                
                image_path = image_dir / item['image']
                
                sample = {
                    'image': str(image_path.absolute()),
                    'prefix': '<CAPTION_TO_PHRASE_GROUNDING>',
                    'suffix': json.dumps(answer, ensure_ascii=False),
                    'text_input': item['caption']
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("带置信度视觉定位数据转换完成")
