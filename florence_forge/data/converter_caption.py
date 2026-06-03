"""字幕/描述格式转换（COCO Caption / CSV）。"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def coco_caption_to_florence2(coco_json_path: str, output_path: str, 
                            image_dir: str, task_type: str = 'CAPTION') -> None:
    """将COCO格式的字幕数据转换为Florence-2格式
    
    Args:
        coco_json_path: COCO JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
        task_type: 任务类型
    """
    logger.info(f"转换COCO字幕数据: {coco_json_path} -> {output_path}")
    
    coco_json_path = Path(coco_json_path).resolve()
    output_path = Path(output_path).resolve()
    image_dir = Path(image_dir).resolve()
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_id, annotations in tqdm(image_annotations.items(), desc="COCO Caption to Florence-2"):
            if image_id not in images:
                continue

            image_info = images[image_id]
            image_path = image_dir / image_info['file_name']

            if not image_path.exists():
                continue

            for ann in annotations:
                caption = ann['caption']
                
                sample = {
                    'image': str(image_path.resolve()),
                    'prefix': f'<{task_type}>',
                    'suffix': caption
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("COCO字幕数据转换完成")

def csv_caption_to_florence2(csv_path: str, output_path: str, 
                           image_column: str = 'image_path', 
                            caption_column: str = 'caption', 
                            task_type: str = 'CAPTION') -> None:
    """将CSV格式的图像标题数据转换为Florence-2格式
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出文件路径
        image_column: 图像列名
        caption_column: 标题列名
        task_type: 任务类型
    """
    logger.info(f"转换CSV标题数据: {csv_path} -> {output_path}")
    
    # 转换为绝对路径
    csv_path = Path(csv_path).absolute()
    output_path = Path(output_path).absolute()
    
    task_prompts = {
        'CAPTION': '<CAPTION>',
        'DETAILED_CAPTION': '<DETAILED_CAPTION>',
        'MORE_DETAILED_CAPTION': '<MORE_DETAILED_CAPTION>'
    }
    
    if task_type not in task_prompts:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    prompt = task_prompts[task_type]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in tqdm(reader, desc="CSV转Florence-2进度"):
                # 使用绝对路径
                image_path = Path(row[image_column]).absolute()
                sample = {
                    'image': str(image_path),
                    'prefix': prompt,
                    'suffix': row[caption_column]
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("CSV标题数据转换完成")
