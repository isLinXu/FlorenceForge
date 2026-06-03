"""目标检测格式转换（YOLO / COCO / VOC XML）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover
    ET = None

logger = logging.getLogger(__name__)


def yolo_to_florence2_od(yolo_labels_dir: str, output_path: str, 
                        image_dir: str, classes_file: str, 
                        image_ext: str = ".jpg", task_type: str = 'OD') -> None:
    """将YOLO格式的目标检测数据转换为Florence-2格式
    
    Args:
        yolo_labels_dir: YOLO标签文件目录
        output_path: 输出文件路径
        image_dir: 图像文件目录
        classes_file: 类别文件路径
        image_ext: 图像文件扩展名
        task_type: 任务类型
    """
    logger.info(f"转换YOLO数据: {yolo_labels_dir} -> {output_path}")
    
    # 转换为绝对路径
    yolo_labels_dir = Path(yolo_labels_dir).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute()
    classes_file = Path(classes_file).absolute()
    
    # 读取类别文件
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 获取所有标签文件
    label_files = list(yolo_labels_dir.glob("*.txt"))
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for label_file in tqdm(label_files, desc="YOLO转Florence-2进度"):
            # 构建对应的图像文件路径
            image_file = image_dir / f"{label_file.stem}{image_ext}"
            
            if not image_file.exists():
                logger.warning(f"图像文件不存在: {image_file}")
                continue
            
            # 读取图像尺寸
            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logger.error(f"无法读取图像 {image_file}: {e}")
                continue
            
            # 读取YOLO标签
            bboxes = []
            labels = []
            
            with open(label_file, 'r') as label_f:
                for line in label_f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为绝对坐标
                        x1 = (x_center - width / 2) * img_width
                        y1 = (y_center - height / 2) * img_height
                        x2 = (x_center + width / 2) * img_width
                        y2 = (y_center + height / 2) * img_height
                        
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(classes[class_id])
            
            if bboxes:  # 只处理有标注的图像
                answer = {
                    f'<{task_type}>': {
                        'bboxes': bboxes,
                        'labels': labels
                    }
                }
                
                sample = {
                    'image': str(image_file.absolute()),
                    'label_file': str(label_file.absolute()),
                    'prefix': f'<{task_type}>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("YOLO数据转换完成")

def coco_to_florence2_od(coco_json_path: str, output_path: str, 
                        image_dir: str, task_type: str = 'OD') -> None:
    """将COCO格式的目标检测数据转换为Florence-2格式
    
    Args:
        coco_json_path: COCO JSON文件路径
        output_path: 输出文件路径
        image_dir: 图像文件目录
        task_type: 任务类型
    """
    logger.info(f"转换COCO数据: {coco_json_path} -> {output_path}")
    
    # 转换为绝对路径
    coco_json_path = Path(coco_json_path).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute()
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

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
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_id, annotations in tqdm(image_annotations.items(), desc="COCO转Florence-2进度"):
            image_info = images[image_id]
            
            bboxes = []
            labels = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                # 转换为[x1, y1, x2, y2]格式
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                bboxes.append([x1, y1, x2, y2])
                labels.append(categories[ann['category_id']])
            
            # 构建答案
            answer = {
                f'<{task_type}>': {
                    'bboxes': bboxes,
                    'labels': labels
                }
            }
            
            # 使用绝对路径
            image_path = image_dir / image_info['file_name']
            
            # 构建样本
            sample = {
                'image': str(image_path.absolute()),
                'prefix': f'<{task_type}>',
                'suffix': json.dumps(answer, ensure_ascii=False)
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"转换完成，共处理 {len(image_annotations)} 个图像")

def xml_to_florence2_od(xml_dir: str, output_path: str, 
                       image_dir: str = "", task_type: str = 'OD') -> None:
    """将VOC XML格式的目标检测数据转换为Florence-2格式
    
    Args:
        xml_dir: XML文件目录
        output_path: 输出文件路径
        image_dir: 图像文件目录
        task_type: 任务类型
    """
    logger.info(f"转换VOC XML数据: {xml_dir} -> {output_path}")
    
    # 转换为绝对路径
    xml_dir = Path(xml_dir).absolute()
    output_path = Path(output_path).absolute()
    image_dir = Path(image_dir).absolute() if image_dir else Path()
    
    xml_files = list(xml_dir.glob("*.xml"))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for xml_file in tqdm(xml_files, desc="XML转Florence-2进度"):
            if ET is None:
                raise ImportError(
                    "VOC XML转换需要 defusedxml，以避免不可信 XML 解析风险。"
                    "请安装: pip install defusedxml"
                )
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 获取图像信息
            filename = root.find('filename').text
            
            # 获取所有目标
            bboxes = []
            labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            
            if bboxes:  # 只处理有标注的图像
                answer = {
                    f'<{task_type}>': {
                        'bboxes': bboxes,
                        'labels': labels
                    }
                }
                
                # 使用绝对路径
                image_path = image_dir / filename
                
                sample = {
                    'image': str(image_path.absolute()),
                    'xml_file': str(xml_file.absolute()),
                    'prefix': f'<{task_type}>',
                    'suffix': json.dumps(answer, ensure_ascii=False)
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info("VOC XML数据转换完成")
