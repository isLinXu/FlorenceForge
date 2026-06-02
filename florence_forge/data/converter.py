"""FlorenceForge数据格式转换器模块

提供各种数据格式到Florence-2格式的转换功能
"""

import json
import csv
import logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)

def generate_mask_from_polygon(polygon: List[float], image_size: Tuple[int, int], mask_dir: str, mask_filename: str) -> str:
    """根据多边形坐标生成掩码图像
    
    Args:
        polygon: 多边形坐标列表
        image_size: 图像尺寸 (width, height)
        mask_dir: 掩码保存目录
        mask_filename: 掩码文件名
        
    Returns:
        掩码文件的绝对路径
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    # 绘制多边形，填充为1
    draw.polygon(polygon, outline=1, fill=1)
    
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = mask_dir / mask_filename
    mask.save(mask_path)
    return str(mask_path.absolute())

class DataFormatConverter:
    """数据格式转换器
    
    提供各种数据格式到Florence-2格式的转换功能
    """
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def txt_ocr_to_florence2(image_dir: str, txt_dir: str, output_path: str,
                            task_type: str = 'OCR') -> None:
        """将文本OCR数据转换为Florence-2格式
        
        Args:
            image_dir: 图像文件目录
            txt_dir: 文本文件目录
            output_path: 输出文件路径
            task_type: 任务类型
        """
        logger.info(f"转换OCR数据: {image_dir}, {txt_dir} -> {output_path}")
        
        # 转换为绝对路径
        image_dir = Path(image_dir).absolute()
        txt_dir = Path(txt_dir).absolute()
        output_path = Path(output_path).absolute()
        
        task_prompts = {
            'OCR': '<OCR>',
            'OCR_WITH_REGION': '<OCR_WITH_REGION>'
        }
        
        if task_type not in task_prompts:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        prompt = task_prompts[task_type]
        
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for image_file in tqdm(image_files, desc="OCR转Florence-2进度"):
                # 构建对应的文本文件路径
                txt_file = txt_dir / f"{image_file.stem}.txt"
                
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as txt_f:
                        text_content = txt_f.read().strip()
                    
                    sample = {
                        'image': str(image_file.absolute()),
                        'txt_file': str(txt_file.absolute()),
                        'prefix': prompt,
                        'suffix': text_content
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info("OCR数据转换完成")
    
    @staticmethod
    def txt_file_ocr_to_florence2(txt_file_path: str, image_dir: str, output_path: str,
                                 task_type: str = 'OCR') -> None:
        """将TXT文件格式的OCR数据转换为Florence-2格式
        
        TXT文件格式：每行包含图像文件名和OCR内容，用制表符分隔
        格式：image_filename\tOCR_content
        
        Args:
            txt_file_path: TXT文件路径
            image_dir: 图像文件目录
            output_path: 输出文件路径
            task_type: 任务类型
        """
        logger.info(f"转换TXT文件OCR数据: {txt_file_path} -> {output_path}")
        
        # 转换为绝对路径
        txt_file_path = Path(txt_file_path).absolute()
        image_dir = Path(image_dir).absolute()
        output_path = Path(output_path).absolute()
        
        task_prompts = {
            'OCR': '<OCR>',
            'OCR_WITH_REGION': '<OCR_WITH_REGION>'
        }
        
        if task_type not in task_prompts:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        prompt = task_prompts[task_type]
        
        # 检查TXT文件是否存在
        if not txt_file_path.exists():
            raise FileNotFoundError(f"TXT文件不存在: {txt_file_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            with open(txt_file_path, 'r', encoding='utf-8') as f_in:
                for line_num, line in enumerate(tqdm(f_in, desc="TXT OCR转Florence-2进度"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析行：图像文件名\tOCR内容
                    parts = line.split('\t')
                    if len(parts) < 2:
                        logger.warning(f"第{line_num}行格式错误，跳过: {line}")
                        skipped_count += 1
                        continue
                    
                    image_filename = parts[0].strip()
                    ocr_content = '\t'.join(parts[1:]).strip()  # 处理OCR内容中可能包含制表符的情况
                    
                    # 构建图像文件路径
                    image_file = image_dir / image_filename
                    
                    # 检查图像文件是否存在
                    if not image_file.exists():
                        logger.warning(f"图像文件不存在，跳过: {image_file}")
                        skipped_count += 1
                        continue
                    
                    # 构建Florence-2格式样本
                    sample = {
                        'image': str(image_file.absolute()),
                        'txt_file': str(txt_file_path.absolute()),
                        'prefix': prompt,
                        'suffix': ocr_content
                    }
                    
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    processed_count += 1
        
        logger.info(f"TXT OCR数据转换完成: 处理{processed_count}条，跳过{skipped_count}条")
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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

class DataValidator:
    """数据验证器
    
    提供数据格式验证和报告生成功能
    """
    
    @staticmethod
    def validate_florence2_jsonl(jsonl_path: str, 
                                image_base_path: str = "") -> Dict[str, Any]:
        """验证Florence-2 JSONL格式数据
        
        Args:
            jsonl_path: JSONL文件路径
            image_base_path: 图像基础路径
            
        Returns:
            验证报告字典
        """
        logger.info(f"验证JSONL数据: {jsonl_path}")
        
        # 转换为绝对路径
        jsonl_path = Path(jsonl_path).absolute()
        image_base_path = Path(image_base_path).absolute() if image_base_path else Path()
        
        report = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'missing_images': 0,
            'task_distribution': {},
            'errors': []
        }
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="验证JSONL进度"), 1):
                report['total_samples'] += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    # 检查必需字段
                    required_fields = ['image', 'prefix', 'suffix']
                    for field in required_fields:
                        if field not in data:
                            report['errors'].append(f"第{line_num}行: 缺少字段 '{field}'")
                            continue
                    
                    # 检查图像文件是否存在
                    image_path = Path(data['image'])
                    if not image_path.exists():
                        report['missing_images'] += 1
                        report['errors'].append(f"第{line_num}行: 图像文件不存在 '{image_path}'")
                    
                    # 检查标签文件是否存在（如果存在）
                    for field in ['label_file', 'txt_file', 'xml_file', 'mask_dir']:
                        if field in data and not Path(data[field]).exists():
                            report['errors'].append(f"第{line_num}行: 标签文件不存在 '{data[field]}'")
                    
                    # 统计任务类型
                    task_type = data['prefix'].strip('<>')
                    report['task_distribution'][task_type] = report['task_distribution'].get(
                        task_type,
                        0
                    ) + 1
                    
                    report['valid_samples'] += 1
                    
                except json.JSONDecodeError as e:
                    report['invalid_samples'] += 1
                    report['errors'].append(f"第{line_num}行: JSON解析错误 - {e}")
                except Exception as e:
                    report['invalid_samples'] += 1
                    report['errors'].append(f"第{line_num}行: 验证错误 - {e}")
        
        logger.info(f"验证完成: 总计{report['total_samples']}样本，有效{report['valid_samples']}，无效{report['invalid_samples']}")
        return report
    
    @staticmethod
    def generate_validation_report(report: Dict[str, Any], output_path: str) -> None:
        """生成验证报告
        
        Args:
            report: 验证报告数据
            output_path: 输出文件路径
        """
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 数据验证报告\n\n")
            f.write(f"- **总样本数**: {report['total_samples']}\n")
            f.write(f"- **有效样本数**: {report['valid_samples']}\n")
            f.write(f"- **无效样本数**: {report['invalid_samples']}\n")
            f.write(f"- **缺失图像数**: {report['missing_images']}\n\n")
            
            f.write("## 任务分布\n\n")
            for task, count in report['task_distribution'].items():
                f.write(f"- **{task}**: {count} 样本\n")
            
            if report['errors']:
                f.write("\n## 错误详情\n\n")
                for error in report['errors'][:100]:  # 只显示前100个错误
                    f.write(f"- {error}\n")
                
                if len(report['errors']) > 100:
                    f.write(f"\n... 还有 {len(report['errors']) - 100} 个错误未显示\n")