"""OCR 文本格式转换。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


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
