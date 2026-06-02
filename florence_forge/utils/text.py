"""FlorenceForge文本处理工具模块

提供文本清理、解析和格式化功能
"""

import re
import json
import logging
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def clean_text(
    text: str,
    remove_extra_whitespace: bool = True,
    remove_special_chars: bool = False,
    normalize_unicode: bool = True,
    lowercase: bool = False
) -> str:
    """清理文本
    
    Args:
        text: 输入文本
        remove_extra_whitespace: 是否移除多余空白字符
        remove_special_chars: 是否移除特殊字符
        normalize_unicode: 是否标准化Unicode
        lowercase: 是否转换为小写
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # Unicode标准化
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    # 移除多余空白字符
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # 移除特殊字符（保留字母、数字、基本标点）
    if remove_special_chars:
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
    
    # 转换为小写
    if lowercase:
        text = text.lower()
    
    return text

def tokenize_text(
    text: str,
    method: str = 'simple',
    preserve_case: bool = True
) -> List[str]:
    """文本分词
    
    Args:
        text: 输入文本
        method: 分词方法（simple, whitespace, regex）
        preserve_case: 是否保持大小写
        
    Returns:
        词汇列表
    """
    if not text:
        return []
    
    if not preserve_case:
        text = text.lower()
    
    if method == 'simple':
        # 简单的基于标点和空白的分词
        tokens = re.findall(r'\b\w+\b', text)
    elif method == 'whitespace':
        # 基于空白字符分词
        tokens = text.split()
    elif method == 'regex':
        # 更复杂的正则表达式分词
        tokens = re.findall(r'\w+|[.,!?;]', text)
    else:
        raise ValueError(f"不支持的分词方法: {method}")
    
    return tokens

def extract_coordinates(
    text: str,
    format_type: str = 'florence'
) -> List[Tuple[float, float, float, float]]:
    """从文本中提取坐标信息
    
    Args:
        text: 包含坐标的文本
        format_type: 坐标格式类型
        
    Returns:
        坐标列表，每个坐标为(x1, y1, x2, y2)
    """
    coordinates = []
    
    if format_type == 'florence':
        # Florence-2格式: <loc_x1><loc_y1><loc_x2><loc_y2>
        pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, text)
        
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            # Florence-2使用1000为基准的归一化坐标
            coordinates.append((x1/1000, y1/1000, x2/1000, y2/1000))
    
    elif format_type == 'bbox':
        # 标准边界框格式: [x1, y1, x2, y2]
        pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            x1, y1, x2, y2 = map(float, match)
            coordinates.append((x1, y1, x2, y2))
    
    elif format_type == 'center_size':
        # 中心点+尺寸格式: (cx, cy, w, h)
        pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            cx, cy, w, h = map(float, match)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            coordinates.append((x1, y1, x2, y2))
    
    return coordinates

def extract_labels_and_coordinates(
    text: str,
    format_type: str = 'florence'
) -> List[Dict[str, Any]]:
    """从文本中提取标签和坐标
    
    Args:
        text: 包含标签和坐标的文本
        format_type: 格式类型
        
    Returns:
        包含标签和坐标的字典列表
    """
    results = []
    
    if format_type == 'florence':
        # Florence-2目标检测格式
        # 例如: "person<loc_100><loc_200><loc_300><loc_400>car<loc_500><loc_600><loc_700><loc_800>"
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, text)
        
        for match in matches:
            label, x1, y1, x2, y2 = match
            label = label.strip()
            
            if label:  # 确保标签不为空
                results.append({
                    'label': label,
                    'bbox': [int(x1)/1000, int(y1)/1000, int(x2)/1000, int(y2)/1000],
                    'confidence': 1.0  # Florence-2通常不提供置信度
                })
    
    elif format_type == 'json':
        # JSON格式
        try:
            data = json.loads(text)
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and 'objects' in data:
                results = data['objects']
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 原始文本: {text[:100]}...")
            # 尝试修复常见的JSON格式问题
            try:
                # 移除可能的前后缀
                cleaned_text = text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                
                # 再次尝试解析
                data = json.loads(cleaned_text.strip())
                if isinstance(data, list):
                    results = data
                elif isinstance(data, dict) and 'objects' in data:
                    results = data['objects']
                    
            except json.JSONDecodeError:
                logger.error(f"JSON解析完全失败，返回空结果: {text[:50]}...")
                results = []
    
    return results

def format_detection_result(
    detections: List[Dict[str, Any]],
    format_type: str = 'florence',
    image_size: Optional[Tuple[int, int]] = None
) -> str:
    """格式化检测结果
    
    Args:
        detections: 检测结果列表
        format_type: 输出格式类型
        image_size: 图像尺寸（用于坐标转换）
        
    Returns:
        格式化的文本
    """
    if not detections:
        return ""
    
    if format_type == 'florence':
        # Florence-2格式
        result_parts = []
        
        for det in detections:
            label = det.get('label', '')
            bbox = det.get('bbox', [])
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # 如果坐标是归一化的，转换为Florence-2格式
                if all(0 <= coord <= 1 for coord in bbox):
                    x1, y1, x2, y2 = int(x1*1000), int(y1*1000), int(x2*1000), int(y2*1000)
                elif image_size:
                    # 如果提供了图像尺寸，进行归一化
                    w, h = image_size
                    x1, y1, x2, y2 = int(x1/w*1000), int(y1/h*1000), int(x2/w*1000), int(y2/h*1000)
                
                result_parts.append(f"{label}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>")
        
        return ''.join(result_parts)
    
    elif format_type == 'json':
        # JSON格式
        return json.dumps(detections, ensure_ascii=False, indent=2)
    
    elif format_type == 'text':
        # 纯文本格式
        result_parts = []
        
        for i, det in enumerate(detections, 1):
            label = det.get('label', 'unknown')
            bbox = det.get('bbox', [])
            confidence = det.get('confidence', 0.0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                result_parts.append(
                    f"{i}. {label}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}] (conf: {confidence:.3f})"
                )
            else:
                result_parts.append(f"{i}. {label} (conf: {confidence:.3f})")
        
        return '\n'.join(result_parts)
    
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")

def parse_ocr_result(text: str) -> List[Dict[str, Any]]:
    """解析OCR结果
    
    Args:
        text: OCR结果文本
        
    Returns:
        解析后的文本块列表
    """
    # 简单的OCR结果解析
    # 假设格式为: "text1<loc_x1><loc_y1><loc_x2><loc_y2>text2<loc_x3><loc_y3><loc_x4><loc_y4>"
    
    results = []
    pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    matches = re.findall(pattern, text)
    
    for match in matches:
        text_content, x1, y1, x2, y2 = match
        text_content = text_content.strip()
        
        if text_content:
            results.append({
                'text': text_content,
                'bbox': [int(x1)/1000, int(y1)/1000, int(x2)/1000, int(y2)/1000]
            })
    
    return results

def extract_caption_keywords(
    caption: str,
    min_length: int = 3,
    exclude_words: Optional[List[str]] = None
) -> List[str]:
    """从图像描述中提取关键词
    
    Args:
        caption: 图像描述
        min_length: 最小词长
        exclude_words: 排除词列表
        
    Returns:
        关键词列表
    """
    if exclude_words is None:
        exclude_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ]
    
    # 分词并清理
    words = tokenize_text(caption.lower(), method='simple')
    
    # 过滤关键词
    keywords = []
    for word in words:
        if (len(word) >= min_length and 
            word not in exclude_words and 
            word.isalpha()):
            keywords.append(word)
    
    # 去重并保持顺序
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords

def calculate_text_similarity(
    text1: str,
    text2: str,
    method: str = 'jaccard'
) -> float:
    """计算文本相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        method: 相似度计算方法
        
    Returns:
        相似度分数（0-1）
    """
    if not text1 or not text2:
        return 0.0
    
    # 分词
    tokens1 = set(tokenize_text(text1.lower()))
    tokens2 = set(tokenize_text(text2.lower()))
    
    if method == 'jaccard':
        # Jaccard相似度
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    elif method == 'cosine':
        # 简单的余弦相似度（基于词频）
        all_tokens = tokens1 | tokens2
        
        if not all_tokens:
            return 0.0
        
        # 计算词频向量
        counter1 = Counter(tokenize_text(text1.lower()))
        counter2 = Counter(tokenize_text(text2.lower()))
        
        vec1 = [counter1.get(token, 0) for token in all_tokens]
        vec2 = [counter2.get(token, 0) for token in all_tokens]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    elif method == 'overlap':
        # 重叠系数
        intersection = len(tokens1 & tokens2)
        min_size = min(len(tokens1), len(tokens2))
        return intersection / min_size if min_size > 0 else 0.0
    
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")

class TextProcessor:
    """文本处理器
    
    提供统一的文本处理接口
    """
    
    def __init__(
        self,
        clean_config: Optional[Dict[str, Any]] = None,
        tokenize_config: Optional[Dict[str, Any]] = None
    ):
        """初始化文本处理器
        
        Args:
            clean_config: 文本清理配置
            tokenize_config: 分词配置
        """
        self.clean_config = clean_config or {
            'remove_extra_whitespace': True,
            'normalize_unicode': True
        }
        
        self.tokenize_config = tokenize_config or {
            'method': 'simple',
            'preserve_case': True
        }
    
    def process_text(self, text: str) -> str:
        """处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        return clean_text(text, **self.clean_config)
    
    def tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 输入文本
            
        Returns:
            词汇列表
        """
        cleaned_text = self.process_text(text)
        return tokenize_text(cleaned_text, **self.tokenize_config)
    
    def extract_entities(
        self,
        text: str,
        entity_type: str = 'detection'
    ) -> List[Dict[str, Any]]:
        """提取实体
        
        Args:
            text: 输入文本
            entity_type: 实体类型（detection, ocr等）
            
        Returns:
            实体列表
        """
        if entity_type == 'detection':
            return extract_labels_and_coordinates(text)
        elif entity_type == 'ocr':
            return parse_ocr_result(text)
        else:
            raise ValueError(f"不支持的实体类型: {entity_type}")
    
    def format_output(
        self,
        data: List[Dict[str, Any]],
        format_type: str = 'text'
    ) -> str:
        """格式化输出
        
        Args:
            data: 要格式化的数据
            format_type: 格式类型
            
        Returns:
            格式化的文本
        """
        return format_detection_result(data, format_type)

def validate_florence_format(text: str) -> bool:
    """验证Florence-2格式的有效性
    
    Args:
        text: 要验证的文本
        
    Returns:
        是否为有效格式
    """
    # 检查是否包含有效的坐标标记
    pattern = r'<loc_\d+>'
    matches = re.findall(pattern, text)
    
    # 坐标标记应该成对出现（4个一组）
    return len(matches) % 4 == 0

def convert_coordinates(
    coords: List[float],
    from_format: str,
    to_format: str,
    image_size: Optional[Tuple[int, int]] = None
) -> List[float]:
    """转换坐标格式
    
    Args:
        coords: 坐标列表
        from_format: 源格式（normalized, absolute, florence）
        to_format: 目标格式
        image_size: 图像尺寸（用于绝对坐标转换）
        
    Returns:
        转换后的坐标
    """
    if len(coords) != 4:
        raise ValueError("坐标必须包含4个值")
    
    x1, y1, x2, y2 = coords
    
    # 转换为归一化坐标（中间格式）
    if from_format == 'normalized':
        norm_coords = [x1, y1, x2, y2]
    elif from_format == 'absolute':
        if not image_size:
            raise ValueError("绝对坐标转换需要图像尺寸")
        w, h = image_size
        norm_coords = [x1/w, y1/h, x2/w, y2/h]
    elif from_format == 'florence':
        norm_coords = [x1/1000, y1/1000, x2/1000, y2/1000]
    else:
        raise ValueError(f"不支持的源格式: {from_format}")
    
    # 从归一化坐标转换为目标格式
    if to_format == 'normalized':
        return norm_coords
    elif to_format == 'absolute':
        if not image_size:
            raise ValueError("绝对坐标转换需要图像尺寸")
        w, h = image_size
        return [norm_coords[0]*w, norm_coords[1]*h, norm_coords[2]*w, norm_coords[3]*h]
    elif to_format == 'florence':
        return [int(coord*1000) for coord in norm_coords]
    else:
        raise ValueError(f"不支持的目标格式: {to_format}")