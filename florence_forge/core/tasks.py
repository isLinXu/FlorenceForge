"""Florence-2任务定义模块

定义所有支持的Florence-2任务类型及其配置
"""

from enum import Enum
from typing import Dict, Any, List, Optional

class TaskCategory(Enum):
    """任务类别枚举"""
    IMAGE_CAPTIONING = "图像描述"
    OBJECT_DETECTION = "目标检测"
    REGION_ANALYSIS = "区域分析"
    TEXT_RECOGNITION = "文字识别"
    IMAGE_SEGMENTATION = "图像分割"

class TaskOutputType(Enum):
    """任务输出类型枚举"""
    TEXT = "text"
    STRUCTURED = "structured"

# Florence-2所有支持的任务类型及其配置
FLORENCE2_TASKS: Dict[str, Dict[str, Any]] = {
    # 图像描述类任务
    'CAPTION': {
        'prompt': '<CAPTION>',
        'category': TaskCategory.IMAGE_CAPTIONING,
        'description': '基础图像标题生成',
        'has_text_input': False,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 256,
        'num_beams': 3
    },
    'DETAILED_CAPTION': {
        'prompt': '<DETAILED_CAPTION>',
        'category': TaskCategory.IMAGE_CAPTIONING,
        'description': '详细图像标题生成',
        'has_text_input': False,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'MORE_DETAILED_CAPTION': {
        'prompt': '<MORE_DETAILED_CAPTION>',
        'category': TaskCategory.IMAGE_CAPTIONING,
        'description': '更详细图像标题生成',
        'has_text_input': False,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 1024,
        'num_beams': 3
    },
    'CAPTION_TO_PHRASE_GROUNDING': {
        'prompt': '<CAPTION_TO_PHRASE_GROUNDING>',
        'category': TaskCategory.IMAGE_CAPTIONING,
        'description': '标题到短语定位',
        'has_text_input': True,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'DENSE_REGION_CAPTION': {
        'prompt': '<DENSE_REGION_CAPTION>',
        'category': TaskCategory.IMAGE_CAPTIONING,
        'description': '密集区域标题生成',
        'has_text_input': False,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 1024,
        'num_beams': 3
    },
    
    # 目标检测类任务
    'OD': {
        'prompt': '<OD>',
        'category': TaskCategory.OBJECT_DETECTION,
        'description': '通用目标检测',
        'has_text_input': False,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'OPEN_VOCABULARY_DETECTION': {
        'prompt': '<OPEN_VOCABULARY_DETECTION>',
        'category': TaskCategory.OBJECT_DETECTION,
        'description': '开放词汇目标检测',
        'has_text_input': True,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    
    # 区域分析类任务
    'REGION_PROPOSAL': {
        'prompt': '<REGION_PROPOSAL>',
        'category': TaskCategory.REGION_ANALYSIS,
        'description': '区域提议生成',
        'has_text_input': False,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'REGION_TO_CATEGORY': {
        'prompt': '<REGION_TO_CATEGORY>',
        'category': TaskCategory.REGION_ANALYSIS,
        'description': '区域到类别分类',
        'has_text_input': True,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 128,
        'num_beams': 3
    },
    'REGION_TO_DESCRIPTION': {
        'prompt': '<REGION_TO_DESCRIPTION>',
        'category': TaskCategory.REGION_ANALYSIS,
        'description': '区域到描述生成',
        'has_text_input': True,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 256,
        'num_beams': 3
    },
    
    # 文字识别类任务
    'OCR': {
        'prompt': '<OCR>',
        'category': TaskCategory.TEXT_RECOGNITION,
        'description': '光学字符识别',
        'has_text_input': False,
        'output_type': TaskOutputType.TEXT,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'OCR_WITH_REGION': {
        'prompt': '<OCR_WITH_REGION>',
        'category': TaskCategory.TEXT_RECOGNITION,
        'description': '带区域的光学字符识别',
        'has_text_input': False,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    
    # 图像分割类任务
    'REGION_TO_SEGMENTATION': {
        'prompt': '<REGION_TO_SEGMENTATION>',
        'category': TaskCategory.IMAGE_SEGMENTATION,
        'description': '区域到分割',
        'has_text_input': True,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    },
    'REFERRING_EXPRESSION_SEGMENTATION': {
        'prompt': '<REFERRING_EXPRESSION_SEGMENTATION>',
        'category': TaskCategory.IMAGE_SEGMENTATION,
        'description': '参考表达式分割',
        'has_text_input': True,
        'output_type': TaskOutputType.STRUCTURED,
        'max_new_tokens': 512,
        'num_beams': 3
    }
}

def get_tasks_by_category(category: TaskCategory) -> Dict[str, Dict[str, Any]]:
    """根据类别获取任务
    
    Args:
        category: 任务类别
        
    Returns:
        该类别下的所有任务
    """
    return {
        task_name: task_config 
        for task_name, task_config in FLORENCE2_TASKS.items() 
        if task_config['category'] == category
    }

def get_task_config(task_name: str) -> Dict[str, Any]:
    """获取指定任务的配置
    
    Args:
        task_name: 任务名称
        
    Returns:
        任务配置
        
    Raises:
        KeyError: 如果任务不存在
    """
    if task_name not in FLORENCE2_TASKS:
        raise KeyError(f"未知任务类型: {task_name}")
    return FLORENCE2_TASKS[task_name]

def validate_task_name(task_name: str) -> bool:
    """验证任务名称是否有效
    
    Args:
        task_name: 任务名称
        
    Returns:
        是否有效
    """
    return task_name in FLORENCE2_TASKS

def list_all_tasks() -> list:
    """列出所有支持的任务
    
    Returns:
        所有任务名称列表
    """
    return list(FLORENCE2_TASKS.keys())