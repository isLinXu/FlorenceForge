"""Florence-2任务定义模块

定义所有支持的Florence-2任务类型及其配置
"""

from enum import Enum
from typing import Dict, Any

from pydantic import BaseModel, Field


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


class TaskConfig(BaseModel):
    """单个任务的配置模型（Pydantic v2）

    替代原有的裸字典，提供运行时校验和 IDE 补全。
    """

    prompt: str = Field(description="任务 prompt 标记")
    category: TaskCategory = Field(description="任务类别")
    description: str = Field(description="任务描述")
    has_text_input: bool = Field(default=False, description="是否接受文本输入")
    output_type: TaskOutputType = Field(default=TaskOutputType.TEXT, description="输出类型")
    max_new_tokens: int = Field(default=512, ge=1, description="最大生成 token 数")
    num_beams: int = Field(default=3, ge=1, description="beam search 宽度")
    is_visual_primitive: bool = Field(default=False, description="是否为视觉原语任务")
    is_tvp: bool = Field(default=False, description="是否为 TVP 思维链任务")
    is_agentic: bool = Field(default=False, description="是否为 Agentic 元认知推理任务")

    model_config = {"frozen": True}

# Florence-2所有支持的任务类型及其配置
FLORENCE2_TASKS: Dict[str, TaskConfig] = {
    # 图像描述类任务
    'CAPTION': TaskConfig(
        prompt='<CAPTION>',
        category=TaskCategory.IMAGE_CAPTIONING,
        description='基础图像标题生成',
        has_text_input=False,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=256,
        num_beams=3,
    ),
    'DETAILED_CAPTION': TaskConfig(
        prompt='<DETAILED_CAPTION>',
        category=TaskCategory.IMAGE_CAPTIONING,
        description='详细图像标题生成',
        has_text_input=False,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=512,
        num_beams=3,
    ),
    'MORE_DETAILED_CAPTION': TaskConfig(
        prompt='<MORE_DETAILED_CAPTION>',
        category=TaskCategory.IMAGE_CAPTIONING,
        description='更详细图像标题生成',
        has_text_input=False,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=1024,
        num_beams=3,
    ),
    'CAPTION_TO_PHRASE_GROUNDING': TaskConfig(
        prompt='<CAPTION_TO_PHRASE_GROUNDING>',
        category=TaskCategory.IMAGE_CAPTIONING,
        description='标题到短语定位',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    'DENSE_REGION_CAPTION': TaskConfig(
        prompt='<DENSE_REGION_CAPTION>',
        category=TaskCategory.IMAGE_CAPTIONING,
        description='密集区域标题生成',
        has_text_input=False,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=1024,
        num_beams=3,
    ),
    
    # 目标检测类任务
    'OD': TaskConfig(
        prompt='<OD>',
        category=TaskCategory.OBJECT_DETECTION,
        description='通用目标检测',
        has_text_input=False,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    'OPEN_VOCABULARY_DETECTION': TaskConfig(
        prompt='<OPEN_VOCABULARY_DETECTION>',
        category=TaskCategory.OBJECT_DETECTION,
        description='开放词汇目标检测',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    
    # 区域分析类任务
    'REGION_PROPOSAL': TaskConfig(
        prompt='<REGION_PROPOSAL>',
        category=TaskCategory.REGION_ANALYSIS,
        description='区域提议生成',
        has_text_input=False,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    'REGION_TO_CATEGORY': TaskConfig(
        prompt='<REGION_TO_CATEGORY>',
        category=TaskCategory.REGION_ANALYSIS,
        description='区域到类别分类',
        has_text_input=True,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=128,
        num_beams=3,
    ),
    'REGION_TO_DESCRIPTION': TaskConfig(
        prompt='<REGION_TO_DESCRIPTION>',
        category=TaskCategory.REGION_ANALYSIS,
        description='区域到描述生成',
        has_text_input=True,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=256,
        num_beams=3,
    ),
    
    # 文字识别类任务
    'OCR': TaskConfig(
        prompt='<OCR>',
        category=TaskCategory.TEXT_RECOGNITION,
        description='光学字符识别',
        has_text_input=False,
        output_type=TaskOutputType.TEXT,
        max_new_tokens=512,
        num_beams=3,
    ),
    'OCR_WITH_REGION': TaskConfig(
        prompt='<OCR_WITH_REGION>',
        category=TaskCategory.TEXT_RECOGNITION,
        description='带区域的光学字符识别',
        has_text_input=False,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    
    # 图像分割类任务
    'REGION_TO_SEGMENTATION': TaskConfig(
        prompt='<REGION_TO_SEGMENTATION>',
        category=TaskCategory.IMAGE_SEGMENTATION,
        description='区域到分割',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),
    'REFERRING_EXPRESSION_SEGMENTATION': TaskConfig(
        prompt='<REFERRING_EXPRESSION_SEGMENTATION>',
        category=TaskCategory.IMAGE_SEGMENTATION,
        description='参考表达式分割',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
    ),

    # 视觉原语（Visual Primitive）任务
    'OD_VP': TaskConfig(
        prompt='<OD>',
        category=TaskCategory.OBJECT_DETECTION,
        description='视觉原语目标检测',
        has_text_input=False,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
        is_visual_primitive=True,
    ),
    'COUNT_VP': TaskConfig(
        prompt='<COUNT>',
        category=TaskCategory.OBJECT_DETECTION,
        description='视觉原语计数',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=256,
        num_beams=3,
        is_visual_primitive=True,
    ),
    'PHRASE_GROUNDING_VP': TaskConfig(
        prompt='<CAPTION_TO_PHRASE_GROUNDING>',
        category=TaskCategory.OBJECT_DETECTION,
        description='视觉原语短语定位',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=512,
        num_beams=3,
        is_visual_primitive=True,
    ),

    # TVP (Thinking with Visual Primitives) 思维链任务
    'COUNT_VP_COT': TaskConfig(
        prompt='<COUNT>',
        category=TaskCategory.OBJECT_DETECTION,
        description='TVP 计数思维链（CoT + VP grounding）',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=1024,
        num_beams=3,
        is_visual_primitive=True,
        is_tvp=True,
    ),
    'SPATIAL_VP': TaskConfig(
        prompt='<OPEN_VOCABULARY_DETECTION>',
        category=TaskCategory.REGION_ANALYSIS,
        description='TVP 空间推理思维链',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=1024,
        num_beams=3,
        is_visual_primitive=True,
        is_tvp=True,
    ),
    'MAZE_VP': TaskConfig(
        prompt='<REGION_PROPOSAL>',
        category=TaskCategory.REGION_ANALYSIS,
        description='TVP 迷宫导航（point 原语 + DFS 探索链）',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=2048,
        num_beams=1,
        is_visual_primitive=True,
        is_tvp=True,
    ),
    'PATH_VP': TaskConfig(
        prompt='<REGION_PROPOSAL>',
        category=TaskCategory.REGION_ANALYSIS,
        description='TVP 路径追踪（point 轨迹原语）',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=1536,
        num_beams=1,
        is_visual_primitive=True,
        is_tvp=True,
    ),

    # Agentic meta-cognitive reasoning tasks
    'AGENTIC_COUNT': TaskConfig(
        prompt='<COUNT>',
        category=TaskCategory.OBJECT_DETECTION,
        description='Agentic counting with meta-cognitive chain',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=2048,
        num_beams=1,
        is_visual_primitive=True,
        is_agentic=True,
    ),
    'AGENTIC_SPATIAL': TaskConfig(
        prompt='<OPEN_VOCABULARY_DETECTION>',
        category=TaskCategory.REGION_ANALYSIS,
        description='Agentic spatial reasoning chain',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=2048,
        num_beams=1,
        is_visual_primitive=True,
        is_agentic=True,
    ),
    'AGENTIC_MAZE': TaskConfig(
        prompt='<REGION_PROPOSAL>',
        category=TaskCategory.REGION_ANALYSIS,
        description='Agentic maze navigation with multi-step exploration',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=4096,
        num_beams=1,
        is_visual_primitive=True,
        is_agentic=True,
    ),
    'AGENTIC_GROUNDING': TaskConfig(
        prompt='<CAPTION_TO_PHRASE_GROUNDING>',
        category=TaskCategory.OBJECT_DETECTION,
        description='Agentic phrase grounding with verification',
        has_text_input=True,
        output_type=TaskOutputType.STRUCTURED,
        max_new_tokens=2048,
        num_beams=1,
        is_visual_primitive=True,
        is_agentic=True,
    ),
}

TVP_TASK_NAMES: tuple = tuple(
    name for name, cfg in FLORENCE2_TASKS.items() if cfg.is_tvp
)

def get_tasks_by_category(category: TaskCategory) -> Dict[str, TaskConfig]:
    """根据类别获取任务
    
    Args:
        category: 任务类别
        
    Returns:
        该类别下的所有任务
    """
    return {
        task_name: task_config 
        for task_name, task_config in FLORENCE2_TASKS.items() 
        if task_config.category == category
    }

def get_task_config(task_name: str) -> Dict[str, Any]:
    """获取指定任务的配置

    Args:
        task_name: 任务名称
        
    Returns:
        任务配置字典（向后兼容）
        
    Raises:
        KeyError: 如果任务不存在
    """
    if task_name not in FLORENCE2_TASKS:
        raise KeyError(f"未知任务类型: {task_name}")
    return FLORENCE2_TASKS[task_name].model_dump()

def get_task_config_typed(task_name: str) -> TaskConfig:
    """获取指定任务的类型化配置

    Args:
        task_name: 任务名称

    Returns:
        TaskConfig 实例

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


def is_tvp_task(task_name: str) -> bool:
    """Return whether *task_name* is a TVP chain-of-thought task."""
    if task_name not in FLORENCE2_TASKS:
        return False
    return FLORENCE2_TASKS[task_name].is_tvp


def get_tvp_tasks() -> Dict[str, TaskConfig]:
    """Return all registered TVP task configurations."""
    return {
        name: cfg for name, cfg in FLORENCE2_TASKS.items() if cfg.is_tvp
    }

def list_vp_tasks() -> Dict[str, TaskConfig]:
    """列出所有视觉原语（VP）任务

    Returns:
        VP 任务名称到配置的映射
    """
    return {
        name: config for name, config in FLORENCE2_TASKS.items()
        if config.is_visual_primitive
    }

AGENTIC_TASK_NAMES: tuple = tuple(
    name for name, cfg in FLORENCE2_TASKS.items() if cfg.is_agentic
)


def is_agentic_task(task_name: str) -> bool:
    """Return whether *task_name* is an Agentic meta-cognitive task."""
    if task_name not in FLORENCE2_TASKS:
        return False
    return FLORENCE2_TASKS[task_name].is_agentic


def get_agentic_tasks() -> Dict[str, TaskConfig]:
    """Return all registered Agentic task configurations."""
    return {
        name: cfg for name, cfg in FLORENCE2_TASKS.items() if cfg.is_agentic
    }

