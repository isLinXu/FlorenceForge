"""掩码生成辅助。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


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
