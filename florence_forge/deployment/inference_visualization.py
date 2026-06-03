"""推理结果可视化（从 ``inference.py`` 抽出）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from florence_forge.utils.plot_backend import finalize_matplotlib_figure

from .inference_parsing import clean_text_prefix

logger = logging.getLogger(__name__)

def visualize_detections(self, image, detections: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Visualize detection results on the image.
        
        Args:
            image: The original image (PIL Image).
            detections: A list of detection results.
            save_path: The path to save the visualization (optional).
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
        except ImportError as e:
            logger.error(f"Visualization dependencies are not installed: {e}")
            return
        
        # Ensure it's a PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                logger.error("Unsupported image format for visualization.")
                return
        
        # Create a matplotlib figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Define a list of colors for bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        # Draw detection boxes and labels
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection.get('confidence', 1.0)
            
            # Select a color
            color = colors[i % len(colors)]
            
            # Create a rectangle patch
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add the label
            label_text = f"{label} ({confidence:.2f})" if confidence < 1.0 else label
            ax.text(
                x1, y1 - 5, label_text,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
        plt.title(f'Detection Results - Found {len(detections)} objects')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Visualization result saved to: {save_path}")
        finalize_matplotlib_figure()



def visualize_bboxes(self, image, bboxes, save_path):
        from PIL import ImageDraw

        if not save_path:
            logger.warning("未提供保存路径，无法可视化边界框")
            return

        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(bbox, outline="red", width=2)

        try:
            image.save(save_path)
            logger.info(f"边界框可视化结果已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存边界框可视化结果失败: {e}")



def visualize_ocr_with_region(self, image, ocr_results: List[Dict[str, Any]], save_path: Optional[str]):
        from PIL import ImageDraw, ImageFont

        if not save_path:
            logger.warning("未提供保存路径，无法可视化OCR结果")
            return

        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Font selection for Chinese characters
        font_size = 15
        try:
            # Construct path to the font file relative to this file
            current_dir = Path(__file__).parent
            font_path = current_dir.parent.parent / 'assets' / 'fonts' / 'SourceHanSansSC-Regular.ttf'
            font = ImageFont.truetype(str(font_path), font_size)
        except IOError:
            logger.warning(f"未能加载思源黑体字体，将回退到默认字体。请检查路径：{font_path}")
            try:
                # Common fonts for Chinese on different OS
                font = ImageFont.truetype("SimHei.ttf", font_size) # Windows/Linux
            except IOError:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/STHeitiLight.ttc", font_size) # macOS
                except IOError:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()

        for result in ocr_results:
            polygon = result['polygon']
            text = result['text']
            
            # Draw polygon with semi-transparent fill
            draw.polygon(polygon, outline='lime', fill=(0, 255, 0, 60))
            
            # Position text at the top-left corner of the polygon
            text_position = polygon[0]
            draw.text(text_position, text, fill='red', font=font)

        try:
            image.convert('RGB').save(save_path)
            logger.info(f"OCR可视化结果已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存OCR可视化结果失败: {e}")



def visualize_caption(self, image, caption: str, save_path: Optional[str] = None):
        """Visualize caption on the image.

        Args:
            image: The original image (PIL Image).
            caption: The caption text.
            save_path: The path to save the visualization (optional).
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import textwrap
            import re
        except ImportError as e:
            logger.error(f"Visualization dependencies are not installed: {e}")
            return

        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                logger.error("Unsupported image format for visualization.")
                return

        # Clean caption
        caption = caption.replace('</s>', '').replace('<s>', '').strip()
        caption = clean_text_prefix(caption)

        # Create a drawing context, use RGBA for transparency
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Dynamically adjust font size based on image height
        font_size = max(36, image.height // 25)  # Adjust font size relative to image height, with a minimum
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # Fallback to a default font, try to get a font with a size
            try:
                font = ImageFont.load_default(size=font_size)
            except AttributeError:
                # Older PIL versions might not support size in load_default
                font = ImageFont.load_default()

        image_width, image_height = image.size

        # Wrap text if it's too long
        avg_char_width = font.getlength('a')
        wrap_width = int(image_width / avg_char_width * 1.5) if avg_char_width > 0 else 60
        wrapped_caption = textwrap.fill(caption, width=wrap_width)
        
        # Calculate text bounding box with wrapped text
        text_bbox = draw.textbbox((0, 0), wrapped_caption, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        # Create a rectangle for the text background at the top
        background_color = (255, 255, 0, 180)  # Semi-transparent yellow
        draw.rectangle(
            [(0, 0), (image_width, text_height + 20)],
            fill=background_color
        )

        # Draw the text at the top with a more conspicuous color
        draw.text(
            (10, 10),  # Padding from top-left
            wrapped_caption,
            font=font,
            fill=(0, 0, 0)  # Black text for high contrast
        )
        
        # Convert back to RGB if saving as a format that doesn't support alpha
        if save_path and Path(save_path).suffix.lower() in ['.jpg', '.jpeg']:
            image = image.convert('RGB')

        if save_path:
            image.save(save_path)
            logger.info(f"Caption visualization saved to: {save_path}")
        else:
            image.show()
    


def visualize_segmentation(
        self, 
        image: 'PIL.Image.Image', 
        polygons: List[List[Tuple[int, int]]], 
        save_path: Optional[str] = None,
        color: Tuple[int, int, int] = (255, 0, 0),  # 默认为红色
        alpha: float = 0.5  # 半透明
    ) -> None:
        """在图像上可视化分割掩码"""
        from PIL import Image, ImageDraw

        logger.debug(f"接收到 {len(polygons)} 个多边形进行可视化。")
        if not polygons:
            logger.warning("未提供多边形数据，跳过可视化。")
            # Even if no polygons, save the original image if a path is given,
            # so the user knows the process ran.
            if save_path:
                image.convert('RGB').save(save_path)
                logger.info(f"已保存原始图像到: {save_path}，因为没有分割数据可绘制。")
            return

        # 确保原始图像是RGBA模式，以便进行alpha合成
        base_image = image.convert('RGBA')
        # 创建一个透明的覆盖层用于绘制
        overlay = Image.new('RGBA', base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        polygons_drawn = 0
        for i, polygon in enumerate(polygons):
            logger.debug(f"正在处理多边形 #{i}，该多边形有 {len(polygon)} 个顶点。")
            if len(polygon) > 2:
                # 在透明覆盖层上绘制半透明多边形
                draw.polygon(polygon, fill=color + (int(255 * alpha),), outline=color)
                polygons_drawn += 1
            else:
                logger.warning(f"多边形 #{i} 的顶点数不足（{len(polygon)} <= 2），无法绘制。")

        if polygons_drawn == 0:
            logger.warning("没有绘制任何多边形，因为所有提供的多边形顶点数都不足。")

        # 将带有掩码的覆盖层与基础图像混合
        blended_image = Image.alpha_composite(base_image, overlay)
        final_image = blended_image.convert('RGB')

        if save_path:
            final_image.save(save_path)
            if polygons_drawn > 0:
                logger.info(f"分割可视化结果（绘制了 {polygons_drawn} 个多边形）已保存到: {save_path}")
            else:
                logger.warning(f"可视化图像已保存到 {save_path}，但未绘制任何分割掩码。")
        else:
            try:
                final_image.show()
            except Exception as e:
                logger.warning(f"无法显示图像，请检查显示环境: {e}")

