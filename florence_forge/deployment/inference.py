#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎

提供高效的模型推理和批处理功能
"""

import torch
import torch.nn as nn
import logging
import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, List, Dict, Any, Tuple
from queue import Queue, Empty

from ..utils.torch_serialization import safe_torch_load

logger = logging.getLogger(__name__)

def _clean_text_prefix(text: str) -> str:
    """Removes prefixes from text that end with '>' or '＞'."""
    import re
    # Split by the last occurrence of '>' or '＞' and take the part after it.
    # This is more robust against multiple separators.
    return re.split(r'[>＞]', text)[-1].strip()


class InferenceEngine:
    """推理引擎
    
    提供高效的模型推理功能
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str, Path],
        device: str = "auto",
        batch_size: int = 1,
        use_amp: bool = False,
        compile_model: bool = False,
        allow_unsafe_torch_load: bool = False,
        model_revision: Optional[str] = None,
    ):
        """初始化推理引擎
        
        Args:
            model: 模型或模型路径
            device: 设备类型
            batch_size: 批处理大小
            use_amp: 是否使用自动混合精度
            compile_model: 是否编译模型
            allow_unsafe_torch_load: 是否允许对本地 .pt/.pth 文件使用
                weights_only=False 的 pickle 反序列化。仅对可信文件启用。
            model_revision: HuggingFace 模型/处理器 revision（分支、tag 或 commit）。
                生产环境建议传入具体 commit hash 以固定供应链输入。
        """
        self.device = self._setup_device(device)
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.compile_model = compile_model
        self.allow_unsafe_torch_load = allow_unsafe_torch_load
        self.model_revision = model_revision
        
        # 加载模型
        self.model = self._load_model(model)
        self.model.eval()
        
        # 性能统计
        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_inference_time": 0.0,
            "throughput": 0.0
        }
        
        # 预处理和后处理函数
        self.preprocessor: Optional[Callable] = None
        self.postprocessor: Optional[Callable] = None
        
        logger.info(f"推理引擎初始化完成，设备: {self.device}")

    def _load_torch_file(self, model_identifier: str) -> nn.Module:
        """安全加载本地 Torch 文件。

        默认使用 weights_only=True，避免对不可信 .pt/.pth 执行 pickle 反序列化。
        需要加载整模型 pickle 时，调用方必须显式允许，或设置环境变量
        FLORENCE_FORGE_ALLOW_UNSAFE_TORCH_LOAD=1。
        """
        allow_unsafe = self.allow_unsafe_torch_load or (
            os.environ.get("FLORENCE_FORGE_ALLOW_UNSAFE_TORCH_LOAD") == "1"
        )

        try:
            loaded = safe_torch_load(
                model_identifier,
                map_location=self.device,
                context="Inference model",
            )
        except Exception as safe_exc:
            if not allow_unsafe:
                raise ValueError(
                    "安全加载本地 Torch 文件失败。FlorenceForge 默认使用 "
                    "torch.load(weights_only=True)；如果该文件是可信来源的整模型 "
                    "pickle，请传入 allow_unsafe_torch_load=True 或设置 "
                    "FLORENCE_FORGE_ALLOW_UNSAFE_TORCH_LOAD=1。"
                ) from safe_exc
            logger.warning(
                "正在使用 weights_only=False 加载本地 Torch 文件。"
                "这会执行 pickle 反序列化，只应对可信文件启用。"
            )
            try:
                loaded = torch.load(
                    model_identifier,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                loaded = torch.load(model_identifier, map_location=self.device)

        if not isinstance(loaded, nn.Module):
            raise TypeError(
                f"本地 Torch 文件加载结果是 {type(loaded).__name__}，不是 nn.Module。"
                "如果这是 state_dict，请先构建模型结构并传入模型实例。"
            )
        return loaded
    
    def _parse_florence2_output(self, output_text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Parse Florence2 model output and scale coordinates.

        Args:
            output_text: The output text from the Florence2 model.
            image_size: The original image size (width, height).

        Returns:
            A list of parsed detection results.
        """
        import re

        detections = []
        # The pattern is designed to capture a label followed by four location tokens.
        # Example: `cat<loc_29><loc_43><loc_935><loc_945>`
        pattern = r"(?P<label>[^<]+)<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)><loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"

        
        image_width, image_height = image_size

        for match in re.finditer(pattern, output_text):
            # Clean the label by removing special characters and unwanted keywords
            label = match.group('label')
            label = label.replace('</s>', '').replace('<s>', '').strip()
            label = _clean_text_prefix(label)
            # Strip leading/trailing underscores and spaces from the label
            label = label.strip(' _')

            # Skip if the label is empty after cleaning
            if not label:
                continue
            
            # The model's output coordinates are normalized to a 1000x1000 space and need to be scaled
            x1 = int(match.group('x1')) * image_width / 1000
            y1 = int(match.group('y1')) * image_height / 1000
            x2 = int(match.group('x2')) * image_width / 1000
            y2 = int(match.group('y2')) * image_height / 1000

            detections.append({
                'label': label,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': 1.0  # Florence-2 does not provide confidence scores
            })
        
        return detections
    
    def _visualize_detections(self, image, detections: List[Dict[str, Any]], save_path: Optional[str] = None):
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
        else:
            plt.show()
        
        plt.close()

    def _visualize_bboxes(self, image, bboxes, save_path):
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

    def _parse_ocr_with_region(self, model_output: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        import re
        width, height = image_size
        results = []
        
        # Clean the output
        cleaned_output = model_output.replace('<s>', '').replace('</s>', '').strip()
        
        # Regex to find text followed by 8 location tags (a quadrilateral)
        pattern = r"([^<]+)((?:<loc_\d+>){8})"
        
        matches = re.findall(pattern, cleaned_output)
        
        for text, loc_str in matches:
            text = text.replace('</s>', '').replace('<s>', '').strip()
            text = _clean_text_prefix(text)
            if not text:
                continue
            
            loc_matches = re.findall(r'\d+', loc_str)
            if len(loc_matches) == 8:
                coords = [int(c) for c in loc_matches]
                polygon = [
                    (coords[0] * width // 1000, coords[1] * height // 1000),
                    (coords[2] * width // 1000, coords[3] * height // 1000),
                    (coords[4] * width // 1000, coords[5] * height // 1000),
                    (coords[6] * width // 1000, coords[7] * height // 1000),
                ]
                results.append({'text': text, 'polygon': polygon})
        return results

    def _visualize_ocr_with_region(self, image, ocr_results: List[Dict[str, Any]], save_path: Optional[str]):
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

    def _parse_bboxes(self, model_output: str, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        import re
        width, height = image_size
        bboxes = []
        # Updated regex to find all bounding box coordinates
        bbox_matches = re.findall(r'\<loc_(\d+)>\<loc_(\d+)>\<loc_(\d+)>\<loc_(\d+)>', model_output)
        for match in bbox_matches:
            xmin = int(match[0]) * width // 1000
            ymin = int(match[1]) * height // 1000
            xmax = int(match[2]) * width // 1000
            ymax = int(match[3]) * height // 1000
            bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes

    def _visualize_caption(self, image, caption: str, save_path: Optional[str] = None):
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
        caption = _clean_text_prefix(caption)

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
    
    def _setup_device(self, device: str) -> torch.device:
        """设置设备
        
        Args:
            device: 设备类型
            
        Returns:
            PyTorch设备对象
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                mps_backend = vars(torch.backends).get("mps")
                mps_available = (
                    mps_backend is not None
                    and callable(getattr(mps_backend, "is_available", None))
                    and mps_backend.is_available()
                )
                device = "mps" if mps_available else "cpu"
        
        return torch.device(device)

    def _build_model_config_kwargs(
        self,
        model_name: str,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构造部署加载路径使用的 ModelConfig 参数。"""
        config_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "device": str(self.device),
            "use_lora": False,
        }
        effective_revision = getattr(self, "model_revision", None) or revision
        if effective_revision:
            config_kwargs["revision"] = effective_revision
        return config_kwargs
    
    def _load_model(self, model: Union[nn.Module, str, Path]) -> nn.Module:
        """加载模型，支持本地路径、Hugging Face模型ID、LoRA模型和TorchScript。
    
        Args:
            model: 模型实例、模型路径（str或Path）或Hugging Face模型ID（str）。
            
        Returns:
            加载并移动到指定设备的模型。
        """
        if isinstance(model, nn.Module):
            loaded_model = model
        else:
            model_identifier = str(model)
            model_path = Path(model_identifier)

            # 优先处理本地文件，如 .pt 或 .pth
            if model_path.suffix in ['.pt', '.pth'] and model_path.is_file():
                logger.info(f"尝试加载本地Torch模型文件: {model_identifier}")
                try:
                    # 尝试加载TorchScript模型
                    loaded_model = torch.jit.load(model_identifier, map_location=self.device)
                    logger.info("TorchScript模型加载成功")
                except Exception:
                    # 如果失败，尝试加载状态字典
                    logger.info("TorchScript加载失败，尝试安全加载PyTorch模型文件")
                    loaded_model = self._load_torch_file(model_identifier)
                    logger.info("PyTorch模型文件加载成功")
            else:
                # 处理Hugging Face模型（本地目录或Hub ID）和LoRA模型
                try:
                    from ..core.model import Florence2MultiTaskModel, ModelConfig

                    # 检查是否为本地LoRA模型目录
                    if model_path.is_dir() and (model_path / "adapter_config.json").exists():
                        logger.info(f"检测到本地LoRA模型: {model_identifier}")
                        import json
                        with open(model_path / "adapter_config.json", 'r') as f:
                            adapter_config = json.load(f)
                        
                        base_model_name = adapter_config.get('base_model_name_or_path', 'microsoft/Florence-2-base')
                        config = ModelConfig(**self._build_model_config_kwargs(
                            base_model_name,
                            revision=adapter_config.get("revision"),
                        ))
                        
                        loaded_model = Florence2MultiTaskModel.load_pretrained(
                            model_identifier, 
                            config=config, 
                            is_peft_model=True
                        )
                        logger.info("LoRA模型加载成功")
                    else:
                        # 加载Hugging Face基础模型（来自Hub或本地）
                        logger.info(f"尝试加载Hugging Face模型: {model_identifier}")
                        config = ModelConfig(**self._build_model_config_kwargs(model_identifier))
                        loaded_model = Florence2MultiTaskModel(config)
                        # 显式加载模型和处理器（延迟加载模式）
                        loaded_model.load()
                        logger.info("Hugging Face模型加载成功")

                except ImportError as e:
                    logger.error(f"无法导入核心模型组件: {e}")
                    raise ValueError(f"加载Hugging Face模型需要 `florence_forge.core.model` 支持。")
                except Exception as e:
                    logger.error(f"加载Hugging Face模型 '{model_identifier}' 失败: {e}")
                    raise ValueError(f"无法加载模型。请检查路径或模型ID是否正确，以及是否需要网络连接。")
        
        if not hasattr(loaded_model, 'eval'):
            raise TypeError(f"加载结果 {type(loaded_model).__name__} 不支持 eval()，无法用于推理")

        # 移动到指定设备
        if hasattr(loaded_model, 'to') and hasattr(loaded_model, '__class__') and 'Florence2MultiTaskModel' in str(loaded_model.__class__):
            # Florence2MultiTaskModel有自己的to方法
            loaded_model = loaded_model.to(self.device)
        elif hasattr(loaded_model, 'to'):
            loaded_model = loaded_model.to(self.device)
        else:
            logger.warning(f"模型 {type(loaded_model)} 不支持.to()方法，跳过设备移动")
        
        # 编译模型（如果支持）
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                loaded_model = torch.compile(loaded_model)
                logger.info("模型编译完成")
            except Exception as e:
                logger.warning(f"模型编译失败: {e}")
        
        return loaded_model
    
    def set_preprocessor(self, preprocessor: Callable) -> None:
        """设置预处理函数
        
        Args:
            preprocessor: 预处理函数
        """
        self.preprocessor = preprocessor
    
    def set_postprocessor(self, postprocessor: Callable) -> None:
        """设置后处理函数
        
        Args:
            postprocessor: 后处理函数
        """
        self.postprocessor = postprocessor
    
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, List, 'PIL.Image.Image'],
        task_prompt: Optional[str] = None,
        text_input: Optional[str] = None,
        return_raw: bool = False,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Union[torch.Tensor, Any]:
        """单次预测
        
        Args:
            inputs: 输入数据（支持PIL Image、numpy数组、tensor等）
            return_raw: 是否返回原始输出
            task_prompt: 任务提示（Florence2模型使用）
            text_input: 文本输入（某些任务需要）
            visualize: 是否在原图上可视化检测结果
            save_path: 可视化结果保存路径（如果不指定则显示图像）
            
        Returns:
            预测结果
        """
        start_time = time.time()
        
        # 预处理
        if self.preprocessor:
            processed_inputs = self.preprocessor(inputs, task_prompt=task_prompt, text_input=text_input)
        else:
            # 默认预处理逻辑 - 直接使用输入，因为Florence2Dataset不存在
            processed_inputs = inputs
        
        # 检查是否为Florence2模型 - 增强检测逻辑
        is_florence2_model = (
            hasattr(self.model, 'generate') and 
            hasattr(self.model, 'processor') and
            hasattr(self.model, '__class__') and
            ('Florence2MultiTaskModel' in str(self.model.__class__) or 
             'florence' in str(self.model.__class__).lower())
        )
        
        # 调试信息
        if hasattr(self.model, '__class__'):
            logger.debug(f"模型类型: {self.model.__class__}, Florence2检测结果: {is_florence2_model}")
        
        # 处理PIL Image
        if hasattr(inputs, 'mode') and hasattr(inputs, 'size'):  # PIL Image检测
            try:
                from PIL import Image
                if isinstance(inputs, Image.Image):
                    if is_florence2_model:
                        # Florence2模型：直接使用generate方法
                        try:
                            # 使用Florence2MultiTaskModel的generate方法进行推理
                            with torch.no_grad():
                                if self.use_amp:
                                    with torch.autocast(device_type=self.device.type):
                                        generated_text = self.model.generate(
                                            images=inputs,
                                            task_prompt=task_prompt,
                                            text_input=text_input
                                        )
                                else:
                                    generated_text = self.model.generate(
                                        images=inputs,
                                        task_prompt=task_prompt,
                                        text_input=text_input
                                    )
                            
                            # 处理生成的文本结果
                            if generated_text is not None:
                                if isinstance(generated_text, torch.Tensor):
                                    # 如果返回的是tensor，需要进一步处理
                                    outputs = str(generated_text)
                                elif isinstance(generated_text, (list, tuple)):
                                    # 如果返回的是列表或元组，取第一个元素
                                    outputs = str(generated_text[0]) if generated_text else ""
                                else:
                                    # 如果是字符串或其他类型，直接使用
                                    outputs = str(generated_text)
                            else:
                                outputs = ""
                            
                            # 可视化检测结果（如果启用）
                            # 可视化结果（如果启用）
                            if visualize and outputs and isinstance(inputs, Image.Image):
                                if '<OD>' in task_prompt or 'detection' in task_prompt.lower():
                                    try:
                                        detections = self._parse_florence2_output(outputs, image_size=inputs.size)
                                        if detections:
                                            self._visualize_detections(inputs, detections, save_path)
                                            logger.info(f"检测到 {len(detections)} 个目标并已可视化")
                                        else:
                                            logger.warning("未检测到任何目标")
                                    except Exception as viz_e:
                                        logger.error(f"检测可视化失败: {viz_e}")
                                elif 'segmentation' in task_prompt.lower() or 'REGION_TO_SEGMENTATION' in task_prompt:
                                    try:
                                        segmentation_data = self._parse_segmentation_output(outputs, image_size=inputs.size)
                                        if segmentation_data:
                                            self._visualize_segmentation(inputs.copy(), segmentation_data, save_path)
                                            logger.info(f"分割结果已可视化")
                                        else:
                                            logger.warning("未解析到分割数据")
                                    except Exception as viz_e:
                                        logger.error(f"分割可视化失败: {viz_e}")
                                elif '<REGION_PROPOSAL>' in task_prompt:
                                    try:
                                        bboxes = self._parse_bboxes(outputs, image_size=inputs.size)
                                        if bboxes:
                                            self._visualize_bboxes(inputs.copy(), bboxes, save_path)
                                            logger.info(f"区域提议结果已可视化")
                                        else:
                                            logger.warning("未解析到区域提议")
                                    except Exception as viz_e:
                                        logger.error(f"区域提议可视化失败: {viz_e}")
                                elif 'OCR_WITH_REGION' in task_prompt:
                                    try:
                                        ocr_results = self._parse_ocr_with_region(outputs, image_size=inputs.size)
                                        if ocr_results:
                                            self._visualize_ocr_with_region(inputs.copy(), ocr_results, save_path)
                                            logger.info(f"OCR区域结果已可视化")
                                        else:
                                            logger.warning("未解析到OCR区域结果")
                                    except Exception as viz_e:
                                         logger.error(f"OCR区域可视化失败: {viz_e}")
                                elif '<REGION_TO_CATEGORY>' in task_prompt:
                                    try:
                                        detections = self._parse_florence2_output(outputs, image_size=inputs.size)
                                        if detections:
                                            self._visualize_detections(inputs, detections, save_path)
                                            logger.info(f"检测到 {len(detections)} 个目标并已可视化")
                                        else:
                                            logger.warning("未检测到任何目标")
                                    except Exception as viz_e:
                                        logger.error(f"检测可视化失败: {viz_e}")
                                else:
                                    # For captioning or other tasks, visualize the text on the image
                                    try:
                                        self._visualize_caption(inputs.copy(), outputs, save_path)
                                    except Exception as viz_e:
                                        logger.error(f"标题可视化失败: {viz_e}")

                            
                        except Exception as e:
                            logger.error(f"Florence2模型推理失败: {e}")
                            logger.error(f"错误类型: {type(e).__name__}")
                            import traceback
                            logger.debug(f"详细错误信息: {traceback.format_exc()}")
                            
                            # 对于embedding相关错误，直接返回空结果而不是回退
                            if 'embedding' in str(e).lower() or 'indices' in str(e).lower():
                                logger.warning("检测到embedding相关错误，返回空结果")
                                outputs = ""
                            else:
                                # 其他错误才尝试回退到普通处理方式
                                logger.info("尝试回退到普通tensor处理方式")
                                try:
                                    if inputs.mode != 'RGB':
                                        inputs = inputs.convert('RGB')
                                    inputs = np.array(inputs)
                                    inputs = torch.from_numpy(inputs).permute(2, 0, 1).float() / 255.0
                                    inputs = inputs.to(self.device).unsqueeze(0)
                                    
                                    with torch.no_grad():
                                        if self.use_amp:
                                            with torch.autocast(device_type=self.device.type):
                                                outputs = self.model(inputs)
                                        else:
                                            outputs = self.model(inputs)
                                except Exception as fallback_e:
                                    logger.error(f"回退处理也失败: {fallback_e}")
                                    outputs = ""
                    else:
                        # 普通模型：转换为tensor
                        if inputs.mode != 'RGB':
                            inputs = inputs.convert('RGB')
                        inputs = np.array(inputs)
                        inputs = torch.from_numpy(inputs).permute(2, 0, 1).float() / 255.0
                        inputs = inputs.to(self.device).unsqueeze(0)
                        
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.autocast(device_type=self.device.type):
                                    outputs = self.model(inputs)
                            else:
                                outputs = self.model(inputs)
                        
            except ImportError:
                logger.warning("PIL未安装，无法处理PIL Image")
                # 确保输入是tensor并在正确设备上
                if not isinstance(inputs, torch.Tensor):
                    try:
                        inputs = torch.tensor(inputs)
                    except Exception as e:
                        raise ValueError(f"无法将输入转换为tensor: {e}. 输入类型: {type(inputs)}")
                inputs = inputs.to(self.device)
                
                # 添加batch维度（如果需要）
                if inputs.dim() == 3:  # 假设是单张图像
                    inputs = inputs.unsqueeze(0)
                
                # 推理
                with torch.no_grad():
                    if self.use_amp:
                        with torch.autocast(device_type=self.device.type):
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)
        else:
            # 非PIL Image输入的处理
            if not isinstance(inputs, torch.Tensor):
                try:
                    inputs = torch.tensor(inputs)
                except Exception as e:
                    raise ValueError(f"无法将输入转换为tensor: {e}. 输入类型: {type(inputs)}")
            inputs = inputs.to(self.device)
            
            # 添加batch维度（如果需要）
            if inputs.dim() == 3:  # 假设是单张图像
                inputs = inputs.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
        
        # 后处理
        if not return_raw and self.postprocessor is not None:
            outputs = self.postprocessor(outputs)
        
        # 更新统计信息
        inference_time = time.time() - start_time
        self._update_stats(inference_time)
        
        return outputs
    
    def predict_batch(
        self,
        inputs_list: List[Union[torch.Tensor, np.ndarray, 'PIL.Image.Image']],
        batch_size: Optional[int] = None,
        task_prompt: str = "<OD>",
        text_input: Optional[str] = None,
        visualize: bool = False,
        save_dir: Optional[str] = None
    ) -> List[Any]:
        """批量预测
        
        Args:
            inputs_list: 输入数据列表（支持PIL Image、numpy数组、tensor等）
            batch_size: 批处理大小
            task_prompt: 任务提示（Florence2模型使用）
            text_input: 文本输入（某些任务需要）
            visualize: 是否在原图上可视化检测结果
            save_dir: 可视化结果保存目录（如果不指定则显示图像）
            
        Returns:
            预测结果列表
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 检查是否为Florence2模型 - 增强检测逻辑
        is_florence2_model = (
            hasattr(self.model, 'generate') and 
            hasattr(self.model, 'processor') and
            hasattr(self.model, '__class__') and
            ('Florence2MultiTaskModel' in str(self.model.__class__) or 
             'florence' in str(self.model.__class__).lower())
        )
        
        # 调试信息
        if hasattr(self.model, '__class__'):
            logger.debug(f"批量推理 - 模型类型: {self.model.__class__}, Florence2检测结果: {is_florence2_model}")
        
        results = []
        
        if is_florence2_model:
            # Florence2模型：逐个处理（因为processor不支持真正的批处理）
            for i, inp in enumerate(inputs_list):
                try:
                    # 为每个图像生成保存路径
                    save_path = None
                    if visualize and save_dir:
                        import os
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"result_{i:04d}.png")
                    
                    result = self.predict(
                        inp, 
                        task_prompt=task_prompt, 
                        text_input=text_input,
                        visualize=visualize,
                        save_path=save_path
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量推理中单个样本失败: {e}")
                    results.append("")
        else:
            # 普通模型：使用原有的批处理逻辑
            for i in range(0, len(inputs_list), batch_size):
                batch_inputs = inputs_list[i:i + batch_size]
                
                # 预处理批次
                if self.preprocessor is not None:
                    batch_inputs = [self.preprocessor(inp) for inp in batch_inputs]
                
                # 处理PIL Image并转换为tensor
                processed_inputs = []
                for inp in batch_inputs:
                    # 处理PIL Image
                    if hasattr(inp, 'mode') and hasattr(inp, 'size'):  # PIL Image检测
                        try:
                            from PIL import Image
                            if isinstance(inp, Image.Image):
                                # 转换为RGB格式
                                if inp.mode != 'RGB':
                                    inp = inp.convert('RGB')
                                # 转换为numpy数组
                                inp = np.array(inp)
                                # 转换为tensor (H, W, C) -> (C, H, W)
                                inp = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
                        except ImportError:
                            logger.warning("PIL未安装，无法处理PIL Image")
                    
                    # 确保是tensor
                    if not isinstance(inp, torch.Tensor):
                        try:
                            inp = torch.tensor(inp)
                        except Exception as e:
                            raise ValueError(f"无法将输入转换为tensor: {e}. 输入类型: {type(inp)}")
                    
                    processed_inputs.append(inp)
                
                # 堆叠tensor
                batch_tensor = torch.stack(processed_inputs).to(self.device)
                
                # 批量推理
                start_time = time.time()
                
                with torch.no_grad():
                    if self.use_amp:
                        with torch.autocast(device_type=self.device.type):
                            batch_outputs = self.model(batch_tensor)
                    else:
                        batch_outputs = self.model(batch_tensor)
                
                # 后处理
                if self.postprocessor is not None:
                    batch_results = [
                        self.postprocessor(output.unsqueeze(0))
                        for output in batch_outputs
                    ]
                else:
                    batch_results = [output for output in batch_outputs]
                
                results.extend(batch_results)
                
                # 更新统计信息
                inference_time = time.time() - start_time
                self._update_stats(inference_time, batch_size=len(batch_inputs))
        
        return results
    
    def predict_stream(
        self,
        input_generator,
        max_workers: int = 4
    ):
        """流式预测
        
        Args:
            input_generator: 输入数据生成器
            max_workers: 最大工作线程数
            
        Yields:
            预测结果
        """
        input_queue = Queue(maxsize=max_workers * 2)
        result_queue = Queue()
        
        def producer():
            """生产者线程"""
            try:
                for inputs in input_generator:
                    input_queue.put(inputs)
                input_queue.put(None)  # 结束标志
            except Exception as e:
                logger.error(f"生产者线程错误: {e}")
                input_queue.put(None)
        
        def consumer():
            """消费者线程"""
            try:
                while True:
                    try:
                        inputs = input_queue.get(timeout=1.0)
                        if inputs is None:
                            break
                        
                        result = self.predict(inputs)
                        result_queue.put(result)
                        
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"消费者线程错误: {e}")
                        result_queue.put(None)
            finally:
                result_queue.put(None)  # 结束标志
        
        # 启动线程
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # 生成结果
        try:
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result is None:
                        break
                    yield result
                except Empty:
                    continue
        finally:
            producer_thread.join()
            consumer_thread.join()
    
    def benchmark(
        self,
        input_shape: tuple,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """性能基准测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            warmup_runs: 预热次数
            
        Returns:
            性能指标
        """
        # 创建随机输入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 预热
        logger.info(f"预热 {warmup_runs} 次...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        
        # 同步（如果是CUDA）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 基准测试
        logger.info(f"基准测试 {num_runs} 次...")
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # 计算统计信息
        times = np.array(times)
        
        results = {
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "median_time_ms": np.median(times) * 1000,
            "fps": 1.0 / np.mean(times),
            "throughput_samples_per_sec": input_shape[0] / np.mean(times) if len(input_shape) > 0 else 1.0 / np.mean(times)
        }
        
        logger.info(f"基准测试完成: {results['avg_time_ms']:.2f}ms, {results['fps']:.2f}FPS")
        
        return results
    
    def profile_memory(self, input_shape: tuple) -> Dict[str, float]:
        """内存使用分析
        
        Args:
            input_shape: 输入形状
            
        Returns:
            内存使用信息
        """
        if self.device.type != 'cuda':
            logger.warning("内存分析仅支持CUDA设备")
            return {}
        
        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 创建输入并进行推理
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        results = {
            "initial_memory_mb": initial_memory / (1024 ** 2),
            "peak_memory_mb": peak_memory / (1024 ** 2),
            "current_memory_mb": current_memory / (1024 ** 2),
            "memory_increase_mb": (current_memory - initial_memory) / (1024 ** 2)
        }
        
        return results
    
    def _update_stats(self, inference_time: float, batch_size: int = 1) -> None:
        """更新统计信息
        
        Args:
            inference_time: 推理时间
            batch_size: 批处理大小
        """
        self.stats["total_inferences"] += batch_size
        self.stats["total_time"] += inference_time
        self.stats["avg_inference_time"] = self.stats["total_time"] / self.stats["total_inferences"]
        self.stats["throughput"] = self.stats["total_inferences"] / self.stats["total_time"]
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_inference_time": 0.0,
            "throughput": 0.0
        }
    
    def save_engine(self, save_path: Union[str, Path]) -> None:
        """保存推理引擎配置
        
        Args:
            save_path: 保存路径
        """
        import json
        
        config = {
            "device": str(self.device),
            "batch_size": self.batch_size,
            "use_amp": self.use_amp,
            "compile_model": self.compile_model,
            "stats": self.stats
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"推理引擎配置已保存到: {save_path}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def _parse_segmentation_output(self, output_text: str, image_size: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """Parse segmentation output from Florence2 model and scale coordinates.

        Args:
            output_text: The output text from the Florence2 model.
            image_size: The original image size (width, height).

        Returns:
            A list of polygons, where each polygon is a list of (x, y) coordinates.
        """
        import re

        all_polygons = []
        image_width, image_height = image_size

        # Corrected regex to find all sequences of location tokens that form a polygon.
        # Using a non-capturing group (?:...) to ensure the whole sequence is returned by findall.
        polygon_texts = re.findall(r'(?:<loc_\d+><loc_\d+>)+', output_text)

        for poly_text in polygon_texts:
            polygon = []
            # Find all coordinate pairs in the current polygon text
            loc_matches = re.finditer(r'<loc_(?P<x>\d+)><loc_(?P<y>\d+)>', poly_text)
            for match in loc_matches:
                # Scale coordinates from 1000x1000 space to original image size
                x = int(int(match.group('x')) * image_width / 1000)
                y = int(int(match.group('y')) * image_height / 1000)
                polygon.append((x, y))
            
            if polygon:
                all_polygons.append(polygon)
        
        return all_polygons

    def _visualize_segmentation(
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 清理资源
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
