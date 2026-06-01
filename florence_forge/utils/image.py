"""FlorenceForge图像处理工具模块

提供图像加载、处理和转换功能
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms


def _validate_positive_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸必须为正数，收到: {(width, height)}")


def _scale_dimension(value: float) -> int:
    return max(1, int(round(value)))


def load_image(image_path: Union[str, Path], mode: str = "RGB") -> Image.Image:
    """加载图像

    Args:
        image_path: 图像路径
        mode: 图像模式（RGB, RGBA, L等）

    Returns:
        PIL图像对象
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    try:
        image = Image.open(image_path)
        if mode and image.mode != mode:
            image = image.convert(mode)
        return image
    except Exception as e:
        raise ValueError(f"无法加载图像 {image_path}: {e}")


def save_image(
    image: Image.Image,
    save_path: Union[str, Path],
    quality: int = 95,
    optimize: bool = True,
) -> None:
    """保存图像

    Args:
        image: PIL图像对象
        save_path: 保存路径
        quality: JPEG质量（1-100）
        optimize: 是否优化
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 根据文件扩展名确定格式
    format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }

    file_format = format_map.get(save_path.suffix.lower(), "JPEG")

    save_kwargs = {"format": file_format, "optimize": optimize}
    if file_format == "JPEG":
        save_kwargs["quality"] = quality

    image.save(save_path, **save_kwargs)


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    method: str = "lanczos",
    maintain_aspect: bool = True,
) -> Image.Image:
    """调整图像大小

    Args:
        image: PIL图像对象
        size: 目标大小（单个值或(width, height)）
        method: 重采样方法
        maintain_aspect: 是否保持宽高比

    Returns:
        调整大小后的图像
    """
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"目标图像尺寸必须为正数，收到: {size}")
        if maintain_aspect:
            # 保持宽高比，以较长边为准
            w, h = image.size
            if w > h:
                new_w, new_h = size, _scale_dimension(h * size / w)
            else:
                new_w, new_h = _scale_dimension(w * size / h), size
        else:
            new_w, new_h = size, size
    else:
        new_w, new_h = size
        _validate_positive_size(new_w, new_h)
        if maintain_aspect:
            # 保持宽高比，适应目标尺寸
            w, h = image.size
            ratio = min(new_w / w, new_h / h)
            new_w, new_h = _scale_dimension(w * ratio), _scale_dimension(h * ratio)

    _validate_positive_size(new_w, new_h)

    # 重采样方法映射
    method_map = {
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }

    resample = method_map.get(method.lower(), Image.LANCZOS)

    return image.resize((new_w, new_h), resample)


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """裁剪图像

    Args:
        image: PIL图像对象
        bbox: 边界框(left, top, right, bottom)

    Returns:
        裁剪后的图像
    """
    return image.crop(bbox)


def pad_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Union[int, Tuple[int, ...]] = 0,
    position: str = "center",
) -> Image.Image:
    """填充图像到目标大小

    Args:
        image: PIL图像对象
        target_size: 目标大小(width, height)
        fill_color: 填充颜色
        position: 图像位置（center, top-left, top-right, bottom-left, bottom-right）

    Returns:
        填充后的图像
    """
    target_w, target_h = target_size
    img_w, img_h = image.size

    if img_w >= target_w and img_h >= target_h:
        return image

    # 创建新图像
    new_image = Image.new(image.mode, target_size, fill_color)

    # 计算粘贴位置
    if position == "center":
        x = (target_w - img_w) // 2
        y = (target_h - img_h) // 2
    elif position == "top-left":
        x, y = 0, 0
    elif position == "top-right":
        x, y = target_w - img_w, 0
    elif position == "bottom-left":
        x, y = 0, target_h - img_h
    elif position == "bottom-right":
        x, y = target_w - img_w, target_h - img_h
    else:
        x, y = 0, 0

    new_image.paste(image, (x, y))
    return new_image


def normalize_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    """标准化图像

    Args:
        image: 图像（PIL、numpy数组或torch张量）
        mean: 均值
        std: 标准差

    Returns:
        标准化后的张量
    """
    # 默认ImageNet标准化参数
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # 转换为张量
    if isinstance(image, Image.Image):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        return transform(image)
    elif isinstance(image, np.ndarray):
        # 假设numpy数组是HWC格式，值范围0-255
        tensor = torch.from_numpy(image).float() / 255.0
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW

        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(tensor)
    elif isinstance(image, torch.Tensor):
        # 假设已经是0-1范围的CHW格式
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(image)
    else:
        raise ValueError(f"不支持的图像类型: {type(image)}")


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    """反标准化图像张量

    Args:
        tensor: 标准化的图像张量
        mean: 均值
        std: 标准差

    Returns:
        反标准化的张量
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将张量转换为PIL图像

    Args:
        tensor: 图像张量（CHW格式，值范围0-1）

    Returns:
        PIL图像
    """
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("只支持单张图像张量，批量张量请先选择一个样本")
        tensor = tensor.squeeze(0)
    if tensor.dim() not in {2, 3}:
        raise ValueError(f"图像张量必须是 2D/3D 或单样本 4D，收到 {tensor.dim()}D")

    # 确保值在0-1范围内
    tensor = torch.clamp(tensor, 0, 1)

    # 转换为numpy数组
    if tensor.dim() == 3:
        array = tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        array = tensor.detach().cpu().numpy()

    # 转换为0-255范围
    array = (array * 255).astype(np.uint8)

    return Image.fromarray(array)


def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    colors: Optional[List[str]] = None,
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """在图像上绘制边界框

    Args:
        image: PIL图像
        boxes: 边界框列表，每个框为[x1, y1, x2, y2]
        labels: 标签列表
        scores: 置信度分数列表
        colors: 颜色列表
        line_width: 线宽
        font_size: 字体大小

    Returns:
        绘制了边界框的图像
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # 默认颜色
    if colors is None:
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown"]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 绘制标签和分数
        if labels or scores:
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if scores and i < len(scores):
                text_parts.append(f"{scores[i]:.2f}")

            if text_parts:
                text = " ".join(text_parts)

                # 计算文本大小
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 绘制文本背景
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color
                )

                # 绘制文本
                draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

    return image


def draw_segmentation_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """在图像上绘制分割掩码

    Args:
        image: PIL图像
        mask: 分割掩码（二值或多类）
        alpha: 透明度
        color: 掩码颜色（如果为None则自动生成）

    Returns:
        绘制了掩码的图像
    """
    image = image.copy()

    if color is None:
        color = (255, 0, 0)  # 默认红色

    # 创建彩色掩码
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask > 0] = color

    # 转换为PIL图像
    mask_image = Image.fromarray(colored_mask)

    # 混合图像
    return Image.blend(image, mask_image, alpha)


class ImageProcessor:
    """图像处理器

    提供一系列图像处理功能的封装
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """初始化图像处理器

        Args:
            target_size: 目标图像大小
            normalize: 是否标准化
            mean: 标准化均值
            std: 标准化标准差
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # 构建变换管道
        self._build_transforms()

    def _build_transforms(self) -> None:
        """构建变换管道"""
        transform_list = []

        if self.target_size:
            transform_list.append(transforms.Resize(self.target_size))

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.transform = transforms.Compose(transform_list)

    def process_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """处理单张图像

        Args:
            image: 图像路径或PIL图像

        Returns:
            处理后的张量
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)

        return self.transform(image)

    def process_batch(
        self, images: List[Union[str, Path, Image.Image]]
    ) -> torch.Tensor:
        """批量处理图像

        Args:
            images: 图像列表

        Returns:
            批量张量
        """
        processed_images = []

        for image in images:
            processed_image = self.process_image(image)
            processed_images.append(processed_image)

        return torch.stack(processed_images)

    def preprocess_for_model(
        self, image: Union[str, Path, Image.Image], add_batch_dim: bool = True
    ) -> torch.Tensor:
        """为模型预处理图像

        Args:
            image: 输入图像
            add_batch_dim: 是否添加批次维度

        Returns:
            预处理后的张量
        """
        tensor = self.process_image(image)

        if add_batch_dim:
            tensor = tensor.unsqueeze(0)

        return tensor

    def postprocess_from_model(
        self, tensor: torch.Tensor, remove_batch_dim: bool = True
    ) -> Image.Image:
        """从模型输出后处理图像

        Args:
            tensor: 模型输出张量
            remove_batch_dim: 是否移除批次维度

        Returns:
            PIL图像
        """
        if remove_batch_dim and tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # 反标准化
        if self.normalize:
            tensor = denormalize_image(tensor, self.mean, self.std)

        return tensor_to_pil(tensor)


def create_image_grid(
    images: List[Image.Image],
    grid_size: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    spacing: int = 2,
    background_color: Union[int, Tuple[int, ...]] = 255,
) -> Image.Image:
    """创建图像网格

    Args:
        images: 图像列表
        grid_size: 网格大小(rows, cols)，如果为None则自动计算
        image_size: 每个图像的大小，如果为None则使用原始大小
        spacing: 图像间距
        background_color: 背景颜色

    Returns:
        网格图像
    """
    if not images:
        raise ValueError("图像列表不能为空")

    # 确定网格大小
    if grid_size is None:
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    # 确定图像大小
    if image_size is None:
        # 使用第一张图像的大小
        image_size = images[0].size

    # 调整所有图像大小
    resized_images = []
    for img in images:
        if img.size != image_size:
            img = resize_image(img, image_size, maintain_aspect=False)
        resized_images.append(img)

    # 计算网格图像大小
    img_w, img_h = image_size
    grid_w = cols * img_w + (cols - 1) * spacing
    grid_h = rows * img_h + (rows - 1) * spacing

    # 创建网格图像
    grid_image = Image.new("RGB", (grid_w, grid_h), background_color)

    # 粘贴图像
    for i, img in enumerate(resized_images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        x = col * (img_w + spacing)
        y = row * (img_h + spacing)

        grid_image.paste(img, (x, y))

    return grid_image


def apply_augmentation(
    image: Image.Image, augmentation_type: str, **kwargs
) -> Image.Image:
    """应用数据增强

    Args:
        image: 输入图像
        augmentation_type: 增强类型
        **kwargs: 增强参数

    Returns:
        增强后的图像
    """
    if augmentation_type == "rotation":
        angle = kwargs.get("angle", 10)
        return image.rotate(angle, expand=True)

    elif augmentation_type == "flip":
        direction = kwargs.get("direction", "horizontal")
        if direction == "horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            return image.transpose(Image.FLIP_TOP_BOTTOM)

    elif augmentation_type == "brightness":
        factor = kwargs.get("factor", 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    elif augmentation_type == "contrast":
        factor = kwargs.get("factor", 1.2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    elif augmentation_type == "saturation":
        factor = kwargs.get("factor", 1.2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    else:
        raise ValueError(f"不支持的增强类型: {augmentation_type}")
