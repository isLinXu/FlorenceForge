"""图像数据增强"""

import random
from typing import Tuple, List, Optional
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class ImageAugmentation:
    """图像数据增强"""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def random_brightness(self, image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """随机亮度调整"""
        if random.random() < self.probability:
            factor = random.uniform(*factor_range)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        return image
    
    def random_contrast(self, image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """随机对比度调整"""
        if random.random() < self.probability:
            factor = random.uniform(*factor_range)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        return image
    
    def random_saturation(self, image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """随机饱和度调整"""
        if random.random() < self.probability:
            factor = random.uniform(*factor_range)
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        return image
    
    def random_blur(self, image: Image.Image, radius_range: Tuple[float, float] = (0.1, 2.0)) -> Image.Image:
        """随机模糊"""
        if random.random() < self.probability:
            radius = random.uniform(*radius_range)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image
    
    def random_noise(self, image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
        """添加随机噪声"""
        if random.random() < self.probability:
            img_array = np.array(image)
            noise = np.random.normal(0, noise_factor * 255, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return image
    
    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        """应用所有增强"""
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_saturation(image)
        image = self.random_blur(image)
        image = self.random_noise(image)
        return image
