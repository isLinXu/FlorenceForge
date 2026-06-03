from __future__ import annotations

"""多模态对齐评估指标

提供图像-文本对齐、跨模态检索等多模态评估功能
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict

from ...utils.optional_dependencies import missing_dependency_message

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("多模态评估功能", "torch")
    )

try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    Image = None
    CLIP_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("CLIP相关功能", "clip-by-openai", "evaluation")
    )

try:
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("BLIP相关功能", "transformers", "evaluation")
    )

try:
    from sklearn.metrics import average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("部分评估指标", "scikit-learn")
    )

logger = logging.getLogger(__name__)


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


class MultiModalMetrics:
    """多模态对齐评估指标
    
    提供图像-文本对齐、跨模态检索等评估功能
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        blip_model_name: str = "Salesforce/blip-itm-base-coco",
        device: Optional[torch.device] = None
    ):
        """初始化多模态指标计算器
        
        Args:
            clip_model_name: CLIP模型名称
            blip_model_name: BLIP模型名称
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化CLIP模型
        if CLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
                self.clip_available = True
                logger.info(f"CLIP模型 {clip_model_name} 加载成功")
            except Exception as e:
                logger.warning(f"CLIP模型加载失败: {e}")
                self.clip_available = False
        else:
            self.clip_available = False
        
        # 初始化BLIP模型
        if BLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
                self.blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_name)
                self.blip_model.to(self.device)
                self.blip_model.eval()
                self.blip_available = True
                logger.info(f"BLIP模型 {blip_model_name} 加载成功")
            except Exception as e:
                logger.warning(f"BLIP模型加载失败: {e}")
                self.blip_available = False
        else:
            self.blip_available = False
    
    def calculate_image_text_matching_score(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        method: str = "clip",
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """计算图像-文本匹配分数
        
        Args:
            images: 图像路径或PIL图像列表
            texts: 文本列表
            method: 使用的方法 ('clip', 'blip')
            batch_size: 批处理大小
            
        Returns:
            图像-文本匹配结果字典
        """
        if len(images) != len(texts):
            raise ValueError("图像和文本数量不匹配")
        
        if method == "clip" and self.clip_available:
            return self._calculate_clip_itm_score(images, texts, batch_size)
        elif method == "blip" and self.blip_available:
            return self._calculate_blip_itm_score(images, texts, batch_size)
        else:
            logger.warning(f"方法 {method} 不可用，返回默认值")
            return {"matching_score": 0.0, "individual_scores": [0.0] * len(images)}
    
    def _calculate_clip_itm_score(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        batch_size: int
    ) -> Dict[str, Any]:
        """使用CLIP计算图像-文本匹配分数"""
        try:
            scores = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                # 预处理图像
                if isinstance(batch_images[0], str):
                    processed_images = torch.stack([
                        self.clip_preprocess(_load_rgb_image(img_path))
                        for img_path in batch_images
                    ]).to(self.device)
                else:
                    processed_images = torch.stack([
                        self.clip_preprocess(img)
                        for img in batch_images
                    ]).to(self.device)
                
                # 编码文本
                text_tokens = clip.tokenize(batch_texts).to(self.device)
                
                with torch.no_grad():
                    # 获取特征
                    image_features = self.clip_model.encode_image(processed_images)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # 归一化
                    image_features = F.normalize(image_features, dim=1)
                    text_features = F.normalize(text_features, dim=1)
                    
                    # 计算相似度
                    similarity = torch.sum(image_features * text_features, dim=1)
                    scores.extend(similarity.cpu().numpy())
            
            avg_score = np.mean(scores)
            
            return {
                "matching_score": float(avg_score),
                "individual_scores": scores,
                "method": "clip",
                "score_distribution": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
            }
            
        except Exception as e:
            logger.error(f"CLIP图像-文本匹配计算失败: {e}")
            return {"matching_score": 0.0, "individual_scores": [0.0] * len(images)}
    
    def _calculate_blip_itm_score(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        batch_size: int
    ) -> Dict[str, Any]:
        """使用BLIP计算图像-文本匹配分数"""
        try:
            scores = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                # 预处理
                if isinstance(batch_images[0], str):
                    batch_images = [_load_rgb_image(img_path) for img_path in batch_images]
                
                inputs = self.blip_processor(
                    images=batch_images,
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.blip_model(**inputs)
                    # BLIP ITM分数
                    itm_scores = torch.softmax(outputs.logits, dim=1)[:, 1]  # 匹配概率
                    scores.extend(itm_scores.cpu().numpy())
            
            avg_score = np.mean(scores)
            
            return {
                "matching_score": float(avg_score),
                "individual_scores": scores,
                "method": "blip",
                "score_distribution": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
            }
            
        except Exception as e:
            logger.error(f"BLIP图像-文本匹配计算失败: {e}")
            return {"matching_score": 0.0, "individual_scores": [0.0] * len(images)}
    
    def calculate_cross_modal_retrieval_metrics(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        k_values: List[int] = [1, 5, 10],
        method: str = "clip"
    ) -> Dict[str, Any]:
        """计算跨模态检索指标
        
        Args:
            images: 图像列表
            texts: 文本列表
            k_values: Top-K值列表
            method: 使用的方法
            
        Returns:
            跨模态检索指标结果
        """
        if not self.clip_available:
            logger.warning("CLIP模型不可用，返回默认值")
            return {f"recall_at_{k}": 0.0 for k in k_values}
        
        try:
            # 获取图像和文本特征
            image_features, text_features = self._extract_multimodal_features(images, texts)
            
            # 计算相似度矩阵
            similarity_matrix = torch.matmul(image_features, text_features.T)
            
            # 图像到文本检索
            i2t_metrics = self._calculate_retrieval_metrics(
                similarity_matrix, k_values, "image_to_text"
            )
            
            # 文本到图像检索
            t2i_metrics = self._calculate_retrieval_metrics(
                similarity_matrix.T, k_values, "text_to_image"
            )
            
            # 合并结果
            results = {
                "image_to_text": i2t_metrics,
                "text_to_image": t2i_metrics,
                "method": method
            }
            
            # 计算平均指标
            for k in k_values:
                i2t_recall = i2t_metrics[f"recall_at_{k}"]
                t2i_recall = t2i_metrics[f"recall_at_{k}"]
                results[f"avg_recall_at_{k}"] = (i2t_recall + t2i_recall) / 2
            
            logger.info(f"跨模态检索指标计算完成，平均R@1: {results['avg_recall_at_1']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"跨模态检索指标计算失败: {e}")
            return {f"recall_at_{k}": 0.0 for k in k_values}
    
    def _extract_multimodal_features(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取多模态特征"""
        # 预处理图像
        if isinstance(images[0], str):
            processed_images = torch.stack([
                self.clip_preprocess(_load_rgb_image(img_path))
                for img_path in images
            ]).to(self.device)
        else:
            processed_images = torch.stack([
                self.clip_preprocess(img)
                for img in images
            ]).to(self.device)
        
        # 编码文本
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            # 获取特征
            image_features = self.clip_model.encode_image(processed_images)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # 归一化
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        return image_features, text_features
    
    def _calculate_retrieval_metrics(
        self,
        similarity_matrix: torch.Tensor,
        k_values: List[int],
        direction: str
    ) -> Dict[str, float]:
        """计算检索指标"""
        n_samples = similarity_matrix.size(0)
        
        # 获取排序索引
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        # 计算Recall@K
        metrics = {}
        for k in k_values:
            correct = 0
            for i in range(n_samples):
                if i in sorted_indices[i, :k]:
                    correct += 1
            
            recall_at_k = correct / n_samples
            metrics[f"recall_at_{k}"] = recall_at_k
        
        # 计算Mean Rank
        ranks = []
        for i in range(n_samples):
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        metrics["mean_rank"] = float(np.mean(ranks))
        metrics["median_rank"] = float(np.median(ranks))
        
        return metrics
    
    def calculate_modality_gap(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str]
    ) -> Dict[str, float]:
        """计算模态间隙
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            模态间隙指标
        """
        if not self.clip_available:
            logger.warning("CLIP模型不可用，返回默认值")
            return {"modality_gap": 0.0}
        
        try:
            # 获取特征
            image_features, text_features = self._extract_multimodal_features(images, texts)
            
            # 计算特征中心
            image_center = torch.mean(image_features, dim=0)
            text_center = torch.mean(text_features, dim=0)
            
            # 计算模态间隙（中心点之间的距离）
            modality_gap = torch.norm(image_center - text_center).item()
            
            # 计算模态内方差
            image_variance = torch.var(image_features, dim=0).mean().item()
            text_variance = torch.var(text_features, dim=0).mean().item()
            
            # 计算跨模态相关性
            cross_modal_correlation = F.cosine_similarity(
                image_center.unsqueeze(0),
                text_center.unsqueeze(0)
            ).item()
            
            return {
                "modality_gap": modality_gap,
                "image_variance": image_variance,
                "text_variance": text_variance,
                "cross_modal_correlation": cross_modal_correlation,
                "normalized_gap": modality_gap / (image_variance + text_variance + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"模态间隙计算失败: {e}")
            return {"modality_gap": 0.0}
    
    def calculate_alignment_consistency(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        perturbation_strength: float = 0.1
    ) -> Dict[str, float]:
        """计算对齐一致性
        
        Args:
            images: 图像列表
            texts: 文本列表
            perturbation_strength: 扰动强度
            
        Returns:
            对齐一致性指标
        """
        if not self.clip_available:
            logger.warning("CLIP模型不可用，返回默认值")
            return {"alignment_consistency": 0.0}
        
        try:
            # 原始特征
            original_image_features, original_text_features = self._extract_multimodal_features(
                images, texts
            )
            
            # 添加噪声扰动
            noise_image = torch.randn_like(original_image_features) * perturbation_strength
            noise_text = torch.randn_like(original_text_features) * perturbation_strength
            
            perturbed_image_features = F.normalize(
                original_image_features + noise_image, dim=1
            )
            perturbed_text_features = F.normalize(
                original_text_features + noise_text, dim=1
            )
            
            # 计算原始相似度
            original_similarities = torch.sum(
                original_image_features * original_text_features, dim=1
            )
            
            # 计算扰动后相似度
            perturbed_similarities = torch.sum(
                perturbed_image_features * perturbed_text_features, dim=1
            )
            
            # 计算一致性（相关系数）
            consistency = F.cosine_similarity(
                original_similarities.unsqueeze(0),
                perturbed_similarities.unsqueeze(0)
            ).item()
            
            # 计算稳定性（差异的标准差）
            stability = 1.0 - torch.std(original_similarities - perturbed_similarities).item()
            
            return {
                "alignment_consistency": consistency,
                "alignment_stability": stability,
                "perturbation_strength": perturbation_strength,
                "similarity_change_mean": torch.mean(
                    torch.abs(original_similarities - perturbed_similarities)
                ).item()
            }
            
        except Exception as e:
            logger.error(f"对齐一致性计算失败: {e}")
            return {"alignment_consistency": 0.0}
    
    def calculate_comprehensive_multimodal_score(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """计算综合多模态评分
        
        Args:
            images: 图像列表
            texts: 文本列表
            weights: 各指标权重
            
        Returns:
            综合多模态评分结果
        """
        default_weights = {
            "matching_score": 0.4,
            "retrieval_score": 0.3,
            "modality_gap": 0.2,
            "alignment_consistency": 0.1
        }
        
        if weights is None:
            weights = default_weights
        
        results = {}
        total_score = 0.0
        total_weight = 0.0
        
        # 图像-文本匹配分数
        if weights.get("matching_score", 0) > 0:
            matching_result = self.calculate_image_text_matching_score(images, texts)
            results["matching"] = matching_result
            total_score += matching_result["matching_score"] * weights["matching_score"]
            total_weight += weights["matching_score"]
        
        # 跨模态检索分数
        if weights.get("retrieval_score", 0) > 0:
            retrieval_result = self.calculate_cross_modal_retrieval_metrics(images, texts)
            results["retrieval"] = retrieval_result
            # 使用R@1作为检索分数
            retrieval_score = retrieval_result.get("avg_recall_at_1", 0.0)
            total_score += retrieval_score * weights["retrieval_score"]
            total_weight += weights["retrieval_score"]
        
        # 模态间隙（越小越好，需要转换）
        if weights.get("modality_gap", 0) > 0:
            gap_result = self.calculate_modality_gap(images, texts)
            results["modality_gap"] = gap_result
            # 转换为分数（1 - normalized_gap）
            gap_score = max(0.0, 1.0 - gap_result.get("normalized_gap", 1.0))
            total_score += gap_score * weights["modality_gap"]
            total_weight += weights["modality_gap"]
        
        # 对齐一致性
        if weights.get("alignment_consistency", 0) > 0:
            consistency_result = self.calculate_alignment_consistency(images, texts)
            results["alignment_consistency"] = consistency_result
            consistency_score = consistency_result.get("alignment_consistency", 0.0)
            total_score += consistency_score * weights["alignment_consistency"]
            total_weight += weights["alignment_consistency"]
        
        # 计算综合分数
        comprehensive_score = total_score / total_weight if total_weight > 0 else 0.0
        
        results["comprehensive_score"] = comprehensive_score
        results["weights_used"] = {k: v for k, v in weights.items() if k in ["matching_score", "retrieval_score", "modality_gap", "alignment_consistency"]}
        
        logger.info(f"综合多模态评分: {comprehensive_score:.4f}")
        
        return results
