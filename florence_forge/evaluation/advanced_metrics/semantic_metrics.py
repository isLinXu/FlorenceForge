from __future__ import annotations

"""语义相似度评估指标

提供基于深度学习的语义相似度评估功能，包括BERT Score、CLIP Score等
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ...utils.optional_dependencies import missing_dependency_message

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("BERT Score功能", "transformers", "evaluation")
    )

try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    Image = None
    CLIP_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("CLIP Score功能", "clip-by-openai", "evaluation")
    )

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("部分语义相似度功能", "sentence-transformers")
    )

logger = logging.getLogger(__name__)


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


class SemanticMetrics:
    """语义相似度评估指标

    提供多种基于深度学习的语义相似度评估方法
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        clip_model_name: str = "ViT-B/32",
        sentence_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None
    ):
        """初始化语义指标计算器

        Args:
            bert_model_name: BERT模型名称
            clip_model_name: CLIP模型名称
            sentence_model_name: Sentence Transformer模型名称
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化BERT模型
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                self.bert_model = AutoModel.from_pretrained(bert_model_name)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                self.bert_available = True
                logger.info(f"BERT模型 {bert_model_name} 加载成功")
            except Exception as e:
                logger.warning(f"BERT模型加载失败: {e}")
                self.bert_available = False
        else:
            self.bert_available = False

        # 初始化CLIP模型
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
                self.clip_available = True
                logger.info(f"CLIP模型 {clip_model_name} 加载成功")
            except Exception as e:
                logger.warning(f"CLIP模型加载失败: {e}")
                self.clip_available = False
        else:
            self.clip_available = False

        # 初始化Sentence Transformer模型
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name)
                self.sentence_model.to(self.device)
                self.sentence_available = True
                logger.info(f"Sentence Transformer模型 {sentence_model_name} 加载成功")
            except Exception as e:
                logger.warning(f"Sentence Transformer模型加载失败: {e}")
                self.sentence_available = False
        else:
            self.sentence_available = False

    def calculate_bert_score(
        self,
        references: List[str],
        candidates: List[str],
        lang: str = "en",
        verbose: bool = False
    ) -> Dict[str, float]:
        """计算BERT Score

        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            lang: 语言代码
            verbose: 是否输出详细信息

        Returns:
            BERT Score结果字典
        """
        if not self.bert_available:
            logger.warning("BERT模型不可用，返回默认值")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量不匹配")

        try:
            # 获取BERT embeddings
            ref_embeddings = self._get_bert_embeddings(references)
            cand_embeddings = self._get_bert_embeddings(candidates)

            # 计算相似度矩阵
            similarity_matrix = torch.cosine_similarity(
                cand_embeddings.unsqueeze(1),
                ref_embeddings.unsqueeze(0),
                dim=2
            )

            # 计算precision, recall, f1
            precision = similarity_matrix.max(dim=1)[0].mean().item()
            recall = similarity_matrix.max(dim=0)[0].mean().item()
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            result = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            if verbose:
                logger.info(f"BERT Score - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

            return result

        except Exception as e:
            logger.error(f"BERT Score计算失败: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def calculate_clip_score(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """计算CLIP Score

        Args:
            images: 图像路径或PIL图像列表
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            CLIP Score结果字典
        """
        if not self.clip_available:
            logger.warning("CLIP模型不可用，返回默认值")
            return {"clip_score": 0.0, "image_text_similarity": 0.0}

        if len(images) != len(texts):
            raise ValueError("图像和文本数量不匹配")

        try:
            scores = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]

                # 预处理图像
                if isinstance(batch_images[0], str):
                    # 图像路径
                    processed_images = torch.stack([
                        self.clip_preprocess(_load_rgb_image(img_path))
                        for img_path in batch_images
                    ]).to(self.device)
                else:
                    # PIL图像
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

            clip_score = np.mean(scores)

            result = {
                "clip_score": float(clip_score),
                "image_text_similarity": float(clip_score),
                "individual_scores": scores
            }

            logger.info(f"CLIP Score: {clip_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"CLIP Score计算失败: {e}")
            return {"clip_score": 0.0, "image_text_similarity": 0.0}

    def calculate_sentence_similarity(
        self,
        references: List[str],
        candidates: List[str],
        metric: str = "cosine"
    ) -> Dict[str, float]:
        """计算句子级语义相似度

        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            metric: 相似度度量方法 ('cosine', 'euclidean', 'manhattan')

        Returns:
            语义相似度结果字典
        """
        if not self.sentence_available:
            logger.warning("Sentence Transformer模型不可用，返回默认值")
            return {"semantic_similarity": 0.0}

        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量不匹配")

        try:
            # 获取句子embeddings
            ref_embeddings = self.sentence_model.encode(references, convert_to_tensor=True)
            cand_embeddings = self.sentence_model.encode(candidates, convert_to_tensor=True)

            # 计算相似度
            if metric == "cosine":
                similarities = F.cosine_similarity(ref_embeddings, cand_embeddings, dim=1)
            elif metric == "euclidean":
                distances = F.pairwise_distance(ref_embeddings, cand_embeddings)
                similarities = 1 / (1 + distances)  # 转换为相似度
            elif metric == "manhattan":
                distances = torch.sum(torch.abs(ref_embeddings - cand_embeddings), dim=1)
                similarities = 1 / (1 + distances)  # 转换为相似度
            else:
                raise ValueError(f"不支持的相似度度量方法: {metric}")

            avg_similarity = similarities.mean().item()

            result = {
                "semantic_similarity": avg_similarity,
                "individual_similarities": similarities.cpu().numpy().tolist(),
                "metric": metric
            }

            logger.info(f"语义相似度 ({metric}): {avg_similarity:.4f}")
            return result

        except Exception as e:
            logger.error(f"语义相似度计算失败: {e}")
            return {"semantic_similarity": 0.0}

    def _get_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        """获取BERT embeddings"""
        embeddings = []

        for text in texts:
            # 分词
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS] token的embedding
                embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)

    def calculate_comprehensive_semantic_score(
        self,
        references: List[str],
        candidates: List[str],
        images: Optional[List[Union[str, Image.Image]]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """计算综合语义评分

        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            images: 图像列表（可选）
            weights: 各指标权重

        Returns:
            综合语义评分结果
        """
        default_weights = {
            "bert_score": 0.4,
            "sentence_similarity": 0.4,
            "clip_score": 0.2
        }

        if weights is None:
            weights = default_weights

        results = {}
        total_score = 0.0
        total_weight = 0.0

        # BERT Score
        if self.bert_available and weights.get("bert_score", 0) > 0:
            bert_result = self.calculate_bert_score(references, candidates)
            results["bert_score"] = bert_result
            total_score += bert_result["f1"] * weights["bert_score"]
            total_weight += weights["bert_score"]

        # Sentence Similarity
        if self.sentence_available and weights.get("sentence_similarity", 0) > 0:
            sentence_result = self.calculate_sentence_similarity(references, candidates)
            results["sentence_similarity"] = sentence_result
            total_score += sentence_result["semantic_similarity"] * weights["sentence_similarity"]
            total_weight += weights["sentence_similarity"]

        # CLIP Score (如果提供了图像)
        if images and self.clip_available and weights.get("clip_score", 0) > 0:
            clip_result = self.calculate_clip_score(images, candidates)
            results["clip_score"] = clip_result
            total_score += clip_result["clip_score"] * weights["clip_score"]
            total_weight += weights["clip_score"]

        # 计算综合分数
        comprehensive_score = total_score / total_weight if total_weight > 0 else 0.0

        results["comprehensive_score"] = comprehensive_score
        results["weights_used"] = {k: v for k, v in weights.items() if k in results}

        logger.info(f"综合语义评分: {comprehensive_score:.4f}")

        return results
