"""多模态评估指标计算器

基于MetricCalculator基类实现的多模态对齐评估指标，
包括图像-文本匹配、跨模态检索、模态间隙等高级多模态评估功能。
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from ..metrics import MetricCalculator
from .multimodal_metrics import MultiModalMetrics

logger = logging.getLogger(__name__)

class MultiModalMetricsCalculator(MetricCalculator):
    """多模态评估指标计算器
    
    继承自MetricCalculator基类，专门用于计算多模态对齐指标，
    包括图像-文本匹配、跨模态检索、模态间隙等高级多模态评估功能。
    """
    
    def __init__(self, task_type: str = "multimodal", **kwargs):
        """初始化多模态指标计算器
        
        Args:
            task_type: 任务类型，默认为"multimodal"
            **kwargs: 传递给MultiModalMetrics的额外参数
        """
        super().__init__(task_type)
        self.multimodal_metrics = MultiModalMetrics(**kwargs)
        self.images: List[Any] = []
        self.image_features: List[np.ndarray] = []
        self.text_features: List[np.ndarray] = []
        
    def add_batch(
        self,
        predictions: List[str],
        references: List[str],
        images: Optional[List[Any]] = None,
        **kwargs
    ) -> None:
        """添加一批预测和参考数据
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            images: 图像数据列表
            **kwargs: 其他参数
        """
        super().add_batch(predictions, references, **kwargs)
        
        # 存储图像数据
        if images is not None:
            self.images.extend(images)
            
            # 预计算图像和文本特征以提高效率
            try:
                for image, pred, ref in zip(images, predictions, references):
                    # 提取图像特征
                    img_feat = self.multimodal_metrics._extract_image_features([image])
                    if img_feat is not None and len(img_feat) > 0:
                        self.image_features.append(img_feat[0])
                    else:
                        self.image_features.append(np.zeros(512))  # 默认特征维度
                    
                    # 提取文本特征
                    text_feat = self.multimodal_metrics._extract_text_features([pred])
                    if text_feat is not None and len(text_feat) > 0:
                        self.text_features.append(text_feat[0])
                    else:
                        self.text_features.append(np.zeros(512))  # 默认特征维度
            except Exception as e:
                logger.warning(f"特征提取失败: {e}")
        else:
            # 如果没有提供图像，添加占位符
            self.images.extend([None] * len(predictions))
            self.image_features.extend([np.zeros(512)] * len(predictions))
            self.text_features.extend([np.zeros(512)] * len(predictions))
    
    def compute(self) -> Dict[str, float]:
        """计算多模态评估指标
        
        Returns:
            包含各种多模态对齐指标的字典
        """
        if not self.predictions or not self.references:
            logger.warning("没有足够的数据进行多模态指标计算")
            return {}
        
        metrics = {}
        
        try:
            # 计算基础指标
            base_metrics = super().compute()
            metrics.update(base_metrics)
            
            # 过滤有效的图像数据
            valid_indices = [i for i, img in enumerate(self.images) if img is not None]
            
            if not valid_indices:
                logger.warning("没有有效的图像数据，跳过多模态指标计算")
                return metrics
            
            valid_images = [self.images[i] for i in valid_indices]
            valid_predictions = [self.predictions[i] for i in valid_indices]
            valid_references = [self.references[i] for i in valid_indices]
            
            # 计算图像-文本匹配分数
            logger.info("计算图像-文本匹配分数...")
            matching_scores = self.multimodal_metrics.calculate_image_text_matching(
                valid_predictions, valid_images
            )
            if matching_scores:
                metrics.update({
                    'image_text_matching_mean': np.mean(matching_scores),
                    'image_text_matching_std': np.std(matching_scores),
                    'image_text_matching_min': np.min(matching_scores),
                    'image_text_matching_max': np.max(matching_scores)
                })
            
            # 计算跨模态检索指标
            logger.info("计算跨模态检索指标...")
            retrieval_metrics = self.multimodal_metrics.calculate_cross_modal_retrieval(
                valid_predictions, valid_images
            )
            if retrieval_metrics:
                metrics.update({
                    f'text_to_image_{k}': v for k, v in retrieval_metrics.items() 
                    if k.startswith('text_to_image')
                })
                metrics.update({
                    f'image_to_text_{k}': v for k, v in retrieval_metrics.items() 
                    if k.startswith('image_to_text')
                })
            
            # 计算模态间隙
            logger.info("计算模态间隙...")
            modality_gap = self.multimodal_metrics.calculate_modality_gap(
                valid_predictions, valid_images
            )
            if modality_gap is not None:
                metrics['modality_gap'] = modality_gap
            
            # 计算对齐一致性
            logger.info("计算对齐一致性...")
            alignment_consistency = self.multimodal_metrics.calculate_alignment_consistency(
                valid_predictions, valid_images
            )
            if alignment_consistency is not None:
                metrics['alignment_consistency'] = alignment_consistency
            
            # 计算综合多模态分数
            multimodal_score = self.multimodal_metrics.calculate_multimodal_score(
                valid_predictions, valid_images
            )
            if multimodal_score is not None:
                metrics['multimodal_score'] = multimodal_score
            
            # 添加统计信息
            metrics.update({
                'num_multimodal_samples': len(valid_indices),
                'multimodal_coverage': len(valid_indices) / len(self.predictions),
                'avg_image_text_feature_similarity': self._calculate_avg_feature_similarity(valid_indices)
            })
            
        except Exception as e:
            logger.error(f"多模态指标计算失败: {e}")
            # 返回基础指标作为后备
            metrics = super().compute()
        
        return metrics
    
    def _calculate_avg_feature_similarity(self, valid_indices: List[int]) -> float:
        """计算平均特征相似度
        
        Args:
            valid_indices: 有效样本索引列表
            
        Returns:
            平均特征相似度
        """
        if not valid_indices or not self.image_features or not self.text_features:
            return 0.0
        
        similarities = []
        for i in valid_indices:
            if i < len(self.image_features) and i < len(self.text_features):
                img_feat = self.image_features[i]
                text_feat = self.text_features[i]
                
                # 计算余弦相似度
                if np.linalg.norm(img_feat) > 0 and np.linalg.norm(text_feat) > 0:
                    similarity = np.dot(img_feat, text_feat) / (
                        np.linalg.norm(img_feat) * np.linalg.norm(text_feat)
                    )
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def reset(self) -> None:
        """重置计算器状态"""
        super().reset()
        self.images.clear()
        self.image_features.clear()
        self.text_features.clear()
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """获取详细的评估结果
        
        Returns:
            包含详细评估信息的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        results = {
            'summary': self.compute(),
            'sample_details': [],
            'modality_analysis': {}
        }
        
        try:
            # 过滤有效的图像数据
            valid_indices = [i for i, img in enumerate(self.images) if img is not None]
            
            if valid_indices:
                valid_images = [self.images[i] for i in valid_indices]
                valid_predictions = [self.predictions[i] for i in valid_indices]
                
                # 计算每个样本的详细结果
                matching_scores = self.multimodal_metrics.calculate_image_text_matching(
                    valid_predictions, valid_images
                )
                
                for idx, i in enumerate(valid_indices):
                    sample_result = {
                        'index': i,
                        'prediction': self.predictions[i],
                        'reference': self.references[i],
                        'has_image': True,
                        'exact_match': self.predictions[i].strip().lower() == self.references[i].strip().lower()
                    }
                    
                    if matching_scores and idx < len(matching_scores):
                        sample_result['image_text_matching'] = matching_scores[idx]
                    
                    if idx < len(self.image_features) and idx < len(self.text_features):
                        sample_result['feature_similarity'] = self._calculate_feature_similarity(
                            self.image_features[i], self.text_features[i]
                        )
                    
                    results['sample_details'].append(sample_result)
            
            # 添加没有图像的样本
            for i, (pred, ref) in enumerate(zip(self.predictions, self.references)):
                if i not in valid_indices:
                    sample_result = {
                        'index': i,
                        'prediction': pred,
                        'reference': ref,
                        'has_image': False,
                        'exact_match': pred.strip().lower() == ref.strip().lower()
                    }
                    results['sample_details'].append(sample_result)
            
            # 模态分析
            results['modality_analysis'] = {
                'total_samples': len(self.predictions),
                'samples_with_images': len(valid_indices),
                'image_coverage': len(valid_indices) / len(self.predictions) if self.predictions else 0,
                'avg_prediction_length': np.mean([len(pred.split()) for pred in self.predictions]),
                'avg_reference_length': np.mean([len(ref.split()) for ref in self.references])
            }
        
        except Exception as e:
            logger.error(f"获取详细结果失败: {e}")
        
        return results
    
    def _calculate_feature_similarity(self, img_feat: np.ndarray, text_feat: np.ndarray) -> float:
        """计算单个样本的特征相似度
        
        Args:
            img_feat: 图像特征
            text_feat: 文本特征
            
        Returns:
            特征相似度
        """
        if np.linalg.norm(img_feat) > 0 and np.linalg.norm(text_feat) > 0:
            return float(np.dot(img_feat, text_feat) / (
                np.linalg.norm(img_feat) * np.linalg.norm(text_feat)
            ))
        return 0.0
    
    def compute_image_text_matching(self, images: Optional[List[Any]] = None, texts: Optional[List[str]] = None) -> Dict[str, float]:
        """计算图像-文本匹配分数
        
        Args:
            images: 图像列表（可选，如果不提供则使用内部存储的图像）
            texts: 文本列表（可选，如果不提供则使用内部存储的预测文本）
        
        Returns:
            图像-文本匹配分数字典
        """
        # 使用提供的参数或内部存储的数据
        image_data = images if images is not None else self.images
        text_data = texts if texts is not None else self.predictions
        
        if not text_data or not image_data:
            return {'matching_score': 0.0, 'individual_scores': []}
        
        try:
            valid_indices = [i for i, img in enumerate(image_data) if img is not None]
            if not valid_indices:
                return {'matching_score': 0.0, 'individual_scores': []}
            
            valid_images = [image_data[i] for i in valid_indices]
            valid_predictions = [text_data[i] for i in valid_indices]
            
            matching_scores = self.multimodal_metrics.calculate_image_text_matching(
                valid_predictions, valid_images
            )
            
            if matching_scores:
                return {
                    'matching_score': np.mean(matching_scores),
                    'individual_scores': matching_scores
                }
            else:
                return {'matching_score': 0.0, 'individual_scores': []}
        
        except Exception as e:
            logger.error(f"计算图像-文本匹配分数失败: {e}")
            return {'matching_score': 0.0, 'individual_scores': []}
    
    def compute_cross_modal_retrieval(self) -> Dict[str, float]:
        """计算跨模态检索指标
        
        Returns:
            跨模态检索指标字典
        """
        if not self.predictions or not self.images:
            return {}
        
        try:
            valid_indices = [i for i, img in enumerate(self.images) if img is not None]
            if not valid_indices:
                return {}
            
            valid_images = [self.images[i] for i in valid_indices]
            valid_predictions = [self.predictions[i] for i in valid_indices]
            
            retrieval_metrics = self.multimodal_metrics.calculate_cross_modal_retrieval(
                valid_predictions, valid_images
            )
            
            return retrieval_metrics if retrieval_metrics else {}
        
        except Exception as e:
            logger.error(f"计算跨模态检索指标失败: {e}")
            return {}
    
    def analyze_cross_modal_performance(self) -> Dict[str, Any]:
        """分析跨模态性能
        
        Returns:
            跨模态性能分析结果
        """
        if not self.predictions or not self.images:
            return {}
        
        analysis = {
            'retrieval_performance': {},
            'alignment_quality': {},
            'modality_distribution': {},
            'recommendations': []
        }
        
        try:
            valid_indices = [i for i, img in enumerate(self.images) if img is not None]
            
            if valid_indices:
                valid_images = [self.images[i] for i in valid_indices]
                valid_predictions = [self.predictions[i] for i in valid_indices]
                
                # 检索性能分析
                retrieval_metrics = self.multimodal_metrics.calculate_cross_modal_retrieval(
                    valid_predictions, valid_images
                )
                if retrieval_metrics:
                    analysis['retrieval_performance'] = retrieval_metrics
                
                # 对齐质量分析
                matching_scores = self.multimodal_metrics.calculate_image_text_matching(
                    valid_predictions, valid_images
                )
                if matching_scores:
                    analysis['alignment_quality'] = {
                        'mean_matching_score': np.mean(matching_scores),
                        'std_matching_score': np.std(matching_scores),
                        'high_quality_ratio': sum(1 for s in matching_scores if s > 0.8) / len(matching_scores),
                        'low_quality_ratio': sum(1 for s in matching_scores if s < 0.3) / len(matching_scores)
                    }
                
                # 模态分布分析
                analysis['modality_distribution'] = {
                    'image_coverage': len(valid_indices) / len(self.predictions),
                    'avg_text_length': np.mean([len(pred.split()) for pred in valid_predictions]),
                    'text_length_variance': np.var([len(pred.split()) for pred in valid_predictions])
                }
                
                # 生成建议
                if analysis['modality_distribution']['image_coverage'] < 0.5:
                    analysis['recommendations'].append("图像覆盖率较低，建议增加图像数据")
                
                if 'mean_matching_score' in analysis['alignment_quality']:
                    if analysis['alignment_quality']['mean_matching_score'] < 0.5:
                        analysis['recommendations'].append("图像-文本匹配度较低，建议优化多模态对齐")
                    elif analysis['alignment_quality']['mean_matching_score'] > 0.8:
                        analysis['recommendations'].append("多模态对齐质量良好")
                
                if 'text_to_image_recall@1' in analysis['retrieval_performance']:
                    if analysis['retrieval_performance']['text_to_image_recall@1'] < 0.3:
                        analysis['recommendations'].append("跨模态检索性能较低，建议改进特征提取")
        
        except Exception as e:
            logger.error(f"跨模态性能分析失败: {e}")
        
        return analysis