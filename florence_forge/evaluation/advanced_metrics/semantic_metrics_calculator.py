"""语义评估指标计算器

基于MetricCalculator基类实现的语义相似度评估指标，
包括BERT Score、CLIP Score和句子相似度等高级语义评估功能。
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

from ..metrics import MetricCalculator
from .semantic_metrics import SemanticMetrics

logger = logging.getLogger(__name__)

class SemanticMetricsCalculator(MetricCalculator):
    """语义评估指标计算器
    
    继承自MetricCalculator基类，专门用于计算语义相似度指标，
    包括BERT Score、CLIP Score、句子相似度等高级语义评估功能。
    """
    
    def __init__(self, task_type: str = "semantic", **kwargs):
        """初始化语义指标计算器
        
        Args:
            task_type: 任务类型，默认为"semantic"
            **kwargs: 传递给SemanticMetrics的额外参数
        """
        super().__init__(task_type)
        self.semantic_metrics = SemanticMetrics(**kwargs)
        self.images: List[Any] = []  # 存储图像数据（用于CLIP Score）
        
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
            images: 图像数据列表（用于CLIP Score计算）
            **kwargs: 其他参数
        """
        super().add_batch(predictions, references, **kwargs)
        
        # 存储图像数据
        if images is not None:
            self.images.extend(images)
        else:
            # 如果没有提供图像，添加None占位符
            self.images.extend([None] * len(predictions))
    
    def compute(self) -> Dict[str, float]:
        """计算语义评估指标
        
        Returns:
            包含各种语义相似度指标的字典
        """
        if not self.predictions or not self.references:
            logger.warning("没有足够的数据进行语义指标计算")
            return {}
        
        metrics = {}
        
        try:
            # 计算基础指标
            base_metrics = super().compute()
            metrics.update(base_metrics)
            
            # 计算BERT Score
            logger.info("计算BERT Score...")
            bert_scores = self.semantic_metrics.calculate_bert_score(
                self.predictions, self.references
            )
            if bert_scores:
                metrics.update({
                    'bert_score_precision': np.mean([score['precision'] for score in bert_scores]),
                    'bert_score_recall': np.mean([score['recall'] for score in bert_scores]),
                    'bert_score_f1': np.mean([score['f1'] for score in bert_scores])
                })
            
            # 计算句子相似度
            logger.info("计算句子相似度...")
            sentence_similarities = self.semantic_metrics.calculate_sentence_similarity(
                self.predictions, self.references
            )
            if sentence_similarities:
                metrics['sentence_similarity'] = np.mean(sentence_similarities)
                metrics['sentence_similarity_std'] = np.std(sentence_similarities)
            
            # 如果有图像数据，计算CLIP Score
            valid_images = [img for img in self.images if img is not None]
            if valid_images and len(valid_images) == len(self.predictions):
                logger.info("计算CLIP Score...")
                clip_scores = self.semantic_metrics.calculate_clip_score(
                    self.predictions, valid_images
                )
                if clip_scores:
                    metrics['clip_score'] = np.mean(clip_scores)
                    metrics['clip_score_std'] = np.std(clip_scores)
            
            # 计算综合语义分数
            semantic_score = self.semantic_metrics.calculate_semantic_score(
                self.predictions, self.references, valid_images if valid_images else None
            )
            if semantic_score is not None:
                metrics['semantic_score'] = semantic_score
            
            # 添加统计信息
            metrics.update({
                'num_samples': len(self.predictions),
                'num_with_images': len(valid_images),
                'avg_prediction_length': np.mean([len(pred.split()) for pred in self.predictions]),
                'avg_reference_length': np.mean([len(ref.split()) for ref in self.references])
            })
            
        except Exception as e:
            logger.error(f"语义指标计算失败: {e}")
            # 返回基础指标作为后备
            metrics = super().compute()
        
        return metrics
    
    def reset(self) -> None:
        """重置计算器状态"""
        super().reset()
        self.images.clear()
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """获取详细的评估结果
        
        Returns:
            包含详细评估信息的字典
        """
        if not self.predictions or not self.references:
            return {}
        
        results = {
            'summary': self.compute(),
            'sample_details': []
        }
        
        # 计算每个样本的详细结果
        try:
            bert_scores = self.semantic_metrics.calculate_bert_score(
                self.predictions, self.references
            )
            sentence_similarities = self.semantic_metrics.calculate_sentence_similarity(
                self.predictions, self.references
            )
            
            valid_images = [img for img in self.images if img is not None]
            clip_scores = []
            if valid_images and len(valid_images) == len(self.predictions):
                clip_scores = self.semantic_metrics.calculate_clip_score(
                    self.predictions, valid_images
                )
            
            for i, (pred, ref) in enumerate(zip(self.predictions, self.references)):
                sample_result = {
                    'index': i,
                    'prediction': pred,
                    'reference': ref,
                    'exact_match': pred.strip().lower() == ref.strip().lower()
                }
                
                if bert_scores and i < len(bert_scores):
                    sample_result['bert_score'] = bert_scores[i]
                
                if sentence_similarities and i < len(sentence_similarities):
                    sample_result['sentence_similarity'] = sentence_similarities[i]
                
                if clip_scores and i < len(clip_scores):
                    sample_result['clip_score'] = clip_scores[i]
                
                results['sample_details'].append(sample_result)
        
        except Exception as e:
            logger.error(f"获取详细结果失败: {e}")
        
        return results
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析语义评估性能
        
        Returns:
            性能分析结果
        """
        if not self.predictions or not self.references:
            return {}
        
        analysis = {
            'data_quality': {},
            'performance_distribution': {},
            'recommendations': []
        }
        
        try:
            # 数据质量分析
            pred_lengths = [len(pred.split()) for pred in self.predictions]
            ref_lengths = [len(ref.split()) for ref in self.references]
            
            analysis['data_quality'] = {
                'avg_prediction_length': np.mean(pred_lengths),
                'avg_reference_length': np.mean(ref_lengths),
                'length_variance_pred': np.var(pred_lengths),
                'length_variance_ref': np.var(ref_lengths),
                'empty_predictions': sum(1 for pred in self.predictions if not pred.strip()),
                'empty_references': sum(1 for ref in self.references if not ref.strip())
            }
            
            # 性能分布分析
            sentence_similarities = self.semantic_metrics.calculate_sentence_similarity(
                self.predictions, self.references
            )
            
            if sentence_similarities:
                analysis['performance_distribution'] = {
                    'similarity_mean': np.mean(sentence_similarities),
                    'similarity_std': np.std(sentence_similarities),
                    'similarity_min': np.min(sentence_similarities),
                    'similarity_max': np.max(sentence_similarities),
                    'high_similarity_ratio': sum(1 for s in sentence_similarities if s > 0.8) / len(sentence_similarities)
                }
            
            # 生成建议
            if analysis['data_quality']['empty_predictions'] > 0:
                analysis['recommendations'].append("检测到空预测，建议检查模型输出")
            
            if analysis['data_quality']['length_variance_pred'] > 100:
                analysis['recommendations'].append("预测长度方差较大，建议检查模型一致性")
            
            if 'similarity_mean' in analysis['performance_distribution']:
                if analysis['performance_distribution']['similarity_mean'] < 0.5:
                    analysis['recommendations'].append("语义相似度较低，建议优化模型或数据质量")
                elif analysis['performance_distribution']['similarity_mean'] > 0.9:
                    analysis['recommendations'].append("语义相似度很高，模型表现良好")
        
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
        
        return analysis
    
    def compute_bert_score(self, predictions: List[str], references: List[str]) -> List[Dict[str, float]]:
        """计算BERT Score
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            BERT Score结果列表
        """
        return self.semantic_metrics.calculate_bert_score(predictions, references)
    
    def compute_clip_score(self, predictions: List[str], images: List[Any]) -> List[float]:
        """计算CLIP Score
        
        Args:
            predictions: 预测文本列表
            images: 图像列表
            
        Returns:
            CLIP Score结果列表
        """
        return self.semantic_metrics.calculate_clip_score(predictions, images)