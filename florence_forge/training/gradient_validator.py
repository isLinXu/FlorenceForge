"""梯度检查和验证机制

提供训练过程中的梯度监控、异常检测和验证功能，确保训练过程的正确性。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """梯度统计信息"""
    step: int
    total_norm: float
    max_grad: float
    min_grad: float
    mean_grad: float
    std_grad: float
    zero_grad_ratio: float
    inf_grad_count: int
    nan_grad_count: int
    layer_norms: Dict[str, float]
    layer_stats: Dict[str, Dict[str, float]]


@dataclass
class GradientValidationConfig:
    """梯度验证配置"""
    # 梯度范数阈值
    max_grad_norm_threshold: float = 10.0
    min_grad_norm_threshold: float = 1e-8
    
    # 梯度爆炸检测
    explosion_multiplier: float = 10.0  # 相对于历史平均值的倍数
    explosion_absolute_threshold: float = 100.0
    
    # 梯度消失检测
    vanishing_threshold: float = 1e-7
    vanishing_ratio_threshold: float = 0.9  # 90%的梯度小于阈值
    
    # 统计窗口大小
    history_window_size: int = 100
    max_stats_history: int = 1000
    
    # 异常检测
    detect_nan: bool = True
    detect_inf: bool = True
    detect_explosion: bool = True
    detect_vanishing: bool = True
    
    # 层级监控
    monitor_layer_gradients: bool = True
    layer_norm_threshold: float = 50.0
    
    # 报告设置
    log_frequency: int = 10  # 每N步记录一次详细信息
    save_stats: bool = True
    stats_save_frequency: int = 100


class GradientValidator:
    """梯度验证器
    
    监控训练过程中的梯度状态，检测异常情况并提供诊断信息。
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[GradientValidationConfig] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        初始化梯度验证器
        
        Args:
            model: 要监控的模型
            config: 验证配置
            output_dir: 输出目录
        """
        self.model = model
        self.config = config or GradientValidationConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        
        # 统计历史
        self.gradient_history: deque = deque(maxlen=self.config.history_window_size)
        self.norm_history: deque = deque(maxlen=self.config.history_window_size)
        self.layer_norm_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_window_size)
        )
        
        # 异常计数
        self.anomaly_counts = {
            'nan_gradients': 0,
            'inf_gradients': 0,
            'gradient_explosion': 0,
            'gradient_vanishing': 0,
            'layer_explosion': 0
        }
        
        # 统计信息
        self.stats_history: List[GradientStats] = []
        self.current_step = 0
        
        # 创建输出目录
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.stats_file = self.output_dir / "gradient_stats.json"
            self.anomaly_file = self.output_dir / "gradient_anomalies.json"
    
    def validate_gradients(self, step: int) -> Tuple[bool, Dict[str, Any]]:
        """
        验证当前步骤的梯度
        
        Args:
            step: 当前训练步骤
            
        Returns:
            (is_valid, validation_info): 验证结果和详细信息
        """
        self.current_step = step
        
        # 收集梯度统计
        stats = self._collect_gradient_stats(step)
        self.stats_history.append(stats)
        if self.config.max_stats_history > 0 and len(self.stats_history) > self.config.max_stats_history:
            self.stats_history = self.stats_history[-self.config.max_stats_history:]
        elif self.config.max_stats_history == 0:
            self.stats_history.clear()
        
        # 检测异常
        anomalies = self._detect_anomalies(stats)
        
        # 判断是否有严重异常
        is_valid = not any([
            anomalies.get('nan_gradients', False),
            anomalies.get('inf_gradients', False),
            anomalies.get('severe_explosion', False)
        ])
        
        # 构建验证信息
        validation_info = {
            'step': step,
            'gradient_stats': asdict(stats),
            'anomalies': anomalies,
            'is_valid': is_valid,
            'recommendations': self._generate_recommendations(stats, anomalies)
        }
        
        # 记录日志
        self._log_validation_results(validation_info)
        
        # 保存统计信息
        if self.config.save_stats and step % self.config.stats_save_frequency == 0:
            self._save_stats()
        
        return is_valid, validation_info
    
    def _collect_gradient_stats(self, step: int) -> GradientStats:
        """收集梯度统计信息"""
        all_grads = []
        layer_norms = {}
        layer_stats = {}
        inf_count = 0
        nan_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # 检查NaN和Inf
                if torch.isnan(grad).any():
                    nan_count += torch.isnan(grad).sum().item()
                if torch.isinf(grad).any():
                    inf_count += torch.isinf(grad).sum().item()
                
                # 过滤有效梯度
                valid_grad = grad[torch.isfinite(grad)]
                if valid_grad.numel() > 0:
                    all_grads.append(valid_grad.flatten())
                    
                    # 层级统计
                    if self.config.monitor_layer_gradients:
                        layer_norm = torch.norm(valid_grad).item()
                        layer_norms[name] = layer_norm
                        
                        layer_stats[name] = {
                            'norm': layer_norm,
                            'mean': valid_grad.mean().item(),
                            'std': valid_grad.std().item(),
                            'max': valid_grad.max().item(),
                            'min': valid_grad.min().item()
                        }
        
        # 计算全局统计
        if all_grads:
            all_grads_tensor = torch.cat(all_grads)
            total_norm = torch.norm(all_grads_tensor).item()
            max_grad = all_grads_tensor.max().item()
            min_grad = all_grads_tensor.min().item()
            mean_grad = all_grads_tensor.mean().item()
            std_grad = all_grads_tensor.std().item()
            zero_grad_ratio = (all_grads_tensor.abs() < 1e-10).float().mean().item()
        else:
            total_norm = max_grad = min_grad = mean_grad = std_grad = zero_grad_ratio = 0.0
        
        # 更新历史
        self.norm_history.append(total_norm)
        for name, norm in layer_norms.items():
            self.layer_norm_history[name].append(norm)
        
        return GradientStats(
            step=step,
            total_norm=total_norm,
            max_grad=max_grad,
            min_grad=min_grad,
            mean_grad=mean_grad,
            std_grad=std_grad,
            zero_grad_ratio=zero_grad_ratio,
            inf_grad_count=inf_count,
            nan_grad_count=nan_count,
            layer_norms=layer_norms,
            layer_stats=layer_stats
        )
    
    def _detect_anomalies(self, stats: GradientStats) -> Dict[str, Any]:
        """检测梯度异常"""
        anomalies = {}
        
        # 检测NaN和Inf
        if self.config.detect_nan and stats.nan_grad_count > 0:
            anomalies['nan_gradients'] = True
            anomalies['nan_count'] = stats.nan_grad_count
            self.anomaly_counts['nan_gradients'] += 1
            logger.warning(f"步骤 {stats.step}: 检测到 {stats.nan_grad_count} 个NaN梯度")
        
        if self.config.detect_inf and stats.inf_grad_count > 0:
            anomalies['inf_gradients'] = True
            anomalies['inf_count'] = stats.inf_grad_count
            self.anomaly_counts['inf_gradients'] += 1
            logger.warning(f"步骤 {stats.step}: 检测到 {stats.inf_grad_count} 个Inf梯度")
        
        # 检测梯度爆炸
        if self.config.detect_explosion:
            explosion_detected = False
            
            # 绝对阈值检测
            if stats.total_norm > self.config.explosion_absolute_threshold:
                explosion_detected = True
                anomalies['absolute_explosion'] = True
            
            # 相对阈值检测（相对于历史平均值）
            if len(self.norm_history) >= 10:
                avg_norm = np.mean(list(self.norm_history)[:-1])  # 排除当前值
                if stats.total_norm > avg_norm * self.config.explosion_multiplier:
                    explosion_detected = True
                    anomalies['relative_explosion'] = True
                    anomalies['explosion_ratio'] = stats.total_norm / avg_norm
            
            if explosion_detected:
                self.anomaly_counts['gradient_explosion'] += 1
                anomalies['gradient_explosion'] = True
                # 严重爆炸（超过绝对阈值的10倍）
                if stats.total_norm > self.config.explosion_absolute_threshold * 10:
                    anomalies['severe_explosion'] = True
                logger.warning(f"步骤 {stats.step}: 检测到梯度爆炸，范数={stats.total_norm:.6f}")
        
        # 检测梯度消失
        if self.config.detect_vanishing:
            if stats.total_norm < self.config.min_grad_norm_threshold:
                anomalies['gradient_vanishing'] = True
                self.anomaly_counts['gradient_vanishing'] += 1
                logger.warning(f"步骤 {stats.step}: 检测到梯度消失，范数={stats.total_norm:.2e}")
            
            # 检测大部分梯度过小
            if stats.zero_grad_ratio > self.config.vanishing_ratio_threshold:
                anomalies['high_zero_ratio'] = True
                anomalies['zero_ratio'] = stats.zero_grad_ratio
                logger.warning(f"步骤 {stats.step}: {stats.zero_grad_ratio:.1%} 的梯度接近零")
        
        # 检测层级异常
        if self.config.monitor_layer_gradients:
            layer_anomalies = {}
            for name, norm in stats.layer_norms.items():
                if norm > self.config.layer_norm_threshold:
                    layer_anomalies[name] = {
                        'type': 'explosion',
                        'norm': norm,
                        'threshold': self.config.layer_norm_threshold
                    }
                    self.anomaly_counts['layer_explosion'] += 1
            
            if layer_anomalies:
                anomalies['layer_anomalies'] = layer_anomalies
                logger.warning(f"步骤 {stats.step}: 检测到 {len(layer_anomalies)} 个层级梯度异常")
        
        return anomalies
    
    def _generate_recommendations(self, stats: GradientStats, anomalies: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if anomalies.get('nan_gradients'):
            recommendations.extend([
                "检测到NaN梯度，建议：",
                "1. 检查损失函数是否包含无效操作",
                "2. 降低学习率",
                "3. 使用梯度裁剪",
                "4. 检查输入数据是否包含NaN值"
            ])
        
        if anomalies.get('inf_gradients'):
            recommendations.extend([
                "检测到Inf梯度，建议：",
                "1. 降低学习率",
                "2. 使用梯度裁剪",
                "3. 检查模型权重初始化"
            ])
        
        if anomalies.get('gradient_explosion'):
            recommendations.extend([
                "检测到梯度爆炸，建议：",
                "1. 降低学习率",
                "2. 使用梯度裁剪（当前阈值可能过高）",
                "3. 使用更稳定的优化器（如AdamW）",
                "4. 检查网络架构是否过深"
            ])
        
        if anomalies.get('gradient_vanishing'):
            recommendations.extend([
                "检测到梯度消失，建议：",
                "1. 增加学习率",
                "2. 使用残差连接",
                "3. 使用更好的权重初始化",
                "4. 考虑使用BatchNorm或LayerNorm"
            ])
        
        if anomalies.get('high_zero_ratio'):
            recommendations.append(f"大量梯度接近零（{anomalies['zero_ratio']:.1%}），可能存在梯度消失问题")
        
        if anomalies.get('layer_anomalies'):
            layer_names = list(anomalies['layer_anomalies'].keys())
            recommendations.append(f"层级异常：{', '.join(layer_names[:3])}{'等' if len(layer_names) > 3 else ''}")
        
        return recommendations
    
    def _log_validation_results(self, validation_info: Dict[str, Any]) -> None:
        """记录验证结果"""
        step = validation_info['step']
        stats = validation_info['gradient_stats']
        anomalies = validation_info['anomalies']
        
        # 基本信息每步都记录
        if not validation_info['is_valid']:
            logger.error(f"步骤 {step}: 梯度验证失败")
        
        # 详细信息按频率记录
        if step % self.config.log_frequency == 0:
            logger.info(
                f"步骤 {step} 梯度统计: "
                f"范数={stats['total_norm']:.6f}, "
                f"均值={stats['mean_grad']:.2e}, "
                f"标准差={stats['std_grad']:.2e}, "
                f"零梯度比例={stats['zero_grad_ratio']:.1%}"
            )
            
            if anomalies:
                logger.warning(f"步骤 {step} 检测到异常: {list(anomalies.keys())}")
            
            # 记录建议
            recommendations = validation_info['recommendations']
            if recommendations:
                logger.info(f"优化建议: {'; '.join(recommendations[:3])}")
    
    def _save_stats(self) -> None:
        """保存统计信息到文件"""
        if not self.output_dir:
            return
        
        try:
            # 保存梯度统计
            stats_data = {
                'config': asdict(self.config),
                'stats_history': [asdict(stats) for stats in self.stats_history[-self.config.stats_save_frequency:]],
                'anomaly_counts': self.anomaly_counts.copy(),
                'summary': self.get_summary()
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"梯度统计已保存到: {self.stats_file}")
            
        except Exception as e:
            logger.error(f"保存梯度统计失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-min(50, len(self.stats_history)):]
        norms = [s.total_norm for s in recent_stats]
        
        return {
            'total_steps': len(self.stats_history),
            'anomaly_counts': self.anomaly_counts.copy(),
            'recent_gradient_stats': {
                'avg_norm': np.mean(norms) if norms else 0.0,
                'std_norm': np.std(norms) if norms else 0.0,
                'max_norm': max(norms) if norms else 0.0,
                'min_norm': min(norms) if norms else 0.0
            },
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """计算梯度健康评分（0-100）"""
        if not self.stats_history:
            return 100.0
        
        score = 100.0
        total_steps = len(self.stats_history)
        
        # 异常惩罚
        for anomaly_type, count in self.anomaly_counts.items():
            if count > 0:
                penalty = min(count / total_steps * 100, 50)  # 最多扣50分
                if anomaly_type in ['nan_gradients', 'inf_gradients']:
                    penalty *= 2  # NaN和Inf更严重
                score -= penalty
        
        # 稳定性奖励
        if len(self.norm_history) >= 10:
            norm_std = np.std(list(self.norm_history))
            norm_mean = np.mean(list(self.norm_history))
            if norm_mean > 0:
                cv = norm_std / norm_mean  # 变异系数
                if cv < 0.5:  # 变异系数小于0.5认为稳定
                    score += min(10, (0.5 - cv) * 20)
        
        return max(0.0, min(100.0, score))
    
    def reset(self) -> None:
        """重置验证器状态"""
        self.gradient_history.clear()
        self.norm_history.clear()
        self.layer_norm_history.clear()
        self.anomaly_counts = {key: 0 for key in self.anomaly_counts}
        self.stats_history.clear()
        self.current_step = 0
        
        logger.info("梯度验证器已重置")
    
    def export_report(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """导出验证报告"""
        if file_path is None:
            file_path = self.output_dir / "gradient_validation_report.json" if self.output_dir else "gradient_report.json"
        
        report = {
            'validation_config': asdict(self.config),
            'summary': self.get_summary(),
            'detailed_stats': [asdict(stats) for stats in self.stats_history],
            'anomaly_timeline': self._get_anomaly_timeline(),
            'layer_analysis': self._get_layer_analysis()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"梯度验证报告已导出到: {file_path}")
        return str(file_path)
    
    def _get_anomaly_timeline(self) -> List[Dict[str, Any]]:
        """获取异常时间线"""
        timeline = []
        for stats in self.stats_history:
            if (stats.nan_grad_count > 0 or stats.inf_grad_count > 0 or 
                stats.total_norm > self.config.explosion_absolute_threshold or
                stats.total_norm < self.config.min_grad_norm_threshold):
                
                timeline.append({
                    'step': stats.step,
                    'total_norm': stats.total_norm,
                    'nan_count': stats.nan_grad_count,
                    'inf_count': stats.inf_grad_count,
                    'zero_ratio': stats.zero_grad_ratio
                })
        
        return timeline
    
    def _get_layer_analysis(self) -> Dict[str, Any]:
        """获取层级分析"""
        if not self.config.monitor_layer_gradients or not self.stats_history:
            return {}
        
        layer_analysis = {}
        
        # 收集所有层的统计信息
        for stats in self.stats_history[-50:]:  # 最近50步
            for layer_name, layer_stats in stats.layer_stats.items():
                if layer_name not in layer_analysis:
                    layer_analysis[layer_name] = {
                        'norms': [],
                        'means': [],
                        'stds': []
                    }
                
                layer_analysis[layer_name]['norms'].append(layer_stats['norm'])
                layer_analysis[layer_name]['means'].append(layer_stats['mean'])
                layer_analysis[layer_name]['stds'].append(layer_stats['std'])
        
        # 计算汇总统计
        summary = {}
        for layer_name, data in layer_analysis.items():
            summary[layer_name] = {
                'avg_norm': np.mean(data['norms']),
                'std_norm': np.std(data['norms']),
                'max_norm': max(data['norms']),
                'stability_score': 1.0 / (1.0 + np.std(data['norms']) / (np.mean(data['norms']) + 1e-8))
            }
        
        return summary
