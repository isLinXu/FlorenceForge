"""鲁棒性评估指标

提供模型鲁棒性评估功能，包括对抗样本测试、噪声鲁棒性、数据分布偏移等
"""

import logging
import warnings
import time
import psutil
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np

from ...utils.optional_dependencies import missing_dependency_message

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("部分鲁棒性评估功能", "torch")
    )

try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn(
        missing_dependency_message("图像扰动功能", "Pillow")
    )

logger = logging.getLogger(__name__)

@dataclass
class RobustnessTestResult:
    """鲁棒性测试结果"""
    original_prediction: Any
    perturbed_prediction: Any
    perturbation_strength: float
    success_rate: float
    confidence_drop: float
    prediction_changed: bool
    test_type: str
    metadata: Dict[str, Any]

class RobustnessMetrics:
    """鲁棒性评估指标
    
    提供全面的模型鲁棒性评估功能
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        """初始化鲁棒性评估器
        
        Args:
            device: 计算设备
            random_seed: 随机种子
        """
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
    
    def generate_adversarial_examples(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10
    ) -> Tuple[Any, List[RobustnessTestResult]]:
        """生成对抗样本
        
        Args:
            model: 待测试模型
            inputs: 输入数据
            targets: 目标标签
            attack_type: 攻击类型 ('fgsm', 'pgd', 'c&w')
            epsilon: 扰动强度
            alpha: 步长
            num_steps: 迭代步数
            
        Returns:
            (adversarial_inputs, test_results): 对抗样本和测试结果
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch未安装，无法生成对抗样本")
            return inputs, []
        
        try:
            model.eval()
            results = []
            
            if attack_type.lower() == "fgsm":
                adv_inputs = self._fgsm_attack(model, inputs, targets, epsilon)
            elif attack_type.lower() == "pgd":
                adv_inputs = self._pgd_attack(model, inputs, targets, epsilon, alpha, num_steps)
            elif attack_type.lower() == "c&w":
                adv_inputs = self._cw_attack(model, inputs, targets, epsilon)
            else:
                logger.warning(f"未知的攻击类型: {attack_type}")
                return inputs, []
            
            # 评估对抗样本效果
            with torch.no_grad():
                original_outputs = model(inputs)
                adversarial_outputs = model(adv_inputs)
                
                for i in range(len(inputs)):
                    # 计算置信度下降
                    orig_conf = torch.max(F.softmax(original_outputs[i], dim=-1)).item()
                    adv_conf = torch.max(F.softmax(adversarial_outputs[i], dim=-1)).item()
                    confidence_drop = orig_conf - adv_conf
                    
                    # 检查预测是否改变
                    orig_pred = torch.argmax(original_outputs[i]).item()
                    adv_pred = torch.argmax(adversarial_outputs[i]).item()
                    prediction_changed = orig_pred != adv_pred
                    
                    result = RobustnessTestResult(
                        original_prediction=orig_pred,
                        perturbed_prediction=adv_pred,
                        perturbation_strength=epsilon,
                        success_rate=1.0 if prediction_changed else 0.0,
                        confidence_drop=confidence_drop,
                        prediction_changed=prediction_changed,
                        test_type=f"adversarial_{attack_type}",
                        metadata={
                            "original_confidence": orig_conf,
                            "adversarial_confidence": adv_conf,
                            "epsilon": epsilon
                        }
                    )
                    results.append(result)
            
            return adv_inputs, results
            
        except Exception as e:
            logger.error(f"对抗样本生成失败: {e}")
            return inputs, []
    
    def _fgsm_attack(
        self,
        model: Any,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """FGSM攻击"""
        inputs.requires_grad_()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        sign_data_grad = inputs.grad.data.sign()
        perturbed_inputs = inputs + epsilon * sign_data_grad
        
        return torch.clamp(perturbed_inputs, 0, 1)
    
    def _pgd_attack(
        self,
        model: Any,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float,
        alpha: float,
        num_steps: int
    ) -> torch.Tensor:
        """PGD攻击"""
        perturbed_inputs = inputs.clone().detach()
        
        for _ in range(num_steps):
            perturbed_inputs.requires_grad_()
            outputs = model(perturbed_inputs)
            loss = F.cross_entropy(outputs, targets)
            
            model.zero_grad()
            loss.backward()
            
            # 更新扰动
            sign_data_grad = perturbed_inputs.grad.data.sign()
            perturbed_inputs = perturbed_inputs.detach() + alpha * sign_data_grad
            
            # 投影到epsilon球内
            delta = torch.clamp(perturbed_inputs - inputs, min=-epsilon, max=epsilon)
            perturbed_inputs = torch.clamp(inputs + delta, 0, 1).detach()
        
        return perturbed_inputs
    
    def _cw_attack(
        self,
        model: Any,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """C&W攻击（简化版）"""
        # 这里实现简化版的C&W攻击
        # 实际应用中可能需要更复杂的实现
        return self._pgd_attack(model, inputs, targets, epsilon, epsilon/10, 20)
    
    def test_noise_robustness(
        self,
        model: Any,
        inputs: Any,
        noise_types: List[str] = None,
        noise_levels: List[float] = None
    ) -> List[RobustnessTestResult]:
        """测试噪声鲁棒性
        
        Args:
            model: 待测试模型
            inputs: 输入数据
            noise_types: 噪声类型列表
            noise_levels: 噪声强度列表
            
        Returns:
            测试结果列表
        """
        if noise_types is None:
            noise_types = ["gaussian", "uniform", "salt_pepper"]
        
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        results = []
        
        try:
            model.eval()
            
            with torch.no_grad():
                original_outputs = model(inputs)
                
                for noise_type in noise_types:
                    for noise_level in noise_levels:
                        # 添加噪声
                        noisy_inputs = self._add_noise(inputs, noise_type, noise_level)
                        noisy_outputs = model(noisy_inputs)
                        
                        # 计算鲁棒性指标
                        for i in range(len(inputs)):
                            orig_pred = torch.argmax(original_outputs[i]).item()
                            noisy_pred = torch.argmax(noisy_outputs[i]).item()
                            
                            orig_conf = torch.max(F.softmax(original_outputs[i], dim=-1)).item()
                            noisy_conf = torch.max(F.softmax(noisy_outputs[i], dim=-1)).item()
                            
                            result = RobustnessTestResult(
                                original_prediction=orig_pred,
                                perturbed_prediction=noisy_pred,
                                perturbation_strength=noise_level,
                                success_rate=1.0 if orig_pred != noisy_pred else 0.0,
                                confidence_drop=orig_conf - noisy_conf,
                                prediction_changed=orig_pred != noisy_pred,
                                test_type=f"noise_{noise_type}",
                                metadata={
                                    "noise_type": noise_type,
                                    "noise_level": noise_level,
                                    "original_confidence": orig_conf,
                                    "noisy_confidence": noisy_conf
                                }
                            )
                            results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"噪声鲁棒性测试失败: {e}")
            return []
    
    def _add_noise(
        self,
        inputs: torch.Tensor,
        noise_type: str,
        noise_level: float
    ) -> torch.Tensor:
        """添加噪声"""
        if noise_type == "gaussian":
            noise = torch.randn_like(inputs) * noise_level
            return torch.clamp(inputs + noise, 0, 1)
        
        elif noise_type == "uniform":
            noise = (torch.rand_like(inputs) - 0.5) * 2 * noise_level
            return torch.clamp(inputs + noise, 0, 1)
        
        elif noise_type == "salt_pepper":
            mask = torch.rand_like(inputs) < noise_level
            noisy_inputs = inputs.clone()
            noisy_inputs[mask] = torch.rand_like(noisy_inputs[mask])
            return noisy_inputs
        
        else:
            logger.warning(f"未知的噪声类型: {noise_type}")
            return inputs
    
    def test_image_transformations(
        self,
        model: Any,
        images: Any,
        transformations: List[str] = None
    ) -> List[RobustnessTestResult]:
        """测试图像变换鲁棒性
        
        Args:
            model: 待测试模型
            images: 输入图像
            transformations: 变换类型列表
            
        Returns:
            测试结果列表
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL未安装，无法进行图像变换测试")
            return []
        
        if transformations is None:
            transformations = ["blur", "brightness", "contrast", "rotation"]
        
        results = []
        
        try:
            model.eval()
            
            with torch.no_grad():
                original_outputs = model(images)
                
                for transform_type in transformations:
                    # 应用图像变换
                    transformed_images = self._apply_image_transform(images, transform_type)
                    transformed_outputs = model(transformed_images)
                    
                    # 计算鲁棒性指标
                    for i in range(len(images)):
                        orig_pred = torch.argmax(original_outputs[i]).item()
                        trans_pred = torch.argmax(transformed_outputs[i]).item()
                        
                        orig_conf = torch.max(F.softmax(original_outputs[i], dim=-1)).item()
                        trans_conf = torch.max(F.softmax(transformed_outputs[i], dim=-1)).item()
                        
                        result = RobustnessTestResult(
                            original_prediction=orig_pred,
                            perturbed_prediction=trans_pred,
                            perturbation_strength=1.0,  # 变换强度
                            success_rate=1.0 if orig_pred != trans_pred else 0.0,
                            confidence_drop=orig_conf - trans_conf,
                            prediction_changed=orig_pred != trans_pred,
                            test_type=f"transform_{transform_type}",
                            metadata={
                                "transform_type": transform_type,
                                "original_confidence": orig_conf,
                                "transformed_confidence": trans_conf
                            }
                        )
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"图像变换鲁棒性测试失败: {e}")
            return []
    
    def _apply_image_transform(
        self,
        images: torch.Tensor,
        transform_type: str
    ) -> torch.Tensor:
        """应用图像变换"""
        # 这里实现基本的图像变换
        # 实际应用中可能需要更复杂的变换
        if transform_type == "blur":
            # 简单的高斯模糊
            kernel = torch.ones(1, 1, 3, 3) / 9.0
            return F.conv2d(images, kernel, padding=1)
        
        elif transform_type == "brightness":
            # 亮度调整
            factor = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2倍亮度
            return torch.clamp(images * factor, 0, 1)
        
        elif transform_type == "contrast":
            # 对比度调整
            factor = 0.5 + torch.rand(1).item()  # 0.5-1.5倍对比度
            mean = torch.mean(images, dim=(2, 3), keepdim=True)
            return torch.clamp(factor * (images - mean) + mean, 0, 1)
        
        else:
            return images
    
    def calculate_robustness_score(
        self,
        test_results: List[RobustnessTestResult]
    ) -> Dict[str, float]:
        """计算综合鲁棒性分数
        
        Args:
            test_results: 测试结果列表
            
        Returns:
            鲁棒性评分字典
        """
        if not test_results:
            return {"overall_score": 0.0}
        
        # 按测试类型分组
        results_by_type = {}
        for result in test_results:
            test_type = result.test_type
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        scores = {}
        
        # 计算每种测试类型的分数
        for test_type, type_results in results_by_type.items():
            # 计算成功率（预测未改变的比例）
            unchanged_rate = 1.0 - np.mean([r.success_rate for r in type_results])
            
            # 计算平均置信度保持率
            confidence_retention = 1.0 - np.mean([
                max(0, r.confidence_drop) for r in type_results
            ])
            
            # 综合分数
            type_score = 0.6 * unchanged_rate + 0.4 * confidence_retention
            scores[f"{test_type}_score"] = float(type_score)
        
        # 计算总体分数
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        scores["overall_score"] = float(overall_score)
        
        # 添加统计信息
        scores["total_tests"] = len(test_results)
        scores["test_types"] = list(results_by_type.keys())
        
        return scores
    
    def generate_robustness_report(
        self,
        test_results: List[RobustnessTestResult]
    ) -> Dict[str, Any]:
        """生成鲁棒性评估报告
        
        Args:
            test_results: 测试结果列表
            
        Returns:
            详细的鲁棒性报告
        """
        scores = self.calculate_robustness_score(test_results)
        
        # 按测试类型分析
        type_analysis = {}
        results_by_type = {}
        
        for result in test_results:
            test_type = result.test_type
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        for test_type, type_results in results_by_type.items():
            type_analysis[test_type] = {
                "total_samples": len(type_results),
                "attack_success_rate": np.mean([r.success_rate for r in type_results]),
                "avg_confidence_drop": np.mean([r.confidence_drop for r in type_results]),
                "max_confidence_drop": np.max([r.confidence_drop for r in type_results]),
                "min_confidence_drop": np.min([r.confidence_drop for r in type_results])
            }
        
        # 脆弱性分析
        vulnerability_analysis = {
            "most_vulnerable_test": max(
                results_by_type.keys(),
                key=lambda t: type_analysis[t]["attack_success_rate"]
            ) if results_by_type else None,
            "most_robust_test": min(
                results_by_type.keys(),
                key=lambda t: type_analysis[t]["attack_success_rate"]
            ) if results_by_type else None
        }
        
        # 改进建议
        recommendations = self._generate_robustness_recommendations(scores, type_analysis)
        
        return {
            "summary": scores,
            "detailed_analysis": type_analysis,
            "vulnerability_analysis": vulnerability_analysis,
            "recommendations": recommendations,
            "test_metadata": {
                "total_tests": len(test_results),
                "test_types": list(results_by_type.keys()),
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def _generate_robustness_recommendations(
        self,
        scores: Dict[str, float],
        type_analysis: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """生成鲁棒性改进建议"""
        recommendations = []
        
        overall_score = scores.get("overall_score", 0.0)
        
        if overall_score < 0.3:
            recommendations.append("模型鲁棒性较差，建议进行对抗训练")
        elif overall_score < 0.6:
            recommendations.append("模型鲁棒性中等，建议增加数据增强")
        else:
            recommendations.append("模型鲁棒性良好")
        
        # 针对特定攻击类型的建议
        for test_type, analysis in type_analysis.items():
            if analysis["attack_success_rate"] > 0.5:
                if "adversarial" in test_type:
                    recommendations.append(f"对{test_type}攻击脆弱，建议进行对抗训练")
                elif "noise" in test_type:
                    recommendations.append(f"对{test_type}噪声敏感，建议增加噪声数据增强")
                elif "transform" in test_type:
                    recommendations.append(f"对{test_type}变换敏感，建议增加几何变换增强")
        
        return recommendations
