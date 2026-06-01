"""测试高级评估指标功能"""

import unittest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import tempfile
import os

from florence_forge.evaluation.advanced_metrics import (
    SemanticMetricsCalculator,
    MultiModalMetricsCalculator,
    RobustnessMetricsCalculator,
    EfficiencyMetricsCalculator
)


class TestSemanticMetricsCalculator(unittest.TestCase):
    """测试语义指标计算器"""
    
    def setUp(self):
        self.calculator = SemanticMetricsCalculator()
        self.predictions = ["A cat sitting on a table", "A dog running in the park"]
        self.references = ["A cat is on the table", "A dog is running in a park"]
    
    def test_bert_score_calculation(self):
        """测试BERT Score计算"""
        try:
            scores = self.calculator.compute_bert_score(self.predictions, self.references)
            self.assertIn('precision', scores)
            self.assertIn('recall', scores)
            self.assertIn('f1', scores)
            self.assertTrue(0 <= scores['f1'] <= 1)
        except ImportError:
            self.skipTest("bert_score库未安装")
    
    def test_clip_score_calculation(self):
        """测试CLIP Score计算"""
        try:
            # 创建模拟图像
            images = [Image.new('RGB', (224, 224), color='red') for _ in range(2)]
            scores = self.calculator.compute_clip_score(images, self.predictions)
            self.assertIsInstance(scores, dict)
            self.assertIn('clip_score', scores)
        except ImportError:
            self.skipTest("clip相关库未安装")


class TestMultiModalMetricsCalculator(unittest.TestCase):
    """测试多模态指标计算器"""
    
    def setUp(self):
        self.calculator = MultiModalMetricsCalculator()
    
    def test_image_text_matching(self):
        """测试图像文本匹配分数"""
        try:
            # 创建模拟数据
            images = [Image.new('RGB', (224, 224), color='red')]
            texts = ["A red image"]
            
            scores = self.calculator.compute_image_text_matching(images, texts)
            self.assertIsInstance(scores, dict)
            self.assertIn('matching_score', scores)
        except ImportError:
            self.skipTest("CLIP相关库未安装")


class TestRobustnessMetricsCalculator(unittest.TestCase):
    """测试鲁棒性指标计算器"""
    
    def setUp(self):
        self.calculator = RobustnessMetricsCalculator()
    
    def test_adversarial_robustness(self):
        """测试对抗鲁棒性评估"""
        # 模拟模型函数
        def mock_model(x):
            return torch.rand(x.shape[0], 10)  # 10类分类
        
        # 创建模拟输入
        inputs = torch.randn(5, 3, 224, 224)
        labels = torch.randint(0, 10, (5,))
        
        try:
            scores = self.calculator.evaluate_adversarial_robustness(
                mock_model, inputs, labels
            )
            self.assertIsInstance(scores, dict)
            self.assertIn('clean_accuracy', scores)
            self.assertIn('adversarial_accuracy', scores)
        except ImportError:
            self.skipTest("adversarial相关库未安装")
    
    def test_noise_robustness(self):
        """测试噪声鲁棒性评估"""
        def mock_model(x):
            return torch.rand(x.shape[0], 10)
        
        inputs = torch.randn(5, 3, 224, 224)
        labels = torch.randint(0, 10, (5,))
        
        scores = self.calculator.evaluate_noise_robustness(
            mock_model, inputs, labels
        )
        self.assertIsInstance(scores, dict)
        self.assertIn('clean_accuracy', scores)
        self.assertIn('gaussian_noise_accuracy', scores)


class TestEfficiencyMetricsCalculator(unittest.TestCase):
    """测试效率指标计算器"""
    
    def setUp(self):
        self.calculator = EfficiencyMetricsCalculator()
    
    def test_inference_speed(self):
        """测试推理速度计算"""
        def mock_model(x):
            import time
            time.sleep(0.01)  # 模拟计算时间
            return torch.rand(x.shape[0], 10)
        
        inputs = [torch.randn(10, 3, 224, 224)]
        
        metrics = self.calculator.measure_inference_speed(
            mock_model, inputs, num_iterations=5
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('avg_inference_time', metrics)
        self.assertIn('inference_speed', metrics)
        self.assertIn('std_inference_time', metrics)
        self.assertGreater(metrics['avg_inference_time'], 0)
        self.assertGreater(metrics['inference_speed'], 0)
    
    def test_memory_usage(self):
        """测试内存使用计算"""
        class MockModel:
            def __call__(self, x):
                # 创建一些临时张量来消耗内存
                temp = torch.randn(100, 100, 100)
                return torch.rand(x.shape[0], 10)
            
            def parameters(self):
                return [torch.randn(100, 100)]
        
        mock_model = MockModel()
        inputs = torch.randn(5, 3, 224, 224)
        
        metrics = self.calculator.measure_memory_usage(mock_model, inputs)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('peak_memory', metrics)
        self.assertIn('memory_efficiency', metrics)
        self.assertGreaterEqual(metrics['peak_memory'], 0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_all_calculators_import(self):
        """测试所有计算器都能正确导入"""
        from florence_forge.evaluation import (
            SemanticMetricsCalculator,
            MultiModalMetricsCalculator,
            RobustnessMetricsCalculator,
            EfficiencyMetricsCalculator
        )
        
        # 验证所有类都能实例化
        semantic_calc = SemanticMetricsCalculator()
        multimodal_calc = MultiModalMetricsCalculator()
        robustness_calc = RobustnessMetricsCalculator()
        efficiency_calc = EfficiencyMetricsCalculator()
        
        self.assertIsNotNone(semantic_calc)
        self.assertIsNotNone(multimodal_calc)
        self.assertIsNotNone(robustness_calc)
        self.assertIsNotNone(efficiency_calc)
    
    def test_benchmark_evaluator_with_new_metrics(self):
        """测试BenchmarkEvaluator与新指标的集成"""
        from florence_forge.evaluation import BenchmarkEvaluator
        from unittest.mock import Mock
        
        # 创建模拟模型
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        
        # 创建评估器实例
        evaluator = BenchmarkEvaluator(model=mock_model)
        
        # 验证新的指标计算器已正确初始化
        self.assertIsNotNone(evaluator.semantic_calculator)
        self.assertIsNotNone(evaluator.multimodal_calculator)
        self.assertIsNotNone(evaluator.robustness_calculator)
        self.assertIsNotNone(evaluator.efficiency_calculator)


if __name__ == '__main__':
    unittest.main()