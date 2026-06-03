"""测试高级评估指标功能"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

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
        expected_scores = [
            {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"precision": 0.7, "recall": 0.6, "f1": 0.65},
        ]

        with patch.object(
            self.calculator.semantic_metrics,
            "calculate_bert_score",
            return_value=expected_scores,
        ) as mock_calculate:
            scores = self.calculator.compute_bert_score(
                self.predictions,
                self.references,
            )

        self.assertEqual(scores, expected_scores)
        mock_calculate.assert_called_once_with(self.predictions, self.references)
    
    def test_clip_score_calculation(self):
        """测试CLIP Score计算"""
        images = [Image.new("RGB", (32, 32), color="red") for _ in range(2)]
        expected_scores = [0.3, 0.6]

        with patch.object(
            self.calculator.semantic_metrics,
            "calculate_clip_score",
            return_value=expected_scores,
        ) as mock_calculate:
            scores = self.calculator.compute_clip_score(self.predictions, images)

        self.assertEqual(scores, expected_scores)
        mock_calculate.assert_called_once_with(self.predictions, images)


class TestMultiModalMetricsCalculator(unittest.TestCase):
    """测试多模态指标计算器"""
    
    def setUp(self):
        self.calculator = MultiModalMetricsCalculator()
    
    def test_image_text_matching(self):
        """测试图像文本匹配分数"""
        images = [Image.new("RGB", (32, 32), color="red")]
        texts = ["A red image"]

        with patch.object(
            self.calculator.multimodal_metrics,
            "calculate_image_text_matching",
            return_value=[0.75],
            create=True,
        ) as mock_calculate:
            scores = self.calculator.compute_image_text_matching(images, texts)

        self.assertIsInstance(scores, dict)
        self.assertEqual(scores["matching_score"], 0.75)
        self.assertEqual(scores["individual_scores"], [0.75])
        mock_calculate.assert_called_once_with(texts, images)


class TestRobustnessMetricsCalculator(unittest.TestCase):
    """测试鲁棒性指标计算器"""
    
    def setUp(self):
        self.calculator = RobustnessMetricsCalculator()
        self.inputs = torch.zeros(5, 3, 4, 4)
        self.inputs[:, 0, 0, 0] = torch.tensor([0.49, 0.51, 0.495, 0.52, 0.494])
        self.labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
        self.fixed_noise = torch.full_like(self.inputs, 2.0)

    @staticmethod
    def _deterministic_model(x):
        feature = x[:, 0, 0, 0]
        margin = (feature - 0.5) * 10
        other_logits = torch.full((x.shape[0], 8), -10.0, dtype=x.dtype, device=x.device)
        return torch.cat(
            [
                (-margin).unsqueeze(1),
                margin.unsqueeze(1),
                other_logits,
            ],
            dim=1,
        )
    
    def test_adversarial_robustness(self):
        """测试对抗鲁棒性评估"""
        with patch(
            "florence_forge.evaluation.advanced_metrics.robustness_metrics_calculator.torch.randn_like",
            return_value=self.fixed_noise,
        ):
            scores = self.calculator.evaluate_adversarial_robustness(
                self._deterministic_model,
                self.inputs,
                self.labels,
            )

        self.assertIsInstance(scores, dict)
        self.assertAlmostEqual(scores["clean_accuracy"], 1.0)
        self.assertAlmostEqual(scores["adversarial_accuracy"], 0.4)
        self.assertAlmostEqual(scores["robustness_drop"], 0.6)
    
    def test_noise_robustness(self):
        """测试噪声鲁棒性评估"""
        with patch(
            "florence_forge.evaluation.advanced_metrics.robustness_metrics_calculator.torch.randn_like",
            return_value=self.fixed_noise,
        ):
            scores = self.calculator.evaluate_noise_robustness(
                self._deterministic_model,
                self.inputs,
                self.labels,
            )

        self.assertIsInstance(scores, dict)
        self.assertAlmostEqual(scores["clean_accuracy"], 1.0)
        self.assertAlmostEqual(scores["gaussian_noise_accuracy"], 0.4)
        self.assertAlmostEqual(scores["robustness_drop"], 0.6)
        self.assertEqual(scores["noise_levels"], [0.1, 0.2, 0.3])
        self.assertEqual(len(scores["accuracies_per_noise"]), 3)
        for accuracy in scores["accuracies_per_noise"]:
            self.assertAlmostEqual(accuracy, 0.4)


class TestEfficiencyMetricsCalculator(unittest.TestCase):
    """测试效率指标计算器"""
    
    def setUp(self):
        self.calculator = EfficiencyMetricsCalculator()
    
    def test_inference_speed(self):
        """测试推理速度计算"""
        def mock_model(x):
            return torch.zeros(x.shape[0], 10)

        inputs = [torch.zeros(2, 3, 16, 16)]
        time_values = iter([0.0, 0.1, 0.2, 0.3])

        with patch(
            "florence_forge.evaluation.advanced_metrics.efficiency_metrics_calculator.time.time",
            side_effect=lambda: next(time_values),
        ):
            metrics = self.calculator.measure_inference_speed(
                mock_model,
                inputs,
                num_iterations=2,
            )

        self.assertIsInstance(metrics, dict)
        self.assertIn("avg_inference_time", metrics)
        self.assertIn("inference_speed", metrics)
        self.assertIn("std_inference_time", metrics)
        self.assertAlmostEqual(metrics["avg_inference_time"], 0.1)
        self.assertAlmostEqual(metrics["std_inference_time"], 0.0)
        self.assertAlmostEqual(metrics["inference_speed"], 10.0)
        self.assertEqual(metrics["total_samples"], 2)
    
    def test_memory_usage(self):
        """测试内存使用计算"""
        class MockModel:
            def __call__(self, x):
                return torch.zeros(x.shape[0], 10)
            
            def parameters(self):
                return [torch.zeros(4, 4)]

        fake_process = Mock()
        fake_process.memory_info.side_effect = [
            SimpleNamespace(rss=100 * 1024 * 1024),
            SimpleNamespace(rss=140 * 1024 * 1024),
        ]

        with patch("psutil.Process", return_value=fake_process), patch(
            "gc.collect",
            return_value=None,
        ):
            metrics = self.calculator.measure_memory_usage(
                MockModel(),
                torch.zeros(2, 3, 16, 16),
            )

        self.assertIsInstance(metrics, dict)
        self.assertIn("peak_memory", metrics)
        self.assertIn("memory_efficiency", metrics)
        self.assertAlmostEqual(metrics["initial_memory"], 100.0)
        self.assertAlmostEqual(metrics["peak_memory"], 140.0)
        self.assertAlmostEqual(metrics["memory_used"], 40.0)
        self.assertEqual(metrics["model_params"], 16)
        self.assertGreater(metrics["memory_efficiency"], 0.0)


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
