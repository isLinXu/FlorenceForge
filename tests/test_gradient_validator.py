"""梯度验证器回归测试。"""

import torch

from florence_forge.training.gradient_validator import (
    GradientValidationConfig,
    GradientValidator,
)


def _model_with_grads():
    model = torch.nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    return model


def test_gradient_validator_stats_history_is_bounded():
    model = _model_with_grads()
    config = GradientValidationConfig(
        max_stats_history=2,
        save_stats=False,
        detect_explosion=False,
        detect_vanishing=False,
    )
    validator = GradientValidator(model, config=config)

    for step in range(5):
        validator.validate_gradients(step)

    assert [stats.step for stats in validator.stats_history] == [3, 4]


def test_gradient_validator_can_disable_stats_history():
    model = _model_with_grads()
    config = GradientValidationConfig(max_stats_history=0, save_stats=False)
    validator = GradientValidator(model, config=config)

    validator.validate_gradients(1)

    assert validator.stats_history == []
