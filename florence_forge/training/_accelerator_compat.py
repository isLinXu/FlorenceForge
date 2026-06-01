#!/usr/bin/env python3
"""
Accelerator compatibility module.

Provides a fallback Accelerator class when the `accelerate` package is not installed.
This avoids duplicating the fallback definition across multiple modules.
"""

try:
    from accelerate import Accelerator
except ImportError:

    class Accelerator:
        """Fallback Accelerator when `accelerate` is not installed.

        Provides no-op implementations of the core Accelerator interface
        so that training code can run without distributed/mixed-precision support.
        """

        def __init__(self, *args, **kwargs):
            self._gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)

        def prepare(self, *args):
            """准备模型/优化器/数据加载器（无分布式时直接返回原对象）"""
            if len(args) == 1:
                return args[0]
            return args

        def backward(self, loss):
            loss.backward()

        def step(self, optimizer):
            optimizer.step()

        def zero_grad(self, optimizer):
            optimizer.zero_grad()

        def wait_for_everyone(self):
            pass

        def save_state(self, *args, **kwargs):
            pass

        def load_state(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            print(*args, **kwargs)

        def log(self, *args, **kwargs):
            pass

        def end_training(self):
            pass

        @property
        def is_main_process(self):
            return True

        @property
        def is_local_main_process(self):
            return True

        @property
        def device(self):
            import torch
            return torch.device("cpu")

        @property
        def sync_gradients(self):
            return False

        def unwrap_model(self, model):
            return model

        def accumulate(self, model):
            """梯度累积上下文管理器

            在无分布式环境下，仅当步数对齐时才执行优化器步骤。
            """
            import contextlib
            return contextlib.nullcontext()

        def clip_grad_norm_(self, parameters, max_norm, **kwargs):
            """梯度裁剪，返回梯度范数"""
            import torch
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm, **kwargs)

        def save_model(self, model, output_dir, **kwargs):
            """保存模型"""
            import os
            os.makedirs(output_dir, exist_ok=True)
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(output_dir)
            else:
                import torch
                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
