"""Integration callbacks that bridge training to external subsystems:
MoE routing diagnostics and the unified monitoring stack (WandB/SwanLab/TensorBoard).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import TrainerCallback

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid cycles
    from ...training.trainer import MultiTaskTrainer


class MoECallback(TrainerCallback):
    """MoE 诊断回调：自动监控和报告专家利用率、Gini 系数、溢出 token 数

    当训练器启用 MoE 时，此回调在每个 epoch 结束时收集并记录：
    - 专家利用率分布（Gini 系数，越接近 0 表示负载越均衡）
    - 容量溢出 token 总数
    - 逐层路由统计摘要（平均门控权重、token 分配比例）

    使用方式：
        from florence_forge.core.callbacks import MoECallback
        callback = MoECallback(log_frequency=1)
        trainer.callbacks.append(callback)
    """

    def __init__(self, log_frequency: int = 1):
        """初始化 MoE 诊断回调

        Args:
            log_frequency: 每多少个 epoch 记录一次详细逐层路由统计
        """
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__ + ".MoECallback")

    def _get_moe_adapter(self, trainer: "MultiTaskTrainer") -> Optional[Any]:
        """从 trainer 获取 MoETrainingAdapter 实例。"""
        training_loop = getattr(trainer, "training_loop", None)
        if training_loop is not None:
            adapter = getattr(training_loop, "_moe_adapter", None)
            if adapter is not None:
                return adapter
        # 回退：检查 trainer 是否有直接挂载的 moe_adapter
        return getattr(trainer, "moe_adapter", None)

    def on_epoch_end(
        self,
        trainer: "MultiTaskTrainer",
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        moe_adapter = self._get_moe_adapter(trainer)
        if moe_adapter is None or not moe_adapter.is_injected():
            return

        try:
            routing_summary = moe_adapter.summarize_routing()
            gini = moe_adapter.get_routing_gini()
            overflow_tokens = moe_adapter.get_total_overflow_tokens()

            self.logger.info(
                "[MoE] Epoch %d: Gini=%.4f, OverflowTokens=%d, "
                "Layers=%d, Experts=%d, TopK=%d",
                epoch + 1,
                gini,
                overflow_tokens,
                routing_summary.get("num_moe_layers", 0),
                routing_summary.get("num_experts", 0),
                routing_summary.get("top_k", 0),
            )

            # 将 MoE 指标注入 logs，供后续回调（TensorBoard/WandB）消费
            if logs is not None:
                logs["moe_gini"] = gini
                logs["moe_overflow_tokens"] = overflow_tokens
                logs["moe_num_layers"] = routing_summary.get("num_moe_layers", 0)
                logs["moe_num_experts"] = routing_summary.get("num_experts", 0)

            # 按频率输出详细逐层统计
            if (epoch + 1) % self.log_frequency == 0:
                for layer_info in routing_summary.get("layers", []):
                    layer_idx = layer_info.get("layer_index", 0)
                    avg_weights = layer_info.get("avg_gate_weights", [])
                    token_dist = layer_info.get("token_distribution", [])
                    overflow = layer_info.get("overflow_tokens")
                    self.logger.info(
                        "[MoE] Layer %d: avg_weights=%s, token_dist=%s, overflow=%s",
                        layer_idx,
                        avg_weights,
                        token_dist,
                        overflow,
                    )
        except Exception as e:
            self.logger.warning("MoE 回调统计收集失败: %s", e)

    def on_train_end(
        self,
        trainer: "MultiTaskTrainer",
        config: Any,
    ) -> None:
        moe_adapter = self._get_moe_adapter(trainer)
        if moe_adapter is None or not moe_adapter.is_injected():
            return
        try:
            final_summary = moe_adapter.summarize_routing()
            gini = moe_adapter.get_routing_gini()
            overflow_tokens = moe_adapter.get_total_overflow_tokens()
            self.logger.info(
                "[MoE] 训练结束: 最终 Gini=%.4f, 总溢出 token=%d, 层数=%d",
                gini,
                overflow_tokens,
                final_summary.get("num_moe_layers", 0),
            )
        except Exception as e:
            self.logger.warning("MoE 训练结束统计收集失败: %s", e)


class MonitoringCallback(TrainerCallback):
    """统一监控回调——将 TrainingMonitor 集成到 Callback 体系

    此回调消除了 Callback 系统和 TrainingMonitor 之间的并行问题。
    所有监控操作（WandB、SwanLab、TensorBoard）都通过此回调统一管理，
    训练器无需直接调用 TrainingMonitor。

    使用方式：
        from florence_forge.training.monitoring import MonitoringConfig, TrainingMonitor

        monitor_config = MonitoringConfig(enable_wandb=True, wandb_project="my-project")
        monitor = TrainingMonitor(monitor_config, output_dir="./output")
        callback = MonitoringCallback(monitor, log_frequency=10)
        trainer.callbacks.append(callback)
    """

    def __init__(self, monitor=None, log_frequency: int = 10, log_gradients: bool = False):
        """初始化监控回调

        Args:
            monitor: TrainingMonitor 实例。如果为 None，则创建一个默认的
                     TensorBoard-only 监控器。
            log_frequency: 每多少步记录一次指标
            log_gradients: 是否记录梯度信息
        """
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        self.logger = logging.getLogger(__name__ + ".MonitoringCallback")

        # 延迟初始化 monitor（避免在 import 时触发依赖检查）
        self._monitor = monitor
        self._owns_monitor = False

    def _ensure_monitor(self, output_dir: Optional[str] = None):
        """确保 monitor 已初始化"""
        if self._monitor is not None:
            return
        try:
            from ...training.monitoring import MonitoringConfig, TrainingMonitor
            config = MonitoringConfig(enable_tensorboard=True)
            self._monitor = TrainingMonitor(config, output_dir=output_dir or "./outputs")
            self._owns_monitor = True
        except Exception as e:
            self.logger.warning(f"MonitoringCallback: 无法创建默认监控器: {e}")

    def on_train_begin(self, trainer, config):
        output_dir = getattr(config, 'output_dir', './outputs')
        self._ensure_monitor(output_dir)
        if self._monitor is not None and hasattr(self._monitor, 'config'):
            if getattr(self._monitor.config, 'log_model_architecture', False):
                try:
                    self._monitor.log_model_architecture(trainer.model)
                except Exception as e:
                    self.logger.warning(f"记录模型架构失败: {e}")

    def on_step_end(self, trainer, step, logs=None):
        if self._monitor is None or step % self.log_frequency != 0:
            return
        if logs:
            try:
                self._monitor.log_metrics(logs, step, prefix="train")
            except Exception as e:
                self.logger.warning(f"记录训练指标失败: {e}")

        # 记录梯度
        if self.log_gradients and hasattr(trainer, 'model'):
            try:
                self._monitor.log_gradients(trainer.model, step)
            except Exception as e:
                self.logger.warning(f"记录梯度失败: {e}")

    def on_eval_end(self, trainer, logs=None):
        if self._monitor is None or not logs:
            return
        step = getattr(trainer, 'global_step', 0)
        try:
            self._monitor.log_metrics(logs, step, prefix="eval")
        except Exception as e:
            self.logger.warning(f"记录评估指标失败: {e}")

    def on_epoch_end(self, trainer, epoch, logs=None):
        """记录 epoch 级指标到监控器"""
        if self._monitor is None or not logs:
            return
        try:
            # 提取训练和验证指标
            train_metrics = logs.get("train_metrics", {})
            val_metrics = logs.get("val_metrics", {})
            epoch_metrics = {}
            if train_metrics:
                epoch_metrics.update({f"epoch/{k}": v for k, v in train_metrics.items()})
            if val_metrics:
                epoch_metrics.update({f"epoch/{k}": v for k, v in val_metrics.items()})
            if epoch_metrics:
                self._monitor.log_metrics(epoch_metrics, epoch, prefix="epoch")
        except Exception as e:
            self.logger.warning(f"记录 epoch 指标失败: {e}")

    def on_train_end(self, trainer, config):
        if self._monitor is not None:
            try:
                self._monitor.finish()
            except Exception as e:
                self.logger.warning(f"结束监控失败: {e}")
