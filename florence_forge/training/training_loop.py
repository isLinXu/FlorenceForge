"""训练循环核心逻辑

由 ``trainer.MultiTaskTrainer`` 组合使用。
"""
import logging
import time
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.config import TrainingConfig
from ..utils.training_logging import (
    format_training_step,
    resolve_total_steps,
    should_log_step,
)

logger = logging.getLogger(__name__)


def supervised_label_count(labels: torch.Tensor) -> int:
    """Count label tokens that participate in the loss."""
    if not isinstance(labels, torch.Tensor):
        return 0
    return int((labels != -100).sum().item())


class TrainingLoop:
    """训练循环管理器
    
    封装训练和验证的核心循环逻辑
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        accelerator=None,
        callback_manager=None
    ):
        """初始化训练循环
        
        Args:
            model: 训练模型
            config: 训练配置
            accelerator: Accelerate 加速器
            callback_manager: 回调管理器
        """
        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.callback_manager = callback_manager
        
        # 训练状态
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        self._train_start_time: Optional[float] = None
        
        # NaN/Inf loss tracking
        self._nan_loss_count = 0
        self._inf_loss_count = 0
        
        # Log hooks
        self._log_hooks: List = []
        
        # MoE adapter (optional)
        self._moe_adapter: Optional[Any] = None
        if getattr(config, "use_moe", False):
            try:
                from ..training.moe import MoETrainingAdapter
                from ..training.moe.moe_config import MoEConfig
                moe_cfg = MoEConfig(
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    aux_loss_weight=config.moe_aux_loss_weight,
                    z_loss_weight=config.moe_z_loss_weight,
                    capacity_factor=config.moe_capacity_factor,
                )
                self._moe_adapter = MoETrainingAdapter(moe_cfg)
                target_pattern = config.moe_target_layers or r"encoder\.layer\.([0-9]+)"
                self._moe_adapter.inject_moe_into_model(model, target_layer_pattern=target_pattern)
                logger.info(f"MoE 已注入模型：{config.moe_num_experts} experts, top_k={config.moe_top_k}")
            except Exception as exc:
                logger.warning(f"MoE 注入失败，将继续不使用 MoE 训练: {exc}")
                self._moe_adapter = None
    
    def train_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        epoch: int,
        gradient_validator=None,
        memory_monitor=None
    ) -> Dict[str, float]:
        """训练一个 epoch
        
        Args:
            train_dataloader: 训练数据加载器
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            epoch: 当前 epoch 编号
            gradient_validator: 梯度验证器（可选）
            memory_monitor: 内存监控器（可选）
        
        Returns:
            训练指标字典
        """
        self.model.train()
        if self._train_start_time is None:
            self._train_start_time = time.perf_counter()
        
        # 指标统计
        epoch_loss = 0.0
        task_losses = defaultdict(float)
        task_samples = defaultdict(int)
        batch_count = 0
        total_steps = resolve_total_steps(
            train_dataloader,
            self.config.num_epochs,
            getattr(self.config, "max_steps", None),
        )
        
        # 进度条
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            disable=not self.accelerator.is_local_main_process if self.accelerator else False
        )
        
        # 触发 epoch 开始回调
        if self.callback_manager:
            self.callback_manager.on_epoch_begin(epoch, mode='train')
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None or (isinstance(batch, dict) and batch.get("is_empty", False)):
                continue
            step_start_time = time.perf_counter()

            # 触发 batch 开始回调
            if self.callback_manager:
                self.callback_manager.on_batch_begin(batch_idx, batch)
            
            # 前向传播
            batch = self._move_batch_to_device(batch)
            model_inputs = self._prepare_model_inputs(batch)
            labels = model_inputs.get("labels")
            if labels is None:
                logger.warning("Batch %s has no labels; skipping training step", batch_idx)
                optimizer.zero_grad()
                continue
            if supervised_label_count(labels) == 0:
                logger.warning(
                    "Batch %s 无有效监督 token（labels 全为 -100）；"
                    "请检查 caption/suffix 是否为空或过短",
                    batch_idx,
                )
                optimizer.zero_grad()
                continue

            accelerator_handles_accumulation = self._accelerator_handles_accumulation()
            skip_optimizer_step = False
            
            with self.accelerator.accumulate(self.model) if self.accelerator else torch.enable_grad():
                outputs = self.model(**model_inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                # 叠加 MoE 辅助损失（如果启用）
                if self._moe_adapter is not None:
                    loss = self._moe_adapter.loss_hook(loss)
                
                if not torch.isfinite(loss).all():
                    if torch.isnan(loss).any():
                        self._nan_loss_count += 1
                    if torch.isinf(loss).any():
                        self._inf_loss_count += 1
                    logger.warning(
                        "Batch %s loss 非有限值 (%s)；已跳过反传。"
                        "MPS 上可关闭 use_fp16；单样本训练请使用更长的 caption suffix",
                        batch_idx,
                        loss.detach().float().mean().item(),
                    )
                    skip_optimizer_step = True
                else:
                    # 原生 PyTorch 路径需要手动缩放；Accelerate 会在 backward 中处理。
                    if not accelerator_handles_accumulation and self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    # 反向传播
                    if self.accelerator:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                
                # 梯度验证（调试用）
                gradient_valid = True
                if not skip_optimizer_step and gradient_validator and batch_idx % 10 == 0:
                    try:
                        if hasattr(gradient_validator, "validate_gradients"):
                            gradient_valid, _ = gradient_validator.validate_gradients(self.global_step)
                        elif hasattr(gradient_validator, "check_gradients"):
                            result = gradient_validator.check_gradients(self.model, self.global_step)
                            if isinstance(result, bool):
                                gradient_valid = result
                        if not gradient_valid:
                            logger.warning("Step %s gradient validation failed; skipping optimizer step", self.global_step)
                    except Exception as exc:
                        logger.error("Gradient validation failed at step %s: %s", self.global_step, exc)
                
                # 梯度裁剪和优化器步进
                should_step = (
                    self.accelerator.sync_gradients
                    if accelerator_handles_accumulation
                    else (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                )
                if skip_optimizer_step or not gradient_valid:
                    if should_step:
                        optimizer.zero_grad()
                elif should_step:
                    if self.config.max_grad_norm > 0:
                        if self.accelerator:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                    
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
            
            if skip_optimizer_step:
                continue
            
            # 统计指标
            actual_loss = (
                loss.item()
                if accelerator_handles_accumulation
                else loss.item() * self.config.gradient_accumulation_steps
            )
            epoch_loss += actual_loss
            batch_count += 1
            
            # 任务级别统计
            task_type, sample_count = self._get_task_type_and_count(batch)
            if task_type is not None:
                task_losses[task_type] += actual_loss
                task_samples[task_type] += sample_count

            # MoE 损失统计（供日志和指标使用）
            moe_aux = 0.0
            moe_z = 0.0
            if self._moe_adapter is not None:
                try:
                    moe_aux = self._moe_adapter.get_auxiliary_loss().item()
                    moe_z = self._moe_adapter.get_router_z_loss().item()
                except Exception:
                    pass

            current_lr = self._get_current_lr(lr_scheduler)
            step_time = time.perf_counter() - step_start_time
            completed_step = self.global_step + 1
            can_log = self.accelerator is None or self.accelerator.is_local_main_process
            if can_log and should_log_step(
                completed_step,
                self.config.logging_steps,
                total_steps,
            ):
                log_metrics = {
                    "loss": actual_loss,
                    "learning_rate": current_lr,
                    "time_per_step": step_time,
                }
                if self._moe_adapter is not None:
                    log_metrics["moe_aux_loss"] = moe_aux
                    log_metrics["moe_z_loss"] = moe_z
                    log_metrics["moe_gini"] = self._moe_adapter.get_routing_gini()
                logger.info(
                    format_training_step(
                        completed_step=completed_step,
                        total_steps=total_steps,
                        epoch=epoch + 1,
                        total_epochs=self.config.num_epochs,
                        metrics=log_metrics,
                        task_type=task_type,
                        elapsed_seconds=time.perf_counter() - self._train_start_time,
                    )
                )
            
            # 更新进度条
            progress_bar_postfix = {
                'loss': f"{actual_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'step_s': f"{step_time:.2f}",
            }
            if self._moe_adapter is not None:
                progress_bar_postfix['aux'] = f"{moe_aux:.4f}"
            progress_bar.set_postfix(progress_bar_postfix)
            
            # 触发 batch 结束回调
            if self.callback_manager:
                step_metrics = {
                    'loss': actual_loss,
                    'learning_rate': current_lr,
                    'global_step': self.global_step
                }
                self.callback_manager.on_batch_end(batch_idx, step_metrics)
            
            self.global_step += 1
            
            # 内存监控
            if memory_monitor and batch_idx % 100 == 0:
                memory_monitor.check_memory(self.global_step)

            # max_steps 硬上限：达到后立即终止当前 epoch
            # （与 v1 trainer.py 行为对齐，max_steps 优先于 num_epochs）
            if self._max_steps_reached():
                logger.info(
                    "🏁 已达到 max_steps=%s，提前结束当前 epoch",
                    self.config.max_steps,
                )
                break
        
        # 计算平均指标
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        # 任务级别平均损失
        task_avg_losses = {
            task: task_losses[task] / task_samples[task]
            for task in task_losses if task_samples[task] > 0
        }
        
        metrics = {
            'loss': avg_loss,
            'learning_rate': self._get_current_lr(lr_scheduler),
            **{f'task_{task}_loss': loss for task, loss in task_avg_losses.items()}
        }
        
        # MoE 路由统计（如果启用）
        if self._moe_adapter is not None:
            try:
                metrics['moe_gini'] = self._moe_adapter.get_routing_gini()
                metrics['moe_overflow_tokens'] = self._moe_adapter.get_total_overflow_tokens()
            except Exception:
                pass
        
        # 触发 epoch 结束回调
        if self.callback_manager:
            self.callback_manager.on_epoch_end(epoch, metrics, mode='train')
        
        return metrics
    
    def validate_epoch(
        self,
        val_dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """验证一个 epoch
        
        Args:
            val_dataloader: 验证数据加载器
            epoch: 当前 epoch 编号
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        # 指标统计
        epoch_loss = 0.0
        task_losses = defaultdict(float)
        task_samples = defaultdict(int)
        batch_count = 0
        
        # 进度条
        progress_bar = tqdm(
            val_dataloader,
            desc=f"Validation Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process if self.accelerator else False
        )
        
        # 触发验证开始回调
        if self.callback_manager:
            self.callback_manager.on_epoch_begin(epoch, mode='eval')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None or (isinstance(batch, dict) and batch.get("is_empty", False)):
                    continue

                batch = self._move_batch_to_device(batch)
                model_inputs = self._prepare_model_inputs(batch)
                if model_inputs.get("labels") is None:
                    logger.warning("Batch %s has no labels; skipping validation step", batch_idx)
                    continue

                autocast_ctx = (
                    self.accelerator.autocast()
                    if self.accelerator is not None
                    else nullcontext()
                )
                with autocast_ctx:
                    outputs = self.model(**model_inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                # 统计指标
                actual_loss = loss.item()
                epoch_loss += actual_loss
                batch_count += 1
                
                # 任务级别统计
                task_type, sample_count = self._get_task_type_and_count(batch)
                if task_type is not None:
                    task_losses[task_type] += actual_loss
                    task_samples[task_type] += sample_count
                
                # 更新进度条
                progress_bar.set_postfix({'val_loss': f"{actual_loss:.4f}"})
        
        # 计算平均指标
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        # 任务级别平均损失
        task_avg_losses = {
            task: task_losses[task] / task_samples[task]
            for task in task_losses if task_samples[task] > 0
        }
        
        metrics = {
            'val_loss': avg_loss,
            **{f'val_task_{task}_loss': loss for task, loss in task_avg_losses.items()}
        }
        
        # 触发验证结束回调
        if self.callback_manager:
            self.callback_manager.on_epoch_end(epoch, metrics, mode='eval')
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将 batch 移动到目标设备
        
        Args:
            batch: 输入 batch
        
        Returns:
            移动后的 batch
        """
        if self.accelerator is not None:
            # accelerate 会自动处理设备转移
            return batch
        
        device = next(self.model.parameters()).device
        moved_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved_batch[k] = v.to(device)
            else:
                moved_batch[k] = v
        return moved_batch

    def _max_steps_reached(self) -> bool:
        """Return True when a positive ``max_steps`` budget has been consumed."""
        max_steps = getattr(self.config, "max_steps", None)
        return bool(max_steps) and max_steps > 0 and self.global_step >= max_steps

    def _accelerator_handles_accumulation(self) -> bool:
        """Return True when using the real Accelerate implementation."""
        if self.accelerator is None:
            return False
        return self.accelerator.__class__.__module__.startswith("accelerate")

    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Strip dataloader metadata before calling the model."""
        allowed_keys = {
            "input_ids",
            "pixel_values",
            "attention_mask",
            "labels",
            "decoder_input_ids",
            "decoder_attention_mask",
            "position_ids",
            "bbox",
            "inputs_embeds",
        }
        return {
            key: value
            for key, value in batch.items()
            if key in allowed_keys and value is not None
        }

    def _get_task_type_and_count(self, batch: Dict[str, Any]) -> Tuple[Optional[str], int]:
        """Return representative task type and batch sample count."""
        task_types = batch.get("task_types")
        if task_types is None:
            task_types = batch.get("task_type")

        if isinstance(task_types, (list, tuple)):
            task_type = task_types[0] if task_types else None
            return task_type, len(task_types)

        batch_size = 1
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 0:
            batch_size = input_ids.shape[0]

        if isinstance(task_types, str):
            return task_types, batch_size
        return None, batch_size

    def _get_current_lr(self, lr_scheduler) -> float:
        """Read the current learning rate from a scheduler if present."""
        if lr_scheduler is None:
            return 0.0
        if hasattr(lr_scheduler, "get_last_lr"):
            values = lr_scheduler.get_last_lr()
            return float(values[0]) if values else 0.0
        return 0.0
    
    def should_early_stop(
        self,
        current_metric: float,
        patience: int
    ) -> bool:
        """判断是否应该早停
        
        Args:
            current_metric: 当前监控指标
            patience: 耐心值
        
        Returns:
            是否应该早停
        """
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"🛑 早停触发：{patience} 个 epoch 无改善")
                return True
            return False

    # ------------------------------------------------------------------
    # Log hooks (v3 enhancement)
    # ------------------------------------------------------------------

    def add_log_hook(self, hook: Callable[[str, Dict], None]) -> None:
        """Register a log hook callable(event_name, data_dict)."""
        if not callable(hook):
            raise TypeError("hook must be callable")
        self._log_hooks.append(hook)

    def _emit_log(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered log hooks, swallowing errors."""
        for hook in self._log_hooks:
            try:
                hook(event, data)
            except Exception as e:
                logger.debug("Log hook raised, suppressed: %s", e)
