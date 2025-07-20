#!/usr/bin/env python3
"""
Florence Forge - 多任务训练器

提供完整的多任务训练功能，包括训练循环、评估、检查点管理等
"""

import os
import csv
import time
import json
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
except ImportError:
    # 如果accelerate不可用，定义占位符
    class Accelerator:
        def __init__(self, *args, **kwargs):
            pass
        def prepare(self, *args):
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

from ..core.model import Florence2MultiTaskModel
from ..core.config import TrainingConfig
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader
from .scheduler import TaskScheduler
from .lora_manager import LoRAManager
from .model_merger import ModelMerger
from .visualizer import TrainingVisualizer
from .monitoring import TrainingMonitor

logger = logging.getLogger(__name__)

class MultiTaskTrainer:
    """多任务训练器
    
    提供完整的多任务训练功能
    """
    
    def __init__(
        self,
        model: Florence2MultiTaskModel,
        train_dataset: MultiTaskDataset,
        val_dataset: Optional[MultiTaskDataset] = None,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None
    ):
        """初始化训练器
        
        Args:
            model: Florence2多任务模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 训练配置
            accelerator: Accelerate加速器
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # 初始化加速器
        if accelerator is None:
            # 自动检测设备并配置加速器
            import torch
            
            # 设备检测和配置
            self._setup_device()
            
            # 确定混合精度设置
            # mixed_precision = "no"
            # if torch.cuda.is_available() and self.config.device != "cpu":
            #     if self.config.use_bf16:
            #         mixed_precision = "bf16"
            #     elif self.config.use_fp16:
            #         mixed_precision = "fp16"

             # 确定混合精度设置（添加硬件和版本检查）
            mixed_precision = "no"
            if torch.cuda.is_available() and self.config.device != "cpu":
                # 检查PyTorch版本是否支持BF16
                pt_version_ok = (torch.__version__ >= "1.10")
                
                # 检查硬件是否支持BF16
                try:
                    hw_support_bf16 = torch.cuda.is_bf16_supported()
                except AttributeError:  # 旧版本PyTorch没有此方法
                    hw_support_bf16 = False
                
                # 只有当版本和硬件都支持时才使用BF16
                if self.config.use_bf16 and pt_version_ok and hw_support_bf16:
                    mixed_precision = "bf16"
                    print("✅ 使用BF16混合精度加速训练")
                elif self.config.use_bf16:  # 请求BF16但不满足条件
                    print("⚠️ BF16不可用，原因: "
                          f"PyTorch版本要求(>=1.10): {torch.__version__}, "
                          f"硬件支持: {hw_support_bf16}")
                    # 自动降级到FP16或禁用
                    if self.config.use_fp16:
                        mixed_precision = "fp16"
                        print("✅ 回退到FP16混合精度")
                    else:
                        print("⚠️ 禁用混合精度")
                elif self.config.use_fp16:  # 使用FP16
                    mixed_precision = "fp16"
            
            self.accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                log_with="tensorboard" if self.config.logging_dir else None,
                project_dir=self.config.logging_dir
            )
        else:
            self.accelerator = accelerator
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf') if not self.config.greater_is_better else float('-inf')
        self.patience_counter = 0
        
        # 组件初始化
        self.task_scheduler = None
        self.lora_manager = None
        self.model_merger = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 指标记录
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.step_metrics = []
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化可视化器
        self.visualizer = TrainingVisualizer(str(self.output_dir))
        
        # 初始化监控器
        self.monitor = None
        if hasattr(self.config, 'monitoring_config') and self.config.monitoring_config:
            self.monitor = TrainingMonitor(
                config=self.config.monitoring_config,
                output_dir=str(self.output_dir)
            )
        
        logger.info("多任务训练器初始化完成")
    
    def _setup_device(self) -> None:
        """设置和检测设备"""
        import torch
        
        if self.config.device == "auto":
            # 自动检测最佳设备
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"检测到 {device_count} 个CUDA设备")
                
                # 选择显存最大的GPU
                best_device = 0
                max_memory = 0
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory = props.total_memory / (1024**3)  # GB
                    logger.info(f"GPU {i}: {props.name}, 显存: {memory:.1f}GB")
                    if memory > max_memory:
                        max_memory = memory
                        best_device = i
                
                self.config.device = f"cuda:{best_device}"
                logger.info(f"自动选择设备: {self.config.device} (显存: {max_memory:.1f}GB)")
            elif torch.backends.mps.is_available():
                self.config.device = "mps"
                logger.info("自动选择设备: mps (Apple Silicon GPU)")
            else:
                self.config.device = "cpu"
                logger.info("自动选择设备: cpu")
        else:
            # 验证指定的设备是否可用
            if self.config.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA不可用，回退到CPU")
                    self.config.device = "cpu"
                else:
                    device_id = int(self.config.device.split(":")[1]) if ":" in self.config.device else 0
                    if device_id >= torch.cuda.device_count():
                        logger.warning(f"GPU {device_id} 不存在，使用GPU 0")
                        self.config.device = "cuda:0"
            elif self.config.device == "mps":
                if not torch.backends.mps.is_available():
                    logger.warning(f"MPS不可用，回退到CPU")
                    self.config.device = "cpu"
            
            logger.info(f"使用指定设备: {self.config.device}")
        
        # 设置PyTorch默认设备
        torch.cuda.set_device(self.config.device) if self.config.device.startswith("cuda") else None
    
    def setup_training(self) -> None:
        """设置训练组件"""
        logger.info("正在设置训练组件...")
        
        # 设置任务调度器
        task_types = list(self.train_dataset.task_indices.keys())
        self.task_scheduler = TaskScheduler(
            task_types=task_types,
            config=self.config.task_scheduling_config
        )
        
        # 设置LoRA管理器和模型合并器
        if self.config.model_config.use_lora:
            self.lora_manager = LoRAManager(self.config.model_config.lora_config)
            self.model_merger = ModelMerger(self.lora_manager)
            
            # 为每个任务创建LoRA配置
            for task_type in task_types:
                self.lora_manager.create_task_config(task_type)
        
        # 设置数据加载器
        self._setup_dataloaders()
        
        # 设置优化器和调度器
        self._setup_optimizer()
        self._setup_lr_scheduler()
        
        # 使用accelerator准备组件（暂时跳过数据加载器）
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        
        # 暂时不使用accelerate准备数据加载器，直接使用原始的
        # self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        
        if self.val_dataloader is not None:
            # self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
            pass
        
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        # 初始化CSV日志记录器
        self._init_csv_logger()
        
        logger.info("训练组件设置完成")
    
    def _setup_dataloaders(self) -> None:
        """设置数据加载器"""
        # 训练数据加载器
        train_loader = TaskDataLoader(
            dataset=self.train_dataset,
            config=self.config.data_config,
            sampling_strategy=self.config.task_scheduling_config.strategy
        )
        self.train_dataloader = train_loader.get_dataloader()
        
        # 验证数据加载器
        if self.val_dataset is not None:
            # 创建验证配置的副本，避免修改原始配置
            val_config = deepcopy(self.config.data_config)
            val_config.shuffle = False  # 验证时不打乱
            val_config.drop_last = False  # 验证时保留所有数据
            
            val_loader = TaskDataLoader(
                dataset=self.val_dataset,
                config=val_config,
                sampling_strategy="random"
            )
            self.val_dataloader = val_loader.get_dataloader()
            logger.info(f"验证数据加载器已创建，批次数: {len(self.val_dataloader)}")
    
    def _setup_optimizer(self) -> None:
        """设置优化器"""
        opt_config = self.config.optimization_config
        
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=(opt_config.adam_beta1, opt_config.adam_beta2),
            eps=opt_config.adam_epsilon
        )
        
        logger.info(f"优化器设置完成，学习率: {opt_config.learning_rate}")
    
    def _setup_lr_scheduler(self) -> None:
        """设置学习率调度器"""
        opt_config = self.config.optimization_config
        
        # 计算总训练步数
        if self.config.max_steps:
            num_training_steps = self.config.max_steps
        else:
            num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        
        # 计算预热步数
        if opt_config.warmup_steps:
            num_warmup_steps = opt_config.warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * opt_config.warmup_ratio)
        
        # 创建调度器
        if opt_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif opt_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.lr_scheduler = get_scheduler(
                opt_config.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        logger.info(
            f"学习率调度器设置完成，类型: {opt_config.lr_scheduler_type}, "
            f"预热步数: {num_warmup_steps}, 总步数: {num_training_steps}"
        )
    
    def _init_csv_logger(self) -> None:
        """初始化CSV日志记录器"""
        self.step_csv_path = self.output_dir / "step_metrics.csv"
        self.epoch_csv_path = self.output_dir / "epoch_metrics.csv"
        
        # 初始化步骤指标CSV
        with open(self.step_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'task_type', 'loss', 'learning_rate', 
                'grad_norm', 'time_per_step'
            ])
        
        # 初始化轮次指标CSV
        with open(self.epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'learning_rate',
                'best_metric', 'patience_counter'
            ])
    
    def train(self) -> Dict[str, Any]:
        """开始训练
        
        Returns:
            训练结果字典
        """
        logger.info("开始多任务训练...")
        
        # 设置训练组件
        self.setup_training()
        
        # 保存配置
        self._save_config()
        
        # 记录模型架构到监控器
        if self.monitor:
            self.monitor.log_model_architecture(self.model)
        
        # 训练循环
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_metrics = self._train_epoch()
                
                # 验证
                val_metrics = None
                if self.val_dataset is not None:
                    # 每个epoch都进行验证，或者按照eval_steps间隔
                    if self.config.eval_steps <= 1 or (epoch + 1) % self.config.eval_steps == 0:
                        val_metrics = self._validate_epoch()
                        logger.info(f"Epoch {epoch + 1}: 验证完成，验证样本数: {len(self.val_dataloader.dataset)}")
                
                # 记录epoch指标
                self._record_epoch_metrics(train_metrics, val_metrics)
                
                # 保存检查点
                if (epoch + 1) % self.config.save_steps == 0:
                    self._save_checkpoint(epoch)
                
                # 早停检查
                if self._should_early_stop(val_metrics):
                    logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                # 更新任务权重
                if self.task_scheduler.should_update_weights():
                    self.task_scheduler.auto_adjust_weights()
        
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            raise
        
        finally:
            # 保存最终模型
            self._save_final_model()
            
            # 生成可视化报告
            try:
                logger.info("正在生成训练可视化报告...")
                report_path = self.visualizer.generate_training_report()
                if report_path:
                    logger.info(f"训练报告已生成: {report_path}")
                else:
                    logger.warning("训练报告生成失败")
            except Exception as e:
                logger.error(f"生成可视化报告时出错: {e}")
            
            # 关闭监控器
            if self.monitor:
                self.monitor.finish()
            
            # 计算训练时间
            total_time = time.time() - start_time
            logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        
        return self._get_training_summary()
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # 跳过空批次
            if batch is None or batch.get("is_empty", False):
                continue
                
            # 处理单个样本或批次
            if isinstance(batch, list):
                # 处理多个样本的批次
                sample = batch[0]  # 暂时只处理第一个样本
            else:
                # 处理单个样本
                sample = batch
                
            # 从批次数据中获取实际任务类型
            if isinstance(batch, dict) and 'task_type' in batch:
                task_type = batch['task_type']
            elif hasattr(sample, 'task_type'):
                task_type = sample.task_type
            else:
                # 作为备选方案才使用调度器
                task_type = self.task_scheduler.select_task(self.current_epoch)
                logger.warning(f"无法从数据中获取任务类型，使用调度器选择: {task_type}")
            
            # 切换LoRA适配器（如果使用）
            if self.lora_manager and hasattr(self.model, 'set_adapter'):
                self.lora_manager.switch_adapter(self.model, task_type)
            
            # 确保数据在正确的设备上
            device = next(self.model.parameters()).device
            input_ids = sample["input_ids"].to(device)
            pixel_values = sample["pixel_values"].to(device)
            attention_mask = sample.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 前向传播
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=input_ids  # 对于生成任务，labels通常是input_ids
                )
                
                loss = outputs.loss
                
                # 反向传播
                self.accelerator.backward(loss)
                
                # 梯度裁剪
                if self.config.optimization_config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimization_config.max_grad_norm
                    )
                
                # 优化器步骤
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # 记录指标
            step_time = time.time() - step_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算梯度范数
            grad_norm = 0.0
            if self.config.optimization_config.max_grad_norm > 0:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** (1. / 2)
            
            # 更新任务性能
            self.task_scheduler.update_task_performance(task_type, loss.item())
            
            # 记录步骤指标 - 修改逻辑确保小数据集也能记录足够的指标
            should_log = (
                self.global_step % self.config.logging_steps == 0 or  # 按原有间隔记录
                self.global_step == 1 or  # 记录第一步
                (self.global_step % max(1, len(self.train_dataloader) // 3) == 0)  # 每个epoch至少记录3次
            )
            if should_log:
                self._record_step_metrics(
                    step=self.global_step,
                    epoch=self.current_epoch,
                    task_type=task_type,
                    loss=loss.item(),
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    time_per_step=step_time
                )
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'task': task_type
            })
            
            # 收集epoch指标
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['learning_rate'].append(current_lr)
            
            self.global_step += 1
            
            # 检查是否达到最大步数
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # 计算epoch平均指标
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        return avg_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_local_main_process
            ):
                # 确保数据在正确的设备上
                device = next(self.model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                val_metrics['loss'].append(loss.item())
        
        # 计算平均指标
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        
        return avg_metrics
    
    def _record_step_metrics(
        self,
        step: int,
        epoch: int,
        task_type: str,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        time_per_step: float
    ) -> None:
        """记录步骤指标"""
        # 记录到CSV
        with open(self.step_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, epoch, task_type, loss, learning_rate,
                grad_norm, time_per_step
            ])
        
        # 记录到内存
        self.step_metrics.append({
            'step': step,
            'epoch': epoch,
            'task_type': task_type,
            'loss': loss,
            'learning_rate': learning_rate,
            'grad_norm': grad_norm,
            'time_per_step': time_per_step
        })
        
        # 记录到accelerator（如果配置了）
        if self.accelerator.is_local_main_process:
            self.accelerator.log({
                'train/loss': loss,
                'train/learning_rate': learning_rate,
                'train/grad_norm': grad_norm,
                'train/time_per_step': time_per_step
            }, step=step)
        
        # 记录到监控器（WandB, SwanLab, TensorBoard）
        if self.monitor:
            metrics = {
                'loss': loss,
                'learning_rate': learning_rate,
                'grad_norm': grad_norm,
                'time_per_step': time_per_step,
                f'task_{task_type}_loss': loss
            }
            self.monitor.log_metrics(metrics, step, prefix='train')
            
            # 记录梯度信息（如果启用）
            if self.monitor.config.log_gradients:
                self.monitor.log_gradients(self.model, step)
    
    def _record_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ) -> None:
        """记录epoch指标"""
        train_loss = train_metrics.get('loss', 0.0)
        val_loss = val_metrics.get('loss') if val_metrics else None
        current_lr = train_metrics.get('learning_rate', 0.0)
        
        # 记录到CSV
        with open(self.epoch_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch, train_loss, val_loss if val_loss is not None else '', current_lr,
                self.best_metric, self.patience_counter
            ])
        
        # 记录到内存
        self.train_metrics['loss'].append(train_loss)
        if val_metrics and val_loss is not None:
            self.val_metrics['loss'].append(val_loss)
        
        # 记录到accelerator
        if self.accelerator.is_local_main_process:
            log_dict = {
                'epoch/train_loss': train_loss,
                'epoch/learning_rate': current_lr
            }
            if val_metrics and val_loss is not None:
                log_dict['epoch/val_loss'] = val_loss
            
            self.accelerator.log(log_dict, step=self.current_epoch)
        
        # 记录到监控器（WandB, SwanLab, TensorBoard）
        if self.monitor:
            epoch_metrics = {
                'train_loss': train_loss,
                'learning_rate': current_lr
            }
            if val_metrics and val_loss is not None:
                epoch_metrics['val_loss'] = val_loss
            
            self.monitor.log_metrics(epoch_metrics, self.current_epoch, prefix='epoch')
        
        # 构建日志信息
        log_msg = (
            f"Epoch {self.current_epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
        )
        if val_loss is not None:
            log_msg += f"val_loss={val_loss:.4f}, "
        else:
            log_msg += "val_loss=N/A, "
        log_msg += f"lr={current_lr:.2e}"
        
        logger.info(log_msg)
    
    def _should_early_stop(self, val_metrics: Optional[Dict[str, float]]) -> bool:
        """检查是否应该早停
        
        Args:
            val_metrics: 验证指标
            
        Returns:
            是否应该早停
        """
        if val_metrics is None or self.config.early_stopping_patience <= 0:
            return False
        
        current_metric = val_metrics.get(self.config.metric_for_best_model.replace('eval_', ''), 0.0)
        
        # 检查是否有改进
        improved = False
        if self.config.greater_is_better:
            if current_metric > self.best_metric + self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        else:
            if current_metric < self.best_metric - self.config.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        
        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int) -> None:
        """保存检查点"""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.accelerator.save_model(self.model, checkpoint_dir)
        
        # 保存训练状态
        training_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'task_scheduler_state': self.task_scheduler.save_state() if self.task_scheduler else None
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # 保存LoRA管理器状态
        if self.lora_manager:
            self.lora_manager.save_manager_state(checkpoint_dir / "lora_manager_state.json")
        
        logger.info(f"检查点已保存到: {checkpoint_dir}")
        
        # 清理旧检查点
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点"""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-epoch-"):
                checkpoint_dirs.append(path)
        
        # 按创建时间排序
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime)
        
        # 删除多余的检查点
        while len(checkpoint_dirs) > self.config.save_total_limit:
            old_checkpoint = checkpoint_dirs.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"已删除旧检查点: {old_checkpoint}")
    
    def _save_final_model(self) -> None:
        """保存最终模型"""
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.accelerator.save_model(self.model, final_model_dir)
        
        # 保存LoRA适配器
        if self.lora_manager:
            self.lora_manager.save_adapter(self.model, final_model_dir / "lora_adapters")
            
            # 可选：保存合并后的模型
            if hasattr(self.config, 'save_merged_model') and self.config.save_merged_model:
                merged_model_dir = final_model_dir / "merged_model"
                self.save_merged_model(merged_model_dir)
        
        logger.info(f"最终模型已保存到: {final_model_dir}")
    
    def _save_config(self) -> None:
        """保存训练配置"""
        config_path = self.output_dir / "training_config.json"
        self.config.save_to_file(config_path)
        
        # 保存数据集统计信息
        dataset_stats = {
            'train_dataset': self.train_dataset.get_task_statistics(),
            'val_dataset': self.val_dataset.get_task_statistics() if self.val_dataset else None
        }
        
        with open(self.output_dir / "dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要
        
        Returns:
            训练摘要字典
        """
        return {
            'total_epochs': self.current_epoch + 1,
            'epochs_completed': self.current_epoch + 1,  # 添加cli.py期望的字段
            'total_steps': self.global_step,
            'best_metric': self.best_metric,
            'final_loss': self.train_metrics['loss'][-1] if self.train_metrics['loss'] else 0.0,  # 添加cli.py期望的字段
            'final_train_loss': self.train_metrics['loss'][-1] if self.train_metrics['loss'] else 0.0,
            'final_val_loss': self.val_metrics['loss'][-1] if self.val_metrics['loss'] else 0.0,
            'task_scheduler_stats': self.task_scheduler.get_statistics() if self.task_scheduler else None,
            'output_dir': str(self.output_dir)
        }
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint_path = Path(checkpoint_path)
        
        # 加载训练状态
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location='cpu')
            
            self.current_epoch = training_state['epoch']
            self.global_step = training_state['global_step']
            self.best_metric = training_state['best_metric']
            self.patience_counter = training_state['patience_counter']
            
            if self.optimizer and 'optimizer_state_dict' in training_state:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            if self.lr_scheduler and training_state.get('lr_scheduler_state_dict'):
                self.lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])
            
            if self.task_scheduler and training_state.get('task_scheduler_state'):
                self.task_scheduler.load_state(training_state['task_scheduler_state'])
        
        # 加载LoRA管理器状态
        lora_state_path = checkpoint_path / "lora_manager_state.json"
        if self.lora_manager and lora_state_path.exists():
            self.lora_manager.load_manager_state(lora_state_path)
        
        logger.info(f"检查点已加载: {checkpoint_path}")
    
    def save_merged_model(self, output_dir: Union[str, Path]) -> None:
        """保存合并后的模型
        
        Args:
            output_dir: 输出目录
        """
        if not self.model_merger:
            logger.warning("模型合并器未初始化，无法保存合并模型")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 合并并保存模型
        merged_model = self.model_merger.merge_all_adapters(self.model)
        self.accelerator.save_model(merged_model, output_dir)
        
        logger.info(f"合并后的模型已保存到: {output_dir}")