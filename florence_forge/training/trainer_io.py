#!/usr/bin/env python3
"""Checkpoint and model IO helpers for the v1 multi-task trainer."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch

from ..utils.torch_serialization import safe_torch_load_cpu

logger = logging.getLogger(__name__)


class TrainerIOMixin:
    """Persistence helpers mixed into ``MultiTaskTrainer``.

    The methods are intentionally behavior-preserving extracts from
    ``trainer.py`` so checkpointing can evolve separately from the train loop.
    """

    def _save_checkpoint(self, epoch: int) -> None:
        """保存检查点（支持异步保存，不阻塞训练循环）

        默认使用异步保存（在后台线程中执行），可通过设置
        config.async_checkpoint = False 禁用。
        """
        # 等待上一个检查点保存完成（避免并发保存冲突）
        if self._last_checkpoint_future is not None:
            try:
                self._last_checkpoint_future.result(timeout=300)  # 最多等待5分钟
            except Exception as e:
                logger.warning(f"等待上一个检查点保存完成时出错: {e}")

        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch + 1}"

        # 决定是否异步保存
        use_async = getattr(self.config, 'async_checkpoint', True)

        if use_async:
            # 异步保存：在后台线程中执行
            self._last_checkpoint_future = self._checkpoint_executor.submit(
                self._do_save_checkpoint, epoch, checkpoint_dir
            )
            logger.info(f"检查点保存已提交到后台线程: {checkpoint_dir}")
        else:
            # 同步保存
            self._do_save_checkpoint(epoch, checkpoint_dir)

    def _do_save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> None:
        """实际执行检查点保存（在同步或异步上下文中调用）"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        try:
            self.accelerator.save_model(self.model, checkpoint_dir)
        except Exception as e:
            logger.error(f"保存模型到检查点失败: {e}")

        # 保存训练状态（优化器状态可选，大模型可关闭以减小检查点体积）
        try:
            training_state = {
                'epoch': epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'patience_counter': self.patience_counter,
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'task_scheduler_state': self.task_scheduler.save_state() if self.task_scheduler else None
            }
            # 仅在未设置 skip_optimizer_state 或显式要求时保存优化器状态
            if not getattr(self.config, 'skip_optimizer_state', False):
                training_state['optimizer_state_dict'] = self.optimizer.state_dict()
            torch.save(training_state, checkpoint_dir / "training_state.pt")
        except Exception as e:
            logger.error(f"保存训练状态失败: {e}")

        # 保存LoRA管理器状态
        if self.lora_manager:
            try:
                self.lora_manager.save_manager_state(checkpoint_dir / "lora_manager_state.json")
            except Exception as e:
                logger.error(f"保存LoRA管理器状态失败: {e}")

        logger.info(f"检查点已保存到: {checkpoint_dir}")

        # 清理旧检查点
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点（带错误隔离）"""
        if self.config.save_total_limit <= 0:
            return

        try:
            checkpoint_dirs = []
            for path in self.output_dir.iterdir():
                if path.is_dir() and path.name.startswith("checkpoint-epoch-"):
                    checkpoint_dirs.append(path)

            # 按创建时间排序
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime)

            # 删除多余的检查点
            while len(checkpoint_dirs) > self.config.save_total_limit:
                old_checkpoint = checkpoint_dirs.pop(0)
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"已删除旧检查点: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"删除旧检查点 {old_checkpoint} 失败: {e}")
        except Exception as e:
            logger.warning(f"清理检查点时出错: {e}")

    def _save_final_model(self) -> None:
        """保存最终模型（带错误隔离）"""
        # 确保异步检查点保存完成
        if self._last_checkpoint_future is not None:
            try:
                self._last_checkpoint_future.result(timeout=600)
            except Exception as e:
                logger.warning(f"等待最终检查点保存完成时出错: {e}")

        # 关闭线程池
        self._checkpoint_executor.shutdown(wait=True)

        # 确保所有CSV数据都已写入
        self._flush_csv_buffer()
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        try:
            self.accelerator.save_model(self.model, final_model_dir)
        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")

        # 保存LoRA适配器
        if self.lora_manager:
            try:
                self.lora_manager.save_adapter(self.model, final_model_dir / "lora_adapters")

                # 可选：保存合并后的模型
                if getattr(self.config, 'save_merged_model', False):
                    merged_model_dir = final_model_dir / "merged_model"
                    self.save_merged_model(merged_model_dir)
            except Exception as e:
                logger.error(f"保存LoRA适配器失败: {e}")

        # 保存梯度验证报告
        if self.gradient_validator:
            try:
                report_path = self.gradient_validator.export_report(
                    self.output_dir / "gradient_validation_report.json"
                )
                logger.info(f"梯度验证报告已保存到: {report_path}")
            except Exception as e:
                logger.error(f"保存梯度验证报告失败: {e}")

        # 保存内存监控报告
        if self.memory_monitor:
            try:
                memory_report_path = self.memory_monitor.export_stats(
                    self.output_dir / "memory_usage_report.json"
                )
                logger.info(f"内存使用报告已保存到: {memory_report_path}")

                # 输出内存优化建议
                recommendations = self.memory_monitor.get_optimization_recommendations()
                logger.info("内存优化建议:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"  {i}. {rec}")
            except Exception as e:
                logger.error(f"保存内存监控报告失败: {e}")

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
        summary = {
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

        # 添加梯度验证摘要
        if self.gradient_validator:
            try:
                gradient_summary = self.gradient_validator.get_summary()
                summary['gradient_validation'] = gradient_summary
            except Exception as e:
                logger.error(f"获取梯度验证摘要失败: {e}")
                summary['gradient_validation'] = {'error': str(e)}

        # 添加内存监控摘要
        if self.memory_monitor:
            try:
                memory_summary = self.memory_monitor.get_memory_summary()
                summary['memory_monitoring'] = memory_summary
            except Exception as e:
                logger.error(f"获取内存监控摘要失败: {e}")
                summary['memory_monitoring'] = {'error': str(e)}

        return summary

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint_path = Path(checkpoint_path)

        # 加载训练状态
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = safe_torch_load_cpu(
                training_state_path,
                context="Training state",
            )

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

        # 合并并保存模型。Accelerate 包装后的模型可能不暴露 merger 需要的属性。
        model_to_merge = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        merged_model = self.model_merger.merge_all_adapters(model_to_merge)
        self.accelerator.save_model(merged_model, output_dir)

        logger.info(f"合并后的模型已保存到: {output_dir}")
