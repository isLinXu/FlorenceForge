"""训练器检查点管理模块（v2 · OO 生命周期版）

``from florence_forge.training import CheckpointManager`` 指向本类。

目录式遗留 API 见 ``checkpoint.py::DirectoryCheckpointManager``（及模块内
``CheckpointManager`` 别名）；工具函数 ``save_model_only`` / ``load_model_only``
亦在该模块。底层序列化与目录式 API 共用 ``_checkpoint_io``。
"""
import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from ..core.config import TrainingConfig
from ._checkpoint_io import atomic_torch_save, load_checkpoint_file, prune_checkpoints

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器
    
    负责模型检查点的保存、加载、清理和合并
    支持异步保存和多检查点管理
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        accelerator=None
    ):
        """初始化检查点管理器
        
        Args:
            model: 训练模型
            config: 训练配置
            accelerator: Accelerate 加速器（可选）
        """
        self.model = model
        self.config = config
        self.accelerator = accelerator
        
        # 检查点保存配置
        self.output_dir = Path(config.output_dir)
        self.keep_checkpoints = getattr(config, "keep_checkpoints", getattr(config, "save_total_limit", 3))
        self.save_best_only = getattr(config, "save_best_only", False)
        
        # 异步保存相关
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._saving_lock = threading.Lock()
        self._current_save_future = None
        
        # 最佳模型跟踪
        self.best_metric_value = float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        async_save: bool = True
    ) -> None:
        """保存检查点
        
        Args:
            epoch: 当前 epoch
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            metrics: 当前指标
            is_best: 是否为最佳模型
            async_save: 是否异步保存
        """
        # 如果设置了 save_best_only 且不是最佳模型，跳过
        if self.save_best_only and not is_best:
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        
        if async_save:
            # 等待上一次异步保存完成
            self._wait_for_pending_save()
            
            # 提交异步保存任务
            self._current_save_future = self._executor.submit(
                self._do_save_checkpoint,
                epoch,
                checkpoint_dir,
                optimizer,
                lr_scheduler,
                metrics,
                is_best
            )
            logger.info(f"📦 检查点保存任务已提交（异步）：{checkpoint_dir}")
        else:
            # 同步保存
            self._do_save_checkpoint(
                epoch,
                checkpoint_dir,
                optimizer,
                lr_scheduler,
                metrics,
                is_best
            )
    
    def _do_save_checkpoint(
        self,
        epoch: int,
        checkpoint_dir: Path,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        metrics: Optional[Dict[str, float]],
        is_best: bool
    ) -> None:
        """执行检查点保存（内部方法）
        
        Args:
            epoch: 当前 epoch
            checkpoint_dir: 检查点目录
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            metrics: 当前指标
            is_best: 是否为最佳模型
        """
        with self._saving_lock:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建检查点数据
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self._get_model_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'metrics': metrics or {},
                'config': self._config_to_dict()
            }
            
            # 保存检查点（原子写，避免崩溃残留损坏文件）
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            atomic_torch_save(checkpoint, checkpoint_path)
            
            # 保存配置
            config_path = checkpoint_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config_to_dict(), f, indent=2, ensure_ascii=False)
            
            # 如果是最佳模型，更新记录
            if is_best:
                self.best_checkpoint_path = checkpoint_dir
                best_marker = checkpoint_dir / "BEST_MODEL"
                best_marker.touch()
            
            logger.info(f"✅ 检查点已保存：{checkpoint_dir}")
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()

    def _config_to_dict(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的配置字典。"""
        if hasattr(self.config, "to_dict"):
            return self.config.to_dict()
        if hasattr(self.config, "model_dump"):
            return self.config.model_dump()
        return dict(getattr(self.config, "__dict__", {}))
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典（兼容 accelerate 和原生 PyTorch）
        
        Returns:
            模型状态字典
        """
        if self.accelerator is not None:
            # 使用 accelerate 的保存方法
            return self.accelerator.get_state_dict(self.model)
        else:
            # 处理 DataParallel / DistributedDataParallel
            if hasattr(self.model, 'module'):
                return self.model.module.state_dict()
            else:
                return self.model.state_dict()
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧检查点（保留最近 N 个，保护最佳模型）"""
        import shutil

        checkpoint_dirs = [
            d for d in self.output_dir.glob("checkpoint-epoch-*") if d.is_dir()
        ]

        def _remove(d: Path) -> None:
            shutil.rmtree(d)
            logger.info(f"🗑️  已删除旧检查点：{d}")

        prune_checkpoints(
            checkpoint_dirs,
            self.keep_checkpoints,
            sort_key=lambda d: d.stat().st_mtime,
            is_protected=lambda d: (d / "BEST_MODEL").exists(),
            remove=_remove,
        )
    
    def _wait_for_pending_save(self) -> None:
        """等待待处理的异步保存任务完成"""
        if self._current_save_future is not None:
            logger.debug("等待上一次异步保存完成...")
            self._current_save_future.result()
            self._current_save_future = None
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径（目录或 .pt 文件）
            optimizer: 优化器（可选，用于恢复优化器状态）
            lr_scheduler: 学习率调度器（可选）
            strict: 是否严格匹配模型参数
        
        Returns:
            检查点元数据（epoch, metrics 等）
        """
        checkpoint_path = Path(checkpoint_path)
        
        # 处理目录路径
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "checkpoint.pt"
        else:
            checkpoint_file = checkpoint_path
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点文件不存在：{checkpoint_file}")
        
        logger.info(f"📂 加载检查点：{checkpoint_file}")
        
        # 加载检查点（统一 fail-closed 安全加载）
        checkpoint = load_checkpoint_file(
            checkpoint_file, map_location="cpu", context="Training checkpoint"
        )
        
        # 恢复模型状态
        if self.accelerator is not None:
            # 使用 accelerate 的加载方法
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint['model_state_dict'],
                strict=strict
            )
        else:
            # 处理 DataParallel / DistributedDataParallel
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=strict
                )
            else:
                self.model.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=strict
                )
        
        # 恢复优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器状态
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        logger.info(f"✅ 检查点加载完成（Epoch {checkpoint.get('epoch', 'unknown')}）")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
    
    def save_final_model(
        self,
        merge_lora: bool = False,
        lora_manager=None
    ) -> None:
        """保存最终模型
        
        Args:
            merge_lora: 是否合并 LoRA 权重
            lora_manager: LoRA 管理器（当 merge_lora=True 时需要）
        """
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # 合并 LoRA 权重（如果需要）
        if merge_lora and lora_manager is not None:
            logger.info("🔀 合并 LoRA 权重...")
            from .model_merger import ModelMerger
            merger = ModelMerger(self.model, lora_manager)
            merged_model = merger.merge_lora_weights()
            model_to_save = merged_model
        else:
            model_to_save = self.model
        
        # 保存模型
        if self.accelerator is not None:
            self.accelerator.save_model(model_to_save, final_dir)
        else:
            # 获取原始模型（unwrap DataParallel）
            if hasattr(model_to_save, 'module'):
                model_to_save = model_to_save.module
            
            # 保存模型状态
            model_to_save.save_pretrained(final_dir)
        
        logger.info(f"💾 最终模型已保存：{final_dir}")
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """获取最佳检查点路径
        
        Returns:
            最佳检查点路径，如果不存在则返回 None
        """
        return self.best_checkpoint_path
    
    def cleanup(self) -> None:
        """清理资源"""
        # 等待所有异步保存完成
        self._wait_for_pending_save()
        self._executor.shutdown(wait=True)
