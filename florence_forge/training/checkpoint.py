"""Model checkpoint management module (v1, 函数式工具集).

⚠️ 仓库内同时存在两个 `CheckpointManager`：
- 本文件 (v1)：`CheckpointManager` + 配套 `create_checkpoint_manager / save_model_only / load_model_only` 函数式工具，被 `trainer.py`（v1 训练栈）和外部脚本使用。
- `checkpoint_manager.py` (v2)：OO 生命周期版，供 `trainer_refactored.py`（v2 训练栈）使用。

二者**分工明确，不要混用**。详情见 `checkpoint_manager.py` 顶部说明。
（v1.1.0 起两者已共用 `_checkpoint_io.py` 的底层序列化原语，见该模块说明。）

This module provides functionality for saving and loading training checkpoints,
including model state, optimizer state, and training metadata.
"""

import json
import torch
import logging
from datetime import datetime
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

from ..utils.torch_serialization import safe_torch_load
from ._checkpoint_io import atomic_torch_save, load_checkpoint_file

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager for model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], max_checkpoints: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoint metadata
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        
        logger.info(f"Checkpoint manager initialized with directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            step: Current training step
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            extra_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'extra_data': extra_data or {}
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        try:
            # Save checkpoint (atomic write to avoid truncated files on crash)
            atomic_torch_save(checkpoint_data, checkpoint_path)
            
            # Update checkpoint metadata
            checkpoint_info = {
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'metrics': metrics or {},
                'timestamp': timestamp,
                'is_best': is_best
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # Update best checkpoint if this is the best
            if is_best:
                self.best_checkpoint = checkpoint_info.copy()
                if metrics and 'eval_loss' in metrics:
                    self.best_metric_value = metrics['eval_loss']
                
                # Save best checkpoint separately
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                atomic_torch_save(checkpoint_data, best_path)
                logger.info(f"New best checkpoint saved: {best_path}")
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint metadata
        """
        try:
            # Load checkpoint without enabling arbitrary pickle execution on supported PyTorch versions.
            checkpoint_data = load_checkpoint_file(
                checkpoint_path,
                map_location=device,
                context="Training checkpoint",
            )
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Extract metadata
            metadata = {
                'epoch': checkpoint_data.get('epoch', 0),
                'step': checkpoint_data.get('step', 0),
                'loss': checkpoint_data.get('loss', 0.0),
                'metrics': checkpoint_data.get('metrics', {}),
                'timestamp': checkpoint_data.get('timestamp', ''),
                'extra_data': checkpoint_data.get('extra_data', {})
            }
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
            
        Returns:
            Best checkpoint metadata or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        
        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None
        
        return self.load_checkpoint(best_path, model, optimizer, scheduler, device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoint_history:
            return None
        
        return self.checkpoint_history[-1]['path']
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                
                # Remove from history
                self.checkpoint_history = [
                    cp for cp in self.checkpoint_history 
                    if cp['path'] != str(checkpoint_path)
                ]
                
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
            return False
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep only the most recent
        self.checkpoint_history.sort(key=lambda x: x['timestamp'])
        
        # Delete oldest checkpoints
        checkpoints_to_delete = self.checkpoint_history[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_delete:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and not checkpoint_info.get('is_best', False):
                try:
                    checkpoint_path.unlink()
                    logger.debug(f"Cleaned up old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoint {checkpoint_path}: {e}")
        
        # Update history
        self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
    
    def save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        
        metadata = {
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'best_metric_value': self.best_metric_value,
            'max_checkpoints': self.max_checkpoints
        }
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Checkpoint metadata saved: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def load_metadata(self) -> None:
        """Load checkpoint metadata from file."""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        
        if not metadata_path.exists():
            logger.debug("No checkpoint metadata file found")
            return
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.checkpoint_history = metadata.get('checkpoint_history', [])
            self.best_checkpoint = metadata.get('best_checkpoint')
            self.best_metric_value = metadata.get('best_metric_value')
            
            logger.debug(f"Checkpoint metadata loaded: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata: {e}")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        return {
            'total_checkpoints': len(self.checkpoint_history),
            'best_checkpoint': self.best_checkpoint,
            'best_metric_value': self.best_metric_value,
            'latest_checkpoint': self.get_latest_checkpoint(),
            'checkpoint_dir': str(self.checkpoint_dir),
            'max_checkpoints': self.max_checkpoints
        }


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    max_checkpoints: int = 5
) -> CheckpointManager:
    """Create a checkpoint manager.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(checkpoint_dir, max_checkpoints)


def save_model_only(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save only the model state dict.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        metadata: Additional metadata to save
    """
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        torch.save(save_data, save_path)
        logger.info(f"Model saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {e}")
        raise


def load_model_only(
    model: torch.nn.Module,
    load_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load only the model state dict.
    
    Args:
        model: Model to load state into
        load_path: Path to load the model from
        device: Device to load model on
        
    Returns:
        Metadata from the saved file
    """
    try:
        load_kwargs = {}
        if device is not None:
            load_kwargs["map_location"] = device
        save_data = safe_torch_load(
            load_path,
            context="Model checkpoint",
            **load_kwargs,
        )
        
        model.load_state_dict(save_data['model_state_dict'])
        
        metadata = save_data.get('metadata', {})
        logger.info(f"Model loaded: {load_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to load model from {load_path}: {e}")
        raise
