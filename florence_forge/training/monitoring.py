#!/usr/bin/env python3
"""
Florence Forge - 训练监控模块

提供 WandB 和 SwanLab 等训练可视化监控功能
"""

import os
import logging
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """监控配置类"""
    
    # 启用的监控工具
    enable_wandb: bool = False
    enable_swanlab: bool = False
    enable_tensorboard: bool = True
    
    # WandB 配置
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # SwanLab 配置
    swanlab_project: Optional[str] = None
    swanlab_experiment_name: Optional[str] = None
    swanlab_description: Optional[str] = None
    swanlab_tags: List[str] = field(default_factory=list)
    swanlab_config: Dict[str, Any] = field(default_factory=dict)
    
    # TensorBoard 配置
    tensorboard_log_dir: Optional[str] = None
    
    # 通用配置
    log_frequency: int = 10  # 每多少步记录一次
    save_model_frequency: int = 500  # 每多少步保存一次模型
    log_gradients: bool = False  # 是否记录梯度
    log_model_architecture: bool = True  # 是否记录模型架构
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enable_wandb": self.enable_wandb,
            "enable_swanlab": self.enable_swanlab,
            "enable_tensorboard": self.enable_tensorboard,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "wandb_tags": self.wandb_tags,
            "wandb_notes": self.wandb_notes,
            "swanlab_project": self.swanlab_project,
            "swanlab_experiment_name": self.swanlab_experiment_name,
            "swanlab_description": self.swanlab_description,
            "swanlab_tags": self.swanlab_tags,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "log_frequency": self.log_frequency,
            "save_model_frequency": self.save_model_frequency,
            "log_gradients": self.log_gradients,
            "log_model_architecture": self.log_model_architecture,
        }


class TrainingMonitor:
    """训练监控器
    
    集成多种监控工具，提供统一的训练监控接口
    """
    
    def __init__(self, config: MonitoringConfig, output_dir: Optional[str] = None):
        """初始化监控器
        
        Args:
            config: 监控配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控工具实例
        self.wandb_run = None
        self.swanlab_run = None
        self.tensorboard_writer = None
        
        # 初始化监控工具
        self._init_monitoring_tools()
        
        logger.info(f"训练监控器初始化完成，启用的工具: "
                   f"WandB={config.enable_wandb}, "
                   f"SwanLab={config.enable_swanlab}, "
                   f"TensorBoard={config.enable_tensorboard}")
    
    def _init_monitoring_tools(self) -> None:
        """初始化监控工具"""
        
        # 初始化 WandB
        if self.config.enable_wandb:
            self._init_wandb()
        
        # 初始化 SwanLab
        if self.config.enable_swanlab:
            self._init_swanlab()
        
        # 初始化 TensorBoard
        if self.config.enable_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(self) -> None:
        """初始化 WandB"""
        try:
            import wandb
            
            # 设置项目名称
            project = self.config.wandb_project or "florence-forge-training"
            
            # 初始化 WandB run
            self.wandb_run = wandb.init(
                project=project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                notes=self.config.wandb_notes,
                config=self.config.wandb_config,
                dir=str(self.output_dir),
                reinit=True
            )
            
            logger.info(f"WandB 初始化成功，项目: {project}")
            
        except ImportError:
            logger.warning("WandB 未安装，跳过 WandB 监控")
            self.config.enable_wandb = False
        except Exception as e:
            logger.error(f"WandB 初始化失败: {e}")
            self.config.enable_wandb = False
    
    def _init_swanlab(self) -> None:
        """初始化 SwanLab"""
        try:
            import swanlab
            
            # 设置项目名称
            project = self.config.swanlab_project or "florence-forge-training"
            
            # 初始化 SwanLab run
            self.swanlab_run = swanlab.init(
                project=project,
                experiment_name=self.config.swanlab_experiment_name,
                description=self.config.swanlab_description,
                config=self.config.swanlab_config,
                logdir=str(self.output_dir / "swanlab_logs")
            )
            
            # 添加标签
            if self.config.swanlab_tags:
                for tag in self.config.swanlab_tags:
                    self.swanlab_run.tag(tag)
            
            logger.info(f"SwanLab 初始化成功，项目: {project}")
            
        except ImportError:
            logger.warning("SwanLab 未安装，跳过 SwanLab 监控")
            self.config.enable_swanlab = False
        except Exception as e:
            logger.error(f"SwanLab 初始化失败: {e}")
            self.config.enable_swanlab = False
    
    def _init_tensorboard(self) -> None:
        """初始化 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # 设置日志目录
            log_dir = self.config.tensorboard_log_dir or str(self.output_dir / "tensorboard_logs")
            
            # 创建 TensorBoard writer
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            
            logger.info(f"TensorBoard 初始化成功，日志目录: {log_dir}")
            
        except ImportError:
            logger.warning("TensorBoard 未安装，跳过 TensorBoard 监控")
            self.config.enable_tensorboard = False
        except Exception as e:
            logger.error(f"TensorBoard 初始化失败: {e}")
            self.config.enable_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, prefix: str = "") -> None:
        """记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀
        """
        if not metrics:
            return
        
        # 添加前缀
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # 记录到 WandB
        if self.config.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"WandB 记录指标失败: {e}")
        
        # 记录到 SwanLab
        if self.config.enable_swanlab and self.swanlab_run:
            try:
                for key, value in metrics.items():
                    self.swanlab_run.log({key: value}, step=step)
            except Exception as e:
                logger.warning(f"SwanLab 记录指标失败: {e}")
        
        # 记录到 TensorBoard
        if self.config.enable_tensorboard and self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"TensorBoard 记录指标失败: {e}")
    
    def log_model_architecture(self, model, input_sample=None) -> None:
        """记录模型架构
        
        Args:
            model: 模型实例
            input_sample: 输入样本（用于可视化模型图）
        """
        if not self.config.log_model_architecture:
            return
        
        # 记录到 WandB
        if self.config.enable_wandb and self.wandb_run:
            try:
                import wandb
                if input_sample is not None:
                    wandb.watch(model, log="all", log_freq=self.config.log_frequency)
            except Exception as e:
                logger.warning(f"WandB 记录模型架构失败: {e}")
        
        # 记录到 TensorBoard
        if self.config.enable_tensorboard and self.tensorboard_writer and input_sample is not None:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"TensorBoard 记录模型架构失败: {e}")
    
    def log_gradients(self, model, step: int) -> None:
        """记录梯度信息
        
        Args:
            model: 模型实例
            step: 当前步数
        """
        if not self.config.log_gradients:
            return
        
        gradient_metrics = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_metrics[f"gradients/{name}_norm"] = grad_norm
        
        if gradient_metrics:
            self.log_metrics(gradient_metrics, step)
    
    def log_learning_rate(self, lr: float, step: int) -> None:
        """记录学习率
        
        Args:
            lr: 学习率
            step: 当前步数
        """
        self.log_metrics({"learning_rate": lr}, step)
    
    def log_images(self, images: Dict[str, Any], step: int) -> None:
        """记录图像
        
        Args:
            images: 图像字典
            step: 当前步数
        """
        # 记录到 WandB
        if self.config.enable_wandb and self.wandb_run:
            try:
                import wandb
                wandb_images = {}
                for key, img in images.items():
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    wandb_images[key] = wandb.Image(img)
                self.wandb_run.log(wandb_images, step=step)
            except Exception as e:
                logger.warning(f"WandB 记录图像失败: {e}")
        
        # 记录到 SwanLab
        if self.config.enable_swanlab and self.swanlab_run:
            try:
                import swanlab
                for key, img in images.items():
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    self.swanlab_run.log({key: swanlab.Image(img)}, step=step)
            except Exception as e:
                logger.warning(f"SwanLab 记录图像失败: {e}")
        
        # 记录到 TensorBoard
        if self.config.enable_tensorboard and self.tensorboard_writer:
            try:
                for key, img in images.items():
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    # 假设图像格式为 (C, H, W) 或 (H, W, C)
                    if len(img.shape) == 3:
                        if img.shape[0] in [1, 3]:  # (C, H, W)
                            self.tensorboard_writer.add_image(key, img, step)
                        else:  # (H, W, C)
                            self.tensorboard_writer.add_image(key, img, step, dataformats='HWC')
            except Exception as e:
                logger.warning(f"TensorBoard 记录图像失败: {e}")
    
    def save_model_artifact(self, model_path: str, step: int, metadata: Optional[Dict] = None) -> None:
        """保存模型工件
        
        Args:
            model_path: 模型路径
            step: 当前步数
            metadata: 元数据
        """
        # 保存到 WandB
        if self.config.enable_wandb and self.wandb_run:
            try:
                import wandb
                artifact = wandb.Artifact(
                    name=f"model-step-{step}",
                    type="model",
                    metadata=metadata or {}
                )
                artifact.add_file(model_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"WandB 保存模型工件失败: {e}")
    
    def finish(self) -> None:
        """结束监控"""
        # 结束 WandB
        if self.config.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("WandB 监控已结束")
            except Exception as e:
                logger.warning(f"WandB 结束失败: {e}")
        
        # 结束 SwanLab
        if self.config.enable_swanlab and self.swanlab_run:
            try:
                self.swanlab_run.finish()
                logger.info("SwanLab 监控已结束")
            except Exception as e:
                logger.warning(f"SwanLab 结束失败: {e}")
        
        # 关闭 TensorBoard
        if self.config.enable_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard 监控已结束")
            except Exception as e:
                logger.warning(f"TensorBoard 结束失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.finish()


def create_monitoring_config(
    enable_wandb: bool = False,
    enable_swanlab: bool = False,
    enable_tensorboard: bool = True,
    project_name: str = "florence-forge-training",
    experiment_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> MonitoringConfig:
    """创建监控配置的便捷函数
    
    Args:
        enable_wandb: 是否启用 WandB
        enable_swanlab: 是否启用 SwanLab
        enable_tensorboard: 是否启用 TensorBoard
        project_name: 项目名称
        experiment_name: 实验名称
        tags: 标签列表
        **kwargs: 其他配置参数
    
    Returns:
        监控配置实例
    """
    config = MonitoringConfig(
        enable_wandb=enable_wandb,
        enable_swanlab=enable_swanlab,
        enable_tensorboard=enable_tensorboard,
        **kwargs
    )
    
    # 设置项目名称
    if enable_wandb:
        config.wandb_project = project_name
        config.wandb_run_name = experiment_name
        config.wandb_tags = tags or []
    
    if enable_swanlab:
        config.swanlab_project = project_name
        config.swanlab_experiment_name = experiment_name
        config.swanlab_tags = tags or []
    
    return config