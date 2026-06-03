#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练可视化模块

提供训练过程中的各种可视化功能，包括损失曲线、学习率曲线等
"""

import json
import logging
from typing import Optional
from pathlib import Path

# 初始化logger
logger = logging.getLogger(__name__)

# 分别检测各个可视化库的可用性
MATLOTLIB_AVAILABLE = False  # 这个变量名有拼写错误，但保留以防有其他地方使用
SEABORN_AVAILABLE = False
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False

# 检测matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logger.debug(f"matplotlib导入失败: {e}")
    MATPLOTLIB_AVAILABLE = False
    
    class MockRcParams:
        def __setitem__(self, key, value):
            pass
        def __getitem__(self, key):
            return None
    
    class plt:
        rcParams = MockRcParams()
        
        @staticmethod
        def subplots(*args, **kwargs):
            raise ImportError("matplotlib not available")
        @staticmethod
        def tight_layout():
            pass
        @staticmethod
        def savefig(*args, **kwargs):
            pass
        @staticmethod
        def close():
            pass
        @staticmethod
        def figure(*args, **kwargs):
            raise ImportError("matplotlib not available")

# 检测seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"seaborn导入失败: {e}")
    SEABORN_AVAILABLE = False
    
    class sns:
        @staticmethod
        def set_style(*args, **kwargs):
            pass
        @staticmethod
        def set_palette(*args, **kwargs):
            pass

# 检测pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"pandas导入失败: {e}")
    PANDAS_AVAILABLE = False
    
    class pd:
        @staticmethod
        def read_csv(*args, **kwargs):
            raise ImportError("pandas not available")
        @staticmethod
        def DataFrame(*args, **kwargs):
            raise ImportError("pandas not available")

# 检测numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    logger.debug(f"numpy导入失败: {e}")
    NUMPY_AVAILABLE = False
    
    class np:
        @staticmethod
        def array(*args, **kwargs):
            raise ImportError("numpy not available")
        @staticmethod
        def linspace(*args, **kwargs):
            raise ImportError("numpy not available")

# 核心可视化功能需要matplotlib和pandas
VISUALIZATION_AVAILABLE = MATPLOTLIB_AVAILABLE and PANDAS_AVAILABLE

# 为了向后兼容，保留原有的变量名
MATLOTLIB_AVAILABLE = VISUALIZATION_AVAILABLE

class TrainingVisualizer:
    """训练可视化器
    
    提供训练过程的可视化功能
    """
    
    def __init__(self, output_dir: str):
        """初始化可视化器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查可视化库的可用性并设置相应的配置
        self._check_visualization_dependencies()
        
        if VISUALIZATION_AVAILABLE:
            # 设置matplotlib中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 如果seaborn可用，设置样式
            if SEABORN_AVAILABLE:
                sns.set_style("whitegrid")
                sns.set_palette("husl")
            else:
                logger.info("Seaborn不可用，将使用matplotlib默认样式")
        else:
            missing_deps = []
            if not MATPLOTLIB_AVAILABLE:
                missing_deps.append("matplotlib")
            if not PANDAS_AVAILABLE:
                missing_deps.append("pandas")
            
            logger.warning(
                f"可视化功能已禁用，缺少依赖: {', '.join(missing_deps)}。"
                f"请安装: pip install {' '.join(missing_deps)}"
            )
    
    def _check_visualization_dependencies(self) -> None:
        """检查并报告可视化依赖的状态"""
        deps_status = {
            "matplotlib": MATPLOTLIB_AVAILABLE,
            "pandas": PANDAS_AVAILABLE,
            "seaborn": SEABORN_AVAILABLE,
            "numpy": NUMPY_AVAILABLE
        }
        
        available_deps = [name for name, available in deps_status.items() if available]
        missing_deps = [name for name, available in deps_status.items() if not available]
        
        if available_deps:
            logger.debug(f"可用的可视化依赖: {', '.join(available_deps)}")
        
        if missing_deps:
            logger.debug(f"缺失的可视化依赖: {', '.join(missing_deps)}")
    
    def plot_loss_curves(self, save_path: Optional[str] = None) -> str:
        """绘制损失曲线
        
        Args:
            save_path: 保存路径，如果为None则使用默认路径
            
        Returns:
            保存的图片路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化功能不可用，跳过损失曲线绘制")
            return ""
            
        try:
            # 读取epoch指标数据
            epoch_csv_path = self.output_dir / "epoch_metrics.csv"
            if not epoch_csv_path.exists():
                logger.warning(f"Epoch metrics file not found: {epoch_csv_path}")
                return ""
            
            df = pd.read_csv(epoch_csv_path)
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制损失曲线
            epochs = df['epoch'] + 1  # 从1开始
            ax1.plot(epochs, df['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
            if 'val_loss' in df.columns and df['val_loss'].sum() > 0:
                ax1.plot(
                    epochs,
                    df['val_loss'],
                    'r-',
                    label='Validation Loss',
                    linewidth=2,
                    marker='s'
                )
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制学习率曲线
            if 'learning_rate' in df.columns:
                ax2.plot(
                    epochs,
                    df['learning_rate'],
                    'g-',
                    label='Learning Rate',
                    linewidth=2,
                    marker='^'
                )
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Curve')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')  # 使用对数刻度
            
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = self.output_dir / "loss_curves.png"
            else:
                save_path = Path(save_path)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Loss curves saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error plotting loss curves: {e}")
            return ""
    
    def plot_step_metrics(self, save_path: Optional[str] = None) -> str:
        """绘制步骤级指标
        
        Args:
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化功能不可用，跳过步骤指标绘制")
            return ""
            
        try:
            # 读取步骤指标数据
            step_csv_path = self.output_dir / "step_metrics.csv"
            if not step_csv_path.exists():
                logger.warning(f"Step metrics file not found: {step_csv_path}")
                return ""
            
            df = pd.read_csv(step_csv_path)
            
            # 创建图形
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制步骤损失
            ax1.plot(df['step'], df['loss'], 'b-', alpha=0.7, linewidth=1)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Step-wise Loss Changes')
            ax1.grid(True, alpha=0.3)
            
            # 绘制学习率变化
            if 'learning_rate' in df.columns:
                ax2.plot(df['step'], df['learning_rate'], 'g-', alpha=0.7, linewidth=1)
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Step-wise Learning Rate Changes')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
            
            # 绘制梯度范数
            if 'grad_norm' in df.columns:
                ax3.plot(df['step'], df['grad_norm'], 'r-', alpha=0.7, linewidth=1)
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('Gradient Norm')
                ax3.set_title('Gradient Norm Changes')
                ax3.grid(True, alpha=0.3)
            
            # 绘制每步耗时
            if 'time_per_step' in df.columns:
                ax4.plot(df['step'], df['time_per_step'], 'm-', alpha=0.7, linewidth=1)
                ax4.set_xlabel('Training Step')
                ax4.set_ylabel('Time per Step (seconds)')
                ax4.set_title('Training Speed Changes')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = self.output_dir / "step_metrics.png"
            else:
                save_path = Path(save_path)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Step metrics plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error plotting step metrics: {e}")
            return ""
    
    def plot_task_distribution(self, save_path: Optional[str] = None) -> str:
        """绘制任务分布图
        
        Args:
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化功能不可用，跳过任务分布图绘制")
            return ""
            
        try:
            # 读取步骤指标数据
            step_csv_path = self.output_dir / "step_metrics.csv"
            if not step_csv_path.exists():
                logger.warning(f"Step metrics file not found: {step_csv_path}")
                return ""
            
            df = pd.read_csv(step_csv_path)
            
            if 'task_type' not in df.columns:
                logger.warning("No task type information in step metrics")
                return ""
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 任务类型分布饼图
            task_counts = df['task_type'].value_counts()
            ax1.pie(task_counts.values, labels=task_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Task Type Distribution')
            
            # 任务类型损失对比
            task_loss = df.groupby('task_type')['loss'].mean().sort_values()
            ax2.bar(task_loss.index, task_loss.values)
            ax2.set_xlabel('Task Type')
            ax2.set_ylabel('Average Loss')
            ax2.set_title('Average Loss Comparison by Task Type')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = self.output_dir / "task_distribution.png"
            else:
                save_path = Path(save_path)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Task distribution plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error plotting task distribution: {e}")
            return ""
    
    def generate_training_report(self) -> str:
        """生成训练报告
        
        Returns:
            报告文件路径
        """
        try:
            # 生成所有可视化图表
            loss_curve_path = self.plot_loss_curves()
            step_metrics_path = self.plot_step_metrics()
            task_dist_path = self.plot_task_distribution()
            
            # 读取训练统计信息
            stats_path = self.output_dir / "dataset_statistics.json"
            training_stats = {}
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    training_stats = json.load(f)
            
            # 生成HTML报告
            report_path = self.output_dir / "training_report.html"
            
            # 为HTML报告准备图片路径
            loss_curve_img = Path(loss_curve_path).name if loss_curve_path else ""
            step_metrics_img = Path(step_metrics_path).name if step_metrics_path else ""
            task_dist_img = Path(task_dist_path).name if task_dist_path else ""

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Florence2 Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .metric {{ margin: 10px 0; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Florence2 Multi-task Training Report</h1>
    <p><strong>Generated at:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Training Statistics</h2>
    <div class="metric"><strong>Training Dataset Size:</strong> {training_stats.get('train_dataset', {}).get('total_samples', 'N/A')}</div>
    <div class="metric"><strong>Validation Dataset Size:</strong> {training_stats.get('val_dataset', {}).get('total_samples', 'N/A') if training_stats.get('val_dataset') else 'N/A'}</div>
    <div class="metric"><strong>Task Types:</strong> {', '.join(training_stats.get('train_dataset', {}).get('task_types', []))}</div>
    
    <h2>Loss Curves</h2>
    <div class="chart">
        <img src="{loss_curve_img}" alt="Loss Curves">
    </div>
    
    <h2>Detailed Metrics</h2>
    <div class="chart">
        <img src="{step_metrics_img}" alt="Detailed Metrics">
    </div>
    
    <h2>Task Distribution</h2>
    <div class="chart">
        <img src="{task_dist_img}" alt="Task Distribution">
    </div>
</body>
</html>
"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Training report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating training report: {e}")
            return ""