"""FlorenceForge可视化工具模块

提供训练过程和结果的可视化功能
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from .optional_dependencies import missing_dependency_message
from .plot_backend import finalize_matplotlib_figure


def _get_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("可视化功能", "pandas")
        ) from e


def _get_pil_image():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("可视化功能", "Pillow")
        ) from e


def _get_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("可视化功能", "matplotlib")
        ) from e

    # 按需设置 matplotlib 中文字体，避免模块导入时产生副作用
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt, Rectangle


def _get_confusion_matrix():
    try:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("混淆矩阵绘制", "scikit-learn")
        ) from e


def _get_seaborn():
    try:
        import seaborn as sns
        return sns
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("该可视化功能", "seaborn")
        ) from e


def _get_plotly():
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        return make_subplots, go
    except ImportError as e:
        raise ImportError(
            missing_dependency_message("交互式仪表板", "plotly")
        ) from e

def plot_training_curves(
    metrics_data: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "训练曲线",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """绘制训练曲线
    
    Args:
        metrics_data: 指标数据字典，键为指标名称，值为数值列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    plt, _ = _get_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 损失曲线
    ax1 = axes[0, 0]
    if 'train_loss' in metrics_data:
        ax1.plot(metrics_data['train_loss'], label='训练损失', color='blue')
    if 'val_loss' in metrics_data:
        ax1.plot(metrics_data['val_loss'], label='验证损失', color='red')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax2 = axes[0, 1]
    if 'learning_rate' in metrics_data:
        ax2.plot(metrics_data['learning_rate'], label='学习率', color='green')
        ax2.set_title('学习率曲线')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax3 = axes[1, 0]
    accuracy_metrics = [k for k in metrics_data.keys() if 'accuracy' in k.lower()]
    for metric in accuracy_metrics:
        ax3.plot(metrics_data[metric], label=metric)
    if accuracy_metrics:
        ax3.set_title('准确率曲线')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 其他指标
    ax4 = axes[1, 1]
    other_metrics = [k for k in metrics_data.keys() 
                    if k not in ['train_loss', 'val_loss', 'learning_rate'] 
                    and 'accuracy' not in k.lower()]
    for metric in other_metrics[:5]:  # 最多显示5个指标
        ax4.plot(metrics_data[metric], label=metric)
    if other_metrics:
        ax4.set_title('其他指标')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def plot_task_distribution(
    task_counts: Dict[str, int],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "任务分布",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """绘制任务分布图
    
    Args:
        task_counts: 任务计数字典
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    plt, _ = _get_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    tasks = list(task_counts.keys())
    counts = list(task_counts.values())
    
    # 柱状图
    bars = ax1.bar(tasks, counts, color=plt.cm.Set3(np.linspace(0, 1, len(tasks))))
    ax1.set_title('任务样本数量')
    ax1.set_xlabel('任务类型')
    ax1.set_ylabel('样本数量')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    # 饼图
    ax2.pie(counts, labels=tasks, autopct='%1.1f%%', startangle=90,
           colors=plt.cm.Set3(np.linspace(0, 1, len(tasks))))
    ax2.set_title('任务比例分布')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def visualize_detection_results(
    image: Union[str, Path, Any],
    detections: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "检测结果",
    figsize: Tuple[int, int] = (12, 8),
    show_confidence: bool = True
) -> None:
    """可视化目标检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
        show_confidence: 是否显示置信度
    """
    Image = _get_pil_image()
    plt, Rectangle = _get_matplotlib()
    # 加载图像
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("图像必须是路径或PIL.Image对象")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    
    # 颜色映射
    colors = plt.cm.Set1(np.linspace(0, 1, len(detections)))
    
    img_width, img_height = image.size
    
    for i, detection in enumerate(detections):
        bbox = detection.get('bbox', [])
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 如果坐标是归一化的，转换为绝对坐标
            if all(0 <= coord <= 1 for coord in bbox):
                x1, y1, x2, y2 = x1*img_width, y1*img_height, x2*img_width, y2*img_height
            
            # 绘制边界框
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            text = label
            if show_confidence and confidence > 0:
                text += f' ({confidence:.2f})'
            
            ax.text(x1, y1-5, text, fontsize=10, color=colors[i],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "混淆矩阵",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    plt, _ = _get_matplotlib()
    sns = _get_seaborn()
    confusion_matrix = _get_confusion_matrix()
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "指标比较",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """绘制指标比较图
    
    Args:
        metrics_dict: 指标字典，格式为 {model_name: {metric_name: value}}
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    pd = _get_pandas()
    plt, _ = _get_matplotlib()
    # 转换为DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 柱状图
    df.plot(kind='bar', ax=axes[0], rot=45)
    axes[0].set_title('指标对比')
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('指标值')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 雷达图
    metrics = list(df.columns)
    models = list(df.index)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax = axes[1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    
    for i, model in enumerate(models):
        values = df.loc[model].tolist()
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_ylim(0, 1)
    ax.set_title('雷达图对比')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def create_evaluation_dashboard(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "评估仪表板"
) -> None:
    """创建评估仪表板
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
        title: 仪表板标题
    """
    make_subplots, go = _get_plotly()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('任务性能', '指标分布', '错误分析', '样本难度'),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'pie'}, {'type': 'scatter'}]]
    )
    
    # 任务性能柱状图
    if 'task_performance' in results:
        task_perf = results['task_performance']
        fig.add_trace(
            go.Bar(
                x=list(task_perf.keys()),
                y=list(task_perf.values()),
                name='任务性能'
            ),
            row=1, col=1
        )
    
    # 指标分布箱线图
    if 'metric_distributions' in results:
        metric_dist = results['metric_distributions']
        for metric, values in metric_dist.items():
            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric
                ),
                row=1, col=2
            )
    
    # 错误类型饼图
    if 'error_analysis' in results:
        error_analysis = results['error_analysis']
        fig.add_trace(
            go.Pie(
                labels=list(error_analysis.keys()),
                values=list(error_analysis.values()),
                name='错误分析'
            ),
            row=2, col=1
        )
    
    # 样本难度散点图
    if 'sample_difficulty' in results:
        difficulty = results['sample_difficulty']
        fig.add_trace(
            go.Scatter(
                x=difficulty.get('complexity', []),
                y=difficulty.get('performance', []),
                mode='markers',
                name='样本难度',
                text=difficulty.get('labels', []),
                hovertemplate='复杂度: %{x}<br>性能: %{y}<br>%{text}'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "注意力热力图",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重矩阵
        tokens: 词汇列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    plt, _ = _get_matplotlib()
    sns = _get_seaborn()
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=False,
        cbar=True
    )
    
    plt.title(title)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def plot_loss_landscape(
    loss_surface: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "损失地形图",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """绘制损失地形图
    
    Args:
        loss_surface: 损失表面数据
        save_path: 保存路径
        title: 图表标题
        figsize: 图像尺寸
    """
    plt, _ = _get_matplotlib()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(loss_surface.shape[0])
    y = np.arange(loss_surface.shape[1])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, loss_surface.T, cmap='viridis', alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('参数维度1')
    ax.set_ylabel('参数维度2')
    ax.set_zlabel('损失值')
    
    fig.colorbar(surf)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    finalize_matplotlib_figure()

def create_training_report(
    training_history: Dict[str, Any],
    model_info: Dict[str, Any],
    save_dir: Union[str, Path],
    report_name: str = "training_report"
) -> None:
    """创建训练报告
    
    Args:
        training_history: 训练历史数据
        model_info: 模型信息
        save_dir: 保存目录
        report_name: 报告名称
    """
    pd = _get_pandas()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制训练曲线
    if 'metrics' in training_history:
        plot_training_curves(
            training_history['metrics'],
            save_path=save_dir / f"{report_name}_curves.png",
            title="训练过程曲线"
        )
    
    # 绘制任务分布
    if 'task_distribution' in training_history:
        plot_task_distribution(
            training_history['task_distribution'],
            save_path=save_dir / f"{report_name}_task_dist.png",
            title="训练任务分布"
        )
    
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>训练报告 - {report_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>训练报告</h1>
            <h2>{report_name}</h2>
        </div>
        
        <div class="section">
            <h3>模型信息</h3>
            <div class="metric">模型名称: {model_info.get('name', 'Unknown')}</div>
            <div class="metric">参数数量: {model_info.get('parameters', 'Unknown')}</div>
            <div class="metric">训练时间: {model_info.get('training_time', 'Unknown')}</div>
        </div>
        
        <div class="section">
            <h3>训练曲线</h3>
            <img src="{report_name}_curves.png" alt="训练曲线">
        </div>
        
        <div class="section">
            <h3>任务分布</h3>
            <img src="{report_name}_task_dist.png" alt="任务分布">
        </div>
        
        <div class="section">
            <h3>最终指标</h3>
    """
    
    # 添加最终指标
    if 'final_metrics' in training_history:
        for metric, value in training_history['final_metrics'].items():
            html_content += f'<div class="metric">{metric}: {value:.4f}</div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(save_dir / f"{report_name}.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 保存JSON数据
    report_data = {
        'training_history': training_history,
        'model_info': model_info,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    with open(save_dir / f"{report_name}_data.json", 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

class VisualizationManager:
    """可视化管理器
    
    统一管理各种可视化功能
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """初始化可视化管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        plt, _ = _get_matplotlib()
        sns = _get_seaborn()
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_and_save(
        self,
        plot_func,
        filename: str,
        *args,
        **kwargs
    ) -> Path:
        """绘制并保存图表
        
        Args:
            plot_func: 绘图函数
            filename: 文件名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            保存路径
        """
        save_path = self.output_dir / filename
        kwargs['save_path'] = save_path
        
        plot_func(*args, **kwargs)
        
        return save_path
    
    def create_summary_dashboard(
        self,
        data: Dict[str, Any],
        title: str = "总结仪表板"
    ) -> Path:
        """创建总结仪表板
        
        Args:
            data: 数据字典
            title: 仪表板标题
            
        Returns:
            保存路径
        """
        save_path = self.output_dir / "dashboard.html"
        
        create_evaluation_dashboard(
            data,
            save_path=save_path,
            title=title
        )
        
        return save_path
    
    def generate_report(
        self,
        training_history: Dict[str, Any],
        model_info: Dict[str, Any],
        report_name: str = "report"
    ) -> Path:
        """生成完整报告
        
        Args:
            training_history: 训练历史
            model_info: 模型信息
            report_name: 报告名称
            
        Returns:
            报告目录路径
        """
        create_training_report(
            training_history,
            model_info,
            self.output_dir,
            report_name
        )
        
        return self.output_dir
