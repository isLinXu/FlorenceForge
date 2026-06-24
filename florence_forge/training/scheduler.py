#!/usr/bin/env python3
"""
Florence Forge - 任务调度器

提供多种任务调度策略，用于多任务训练中的任务采样和权重调整
"""

import random
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque, defaultdict

from ..core.config import TaskSchedulingConfig

logger = logging.getLogger(__name__)

class TaskScheduler:
    """任务调度器
    
    负责在多任务训练过程中决定每个步骤使用哪个任务
    """
    
    def __init__(
        self,
        task_types: List[str],
        config: Optional[TaskSchedulingConfig] = None,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """初始化任务调度器
        
        Args:
            task_types: 任务类型列表
            config: 调度配置
            initial_weights: 初始任务权重
        """
        self.task_types = task_types
        self.config = config or TaskSchedulingConfig()
        
        # 初始化任务权重
        if initial_weights:
            self.task_weights = initial_weights.copy()
        else:
            self.task_weights = {task: 1.0 for task in task_types}
        
        # 调度状态
        self.current_step = 0
        self.task_history = deque(maxlen=1000)  # 记录最近的任务选择历史
        self.task_performance = defaultdict(list)  # 记录任务性能历史
        self.task_counts = defaultdict(int)  # 记录任务选择次数
        
        # 课程学习状态
        self.curriculum_stage = 0
        self.curriculum_tasks = self._init_curriculum_order()
        
        logger.info(f"任务调度器初始化完成，策略: {self.config.strategy}")
        logger.info(f"任务类型: {task_types}")
        logger.info(f"初始权重: {self.task_weights}")
    
    def _init_curriculum_order(self) -> List[str]:
        """初始化课程学习的任务顺序
        
        Returns:
            按难度排序的任务列表
        """
        # 从配置中读取任务复杂度（如果提供）
        if self.config.task_complexity is not None:
            task_complexity = self.config.task_complexity
            logger.info(f"使用配置中的任务复杂度: {task_complexity}")
        else:
            # 默认复杂度（可被子类或配置文件覆盖）
            task_complexity = self._get_default_task_complexity()
        
        # 按复杂度排序
        sorted_tasks = sorted(
            self.task_types,
            key=lambda x: task_complexity.get(x, 3)
        )
        
        return sorted_tasks
    
    @staticmethod
    def _get_default_task_complexity() -> Dict[str, int]:
        """获取默认任务复杂度
        
        Returns:
            任务复杂度字典（任务名 -> 复杂度 1-10）
        """
        return {
            'CAPTION': 1,
            'DETAILED_CAPTION': 2,
            'MORE_DETAILED_CAPTION': 3,
            'OCR': 2,
            'OCR_WITH_REGION': 3,
            'OD': 4,
            'OPEN_VOCABULARY_DETECTION': 5,
            'REGION_PROPOSAL': 3,
            'REGION_TO_CATEGORY': 2,
            'REGION_TO_DESCRIPTION': 3,
            'REGION_TO_SEGMENTATION': 5,
            'REFERRING_EXPRESSION_SEGMENTATION': 6,
            'CAPTION_TO_PHRASE_GROUNDING': 4,
            'DENSE_REGION_CAPTION': 5
        }
    
    def select_task(self, epoch: Optional[int] = None) -> str:
        """选择下一个任务
        
        Args:
            epoch: 当前训练轮次
            
        Returns:
            选择的任务类型
        """
        self.current_step += 1
        
        # 如果只有一个任务类型，直接返回
        if len(self.task_types) == 1:
            task = self.task_types[0]
        elif self.config.strategy == "round_robin":
            task = self._round_robin_selection()
        elif self.config.strategy == "weighted":
            task = self._weighted_selection()
        elif self.config.strategy == "curriculum":
            task = self._curriculum_selection(epoch)
        elif self.config.strategy == "adaptive":
            task = self._adaptive_selection()
        else:
            # 默认使用轮询
            task = self._round_robin_selection()
        
        # 记录选择历史
        self.task_history.append(task)
        self.task_counts[task] += 1
        
        return task
    
    def _round_robin_selection(self) -> str:
        """轮询选择任务"""
        task_index = (self.current_step - 1) % len(self.task_types)
        return self.task_types[task_index]
    
    def _weighted_selection(self) -> str:
        """基于权重的随机选择"""
        tasks = list(self.task_weights.keys())
        weights = list(self.task_weights.values())
        
        # 应用温度参数
        if self.config.temperature != 1.0:
            weights = [w ** (1.0 / self.config.temperature) for w in weights]
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(tasks)] * len(tasks)
        
        return np.random.choice(tasks, p=weights)
    
    def _curriculum_selection(self, epoch: Optional[int] = None) -> str:
        """课程学习任务选择"""
        if epoch is None:
            epoch = self.current_step // 1000  # 估算轮次
        
        # 确定当前课程阶段
        if epoch < self.config.curriculum_start_epoch:
            # 课程学习开始前，使用轮询
            return self._round_robin_selection()
        elif epoch >= self.config.curriculum_end_epoch:
            # 课程学习结束后，使用加权选择
            return self._weighted_selection()
        else:
            # 课程学习阶段
            progress = (epoch - self.config.curriculum_start_epoch) / (
                self.config.curriculum_end_epoch - self.config.curriculum_start_epoch
            )
            
            # 根据进度确定可用任务数量
            num_available_tasks = max(
                1,
                int(len(self.curriculum_tasks) * progress) + 1
            )
            
            available_tasks = self.curriculum_tasks[:num_available_tasks]
            
            # 从可用任务中随机选择
            return random.choice(available_tasks)
    
    def _adaptive_selection(self) -> str:
        """自适应任务选择
        
        基于任务性能动态调整权重
        """
        # 如果没有性能数据，使用加权选择
        if not self.task_performance:
            return self._weighted_selection()
        
        # 计算每个任务的平均性能
        task_avg_performance = {}
        for task, performances in self.task_performance.items():
            if performances:
                task_avg_performance[task] = np.mean(performances[-10:])  # 最近10次的平均值
            else:
                task_avg_performance[task] = 0.0
        
        # 根据性能调整权重（性能差的任务获得更高权重）
        if task_avg_performance:
            max_loss = max(task_avg_performance.values())
            adjusted_weights = {}
            for task in self.task_types:
                if task in task_avg_performance:
                    # 性能差的任务权重更高
                    adjusted_weights[task] = max_loss - task_avg_performance[task] + 0.1
                else:
                    adjusted_weights[task] = 1.0
            
            # 临时更新权重进行选择
            original_weights = self.task_weights.copy()
            self.task_weights = adjusted_weights
            selected_task = self._weighted_selection()
            self.task_weights = original_weights
            
            return selected_task
        else:
            return self._weighted_selection()
    
    def update_task_performance(self, task_type: str, loss: float) -> None:
        """更新任务性能
        
        Args:
            task_type: 任务类型
            loss: 任务损失值
        """
        self.task_performance[task_type].append(loss)
        
        # 限制历史记录长度
        if len(self.task_performance[task_type]) > 100:
            self.task_performance[task_type] = self.task_performance[task_type][-100:]
    
    def update_task_weights(self, new_weights: Dict[str, float]) -> None:
        """更新任务权重
        
        Args:
            new_weights: 新的任务权重
        """
        self.task_weights.update(new_weights)
        logger.info(f"任务权重已更新: {self.task_weights}")
    
    def get_task_distribution(self, window_size: int = 100) -> Dict[str, float]:
        """获取最近的任务分布
        
        Args:
            window_size: 统计窗口大小
            
        Returns:
            任务分布字典
        """
        if not self.task_history:
            return {task: 0.0 for task in self.task_types}
        
        recent_history = list(self.task_history)[-window_size:]
        task_counts = defaultdict(int)
        
        for task in recent_history:
            task_counts[task] += 1
        
        total_count = len(recent_history)
        return {
            task: count / total_count
            for task, count in task_counts.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息
        
        Returns:
            统计信息字典
        """
        total_selections = sum(self.task_counts.values())
        
        return {
            "current_step": self.current_step,
            "strategy": self.config.strategy,
            "task_weights": self.task_weights.copy(),
            "task_counts": dict(self.task_counts),
            "task_distribution": {
                task: count / total_selections if total_selections > 0 else 0.0
                for task, count in self.task_counts.items()
            },
            "recent_distribution": self.get_task_distribution(),
            "curriculum_stage": self.curriculum_stage,
            "performance_available": bool(self.task_performance)
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.current_step = 0
        self.task_history.clear()
        self.task_performance.clear()
        self.task_counts.clear()
        self.curriculum_stage = 0
        
        logger.info("调度器统计信息已重置")
    
    def save_state(self) -> Dict[str, Any]:
        """保存调度器状态
        
        Returns:
            状态字典
        """
        return {
            "task_types": self.task_types,
            "config": self.config.to_dict(),
            "task_weights": self.task_weights,
            "current_step": self.current_step,
            "task_history": list(self.task_history),
            "task_performance": dict(self.task_performance),
            "task_counts": dict(self.task_counts),
            "curriculum_stage": self.curriculum_stage,
            "curriculum_tasks": self.curriculum_tasks
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """加载调度器状态
        
        Args:
            state: 状态字典
        """
        self.task_types = state["task_types"]
        self.config = TaskSchedulingConfig.from_dict(state["config"])
        self.task_weights = state["task_weights"]
        self.current_step = state["current_step"]
        self.task_history = deque(state["task_history"], maxlen=1000)
        self.task_performance = defaultdict(list, state["task_performance"])
        self.task_counts = defaultdict(int, state["task_counts"])
        self.curriculum_stage = state["curriculum_stage"]
        self.curriculum_tasks = state["curriculum_tasks"]
        
        logger.info("调度器状态已加载")
    
    def should_update_weights(self) -> bool:
        """判断是否应该更新权重
        
        Returns:
            是否应该更新
        """
        return (
            self.config.update_frequency > 0 and
            self.current_step % self.config.update_frequency == 0
        )
    
    def auto_adjust_weights(self, performance_threshold: float = 0.1) -> None:
        """自动调整任务权重
        
        Args:
            performance_threshold: 性能差异阈值
        """
        if not self.task_performance:
            return
        
        # 计算每个任务的平均性能
        avg_performances = {}
        for task, performances in self.task_performance.items():
            if performances:
                avg_performances[task] = np.mean(performances[-20:])  # 最近20次
        
        if len(avg_performances) < 2:
            return
        
        # 找出性能最好和最差的任务
        best_performance = min(avg_performances.values())  # 损失越小越好
        worst_performance = max(avg_performances.values())
        
        # 如果性能差异超过阈值，调整权重
        if worst_performance - best_performance > performance_threshold:
            new_weights = {}
            for task in self.task_types:
                if task in avg_performances:
                    # 性能差的任务获得更高权重
                    performance_ratio = avg_performances[task] / best_performance
                    new_weights[task] = min(performance_ratio, 3.0)  # 限制最大权重
                else:
                    new_weights[task] = 1.0
            
            self.update_task_weights(new_weights)
            logger.info(f"自动调整权重完成，性能差异: {worst_performance - best_performance:.4f}")