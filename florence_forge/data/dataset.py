# """FlorenceForge多任务数据集模块

# 提供多任务数据集的加载、处理和采样功能
# """

# import json
# import random
# import logging
# from collections import defaultdict
# from typing import Optional, Dict, Any, List, Union, Tuple
# from pathlib import Path
# from torch.utils.data import Dataset
# from PIL import Image

# try:
#     from ..core.tasks import FLORENCE2_TASKS, validate_task_name
# except ImportError:
#     # 兼容性处理：如果相对导入失败，尝试绝对导入
#     try:
#         from florence_forge.core.tasks import FLORENCE2_TASKS, validate_task_name
#     except ImportError:
#         # 如果都失败，创建默认配置
#         FLORENCE2_TASKS = {}
#         def validate_task_name(task_name):
#             """验证任务名称是否有效
            
#             Args:
#                 task_name (str): 要验证的任务名称
                
#             Returns:
#                 bool: 如果任务名称在支持的任务列表中返回True，否则返回False
                
#             Note:
#                 这是一个兼容性函数，当FLORENCE2_TASKS不可用时提供基本的任务验证
#             """
#             return task_name in FLORENCE2_TASKS

# try:
#     from ..core.config import DataConfig
# except ImportError:
#     try:
#         from florence_forge.core.config import DataConfig
#     except ImportError:
#         # 创建一个简单的DataConfig类
#         class DataConfig:
#             def __init__(self, **kwargs):
#                 """初始化数据配置对象
                
#                 Args:
#                     **kwargs: 任意关键字参数，将作为配置属性设置到对象上
                    
#                 Note:
#                     这是一个简化的DataConfig实现，用于在无法导入完整配置模块时提供基本功能
#                 """
#                 for k, v in kwargs.items():
#                     setattr(self, k, v)

# logger = logging.getLogger(__name__)

# class TaskSample:
#     """任务样本数据结构"""
    
#     def __init__(
#         self,
#         task_type: str,
#         image_path: str,
#         prefix: str,
#         suffix: str,
#         weight: float = 1.0,
#         metadata: Optional[Dict[str, Any]] = None
#     ):
#         self.task_type = task_type
#         self.image_path = image_path
#         self.prefix = prefix
#         self.suffix = suffix
#         self.weight = weight
#         self.metadata = metadata or {}
    
#     def to_dict(self) -> Dict[str, Any]:
#         """转换为字典格式"""
#         return {
#             "task_type": self.task_type,
#             "image_path": self.image_path,
#             "prefix": self.prefix,
#             "suffix": self.suffix,
#             "weight": self.weight,
#             "metadata": self.metadata
#         }
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'TaskSample':
#         """从字典创建样本"""
#         return cls(
#             task_type=data["task_type"],
#             image_path=data["image_path"],
#             prefix=data["prefix"],
#             suffix=data["suffix"],
#             weight=data.get("weight", 1.0),
#             metadata=data.get("metadata", {})
#         )

# class MultiTaskDataset(Dataset):
#     """多任务数据集
    
#     支持多种任务类型的数据加载和处理
#     """
    
#     def __init__(
#         self,
#         data_configs: List[Dict[str, Any]],
#         image_base_path: str = "",
#         config: Optional[DataConfig] = None,
#         processor=None
#     ):
#         """初始化多任务数据集
        
#         Args:
#             data_configs: 数据配置列表，每个配置包含任务类型和数据路径
#             image_base_path: 图像文件基础路径
#             config: 数据配置
#             processor: 数据处理器
#         """
#         self.data_configs = data_configs
#         self.image_base_path = Path(image_base_path)
#         self.config = config or DataConfig()
#         self.processor = processor
        
#         self.samples: List[TaskSample] = []
#         self.task_weights: Dict[str, float] = {}
#         self.task_indices: Dict[str, List[int]] = defaultdict(list)
        
#         self._validate_configs()
#         self._load_all_tasks()
#         self._calculate_task_weights()
#         self._build_task_indices()
        
#         logger.info(f"数据集初始化完成，总样本数: {len(self.samples)}")
    
#     def _validate_configs(self) -> None:
#         """验证数据配置"""
#         for i, config in enumerate(self.data_configs):
#             if "task_type" not in config:
#                 raise ValueError(f"配置 {i} 缺少 task_type 字段")
#             if "data_path" not in config:
#                 raise ValueError(f"配置 {i} 缺少 data_path 字段")
            
#             task_type = config["task_type"]
#             if not validate_task_name(task_type):
#                 raise ValueError(f"未知任务类型: {task_type}")
            
#             data_path = Path(config["data_path"])
#             if not data_path.exists():
#                 raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
#     def _load_all_tasks(self) -> None:
#         """加载所有任务数据"""
#         task_counts = defaultdict(int)
        
#         for config in self.data_configs:
#             task_type = config["task_type"]
#             data_path = config["data_path"]
#             weight = config.get("weight", 1.0)
            
#             logger.info(f"正在加载任务: {task_type}, 路径: {data_path}")
            
#             samples_loaded = self._load_task_data(task_type, data_path, weight)
#             task_counts[task_type] += samples_loaded
        
#         logger.info(f"数据加载完成，各任务样本数: {dict(task_counts)}")
    
#     def _load_task_data(self, task_type: str, data_path: str, weight: float) -> int:
#         """加载单个任务的数据
        
#         Args:
#             task_type: 任务类型
#             data_path: 数据文件路径
#             weight: 任务权重
            
#         Returns:
#             加载的样本数量
#         """
#         samples_loaded = 0
#         max_samples = self.config.max_samples_per_task
        
#         try:
#             with open(data_path, 'r', encoding='utf-8') as f:
#                 for line_num, line in enumerate(f, 1):
#                     if max_samples and samples_loaded >= max_samples:
#                         break
                    
#                     try:
#                         data = json.loads(line.strip())
                        
#                         # 构建图像路径
#                         image_path = self.image_base_path / data["image"]
                        
#                         # 创建样本
#                         sample = TaskSample(
#                             task_type=task_type,
#                             image_path=str(image_path),
#                             prefix=data["prefix"],
#                             suffix=data["suffix"],
#                             weight=weight,
#                             metadata={
#                                 "source_file": data_path,
#                                 "line_number": line_num
#                             }
#                         )
                        
#                         self.samples.append(sample)
#                         samples_loaded += 1
                        
#                     except json.JSONDecodeError as e:
#                         logger.warning(f"解析JSON失败 {data_path}:{line_num}: {e}")
#                         continue
#                     except KeyError as e:
#                         logger.warning(f"缺少必要字段 {data_path}:{line_num}: {e}")
#                         continue
        
#         except Exception as e:
#             logger.error(f"加载任务数据失败 {task_type}: {e}")
#             raise
        
#         return samples_loaded
    
#     def _calculate_task_weights(self) -> None:
#         """计算任务权重以实现平衡采样"""
#         if not self.config.use_balanced_sampling:
#             return
        
#         task_counts = defaultdict(int)
#         for sample in self.samples:
#             task_counts[sample.task_type] += 1
        
#         if not task_counts:
#             return
        
#         max_count = max(task_counts.values())
#         for task_type, count in task_counts.items():
#             self.task_weights[task_type] = max_count / count
        
#         logger.info(f"任务权重: {self.task_weights}")
    
#     def _build_task_indices(self) -> None:
#         """构建任务索引映射"""
#         for idx, sample in enumerate(self.samples):
#             self.task_indices[sample.task_type].append(idx)
    
#     def __len__(self) -> int:
#         """返回数据集大小"""
#         return len(self.samples)
    
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """获取单个样本
        
#         Args:
#             idx: 样本索引
            
#         Returns:
#             处理后的样本数据
#         """
#         sample = self.samples[idx]
        
#         # 加载图像
#         try:
#             image = Image.open(sample.image_path).convert('RGB')
#         except Exception as e:
#             logger.warning(f"无法加载图像 {sample.image_path}: {e}")
#             # 返回空白图像作为备选
#             image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
#         # 构建提示和答案
#         prompt, answer = self._build_prompt_and_answer(sample)
        
#         # 如果有处理器，进行预处理
#         if self.processor is not None:
#             # Florence-2的processor只需要任务token，不需要前缀
#             task_config = FLORENCE2_TASKS[sample.task_type]
#             task_token = task_config["prompt"]
            
#             processed = self.processor(
#                 text=task_token,
#                 images=image,
#                 return_tensors="pt"
#             )
#             # 移除批次维度（因为processor返回的是批次格式）
#             for key, value in processed.items():
#                 if hasattr(value, 'squeeze'):
#                     processed[key] = value.squeeze(0)
            
#             result = {
#                 "prompt": prompt,
#                 "answer": answer,
#                 "task_type": sample.task_type,
#                 "weight": sample.weight,
#                 "metadata": sample.metadata
#             }
#             result.update(processed)
#         else:
#             # 如果没有处理器，保留原始格式
#             result = {
#                 "image": image,
#                 "prompt": prompt,
#                 "answer": answer,
#                 "task_type": sample.task_type,
#                 "weight": sample.weight,
#                 "metadata": sample.metadata
#             }
        
#         return result
    
#     def _build_prompt_and_answer(self, sample: TaskSample) -> Tuple[str, str]:
#         """构建提示和答案
        
#         Args:
#             sample: 任务样本
            
#         Returns:
#             (提示, 答案) 元组
#         """
#         task_config = FLORENCE2_TASKS[sample.task_type]
#         task_prompt = task_config["prompt"]
        
#         # 构建完整提示
#         if sample.prefix:
#             prompt = f"{task_prompt}{sample.prefix}"
#         else:
#             prompt = task_prompt
        
#         # 答案就是suffix
#         answer = sample.suffix
        
#         return prompt, answer
    
#     def get_task_samples(self, task_type: str) -> List[int]:
#         """获取指定任务的样本索引
        
#         Args:
#             task_type: 任务类型
            
#         Returns:
#             样本索引列表
#         """
#         return self.task_indices.get(task_type, [])
    
#     def get_task_statistics(self) -> Dict[str, Any]:
#         """获取任务统计信息
        
#         Returns:
#             统计信息字典
#         """
#         task_counts = defaultdict(int)
#         for sample in self.samples:
#             task_counts[sample.task_type] += 1
        
#         return {
#             "total_samples": len(self.samples),
#             "task_counts": dict(task_counts),
#             "task_weights": self.task_weights,
#             "num_tasks": len(task_counts)
#         }
    
#     def sample_by_task(self, task_type: str, num_samples: int) -> List[int]:
#         """按任务类型采样
        
#         Args:
#             task_type: 任务类型
#             num_samples: 采样数量
            
#         Returns:
#             采样的索引列表
#         """
#         task_indices = self.get_task_samples(task_type)
#         if not task_indices:
#             return []
        
#         if num_samples >= len(task_indices):
#             return task_indices.copy()
        
#         return random.sample(task_indices, num_samples)
    
#     def create_subset(self, indices: List[int]) -> 'MultiTaskDataset':
#         """创建子集
        
#         Args:
#             indices: 样本索引列表
            
#         Returns:
#             子数据集
#         """
#         subset_samples = [self.samples[i] for i in indices]
        
#         # 创建新的数据集实例
#         subset = MultiTaskDataset.__new__(MultiTaskDataset)
#         subset.data_configs = self.data_configs
#         subset.image_base_path = self.image_base_path
#         subset.config = self.config
#         subset.processor = self.processor
#         subset.samples = subset_samples
#         subset.task_weights = self.task_weights.copy()
        
#         # 重新构建任务索引
#         subset.task_indices = defaultdict(list)
#         subset._build_task_indices()
        
#         return subset
    
#     def save_to_file(self, file_path: Union[str, Path]) -> None:
#         """保存数据集到文件
        
#         Args:
#             file_path: 文件路径
#         """
#         file_path = Path(file_path)
#         file_path.parent.mkdir(parents=True, exist_ok=True)
        
#         data = {
#             "data_configs": self.data_configs,
#             "image_base_path": str(self.image_base_path),
#             "config": self.config.to_dict(),
#             "samples": [sample.to_dict() for sample in self.samples],
#             "task_weights": self.task_weights
#         }
        
#         with open(file_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"数据集已保存到: {file_path}")
    
#     @classmethod
#     def load_from_file(
#         cls, 
#         file_path: Union[str, Path], 
#         processor=None
#     ) -> 'MultiTaskDataset':
#         """从文件加载数据集
        
#         Args:
#             file_path: 文件路径
#             processor: 数据处理器
            
#         Returns:
#             数据集实例
#         """
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 创建新实例
#         dataset = cls.__new__(cls)
#         dataset.data_configs = data["data_configs"]
#         dataset.image_base_path = Path(data["image_base_path"])
#         dataset.config = DataConfig.from_dict(data["config"])
#         dataset.processor = processor
#         dataset.samples = [TaskSample.from_dict(s) for s in data["samples"]]
#         dataset.task_weights = data["task_weights"]
        
#         # 重新构建任务索引
#         dataset.task_indices = defaultdict(list)
#         dataset._build_task_indices()
        
#         logger.info(f"数据集已从文件加载: {file_path}")
#         return dataset


"""FlorenceForge多任务数据集模块

提供多任务数据集的加载、处理和采样功能
"""

import json
import random
import logging
import torch
from collections import defaultdict
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

try:
    from ..core.tasks import FLORENCE2_TASKS, validate_task_name
except ImportError:
    # 兼容性处理：如果相对导入失败，尝试绝对导入
    try:
        from florence_forge.core.tasks import FLORENCE2_TASKS, validate_task_name
    except ImportError:
        # 如果都失败，创建默认配置
        FLORENCE2_TASKS = {}
        def validate_task_name(task_name):
            """验证任务名称是否有效
            
            Args:
                task_name (str): 要验证的任务名称
                
            Returns:
                bool: 如果任务名称在支持的任务列表中返回True，否则返回False
                
            Note:
                这是一个兼容性函数，当FLORENCE2_TASKS不可用时提供基本的任务验证
            """
            return task_name in FLORENCE2_TASKS

try:
    from ..core.config import DataConfig
except ImportError:
    try:
        from florence_forge.core.config import DataConfig
    except ImportError:
        # 创建一个简单的DataConfig类
        class DataConfig:
            def __init__(self, **kwargs):
                """初始化数据配置对象
                
                Args:
                    **kwargs: 任意关键字参数，将作为配置属性设置到对象上
                    
                Note:
                    这是一个简化的DataConfig实现，用于在无法导入完整配置模块时提供基本功能
                """
                for k, v in kwargs.items():
                    setattr(self, k, v)

logger = logging.getLogger(__name__)

class TaskSample:
    """任务样本数据结构"""
    
    def __init__(
        self,
        task_type: str,
        image_path: str,
        prefix: str,
        suffix: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_type = task_type
        self.image_path = image_path
        self.prefix = prefix
        self.suffix = suffix
        self.weight = weight
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_type": self.task_type,
            "image_path": self.image_path,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "weight": self.weight,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSample':
        """从字典创建样本"""
        return cls(
            task_type=data["task_type"],
            image_path=data["image_path"],
            prefix=data["prefix"],
            suffix=data["suffix"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {})
        )

class MultiTaskDataset(Dataset):
    """多任务数据集
    
    支持多种任务类型的数据加载和处理
    """
    
    def __init__(
        self,
        data_configs: List[Dict[str, Any]],
        image_base_path: str = "",
        config: Optional[DataConfig] = None,
        processor=None
    ):
        """初始化多任务数据集
        
        Args:
            data_configs: 数据配置列表，每个配置包含任务类型和数据路径
            image_base_path: 图像文件基础路径
            config: 数据配置
            processor: 数据处理器
        """
        self.data_configs = data_configs
        self.image_base_path = Path(image_base_path)
        self.config = config or DataConfig()
        self.processor = processor
        
        self.samples: List[TaskSample] = []
        self.task_weights: Dict[str, float] = {}
        self.task_indices: Dict[str, List[int]] = defaultdict(list)
        
        self._validate_configs()
        self._load_all_tasks()
        self._calculate_task_weights()
        self._build_task_indices()
        
        logger.info(f"数据集初始化完成，总样本数: {len(self.samples)}")
    
    def _validate_configs(self) -> None:
        """验证数据配置"""
        for i, config in enumerate(self.data_configs):
            if "task_type" not in config:
                raise ValueError(f"配置 {i} 缺少 task_type 字段")
            if "data_path" not in config:
                raise ValueError(f"配置 {i} 缺少 data_path 字段")
            
            task_type = config["task_type"]
            if not validate_task_name(task_type):
                raise ValueError(f"未知任务类型: {task_type}")
            
            data_path = Path(config["data_path"])
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    def _load_all_tasks(self) -> None:
        """加载所有任务数据"""
        task_counts = defaultdict(int)
        
        for config in self.data_configs:
            task_type = config["task_type"]
            data_path = config["data_path"]
            weight = config.get("weight", 1.0)
            
            logger.info(f"正在加载任务: {task_type}, 路径: {data_path}")
            
            samples_loaded = self._load_task_data(task_type, data_path, weight)
            task_counts[task_type] += samples_loaded
        
        logger.info(f"数据加载完成，各任务样本数: {dict(task_counts)}")
    
    def _load_task_data(self, task_type: str, data_path: str, weight: float) -> int:
        """加载单个任务的数据
        
        Args:
            task_type: 任务类型
            data_path: 数据文件路径
            weight: 任务权重
            
        Returns:
            加载的样本数量
        """
        samples_loaded = 0
        max_samples = self.config.max_samples_per_task
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and samples_loaded >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # 构建图像路径
                        image_path = self.image_base_path / data["image"]
                        
                        # 创建样本
                        sample = TaskSample(
                            task_type=task_type,
                            image_path=str(image_path),
                            prefix=data["prefix"],
                            suffix=data["suffix"],
                            weight=weight,
                            metadata={
                                "source_file": data_path,
                                "line_number": line_num
                            }
                        )
                        
                        self.samples.append(sample)
                        samples_loaded += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON失败 {data_path}:{line_num}: {e}")
                        continue
                    except KeyError as e:
                        logger.warning(f"缺少必要字段 {data_path}:{line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"加载任务数据失败 {task_type}: {e}")
            raise
        
        return samples_loaded
    
    def _calculate_task_weights(self) -> None:
        """计算任务权重以实现平衡采样"""
        if not self.config.use_balanced_sampling:
            return
        
        task_counts = defaultdict(int)
        for sample in self.samples:
            task_counts[sample.task_type] += 1
        
        if not task_counts:
            return
        
        max_count = max(task_counts.values())
        for task_type, count in task_counts.items():
            self.task_weights[task_type] = max_count / count
        
        logger.info(f"任务权重: {self.task_weights}")
    
    def _build_task_indices(self) -> None:
        """构建任务索引映射"""
        for idx, sample in enumerate(self.samples):
            self.task_indices[sample.task_type].append(idx)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            处理后的样本数据
        """
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图像 {sample.image_path}: {e}")
            # 返回空白图像作为备选
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # 构建提示和答案
        prompt, answer = self._build_prompt_and_answer(sample)
        
        # 如果有处理器，进行预处理
        if self.processor is not None:
            # Florence-2的processor只需要任务token，不需要前缀
            task_config = FLORENCE2_TASKS[sample.task_type]
            task_token = task_config["prompt"]
            
            processed = self.processor(
                text=task_token,
                images=image,
                return_tensors="pt"
            )
            # 移除批次维度（因为processor返回的是批次格式）
            for key, value in processed.items():
                if hasattr(value, 'squeeze'):
                    processed[key] = value.squeeze(0)
            
            result = {
                "prompt": prompt,
                "answer": answer,
                "task_type": sample.task_type,
                "weight": sample.weight,
                "metadata": sample.metadata
            }
            result.update(processed)
        else:
            # 如果没有处理器，保留原始格式
            result = {
                "image": image,
                "prompt": prompt,
                "answer": answer,
                "task_type": sample.task_type,
                "weight": sample.weight,
                "metadata": sample.metadata
            }
        
        return result
    
    def _build_prompt_and_answer(self, sample: TaskSample) -> Tuple[str, str]:
        """构建提示和答案
        
        Args:
            sample: 任务样本
            
        Returns:
            (提示, 答案) 元组
        """
        task_config = FLORENCE2_TASKS[sample.task_type]
        task_prompt = task_config["prompt"]
        
        # 构建完整提示
        if sample.prefix:
            prompt = f"{task_prompt}{sample.prefix}"
        else:
            prompt = task_prompt
        
        # 答案就是suffix
        answer = sample.suffix
        
        return prompt, answer
    
    def get_task_samples(self, task_type: str) -> List[int]:
        """获取指定任务的样本索引
        
        Args:
            task_type: 任务类型
            
        Returns:
            样本索引列表
        """
        return self.task_indices.get(task_type, [])
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息
        
        Returns:
            统计信息字典
        """
        task_counts = defaultdict(int)
        for sample in self.samples:
            task_counts[sample.task_type] += 1
        
        return {
            "total_samples": len(self.samples),
            "task_counts": dict(task_counts),
            "task_weights": self.task_weights,
            "num_tasks": len(task_counts)
        }
    
    def sample_by_task(self, task_type: str, num_samples: int) -> List[int]:
        """按任务类型采样
        
        Args:
            task_type: 任务类型
            num_samples: 采样数量
            
        Returns:
            采样的索引列表
        """
        task_indices = self.get_task_samples(task_type)
        if not task_indices:
            return []
        
        if num_samples >= len(task_indices):
            return task_indices.copy()
        
        return random.sample(task_indices, num_samples)
    
    def create_subset(self, indices: List[int]) -> 'MultiTaskDataset':
        """创建子集
        
        Args:
            indices: 样本索引列表
            
        Returns:
            子数据集
        """
        subset_samples = [self.samples[i] for i in indices]
        
        # 创建新的数据集实例
        subset = MultiTaskDataset.__new__(MultiTaskDataset)
        subset.data_configs = self.data_configs
        subset.image_base_path = self.image_base_path
        subset.config = self.config
        subset.processor = self.processor
        subset.samples = subset_samples
        subset.task_weights = self.task_weights.copy()
        
        # 重新构建任务索引
        subset.task_indices = defaultdict(list)
        subset._build_task_indices()
        
        return subset
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存数据集到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "data_configs": self.data_configs,
            "image_base_path": str(self.image_base_path),
            "config": self.config.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "task_weights": self.task_weights
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集已保存到: {file_path}")
    
    @classmethod
    def load_from_file(
        cls, 
        file_path: Union[str, Path], 
        processor=None
    ) -> 'MultiTaskDataset':
        """从文件加载数据集
        
        Args:
            file_path: 文件路径
            processor: 数据处理器
            
        Returns:
            数据集实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建新实例
        dataset = cls.__new__(cls)
        dataset.data_configs = data["data_configs"]
        dataset.image_base_path = Path(data["image_base_path"])
        dataset.config = DataConfig.from_dict(data["config"])
        dataset.processor = processor
        dataset.samples = [TaskSample.from_dict(s) for s in data["samples"]]
        dataset.task_weights = data["task_weights"]
        
        # 重新构建任务索引
        dataset.task_indices = defaultdict(list)
        dataset._build_task_indices()
        
        logger.info(f"数据集已从文件加载: {file_path}")
        return dataset

class Florence2Dataset(MultiTaskDataset):
    """Florence-2特定任务的数据集"""

    @staticmethod
    def preprocess_for_inference(
        images: Union[Image.Image, List[Image.Image]],
        task_prompt: str,
        processor: Any,
        text_input: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """为单张或批量图像进行推理预处理

        Args:
            images: 单张PIL图像或图像列表
            task_prompt: 任务提示，例如 '<OD>'
            processor: Hugging Face处理器
            text_input: 额外的文本输入，用于需要文本条件的任务

        Returns:
            一个包含 'pixel_values' 和 'input_ids' 的字典
        """
        if not isinstance(images, list):
            images = [images]

        # 根据任务类型构建最终的文本提示
        if text_input:
            # 对于需要额外文本输入的任务，将其附加到任务提示后
            # 确保格式与模型训练时一致
            full_prompt = f"{task_prompt}{text_input}"
        else:
            full_prompt = task_prompt

        # 使用处理器进行预处理
        # `padding="longest"` 和 `truncation=True` 是推荐的最佳实践
        inputs = processor(
            text=[full_prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024  # 根据需要调整
        )

        return inputs