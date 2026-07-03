"""FlorenceForge多任务评估器模块

提供完整的多任务模型评估功能
"""

import json
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, Union, List
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import PIL

from ..core.tasks import FLORENCE2_TASKS
from ..data.collate import Florence2Collator
from ..data.dataset import MultiTaskDataset
from .metrics import get_metric_calculator
from .evaluator_reporting import (
    compute_overall_metrics,
    export_bad_cases_to_jsonl,
    infer_case_score,
    iter_result_items,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)

class MultiTaskEvaluator:
    """多任务评估器
    
    提供完整的多任务模型评估功能
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """初始化评估器

        Args:
            model: 多任务模型（需实现 generate 和 processor 接口）
            device: 计算设备
        """
        self.model = model
        self.device = self._resolve_device(device)

        # 运行时检查模型是否具备评估所需的接口
        if not hasattr(model, 'generate'):
            raise TypeError(f"评估器要求模型实现 generate() 方法，但 {type(model).__name__} 未提供")
        if not callable(self._safe_getattr(model, 'decode')) and self._safe_getattr(model, 'processor') is None:
            raise TypeError(
                f"评估器要求模型实现 decode() 方法或具备 processor 属性，"
                f"但 {type(model).__name__} 未提供"
            )

        # 将模型移到指定设备。对于 accelerate device_map/offload 的模型，
        # 再次调用 .to(...) 会直接报错，此时保留其当前分发状态即可。
        if self._can_move_model(model):
            self.model.to(self.device)
        else:
            logger.info("检测到 accelerate 分发/卸载模型，跳过显式 model.to(device)")

        # 评估结果存储
        self.evaluation_results = {}
        self.task_metrics = {}

        logger.info(f"多任务评估器初始化完成，使用设备: {self.device}")

    def _can_move_model(self, model: nn.Module) -> bool:
        """判断模型是否适合显式调用 ``.to(device)``。"""
        inner_model = self._safe_getattr(model, "model", None)
        candidates = [m for m in (model, inner_model) if m is not None]
        for candidate in candidates:
            hf_device_map = self._safe_getattr(candidate, "hf_device_map", None)
            if hf_device_map:
                return False
            if self._safe_getattr(candidate, "_hf_hook", None) is not None:
                return False
        return True

    def _safe_getattr(self, obj: Any, name: str, default: Any = None) -> Any:
        """安全获取属性，避免未加载 processor 等 RuntimeError 打断接口检测。"""
        try:
            return getattr(obj, name)
        except (AttributeError, RuntimeError):
            return default

    def _decode_token_ids(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """解码 token ids，优先使用模型级 decode，回退到 processor.batch_decode。"""
        decode = self._safe_getattr(self.model, 'decode')
        if callable(decode):
            return decode(token_ids, skip_special_tokens=skip_special_tokens)

        processor = self._safe_getattr(self.model, 'processor')
        if processor is not None and hasattr(processor, 'batch_decode'):
            return processor.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

        raise RuntimeError("模型未提供 decode() 或 processor.batch_decode()，无法解码评估结果")

    def _should_preserve_special_tokens(self, task_type: str) -> bool:
        """检测/区域类任务需要保留 ``<loc_*>`` 等特殊 token 才能正确评估。"""
        if not isinstance(task_type, str):
            return False
        task_upper = task_type.upper()
        return task_upper in {
            "OD",
            "OD_VP",
            "REGION_PROPOSAL",
            "CAPTION_TO_PHRASE_GROUNDING",
            "PHRASE_GROUNDING_VP",
        }

    def _decode_by_task_types(
        self,
        token_ids: torch.Tensor,
        task_types: List[str],
    ) -> List[str]:
        """按任务逐样本解码，确保 OD 等任务保留 loc tokens。"""
        if not isinstance(token_ids, torch.Tensor):
            return []
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        decoded: List[str] = []
        for index, task_type in enumerate(task_types):
            sample_ids = token_ids[index].unsqueeze(0)
            decoded.extend(
                self._decode_token_ids(
                    sample_ids,
                    skip_special_tokens=not self._should_preserve_special_tokens(task_type),
                )
            )
        return decoded

    def _get_batch_task_types(self, batch: Dict[str, Any]) -> List[str]:
        """返回与 batch size 对齐的任务类型列表。"""
        task_types = batch.get('task_types')
        if task_types:
            return list(task_types)

        task_type = batch.get('task_type')
        if isinstance(task_type, list):
            return task_type
        if isinstance(task_type, str):
            batch_size = batch['input_ids'].shape[0]
            return [task_type] * batch_size
        return []

    def _get_reference_ids(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """获取参考答案 token，避免在 Tensor 上使用布尔 or。"""
        reference_ids = batch.get('reference_ids')
        if reference_ids is None:
            reference_ids = batch.get('labels')
        if isinstance(reference_ids, torch.Tensor):
            pad_token_id = self._resolve_pad_token_id()
            reference_ids = reference_ids.clone()
            reference_ids[reference_ids == -100] = pad_token_id
        return reference_ids

    def _get_generation_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """评估生成优先使用仅 prompt 输入，避免把 reference token 一起喂给模型。"""
        input_ids = batch.get("prompt_input_ids")
        attention_mask = batch.get("prompt_attention_mask")
        if input_ids is None:
            input_ids = batch["input_ids"]
        if attention_mask is None:
            attention_mask = batch.get("attention_mask")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": batch["pixel_values"],
        }

    def _resolve_pad_token_id(self) -> int:
        """解析模型 processor/tokenizer 的 pad_token_id，找不到时回退 0。"""
        processor = self._safe_getattr(self.model, 'processor')
        tokenizer = getattr(processor, 'tokenizer', None) if processor is not None else None
        pad_token_id = getattr(
            processor,
            'pad_token_id',
            getattr(tokenizer, 'pad_token_id', 0) or 0,
        )
        try:
            return int(pad_token_id)
        except (TypeError, ValueError):
            return 0

    def _extract_generated_tokens(
        self,
        generated_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """只在模型返回完整 prompt+answer 序列时剥离 prompt tokens。

        Encoder-decoder 模型通常只返回新生成 tokens；decoder-only 风格模型可能返回
        prompt + generated tokens。盲目按 input length 裁剪会把前者裁成空序列。
        """
        if not isinstance(generated_ids, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            return generated_ids
        if generated_ids.dim() != 2 or input_ids.dim() != 2:
            return generated_ids

        input_len = input_ids.shape[1]
        if generated_ids.shape[1] <= input_len:
            return generated_ids

        prefix = generated_ids[:, :input_len]
        try:
            input_for_compare = input_ids.to(prefix.device)
            if torch.equal(prefix, input_for_compare):
                return generated_ids[:, input_len:]
        except Exception:
            logger.debug("无法比较生成结果前缀，保留完整生成序列", exc_info=True)

        return generated_ids

    def _resolve_device(self, device) -> torch.device:
        """解析设备参数，支持 auto / 字符串 / torch.device"""
        import torch
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    def _get_collate_fn(self, dataset: MultiTaskDataset):
        """返回数据集 collate_fn，兼容旧数据集对象。

        回退构造 Florence2Collator 时使用模型实际的 pad_token_id，
        避免硬编码 0 在非零 pad token 的模型上导致 padding 错位。
        """
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            return collate_fn
        return Florence2Collator(pad_token_id=self._resolve_pad_token_id())

    def _resolve_num_workers(self, dataset: MultiTaskDataset, num_workers: int) -> int:
        """避免 DataLoader worker 丢失 processor/backend 后返回未编码样本。"""
        if num_workers <= 0:
            return num_workers

        if getattr(dataset, "processor", None) is not None or getattr(dataset, "backend", None) is not None:
            logger.warning(
                "评估数据集依赖 processor/backend 在线编码，已将 num_workers 设为 0，"
                "避免 worker 序列化后丢失编码器。"
            )
            return 0
        return num_workers

    def _make_dataloader(
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """创建评估 DataLoader，并应用与训练数据加载器一致的 worker 安全策略。"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._resolve_num_workers(dataset, num_workers),
            collate_fn=self._get_collate_fn(dataset),
        )

    def _evaluate_batch(
        self, batch: Dict[str, Any]
    ) -> Optional[List[tuple]]:
        """对单个批次执行 生成 → 解码 → 参考解码 的公共流程。

        ``evaluate_dataset`` 与 ``evaluate_task`` 之前各自重复了"移动数据到设备、
        取生成输入、generate、剥离 prompt、按任务解码预测与参考"这一整段逻辑。
        此处统一抽取为单一实现。

        Returns:
            ``(pred_raw, ref_raw, task_type)`` 三元组列表；若该批次缺少参考答案
            （labels/reference_ids），返回 ``None`` 表示调用方应跳过。
        """
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch_task_types = self._get_batch_task_types(batch)

        generation_inputs = self._get_generation_inputs(batch)
        pred_ids = self.model.generate(
            input_ids=generation_inputs['input_ids'],
            pixel_values=generation_inputs['pixel_values'],
            attention_mask=generation_inputs.get('attention_mask'),
            max_new_tokens=512,
            do_sample=False,
        )

        # 仅在生成结果确实包含 prompt 前缀时剥离
        new_tokens = self._extract_generated_tokens(pred_ids, generation_inputs['input_ids'])
        decoded_predictions = self._decode_by_task_types(new_tokens, batch_task_types)

        reference_ids = self._get_reference_ids(batch)
        if reference_ids is None:
            logger.warning("批次缺少 labels/reference_ids，跳过指标计算")
            return None
        decoded_references = self._decode_by_task_types(reference_ids, batch_task_types)

        return list(zip(decoded_predictions, decoded_references, batch_task_types))

    def evaluate_dataset(
        self,
        dataset: MultiTaskDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples_per_task: Optional[int] = None,
        save_predictions: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """评估整个数据集
        
        Args:
            dataset: 评估数据集
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            max_samples_per_task: 每个任务的最大样本数
            save_predictions: 是否保存预测结果
            output_dir: 输出目录
            
        Returns:
            评估结果字典
        """
        logger.info("开始数据集评估...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 创建数据加载器
        dataloader = self._make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        # 按任务组织指标计算器
        task_calculators = {}
        for task_type in dataset.task_indices.keys():
            task_calculators[task_type] = get_metric_calculator(task_type)
        
        # 存储预测结果
        all_predictions = defaultdict(list)
        
        # 评估统计
        total_samples = 0
        task_sample_counts = defaultdict(int)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                decoded = self._evaluate_batch(batch)
                if decoded is None:
                    continue

                # 按任务类型组织结果
                for pred, ref, task_type in decoded:
                    # 检查样本数限制（计数仅在实际纳入统计时增加，跳过不计数）
                    if (max_samples_per_task and 
                        task_sample_counts[task_type] >= max_samples_per_task):
                        continue
                    
                    # 清理预测和参考文本
                    pred_clean = self._clean_prediction(pred, task_type)
                    ref_clean = self._clean_reference(ref, task_type)
                    
                    # 添加到指标计算器
                    if task_type in task_calculators:
                        task_calculators[task_type].add_batch([pred_clean], [ref_clean])
                    
                    # 存储预测结果
                    if save_predictions:
                        all_predictions[task_type].append({
                            'prediction': pred_clean,
                            'reference': ref_clean,
                            'sample_id': total_samples
                        })
                    
                    task_sample_counts[task_type] += 1
                    total_samples += 1
        
        # 计算各任务指标
        task_metrics = {}
        for task_type, calculator in task_calculators.items():
            if task_sample_counts[task_type] > 0:
                metrics = calculator.compute()
                task_metrics[task_type] = {
                    'metrics': metrics,
                    'sample_count': task_sample_counts[task_type]
                }
        
        # 计算总体指标
        overall_metrics = self._compute_overall_metrics(task_metrics)
        
        # 构建评估结果
        evaluation_time = time.time() - start_time
        
        results = {
            'overall_metrics': overall_metrics,
            'task_metrics': task_metrics,
            'evaluation_info': {
                'total_samples': total_samples,
                'task_sample_counts': dict(task_sample_counts),
                'evaluation_time': evaluation_time,
                'device': str(self.device)
            }
        }
        
        # 保存结果
        if output_dir:
            self._save_evaluation_results(results, all_predictions, output_dir)
        
        # 存储到实例变量
        self.evaluation_results = results
        self.task_metrics = task_metrics
        
        logger.info(f"数据集评估完成，耗时: {evaluation_time:.2f}秒")
        
        return results
    
    def evaluate_task(
        self,
        dataset: MultiTaskDataset,
        task_type: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """评估特定任务
        
        Args:
            dataset: 评估数据集
            task_type: 任务类型
            batch_size: 批次大小
            max_samples: 最大样本数
            
        Returns:
            任务评估结果
        """
        logger.info(f"开始评估任务: {task_type}")
        
        # 创建任务特定的子集
        task_subset = dataset.create_task_subset(task_type, max_samples)
        
        if len(task_subset) == 0:
            logger.warning(f"任务 {task_type} 没有可用样本")
            return {}
        
        # 创建数据加载器
        dataloader = self._make_dataloader(
            task_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # 创建指标计算器
        calculator = get_metric_calculator(task_type)
        
        # 设置模型为评估模式
        self.model.eval()
        
        predictions = []
        references = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估 {task_type}"):
                decoded = self._evaluate_batch(batch)
                if decoded is None:
                    continue

                # 清理和收集结果（task_subset 已按任务过滤，忽略逐样本 task_type）
                for pred, ref, _ in decoded:
                    pred_clean = self._clean_prediction(pred, task_type)
                    ref_clean = self._clean_reference(ref, task_type)
                    
                    predictions.append(pred_clean)
                    references.append(ref_clean)
                    
                    calculator.add_batch([pred_clean], [ref_clean])
        
        # 计算指标
        metrics = calculator.compute()
        evaluation_time = time.time() - start_time
        
        results = {
            'task_type': task_type,
            'metrics': metrics,
            'sample_count': len(predictions),
            'evaluation_time': evaluation_time,
            'predictions': predictions[:10],  # 保存前10个预测示例
            'references': references[:10]
        }
        
        logger.info(f"任务 {task_type} 评估完成，耗时: {evaluation_time:.2f}秒")
        
        return results
    
    def _predict_task_compat(
        self,
        image: Union[str, Path, 'PIL.Image.Image'],
        task_type: str,
    ) -> Union[str, List[str]]:
        """以后端无关的方式对单张图片执行任务预测。

        ``predict_task`` 是 Florence2MultiTaskModel 门面提供的便捷接口，但并非
        所有后端（如 PaliGemma / 通用 HF 后端）都实现它。此前直接调用会在这些
        后端上抛出 ``AttributeError``。这里优先使用 ``predict_task``，否则回退到
        标准的 ``generate(images=..., task_prompt=...)`` 接口。
        """
        if hasattr(self.model, "predict_task"):
            return self.model.predict_task(image, task_type)

        # 回退：从任务表解析 prompt，走通用 generate 接口
        loaded_image = image
        if isinstance(image, (str, Path)):
            from PIL import Image as _PILImage

            loaded_image = _PILImage.open(image).convert("RGB")

        task_config = FLORENCE2_TASKS.get(task_type)
        task_prompt = getattr(task_config, "prompt", None) if task_config else None
        if task_prompt is None:
            task_prompt = task_type
        return self.model.generate(images=loaded_image, task_prompt=task_prompt)

    def evaluate_single_sample(
        self,
        image: Union[str, Path, 'PIL.Image.Image'],
        task_type: str,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """评估单个样本
        
        Args:
            image: 输入图像
            task_type: 任务类型
            reference: 参考答案（可选）
            
        Returns:
            评估结果
        """
        # 使用模型进行预测
        prediction = self._predict_task_compat(image, task_type)
        
        result = {
            'task_type': task_type,
            'prediction': prediction,
            'reference': reference
        }
        
        # 如果有参考答案，计算指标
        if reference:
            calculator = get_metric_calculator(task_type)
            
            pred_clean = self._clean_prediction(prediction, task_type)
            ref_clean = self._clean_reference(reference, task_type)
            
            calculator.add_batch([pred_clean], [ref_clean])
            metrics = calculator.compute()
            
            result['metrics'] = metrics
        
        return result

    def export_bad_cases(
        self,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        threshold: float = 0.5,
        output_dir: Union[str, Path] = "bad_cases",
        filename: str = "bad_cases.jsonl",
    ) -> Path:
        """导出低分样本为 JSONL，便于回流标注。

        Args:
            results: per-sample 结果列表，或包含 predictions/references 的评估结果字典
            threshold: 分数小于等于该阈值的样本会被导出
            output_dir: 输出目录
            filename: JSONL 文件名

        Returns:
            写入的 JSONL 文件路径
        """
        return export_bad_cases_to_jsonl(
            results,
            threshold=threshold,
            output_dir=output_dir,
            filename=filename,
        )

    def _iter_result_items(
        self,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return iter_result_items(results)

    def _infer_case_score(self, item: Dict[str, Any]) -> Optional[float]:
        return infer_case_score(item)
    
    def _clean_prediction(self, prediction: str, task_type: str) -> str:
        """清理预测结果（移除 prompt tokens 保留纯答案）

        Florence-2 的 generate() 会返回完整序列（prompt + answer），
        需要从结果中移除 prompt 部分，只保留真正的预测答案。

        Args:
            prediction: 原始预测结果
            task_type: 任务类型

        Returns:
            清理后的预测结果
        """
        cleaned = prediction.strip()

        # 根据任务类型进行特定清理
        if task_type in FLORENCE2_TASKS:
            task_config = FLORENCE2_TASKS[task_type]

            # 移除任务 prompt
            prefix = task_config.prompt
            if prefix:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()

        return cleaned
    
    def _clean_reference(self, reference: str, task_type: str) -> str:
        """清理参考答案
        
        Args:
            reference: 原始参考答案
            task_type: 任务类型
            
        Returns:
            清理后的参考答案
        """
        # 基本清理
        cleaned = reference.strip()
        
        # 根据任务类型进行特定清理
        if task_type in FLORENCE2_TASKS:
            task_config = FLORENCE2_TASKS[task_type]

            # 移除任务 prompt
            prefix = task_config.prompt
            if prefix:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def _compute_overall_metrics(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        return compute_overall_metrics(task_metrics)

    def _save_evaluation_results(
        self,
        results: Dict[str, Any],
        predictions: Dict[str, List[Dict]],
        output_dir: Union[str, Path]
    ) -> None:
        save_evaluation_results(results, predictions, output_dir)
    
    def get_task_performance_summary(self) -> Dict[str, Any]:
        """获取任务性能摘要
        
        Returns:
            任务性能摘要
        """
        if not self.task_metrics:
            return {}
        
        summary = {}
        
        for task_type, task_data in self.task_metrics.items():
            metrics = task_data['metrics']
            sample_count = task_data['sample_count']
            
            # 选择关键指标
            key_metrics = {}
            
            # 通用指标
            for metric_name in ['accuracy', 'f1', 'bleu', 'rouge1_f1', 'precision', 'recall']:
                if metric_name in metrics:
                    key_metrics[metric_name] = metrics[metric_name]
            
            # 任务特定指标
            if 'caption' in task_type.lower():
                for metric_name in ['word_overlap', 'length_ratio']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'detection' in task_type.lower():
                for metric_name in ['mAP', 'true_positives', 'false_positives']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'ocr' in task_type.lower():
                for metric_name in ['character_accuracy', 'word_accuracy', 'edit_distance']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            elif 'segmentation' in task_type.lower():
                for metric_name in ['mean_iou', 'mean_dice']:
                    if metric_name in metrics:
                        key_metrics[metric_name] = metrics[metric_name]
            
            summary[task_type] = {
                'key_metrics': key_metrics,
                'sample_count': sample_count,
                'all_metrics': metrics
            }
        
        return summary
    
    def compare_with_baseline(
        self,
        baseline_results: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """与基线结果比较
        
        Args:
            baseline_results: 基线评估结果
            output_file: 输出文件路径
            
        Returns:
            比较结果
        """
        if not self.evaluation_results:
            raise ValueError("请先运行评估")
        
        comparison = {
            'current_results': self.evaluation_results,
            'baseline_results': baseline_results,
            'improvements': {},
            'regressions': {}
        }
        
        # 比较总体指标
        current_overall = self.evaluation_results.get('overall_metrics', {})
        baseline_overall = baseline_results.get('overall_metrics', {})
        
        for metric_name in current_overall:
            if metric_name in baseline_overall:
                current_value = current_overall[metric_name]
                baseline_value = baseline_overall[metric_name]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    improvement = current_value - baseline_value
                    relative_improvement = improvement / baseline_value if baseline_value != 0 else 0
                    
                    comparison['improvements'][metric_name] = {
                        'absolute': improvement,
                        'relative': relative_improvement,
                        'current': current_value,
                        'baseline': baseline_value
                    }
        
        # 比较任务级指标
        current_tasks = self.evaluation_results.get('task_metrics', {})
        baseline_tasks = baseline_results.get('task_metrics', {})
        
        task_comparisons = {}
        for task_type in current_tasks:
            if task_type in baseline_tasks:
                task_comparisons[task_type] = self._compare_task_metrics(
                    current_tasks[task_type]['metrics'],
                    baseline_tasks[task_type]['metrics']
                )
        
        comparison['task_comparisons'] = task_comparisons
        
        # 保存比较结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
        
        return comparison
    
    def _compare_task_metrics(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """比较任务指标"""
        comparison = {}
        
        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    improvement = current_value - baseline_value
                    relative_improvement = improvement / baseline_value if baseline_value != 0 else 0
                    
                    comparison[metric_name] = {
                        'absolute': improvement,
                        'relative': relative_improvement,
                        'current': current_value,
                        'baseline': baseline_value
                    }
        
        return comparison

    def evaluate_tvp_benchmark(
        self,
        data_path: Union[str, Path],
        *,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate the loaded model on a TVP JSONL benchmark dataset."""
        from .tvp_benchmark import run_tvp_benchmark

        return run_tvp_benchmark(
            self.model,
            data_path,
            max_samples=max_samples,
        )
