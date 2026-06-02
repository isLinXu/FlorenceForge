"""FlorenceForge Benchmark评估模块

提供标准化的benchmark指标计算和评估功能
"""

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from ..data.dataset import MultiTaskDataset
from .benchmark_cache import BenchmarkCache
from .benchmark_monitoring import (
    BenchmarkMonitor,
    export_monitoring_data,
    get_real_time_status,
    monitoring_data_snapshot,
)
from .benchmark_lazy_metrics import make_default_advanced_metric_calculators
from .benchmark_parallel import run_parallel_dataset_evaluation
from .benchmark_reports import (
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
    generate_pdf_report,
    save_benchmark_results,
)
from .benchmark_statistics import (
    analyze_performance_trends,
    analyze_resource_usage,
    compare_task_metrics,
    compare_with_baseline,
    compute_overall_summary,
    compute_statistical_summary,
    compute_task_performance,
    enhance_results_for_report,
    generate_optimization_recommendations,
)
from .evaluator import MultiTaskEvaluator
from .metrics import get_metric_calculator

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """标准化的benchmark评估器

    提供多种评估协议和指标计算，支持与基线结果比较。
    支持多GPU并行评估、增量评估、实时监控等高级功能。
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "auto",
        config: Optional[Dict[str, Any]] = None,
        enable_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.model = model
        self.device = self._setup_device(device)
        self.config = config or self._get_default_config()

        # 分布式训练设置
        self.enable_distributed = enable_distributed
        self.world_size = world_size
        self.rank = rank

        # 增量评估缓存
        self.cache_dir = Path(self.config.get("cache_dir", "./benchmark_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_incremental = self.config.get("enable_incremental", True)

        # 实时监控
        self.monitoring_enabled = self.config.get("enable_monitoring", True)
        self.monitoring_history_size = max(
            0,
            int(self.config.get("monitoring_history_size", 100) or 0),
        )
        self._benchmark_monitor = BenchmarkMonitor(
            enabled=self.monitoring_enabled,
            history_size=self.monitoring_history_size,
            sample_interval=float(
                self.config.get("monitoring_interval_seconds", 5.0) or 5.0
            ),
            cpu_interval=float(
                self.config.get("monitoring_cpu_interval_seconds", 1.0) or 1.0
            ),
        )
        self.monitoring_data = self._benchmark_monitor.data
        self.monitoring_thread = None

        # 设置模型
        self._setup_model()

        # 初始化评估器
        self.evaluator = MultiTaskEvaluator(model=self.model)

        # 高级评估指标计算器默认懒加载，避免 benchmark 启动时拉起重型模型。
        self._setup_advanced_metric_calculators()

        logger.info(
            f"BenchmarkEvaluator initialized on {self.device} (distributed: {enable_distributed})"
        )

    def _setup_advanced_metric_calculators(self) -> None:
        calculators = make_default_advanced_metric_calculators()
        for attr_name, calculator in calculators.items():
            setattr(self, attr_name, calculator)

        if not self.config.get("lazy_advanced_metrics", True):
            for calculator in calculators.values():
                calculator.load()

    def _setup_device(self, device: Union[str, torch.device]) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif isinstance(device, str):
            return torch.device(device)
        else:
            return device

    def _setup_model(self):
        """设置模型（支持分布式）"""
        self.model.to(self.device)
        self.model.eval()

        if self.enable_distributed:
            if self.world_size <= 1:
                logger.warning(
                    "enable_distributed=True 但 world_size <= 1，跳过 DDP 包装"
                )
                return

            # 初始化分布式环境
            if not dist.is_initialized():
                init_method = self.config.get(
                    "distributed_init_method"
                ) or self.config.get("init_method")
                if init_method is None:
                    has_env_rendezvous = os.environ.get(
                        "MASTER_ADDR"
                    ) and os.environ.get("MASTER_PORT")
                    if has_env_rendezvous:
                        init_method = "env://"
                    else:
                        raise ValueError(
                            "启用分布式 benchmark 需要提供 init_method/"
                            "distributed_init_method，或设置 MASTER_ADDR 与 MASTER_PORT。"
                        )

                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                )

            # 包装模型为DDP
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
                if torch.cuda.device_count() > 0:
                    local_rank = local_rank % torch.cuda.device_count()
                torch.cuda.set_device(local_rank)
                self.model = DDP(self.model, device_ids=[local_rank])
            else:
                self.model = DDP(self.model)

    def _start_monitoring(self):
        """启动实时监控"""
        monitor = getattr(self, "_benchmark_monitor", None)
        if monitor is None:
            return
        monitor.enabled = self.monitoring_enabled
        self.monitoring_thread = monitor.start()

    def _stop_monitoring(self):
        """停止实时监控"""
        monitor = getattr(self, "_benchmark_monitor", None)
        if monitor is not None:
            monitor.stop(timeout=1)
            self.monitoring_thread = monitor.thread

    def _get_cache_key(
        self, dataset_name: str, task_type: str, model_config: Dict
    ) -> str:
        """生成缓存键"""
        return BenchmarkCache.make_key(dataset_name, task_type, model_config)

    def _monitoring_data_snapshot(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的监控数据快照。"""
        monitor = getattr(self, "_benchmark_monitor", None)
        if monitor is not None:
            return monitor.snapshot()
        return monitoring_data_snapshot(self.monitoring_data)

    def _save_cached_results(self, cache_key: str, results: Dict[str, Any]):
        """保存评估结果到缓存（兼容代理，实际逻辑在 BenchmarkCache）"""
        self._cache_helper().save_results(cache_key, results)

    def _load_cached_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """加载缓存的评估结果（兼容代理，实际逻辑在 BenchmarkCache）"""
        return self._cache_helper().load_results(cache_key)

    def _cache_helper(self) -> BenchmarkCache:
        """创建 benchmark cache helper。

        保持为轻量工厂，兼容 __new__ 构造的测试/worker helper。
        """
        return BenchmarkCache(
            cache_dir=self.cache_dir,
            config=self.config,
            enable_incremental=self.enable_incremental,
        )

    def run_benchmark(
        self,
        datasets: Dict[str, MultiTaskDataset],
        output_dir: Union[str, Path],
        save_detailed: bool = True,
        compare_baseline: Optional[Dict[str, Any]] = None,
        use_parallel: bool = False,
        num_gpus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """运行完整的benchmark评估

        Args:
            datasets: 评估数据集字典 {dataset_name: dataset}
            output_dir: 输出目录
            save_detailed: 是否保存详细结果
            compare_baseline: 基线结果用于比较
            use_parallel: 是否使用多GPU并行评估
            num_gpus: 使用的GPU数量

        Returns:
            Benchmark评估结果
        """
        logger.info(f"开始运行benchmark评估，数据集: {list(datasets.keys())}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_start_time = time.time()

        # 启动监控
        self._start_monitoring()

        try:
            # 选择评估策略
            if use_parallel and torch.cuda.device_count() > 1:
                benchmark_results = self._run_parallel_benchmark(
                    datasets, output_path, compare_baseline, save_detailed, num_gpus
                )
            else:
                benchmark_results = self._run_sequential_benchmark(
                    datasets, output_path, compare_baseline, save_detailed
                )

            # 计算总评估时间
            total_time = time.time() - total_start_time
            benchmark_results["benchmark_info"]["total_evaluation_time"] = total_time

            # 添加监控数据
            if self.monitoring_enabled:
                benchmark_results["monitoring_data"] = self._monitoring_data_snapshot()

            self.benchmark_results = benchmark_results

            logger.info(f"Benchmark评估完成，总耗时: {total_time:.2f}秒")

            return benchmark_results

        finally:
            # 停止监控
            self._stop_monitoring()

    def _run_sequential_benchmark(
        self,
        datasets: Dict[str, MultiTaskDataset],
        output_path: Path,
        compare_baseline: Optional[Dict[str, Any]],
        save_detailed: bool,
    ) -> Dict[str, Any]:
        """运行顺序benchmark评估"""
        benchmark_results = {
            "benchmark_info": {
                "model_name": getattr(self.model, "model_name", "Florence2"),
                "device": str(self.device),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config,
                "evaluation_mode": "sequential",
            },
            "dataset_results": {},
            "overall_summary": {},
            "task_performance": {},
        }

        # 评估每个数据集
        for dataset_name, dataset in tqdm(datasets.items(), desc="评估数据集"):
            logger.info(f"评估数据集: {dataset_name}")

            # 检查缓存
            cache_key = self._get_cache_key(
                dataset_name,
                "mixed",  # 混合任务
                {"model_name": getattr(self.model, "model_name", "Florence2")},
            )

            cached_result = self._load_cached_results(cache_key)
            if cached_result:
                dataset_result = cached_result
                logger.info(f"使用缓存结果: {dataset_name}")
            else:
                dataset_result = self._evaluate_dataset(
                    dataset, dataset_name, output_path / dataset_name
                )
                self._save_cached_results(cache_key, dataset_result)

            benchmark_results["dataset_results"][dataset_name] = dataset_result

            # 更新进度
            self.monitoring_data["current_progress"] = len(
                benchmark_results["dataset_results"]
            ) / len(datasets)

        # 计算总体摘要
        benchmark_results["overall_summary"] = self._compute_overall_summary(
            benchmark_results["dataset_results"]
        )

        # 计算任务性能统计
        benchmark_results["task_performance"] = self._compute_task_performance(
            benchmark_results["dataset_results"]
        )

        # 与基线比较
        if compare_baseline:
            benchmark_results["baseline_comparison"] = self._compare_with_baseline(
                benchmark_results, compare_baseline
            )

        # 保存结果
        self._save_benchmark_results(benchmark_results, output_path, save_detailed)

        return benchmark_results

    def _run_parallel_benchmark(
        self,
        datasets: Dict[str, MultiTaskDataset],
        output_path: Path,
        compare_baseline: Optional[Dict[str, Any]],
        save_detailed: bool,
        num_gpus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """运行多GPU并行benchmark评估"""
        parallel_run = run_parallel_dataset_evaluation(
            datasets=datasets,
            output_path=output_path,
            cache_dir=self.cache_dir,
            config=self.config,
            enable_incremental=self.enable_incremental,
            model_template_factory=self._prepare_spawn_model_template,
            num_gpus=num_gpus,
        )
        worker_count = parallel_run.worker_count

        benchmark_results = {
            "benchmark_info": {
                "model_name": getattr(self.model, "model_name", "Florence2"),
                "device": f"cuda (x{worker_count})",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config,
                "evaluation_mode": "parallel",
                "num_gpus": worker_count,
            },
            "dataset_results": parallel_run.dataset_results,
            "overall_summary": {},
            "task_performance": {},
        }

        # 计算总体摘要
        benchmark_results["overall_summary"] = self._compute_overall_summary(
            benchmark_results["dataset_results"]
        )

        # 计算任务性能统计
        benchmark_results["task_performance"] = self._compute_task_performance(
            benchmark_results["dataset_results"]
        )

        # 与基线比较
        if compare_baseline:
            benchmark_results["baseline_comparison"] = self._compare_with_baseline(
                benchmark_results, compare_baseline
            )

        # 保存结果
        self._save_benchmark_results(benchmark_results, output_path, save_detailed)

        return benchmark_results

    def _prepare_spawn_model_template(self) -> nn.Module:
        """Create a CPU model template suitable for torch.multiprocessing.spawn."""
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        try:
            model_template = copy.deepcopy(model_ref)
        except Exception as exc:
            raise RuntimeError(
                "并行 benchmark 需要模型可 deepcopy，以便为每个 spawn worker 创建独立副本。"
                "请改用顺序 benchmark，或为模型提供可序列化的轻量包装。"
            ) from exc

        try:
            model_template = model_template.to(torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError("无法将并行 benchmark 模型模板移动到 CPU") from exc

        model_template.eval()
        return model_template

    def _evaluate_datasets_on_gpu(
        self, datasets: Dict[str, MultiTaskDataset], gpu_id: int, output_path: Path
    ) -> Dict[str, Any]:
        """在指定GPU上评估数据集

        注意：此方法运行在 spawn 上下文的子进程中，每个进程拥有独立的 CUDA 上下文。
        通过 deepcopy state_dict 来避免共享参数访问。
        """
        import copy

        # 设置GPU设备
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        # 创建模型副本并移动到指定GPU（使用 deepcopy 避免共享参数）
        model_ref = self.model
        if hasattr(self.model, "module"):  # 如果是DDP包装的模型
            model_ref = self.model.module

        # Deep copy state_dict 并加载到新设备，避免跨 GPU 共享参数
        try:
            model_copy = copy.deepcopy(model_ref)
        except Exception:
            # deepcopy 可能因自定义对象失败，回退到 state_dict 拷贝
            model_copy = type(model_ref)(model_ref.config)
            model_copy.load_state_dict(
                {k: v.clone() for k, v in model_ref.state_dict().items()}
            )

        model_copy = model_copy.to(device)
        model_copy.eval()

        # 创建评估器
        evaluator = MultiTaskEvaluator(model_copy)

        results = {}
        for dataset_name, dataset in datasets.items():
            logger.info(f"GPU {gpu_id} 评估数据集: {dataset_name}")

            # 检查缓存
            cache_key = self._get_cache_key(
                dataset_name,
                "mixed",
                {
                    "model_name": getattr(model_copy, "model_name", "Florence2"),
                    "gpu_id": gpu_id,
                },
            )

            cached_result = self._load_cached_results(cache_key)
            if cached_result:
                results[dataset_name] = cached_result
                logger.info(f"GPU {gpu_id} 使用缓存结果: {dataset_name}")
            else:
                dataset_result = evaluator.evaluate_dataset(
                    dataset,
                    batch_size=self.config.get("batch_size", 8),
                    num_workers=self.config.get("num_workers", 4),
                    max_samples_per_task=self.config.get("max_samples_per_task"),
                    save_predictions=self.config.get("save_predictions", False),
                    output_dir=output_path / dataset_name,
                )

                self._save_cached_results(cache_key, dataset_result)
                results[dataset_name] = dataset_result

        return results

    def evaluate_single_task(
        self,
        dataset: MultiTaskDataset,
        task_type: str,
        output_dir: Optional[Union[str, Path]] = None,
        detailed_analysis: bool = True,
    ) -> Dict[str, Any]:
        """评估单个任务

        Args:
            dataset: 评估数据集
            task_type: 任务类型
            output_dir: 输出目录
            detailed_analysis: 是否进行详细分析

        Returns:
            任务评估结果
        """
        logger.info(f"开始评估任务: {task_type}")

        # 筛选任务数据
        task_dataset = self._filter_task_dataset(dataset, task_type)

        if len(task_dataset) == 0:
            logger.warning(f"数据集中没有任务 {task_type} 的数据")
            return {}

        # 运行评估
        task_result = self.evaluator.evaluate_task(
            task_dataset,
            task_type,
            batch_size=self.config.get("batch_size", 8),
            max_samples=self.config.get("max_samples_per_task"),
        )

        # 详细分析
        if detailed_analysis:
            task_result["detailed_analysis"] = self._analyze_task_performance(
                task_result, task_type
            )

        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            with open(
                output_path / f"{task_type}_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(task_result, f, indent=2, ensure_ascii=False)

        return task_result

    def compute_standard_metrics(
        self,
        predictions: List[Any],
        references: List[Any],
        task_type: str,
        metric_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """计算标准指标

        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            task_type: 任务类型
            metric_config: 指标配置

        Returns:
            标准指标字典
        """
        if len(predictions) != len(references):
            raise ValueError("预测结果和参考答案数量不匹配")

        # 获取指标计算器
        calculator = get_metric_calculator(task_type)

        # 添加数据
        calculator.add_batch(predictions, references)

        # 计算指标
        metrics = calculator.compute()

        # 应用指标配置
        if metric_config:
            metrics = self._apply_metric_config(metrics, metric_config)

        return metrics

    def generate_benchmark_report(
        self,
        results: Dict[str, Any],
        output_file: Union[str, Path],
        format: str = "markdown",
        include_monitoring: bool = True,
        include_recommendations: bool = True,
    ) -> None:
        """生成benchmark报告

        Args:
            results: benchmark结果
            output_file: 输出文件路径
            format: 报告格式 ('markdown', 'html', 'json')
            include_monitoring: 是否包含监控数据
            include_recommendations: 是否包含优化建议
        """
        output_file = Path(output_file)

        # 增强结果数据
        enhanced_results = self._enhance_results_for_report(
            results, include_monitoring, include_recommendations
        )

        if format == "markdown":
            self._generate_markdown_report(enhanced_results, output_file)
        elif format == "html":
            self._generate_html_report(enhanced_results, output_file)
        elif format == "json":
            self._generate_json_report(enhanced_results, output_file)
        elif format == "pdf":
            self._generate_pdf_report(enhanced_results, output_file)
        else:
            raise ValueError(f"不支持的报告格式: {format}")

        logger.info(f"Benchmark报告已生成: {output_file}")

    def _enhance_results_for_report(
        self,
        results: Dict[str, Any],
        include_monitoring: bool,
        include_recommendations: bool,
    ) -> Dict[str, Any]:
        """增强结果数据用于报告生成"""
        return enhance_results_for_report(
            results,
            include_monitoring=include_monitoring,
            include_recommendations=include_recommendations,
        )

    def _analyze_performance_trends(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能趋势"""
        return analyze_performance_trends(results)

    def _analyze_resource_usage(
        self, monitoring_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析资源使用情况"""
        return analyze_resource_usage(monitoring_data)

    def _generate_optimization_recommendations(
        self, results: Dict[str, Any]
    ) -> List[str]:
        """生成优化建议"""
        return generate_optimization_recommendations(results)

    def _compute_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计摘要"""
        return compute_statistical_summary(results)

    def get_real_time_status(self) -> Dict[str, Any]:
        """获取实时评估状态"""
        monitor = getattr(self, "_benchmark_monitor", None)
        if monitor is not None:
            return monitor.status()
        return get_real_time_status(
            self.monitoring_data,
            enabled=getattr(self, "monitoring_enabled", True),
        )

    def export_monitoring_data(self, output_file: Union[str, Path]) -> None:
        """导出监控数据"""
        monitor = getattr(self, "_benchmark_monitor", None)
        if monitor is not None:
            monitor.export(output_file)
            return
        export_monitoring_data(self.monitoring_data, output_file)

    def _evaluate_dataset(
        self, dataset: MultiTaskDataset, dataset_name: str, output_dir: Path
    ) -> Dict[str, Any]:
        """评估单个数据集"""
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # 运行评估
        eval_results = self.evaluator.evaluate_dataset(
            dataset,
            batch_size=self.config.get("batch_size", 8),
            num_workers=self.config.get("num_workers", 4),
            max_samples_per_task=self.config.get("max_samples_per_task"),
            save_predictions=self.config.get("save_predictions", False),
            output_dir=output_dir,
        )

        evaluation_time = time.time() - start_time

        # 添加数据集特定信息
        dataset_result = {
            "dataset_name": dataset_name,
            "evaluation_time": evaluation_time,
            "dataset_info": {
                "total_samples": len(dataset),
                "task_distribution": dataset.get_task_statistics(),
                "data_path": getattr(dataset, "data_path", "unknown"),
            },
            "metrics": eval_results.get("task_metrics", {}),
            "overall_metrics": eval_results.get("overall_metrics", {}),
        }

        return dataset_result

    def _compute_overall_summary(
        self, dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算总体摘要"""
        return compute_overall_summary(dataset_results)

    def _compute_task_performance(
        self, dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算任务性能统计"""
        return compute_task_performance(dataset_results)

    def _compare_with_baseline(
        self, current_results: Dict[str, Any], baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """与基线结果比较"""
        return compare_with_baseline(current_results, baseline_results)

    def _compare_task_metrics(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """比较任务指标"""
        return compare_task_metrics(current_metrics, baseline_metrics)

    def _filter_task_dataset(
        self, dataset: MultiTaskDataset, task_type: str
    ) -> MultiTaskDataset:
        """筛选特定任务的数据集"""
        # 这里简化实现，实际应该创建新的数据集实例
        # 只包含指定任务的数据
        return dataset  # 临时返回原数据集

    def _analyze_task_performance(
        self, task_result: Dict[str, Any], task_type: str
    ) -> Dict[str, Any]:
        """分析任务性能"""
        analysis = {
            "task_type": task_type,
            "performance_level": "unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        }

        metrics = task_result.get("metrics", {})

        # 简单的性能分析逻辑
        if task_type.lower() in ["caption", "detailed_caption"]:
            bleu_score = metrics.get("bleu_4", 0)
            if bleu_score > 0.3:
                analysis["performance_level"] = "good"
                analysis["strengths"].append("良好的BLEU分数")
            elif bleu_score > 0.2:
                analysis["performance_level"] = "fair"
            else:
                analysis["performance_level"] = "poor"
                analysis["weaknesses"].append("BLEU分数较低")
                analysis["recommendations"].append("考虑增加训练数据或调整模型参数")

        elif task_type.lower() in ["detection", "object_detection"]:
            map_score = metrics.get("mAP", 0)
            if map_score > 0.5:
                analysis["performance_level"] = "good"
                analysis["strengths"].append("良好的mAP分数")
            elif map_score > 0.3:
                analysis["performance_level"] = "fair"
            else:
                analysis["performance_level"] = "poor"
                analysis["weaknesses"].append("mAP分数较低")
                analysis["recommendations"].append("考虑调整检测阈值或增强数据")

        return analysis

    def _apply_metric_config(
        self, metrics: Dict[str, float], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """应用指标配置"""
        # 筛选指标
        if "include_metrics" in config:
            metrics = {
                k: v for k, v in metrics.items() if k in config["include_metrics"]
            }

        # 排除指标
        if "exclude_metrics" in config:
            metrics = {
                k: v for k, v in metrics.items() if k not in config["exclude_metrics"]
            }

        return metrics

    def _save_benchmark_results(
        self, results: Dict[str, Any], output_dir: Path, save_detailed: bool
    ) -> None:
        """保存benchmark结果。"""
        save_benchmark_results(results, output_dir, save_detailed)

    def _generate_markdown_report(
        self, results: Dict[str, Any], output_file: Path
    ) -> None:
        """生成Markdown格式报告。"""
        generate_markdown_report(results, output_file)

    def _generate_html_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成HTML格式报告。"""
        generate_html_report(results, output_file)

    def _generate_json_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成JSON格式报告。"""
        generate_json_report(
            results,
            output_file,
            enable_distributed=self.enable_distributed,
            enable_incremental=self.enable_incremental,
        )

    def _generate_pdf_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成PDF格式报告。"""
        generate_pdf_report(
            results,
            output_file,
            enable_distributed=self.enable_distributed,
            enable_incremental=self.enable_incremental,
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "batch_size": 8,
            "num_workers": 4,
            "max_samples_per_task": None,
            "save_predictions": False,
            "compute_detailed_metrics": True,
            "metric_config": {
                "include_metrics": None,  # None表示包含所有指标
                "exclude_metrics": [],
                "custom_thresholds": {},
            },
        }
