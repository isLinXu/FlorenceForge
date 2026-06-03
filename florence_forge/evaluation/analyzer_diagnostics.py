"""评估结果深度诊断：聚类、瓶颈、数据质量与行为分析。"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from . import analyzer_deps
from . import analyzer_scoring

logger = logging.getLogger(__name__)


def classify_error_type(prediction: str, reference: str, task_type: str) -> str:
    if not prediction.strip():
        return "empty_prediction"
    if len(prediction) < len(reference) * 0.3:
        return "too_short"
    if len(prediction) > len(reference) * 2.0:
        return "too_long"
    if task_type in ["object_detection", "phrase_grounding"]:
        if "<loc_" not in prediction and "<loc_" in reference:
            return "missing_location"
        if prediction.count("<loc_") != reference.count("<loc_"):
            return "location_count_mismatch"
    if task_type == "ocr":
        char_diff = abs(len(prediction) - len(reference)) / max(len(reference), 1)
        if char_diff > 0.5:
            return "character_mismatch"
    return "content_mismatch"


def extract_error_features(prediction: str, reference: str, task_type: str) -> List[float]:
    features = [
        len(prediction),
        len(reference),
        abs(len(prediction) - len(reference)),
        len(prediction) / max(len(reference), 1),
        1 if prediction.strip() == "" else 0,
        prediction.count(" "),
        prediction.count("\n"),
    ]
    if task_type in ["object_detection", "phrase_grounding"]:
        features.extend([prediction.count("<loc_"), reference.count("<loc_")])
    else:
        features.extend([0, 0])
    common_chars = set(prediction.lower()) & set(reference.lower())
    features.append(len(common_chars))
    features.append(len(common_chars) / max(len(set(reference.lower())), 1))
    return features


def perform_error_clustering(
    features: List[List[float]], method: str, n_clusters: int
) -> Dict[str, Any]:
    if not features:
        return {"labels": [], "n_clusters": 0}
    try:
        KMeans, DBSCAN, silhouette_score = analyzer_deps.load_clustering_dependencies()
    except ImportError as exc:
        logger.warning("%s", exc)
        return {
            "labels": [0] * len(features),
            "n_clusters": 0,
            "silhouette_score": None,
            "error": str(exc),
        }

    features_array = np.array(features)
    if method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
            features_array
        )
        sil = (
            float(silhouette_score(features_array, labels))
            if len(set(labels)) > 1
            else 0.0
        )
        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": sil,
        }
    if method == "dbscan":
        labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(features_array)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        sil = (
            float(silhouette_score(features_array, labels))
            if n_found > 1
            else 0.0
        )
        return {
            "labels": labels.tolist(),
            "n_clusters": n_found,
            "silhouette_score": sil,
        }
    raise ValueError(f"不支持的聚类方法: {method}")


def analyze_error_clusters(error_samples: List[Dict], labels: List[int]) -> Dict[str, Any]:
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for sample, label in zip(error_samples, labels):
        clusters[label].append(sample)
    pattern_analysis = {}
    for cluster_id, samples in clusters.items():
        if cluster_id == -1:
            continue
        error_types = [s["error_type"] for s in samples]
        task_types = [s["task_type"] for s in samples]
        pattern_analysis[f"cluster_{cluster_id}"] = {
            "size": len(samples),
            "dominant_error_type": Counter(error_types).most_common(1)[0][0],
            "error_type_distribution": dict(Counter(error_types)),
            "task_distribution": dict(Counter(task_types)),
            "sample_examples": samples[:3],
        }
    return pattern_analysis


def generate_error_pattern_recommendations(pattern_analysis: Dict[str, Any]) -> List[str]:
    recommendations = []
    for cluster_info in pattern_analysis.values():
        dominant_error = cluster_info["dominant_error_type"]
        cluster_size = cluster_info["size"]
        if dominant_error == "empty_prediction" and cluster_size > 5:
            recommendations.append(
                f"发现{cluster_size}个空预测错误，建议检查模型输入预处理和解码逻辑"
            )
        elif dominant_error == "missing_location" and cluster_size > 3:
            recommendations.append(
                f"发现{cluster_size}个位置标记缺失错误，建议加强位置检测训练"
            )
        elif dominant_error in ("too_short", "too_long") and cluster_size > 5:
            recommendations.append(
                f"发现{cluster_size}个长度异常错误，建议调整生成长度控制参数"
            )
    return recommendations


def cluster_error_patterns(
    evaluation_results: Dict[str, Any],
    clustering_method: str = "kmeans",
    n_clusters: int = 5,
) -> Dict[str, Any]:
    if not analyzer_deps.clustering_available():
        logger.warning("聚类分析库不可用，跳过错误模式分析")
        return {}

    error_samples = []
    error_features = []
    for task_type, task_data in evaluation_results.get("task_metrics", {}).items():
        for pred_data in task_data.get("predictions", []):
            prediction = pred_data.get("prediction", "")
            reference = pred_data.get("reference", "")
            if prediction == reference:
                continue
            error_samples.append(
                {
                    "task_type": task_type,
                    "prediction": prediction,
                    "reference": reference,
                    "error_type": classify_error_type(prediction, reference, task_type),
                }
            )
            error_features.append(
                extract_error_features(prediction, reference, task_type)
            )

    if not error_samples:
        return {"message": "没有发现错误样本"}

    cluster_results = perform_error_clustering(
        error_features, clustering_method, n_clusters
    )
    pattern_analysis = analyze_error_clusters(error_samples, cluster_results["labels"])
    return {
        "total_errors": len(error_samples),
        "clustering_method": clustering_method,
        "n_clusters": cluster_results.get("n_clusters", n_clusters),
        "silhouette_score": cluster_results.get("silhouette_score"),
        "error_patterns": pattern_analysis,
        "cluster_distribution": Counter(cluster_results["labels"]),
        "recommendations": generate_error_pattern_recommendations(pattern_analysis),
    }


def get_system_info() -> Dict[str, Any]:
    system_info = {"timestamp": datetime.now().isoformat()}
    psutil = analyzer_deps.load_psutil()
    if not psutil:
        return system_info
    try:
        system_info.update(
            {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            }
        )
    except Exception as exc:
        logger.warning("获取系统信息失败: %s", exc)
    return system_info


def identify_task_issues(task_info: Dict[str, Any]) -> List[str]:
    issues = []
    if task_info["score"] < 0.3:
        issues.append("性能严重不足")
    if task_info["sample_count"] < 50:
        issues.append("训练样本不足")
    metrics = task_info["metrics"]
    if metrics.get("accuracy", 1) < 0.5:
        issues.append("准确率过低")
    if metrics.get("f1", 1) < 0.4:
        issues.append("F1分数过低")
    if metrics.get("bleu", 1) < 0.2:
        issues.append("BLEU分数过低")
    return issues


def analyze_metric_bottlenecks(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    metric_scores: Dict[str, List[float]] = defaultdict(list)
    for task_data in task_metrics.values():
        for metric_name, value in task_data["metrics"].items():
            if isinstance(value, (int, float)):
                metric_scores[metric_name].append(value)
    bottlenecks = {}
    for metric_name, scores in metric_scores.items():
        if scores:
            avg_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            bottlenecks[metric_name] = {
                "average": avg_score,
                "minimum": min_score,
                "std_deviation": float(np.std(scores)),
                "is_bottleneck": avg_score < 0.5 or min_score < 0.2,
            }
    return bottlenecks


def analyze_resource_bottlenecks() -> Dict[str, Any]:
    resource_analysis: Dict[str, Any] = {}
    psutil = analyzer_deps.load_psutil()
    if not psutil:
        return resource_analysis
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        resource_analysis["cpu"] = {
            "usage_percent": cpu_percent,
            "is_bottleneck": cpu_percent > 80,
        }
        resource_analysis["memory"] = {
            "usage_percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "is_bottleneck": memory.percent > 85,
        }
        resource_analysis["disk"] = {
            "usage_percent": disk.percent,
            "free_gb": disk.free / (1024**3),
            "is_bottleneck": disk.percent > 90,
        }
    except Exception as exc:
        logger.warning("资源分析失败: %s", exc)
        resource_analysis["error"] = str(exc)
    return resource_analysis


def generate_bottleneck_recommendations(diagnosis: Dict[str, Any]) -> List[str]:
    recommendations = []
    for task_info in diagnosis.get("task_bottlenecks", {}).get(
        "worst_performing_tasks", []
    ):
        if "性能严重不足" in task_info["issues"]:
            recommendations.append(
                f"任务{task_info['task']}性能严重不足，建议重新设计模型架构"
            )
        if "训练样本不足" in task_info["issues"]:
            recommendations.append(
                f"任务{task_info['task']}训练样本不足，建议增加数据或使用数据增强"
            )
    for metric_name, metric_info in diagnosis.get("metric_bottlenecks", {}).items():
        if metric_info.get("is_bottleneck"):
            recommendations.append(f"指标{metric_name}表现不佳，建议针对性优化")
    resource = diagnosis.get("resource_bottlenecks", {})
    if resource.get("cpu", {}).get("is_bottleneck"):
        recommendations.append("CPU使用率过高，建议优化计算逻辑或增加计算资源")
    if resource.get("memory", {}).get("is_bottleneck"):
        recommendations.append("内存使用率过高，建议优化内存管理或增加内存")
    return recommendations


def diagnose_performance_bottlenecks(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "task_bottlenecks": {},
        "metric_bottlenecks": {},
        "resource_bottlenecks": {},
        "recommendations": [],
    }
    task_metrics = evaluation_results.get("task_metrics", {})
    task_performance = []
    for task_type, task_data in task_metrics.items():
        task_performance.append(
            {
                "task": task_type,
                "score": analyzer_scoring.calculate_performance_score(task_data["metrics"]),
                "sample_count": task_data.get("sample_count", 0),
                "metrics": task_data["metrics"],
            }
        )
    task_performance.sort(key=lambda item: item["score"])
    diagnosis["task_bottlenecks"] = {
        "worst_performing_tasks": [
            {
                "task": task["task"],
                "score": task["score"],
                "issues": identify_task_issues(task),
            }
            for task in task_performance[:3]
        ]
    }
    diagnosis["metric_bottlenecks"] = analyze_metric_bottlenecks(task_metrics)
    diagnosis["resource_bottlenecks"] = analyze_resource_bottlenecks()
    diagnosis["recommendations"] = generate_bottleneck_recommendations(diagnosis)
    return diagnosis


def assess_task_data_quality(
    predictions: List[Dict], task_type: str, sample_count: int
) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "sample_count": sample_count,
        "quality_score": 1.0 if sample_count > 0 else 0.0,
        "issues": [],
    }


def analyze_data_distribution(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    return {
        "task_sample_counts": {
            task: data.get("sample_count", 0) for task, data in task_metrics.items()
        }
    }


def identify_quality_issues(
    task_quality: Dict[str, Any], distribution: Dict[str, Any]
) -> List[str]:
    return [
        f"{task}: 质量分数偏低"
        for task, info in task_quality.items()
        if info.get("quality_score", 1.0) < 0.5
    ]


def generate_quality_recommendations(quality_assessment: Dict[str, Any]) -> List[str]:
    if quality_assessment.get("quality_issues"):
        return ["建议检查低质量任务的数据标注与采样分布"]
    return []


def assess_data_quality(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    quality_assessment = {
        "timestamp": datetime.now().isoformat(),
        "overall_quality_score": 0.0,
        "task_quality": {},
        "data_distribution": {},
        "quality_issues": [],
        "recommendations": [],
    }
    task_metrics = evaluation_results.get("task_metrics", {})
    scores = []
    for task_type, task_data in task_metrics.items():
        task_quality = assess_task_data_quality(
            task_data.get("predictions", []),
            task_type,
            task_data.get("sample_count", 0),
        )
        quality_assessment["task_quality"][task_type] = task_quality
        scores.append(task_quality["quality_score"])
    if scores:
        quality_assessment["overall_quality_score"] = float(np.mean(scores))
    distribution = analyze_data_distribution(task_metrics)
    quality_assessment["data_distribution"] = distribution
    quality_assessment["quality_issues"] = identify_quality_issues(
        quality_assessment["task_quality"], distribution
    )
    quality_assessment["recommendations"] = generate_quality_recommendations(
        quality_assessment
    )
    return quality_assessment


def analyze_prediction_patterns(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    return {"tasks": list(task_metrics.keys())}


def analyze_confidence_distribution(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    return {}


def detect_model_bias(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    return {}


def analyze_prediction_consistency(task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    return {}


def generate_behavioral_insights(behavior_analysis: Dict[str, Any]) -> List[str]:
    return []


def analyze_model_behavior(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    task_metrics = evaluation_results.get("task_metrics", {})
    behavior_analysis = {
        "timestamp": datetime.now().isoformat(),
        "prediction_patterns": analyze_prediction_patterns(task_metrics),
        "confidence_analysis": analyze_confidence_distribution(task_metrics),
        "bias_detection": detect_model_bias(task_metrics),
        "consistency_analysis": analyze_prediction_consistency(task_metrics),
        "behavioral_insights": [],
    }
    behavior_analysis["behavioral_insights"] = generate_behavioral_insights(
        behavior_analysis
    )
    return behavior_analysis
