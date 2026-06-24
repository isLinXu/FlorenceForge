#!/usr/bin/env python3
"""COCO128 数据集训练脚本。

该脚本会：
1. 下载 Florence-2-base 模型
2. 下载或复用本地 COCO128 数据集
3. 自动识别 YOLO / COCO 标注布局并转换为 FlorenceForge JSONL
4. 启动目标检测训练
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 配置
FLORENCE_MODEL_NAME = "/home/linxu/PycharmProjects/AI-ModelScope/Florence-2-base"
# COCO128 数据集镜像源列表（按优先级排序）
COCO128_MIRRORS = [
    "https://modelscope.cn/datasets/summary/coco128/resolve/master/coco128.zip",
    "https://ghproxy.com/https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "coco128_training"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "coco128"
DEFAULT_CLASSES_FILE = REPO_ROOT / "data" / "coco80.names"


def download_file(url: str, dest: Path, desc: str = "下载中", timeout: int = 60) -> bool:
    """下载文件并显示进度条。
    
    Returns:
        bool: 下载成功返回 True，失败返回 False
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"下载完成: {dest}")
        return True
    except Exception as e:
        logger.warning(f"下载失败 ({url}): {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_coco128(data_dir: Path) -> Path:
    """下载并解压 COCO128 数据集。"""
    data_dir = data_dir.absolute()
    coco128_dir = data_dir / "coco128"
    
    # 检查是否已存在
    if coco128_dir.exists():
        logger.info(f"COCO128 数据集已存在: {coco128_dir}")
        return coco128_dir
    
    # 下载数据集 - 尝试多个镜像源
    zip_path = data_dir / "coco128.zip"
    logger.info("开始下载 COCO128 数据集...")
    
    download_success = False
    for i, url in enumerate(COCO128_MIRRORS):
        logger.info(f"尝试镜像源 {i+1}/{len(COCO128_MIRRORS)}: {url}")
        if download_file(url, zip_path, desc=f"COCO128 (镜像{i+1})", timeout=120):
            download_success = True
            break
    
    if not download_success:
        raise RuntimeError("所有镜像源下载失败，请检查网络连接或手动下载 COCO128 数据集")
    
    # 解压
    logger.info("解压数据集...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    
    # 删除 zip 文件
    zip_path.unlink()
    
    logger.info(f"COCO128 数据集准备完成: {coco128_dir}")
    return coco128_dir


def download_florence_model() -> None:
    """预下载 Florence-2 模型。"""
    logger.info(f"开始下载 Florence-2 模型: {FLORENCE_MODEL_NAME}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        logger.info("下载模型和处理器...")
        processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_NAME,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_NAME,
            trust_remote_code=True
        )
        
        logger.info("模型下载完成!")
        del model, processor  # 释放内存
        
    except Exception as e:
        logger.warning(f"模型下载过程中出现警告: {e}")
        logger.info("模型将在训练时自动下载")


def _normalize_coco128_root(coco128_dir: Path) -> Path:
    """兼容 data/coco128 与 data/coco128/coco128 两种目录入口。"""
    coco128_dir = coco128_dir.absolute()
    nested_dir = coco128_dir / "coco128"
    if nested_dir.is_dir() and not (coco128_dir / "images").is_dir():
        return nested_dir
    return coco128_dir


def convert_coco128_to_florence2(
    coco128_dir: Path,
    output_jsonl: Path,
    classes_file: Path = DEFAULT_CLASSES_FILE,
) -> None:
    """将 COCO128 数据集转换为 FlorenceForge 目标检测 JSONL。"""
    from florence_forge.data.converter import DataFormatConverter

    coco128_dir = _normalize_coco128_root(coco128_dir)
    images_dir = coco128_dir / "images" / "train2017"
    labels_dir = coco128_dir / "labels" / "train2017"
    annotations_dir = coco128_dir / "annotations"

    if labels_dir.is_dir():
        if not classes_file.exists():
            raise FileNotFoundError(f"未找到 YOLO 类别文件: {classes_file}")
        logger.info("检测到 YOLO 布局，使用 YOLO -> FlorenceForge 转换")
        logger.info(f"标签目录: {labels_dir}")
        logger.info(f"图像目录: {images_dir}")
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        DataFormatConverter.yolo_to_florence2_od(
            yolo_labels_dir=str(labels_dir),
            output_path=str(output_jsonl),
            image_dir=str(images_dir),
            classes_file=str(classes_file),
            task_type="OD",
        )
    else:
        json_files = list(annotations_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"未找到可用标注，既没有 YOLO 标签目录 {labels_dir}，也没有 COCO 标注目录 {annotations_dir}"
            )
        coco_json_path = json_files[0]
        logger.info("检测到 COCO 布局，使用 COCO -> FlorenceForge 转换")
        logger.info(f"使用标注文件: {coco_json_path}")
        logger.info(f"图像目录: {images_dir}")
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        DataFormatConverter.coco_to_florence2_od(
            coco_json_path=str(coco_json_path),
            output_path=str(output_jsonl),
            image_dir=str(images_dir),
            task_type="OD",
        )

    logger.info("开始转换数据集...")
    logger.info(f"数据转换完成: {output_jsonl}")

    # 统计样本数量
    with open(output_jsonl, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    logger.info(f"共转换 {count} 个训练样本")


def create_training_config(output_dir: Path, train_jsonl: Path) -> Path:
    """创建训练配置文件。"""
    import yaml

    config = {
        # 基本训练参数
        "num_epochs": 3,
        "max_steps": None,
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 50,
        
        # 输出目录
        "output_dir": str(output_dir),
        "logging_dir": str(output_dir / "logs"),
        
        # 混合精度训练
        "use_fp16": False,
        "use_bf16": False,
        
        # 梯度设置
        "gradient_accumulation_steps": 4,
        
        # 检查点管理
        "save_total_limit": 3,
        
        # 实验跟踪
        "experiment_name": "florence2_coco128_od",
        "run_name": "coco128_baseline",
        "tags": ["florence2", "object-detection", "coco128"],

        # 任务与数据
        "tasks": ["OD"],
        "task_weights": {"OD": 1.0},
        "train_data_path": str(train_jsonl),
        "device": "cpu",
        
        # 模型配置
        "model_config": {
            "model_name": FLORENCE_MODEL_NAME,
            "trust_remote_code": True,
            "torch_dtype": "float32",
            "device_map": None,
            "device": "cpu",
            "use_lora": True,
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "gradient_checkpointing": True,
        },
        
        # 数据配置
        "data_config": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "drop_last": False,
            "use_augmentation": False,
        },
        
        # 优化配置
        "optimization_config": {
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.1,
        },
        
        # 任务调度配置
        "task_scheduling_config": {
            "strategy": "round_robin",
            "temperature": 1.0,
        },
        
        # 分布式训练配置
        "distributed_config": {
            "enabled": False,
        },
    }
    
    config_path = output_dir / "config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"配置文件已创建: {config_path}")
    return config_path


def run_training(config_path: Path, train_data: Path) -> None:
    """启动训练。"""
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable, "-m", "florence_forge.cli.main", "train",
        "--task", "od",
        "--config", str(config_path),
        "--train-data", str(train_data),
        "--trainer-version", "v2",
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="COCO128 数据集训练脚本")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="数据存储目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="训练输出目录"
    )
    parser.add_argument(
        "--skip-download-model",
        action="store_true",
        help="跳过模型下载"
    )
    parser.add_argument(
        "--skip-download-data",
        action="store_true",
        help="跳过数据集下载"
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="跳过数据转换"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="仅运行训练（需要先准备好数据）"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Florence-2 COCO128 训练流程")
    logger.info("=" * 60)
    
    # 1. 下载模型
    if not args.skip_download_model and not args.train_only:
        logger.info("\n[步骤 1/4] 下载 Florence-2 模型...")
        download_florence_model()
    else:
        logger.info("\n[步骤 1/4] 跳过模型下载")
    
    # 2. 下载数据集
    if not args.skip_download_data and not args.train_only:
        logger.info("\n[步骤 2/4] 下载 COCO128 数据集...")
        coco128_dir = download_coco128(args.data_dir)
    else:
        coco128_dir = args.data_dir / "coco128"
        logger.info(f"\n[步骤 2/4] 使用已有数据: {coco128_dir}")
    
    # 3. 转换数据格式
    train_jsonl = args.output_dir / "train.jsonl"
    if not args.skip_convert and not args.train_only:
        logger.info("\n[步骤 3/4] 转换数据格式...")
        convert_coco128_to_florence2(coco128_dir, train_jsonl)
    else:
        logger.info(f"\n[步骤 3/4] 使用已有数据: {train_jsonl}")
    
    # 4. 创建配置并启动训练
    logger.info("\n[步骤 4/4] 创建配置并启动训练...")
    config_path = create_training_config(args.output_dir, train_jsonl)
    run_training(config_path, train_jsonl)
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info(f"模型保存在: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
