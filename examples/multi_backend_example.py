"""多后端 VLM 统一训练与推理示例

本示例展示如何在 FlorenceForge 框架中：
1. 切换不同的 VLM 后端（Florence-2 / PaliGemma / YouTu-VL / GenericHF）
2. 配置分布式训练（DDP / FSDP / DeepSpeed ZeRO）
3. 使用激活值重计算策略优化显存占用

支持的后端:
- Florence-2 (microsoft/Florence-2-large)
- PaliGemma (google/paligemma-3b-pt-224)
- YouTu-VL (tencent-YouTu/Youtu-VL-4B-Instruct)
- GenericHFBackend（自动适配任意 transformers VLM）

使用方法:
    # 单卡推理
    python examples/multi_backend_example.py --backend florence-2 --task CAPTION --image test.jpg

    # 单卡训练
    python examples/multi_backend_example.py --mode train --backend florence-2 --data data.json

    # 分布式训练（4卡 DDP）
    torchrun --nproc_per_node=4 examples/multi_backend_example.py \
        --mode train --backend florence-2 --data data.json --distributed

    # FSDP 训练（适合大模型）
    torchrun --nproc_per_node=4 examples/multi_backend_example.py \
        --mode train --backend florence-2 --data data.json \
        --distributed --dist-strategy fsdp

    # 显存优化训练（启用梯度检查点）
    python examples/multi_backend_example.py --mode train --backend florence-2 \
        --data data.json --checkpoint-strategy auto

    # 多后端对比推理
    python examples/multi_backend_example.py --mode compare --image test.jpg
"""

import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 后端特定的模型名称映射
MODEL_NAME_MAP = {
    "florence-2": "microsoft/Florence-2-large",
    "florence2": "microsoft/Florence-2-large",
    "paligemma": "google/paligemma-3b-pt-224",
    "paligemma-3b": "google/paligemma-3b-pt-224",
    "youtuvl": "tencent-YouTu/Youtu-VL-4B-Instruct",
    "youtu-vl": "tencent-YouTu/Youtu-VL-4B-Instruct",
    "auto": "tencent-YouTu/Youtu-VL-4B-Instruct",
    "generic-hf": "tencent-YouTu/Youtu-VL-4B-Instruct",
    "hf": "tencent-YouTu/Youtu-VL-4B-Instruct",
}


def create_config(backend_name: str, task_type: str, model_name: str = None,
                   use_gradient_checkpointing: bool = True,
                   checkpoint_strategy: str = "auto"):
    """创建模型配置

    Args:
        backend_name: 后端名称 ("florence-2", "paligemma", "youtuvl" 等)
        task_type: 任务类型 ("CAPTION", "OD", "VQA" 等)
        model_name: 可选，显式指定模型名称（覆盖默认映射）
        use_gradient_checkpointing: 是否启用梯度检查点
        checkpoint_strategy: 激活值重计算策略 (none / full / selective / auto)

    Returns:
        ModelConfig 实例
    """
    from florence_forge.core.config import ModelConfig, LoRAConfig

    resolved_name = model_name or MODEL_NAME_MAP.get(backend_name, backend_name)

    config = ModelConfig(
        model_name=resolved_name,
        backend_name=backend_name,
        use_lora=True,
        lora_config=LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
        ),
        trust_remote_code=True,
        # 显存优化配置
        gradient_checkpointing=use_gradient_checkpointing,
        activation_checkpointing_strategy=checkpoint_strategy,
    )

    return config


def create_dataset_config(backend_name: str):
    """创建数据配置

    不同后端的数据配置可以相同，因为 Dataset 层与后端解耦。
    但缓存目录建议按后端区分。
    """
    from florence_forge.core.config import DataConfig

    cache_dir = f"./cache/{backend_name}"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return DataConfig(
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        use_cache=True,
        cache_dir=cache_dir,
    )


def run_inference(backend_name: str, task_type: str, image_path: str):
    """运行推理示例

    Args:
        backend_name: 后端名称
        task_type: 任务类型
        image_path: 图像路径
    """
    from PIL import Image
    from florence_forge.core.model import Florence2MultiTaskModel

    logger.info(f"=" * 60)
    logger.info(f"后端: {backend_name} | 任务: {task_type}")
    logger.info(f"=" * 60)

    # 创建配置和模型
    config = create_config(backend_name, task_type)
    model = Florence2MultiTaskModel(config)
    model.load()  # 显式加载模型和处理器

    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 获取任务提示（后端自动处理格式差异）
    task_prompt = model._backend.get_task_prompt(task_type)
    logger.info(f"任务提示: {task_prompt}")

    # 执行推理
    result = model.predict_task(
        images=image,
        task_name=task_type,
        max_new_tokens=256,
        num_beams=3,
    )

    logger.info(f"推理结果: {result}")
    return result


def run_training(backend_name: str, data_path: str, output_dir: str,
                 distributed: bool = False, strategy: str = "ddp"):
    """运行训练示例

    Args:
        backend_name: 后端名称
        data_path: 数据路径
        output_dir: 输出目录
        distributed: 是否启用分布式训练
        strategy: 分布式策略 (ddp / fsdp / deepspeed)
    """
    from florence_forge.core.config import (
        TrainingConfig, OptimizationConfig, DistributedConfig
    )
    from florence_forge.core.model import Florence2MultiTaskModel
    from florence_forge.data.dataset import MultiTaskDataset
    from florence_forge.training.trainer import MultiTaskTrainer

    logger.info(f"=" * 60)
    logger.info(f"训练后端: {backend_name}")
    if distributed:
        logger.info(f"分布式策略: {strategy}")
    logger.info(f"=" * 60)

    # 创建配置
    model_config = create_config(backend_name, "CAPTION")
    data_config = create_dataset_config(backend_name)
    opt_config = OptimizationConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # 分布式配置
    dist_config = DistributedConfig(
        enabled=distributed,
        strategy=strategy,
    ) if distributed else DistributedConfig()

    training_config = TrainingConfig(
        num_epochs=3,
        model_config=model_config,
        data_config=data_config,
        optimization_config=opt_config,
        distributed_config=dist_config,
        output_dir=output_dir,
        logging_steps=10,
        save_steps=100,
    )

    # 创建模型
    model = Florence2MultiTaskModel(model_config)
    model.load()  # 显式加载模型和处理器

    # 创建数据集（这里需要真实数据）
    # 示例数据结构: [{"task_type": "CAPTION", "data_path": "data.json"}]
    data_configs = [
        {"task_type": "CAPTION", "data_path": data_path},
    ]

    dataset = MultiTaskDataset(
        data_configs=data_configs,
        config=data_config,
        processor=model.processor,
    )

    # 创建训练器
    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=dataset,
        config=training_config,
    )

    # 设置训练组件
    trainer.setup_training()

    # 开始训练
    trainer.train()

    logger.info(f"训练完成，模型保存到: {output_dir}")


def compare_backends(image_path: str):
    """对比不同后端的推理结果

    Args:
        image_path: 测试图像路径
    """
    backends = ["florence-2", "paligemma", "youtuvl"]
    task = "CAPTION"

    results = {}
    for backend in backends:
        try:
            result = run_inference(backend, task, image_path)
            results[backend] = result
        except Exception as e:
            logger.error(f"{backend} 推理失败: {e}")
            results[backend] = f"ERROR: {e}"

    logger.info("=" * 60)
    logger.info("对比结果")
    logger.info("=" * 60)
    for backend, result in results.items():
        logger.info(f"{backend}: {result}")


def main():
    parser = argparse.ArgumentParser(description="FlorenceForge 多后端示例")
    parser.add_argument(
        "--backend",
        type=str,
        default="florence-2",
        choices=[
            "florence-2", "florence2",
            "paligemma", "paligemma-3b",
            "youtuvl", "youtu-vl",
            "auto", "generic-hf", "hf"
        ],
        help="选择 VLM 后端",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="显式指定模型名称（覆盖 backend 的默认模型）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="CAPTION",
        help="任务类型 (CAPTION, OD, VQA 等)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="推理用的图像路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "train", "compare"],
        help="运行模式",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="训练数据路径（train 模式需要）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="输出目录",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="启用分布式训练（自动检测环境或配置）",
    )
    parser.add_argument(
        "--dist-strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "deepspeed"],
        help="分布式训练策略",
    )
    parser.add_argument(
        "--checkpoint-strategy",
        type=str,
        default="auto",
        choices=["none", "full", "selective", "auto"],
        help="激活值重计算策略（显存优化）",
    )

    args = parser.parse_args()

    # 如果指定了 --model，更新映射
    if args.model:
        MODEL_NAME_MAP[args.backend] = args.model

    if args.mode == "inference":
        if args.image is None:
            logger.error("推理模式需要提供 --image 参数")
            return
        run_inference(args.backend, args.task, args.image)

    elif args.mode == "train":
        if args.data is None:
            logger.error("训练模式需要提供 --data 参数")
            return
        run_training(
            args.backend, args.data, args.output,
            distributed=args.distributed,
            strategy=args.dist_strategy,
        )

    elif args.mode == "compare":
        if args.image is None:
            logger.error("对比模式需要提供 --image 参数")
            return
        compare_backends(args.image)


if __name__ == "__main__":
    main()
