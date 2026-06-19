#!/usr/bin/env python3
"""后训练全流程验证脚本

验证训练完成后的完整工作流：
  1. 模拟训练 → 保存 checkpoint
  2. 加载 checkpoint → 恢复模型状态
  3. 推理（InferenceEngine）→ 单图 + 批量
  4. 评估（MultiTaskEvaluator）→ 指标计算
  5. 部署（FastAPI ModelServer）→ 服务启动验证

使用 mock 模型避免下载真实 Florence-2，聚焦后训练管线本身的正确性。
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 抑制实验性 MoE 警告
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 颜色输出 ────────────────────────────────────────────────────────
PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
SKIP = "\033[93m⚠️  SKIP\033[0m"
INFO = "\033[94mℹ️ \033[0m"


def log_step(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════
#  1. 模拟训练 + 保存 Checkpoint
# ════════════════════════════════════════════════════════════════════

def step_train_and_save() -> Path:
    """创建一个 mock 训练流程并保存 checkpoint。"""
    from florence_forge.training.checkpoint_manager import CheckpointManager
    from florence_forge.core.config import TrainingConfig

    log_step("Step 1: 模拟训练 → 保存 Checkpoint")

    output_dir = Path(tempfile.mkdtemp(prefix="ff_posttrain_"))
    config = TrainingConfig()
    config.output_dir = str(output_dir)
    config.num_epochs = 1
    config.device = "cpu"

    # mock 模型：简单的 seq2seq 风格
    model = nn.Sequential(
        nn.Embedding(100, 32),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 100),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    # 模拟一步训练
    model.train()
    input_ids = torch.randint(0, 100, (2, 10))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    optimizer.step()

    # 保存 checkpoint
    manager = CheckpointManager(model=model, config=config, accelerator=None)
    manager.save_checkpoint(
        epoch=0,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics={"loss": float(loss.item()), "accuracy": 0.85},
        is_best=True,
        async_save=False,
    )

    checkpoint_dir = output_dir / "checkpoint-epoch-0"
    assert checkpoint_dir.exists(), "checkpoint 目录未创建"
    assert (checkpoint_dir / "checkpoint.pt").exists(), "checkpoint.pt 未创建"
    assert (checkpoint_dir / "BEST_MODEL").exists(), "BEST_MODEL 标记未创建"

    print(f"{PASS} 训练完成，checkpoint 保存至 {checkpoint_dir}")
    return checkpoint_dir


# ════════════════════════════════════════════════════════════════════
#  2. 加载 Checkpoint
# ════════════════════════════════════════════════════════════════════

def step_load_checkpoint(checkpoint_dir: Path) -> nn.Module:
    """从 checkpoint 加载模型并验证状态恢复。"""
    from florence_forge.training.checkpoint_manager import CheckpointManager
    from florence_forge.core.config import TrainingConfig

    log_step("Step 2: 加载 Checkpoint → 恢复模型状态")

    config = TrainingConfig()
    config.output_dir = str(checkpoint_dir.parent)
    config.device = "cpu"

    new_model = nn.Sequential(
        nn.Embedding(100, 32),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 100),
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    manager = CheckpointManager(model=new_model, config=config, accelerator=None)
    metadata = manager.load_checkpoint(
        checkpoint_path=checkpoint_dir,
        optimizer=new_optimizer,
    )

    assert metadata["epoch"] == 0, f"epoch 恢复错误: {metadata['epoch']}"
    assert "loss" in metadata["metrics"], "metrics 未恢复"
    print(f"{PASS} Checkpoint 加载成功，epoch={metadata['epoch']}, metrics={metadata['metrics']}")
    return new_model


# ════════════════════════════════════════════════════════════════════
#  3. 保存最终模型 → 推理
# ════════════════════════════════════════════════════════════════════

def step_inference(checkpoint_dir: Path) -> None:
    """将 checkpoint 保存为可推理模型，并用 InferenceEngine 执行预测。"""
    from florence_forge.training.checkpoint_manager import load_model_only
    from florence_forge.deployment.inference import InferenceEngine

    log_step("Step 3: 保存最终模型 → 推理")

    # 构建完整模型并加载权重
    model = nn.Sequential(
        nn.Embedding(100, 32),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 100),
    )
    metadata = load_model_only(model, checkpoint_dir / "checkpoint.pt", device="cpu")
    model.eval()

    # 保存为最终模型目录
    final_dir = checkpoint_dir.parent / "final_model"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    print(f"{INFO} 最终模型保存至 {final_dir}")

    # 创建 mock 推理引擎（不加载 Florence-2，直接用 torch 模型）
    # InferenceEngine 内部调用 predict_pil_image -> forward_tensor，输入为 float 图像张量 [1, 3, H, W]
    # 需要 mock 模型接受 float 张量而非整数索引
    class MockVLM(nn.Module):
        def __init__(self):
            super().__init__()
            # 接受 [B, 3, H, W] 浮点图像张量
            self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 100)

        def forward(self, x):
            # x: [B, 3, H, W]
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)  # [B, 32, 1, 1]
            x = x.view(x.size(0), -1)  # [B, 32]
            return self.fc(x)  # [B, 100]

        def generate(self, images, **kwargs):
            # images 可以是 PIL Image 或 tensor
            with torch.no_grad():
                if hasattr(images, "mode"):
                    # PIL Image -> tensor
                    img = np.array(images.convert("RGB"))
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    tensor = tensor.unsqueeze(0).to(next(self.parameters()).device)
                else:
                    tensor = images
                logits = self.forward(tensor)
            return torch.argmax(logits, dim=-1).unsqueeze(1)

        def decode(self, token_ids, **kwargs):
            return [f"token_{t.item()}" for t in token_ids.flatten()]

    mock_model = MockVLM()
    mock_model.eval()

    # 由于 InferenceEngine 需要 model 有特定接口，这里直接验证 predict 核心逻辑
    engine = InferenceEngine(mock_model, device="cpu", batch_size=1)

    # 创建 mock 图像输入
    mock_image = Image.new("RGB", (224, 224), color="red")

    # 执行预测（虽然 mock 模型不是真正 VLM，但验证推理引擎可以运行）
    result = engine.predict(mock_image, task_prompt="<CAPTION>")
    assert result is not None, "推理结果为空"

    # 验证统计信息
    stats = engine.get_stats()
    assert stats["total_inferences"] >= 1, "推理计数未更新"
    print(f"{PASS} 推理成功: result={result}, stats={stats}")


# ════════════════════════════════════════════════════════════════════
#  4. 评估（MultiTaskEvaluator）
# ════════════════════════════════════════════════════════════════════

def step_evaluation() -> None:
    """使用 mock 评估数据验证 MultiTaskEvaluator 指标计算。"""
    from florence_forge.evaluation.evaluator import MultiTaskEvaluator
    from florence_forge.data.dataset import MultiTaskDataset

    log_step("Step 4: 评估 → MultiTaskEvaluator")

    # 构建 mock 模型（具备 generate + decode 接口，匹配 evaluator 调用签名）
    class MockEvalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.processor = None

        def generate(self, input_ids, pixel_values, attention_mask=None, **kwargs):
            # mock 生成：总是返回固定 token
            batch_size = pixel_values.shape[0] if hasattr(pixel_values, "shape") else 1
            return torch.tensor([[42]] * batch_size)

        def decode(self, token_ids, **kwargs):
            return ["a mock caption" for _ in token_ids]

    model = MockEvalModel()
    evaluator = MultiTaskEvaluator(model, device="cpu")

    # 构建 MultiTaskDataset 而非 raw list
    eval_data = [
        {
            "image": Image.new("RGB", (224, 224), color="blue"),
            "prefix": "<CAPTION>",
            "suffix": "a blue square",
        },
        {
            "image": Image.new("RGB", (224, 224), color="green"),
            "prefix": "<CAPTION>",
            "suffix": "a green circle",
        },
    ]
    # 将图像保存为临时文件，JSONL 中存储路径
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_jsonl = tmp_dir / "eval.jsonl"
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for i, row in enumerate(eval_data):
            img_path = tmp_dir / f"img_{i}.png"
            row["image"].save(img_path)
            f.write(
                json.dumps(
                    {"image": str(img_path), "prefix": row["prefix"], "suffix": row["suffix"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    # 构建 mock processor，模拟 Florence-2 processor 的编码接口
    class MockProcessor:
        def __init__(self):
            self.tokenizer = MockTokenizer()
            self.pad_token_id = 0

        def __call__(self, text, images=None, return_tensors="pt"):
            # 简单编码：每个字符映射为整数
            input_ids = torch.tensor([[ord(c) % 1000 for c in text]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            result = {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
            }
            if images is not None:
                # 将图像转换为固定尺寸的 tensor
                if hasattr(images, "mode"):
                    img = images.convert("RGB").resize((224, 224))
                elif isinstance(images, str):
                    img = Image.open(images).convert("RGB").resize((224, 224))
                else:
                    img = images
                arr = np.array(img).astype(np.float32) / 255.0
                pixel_values = torch.from_numpy(arr).permute(2, 0, 1)
                result["pixel_values"] = pixel_values
            return result

    class MockTokenizer:
        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            input_ids = torch.tensor([[ord(c) % 1000 for c in text]], dtype=torch.long)
            return {"input_ids": input_ids}

    processor = MockProcessor()

    dataset = MultiTaskDataset(
        data_configs=[{"task_type": "CAPTION", "data_path": str(tmp_jsonl), "weight": 1.0}],
        processor=processor,
    )

    # 由于 DataLoader 子进程中 processor 不可 pickle，需要设 num_workers=0
    results = evaluator.evaluate_dataset(dataset, num_workers=0)
    assert "overall_metrics" in results or "task_metrics" in results, "评估结果结构异常"
    print(f"{PASS} 评估完成: {list(results.keys())}")


# ════════════════════════════════════════════════════════════════════
#  5. 部署服务启动验证（FastAPI）
# ════════════════════════════════════════════════════════════════════

def step_deploy_service() -> None:
    """验证 FastAPI 服务可以正常构造和启动。"""
    from florence_forge.deployment.server import ModelServer, FASTAPI_AVAILABLE
    from florence_forge.deployment.inference import InferenceEngine

    log_step("Step 5: 部署 → FastAPI ModelServer 启动验证")

    if not FASTAPI_AVAILABLE:
        print(f"{SKIP} FastAPI 不可用，跳过部署验证")
        return

    # 创建 mock 推理引擎
    class MockServiceModel(nn.Module):
        def generate(self, input_ids, **kwargs):
            return torch.tensor([[1, 2, 3]])

    mock_engine = InferenceEngine(MockServiceModel(), device="cpu", batch_size=1)

    try:
        server = ModelServer(
            inference_engine=mock_engine,
            host="127.0.0.1",
            port=18000,
        )
        # 验证 app 对象已创建
        assert hasattr(server, "app"), "ModelServer 未创建 app"
        print(f"{PASS} ModelServer 构造成功，app 可用")
    except Exception as e:
        print(f"{FAIL} ModelServer 构造失败: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  FlorenceForge 后训练全流程验证")
    print("  Post-Training Pipeline Verification")
    print("=" * 60)

    start_time = time.time()
    errors: List[str] = []

    steps = [
        ("train_and_save", step_train_and_save),
        ("load_checkpoint", step_load_checkpoint),
        ("inference", step_inference),
        ("evaluation", step_evaluation),
        ("deploy_service", step_deploy_service),
    ]

    checkpoint_dir: Path | None = None

    for name, step_fn in steps:
        try:
            if name == "train_and_save":
                checkpoint_dir = step_fn()
            elif name == "load_checkpoint":
                model = step_fn(checkpoint_dir)
            elif name == "inference":
                step_fn(checkpoint_dir)
            else:
                step_fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"{FAIL} {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  验证完成")
    print(f"{'='*60}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  通过步骤: {len(steps) - len(errors)}/{len(steps)}")
    if errors:
        print(f"  失败步骤:")
        for err in errors:
            print(f"    - {err}")
        return 1
    print(f"  {PASS} 所有步骤通过！")
    return 0


if __name__ == "__main__":
    # 确保项目根目录在 Python 路径中（无需安装）
    import os
    project_root = Path(__file__).parent.parent
    os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(project_root))
    sys.exit(main())
