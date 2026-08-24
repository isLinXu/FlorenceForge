#!/usr/bin/env python3
"""MoE CIFAR-10 四配置对比 Benchmark（接入 florence_forge.training.moe 正式实现）。

对比配置（plan.md Stage 2）：
1. dense           — 标准 Linear 分类头
2. moe_dense       — MoE 层，top_k=None（全部专家参与计算）
3. moe_sparse      — MoE 层，top_k=2，无 aux loss
4. moe_sparse_aux  — MoE 层，top_k=2，含 aux loss + z-loss

数据来源：HF datasets 本地缓存（离线）。为消除数据管线开销，首次运行时
将图像预处理（ToTensor + Normalize，无随机增强，保证四配置公平可比）
缓存为张量文件，后续运行直接加载。

用法：
    python scripts/benchmark/moe_cifar10_benchmark.py --config dense --epochs 15
    python scripts/benchmark/moe_cifar10_benchmark.py --config all --epochs 15

每个配置独立运行，结果写入 <output-dir>/<config>.json（含逐 epoch 历史）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from florence_forge.training.moe.moe_adapter import MoETrainingAdapter
from florence_forge.training.moe.moe_config import MoEConfig
from florence_forge.training.moe.moe_layer import MoELayer

CONFIGS = ["dense", "moe_dense", "moe_sparse", "moe_sparse_aux"]
NUM_CLASSES = 10
FEATURE_DIM = 256
NUM_EXPERTS = 8
TOP_K = 2
AUX_W = 0.05
Z_W = 0.001

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_or_load_tensors(data_dir: Path, cache_path: Path):
    """预处理 CIFAR-10 为内存张量（带磁盘缓存）。

    Returns:
        (train_x, train_y, test_x, test_y) 四个 CPU 张量。
    """
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        return blob["train_x"], blob["train_y"], blob["test_x"], blob["test_y"]

    from datasets import load_dataset

    transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
    hf = load_dataset("cifar10", "plain_text", cache_dir=str(data_dir / "hf_cache"))

    def to_tensors(split):
        n = len(hf[split])
        xs = torch.empty(n, 3, 32, 32)
        ys = torch.empty(n, dtype=torch.long)
        for i in range(n):
            item = hf[split][i]
            xs[i] = transform(item["img"])
            ys[i] = int(item["label"])
            if (i + 1) % 10000 == 0:
                print(f"  预处理 {split}: {i + 1}/{n}", flush=True)
        return xs, ys

    print("首次运行：预处理 CIFAR-10 为张量缓存 ...", flush=True)
    train_x, train_y = to_tensors("train")
    test_x, test_y = to_tensors("test")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"train_x": train_x, "train_y": train_y,
         "test_x": test_x, "test_y": test_y},
        cache_path,
    )
    print(f"张量缓存已写入 {cache_path}", flush=True)
    return train_x, train_y, test_x, test_y


class CIFARBackbone(nn.Module):
    """轻量 CNN 特征提取器（3 层 Conv）→ (B, FEATURE_DIM)。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),  # 128*2*2 = 512
            nn.Linear(512, FEATURE_DIM), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class DenseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = CIFARBackbone()
        self.head = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MoEClassifier(nn.Module):
    """CNN backbone + MoELayer 分类头（feature 视为单 token 序列）。"""

    def __init__(self, top_k: int | None, capacity_factor: float | None) -> None:
        super().__init__()
        self.backbone = CIFARBackbone()
        self.moe = MoELayer(
            num_experts=NUM_EXPERTS,
            d_model=FEATURE_DIM,
            d_state=NUM_CLASSES,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).unsqueeze(1)  # (B, 1, FEATURE_DIM)
        return self.moe(feat).squeeze(1)  # (B, NUM_CLASSES)


def build_model(config_name: str) -> tuple[nn.Module, MoETrainingAdapter | None, bool]:
    """构建模型；MoE 配置一律挂载 adapter 用于路由统计，仅 aux 配置应用辅助损失。

    Returns:
        (model, adapter_or_none, apply_aux_loss)
    """
    if config_name == "dense":
        return DenseClassifier(), None, False

    top_k = None if config_name == "moe_dense" else TOP_K
    capacity = None if config_name == "moe_dense" else 1.25
    model = MoEClassifier(top_k=top_k, capacity_factor=capacity)

    cfg = MoEConfig(
        num_experts=NUM_EXPERTS, d_model=FEATURE_DIM, d_state=NUM_CLASSES,
        top_k=TOP_K, aux_loss_weight=AUX_W, z_loss_weight=Z_W,
    )
    adapter = MoETrainingAdapter(cfg)
    adapter._moe_layers = [model.moe]
    return model, adapter, config_name == "moe_sparse_aux"


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             device: torch.device, batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        correct += (model(xb).argmax(dim=-1) == yb).sum().item()
    return correct / x.shape[0]


def routing_stats(adapter: MoETrainingAdapter | None) -> Dict[str, Any]:
    if adapter is None or not adapter.is_injected():
        return {}
    layer = adapter._moe_layers[0]
    stats: Dict[str, Any] = {
        "gini": float(adapter.get_routing_gini()),
        "overflow_tokens": adapter.get_total_overflow_tokens(),
    }
    if layer._routing_sums is not None:
        total = layer._routing_sums.sum().item()
        stats["expert_load_pct"] = [
            round(100 * v / max(total, 1e-9), 2)
            for v in layer._routing_sums.tolist()
        ]
    return stats


def run_config(config_name: str, args: argparse.Namespace,
               tensors) -> Dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_x, train_y, test_x, test_y = tensors

    if args.subset:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(train_x.shape[0], generator=g)[: args.subset]
        train_x, train_y = train_x[idx], train_y[idx]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model, adapter, apply_aux = build_model(config_name)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_acc = 0.0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if apply_aux and adapter is not None:
                loss = loss + adapter.get_auxiliary_loss(loss_weight=AUX_W) \
                    + adapter.get_router_z_loss(loss_weight=Z_W)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            ep_batches += 1

        acc = evaluate(model, test_x, test_y, device)
        best_acc = max(best_acc, acc)
        record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": round(ep_loss / max(ep_batches, 1), 4),
            "test_acc": round(acc, 4),
        }
        record.update(routing_stats(adapter))
        history.append(record)
        print(f"[{config_name}] epoch {epoch + 1}/{args.epochs} "
              f"loss={record['train_loss']:.4f} acc={acc:.4f}", flush=True)

    elapsed = time.perf_counter() - t0
    return {
        "config": config_name,
        "epochs": args.epochs,
        "subset": args.subset,
        "seed": args.seed,
        "device": str(device),
        "params": n_params,
        "final_test_acc": history[-1]["test_acc"],
        "best_test_acc": round(best_acc, 4),
        "train_time_sec": round(elapsed, 1),
        "sec_per_epoch": round(elapsed / args.epochs, 2),
        "final_routing": routing_stats(adapter),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS + ["all"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=None,
                        help="训练集子采样数量（默认全量 50000）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="experiments/moe_cifar10/data")
    parser.add_argument("--output-dir", default="experiments/moe_cifar10/results")
    parser.add_argument("--tensor-cache",
                        default="experiments/moe_cifar10/results/cifar10_tensors.pt")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = build_or_load_tensors(Path(args.data_dir), Path(args.tensor_cache))

    configs = CONFIGS if args.config == "all" else [args.config]
    for name in configs:
        print(f"=== 开始配置: {name} ===", flush=True)
        result = run_config(name, args, tensors)
        out_file = out_dir / f"{name}.json"
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"=== {name} 完成: best_acc={result['best_test_acc']:.4f} "
              f"time={result['train_time_sec']}s → {out_file} ===", flush=True)


if __name__ == "__main__":
    main()
