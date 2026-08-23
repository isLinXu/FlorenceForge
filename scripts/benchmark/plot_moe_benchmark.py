#!/usr/bin/env python3
"""MoE CIFAR-10 benchmark 结果可视化：准确率曲线 / 效率对比 / 专家负载。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS = Path("experiments/moe_cifar10/results")
OUT = Path("experiments/moe_cifar10/results")

CONFIG_LABELS = {
    "dense": "Dense 基线",
    "moe_dense": "MoE-dense (top-k=全部)",
    "moe_sparse": "MoE-sparse (top-k=2)",
    "moe_sparse_aux": "MoE-sparse+aux (top-k=2)",
}
ORDER = ["dense", "moe_dense", "moe_sparse", "moe_sparse_aux"]
PALETTE = {
    "dense": "#6b7280",
    "moe_dense": "#2563eb",
    "moe_sparse": "#dc2626",
    "moe_sparse_aux": "#16a34a",
}


def load_results():
    results = {}
    for name in ORDER:
        results[name] = json.loads((RESULTS / f"{name}.json").read_text())
    return results


def fig_learning_curves(results):
    rows = []
    for name in ORDER:
        for h in results[name]["history"]:
            rows.append({
                "epoch": h["epoch"],
                "test_acc": h["test_acc"] * 100,
                "config": CONFIG_LABELS[name],
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=df, x="epoch", y="test_acc", hue="config", style="config",
        markers=True, dashes=False, palette=[PALETTE[c] for c in ORDER],
        hue_order=[CONFIG_LABELS[c] for c in ORDER],
        style_order=[CONFIG_LABELS[c] for c in ORDER],
        linewidth=2.2, markersize=7, ax=ax,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("MoE CIFAR-10 四配置学习曲线（subset=10240, seed=42, MPS）")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(title="配置", loc="lower right")
    fig.savefig(OUT / "benchmark_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency(results):
    df = pd.DataFrame([
        {
            "config": CONFIG_LABELS[c],
            "best_acc": results[c]["best_test_acc"] * 100,
            "time": results[c]["train_time_sec"],
            "params_k": results[c]["params"] / 1000,
        }
        for c in ORDER
    ])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    sns.barplot(data=df, x="config", y="best_acc", ax=axes[0],
                palette=[PALETTE[c] for c in ORDER], hue="config", legend=False)
    axes[0].set_title("最佳测试准确率 (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Best Acc (%)")
    for i, v in enumerate(df["best_acc"]):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    sns.barplot(data=df, x="config", y="time", ax=axes[1],
                palette=[PALETTE[c] for c in ORDER], hue="config", legend=False)
    axes[1].set_title("训练耗时 (秒, 10 epochs)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("秒")
    for i, v in enumerate(df["time"]):
        axes[1].text(i, v + 3, f"{v:.0f}s", ha="center", fontsize=10)

    sns.barplot(data=df, x="config", y="params_k", ax=axes[2],
                palette=[PALETTE[c] for c in ORDER], hue="config", legend=False)
    axes[2].set_title("参数量 (K)")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("K 参数")
    for i, v in enumerate(df["params_k"]):
        axes[2].text(i, v + 2, f"{v:.0f}K", ha="center", fontsize=10)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("精度 / 效率 / 参数量 三维对比", y=1.02)
    fig.savefig(OUT / "benchmark_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_expert_load(results):
    rows = []
    for name in ["moe_dense", "moe_sparse", "moe_sparse_aux"]:
        loads = results[name]["final_routing"].get("expert_load_pct")
        if not loads:
            continue
        for e, pct in enumerate(loads):
            rows.append({
                "expert": f"E{e}",
                "load": pct,
                "config": CONFIG_LABELS[name],
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="expert", y="load", hue="config", ax=ax,
                palette=[PALETTE[c] for c in ["moe_dense", "moe_sparse", "moe_sparse_aux"]])
    ax.axhline(100 / 8, color="black", linestyle="--", linewidth=1,
               label="理想均衡 (12.5%)")
    ax.set_xlabel("专家")
    ax.set_ylabel("最终批次负载占比 (%)")
    ax.set_title("专家负载分布（最后一个训练批次的路由统计）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(OUT / "benchmark_expert_load.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_plot()
    results = load_results()
    fig_learning_curves(results)
    fig_efficiency(results)
    fig_expert_load(results)
    print("charts saved to", OUT)


if __name__ == "__main__":
    main()
