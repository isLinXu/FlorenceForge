#!/bin/bash

# Florence-2 视觉原语增强任务 LoRA Training 脚本
# 适用于 OD_VP / COUNT_VP / PHRASE_GROUNDING_VP 的 Layer 1 MVP

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 禁用tokenizers并行化警告
export TOKENIZERS_PARALLELISM=false

# 配置文件路径
CONFIG_PATH="configs/examples/visual_primitive_training.yaml"

# 数据路径（请先使用 convert vp-coco-od / vp-coco-count / vp-yolo-od / vp-yolo-count 生成）
TRAIN_DATA="${TRAIN_DATA:-/path/to/your/train_visual_primitive_data.jsonl}"
VAL_DATA="${VAL_DATA:-/path/to/your/val_visual_primitive_data.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/florence2_visual_primitive_lora}"

# 训练参数
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

echo "开始 Florence-2 视觉原语增强 LoRA Training..."
echo "配置文件: $CONFIG_PATH"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "训练模式: Visual Primitive LoRA MVP"
echo "=============================================="

python3 -m florence_forge.cli.main train \
    --task visual_primitive \
    --config "$CONFIG_PATH" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE"

echo "训练完成。"
echo "LoRA适配器保存在: $OUTPUT_DIR"
