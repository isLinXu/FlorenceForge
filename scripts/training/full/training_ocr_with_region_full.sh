#!/bin/bash

# Florence-2 OCR带区域任务 Full Training 脚本
# 使用 Full Training 模式（非LoRA）

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 禁用tokenizers并行化警告
export TOKENIZERS_PARALLELISM=false

# 配置文件路径
CONFIG_PATH="configs/full/ocr_with_region_training.yaml"

# 数据路径（请根据实际情况修改）
TRAIN_DATA="/path/to/your/train_ocr_with_region_data.jsonl"
VAL_DATA="/path/to/your/val_ocr_with_region_data.jsonl"

# 输出目录
OUTPUT_DIR="./outputs/florence2_ocr_with_region_full"

# 训练参数
EPOCHS=5
BATCH_SIZE=1  # Full training使用小batch size
LEARNING_RATE=1e-5

echo "开始 Florence-2 OCR带区域任务 Full Training..."
echo "配置文件: $CONFIG_PATH"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "训练模式: Full Training (非LoRA)"
echo "="*50

# 执行训练
python3 -m florence_forge.cli.main train \
    --task ocr_with_region \
    --config "$CONFIG_PATH" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE

echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"