#!/bin/bash

# Florence-2 多任务训练 LoRA Training 脚本
# 使用 LoRA 训练模式（轻量级微调）

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 禁用tokenizers并行化警告
export TOKENIZERS_PARALLELISM=false

# 配置文件路径
CONFIG_PATH="configs/examples/multitask_training.yaml"

# 数据路径（请根据实际情况修改）
TRAIN_DATA="/path/to/your/train_multitask_data.jsonl"
VAL_DATA="/path/to/your/val_multitask_data.jsonl"

# 输出目录
OUTPUT_DIR="./outputs/florence2_multitask_lora"

# 训练参数
EPOCHS=8
BATCH_SIZE=1  # 多任务训练使用小batch size
LEARNING_RATE=1.5e-5

echo "开始 Florence-2 多任务训练 LoRA Training..."
echo "配置文件: $CONFIG_PATH"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "训练模式: LoRA Training (轻量级微调)"
echo "="*50

# 执行训练
python3 -m florence_forge.cli.main train \
    --task multitask \
    --config "$CONFIG_PATH" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE

echo "训练完成！"
echo "LoRA适配器保存在: $OUTPUT_DIR"