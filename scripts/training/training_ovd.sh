#!/bin/bash

# 禁用MPS设备，强制使用CPU
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# export CUDA_VISIBLE_DEVICES=""
# export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=0

# 禁用tokenizers并行化警告
# export TOKENIZERS_PARALLELISM=false

# 运行目标检测训练
# python -m florence_forge.cli train \
#     --task od \
#     --config /Users/gatilin/Downloads/MIRA/code/florence_forge/configs/examples/object_detection_training.yaml \
#     --train-data /Users/gatilin/Downloads/MIRA/data/coco128/coco128_od.jsonl \
#     --val-data /Users/gatilin/Downloads/MIRA/data/coco128/coco8_od.jsonl \
#     --output-dir /Users/gatilin/Downloads/MIRA/outputs/florence2_object_detection1 \
#     --epochs 2 \
#     --batch-size 16 \
#     --learning-rate 1e-5


# image_dir=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/images/train
# # image_dir=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/images/test

# # train_data=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_test.jsonl

# train_data=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_fix.jsonl
# val_data=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_test.jsonl
# val_image_dir=/svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/images/test
# python3 -m florence_forge.cli train \


# usage: florence_forge_cli train [-h]
#                                 [--task {caption,detailed_caption,more_detailed_caption,detection,od,open_vocabulary_detection,phrase_grounding,dense_region_caption,region_proposal,region_to_category,region_to_description,ocr,ocr_with_region,segmentation,seg,region_to_segmentation,referring_expression_segmentation,multitask,multi}]
#                                 [--config CONFIG] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--output-dir OUTPUT_DIR]
#                                 [--model MODEL] [--train-data TRAIN_DATA] [--val-data VAL_DATA]
#                                 [--device {auto,cpu,cuda,cuda:0,cuda:1,cuda:2,cuda:3,mps}]

# options:
#   -h, --help            show this help message and exit
#   --task {caption,detailed_caption,more_detailed_caption,detection,od,open_vocabulary_detection,phrase_grounding,dense_region_caption,region_proposal,region_to_category,region_to_description,ocr,ocr_with_region,segmentation,seg,region_to_segmentation,referring_expression_segmentation,multitask,multi}, -t {caption,detailed_caption,more_detailed_caption,detection,od,open_vocabulary_detection,phrase_grounding,dense_region_caption,region_proposal,region_to_category,region_to_description,ocr,ocr_with_region,segmentation,seg,region_to_segmentation,referring_expression_segmentation,multitask,multi}
#                         任务类型
#   --config CONFIG, -c CONFIG
#                         配置文件路径
#   --epochs EPOCHS, -e EPOCHS
#                         训练轮数
#   --batch-size BATCH_SIZE, -b BATCH_SIZE
#                         批次大小
#   --lr LR, --learning-rate LR
#                         学习率
#   --output-dir OUTPUT_DIR, -o OUTPUT_DIR
#                         输出目录
#   --model MODEL, -m MODEL
#                         模型名称
#   --train-data TRAIN_DATA
#                         训练数据文件路径
#   --val-data VAL_DATA   验证数据文件路径
#   --device {auto,cpu,cuda,cuda:0,cuda:1,cuda:2,cuda:3,mps}, -d {auto,cpu,cuda,cuda:0,cuda:1,cuda:2,cuda:3,mps}
#                         训练设备 (默认: auto)

# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# python3 -m florence_forge.cli.main train \
#     --task open_vocabulary_detection \
#     --config /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/florence_forge/configs/examples/object_detection_training.yaml \
#     --train-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_fix.jsonl \
#     --val-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_test.jsonl \
#     --output-dir /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_ovd \
#     --epochs 1 \
#     --batch-size 32 \
#     --learning-rate 1e-5


python3 -m florence_forge.cli.main train \
    --task open_vocabulary_detection \
    --config /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/florence_forge/configs/examples/object_detection_training.yaml \
    --train-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_fix.jsonl \
    --val-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_test.jsonl \
    --output-dir /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_ovd1 \
    --epochs 2 \
    --batch-size 32 \
    --learning-rate 1e-5
    # --epochs 2 \
    # --batch-size 16 \
    # --learning-rate 1e-5