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

# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python3 -m florence_forge.cli.main train \
    --task od \
    --config /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/florence_forge/configs/examples/object_detection_training.yaml \
    --train-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_fix.jsonl \
    --val-data /svap_storage/gatilin/data/datasets/svap_unidet_train/youtu_vehicle_car_ped_nomotor_3cls/labels/od_yolo_test.jsonl \
    --output-dir /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1 \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 1e-5
    # --epochs 2 \
    # --batch-size 16 \
    # --learning-rate 1e-5