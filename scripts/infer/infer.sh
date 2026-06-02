# florence_forge_cli infer
# florence_forge_cli infer --help 
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results --visualize
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results1 --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results2 --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results3 --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results4 --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results5 --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OD>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results6 --visualize --save-visualizations


# florence_forge_cli infer --task-prompt "<CAPTION>" --model /svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_object_detection1/final_model/lora_adapters --input /svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test --output ./results6 --visualize --save-visualizations

# caption
# MODEL_FILE=/svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/FlorenceForge/outputs/florence2_caption/final_model/lora_adapters
# IMAGE_DIR=/svap_storage/gatilin/data/datasets/ped_car_datasets/nomotor_bike_3cls_label_datasets/images/test
# IMAGE_DIR=/Users/gatilin/PycharmProjects/yolo-lab/data/qishu_sample/A车牌所有字符清晰可见


IMAGE_DIR=/Users/gatilin/PycharmProjects/PaddleX/meizi_images/topic_小甜美/unknown
# OUTPUT_DIR=./results_caption
# OUTPUT_DIR=./results_captio1
# OUTPUT_DIR=./results_caption2
# OUTPUT_DIR=./results_caption3

# florence_forge_cli infer --task-prompt "<CAPTION>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations

MODEL_FILE=microsoft/Florence-2-base


# OUTPUT_DIR=./results_region
# OUTPUT_DIR=./results_ocr_region
# OUTPUT_DIR=./results_ocr_region1
# OUTPUT_DIR=./results_ocr_region2

# OUTPUT_DIR=./results_category_region
# OUTPUT_DIR=./results_category_region1
# OUTPUT_DIR=./results_category_region2
# OUTPUT_DIR=./results_category_region3
# OUTPUT_DIR=./results_category_region4

# OUTPUT_DIR=./results_ovd
# OUTPUT_DIR=./results_ovd1
# florence_forge_cli infer --task-prompt "<OPEN_VOCABULARY_DETECTION>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations

# REGION_PROPOSAL
# florence_forge_cli infer --task-prompt "<REGION_TO_SEGMENTATION>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<REGION_PROPOSAL>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<OCR_WITH_REGION>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations
# florence_forge_cli infer --task-prompt "<REGION_TO_CATEGORY>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations

# OUTPUT_DIR=./results_caption
OUTPUT_DIR=./images_caption

florence_forge_cli infer --task-prompt "<CAPTION>" --model $MODEL_FILE --input $IMAGE_DIR --output $OUTPUT_DIR --visualize --save-visualizations




# OUTPUT_DIR=./results_ovd
# IMAGE_DIR=./images
# MODEL_FILE=./model

# florence_forge_cli infer \
#     --task-prompt "<OPEN_VOCABULARY_DETECTION>" \
#     --model $MODEL_FILE \
#     --input $IMAGE_DIR \
#     --output $OUTPUT_DIR \
#     --text-input "a car" \
#     --visualize \
#     --save-visualizations