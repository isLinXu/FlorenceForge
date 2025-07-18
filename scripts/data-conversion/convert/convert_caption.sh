



# json_file=/svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/Florence-2-Fine-tuning-main/MIRA/data/coco_caption/captions_train2017.json
# images_dir=/svap_storage/dataset/SVAP_Public_Datasets/coco/images/train2017
# output_file=/svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/Florence-2-Fine-tuning-main/MIRA/data/coco_caption/coco_train_caption.jsonl


json_file=/svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/Florence-2-Fine-tuning-main/MIRA/data/coco_caption/captions_val2017.json
images_dir=/svap_storage/dataset/SVAP_Public_Datasets/coco/images/train2017
output_file=/svap_storage/gatilin/workspaces/working/GatilinLAB/vlm_learning/Florence-2-Fine-tuning-main/MIRA/data/coco_caption/coco_val_caption.jsonl


# florence-forge convert coco-caption --json-file /Users/gatilin/Downloads/MIRA/code/florence_forge/data/annotations/captions_train2017.json --images-dir /Users/gatilin/Downloads/MIRA/code/florence_forge/data/train2017 --output /Users/gatilin/Downloads/MIRA/code/florence_forge/data/annotations/train_caption.jsonl
florence_forge_cli convert coco-caption --json-file $json_file --images-dir $images_dir --output $output_file 