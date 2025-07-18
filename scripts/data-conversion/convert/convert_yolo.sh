# florence_forge_cli convert yolo --help
# usage: florence_forge_cli convert yolo [-h] --labels-dir LABELS_DIR --images-dir IMAGES_DIR --classes-file CLASSES_FILE --output OUTPUT
#                                        [--image-ext IMAGE_EXT] [--task-type TASK_TYPE]
# labels_dir=/Users/gatilin/Downloads/MIRA/data/coco128/labels/train2017
# images_dir=/Users/gatilin/Downloads/MIRA/data/coco128/images/train2017
# classes_file=/Users/gatilin/Downloads/MIRA/data/coco128/classes.txt
# output=/Users/gatilin/Downloads/MIRA/data/coco128/coco128_od.jsonl

labels_dir=/Users/gatilin/Downloads/MIRA/data/coco8/labels/train
images_dir=/Users/gatilin/Downloads/MIRA/data/coco8/images/train
classes_file=/Users/gatilin/Downloads/MIRA/data/coco128/classes.txt
output=/Users/gatilin/Downloads/MIRA/data/coco128/coco8_od.jsonl
image_ext=.jpg
task_type=OD
florence_forge_cli convert yolo --labels-dir $labels_dir --images-dir $images_dir --classes-file $classes_file --output $output --task-type $task_type

# options:
#   -h, --help            show this help message and exit
#   --labels-dir LABELS_DIR
#                         YOLO标签文件目录
#   --images-dir IMAGES_DIR
#                         图像文件目录
#   --classes-file CLASSES_FILE
#                         类别文件路径
#   --output OUTPUT, -o OUTPUT
#                         输出文件路径
#   --image-ext IMAGE_EXT
#                         图像文件扩展名
#   --task-type TASK_TYPE
#                         任务类型