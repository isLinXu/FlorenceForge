#!/bin/bash

# Florence Forge OCR数据转换脚本
# 用法: ./convert_ocr.sh <txt_file> <images_dir> <output_file>
# 
# 参数说明:
# txt_file: 包含图像文件名和OCR内容的txt文件，格式为: "图像文件名\tOCR内容"
# images_dir: 图像文件所在目录
# output_file: 输出的Florence-2格式文件

# 检查参数
if [ $# -ne 3 ]; then
    echo "用法: $0 <txt_file> <images_dir> <output_file>"
    echo "示例: $0 ocr_data.txt ./images ./output.jsonl"
    echo ""
    echo "txt文件格式示例:"
    echo "0-浙NJVJLH.jpg\t浙NJVJLH"
    echo "1-辽GM06R4.jpg\t辽GM06R4"
    exit 1
fi

TXT_FILE="$1"
IMAGES_DIR="$2"
OUTPUT_FILE="$3"

# 检查输入文件是否存在
if [ ! -f "$TXT_FILE" ]; then
    echo "错误: 找不到txt文件: $TXT_FILE"
    exit 1
fi

# 检查图像目录是否存在
if [ ! -d "$IMAGES_DIR" ]; then
    echo "错误: 找不到图像目录: $IMAGES_DIR"
    exit 1
fi

# 创建临时目录来存放单独的文本文件
TEMP_TEXTS_DIR=$(mktemp -d)
echo "创建临时文本目录: $TEMP_TEXTS_DIR"

# 解析txt文件并创建单独的文本文件
echo "正在解析txt文件并创建单独的文本文件..."
while IFS=$'\t' read -r image_name ocr_content; do
    # 跳过空行
    if [ -z "$image_name" ]; then
        continue
    fi
    
    # 获取不带扩展名的文件名
    base_name=$(basename "$image_name" | sed 's/\.[^.]*$//')
    
    # 创建对应的文本文件
    echo "$ocr_content" > "$TEMP_TEXTS_DIR/${base_name}.txt"
    
    echo "处理: $image_name -> ${base_name}.txt"
done < "$TXT_FILE"

echo "文本文件创建完成，共$(ls -1 "$TEMP_TEXTS_DIR" | wc -l)个文件"

# 使用florence_forge_cli进行转换
echo "开始使用florence_forge_cli进行OCR数据转换..."
florence_forge_cli convert ocr \
    --images-dir "$IMAGES_DIR" \
    --texts-dir "$TEMP_TEXTS_DIR" \
    --output "$OUTPUT_FILE" \
    --task-type OCR

# 检查转换是否成功
if [ $? -eq 0 ]; then
    echo "✅ OCR数据转换成功!"
    echo "输出文件: $OUTPUT_FILE"
else
    echo "❌ OCR数据转换失败!"
    exit 1
fi

# 清理临时目录
echo "清理临时文件..."
rm -rf "$TEMP_TEXTS_DIR"

echo "转换完成!"