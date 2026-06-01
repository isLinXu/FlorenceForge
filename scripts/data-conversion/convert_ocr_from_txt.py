#!/usr/bin/env python3
"""
Florence Forge OCR数据转换工具

将包含图像文件名和OCR内容的txt文件转换为Florence-2格式

用法:
    python convert_ocr_from_txt.py <txt_file> <images_dir> <output_file> [--task-type OCR]

参数说明:
    txt_file: 包含图像文件名和OCR内容的txt文件，格式为: "图像文件名\tOCR内容"
    images_dir: 图像文件所在目录
    output_file: 输出的Florence-2格式文件
    --task-type: 任务类型，可选OCR或OCR_WITH_REGION (默认: OCR)

示例:
    python convert_ocr_from_txt.py ocr_data.txt ./images ./output.jsonl
    python convert_ocr_from_txt.py ocr_data.txt ./images ./output.jsonl --task-type OCR_WITH_REGION

txt文件格式示例:
    0-浙NJVJLH.jpg\t浙NJVJLH
    1-辽GM06R4.jpg\t辽GM06R4
    2-川G3LGWX.jpg\t川G3LGWX
    3-陕TBB442.jpg\t陕TBB442
"""

import argparse
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_txt_file(txt_file: str) -> List[Tuple[str, str]]:
    """解析txt文件，返回(图像文件名, OCR内容)的列表"""
    data: List[Tuple[str, str]] = []

    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"警告: 第{line_num}行格式不正确，跳过: {line}")
                    continue

                image_name, ocr_content = parts
                data.append((image_name.strip(), ocr_content.strip()))

    except FileNotFoundError:
        print(f"错误: 找不到txt文件: {txt_file}")
        sys.exit(1)
    except Exception as exc:
        print(f"错误: 读取txt文件时出错: {exc}")
        sys.exit(1)

    return data


def create_text_files(data: List[Tuple[str, str]], temp_dir: str) -> int:
    """在临时目录中为每个图像创建独立的 .txt 文本文件。"""
    count = 0

    for image_name, ocr_content in data:
        base_name = Path(image_name).stem
        txt_file_path = Path(temp_dir) / f"{base_name}.txt"

        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(ocr_content)

            print(f"处理: {image_name} -> {base_name}.txt")
            count += 1

        except Exception as exc:
            print(f"警告: 创建文本文件失败 {txt_file_path}: {exc}")

    return count


def run_florence_cli(images_dir: str, texts_dir: str, output_file: str, task_type: str) -> bool:
    """调用 florence_forge_cli convert ocr 完成最终转换。"""
    cmd = [
        'florence_forge_cli', 'convert', 'ocr',
        '--images-dir', images_dir,
        '--texts-dir', texts_dir,
        '--output', output_file,
        '--task-type', task_type,
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ OCR数据转换成功!")
        print(f"输出文件: {output_file}")
        return True

    except subprocess.CalledProcessError as exc:
        print("❌ OCR数据转换失败!")
        print(f"错误代码: {exc.returncode}")
        print(f"错误输出: {exc.stderr}")
        return False

    except FileNotFoundError:
        print("❌ 找不到florence_forge_cli命令!")
        print("请确保已正确安装Florence Forge并且florence_forge_cli在PATH中")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description='将txt格式的OCR数据转换为Florence-2格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_ocr_from_txt.py ocr_data.txt ./images ./output.jsonl
  python convert_ocr_from_txt.py ocr_data.txt ./images ./output.jsonl --task-type OCR_WITH_REGION
""",
    )

    parser.add_argument('txt_file', help='包含图像文件名和OCR内容的txt文件')
    parser.add_argument('images_dir', help='图像文件所在目录')
    parser.add_argument('output_file', help='输出的Florence-2格式文件')
    parser.add_argument(
        '--task-type',
        choices=['OCR', 'OCR_WITH_REGION'],
        default='OCR',
        help='任务类型 (默认: OCR)',
    )

    args = parser.parse_args()

    if not os.path.isfile(args.txt_file):
        print(f"错误: 找不到txt文件: {args.txt_file}")
        sys.exit(1)

    if not os.path.isdir(args.images_dir):
        print(f"错误: 找不到图像目录: {args.images_dir}")
        sys.exit(1)

    print(f"正在解析txt文件: {args.txt_file}")
    data = parse_txt_file(args.txt_file)

    if not data:
        print("错误: txt文件中没有有效数据")
        sys.exit(1)

    print(f"找到 {len(data)} 条OCR数据")

    temp_dir = tempfile.mkdtemp(prefix='florence_ocr_')
    print(f"创建临时文本目录: {temp_dir}")

    try:
        print("正在创建单独的文本文件...")
        count = create_text_files(data, temp_dir)
        print(f"文本文件创建完成，共 {count} 个文件")

        if count == 0:
            print("错误: 没有成功创建任何文本文件")
            sys.exit(1)

        print("开始使用florence_forge_cli进行OCR数据转换...")
        success = run_florence_cli(
            args.images_dir,
            temp_dir,
            args.output_file,
            args.task_type,
        )

        if not success:
            sys.exit(1)

    finally:
        print("清理临时文件...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("转换完成!")


if __name__ == '__main__':
    main()
