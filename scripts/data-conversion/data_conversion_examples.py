#!/usr/bin/env python3
"""
Florence Forge 数据转换示例脚本

展示如何使用florence_forge_cli的数据转换功能
将各种格式的数据转换为Florence-2训练格式
"""

import sys
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: list, description: str) -> bool:
    """运行命令并处理结果"""
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ {description} 成功完成")
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败")
        logger.error(f"错误: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"❌ 找不到florence_forge_cli命令")
        logger.error("请确保已正确安装florence_forge并且CLI在PATH中")
        return False

def example_yolo_conversion():
    """YOLO格式转换示例"""
    logger.info("\n=== YOLO格式转换示例 ===")
    
    # 示例命令
    cmd = [
        'florence_forge_cli', 'convert', 'yolo',
        '--labels-dir', './data/yolo/labels',
        '--images-dir', './data/yolo/images', 
        '--classes-file', './data/yolo/classes.txt',
        '--output', './data/converted/yolo_to_florence2.jsonl',
        '--image-ext', '.jpg',
        '--task-type', 'OD'
    ]
    
    return run_command(cmd, "YOLO到Florence-2格式转换")

def example_coco_conversion():
    """COCO格式转换示例"""
    logger.info("\n=== COCO格式转换示例 ===")
    
    cmd = [
        'florence_forge_cli', 'convert', 'coco',
        '--json-file', './data/coco/annotations/instances_train2017.json',
        '--images-dir', './data/coco/images/train2017',
        '--output', './data/converted/coco_to_florence2.jsonl'
    ]
    
    return run_command(cmd, "COCO到Florence-2格式转换")

def example_csv_conversion():
    """CSV格式转换示例"""
    logger.info("\n=== CSV格式转换示例 ===")
    
    cmd = [
        'florence_forge_cli', 'convert', 'csv',
        '--csv-file', './data/captions/image_captions.csv',
        '--output', './data/converted/csv_to_florence2.jsonl',
        '--image-column', 'image_path',
        '--caption-column', 'caption',
        '--task-type', 'CAPTION'
    ]
    
    return run_command(cmd, "CSV到Florence-2格式转换")

def example_xml_conversion():
    """VOC XML格式转换示例"""
    logger.info("\n=== VOC XML格式转换示例 ===")
    
    cmd = [
        'florence_forge_cli', 'convert', 'xml',
        '--xml-dir', './data/voc/annotations',
        '--images-dir', './data/voc/images',
        '--output', './data/converted/xml_to_florence2.jsonl'
    ]
    
    return run_command(cmd, "VOC XML到Florence-2格式转换")

def example_ocr_conversion():
    """OCR数据转换示例"""
    logger.info("\n=== OCR数据转换示例 ===")
    
    cmd = [
        'florence_forge_cli', 'convert', 'ocr',
        '--images-dir', './data/ocr/images',
        '--texts-dir', './data/ocr/texts',
        '--output', './data/converted/ocr_to_florence2.jsonl',
        '--task-type', 'OCR'
    ]
    
    return run_command(cmd, "OCR数据到Florence-2格式转换")

def create_sample_data():
    """创建示例数据文件结构"""
    logger.info("\n=== 创建示例数据结构 ===")
    
    # 创建目录结构
    directories = [
        './data/yolo/labels',
        './data/yolo/images',
        './data/coco/annotations',
        './data/coco/images/train2017',
        './data/captions',
        './data/voc/annotations',
        './data/voc/images',
        './data/ocr/images',
        './data/ocr/texts',
        './data/converted'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    
    # 创建示例类别文件
    classes_content = "person\ncar\nbicycle\ndog\ncat\n"
    with open('./data/yolo/classes.txt', 'w') as f:
        f.write(classes_content)
    logger.info("创建示例类别文件: ./data/yolo/classes.txt")
    
    # 创建示例CSV文件
    csv_content = "image_path,caption\n./images/img1.jpg,A person walking in the park\n./images/img2.jpg,A red car on the street\n"
    with open('./data/captions/image_captions.csv', 'w') as f:
        f.write(csv_content)
    logger.info("创建示例CSV文件: ./data/captions/image_captions.csv")
    
    logger.info("✅ 示例数据结构创建完成")

def show_conversion_help():
    """显示转换功能帮助信息"""
    logger.info("\n=== Florence Forge 数据转换帮助 ===")
    
    cmd = ['florence_forge_cli', 'convert', '--help']
    run_command(cmd, "显示转换功能帮助")
    
    # 显示各个子命令的帮助
    for convert_type in ['yolo', 'coco', 'csv', 'xml', 'ocr']:
        logger.info(f"\n--- {convert_type.upper()}转换帮助 ---")
        cmd = ['florence_forge_cli', 'convert', convert_type, '--help']
        run_command(cmd, f"显示{convert_type}转换帮助")

def main():
    """主函数"""
    logger.info("Florence Forge 数据转换示例脚本")
    logger.info("=" * 50)
    
    # 检查参数
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == 'help':
            show_conversion_help()
        elif action == 'setup':
            create_sample_data()
        elif action == 'yolo':
            example_yolo_conversion()
        elif action == 'coco':
            example_coco_conversion()
        elif action == 'csv':
            example_csv_conversion()
        elif action == 'xml':
            example_xml_conversion()
        elif action == 'ocr':
            example_ocr_conversion()
        elif action == 'all':
            # 运行所有转换示例
            create_sample_data()
            success_count = 0
            total_count = 5
            
            if example_yolo_conversion(): success_count += 1
            if example_coco_conversion(): success_count += 1
            if example_csv_conversion(): success_count += 1
            if example_xml_conversion(): success_count += 1
            if example_ocr_conversion(): success_count += 1
            
            logger.info(f"\n=== 转换完成统计 ===")
            logger.info(f"成功: {success_count}/{total_count}")
            
        else:
            logger.error(f"未知操作: {action}")
            print_usage()
    else:
        print_usage()

def print_usage():
    """打印使用说明"""
    print("\n使用方法:")
    print("  python data_conversion_examples.py <action>")
    print("\n可用操作:")
    print("  help   - 显示转换功能帮助信息")
    print("  setup  - 创建示例数据结构")
    print("  yolo   - 运行YOLO转换示例")
    print("  coco   - 运行COCO转换示例")
    print("  csv    - 运行CSV转换示例")
    print("  xml    - 运行XML转换示例")
    print("  ocr    - 运行OCR转换示例")
    print("  all    - 运行所有转换示例")
    print("\n示例:")
    print("  python data_conversion_examples.py setup")
    print("  python data_conversion_examples.py yolo")
    print("  python data_conversion_examples.py all")

if __name__ == '__main__':
    main()