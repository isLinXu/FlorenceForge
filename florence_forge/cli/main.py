#!/usr/bin/env python3
"""
Florence Forge CLI - 命令行接口

提供便捷的命令行接口来运行不同任务的训练配置

使用示例:
    # 运行图像描述任务训练
    florence_forge_cli train --task caption --config configs/examples/caption_training.yaml
    
    # 运行目标检测任务训练
    florence_forge_cli train --task detection --config configs/examples/object_detection_training.yaml
    
    # 运行多任务训练
    florence_forge_cli train --task multitask --config configs/examples/multitask_training.yaml
    
    # 列出所有可用的任务和配置
    florence_forge_cli list-tasks
    
    # 验证配置文件
    florence_forge_cli validate --config configs/examples/caption_training.yaml
    
    # 生成配置模板
    florence_forge_cli generate-config --task ocr --output my_ocr_config.yaml
"""

import argparse
import sys
import os
from typing import Dict, Any
from pathlib import Path

# 在导入torch之前设置MPS设备配置
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# 移除强制CPU设置，允许使用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import yaml
from datetime import datetime

from florence_forge.utils.diagnostics import DEFAULT_MODEL_ID, collect_environment_diagnostics

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 共享常量 / 纯辅助函数（抽离至 _helpers，便于复用且避免循环导入）。
# 这些名称在此重新导出，保持 `florence_forge.cli.main.xxx` 历史导入路径兼容。
from ._helpers import (  # noqa: E402
    TASK_CONFIG_MAPPING,
    TASK_DESCRIPTIONS,
)

# 重型子命令 handler（抽离至 commands，保持本文件聚焦于参数解析与调度）。
# 同样重新导出以兼容历史导入路径（含测试中对这些函数的直接 import）。
from .commands import (  # noqa: E402
    run_agentic_task,
    run_data_conversion,
    run_eval_task,
    run_inference_task,
    run_serve_task,
    run_training_task,
    run_tvp_training_task,
)


def _print_doctor_report(report: Dict[str, Any]) -> None:
    """Print a concise human-readable environment diagnostic report."""
    status = "OK" if report["ok"] else "ISSUES"
    print(f"\n=== Florence Forge Doctor: {status} ===")

    platform_info = report["platform"]
    print(
        "Platform: "
        f"Python {platform_info['python']} / "
        f"{platform_info['system']} {platform_info['release']} / "
        f"{platform_info['machine']}"
    )

    torch_info = report["torch"]
    print(
        "Torch: "
        f"{torch_info.get('version') or 'missing'} | "
        f"selected={torch_info['selected_device']} "
        f"available={torch_info['selected_device_available']}"
    )
    print(
        "Devices: "
        f"MPS={torch_info['mps_available']} "
        f"(built={torch_info['mps_built']}), "
        f"CUDA={torch_info['cuda_available']} "
        f"(count={torch_info['cuda_device_count']})"
    )

    model_info = report["model"]
    if model_info["local_snapshot_exists"]:
        print(f"Model cache: {model_info['local_snapshot']}")
    else:
        print(f"Model cache: missing local snapshot for {model_info['model_id']}")

    missing_required = report.get("missing_required", [])
    if missing_required:
        print("Missing required deps: " + ", ".join(missing_required))
    else:
        print("Required deps: OK")

    optional_missing = [
        dep["package"]
        for dep in report["dependencies"]
        if not dep["required"] and not dep["available"]
    ]
    if optional_missing:
        print("Missing optional deps: " + ", ".join(optional_missing))

    if report.get("warnings"):
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    print(f"Recommended dtype: {report['recommended_torch_dtype']}")
    print(f"Smoke command: {report['suggested_smoke_command']}")


def run_doctor_task(args) -> bool:
    """Run lightweight environment diagnostics."""
    import json

    report = collect_environment_diagnostics(
        requested_device=getattr(args, "device", "auto"),
        model_id=getattr(args, "model_id", DEFAULT_MODEL_ID),
        model_path=getattr(args, "model_path", None),
        require_model=getattr(args, "require_model", False),
    )

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_doctor_report(report)

    return bool(report["ok"])


def setup_cli_logging(verbose: bool = False) -> None:
    """设置CLI日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

def list_available_tasks() -> None:
    """列出所有可用的任务和配置"""
    print("\n=== Florence Forge 可用任务 ===")
    print()

    try:
        from florence_forge.core.tasks import FLORENCE2_TASKS, TaskCategory
    except ImportError as e:
        logger.warning(f"无法导入任务注册表，回退到预定义任务列表: {e}")
        FLORENCE2_TASKS = {}
        TaskCategory = []
    
    if not FLORENCE2_TASKS:
        print("\n⚠️  无可用任务列表:")
        predefined_tasks = {
            'CAPTION': '图像描述生成',
            'DETAILED_CAPTION': '详细图像描述',
            'MORE_DETAILED_CAPTION': '更详细图像描述',
            'OD': '目标检测',
            'OPEN_VOCABULARY_DETECTION': '开放词汇检测',
            'CAPTION_TO_PHRASE_GROUNDING': '短语定位',
            'DENSE_REGION_CAPTION': '密集区域描述',
            'REGION_PROPOSAL': '区域提议',
            'REGION_TO_CATEGORY': '区域到类别',
            'REGION_TO_DESCRIPTION': '区域到描述',
            'OCR': '光学字符识别',
            'OCR_WITH_REGION': '带区域的OCR',
            'REGION_TO_SEGMENTATION': '区域到分割',
            'REFERRING_EXPRESSION_SEGMENTATION': '参考表达式分割'
        }
        
        categories = {
            '图像描述': ['CAPTION', 'DETAILED_CAPTION', 'MORE_DETAILED_CAPTION'],
            '目标检测': ['OD', 'OPEN_VOCABULARY_DETECTION', 'CAPTION_TO_PHRASE_GROUNDING', 'REGION_PROPOSAL'],
            '区域分析': ['DENSE_REGION_CAPTION', 'REGION_TO_CATEGORY', 'REGION_TO_DESCRIPTION'],
            '文字识别': ['OCR', 'OCR_WITH_REGION'],
            '图像分割': ['REGION_TO_SEGMENTATION', 'REFERRING_EXPRESSION_SEGMENTATION']
        }
        
        for category, task_list in categories.items():
            print(f"\n📂 {category}:")
            for task_name in task_list:
                description = predefined_tasks.get(task_name, '无描述')
                print(f"  • {task_name}: {description}")
        
        print(f"\n总计: {len(predefined_tasks)} 个任务")
    else:
        # 按类别分组显示Florence-2原生任务
        print("📋 Florence-2 原生任务:")
        for category in TaskCategory:
            tasks = [name for name, config in FLORENCE2_TASKS.items() 
                    if config.category == category]
            if tasks:
                print(f"  {category.value}:")
                for task in tasks:
                    desc = FLORENCE2_TASKS[task].description
                    print(f"    - {task}: {desc}")
    
    print("\n🎯 预配置训练任务:")
    for task_key, description in TASK_DESCRIPTIONS.items():
        config_path = TASK_CONFIG_MAPPING[task_key]
        print(f"  - {task_key}: {description}")
        print(f"    配置文件: {config_path}")
    
    print("\n💡 使用示例:")
    print("  florence_forge_cli train --task caption")
    print("  florence_forge_cli train --task detection --epochs 10")
    print("  florence_forge_cli train --config custom_config.yaml")

def validate_config(config_path: str) -> bool:
    """验证配置文件"""
    try:
        from florence_forge.core.config import TrainingConfig

        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"❌ 配置文件不存在: {config_path}")
            return False
        
        # 尝试加载YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            logger.error(f"❌ 配置文件为空: {config_path}")
            return False
        
        # 基础验证
        required_sections = ['model_config', 'data_config', 'optimization_config']
        missing_sections = []
        
        for section in required_sections:
            if section not in config_data:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"❌ 配置文件缺少必需的部分: {missing_sections}")
            return False
        
        # 进行更详细的验证
        try:
            training_config = TrainingConfig.from_dict(config_data)
            logger.info(f"✅ 配置文件验证通过: {config_path}")
            logger.info(f"   实验名称: {training_config.experiment_name}")
            logger.info(f"   模型: {training_config.model_settings.model_name}")
            logger.info(f"   训练轮数: {training_config.num_epochs}")
            logger.info(f"   批次大小: {training_config.data_settings.batch_size}")
            logger.info(f"   学习率: {training_config.optimization_settings.learning_rate}")
            return True
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")
            return False
            
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML格式错误: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 验证配置文件时出错: {e}")
        return False

def generate_config_template(task: str, output_path: str) -> bool:
    """生成配置模板"""
    try:
        if task not in TASK_CONFIG_MAPPING:
            logger.error(f"❌ 未知任务类型: {task}")
            logger.info(f"可用任务: {', '.join(TASK_CONFIG_MAPPING.keys())}")
            return False
        
        template_path = TASK_CONFIG_MAPPING[task]
        
        # 查找模板文件
        possible_paths = [
            Path(template_path),
            Path.cwd() / template_path,
            Path(__file__).parent.parent / template_path
        ]
        
        template_file = None
        for path in possible_paths:
            if path.exists():
                template_file = path
                break
        
        if not template_file:
            logger.error(f"❌ 找不到模板文件: {template_path}")
            return False
        
        # 复制模板
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_file, 'r', encoding='utf-8') as src:
            content = src.read()
        
        # 修改实验名称和输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = content.replace(
            f'experiment_name: "florence2_{task}_training"',
            f'experiment_name: "florence2_{task}_training_{timestamp}"'
        )
        content = content.replace(
            f'output_dir: "./outputs/florence2_{task}"',
            f'output_dir: "./outputs/florence2_{task}_{timestamp}"'
        )
        
        with open(output_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        logger.info(f"✅ 配置模板已生成: {output_file}")
        logger.info(f"   基于模板: {template_file}")
        logger.info(f"   任务类型: {task}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 生成配置模板时出错: {e}")
        return False


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog='florence_forge_cli',
        description='Florence Forge - Florence-2多任务微调工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练相关
  florence_forge_cli train --task caption
  florence_forge_cli train --task detection --epochs 10
  florence_forge_cli train --config custom_config.yaml
  
  # 推理测试
  florence_forge_cli infer --model ./models/best.pth --input ./test_image.jpg --output ./results
  florence_forge_cli infer --model ./models/best.pth --input ./test_images/ --output ./results --batch-size 4
  florence_forge_cli infer --model ./models/best.pth --input ./test_images/ --output ./results --device cuda --use-amp
  
  # 配置管理
  florence_forge_cli doctor --device mps --require-model
  florence_forge_cli list-tasks
  florence_forge_cli validate --config configs/examples/caption_training.yaml
  florence_forge_cli generate-config --task ocr --output my_config.yaml
  
  # 数据转换
  florence_forge_cli convert yolo --labels-dir ./labels --images-dir ./images --classes-file ./classes.txt --output ./data.jsonl
  florence_forge_cli convert coco --json-file ./annotations.json --images-dir ./images --output ./data.jsonl
  florence_forge_cli convert csv --csv-file ./captions.csv --output ./data.jsonl
  florence_forge_cli convert xml --xml-dir ./annotations --images-dir ./images --output ./data.jsonl
  florence_forge_cli convert ocr --images-dir ./images --texts-dir ./texts --output ./data.jsonl
  
  # VP (Visual Primitive) 数据转换
  florence_forge_cli convert vp-coco --json-file ./annotations.json --images-dir ./images --output ./vp_data.jsonl
  florence_forge_cli convert vp-yolo --labels-dir ./labels --images-dir ./images --classes-file ./classes.txt --output ./vp_data.jsonl
  
  # 结构化VP推理
  florence_forge_cli infer --model ./models/best.pth --input ./test.jpg --output ./results --structured-vp
  
  # 推理服务
  florence_forge_cli serve --model ./models/best.pth --port 8000
  florence_forge_cli serve --model ./models/best.pth --host 0.0.0.0 --port 8080 --device cuda
  
  # 模型评估
  florence_forge_cli eval --model ./models/best.pth --data ./eval_data.jsonl
  florence_forge_cli eval --model ./models/best.pth --data ./eval_data.jsonl --output results.json

更多信息请访问: https://github.com/florenceforge/florence-forge
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 环境诊断命令
    doctor_parser = subparsers.add_parser('doctor', help='检查本地运行环境')
    doctor_parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='要检查的目标设备 (默认: auto)'
    )
    doctor_parser.add_argument(
        '--model-id',
        default=DEFAULT_MODEL_ID,
        help=f'要检查的Hugging Face模型ID (默认: {DEFAULT_MODEL_ID})'
    )
    doctor_parser.add_argument(
        '--model-path',
        help='显式检查的本地模型目录或文件'
    )
    doctor_parser.add_argument(
        '--require-model',
        action='store_true',
        help='本地模型快照不存在时返回失败'
    )
    doctor_parser.add_argument(
        '--json',
        action='store_true',
        help='以JSON格式输出诊断报告'
    )
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='运行训练任务')
    train_parser.add_argument(
        '--task', '-t',
        choices=list(TASK_CONFIG_MAPPING.keys()),
        help='任务类型'
    )
    train_parser.add_argument(
        '--config', '-c',
        help='配置文件路径'
    )
    train_parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='训练轮数'
    )
    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='批次大小'
    )
    train_parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        help='学习率'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        help='输出目录'
    )
    train_parser.add_argument(
        '--override',
        action='append',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        help='覆盖任意配置项 (例如: --override data_config.batch_size 32)'
    )
    train_parser.add_argument(
        '--model', '-m',
        help='模型名称'
    )
    train_parser.add_argument(
        '--train-data',
        help='训练数据文件路径'
    )
    train_parser.add_argument(
        '--val-data',
        help='验证数据文件路径'
    )
    train_parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='训练设备 (默认: auto)'
    )
    train_parser.add_argument(
        '--resume', '-r',
        help='从检查点恢复训练 (检查点目录路径)'
    )
    train_parser.add_argument(
        '--tvp-config',
        help='TVP 阶段 YAML 配置（如 configs/tvp/sft.yaml），走 MultiTaskTrainer 桥接训练'
    )
    train_parser.add_argument(
        '--tvp-pipeline',
        help='TVP 三阶段 pipeline YAML（如 configs/tvp/pipeline.yaml）'
    )
    train_parser.add_argument(
        '--tvp-stage',
        choices=['sft', 'opd', 'grpo'],
        help='仅运行指定 TVP 阶段（需配合 --tvp-config）'
    )
    
    # 列出任务命令
    subparsers.add_parser('list-tasks', help='列出所有可用任务')
    
    # 验证配置命令
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument(
        '--config', '-c',
        required=True,
        help='要验证的配置文件路径'
    )
    
    # 生成配置模板命令
    generate_parser = subparsers.add_parser('generate-config', help='生成配置模板')
    generate_parser.add_argument(
        '--task', '-t',
        required=True,
        choices=list(TASK_CONFIG_MAPPING.keys()),
        help='任务类型'
    )
    generate_parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出文件路径'
    )
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='运行模型推理')
    infer_parser.add_argument(
        '--model', '-m',
        required=True,
        help='训练好的模型文件路径 (.pt, .pth)'
    )
    infer_parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入图像文件或目录路径'
    )
    infer_parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出结果目录'
    )
    infer_parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='推理设备 (默认: auto)'
    )
    infer_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='批处理大小 (默认: 1)'
    )
    infer_parser.add_argument(
        '--use-amp',
        action='store_true',
        help='使用自动混合精度加速推理'
    )
    infer_parser.add_argument(
        '--task-prompt',
        default='<OD>',
        help='Florence2模型的任务提示 (默认: <OD> 目标检测)'
    )
    infer_parser.add_argument(
        '--text-input',
        type=str,
        required=False,  # 明确设置为非必需，因为只有特定任务需要
        help="为需要文本输入的任务提供文本，例如开放词汇检测中的类别名称"
    )
    infer_parser.add_argument(
        '--visualize',
        action='store_true',
        help='在原图上可视化检测结果'
    )
    infer_parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='保存可视化结果到文件（默认只显示）'
    )
    # Structured VP decoding options
    infer_parser.add_argument(
        '--structured-vp',
        action='store_true',
        help='Enable structured visual primitive decoding',
    )
    infer_parser.add_argument(
        '--structured-vp-mode',
        choices=['auto', 'decode', 'off'],
        default='auto',
        help='Structured VP mode: auto/decode/off',
    )
    infer_parser.add_argument(
        '--structured-vp-decode',
        action='store_true',
        default=False,
        help='Force VP decoding',
    )
    infer_parser.add_argument(
        '--vp-box-format',
        default='loc_tokens',
        choices=['loc_tokens', 'json', 'quad'],
        help='VP decode bbox format (default: loc_tokens)',
    )
    infer_parser.add_argument(
        '--structured-vp-marker-style',
        '--vp-marker-style',
        dest='structured_vp_marker_style',
        default='special',
        choices=['special', 'angle_bracket', 'plain'],
        help='VP decode marker style (default: special)',
    )
    infer_parser.add_argument(
        '--structured-vp-max-boxes-per-label',
        '--vp-max-boxes-per-label',
        dest='structured_vp_max_boxes_per_label',
        type=int,
        default=None,
        help='VP decode: max boxes per label',
    )
    infer_parser.add_argument(
        '--structured-vp-max-total-boxes',
        dest='structured_vp_max_total_boxes',
        type=int,
        default=None,
        help='VP decode: max total boxes',
    )
    infer_parser.add_argument(
        '--structured-vp-filter-policy',
        dest='structured_vp_filter_policy',
        choices=['none', 'nms', 'score'],
        default='none',
        help='VP decode: box filter policy (default: none)',
    )
    infer_parser.add_argument(
        '--structured-vp-nms-iou-threshold',
        '--vp-nms-iou-threshold',
        dest='structured_vp_nms_iou_threshold',
        type=float,
        default=None,
        help='VP decode: NMS IoU threshold',
    )
    infer_parser.add_argument(
        '--structured-vp-allowed-labels',
        dest='structured_vp_allowed_labels',
        default=None,
        help='VP decode: allowed labels (comma-separated)',
    )
    
    # 数据转换命令
    convert_parser = subparsers.add_parser('convert', help='数据格式转换')
    convert_subparsers = convert_parser.add_subparsers(dest='convert_type', help='转换类型')
    
    # YOLO转换
    yolo_parser = convert_subparsers.add_parser('yolo', help='YOLO格式转换为Florence-2格式')
    yolo_parser.add_argument('--labels-dir', required=True, help='YOLO标签文件目录')
    yolo_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    yolo_parser.add_argument('--classes-file', required=True, help='类别文件路径')
    yolo_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    yolo_parser.add_argument('--image-ext', default='.jpg', help='图像文件扩展名')
    yolo_parser.add_argument('--task-type', default='OD', help='任务类型')
    
    # COCO转换
    coco_parser = convert_subparsers.add_parser('coco', help='COCO格式转换为Florence-2格式')
    coco_parser.add_argument('--json-file', required=True, help='COCO JSON文件路径')
    coco_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    coco_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    
    # COCO Caption转换
    coco_caption_parser = convert_subparsers.add_parser('coco-caption', help='COCO Caption格式转换为Florence-2格式')
    coco_caption_parser.add_argument('--json-file', required=True, help='COCO JSON文件路径')
    coco_caption_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    coco_caption_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    
    # CSV转换
    csv_parser = convert_subparsers.add_parser('csv', help='CSV格式转换为Florence-2格式')
    csv_parser.add_argument('--csv-file', required=True, help='CSV文件路径')
    csv_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    csv_parser.add_argument('--image-column', default='image', help='图像列名')
    csv_parser.add_argument('--caption-column', default='caption', help='标题列名')
    csv_parser.add_argument(
        '--task-type',
        default='CAPTION',
        choices=['CAPTION',
        'DETAILED_CAPTION',
        'MORE_DETAILED_CAPTION'],
        help='任务类型'
    )
    
    # VOC XML转换
    xml_parser = convert_subparsers.add_parser('xml', help='VOC XML格式转换为Florence-2格式')
    xml_parser.add_argument('--xml-dir', required=True, help='XML文件目录')
    xml_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    xml_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    
    # OCR转换
    ocr_parser = convert_subparsers.add_parser('ocr', help='OCR数据转换为Florence-2格式')
    ocr_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    ocr_parser.add_argument('--texts-dir', required=True, help='文本文件目录')
    ocr_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    ocr_parser.add_argument(
        '--task-type',
        default='OCR',
        choices=['OCR',
        'OCR_WITH_REGION'],
        help='任务类型'
    )
    
    # OCR TXT文件转换
    ocr_txt_parser = convert_subparsers.add_parser('ocr-txt', help='TXT文件OCR数据转换为Florence-2格式')
    ocr_txt_parser.add_argument('--txt-file', required=True, help='TXT文件路径（格式：图像文件名\tOCR内容）')
    ocr_txt_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    ocr_txt_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    ocr_txt_parser.add_argument(
        '--task-type',
        default='OCR',
        choices=['OCR',
        'OCR_WITH_REGION'],
        help='任务类型'
    )

    # ── VP (Visual Primitive) 数据转换 ──────────────────────────────
    vp_coco_parser = convert_subparsers.add_parser(
        'vp-coco', help='COCO标注转视觉原语(VP)格式',
    )
    vp_coco_parser.add_argument('--json-file', required=True, help='COCO JSON文件路径')
    vp_coco_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    vp_coco_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    vp_coco_parser.add_argument(
        '--vp-task', default='OD_VP',
        choices=['OD_VP', 'COUNT_VP', 'PHRASE_GROUNDING_VP', 'COUNT_VP_COT'],
        help='VP/TVP任务类型 (默认: OD_VP)',
    )
    vp_coco_parser.add_argument(
        '--box-format', default='json',
        choices=['json', 'loc_tokens', 'quad'],
        help='VP bbox格式 (默认: json; quad 为 json 别名)',
    )
    vp_coco_parser.add_argument(
        '--counting-mode', default='coarse',
        choices=['coarse', 'fine'],
        help='COUNT_VP_COT 计数思维链模式 (默认: coarse)',
    )
    vp_coco_parser.add_argument(
        '--marker-style', default='special',
        choices=['special', 'angle_bracket'],
        help='VP标记风格 (默认: special)',
    )

    vp_yolo_parser = convert_subparsers.add_parser(
        'vp-yolo', help='YOLO标注转视觉原语(VP)格式',
    )
    vp_yolo_parser.add_argument('--labels-dir', required=True, help='YOLO标签文件目录')
    vp_yolo_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    vp_yolo_parser.add_argument('--classes-file', required=True, help='类别文件路径')
    vp_yolo_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    vp_yolo_parser.add_argument(
        '--vp-task', default='OD_VP',
        choices=['OD_VP', 'COUNT_VP', 'PHRASE_GROUNDING_VP'],
        help='VP任务类型 (默认: OD_VP)',
    )
    vp_yolo_parser.add_argument(
        '--box-format', default='json',
        choices=['json', 'loc_tokens', 'quad'],
        help='VP bbox格式 (默认: json; quad 为 json 别名)',
    )
    vp_yolo_parser.add_argument(
        '--marker-style', default='special',
        choices=['special', 'angle_bracket'],
        help='VP标记风格 (默认: special)',
    )
    vp_yolo_parser.add_argument('--image-ext', default='.jpg', help='图像文件扩展名')

    def _add_tvp_jsonl_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = convert_subparsers.add_parser(name, help=help_text)
        parser.add_argument('--input', required=True, help='输入 JSONL 文件路径')
        parser.add_argument('--images-dir', required=True, help='图像文件目录')
        parser.add_argument('--output', '-o', required=True, help='输出 JSONL 文件路径')
        parser.add_argument(
            '--marker-style', default='special',
            choices=['special', 'angle_bracket'],
            help='VP标记风格 (默认: special)',
        )
        return parser

    tvp_count_parser = convert_subparsers.add_parser(
        'tvp-count-cot', help='COCO 标注转 TVP 计数思维链格式',
    )
    tvp_count_parser.add_argument('--json-file', required=True, help='COCO JSON 文件路径')
    tvp_count_parser.add_argument('--images-dir', required=True, help='图像文件目录')
    tvp_count_parser.add_argument('--output', '-o', required=True, help='输出 JSONL 文件路径')
    tvp_count_parser.add_argument(
        '--counting-mode', default='coarse',
        choices=['coarse', 'fine'],
        help='计数思维链模式 (默认: coarse)',
    )
    tvp_count_parser.add_argument(
        '--marker-style', default='special',
        choices=['special', 'angle_bracket'],
        help='VP标记风格 (默认: special)',
    )

    tvp_maze_parser = _add_tvp_jsonl_parser('tvp-maze', '迷宫 JSONL 转 TVP point 思维链')
    tvp_path_parser = _add_tvp_jsonl_parser('tvp-path', '路径追踪 JSONL 转 TVP point 思维链')
    tvp_spatial_parser = _add_tvp_jsonl_parser('tvp-spatial', '空间推理 JSONL 转 TVP 思维链')

    generate_tvp_maze_parser = convert_subparsers.add_parser(
        'generate-tvp-maze',
        help='合成迷宫数据（PNG + 原始 JSONL）',
    )
    generate_tvp_maze_parser.add_argument('--output-dir', '-o', required=True, help='输出目录')
    generate_tvp_maze_parser.add_argument('--num-samples', type=int, default=100, help='样本数量')
    generate_tvp_maze_parser.add_argument('--rows', type=int, default=8, help='迷宫行数')
    generate_tvp_maze_parser.add_argument('--cols', type=int, default=8, help='迷宫列数')
    generate_tvp_maze_parser.add_argument('--seed', type=int, default=42, help='随机种子')

    generate_tvp_path_parser = convert_subparsers.add_parser(
        'generate-tvp-path',
        help='合成路径追踪数据（PNG + 原始 JSONL）',
    )
    generate_tvp_path_parser.add_argument('--output-dir', '-o', required=True, help='输出目录')
    generate_tvp_path_parser.add_argument('--num-samples', type=int, default=100, help='样本数量')
    generate_tvp_path_parser.add_argument('--seed', type=int, default=42, help='随机种子')

    generate_tvp_spatial_parser = convert_subparsers.add_parser(
        'generate-tvp-spatial',
        help='合成空间推理数据（PNG + 原始 JSONL）',
    )
    generate_tvp_spatial_parser.add_argument('--output-dir', '-o', required=True, help='输出目录')
    generate_tvp_spatial_parser.add_argument('--num-samples', type=int, default=100, help='样本数量')
    generate_tvp_spatial_parser.add_argument('--seed', type=int, default=42, help='随机种子')

    generate_tvp_all_parser = convert_subparsers.add_parser(
        'generate-tvp-all',
        help='一次性合成 maze/path/spatial 三类 TVP 原始数据',
    )
    generate_tvp_all_parser.add_argument('--output-dir', '-o', required=True, help='输出根目录')
    generate_tvp_all_parser.add_argument('--num-samples', type=int, default=8, help='每类样本数量')
    generate_tvp_all_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # ── VP sub-type convert commands (test-aligned) ───────────────
    vp_coco_od_parser = convert_subparsers.add_parser(
        'vp-coco-od', help='COCO OD to VP format (OD_VP task)',
    )
    vp_coco_od_parser.add_argument('--json-file', required=True, help='COCO JSON file path')
    vp_coco_od_parser.add_argument('--images-dir', required=True, help='Image directory')
    vp_coco_od_parser.add_argument('--output', '-o', required=True, help='Output file path')
    vp_coco_od_parser.add_argument(
        '--task-type', default='OD_VP',
        help='Task type (default: OD_VP)',
    )
    vp_coco_od_parser.add_argument(
        '--box-format', default='json',
        choices=['json', 'loc_tokens', 'quad'],
        help='VP bbox format (default: json)',
    )
    vp_coco_od_parser.add_argument(
        '--marker-style', default='special',
        choices=['special', 'angle_bracket', 'plain'],
        help='VP marker style (default: special)',
    )
    vp_coco_od_parser.set_defaults(func=run_data_conversion)

    vp_yolo_count_parser = convert_subparsers.add_parser(
        'vp-yolo-count', help='YOLO to VP counting format (COUNT_VP task)',
    )
    vp_yolo_count_parser.add_argument('--labels-dir', required=True, help='YOLO labels directory')
    vp_yolo_count_parser.add_argument('--images-dir', required=True, help='Image directory')
    vp_yolo_count_parser.add_argument('--classes-file', required=True, help='Classes file path')
    vp_yolo_count_parser.add_argument('--output', '-o', required=True, help='Output file path')
    vp_yolo_count_parser.add_argument(
        '--task-type', default='COUNT_VP',
        help='Task type (default: COUNT_VP)',
    )
    vp_yolo_count_parser.add_argument(
        '--box-format', default='json',
        choices=['json', 'loc_tokens', 'quad'],
        help='VP bbox format (default: json)',
    )
    vp_yolo_count_parser.add_argument(
        '--marker-style', default='special',
        choices=['special', 'angle_bracket', 'plain'],
        help='VP marker style (default: special)',
    )
    vp_yolo_count_parser.add_argument('--image-ext', default='.jpg', help='Image extension')
    vp_yolo_count_parser.set_defaults(func=run_data_conversion)

    vp_jsonl_grounding_parser = convert_subparsers.add_parser(
        'vp-jsonl-grounding', help='VP OD JSONL to grounding VP JSONL',
    )
    vp_jsonl_grounding_parser.add_argument('--input', required=True, help='Input JSONL file path')
    vp_jsonl_grounding_parser.add_argument('--output', '-o', required=True, help='Output JSONL file path')
    vp_jsonl_grounding_parser.add_argument(
        '--task-type', default='PHRASE_GROUNDING_VP',
        help='Task type (default: PHRASE_GROUNDING_VP)',
    )
    vp_jsonl_grounding_parser.add_argument(
        '--box-format', default='loc_tokens',
        choices=['json', 'loc_tokens', 'quad'],
        help='VP bbox format (default: loc_tokens)',
    )
    vp_jsonl_grounding_parser.add_argument(
        '--marker-style', default='plain',
        choices=['special', 'angle_bracket', 'plain'],
        help='VP marker style (default: plain)',
    )
    vp_jsonl_grounding_parser.set_defaults(func=run_data_conversion)

    # ── VP (Visual Primitive) 数据转换 ──────────────────────────────
    # 设置 VP/TVP 子命令的默认处理函数
    for _parser in (
        vp_coco_parser,
        vp_yolo_parser,
        tvp_count_parser,
        tvp_maze_parser,
        tvp_path_parser,
        tvp_spatial_parser,
        generate_tvp_maze_parser,
        generate_tvp_path_parser,
        generate_tvp_spatial_parser,
        generate_tvp_all_parser,
    ):
        _parser.set_defaults(func=run_data_conversion)

    # 推理服务器命令
    serve_parser = subparsers.add_parser('serve', help='启动模型推理服务')
    serve_parser.add_argument(
        '--model', '-m',
        required=True,
        help='训练好的模型文件路径'
    )
    serve_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='服务器监听地址 (默认: 127.0.0.1，仅本机访问；'
             '如需对外暴露请显式指定 0.0.0.0 并自行配置鉴权与网络边界)'
    )
    serve_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='服务器监听端口 (默认: 8000)'
    )
    serve_parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='推理设备 (默认: auto)'
    )
    serve_parser.add_argument(
        '--backend',
        choices=['native', 'vllm'],
        default='native',
        help='推理后端 (默认: native)'
    )
    serve_parser.add_argument(
        '--model-revision',
        help='HuggingFace 模型/处理器 revision（建议生产环境使用具体 commit hash）'
    )
    serve_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='批处理大小 (默认: 1)'
    )
    serve_parser.add_argument(
        '--use-amp',
        action='store_true',
        help='使用自动混合精度加速推理'
    )
    
    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='运行模型评估')
    eval_parser.add_argument(
        '--model', '-m',
        required=True,
        help='训练好的模型文件路径'
    )
    eval_parser.add_argument(
        '--data', '-d',
        required=True,
        help='评估数据文件路径'
    )
    eval_parser.add_argument(
        '--output', '-o',
        help='评估结果输出路径 (JSON格式)'
    )
    eval_parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='评估设备 (默认: auto)'
    )
    eval_parser.add_argument(
        '--benchmark',
        choices=['default', 'tvp'],
        default='default',
        help='评估模式: default=标准多任务评估, tvp=TVP 思维链 benchmark',
    )
    eval_parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='TVP benchmark 最大样本数 (仅 --benchmark tvp 时生效)',
    )

    # ── Agentic 多步推理命令 ──────────────────────────────────────────
    agentic_parser = subparsers.add_parser(
        'agentic',
        help='运行 Agentic 多步视觉推理（目标分解 → 工具调用 → 验证 → 汇总）',
    )
    agentic_parser.add_argument(
        '--model', '-m',
        required=True,
        help='模型路径或 Hugging Face Hub ID',
    )
    agentic_parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入图像文件或目录路径',
    )
    agentic_parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出结果目录',
    )
    agentic_parser.add_argument(
        '--goal',
        required=True,
        help='自然语言目标（如 "detect and count all objects"）',
    )
    agentic_parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps'],
        default='auto',
        help='推理设备 (默认: auto)',
    )
    agentic_parser.add_argument(
        '--use-amp',
        action='store_true',
        help='使用自动混合精度加速推理',
    )
    agentic_parser.add_argument(
        '--max-steps',
        type=int,
        default=12,
        help='最大编排步数 (默认: 12)',
    )
    agentic_parser.add_argument(
        '--max-retries',
        type=int,
        default=1,
        help='每个子任务验证失败时的额外重试次数 (默认: 1)',
    )
    agentic_parser.add_argument(
        '--summarize-every',
        type=int,
        default=3,
        help='每 N 步输出一次状态摘要 (默认: 3)',
    )
    agentic_parser.add_argument(
        '--save-transcript',
        action='store_true',
        help='保存 agentic 元认知 transcript 文件',
    )

    return parser

def main() -> None:
    """主函数"""
    parser = create_parser()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_cli_logging(args.verbose)
    
    # 执行命令
    if args.command == 'doctor':
        success = run_doctor_task(args)
        sys.exit(0 if success else 1)

    elif args.command == 'train':
        # 构建参数覆盖字典
        overrides = {}
        if args.epochs:
            overrides['epochs'] = args.epochs
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.lr:
            overrides['lr'] = args.lr
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        if args.model:
            overrides['model'] = args.model
        if hasattr(args, 'train_data') and args.train_data:
            overrides['train_data'] = args.train_data
        if hasattr(args, 'val_data') and args.val_data:
            overrides['val_data'] = args.val_data
        if args.device:
            overrides['device'] = args.device
        if hasattr(args, 'resume') and args.resume:
            overrides['resume'] = args.resume

        if getattr(args, 'tvp_pipeline', None):
            success = run_tvp_training_task(
                tvp_pipeline=args.tvp_pipeline,
                override=args.override,
                **overrides,
            )
        elif getattr(args, 'tvp_config', None):
            success = run_tvp_training_task(
                tvp_config=args.tvp_config,
                tvp_stage=getattr(args, 'tvp_stage', None),
                override=args.override,
                **overrides,
            )
        else:
            success = run_training_task(
                task=args.task,
                config=args.config,
                override=args.override,
                **overrides
            )
        sys.exit(0 if success else 1)
        
    elif args.command == 'list-tasks':
        list_available_tasks()
        
    elif args.command == 'validate':
        success = validate_config(args.config)
        sys.exit(0 if success else 1)
        
    elif args.command == 'generate-config':
        success = generate_config_template(args.task, args.output)
        sys.exit(0 if success else 1)
        
    elif args.command == 'infer':
        success = run_inference_task(args)
        sys.exit(0 if success else 1)
        
    elif args.command == 'convert':
        success = run_data_conversion(args)
        sys.exit(0 if success else 1)
        
    elif args.command == 'serve':
        success = run_serve_task(args)
        sys.exit(0 if success else 1)
        
    elif args.command == 'eval':
        success = run_eval_task(args)
        sys.exit(0 if success else 1)

    elif args.command == 'agentic':
        success = run_agentic_task(args)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)

# 兼容性函数
def train_command():
    """训练命令入口点"""
    sys.argv = ['florence_forge_cli', 'train'] + sys.argv[1:]
    main()

def eval_command():
    """评估命令入口点"""
    sys.argv = ['florence_forge_cli', 'eval'] + sys.argv[1:]
    main()

def info_command():
    """信息命令入口点"""
    print("\n=== Florence Forge 信息 ===")
    print("版本: 1.0.0")
    print("描述: Florence-2多任务微调库")
    print("GitHub: https://github.com/florenceforge/florence-forge")
    print("文档: https://florenceforge.readthedocs.io")
    print("\n使用 'florence_forge_cli --help' 查看完整帮助")

if __name__ == '__main__':
    main()
