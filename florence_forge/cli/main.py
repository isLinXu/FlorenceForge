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
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# 在导入torch之前设置MPS设备配置
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# 移除强制CPU设置，允许使用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import yaml
from datetime import datetime

# 添加transformers导入
try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

# 简化导入，避免循环依赖
try:
    from florence_forge.core.config import TrainingConfig
    from florence_forge.core.tasks import FLORENCE2_TASKS, TaskCategory, list_all_tasks
    from florence_forge.core.model import Florence2MultiTaskModel
    from florence_forge.training.trainer import MultiTaskTrainer
    from florence_forge.training.config import load_config_from_file
    from florence_forge.data.dataset import MultiTaskDataset
    from florence_forge.data.loader import TaskDataLoader
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入核心配置模块: {e}")
    TrainingConfig = None
    FLORENCE2_TASKS = {}
    TaskCategory = None
    list_all_tasks = None
    Florence2MultiTaskModel = None
    MultiTaskTrainer = None
    load_config_from_file = None
    MultiTaskDataset = None
    TaskDataLoader = None
    CORE_AVAILABLE = False

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 预定义的任务配置映射
TASK_CONFIG_MAPPING = {
    'caption': 'configs/examples/caption_training.yaml',
    'detailed_caption': 'configs/examples/detailed_caption_training.yaml',
    'more_detailed_caption': 'configs/examples/more_detailed_caption_training.yaml',
    'detection': 'configs/examples/object_detection_training.yaml',
    'od': 'configs/examples/object_detection_training.yaml',
    'open_vocabulary_detection': 'configs/examples/open_vocabulary_detection_training.yaml',
    'phrase_grounding': 'configs/examples/phrase_grounding_training.yaml',
    'dense_region_caption': 'configs/examples/dense_region_caption_training.yaml',
    'region_proposal': 'configs/examples/region_proposal_training.yaml',
    'region_to_category': 'configs/examples/region_to_category_training.yaml',
    'region_to_description': 'configs/examples/region_to_description_training.yaml',
    'ocr': 'configs/examples/ocr_training.yaml',
    'ocr_with_region': 'configs/examples/ocr_with_region_training.yaml',
    'segmentation': 'configs/examples/segmentation_training.yaml',
    'seg': 'configs/examples/segmentation_training.yaml',
    'region_to_segmentation': 'configs/examples/region_to_segmentation_training.yaml',
    'referring_expression_segmentation': 'configs/examples/referring_expression_segmentation_training.yaml',
    'multitask': 'configs/examples/multitask_training.yaml',
    'multi': 'configs/examples/multitask_training.yaml'
}

# 任务描述
TASK_DESCRIPTIONS = {
    'caption': '基础图像描述生成任务 (CAPTION)',
    'detailed_caption': '详细图像描述生成任务 (DETAILED_CAPTION)',
    'more_detailed_caption': '更详细图像描述生成任务 (MORE_DETAILED_CAPTION)',
    'detection': '标准目标检测任务 (OD)',
    'open_vocabulary_detection': '开放词汇目标检测任务 (OPEN_VOCABULARY_DETECTION)',
    'phrase_grounding': '短语定位任务 (CAPTION_TO_PHRASE_GROUNDING)',
    'dense_region_caption': '密集区域描述任务 (DENSE_REGION_CAPTION)',
    'region_proposal': '区域提议任务 (REGION_PROPOSAL)',
    'region_to_category': '区域到类别分类任务 (REGION_TO_CATEGORY)',
    'region_to_description': '区域到描述生成任务 (REGION_TO_DESCRIPTION)',
    'ocr': 'OCR文字识别任务 (OCR)',
    'ocr_with_region': '带区域的OCR任务 (OCR_WITH_REGION)',
    'segmentation': '标准图像分割任务',
    'region_to_segmentation': '区域到分割任务 (REGION_TO_SEGMENTATION)',
    'referring_expression_segmentation': '参考表达式分割任务 (REFERRING_EXPRESSION_SEGMENTATION)',
    'multitask': '多任务混合训练 (CAPTION + OD + OCR + SEGMENTATION)'
}

def setup_cli_logging(verbose: bool = False) -> None:
    """设置CLI日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

def list_available_tasks() -> None:
    """列出所有可用的任务和配置"""
    print("\n=== Florence Forge 可用任务 ===")
    print()
    
    if not CORE_AVAILABLE or not FLORENCE2_TASKS:
        print("\n⚠️  核心模块不可用，显示预定义任务列表:")
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
                    if config['category'] == category]
            if tasks:
                print(f"  {category.value}:")
                for task in tasks:
                    desc = FLORENCE2_TASKS[task]['description']
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
        
        # 如果核心模块可用，进行更详细的验证
        if CORE_AVAILABLE and TrainingConfig:
            try:
                training_config = TrainingConfig.from_dict(config_data)
                logger.info(f"✅ 配置文件验证通过: {config_path}")
                logger.info(f"   实验名称: {training_config.experiment_name}")
                logger.info(f"   模型: {training_config.model_config.model_name}")
                logger.info(f"   训练轮数: {training_config.num_epochs}")
                logger.info(f"   批次大小: {training_config.data_config.batch_size}")
                logger.info(f"   学习率: {training_config.optimization_config.learning_rate}")
                return True
            except Exception as e:
                logger.error(f"❌ 配置验证失败: {e}")
                return False
        else:
            logger.info(f"✅ 配置文件基础验证通过: {config_path}")
            logger.info(f"   实验名称: {config_data.get('experiment_name', 'N/A')}")
            logger.info(f"   模型: {config_data.get('model_config', {}).get('model_name', 'N/A')}")
            logger.info(f"   训练轮数: {config_data.get('num_epochs', 'N/A')}")
            logger.info(f"   批次大小: {config_data.get('data_config', {}).get('batch_size', 'N/A')}")
            logger.info(f"   学习率: {config_data.get('optimization_config', {}).get('learning_rate', 'N/A')}")
            return True
            
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

def run_inference_task(args) -> bool:
    """运行推理任务"""
    try:
        from pathlib import Path
        import json
        import glob
        from PIL import Image
        
        # 导入推理引擎
        try:
            from florence_forge.deployment.inference import InferenceEngine
        except ImportError:
            logger.error("❌ 无法导入推理引擎，请检查安装")
            return False
        
        # 验证模型路径
        model_path_str = args.model
        is_hf_hub_id = '/' in model_path_str and not os.path.exists(model_path_str)

        if not is_hf_hub_id:
            model_path = Path(model_path_str)
            if not model_path.exists():
                logger.error(f"❌ 模型文件或目录不存在: {model_path}")
                return False
        else:
            logger.info(f"ℹ️  将从Hugging Face Hub加载模型: {model_path_str}")
            model_path = model_path_str
        
        logger.info(f"🚀 开始推理任务")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   输入路径: {args.input}")
        logger.info(f"   输出目录: {args.output}")
        logger.info(f"   设备: {args.device}")
        
        # 创建推理引擎
        logger.info("🤖 初始化推理引擎...")
        inference_engine = InferenceEngine(
            model=str(model_path),
            device=args.device,
            batch_size=args.batch_size,
            use_amp=args.use_amp
        )
        
        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理输入
        input_path = Path(args.input)
        results = []
        
        if input_path.is_file():
            # 单个文件推理
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                logger.info("📸 处理单张图像...")
                image = Image.open(input_path).convert('RGB')
                # 为Florence2模型添加默认任务提示
                task_prompt = getattr(args, 'task_prompt', '<OD>')  # 默认为目标检测
                
                # 设置可视化参数
                visualize = getattr(args, 'visualize', False)
                save_path = None
                if visualize and getattr(args, 'save_visualizations', False):
                    save_path = output_dir / f"{input_path.stem}_visualization.png"
                
                # 检查是否需要文本输入
                if task_prompt == '<OPEN_VOCABULARY_DETECTION>' and not args.text_input:
                    logger.error(f"❌ 任务 '{task_prompt}' 需要 --text-input 参数.")
                    return False
                text_input = args.text_input

                result = inference_engine.predict(
                    image, 
                    task_prompt=task_prompt,
                    text_input=text_input,
                    visualize=visualize,
                    save_path=str(save_path) if save_path else None
                )
                
                # 保存结果
                result_file = output_dir / f"{input_path.stem}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "image_path": str(input_path),
                        "result": str(result) if not isinstance(result, (dict, list)) else result
                    }, f, indent=2, ensure_ascii=False)
                
                results.append({
                    "image_path": str(input_path),
                    "result_file": str(result_file),
                    "result": result
                })
                
                logger.info(f"✅ 推理完成: {result_file}")
            else:
                logger.error(f"❌ 不支持的文件格式: {input_path.suffix}")
                return False
                
        elif input_path.is_dir():
            # 批量推理
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(str(input_path / ext)))
                image_files.extend(glob.glob(str(input_path / ext.upper())))
            
            if not image_files:
                logger.error(f"❌ 在目录中未找到图像文件: {input_path}")
                return False
            
            # 为Florence2模型添加默认任务提示
            task_prompt = getattr(args, 'task_prompt', '<OD>')  # 默认为目标检测
            
            # 预先检查是否需要文本输入
            if task_prompt == '<OPEN_VOCABULARY_DETECTION>' and not args.text_input:
                logger.error(f"❌ 任务 '{task_prompt}' 需要 --text-input 参数.")
                return False
            text_input = args.text_input
            
            logger.info(f"📸 处理 {len(image_files)} 张图像...")
            
            # 批量处理
            for i, image_file in enumerate(image_files, 1):
                try:
                    image_path = Path(image_file)
                    logger.info(f"处理 {i}/{len(image_files)}: {image_path.name}")
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # 设置可视化参数
                    visualize = getattr(args, 'visualize', False)
                    save_path = None
                    if visualize and getattr(args, 'save_visualizations', False):
                        save_path = output_dir / f"{image_path.stem}_visualization.png"

                    result = inference_engine.predict(
                        image, 
                        task_prompt=task_prompt,
                        text_input=text_input,
                        visualize=visualize,
                        save_path=str(save_path) if save_path else None
                    )
                    
                    # 保存结果
                    result_file = output_dir / f"{image_path.stem}_result.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "image_path": str(image_path),
                            "result": str(result) if not isinstance(result, (dict, list)) else result
                        }, f, indent=2, ensure_ascii=False)
                    
                    results.append({
                        "image_path": str(image_path),
                        "result_file": str(result_file),
                        "result": result
                    })
                    
                except Exception as e:
                    logger.error(f"❌ 处理图像失败 {image_path.name}: {e}")
                    continue
            
            logger.info(f"✅ 批量推理完成，处理了 {len(results)} 张图像")
        else:
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return False
        
        # 保存汇总结果
        summary_file = output_dir / "inference_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": str(model_path),
                "input_path": str(input_path),
                "total_images": len(results),
                "results": results,
                "stats": inference_engine.get_stats()
            }, f, indent=2, ensure_ascii=False)
        
        # 输出统计信息
        stats = inference_engine.get_stats()
        logger.info("📊 推理统计:")
        logger.info(f"   总推理次数: {stats['total_inferences']}")
        logger.info(f"   总耗时: {stats['total_time']:.2f}s")
        logger.info(f"   平均推理时间: {stats['avg_inference_time']:.3f}s")
        logger.info(f"   吞吐量: {stats['throughput']:.2f} images/s")
        logger.info(f"   汇总文件: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 推理任务失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

def run_data_conversion(args) -> bool:
    """运行数据转换任务"""
    try:
        # 导入数据转换器
        from florence_forge.data.converter import DataFormatConverter
        
        logger.info(f"开始数据转换: {args.convert_type}")
        
        if args.convert_type == 'yolo':
            DataFormatConverter.yolo_to_florence2_od(
                yolo_labels_dir=args.labels_dir,
                output_path=args.output,
                image_dir=args.images_dir,
                classes_file=args.classes_file,
                image_ext=args.image_ext,
                task_type=args.task_type
            )
            
        elif args.convert_type == 'coco':
            DataFormatConverter.coco_to_florence2_od(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir
            )
            
        elif args.convert_type == 'coco-caption':
            DataFormatConverter.coco_caption_to_florence2(
                coco_json_path=args.json_file,
                output_path=args.output,
                image_dir=args.images_dir
            )
            
        elif args.convert_type == 'csv':
            DataFormatConverter.csv_caption_to_florence2(
                csv_path=args.csv_file,
                output_path=args.output,
                image_column=args.image_column,
                caption_column=args.caption_column,
                task_type=args.task_type
            )
            
        elif args.convert_type == 'xml':
            DataFormatConverter.xml_to_florence2_od(
                xml_dir=args.xml_dir,
                output_path=args.output,
                image_dir=args.images_dir
            )
            
        elif args.convert_type == 'ocr':
            DataFormatConverter.txt_ocr_to_florence2(
                image_dir=args.images_dir,
                txt_dir=args.texts_dir,
                output_path=args.output,
                task_type=args.task_type
            )
            
        else:
            logger.error(f"❌ 不支持的转换类型: {args.convert_type}")
            return False
        
        logger.info(f"✅ 数据转换完成: {args.output}")
        return True
        
    except ImportError as e:
        logger.error(f"❌ 导入数据转换器失败: {e}")
        logger.error("请确保已正确安装florence_forge或数据转换器模块")
        return False
        
    except Exception as e:
        logger.error(f"❌ 数据转换失败: {e}")
        return False

def run_training_task(
    task: Optional[str] = None,
    config: Optional[str] = None,
    override: Optional[list] = None,
    **overrides
) -> bool:
    # 处理 --override 参数
    if override:
        for key, value in override:
            # 尝试将值转换为适当的类型
            try:
                # 尝试转换为数字
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 如果转换失败，保持原始字符串
                pass
            overrides[key] = value
    """运行训练任务"""
    try:
        # 检查核心模块是否可用
        if not CORE_AVAILABLE:
            logger.error("❌ 核心训练模块不可用，请检查安装")
            return False
        
        # 确定配置文件路径
        if config:
            config_path = Path(config)
        elif task:
            if task not in TASK_CONFIG_MAPPING:
                logger.error(f"❌ 未知任务类型: {task}")
                logger.info(f"可用任务: {', '.join(TASK_CONFIG_MAPPING.keys())}")
                return False
            config_path = Path(TASK_CONFIG_MAPPING[task])
        else:
            logger.error("❌ 必须指定任务类型或配置文件")
            return False
        
        # 查找配置文件
        possible_paths = [
            config_path,
            Path.cwd() / config_path,
            Path(__file__).parent.parent / config_path
        ]
        
        actual_config_path = None
        for path in possible_paths:
            if path.exists():
                actual_config_path = path
                break
        
        if not actual_config_path:
            logger.error(f"❌ 找不到配置文件: {config_path}")
            return False
        
        # 将任务类型映射为正确的任务名称
        task_type_mapping = {
            'od': 'OD',
            'detection': 'OD', 
            'caption': 'CAPTION',
            'detailed_caption': 'DETAILED_CAPTION',
            'more_detailed_caption': 'MORE_DETAILED_CAPTION',
            'open_vocabulary_detection': 'OPEN_VOCABULARY_DETECTION',
            'phrase_grounding': 'CAPTION_TO_PHRASE_GROUNDING',
            'dense_region_caption': 'DENSE_REGION_CAPTION',
            'region_proposal': 'REGION_PROPOSAL',
            'region_to_category': 'REGION_TO_CATEGORY',
            'region_to_description': 'REGION_TO_DESCRIPTION',
            'ocr': 'OCR',
            'ocr_with_region': 'OCR_WITH_REGION',
            'segmentation': 'REFERRING_EXPRESSION_SEGMENTATION',
            'seg': 'REFERRING_EXPRESSION_SEGMENTATION',
            'region_to_segmentation': 'REGION_TO_SEGMENTATION',
            'referring_expression_segmentation': 'REFERRING_EXPRESSION_SEGMENTATION'
        }
        
        # 添加任务类型到覆盖参数中
        if task and task in task_type_mapping:
            overrides['task_type'] = task_type_mapping[task]
        
        logger.info(f"🚀 开始训练任务")
        logger.info(f"   任务类型: {task or 'custom'}")
        logger.info(f"   配置文件: {actual_config_path}")
        
        if overrides:
            logger.info(f"   参数覆盖: {overrides}")
        
        # 加载训练配置
        logger.info("📋 加载训练配置...")
        training_config = load_config_from_file(str(actual_config_path))
        
        # 应用命令行参数覆盖
        if overrides:
            _apply_config_overrides(training_config, overrides)
        
        # 验证配置
        logger.info("✅ 验证训练配置...")
        if not validate_config(str(actual_config_path)):
            return False
        
        # 初始化模型
        logger.info("🤖 初始化模型...")
        model = Florence2MultiTaskModel(training_config.model_config)
        
        # 准备数据集
        logger.info("📊 准备训练数据...")
        train_dataset, val_dataset = _prepare_datasets(training_config)
        
        # 创建训练器
        logger.info("🏋️ 创建训练器...")
        trainer = MultiTaskTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=training_config
        )
        
        # 开始训练
        logger.info("🚀 开始训练...")
        training_summary = trainer.train()
        
        # 输出训练结果
        logger.info("✅ 训练完成!")
        logger.info(f"   最终损失: {training_summary.get('final_loss', 'N/A')}")
        logger.info(f"   最佳指标: {training_summary.get('best_metric', 'N/A')}")
        logger.info(f"   训练轮数: {training_summary.get('epochs_completed', 'N/A')}")
        logger.info(f"   输出目录: {training_config.output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 运行训练任务时出错: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

def _set_nested_attr(obj, attr_str, value):
    """递归设置对象的嵌套属性"""
    attrs = attr_str.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def _apply_config_overrides(config: 'TrainingConfig', overrides: Dict[str, Any]) -> None:
    """应用命令行参数覆盖到配置"""
    try:
        if 'epochs' in overrides and overrides['epochs'] is not None:
            config.num_epochs = overrides['epochs']
            logger.info(f"覆盖训练轮数: {config.num_epochs}")
        
        if 'batch_size' in overrides and overrides['batch_size'] is not None:
            config.data_config.batch_size = overrides['batch_size']
            logger.info(f"覆盖批次大小: {config.data_config.batch_size}")
        
        if 'lr' in overrides and overrides['lr'] is not None:
            config.optimization_config.learning_rate = overrides['lr']
            logger.info(f"覆盖学习率: {config.optimization_config.learning_rate}")
        
        if 'output_dir' in overrides and overrides['output_dir'] is not None:
            config.output_dir = overrides['output_dir']
            logger.info(f"覆盖输出目录: {config.output_dir}")
        
        if 'model' in overrides and overrides['model'] is not None:
            config.model_config.model_name = overrides['model']
            logger.info(f"覆盖模型名称: {config.model_config.model_name}")
        
        if 'train_data_path' in overrides and overrides['train_data_path'] is not None:
            config.train_data_path = overrides['train_data_path']
            logger.info(f"覆盖训练数据路径: {config.train_data_path}")
        
        if 'val_data_path' in overrides and overrides['val_data_path'] is not None:
            config.val_data_path = overrides['val_data_path']
            logger.info(f"覆盖验证数据路径: {config.val_data_path}")
        
        # 添加任务类型覆盖
        # 处理所有其他以.分隔的覆盖
        for key, value in overrides.items():
            if '.' in key and value is not None:
                try:
                    _set_nested_attr(config, key, value)
                    logger.info(f"覆盖配置: {key} = {value}")
                except AttributeError:
                    logger.warning(f"无法设置配置属性: {key}")
            
    except Exception as e:
        logger.warning(f"应用配置覆盖时出错: {e}")

def _prepare_datasets(config: 'TrainingConfig') -> Tuple['MultiTaskDataset', Optional['MultiTaskDataset']]:
    """准备训练和验证数据集"""
    try:
        # 构建数据配置列表
        data_configs = []
        for task in config.tasks:
            # 确保任务类型格式正确
            task_type = task.upper() if task else "CAPTION"
            if config.train_data_path:
                data_configs.append({
                    "task_type": task_type,
                    "data_path": config.train_data_path,
                    "weight": config.task_weights.get(task, 1.0)
                })
        
        if not data_configs:
            # 如果没有配置数据路径，使用默认的示例数据
            data_configs = [{
                "task_type": "CAPTION",
                "data_path": "./data/sample_data.jsonl",
                "weight": 1.0
            }]
            logger.warning("未配置训练数据路径，使用默认示例数据")
        
        # 创建processor
        processor = None
        if AutoProcessor is not None:
            try:
                processor = AutoProcessor.from_pretrained(
                    config.model_config.model_name,
                    trust_remote_code=config.model_config.trust_remote_code
                )
            except Exception as e:
                logger.error(f"处理器加载失败: {e}")
                processor = None
        else:
            logger.warning("AutoProcessor不可用，跳过处理器加载")
        
        # 创建训练数据集
        train_dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path="./data/images",
            config=config.data_config,
            processor=processor
        )
        
        # 创建验证数据集（如果配置了验证数据）
        val_dataset = None
        if config.val_data_path:
            val_data_configs = []
            for task in config.tasks:
                # 确保任务类型格式正确
                task_type = task.upper() if task else "CAPTION"
                val_data_configs.append({
                    "task_type": task_type,
                    "data_path": config.val_data_path,
                    "weight": config.task_weights.get(task, 1.0)
                })
            
            val_dataset = MultiTaskDataset(
                data_configs=val_data_configs,
                image_base_path="./data/images",
                config=config.data_config,
                processor=processor
            )
        
        logger.info(f"训练数据集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"验证数据集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"准备数据集时出错: {e}")
        raise

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
  florence_forge_cli list-tasks
  florence_forge_cli validate --config configs/examples/caption_training.yaml
  florence_forge_cli generate-config --task ocr --output my_config.yaml
  
  # 数据转换
  florence_forge_cli convert yolo --labels-dir ./labels --images-dir ./images --classes-file ./classes.txt --output ./data.jsonl
  florence_forge_cli convert coco --json-file ./annotations.json --images-dir ./images --output ./data.jsonl
  florence_forge_cli convert csv --csv-file ./captions.csv --output ./data.jsonl
  florence_forge_cli convert xml --xml-dir ./annotations --images-dir ./images --output ./data.jsonl
  florence_forge_cli convert ocr --images-dir ./images --texts-dir ./texts --output ./data.jsonl

更多信息请访问: https://github.com/florenceforge/florenceforge
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
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
    if args.command == 'train':
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
            overrides['train_data_path'] = args.train_data
        if hasattr(args, 'val_data') and args.val_data:
            overrides['val_data_path'] = args.val_data
        if args.device:
            overrides['device'] = args.device
        
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
    print("评估功能正在开发中...")
    sys.exit(1)

def info_command():
    """信息命令入口点"""
    print("\n=== Florence Forge 信息 ===")
    print("版本: 1.0.0")
    print("描述: Florence-2多任务微调库")
    print("GitHub: https://github.com/florenceforge/florenceforge")
    print("文档: https://florenceforge.readthedocs.io")
    print("\n使用 'florence_forge_cli --help' 查看完整帮助")

if __name__ == '__main__':
    main()