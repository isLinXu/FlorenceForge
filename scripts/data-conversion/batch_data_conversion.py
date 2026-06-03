#!/usr/bin/env python3
"""
Florence Forge 批量数据转换工作流程脚本

基于配置文件自动执行多种格式的数据转换，
支持数据验证、清理、合并和分割等功能。
"""

import sys
import os
import json
import yaml
import logging
import subprocess
import argparse
from datetime import datetime
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 设置日志
def setup_logging(config: Dict[str, Any]):
    """设置日志配置"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    log_file = log_config.get('file', './logs/data_conversion.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    
    if log_config.get('console', True):
        logger.addHandler(console_handler)
    
    return logger

class DataConversionWorkflow:
    """数据转换工作流程管理器"""
    
    def __init__(self, config_path: str):
        """初始化工作流程"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logging(self.config)
        self.conversion_stats = {
            'total_files': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'errors': []
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"错误：无法加载配置文件 {self.config_path}: {e}")
            sys.exit(1)
    
    def _run_conversion_command(self, cmd: List[str], description: str) -> bool:
        """运行数据转换命令"""
        self.logger.info(f"执行: {description}")
        self.logger.debug(f"命令: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(PROJECT_ROOT)
                if not env.get("PYTHONPATH")
                else str(PROJECT_ROOT) + os.pathsep + env["PYTHONPATH"]
            )
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=3600,  # 1小时超时
                env=env,
            )
            
            self.logger.info(f"✅ {description} 成功完成")
            if result.stdout:
                self.logger.debug(f"输出: {result.stdout}")
            
            self.conversion_stats['successful_conversions'] += 1
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"{description} 失败: {e.stderr}"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['failed_conversions'] += 1
            self.conversion_stats['errors'].append(error_msg)
            
            # 根据配置决定是否继续
            if not self.config.get('error_handling', {}).get('continue_on_error', True):
                raise
            return False
            
        except subprocess.TimeoutExpired:
            error_msg = f"{description} 超时"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['failed_conversions'] += 1
            self.conversion_stats['errors'].append(error_msg)
            return False
            
        except FileNotFoundError:
            error_msg = "找不到florence_forge_cli命令"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['errors'].append(error_msg)
            return False
    
    def _create_output_directories(self):
        """创建输出目录"""
        directories = set()
        
        # 收集所有输出目录
        for source_config in self.config.get('input_sources', {}).values():
            if isinstance(source_config, dict) and 'output_file' in source_config:
                directories.add(Path(source_config['output_file']).parent)
        
        output_config = self.config.get('output', {})
        if 'merged_file' in output_config:
            directories.add(Path(output_config['merged_file']).parent)
        
        split_config = output_config.get('split', {})
        if split_config.get('enabled'):
            directories.add(Path(split_config.get('train_file', '')).parent)
            directories.add(Path(split_config.get('val_file', '')).parent)
        
        # 创建目录
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"创建输出目录: {directory}")
    
    def convert_yolo_data(self, config: Dict[str, Any]) -> bool:
        """转换YOLO格式数据"""
        cmd = [
            sys.executable, '-m', 'florence_forge.cli.main', 'convert', 'yolo',
            '--labels-dir', config['labels_dir'],
            '--images-dir', config['images_dir'],
            '--classes-file', config['classes_file'],
            '--output', config['output_file'],
            '--image-ext', config.get('image_ext', '.jpg'),
            '--task-type', config.get('task_type', 'OD')
        ]
        
        return self._run_conversion_command(cmd, "YOLO格式转换")
    
    def convert_coco_data(self, config: Dict[str, Any]) -> bool:
        """转换COCO格式数据"""
        cmd = [
            sys.executable, '-m', 'florence_forge.cli.main', 'convert', 'coco',
            '--json-file', config['json_file'],
            '--images-dir', config['images_dir'],
            '--output', config['output_file']
        ]
        
        return self._run_conversion_command(cmd, "COCO格式转换")
    
    def convert_csv_data(self, config: Dict[str, Any]) -> bool:
        """转换CSV格式数据"""
        cmd = [
            sys.executable, '-m', 'florence_forge.cli.main', 'convert', 'csv',
            '--csv-file', config['csv_file'],
            '--output', config['output_file'],
            '--image-column', config.get('image_column', 'image_path'),
            '--caption-column', config.get('caption_column', 'caption'),
            '--task-type', config.get('task_type', 'CAPTION')
        ]
        
        return self._run_conversion_command(cmd, "CSV格式转换")
    
    def convert_xml_data(self, config: Dict[str, Any]) -> bool:
        """转换VOC XML格式数据"""
        cmd = [
            sys.executable, '-m', 'florence_forge.cli.main', 'convert', 'xml',
            '--xml-dir', config['xml_dir'],
            '--images-dir', config['images_dir'],
            '--output', config['output_file']
        ]
        
        return self._run_conversion_command(cmd, "VOC XML格式转换")
    
    def convert_ocr_data(self, config: Dict[str, Any]) -> bool:
        """转换OCR数据"""
        cmd = [
            sys.executable, '-m', 'florence_forge.cli.main', 'convert', 'ocr',
            '--images-dir', config['images_dir'],
            '--texts-dir', config['texts_dir'],
            '--output', config['output_file'],
            '--task-type', config.get('task_type', 'OCR')
        ]
        
        return self._run_conversion_command(cmd, "OCR数据转换")
    
    def merge_converted_files(self) -> Optional[str]:
        """合并转换后的文件"""
        output_config = self.config.get('output', {})
        merged_file = output_config.get('merged_file')
        
        if not merged_file:
            self.logger.warning("未配置合并文件路径，跳过合并步骤")
            return None
        
        # 收集所有成功转换的文件
        converted_files = []
        for source_name, source_config in self.config.get('input_sources', {}).items():
            if (isinstance(source_config, dict) and 
                source_config.get('enabled', False) and 
                'output_file' in source_config):
                
                output_file = source_config['output_file']
                if Path(output_file).exists():
                    converted_files.append(output_file)
                    self.logger.info(f"找到转换文件: {output_file}")
        
        if not converted_files:
            self.logger.warning("没有找到可合并的转换文件")
            return None
        
        # 合并文件
        self.logger.info(f"合并 {len(converted_files)} 个文件到 {merged_file}")
        
        try:
            Path(merged_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(merged_file, 'w', encoding='utf-8') as outfile:
                total_samples = 0
                for file_path in converted_files:
                    self.logger.info(f"合并文件: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            line = line.strip()
                            if line:
                                outfile.write(line + '\n')
                                total_samples += 1
            
            self.logger.info(f"✅ 成功合并 {total_samples} 个样本到 {merged_file}")
            return merged_file
            
        except Exception as e:
            error_msg = f"合并文件失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['errors'].append(error_msg)
            return None
    
    def split_data(self, merged_file: str) -> bool:
        """分割数据为训练集和验证集"""
        split_config = self.config.get('output', {}).get('split', {})
        
        if not split_config.get('enabled', False):
            self.logger.info("数据分割未启用，跳过分割步骤")
            return True
        
        train_ratio = split_config.get('train_ratio', 0.8)
        val_ratio = split_config.get('val_ratio', 0.2)
        train_file = split_config.get('train_file', './data/final/train.jsonl')
        val_file = split_config.get('val_file', './data/final/val.jsonl')
        
        self.logger.info(f"分割数据: 训练集 {train_ratio:.1%}, 验证集 {val_ratio:.1%}")
        
        try:
            # 读取所有样本
            samples = []
            with open(merged_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(line)
            
            # 随机打乱
            random.shuffle(samples)
            
            # 计算分割点
            total_samples = len(samples)
            train_count = int(total_samples * train_ratio)
            
            # 分割数据
            train_samples = samples[:train_count]
            val_samples = samples[train_count:]
            
            # 创建输出目录
            Path(train_file).parent.mkdir(parents=True, exist_ok=True)
            Path(val_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 写入训练集
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in train_samples:
                    f.write(sample + '\n')
            
            # 写入验证集
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    f.write(sample + '\n')
            
            self.logger.info(f"✅ 数据分割完成:")
            self.logger.info(f"  训练集: {len(train_samples)} 样本 -> {train_file}")
            self.logger.info(f"  验证集: {len(val_samples)} 样本 -> {val_file}")
            
            return True
            
        except Exception as e:
            error_msg = f"数据分割失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['errors'].append(error_msg)
            return False
    
    def generate_statistics_report(self) -> bool:
        """生成统计报告"""
        stats_config = self.config.get('output', {}).get('statistics', {})
        
        if not stats_config.get('enabled', False):
            self.logger.info("统计报告未启用，跳过报告生成")
            return True
        
        output_file = stats_config.get('output_file', './data/final/conversion_report.json')
        
        try:
            # 生成报告数据
            report = {
                'workflow': self.config.get('workflow', {}),
                'timestamp': datetime.now().isoformat(),
                'conversion_statistics': self.conversion_stats,
                'input_sources': {},
                'output_files': []
            }
            
            # 统计输入源信息
            for source_name, source_config in self.config.get('input_sources', {}).items():
                if isinstance(source_config, dict):
                    report['input_sources'][source_name] = {
                        'enabled': source_config.get('enabled', False),
                        'output_file': source_config.get('output_file', ''),
                        'file_exists': Path(source_config.get('output_file', '')).exists() if source_config.get('output_file') else False
                    }
            
            # 统计输出文件
            output_config = self.config.get('output', {})
            if 'merged_file' in output_config:
                merged_file = output_config['merged_file']
                if Path(merged_file).exists():
                    report['output_files'].append({
                        'type': 'merged',
                        'path': merged_file,
                        'size_bytes': Path(merged_file).stat().st_size
                    })
            
            split_config = output_config.get('split', {})
            if split_config.get('enabled'):
                for file_type, file_path in [('train', split_config.get('train_file')), ('val', split_config.get('val_file'))]:
                    if file_path and Path(file_path).exists():
                        report['output_files'].append({
                            'type': file_type,
                            'path': file_path,
                            'size_bytes': Path(file_path).stat().st_size
                        })
            
            # 写入报告
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 统计报告已生成: {output_file}")
            return True
            
        except Exception as e:
            error_msg = f"生成统计报告失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.conversion_stats['errors'].append(error_msg)
            return False
    
    def run_workflow(self) -> bool:
        """运行完整的数据转换工作流程"""
        self.logger.info("开始数据转换工作流程")
        self.logger.info(f"配置文件: {self.config_path}")
        
        workflow_info = self.config.get('workflow', {})
        self.logger.info(f"工作流程: {workflow_info.get('name', 'Unknown')}")
        self.logger.info(f"描述: {workflow_info.get('description', 'No description')}")
        
        # 创建输出目录
        self._create_output_directories()
        
        # 执行各种格式的数据转换
        conversion_methods = {
            'yolo_detection': self.convert_yolo_data,
            'coco_detection': self.convert_coco_data,
            'csv_captions': self.convert_csv_data,
            'voc_detection': self.convert_xml_data,
            'ocr_data': self.convert_ocr_data
        }
        
        input_sources = self.config.get('input_sources', {})
        enabled_sources = 0
        
        for source_name, source_config in input_sources.items():
            if isinstance(source_config, dict) and source_config.get('enabled', False):
                enabled_sources += 1
                self.conversion_stats['total_files'] += 1
                
                self.logger.info(f"\n处理数据源: {source_name}")
                
                # 查找对应的转换方法
                conversion_method = None
                for method_key, method_func in conversion_methods.items():
                    if method_key in source_name or source_name in method_key:
                        conversion_method = method_func
                        break
                
                if conversion_method:
                    conversion_method(source_config)
                else:
                    self.logger.warning(f"未找到 {source_name} 的转换方法")
        
        if enabled_sources == 0:
            self.logger.warning("没有启用的数据源")
            return False
        
        # 合并转换后的文件
        merged_file = self.merge_converted_files()
        
        # 分割数据
        if merged_file:
            self.split_data(merged_file)
        
        # 生成统计报告
        self.generate_statistics_report()
        
        # 输出最终统计
        self.logger.info("\n=== 工作流程完成 ===")
        self.logger.info(f"总文件数: {self.conversion_stats['total_files']}")
        self.logger.info(f"成功转换: {self.conversion_stats['successful_conversions']}")
        self.logger.info(f"转换失败: {self.conversion_stats['failed_conversions']}")
        
        if self.conversion_stats['errors']:
            self.logger.warning(f"错误数量: {len(self.conversion_stats['errors'])}")
            for error in self.conversion_stats['errors']:
                self.logger.warning(f"  - {error}")
        
        return self.conversion_stats['failed_conversions'] == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Florence Forge 批量数据转换工作流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python batch_data_conversion.py --config configs/data_conversion_workflow.yaml
  python batch_data_conversion.py --config my_workflow.yaml --dry-run
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='工作流程配置文件路径'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅显示将要执行的操作，不实际执行'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)
    
    try:
        # 创建工作流程实例
        workflow = DataConversionWorkflow(args.config)
        
        if args.dry_run:
            print("DRY RUN 模式 - 仅显示配置信息")
            print(f"配置文件: {args.config}")
            print(f"工作流程: {workflow.config.get('workflow', {}).get('name', 'Unknown')}")
            
            input_sources = workflow.config.get('input_sources', {})
            enabled_sources = [name for name, config in input_sources.items() 
                             if isinstance(config, dict) and config.get('enabled', False)]
            
            print(f"启用的数据源: {', '.join(enabled_sources) if enabled_sources else '无'}")
            return
        
        # 运行工作流程
        success = workflow.run_workflow()
        
        if success:
            print("\n✅ 数据转换工作流程成功完成")
            sys.exit(0)
        else:
            print("\n❌ 数据转换工作流程完成，但存在错误")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 工作流程执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
