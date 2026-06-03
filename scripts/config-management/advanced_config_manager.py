#!/usr/bin/env python3
"""
高级配置管理工具

提供配置文件的批量管理、验证、比较和优化功能

功能:
1. 批量验证配置文件
2. 配置文件比较和差异分析
3. 配置参数优化建议
4. 配置文件格式化和标准化
5. 配置模板生成和自定义
6. 硬件适配优化
7. 任务特定配置生成

使用示例:
    # 验证所有示例配置
    python scripts/advanced_config_manager.py validate-all
    
    # 比较两个配置文件
    python scripts/advanced_config_manager.py compare config1.yaml config2.yaml
    
    # 优化配置参数
    python scripts/advanced_config_manager.py optimize --config config.yaml --hardware gpu_info.json
    
    # 格式化配置文件
    python scripts/advanced_config_manager.py format --config config.yaml
    
    # 生成硬件适配配置
    python scripts/advanced_config_manager.py hardware-adapt --base-config base.yaml --gpu-memory 24
"""

import argparse
import sys
import json
import logging
import yaml
from collections import defaultdict
from datetime import datetime
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from florence_forge.core import TrainingConfig, ModelConfig, DataConfig
    from florence_forge.utils import setup_logging
except ImportError as e:
    print(f"错误: 无法导入必要模块: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class AdvancedConfigManager:
    """高级配置管理器"""
    
    def __init__(self):
        """TODO: Add documentation for __init__"""
        self.configs_dir = project_root / "configs"
        self.examples_dir = self.configs_dir / "examples"
        
        # 硬件配置建议
        self.hardware_recommendations = {
            'gpu_memory_thresholds': {
                4: {'batch_size': 1, 'lora_r': 16, 'grad_accum': 8},
                8: {'batch_size': 2, 'lora_r': 32, 'grad_accum': 4},
                12: {'batch_size': 4, 'lora_r': 32, 'grad_accum': 2},
                16: {'batch_size': 6, 'lora_r': 64, 'grad_accum': 2},
                24: {'batch_size': 8, 'lora_r': 64, 'grad_accum': 1},
                32: {'batch_size': 12, 'lora_r': 128, 'grad_accum': 1},
                48: {'batch_size': 16, 'lora_r': 128, 'grad_accum': 1}
            },
            'model_memory_requirements': {
                'microsoft/Florence-2-base': 6,  # GB
                'microsoft/Florence-2-large': 12  # GB
            }
        }
    
    def detect_hardware(self) -> Dict[str, Any]:
        """检测当前硬件配置"""
        try:
            import torch
            
            hardware_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': 0,
                'gpu_memory_gb': 0,
                'gpu_names': []
            }
            
            if torch.cuda.is_available():
                hardware_info['gpu_count'] = torch.cuda.device_count()
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_properties(i).name
                    hardware_info['gpu_memory_gb'] = max(hardware_info['gpu_memory_gb'], gpu_memory)
                    hardware_info['gpu_names'].append(gpu_name)
            
            return hardware_info
            
        except ImportError:
            logger.warning("PyTorch未安装，无法检测GPU信息")
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': False,
                'gpu_count': 0,
                'gpu_memory_gb': 0,
                'gpu_names': []
            }
    
    def find_all_configs(self, directory: Optional[Path] = None) -> List[Path]:
        """查找所有配置文件"""
        if directory is None:
            directory = self.configs_dir
        
        config_files = []
        for pattern in ['*.yaml', '*.yml']:
            config_files.extend(directory.rglob(pattern))
        
        return sorted(config_files)
    
    def validate_config_file(self, config_path: Path) -> Tuple[bool, str, Optional[TrainingConfig]]:
        """验证单个配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 尝试创建TrainingConfig对象
            config = TrainingConfig.from_dict(config_data)
            
            # 详细验证
            errors = []
            warnings = []
            
            # 基本参数检查
            if config.num_epochs <= 0:
                errors.append("num_epochs 必须大于 0")
            if config.num_epochs > 50:
                warnings.append(f"训练轮数 ({config.num_epochs}) 可能过多")
            
            if config.data_config.batch_size <= 0:
                errors.append("batch_size 必须大于 0")
            if config.data_config.batch_size > 64:
                warnings.append(f"批次大小 ({config.data_config.batch_size}) 可能过大")
            
            if config.optimization_config.learning_rate <= 0:
                errors.append("learning_rate 必须大于 0")
            
            # 路径检查
            output_dir = Path(config.output_dir)
            if not output_dir.parent.exists():
                warnings.append(f"输出目录的父目录不存在: {output_dir.parent}")
            
            # LoRA配置检查
            if config.model_config.use_lora:
                lora_config = config.model_config.lora_config
                if lora_config.r <= 0:
                    errors.append("LoRA rank (r) 必须大于 0")
                if lora_config.r > 512:
                    warnings.append(f"LoRA rank ({lora_config.r}) 可能过大")
                if lora_config.lora_alpha < lora_config.r:
                    warnings.append("lora_alpha 通常应该 >= lora_r")
            
            # 学习率合理性检查
            lr = config.optimization_config.learning_rate
            if config.model_config.use_lora:
                if lr > 1e-3:
                    warnings.append(f"LoRA微调学习率 ({lr}) 可能过高")
                elif lr < 1e-6:
                    warnings.append(f"LoRA微调学习率 ({lr}) 可能过低")
            else:
                if lr > 1e-4:
                    warnings.append(f"全参数微调学习率 ({lr}) 可能过高")
                elif lr < 1e-7:
                    warnings.append(f"全参数微调学习率 ({lr}) 可能过低")
            
            # 梯度累积检查
            if config.gradient_accumulation_steps > 16:
                warnings.append(f"梯度累积步数 ({config.gradient_accumulation_steps}) 可能过大")
            
            # 任务调度检查
            if hasattr(config, 'task_scheduling_config'):
                if config.task_scheduling_config.strategy not in ['round_robin', 'weighted', 'curriculum', 'adaptive']:
                    warnings.append(f"未知的任务调度策略: {config.task_scheduling_config.strategy}")
            
            if errors:
                return False, f"错误: {'; '.join(errors)}", None
            
            message = "验证通过"
            if warnings:
                message += f" (警告: {'; '.join(warnings)})"
            
            return True, message, config
            
        except Exception as e:
            return False, f"解析失败: {str(e)}", None
    
    def validate_all_configs(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """批量验证配置文件"""
        config_files = self.find_all_configs(directory)
        results = {
            'total': len(config_files),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': [],
            'summary': {
                'common_issues': defaultdict(int),
                'config_types': defaultdict(int)
            }
        }
        
        logger.info(f"开始验证 {len(config_files)} 个配置文件...")
        
        for config_path in config_files:
            logger.info(f"验证: {config_path.relative_to(project_root)}")
            
            is_valid, message, config = self.validate_config_file(config_path)
            
            result = {
                'file': str(config_path.relative_to(project_root)),
                'valid': is_valid,
                'message': message,
                'has_warnings': 'warning' in message.lower() or '警告' in message
            }
            
            if config:
                result['config_info'] = {
                    'experiment_name': config.experiment_name,
                    'model': config.model_config.model_name,
                    'epochs': config.num_epochs,
                    'batch_size': config.data_config.batch_size,
                    'learning_rate': config.optimization_config.learning_rate,
                    'use_lora': config.model_config.use_lora
                }
                
                # 统计配置类型
                if 'caption' in config_path.name:
                    results['summary']['config_types']['caption'] += 1
                elif 'detection' in config_path.name or 'od' in config_path.name:
                    results['summary']['config_types']['detection'] += 1
                elif 'ocr' in config_path.name:
                    results['summary']['config_types']['ocr'] += 1
                elif 'segmentation' in config_path.name:
                    results['summary']['config_types']['segmentation'] += 1
                elif 'multitask' in config_path.name:
                    results['summary']['config_types']['multitask'] += 1
            
            results['details'].append(result)
            
            if is_valid:
                results['passed'] += 1
                if result['has_warnings']:
                    results['warnings'] += 1
                logger.info(f"  ✅ {message}")
            else:
                results['failed'] += 1
                logger.error(f"  ❌ {message}")
                
                # 统计常见问题
                if 'learning_rate' in message:
                    results['summary']['common_issues']['learning_rate'] += 1
                if 'batch_size' in message:
                    results['summary']['common_issues']['batch_size'] += 1
                if 'LoRA' in message:
                    results['summary']['common_issues']['lora_config'] += 1
        
        return results
    
    def compare_configs(self, config1_path: Path, config2_path: Path) -> Dict[str, Any]:
        """比较两个配置文件"""
        try:
            # 加载配置文件
            with open(config1_path, 'r', encoding='utf-8') as f:
                config1_data = yaml.safe_load(f)
            
            with open(config2_path, 'r', encoding='utf-8') as f:
                config2_data = yaml.safe_load(f)
            
            # 递归比较字典
            differences = self._compare_dicts(config1_data, config2_data)
            
            # 分析差异类型
            diff_analysis = self._analyze_differences(differences)
            
            # 生成文本差异
            config1_text = yaml.dump(config1_data, default_flow_style=False, sort_keys=True)
            config2_text = yaml.dump(config2_data, default_flow_style=False, sort_keys=True)
            
            text_diff = list(difflib.unified_diff(
                config1_text.splitlines(keepends=True),
                config2_text.splitlines(keepends=True),
                fromfile=str(config1_path),
                tofile=str(config2_path)
            ))
            
            return {
                'config1': str(config1_path),
                'config2': str(config2_path),
                'differences': differences,
                'analysis': diff_analysis,
                'text_diff': ''.join(text_diff),
                'summary': {
                    'total_differences': len(differences),
                    'added_keys': len([d for d in differences if d['type'] == 'added']),
                    'removed_keys': len([d for d in differences if d['type'] == 'removed']),
                    'changed_values': len([d for d in differences if d['type'] == 'changed']),
                    'significant_changes': diff_analysis['significant_changes']
                }
            }
            
        except Exception as e:
            logger.error(f"比较配置文件失败: {e}")
            return {}
    
    def _compare_dicts(self, dict1: Dict, dict2: Dict, path: str = "") -> List[Dict[str, Any]]:
        """递归比较字典"""
        differences = []
        
        # 检查dict1中的键
        for key, value1 in dict1.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict2:
                differences.append({
                    'type': 'removed',
                    'path': current_path,
                    'value1': value1,
                    'value2': None
                })
            elif isinstance(value1, dict) and isinstance(dict2[key], dict):
                differences.extend(self._compare_dicts(value1, dict2[key], current_path))
            elif value1 != dict2[key]:
                differences.append({
                    'type': 'changed',
                    'path': current_path,
                    'value1': value1,
                    'value2': dict2[key]
                })
        
        # 检查dict2中新增的键
        for key, value2 in dict2.items():
            if key not in dict1:
                current_path = f"{path}.{key}" if path else key
                differences.append({
                    'type': 'added',
                    'path': current_path,
                    'value1': None,
                    'value2': value2
                })
        
        return differences
    
    def _analyze_differences(self, differences: List[Dict]) -> Dict[str, Any]:
        """分析配置差异"""
        analysis = {
            'significant_changes': [],
            'minor_changes': [],
            'categories': defaultdict(list)
        }
        
        # 重要参数列表
        significant_params = [
            'num_epochs', 'learning_rate', 'batch_size', 'model_name',
            'use_lora', 'lora_config.r', 'lora_config.lora_alpha'
        ]
        
        for diff in differences:
            path = diff['path']
            
            # 分类差异
            if any(param in path for param in significant_params):
                analysis['significant_changes'].append(diff)
            else:
                analysis['minor_changes'].append(diff)
            
            # 按类别分组
            if 'model_config' in path:
                analysis['categories']['model'].append(diff)
            elif 'data_config' in path:
                analysis['categories']['data'].append(diff)
            elif 'optimization_config' in path:
                analysis['categories']['optimization'].append(diff)
            elif 'task_scheduling_config' in path:
                analysis['categories']['task_scheduling'].append(diff)
            else:
                analysis['categories']['general'].append(diff)
        
        return analysis
    
    def optimize_config_for_hardware(
        self,
        config_path: Path,
        hardware_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """基于硬件信息优化配置"""
        try:
            if hardware_info is None:
                hardware_info = self.detect_hardware()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            config = TrainingConfig.from_dict(config_data)
            suggestions = []
            optimized_config = config_data.copy()
            
            gpu_memory = hardware_info.get('gpu_memory_gb', 12)
            gpu_count = hardware_info.get('gpu_count', 1)
            
            # 根据GPU内存优化参数
            optimal_params = self._get_optimal_params_for_memory(gpu_memory)
            
            # 批次大小优化
            current_batch_size = config.data_config.batch_size
            recommended_batch_size = optimal_params['batch_size'] * gpu_count
            
            if current_batch_size != recommended_batch_size:
                suggestions.append({
                    'type': 'batch_size',
                    'current': current_batch_size,
                    'recommended': recommended_batch_size,
                    'reason': f'基于 {gpu_memory}GB GPU 内存和 {gpu_count} 个GPU优化',
                    'impact': 'high'
                })
                optimized_config['data_config']['batch_size'] = recommended_batch_size
            
            # 梯度累积优化
            current_grad_accum = config.gradient_accumulation_steps
            recommended_grad_accum = optimal_params['grad_accum']
            
            if current_grad_accum != recommended_grad_accum:
                suggestions.append({
                    'type': 'gradient_accumulation_steps',
                    'current': current_grad_accum,
                    'recommended': recommended_grad_accum,
                    'reason': f'与批次大小配合，保持有效批次大小',
                    'impact': 'medium'
                })
                optimized_config['gradient_accumulation_steps'] = recommended_grad_accum
            
            # LoRA参数优化
            if config.model_config.use_lora:
                current_lora_r = config.model_config.lora_config.r
                recommended_lora_r = optimal_params['lora_r']
                
                if current_lora_r != recommended_lora_r:
                    suggestions.append({
                        'type': 'lora_rank',
                        'current': current_lora_r,
                        'recommended': recommended_lora_r,
                        'reason': f'基于GPU内存优化LoRA参数量',
                        'impact': 'medium'
                    })
                    optimized_config['model_config']['lora_config']['r'] = recommended_lora_r
                    optimized_config['model_config']['lora_config']['lora_alpha'] = recommended_lora_r * 2
            
            # 混合精度建议
            if gpu_memory >= 16 and not config.use_bf16:
                suggestions.append({
                    'type': 'mixed_precision',
                    'current': 'fp32',
                    'recommended': 'bf16',
                    'reason': '大内存GPU建议使用bf16以提高训练速度',
                    'impact': 'high'
                })
                optimized_config['use_bf16'] = True
                optimized_config['use_fp16'] = False
            
            # 数据加载优化
            cpu_count = hardware_info.get('cpu_count', 4)
            recommended_workers = min(cpu_count, 8)
            current_workers = config.data_config.num_workers
            
            if current_workers != recommended_workers:
                suggestions.append({
                    'type': 'num_workers',
                    'current': current_workers,
                    'recommended': recommended_workers,
                    'reason': f'基于CPU核心数 ({cpu_count}) 优化数据加载',
                    'impact': 'low'
                })
                optimized_config['data_config']['num_workers'] = recommended_workers
            
            return {
                'config_file': str(config_path),
                'hardware_info': hardware_info,
                'suggestions': suggestions,
                'optimized_config': optimized_config,
                'summary': {
                    'total_suggestions': len(suggestions),
                    'high_impact': len([s for s in suggestions if s['impact'] == 'high']),
                    'medium_impact': len([s for s in suggestions if s['impact'] == 'medium']),
                    'low_impact': len([s for s in suggestions if s['impact'] == 'low'])
                }
            }
            
        except Exception as e:
            logger.error(f"硬件优化失败: {e}")
            return {}
    
    def _get_optimal_params_for_memory(self, gpu_memory_gb: float) -> Dict[str, int]:
        """根据GPU内存获取最优参数"""
        thresholds = self.hardware_recommendations['gpu_memory_thresholds']
        
        # 找到最接近的内存阈值
        suitable_threshold = 4  # 默认最小配置
        for threshold in sorted(thresholds.keys()):
            if gpu_memory_gb >= threshold:
                suitable_threshold = threshold
            else:
                break
        
        return thresholds[suitable_threshold]
    
    def generate_hardware_adapted_config(
        self,
        base_config_path: Path,
        output_path: Path,
        hardware_info: Optional[Dict] = None
    ) -> bool:
        """生成硬件适配的配置文件"""
        try:
            optimization_result = self.optimize_config_for_hardware(base_config_path, hardware_info)
            
            if not optimization_result:
                return False
            
            optimized_config = optimization_result['optimized_config']
            
            # 添加硬件信息注释
            header = f"""# 硬件适配配置文件
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 基础配置: {base_config_path.name}
# 硬件信息: GPU {optimization_result['hardware_info']['gpu_memory_gb']:.1f}GB x{optimization_result['hardware_info']['gpu_count']}
# 优化建议数: {optimization_result['summary']['total_suggestions']}

"""
            
            # 写入优化后的配置
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header)
                yaml.dump(
                    optimized_config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    allow_unicode=True
                )
            
            logger.info(f"硬件适配配置已生成: {output_path}")
            logger.info(f"应用了 {optimization_result['summary']['total_suggestions']} 项优化")
            
            return True
            
        except Exception as e:
            logger.error(f"生成硬件适配配置失败: {e}")
            return False
    
    def format_config(self, config_path: Path, output_path: Optional[Path] = None) -> bool:
        """格式化配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 标准化格式
            formatted_yaml = yaml.dump(
                config_data,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
                width=100,
                allow_unicode=True
            )
            
            # 添加格式化信息
            header = f"# 配置文件格式化于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            formatted_content = header + formatted_yaml
            
            # 输出文件
            if output_path is None:
                output_path = config_path
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            logger.info(f"配置文件已格式化: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"格式化配置文件失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Florence-2 高级配置管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # validate-all 命令
    validate_all_parser = subparsers.add_parser('validate-all', help='验证所有配置文件')
    validate_all_parser.add_argument('--directory', help='配置文件目录')
    validate_all_parser.add_argument('--output', help='保存验证结果到JSON文件')
    
    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='比较两个配置文件')
    compare_parser.add_argument('config1', help='第一个配置文件')
    compare_parser.add_argument('config2', help='第二个配置文件')
    compare_parser.add_argument('--output', help='保存比较结果到JSON文件')
    
    # optimize 命令
    optimize_parser = subparsers.add_parser('optimize', help='优化配置参数')
    optimize_parser.add_argument('--config', required=True, help='要优化的配置文件')
    optimize_parser.add_argument('--hardware', help='硬件信息JSON文件')
    optimize_parser.add_argument('--output', help='保存优化建议到JSON文件')
    
    # hardware-adapt 命令
    hardware_adapt_parser = subparsers.add_parser('hardware-adapt', help='生成硬件适配配置')
    hardware_adapt_parser.add_argument('--base-config', required=True, help='基础配置文件')
    hardware_adapt_parser.add_argument('--output', required=True, help='输出配置文件')
    hardware_adapt_parser.add_argument('--gpu-memory', type=float, help='GPU内存大小(GB)')
    hardware_adapt_parser.add_argument('--gpu-count', type=int, default=1, help='GPU数量')
    
    # detect-hardware 命令
    subparsers.add_parser('detect-hardware', help='检测当前硬件配置')
    
    # format 命令
    format_parser = subparsers.add_parser('format', help='格式化配置文件')
    format_parser.add_argument('--config', required=True, help='要格式化的配置文件')
    format_parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    
    manager = AdvancedConfigManager()
    
    if args.command == 'validate-all':
        directory = Path(args.directory) if args.directory else None
        results = manager.validate_all_configs(directory)
        
        # 输出结果
        logger.info(f"\n=== 验证结果 ===")
        logger.info(f"总计: {results['total']} 个文件")
        logger.info(f"通过: {results['passed']} 个")
        logger.info(f"失败: {results['failed']} 个")
        logger.info(f"警告: {results['warnings']} 个")
        
        if results['summary']['config_types']:
            logger.info("\n配置类型分布:")
            for config_type, count in results['summary']['config_types'].items():
                logger.info(f"  {config_type}: {count} 个")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {args.output}")
    
    elif args.command == 'compare':
        config1_path = Path(args.config1)
        config2_path = Path(args.config2)
        
        results = manager.compare_configs(config1_path, config2_path)
        
        if results:
            logger.info(f"\n=== 配置比较结果 ===")
            logger.info(f"文件1: {results['config1']}")
            logger.info(f"文件2: {results['config2']}")
            logger.info(f"差异总数: {results['summary']['total_differences']}")
            logger.info(f"重要变更: {results['summary']['significant_changes']}")
            
            if results['differences']:
                logger.info("\n主要差异:")
                for diff in results['differences'][:10]:
                    logger.info(f"  {diff['type']}: {diff['path']}")
                    if diff['type'] == 'changed':
                        logger.info(f"    {diff['value1']} -> {diff['value2']}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"比较结果已保存到: {args.output}")
    
    elif args.command == 'optimize':
        config_path = Path(args.config)
        
        hardware_info = None
        if args.hardware:
            with open(args.hardware, 'r', encoding='utf-8') as f:
                hardware_info = json.load(f)
        
        results = manager.optimize_config_for_hardware(config_path, hardware_info)
        
        if results:
            logger.info(f"\n=== 硬件优化建议 ===")
            logger.info(f"配置文件: {results['config_file']}")
            logger.info(f"GPU: {results['hardware_info']['gpu_memory_gb']:.1f}GB x{results['hardware_info']['gpu_count']}")
            logger.info(f"建议总数: {results['summary']['total_suggestions']}")
            
            for suggestion in results['suggestions']:
                impact_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[suggestion['impact']]
                logger.info(f"\n{impact_icon} {suggestion['type']}:")
                logger.info(f"  当前值: {suggestion['current']}")
                logger.info(f"  推荐值: {suggestion['recommended']}")
                logger.info(f"  原因: {suggestion['reason']}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"优化建议已保存到: {args.output}")
    
    elif args.command == 'hardware-adapt':
        base_config_path = Path(args.base_config)
        output_path = Path(args.output)
        
        hardware_info = None
        if args.gpu_memory:
            hardware_info = {
                'gpu_memory_gb': args.gpu_memory,
                'gpu_count': args.gpu_count
            }
        
        success = manager.generate_hardware_adapted_config(
            base_config_path, output_path, hardware_info
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'detect-hardware':
        hardware_info = manager.detect_hardware()
        
        logger.info("\n=== 硬件检测结果 ===")
        logger.info(f"CPU核心数: {hardware_info['cpu_count']}")
        logger.info(f"系统内存: {hardware_info['memory_gb']:.1f} GB")
        logger.info(f"GPU可用: {hardware_info['gpu_available']}")
        
        if hardware_info['gpu_available']:
            logger.info(f"GPU数量: {hardware_info['gpu_count']}")
            logger.info(f"GPU内存: {hardware_info['gpu_memory_gb']:.1f} GB")
            logger.info(f"GPU型号: {', '.join(hardware_info['gpu_names'])}")
        
        # 保存硬件信息
        hardware_file = 'hardware_info.json'
        with open(hardware_file, 'w', encoding='utf-8') as f:
            json.dump(hardware_info, f, indent=2, ensure_ascii=False)
        logger.info(f"\n硬件信息已保存到: {hardware_file}")
    
    elif args.command == 'format':
        config_path = Path(args.config)
        output_path = Path(args.output) if args.output else None
        
        success = manager.format_config(config_path, output_path)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
