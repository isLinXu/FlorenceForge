#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型优化器

提供模型量化、剪枝、蒸馏等优化技术
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Union, Optional, Callable, List, Dict, Any
import logging
import copy

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """模型优化器
    
    提供多种模型优化技术
    """
    
    def __init__(self, model: nn.Module):
        """初始化优化器
        
        Args:
            model: 要优化的模型
        """
        self.original_model = model
        self.optimized_model = None
    
    def quantize_dynamic(
        self,
        qconfig_spec: Optional[Dict] = None,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """动态量化
        
        Args:
            qconfig_spec: 量化配置
            dtype: 量化数据类型
            
        Returns:
            量化后的模型
        """
        try:
            model_copy = copy.deepcopy(self.original_model)
            model_copy.eval()
            
            # 默认量化配置
            if qconfig_spec is None:
                qconfig_spec = {
                    nn.Linear: torch.quantization.default_dynamic_qconfig,
                    nn.LSTM: torch.quantization.default_dynamic_qconfig,
                    nn.GRU: torch.quantization.default_dynamic_qconfig
                }
            
            quantized_model = torch.quantization.quantize_dynamic(
                model_copy,
                qconfig_spec,
                dtype=dtype
            )
            
            self.optimized_model = quantized_model
            logger.info("动态量化完成")
            
            return quantized_model
        
        except Exception as e:
            logger.error(f"动态量化失败: {e}")
            raise
    
    def quantize_static(
        self,
        calibration_data_loader,
        qconfig: Optional[torch.quantization.QConfig] = None
    ) -> nn.Module:
        """静态量化
        
        Args:
            calibration_data_loader: 校准数据加载器
            qconfig: 量化配置
            
        Returns:
            量化后的模型
        """
        try:
            model_copy = copy.deepcopy(self.original_model)
            model_copy.eval()
            
            # 设置量化配置
            if qconfig is None:
                qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            model_copy.qconfig = qconfig
            
            # 准备量化
            torch.quantization.prepare(model_copy, inplace=True)
            
            # 校准
            logger.info("开始校准...")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_data_loader):
                    model_copy(data)
                    if batch_idx >= 100:  # 限制校准样本数量
                        break
            
            # 转换为量化模型
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
            
            self.optimized_model = quantized_model
            logger.info("静态量化完成")
            
            return quantized_model
        
        except Exception as e:
            logger.error(f"静态量化失败: {e}")
            raise
    
    def prune_unstructured(
        self,
        pruning_ratio: float = 0.2,
        pruning_method: str = "magnitude"
    ) -> nn.Module:
        """非结构化剪枝
        
        Args:
            pruning_ratio: 剪枝比例
            pruning_method: 剪枝方法
            
        Returns:
            剪枝后的模型
        """
        try:
            import torch.nn.utils.prune as prune
            
            model_copy = copy.deepcopy(self.original_model)
            
            # 收集要剪枝的参数
            parameters_to_prune = []
            for module in model_copy.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            # 应用剪枝
            if pruning_method == "magnitude":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio
                )
            elif pruning_method == "random":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=pruning_ratio
                )
            else:
                raise ValueError(f"不支持的剪枝方法: {pruning_method}")
            
            # 移除剪枝重参数化
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            self.optimized_model = model_copy
            logger.info(f"非结构化剪枝完成，剪枝比例: {pruning_ratio}")
            
            return model_copy
        
        except ImportError:
            logger.error("PyTorch版本不支持剪枝功能")
            raise
        except Exception as e:
            logger.error(f"非结构化剪枝失败: {e}")
            raise
    
    def prune_structured(
        self,
        pruning_ratio: float = 0.2,
        dim: int = 0
    ) -> nn.Module:
        """结构化剪枝
        
        Args:
            pruning_ratio: 剪枝比例
            dim: 剪枝维度
            
        Returns:
            剪枝后的模型
        """
        try:
            import torch.nn.utils.prune as prune
            
            model_copy = copy.deepcopy(self.original_model)
            
            # 对每个线性层和卷积层进行结构化剪枝
            for module in model_copy.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=pruning_ratio,
                        n=2,
                        dim=dim
                    )
                    prune.remove(module, 'weight')
            
            self.optimized_model = model_copy
            logger.info(f"结构化剪枝完成，剪枝比例: {pruning_ratio}")
            
            return model_copy
        
        except ImportError:
            logger.error("PyTorch版本不支持剪枝功能")
            raise
        except Exception as e:
            logger.error(f"结构化剪枝失败: {e}")
            raise
    
    def knowledge_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader,
        num_epochs: int = 10,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = "cpu"
    ) -> nn.Module:
        """知识蒸馏
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            device: 设备
            
        Returns:
            蒸馏后的学生模型
        """
        try:
            device = torch.device(device)
            student_model = student_model.to(device)
            teacher_model = teacher_model.to(device)
            teacher_model.eval()
            
            optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
            criterion_ce = nn.CrossEntropyLoss()
            criterion_kl = nn.KLDivLoss(reduction='batchmean')
            
            logger.info(f"开始知识蒸馏训练，共{num_epochs}轮")
            
            for epoch in range(num_epochs):
                student_model.train()
                total_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    
                    # 学生模型输出
                    student_output = student_model(data)
                    
                    # 教师模型输出
                    with torch.no_grad():
                        teacher_output = teacher_model(data)
                    
                    # 计算损失
                    # 硬标签损失
                    loss_ce = criterion_ce(student_output, target)
                    
                    # 软标签损失（知识蒸馏）
                    loss_kl = criterion_kl(
                        torch.log_softmax(student_output / temperature, dim=1),
                        torch.softmax(teacher_output / temperature, dim=1)
                    ) * (temperature ** 2)
                    
                    # 总损失
                    loss = alpha * loss_kl + (1 - alpha) * loss_ce
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 100 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"Batch {batch_idx}, Loss: {loss.item():.4f}"
                        )
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            self.optimized_model = student_model
            logger.info("知识蒸馏完成")
            
            return student_model
        
        except Exception as e:
            logger.error(f"知识蒸馏失败: {e}")
            raise
    
    def optimize_for_mobile(self) -> nn.Module:
        """移动端优化
        
        Returns:
            优化后的模型
        """
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            
            # 先转换为TorchScript
            model_copy = copy.deepcopy(self.original_model)
            model_copy.eval()
            
            # 这里需要示例输入来trace模型
            # 实际使用时需要提供合适的输入
            logger.warning("移动端优化需要示例输入来trace模型")
            
            # traced_model = torch.jit.trace(model_copy, example_input)
            # optimized_model = optimize_for_mobile(traced_model)
            
            # 暂时返回原模型
            self.optimized_model = model_copy
            logger.info("移动端优化完成（需要提供示例输入）")
            
            return model_copy
        
        except ImportError:
            logger.error("移动端优化功能不可用")
            raise
        except Exception as e:
            logger.error(f"移动端优化失败: {e}")
            raise
    
    def fuse_modules(self, modules_to_fuse: Optional[List[List[str]]] = None) -> nn.Module:
        """模块融合优化
        
        Args:
            modules_to_fuse: 要融合的模块列表
            
        Returns:
            融合后的模型
        """
        try:
            model_copy = copy.deepcopy(self.original_model)
            model_copy.eval()
            
            if modules_to_fuse is None:
                # 自动检测可融合的模块
                modules_to_fuse = self._detect_fusable_modules(model_copy)
            
            if modules_to_fuse:
                torch.quantization.fuse_modules(model_copy, modules_to_fuse, inplace=True)
                logger.info(f"模块融合完成，融合了 {len(modules_to_fuse)} 组模块")
            else:
                logger.info("未发现可融合的模块")
            
            self.optimized_model = model_copy
            return model_copy
        
        except Exception as e:
            logger.error(f"模块融合失败: {e}")
            raise
    
    def _detect_fusable_modules(self, model: nn.Module) -> List[List[str]]:
        """自动检测可融合的模块
        
        Args:
            model: 模型
            
        Returns:
            可融合模块列表
        """
        fusable_modules = []
        
        # 简单的启发式检测
        # 实际应用中可能需要更复杂的逻辑
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                submodules = list(module.children())
                for i in range(len(submodules) - 1):
                    if (isinstance(submodules[i], nn.Conv2d) and 
                        isinstance(submodules[i+1], nn.BatchNorm2d)):
                        fusable_modules.append([f"{name}.{i}", f"{name}.{i+1}"])
        
        return fusable_modules
    
    def compare_models(
        self,
        test_loader,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """比较原模型和优化模型的性能
        
        Args:
            test_loader: 测试数据加载器
            metrics: 要计算的指标
            
        Returns:
            性能比较结果
        """
        if self.optimized_model is None:
            raise ValueError("尚未进行模型优化")
        
        if metrics is None:
            metrics = ['accuracy', 'inference_time', 'model_size']
        
        results = {
            'original': {},
            'optimized': {},
            'improvement': {}
        }
        
        # 评估原模型
        original_metrics = self._evaluate_model(self.original_model, test_loader, metrics)
        results['original'] = original_metrics
        
        # 评估优化模型
        optimized_metrics = self._evaluate_model(self.optimized_model, test_loader, metrics)
        results['optimized'] = optimized_metrics
        
        # 计算改进
        for metric in metrics:
            if metric in original_metrics and metric in optimized_metrics:
                if metric == 'inference_time':
                    # 推理时间越小越好
                    improvement = (original_metrics[metric] - optimized_metrics[metric]) / original_metrics[metric] * 100
                else:
                    # 其他指标越大越好
                    improvement = (optimized_metrics[metric] - original_metrics[metric]) / original_metrics[metric] * 100
                
                results['improvement'][metric] = improvement
        
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader,
        metrics: List[str]
    ) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            metrics: 要计算的指标
            
        Returns:
            性能指标
        """
        import time
        
        results = {}
        model.eval()
        
        if 'accuracy' in metrics:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            results['accuracy'] = correct / total
        
        if 'inference_time' in metrics:
            # 测量推理时间
            times = []
            
            with torch.no_grad():
                for i, (data, _) in enumerate(test_loader):
                    if i >= 100:  # 只测试前100个batch
                        break
                    
                    start_time = time.time()
                    _ = model(data)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
            
            results['inference_time'] = np.mean(times)
        
        if 'model_size' in metrics:
            # 计算模型大小（参数数量）
            total_params = sum(p.numel() for p in model.parameters())
            results['model_size'] = total_params
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要
        
        Returns:
            优化摘要信息
        """
        if self.optimized_model is None:
            return {"status": "未进行优化"}
        
        original_params = sum(p.numel() for p in self.original_model.parameters())
        optimized_params = sum(p.numel() for p in self.optimized_model.parameters())
        
        summary = {
            "original_parameters": original_params,
            "optimized_parameters": optimized_params,
            "parameter_reduction": (original_params - optimized_params) / original_params * 100,
            "compression_ratio": original_params / optimized_params if optimized_params > 0 else float('inf')
        }
        
        return summary