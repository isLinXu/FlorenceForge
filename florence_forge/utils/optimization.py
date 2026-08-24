"""Model optimization utilities.

This module provides tools for model optimization including quantization,
pruning, and performance optimization techniques.
"""

import torch
import torch.nn as nn
import logging
import time
import psutil
import gc
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Model optimization utilities for Florence models."""
    
    def __init__(self, model: nn.Module):
        """Initialize model optimizer.
        
        Args:
            model: Model to optimize
        """
        self.model = model
        self.original_state = None
        self.optimization_history: List[Dict[str, Any]] = []
        
    def save_original_state(self) -> None:
        """Save original model state for restoration."""
        self.original_state = {
            'state_dict': self.model.state_dict(),
            'model_size': self.get_model_size(),
            'param_count': self.count_parameters()
        }
        logger.info("Original model state saved")
    
    def restore_original_state(self) -> None:
        """Restore model to original state."""
        if self.original_state is None:
            logger.warning("No original state saved")
            return
        
        self.model.load_state_dict(self.original_state['state_dict'])
        logger.info("Model restored to original state")
    
    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        dtype: torch.dtype = torch.qint8,
        backend: str = "fbgemm"
    ) -> nn.Module:
        """Apply quantization to the model.

        .. deprecated::
            Prefer :class:`florence_forge.optimization.quantization.ModelQuantizer`,
            which is the unified quantization entry point (bnb-4bit/8bit, GPTQ,
            AWQ and dynamic-int8). This method now delegates its ``dynamic`` path
            to ``ModelQuantizer.quantize_module_dynamic`` and is kept only for
            backward compatibility.

        Args:
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            dtype: Quantization data type
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        import warnings

        warnings.warn(
            "ModelOptimizer.quantize_model is deprecated; use "
            "florence_forge.optimization.quantization.ModelQuantizer instead "
            "(the unified quantization entry point).",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            original_size = self.get_model_size()
            
            if quantization_type == "dynamic":
                # Delegate to the unified quantizer (single source of truth).
                from ..optimization.quantization import ModelQuantizer

                quantized_model = ModelQuantizer.quantize_module_dynamic(
                    self.model,
                    dtype=dtype,
                    target_layers={nn.Linear, nn.Conv2d},
                )
            elif quantization_type == "static":
                # Prepare model for static quantization
                self.model.qconfig = torch.quantization.get_default_qconfig(backend)
                torch.quantization.prepare(self.model, inplace=True)
                
                # Note: In practice, you would need calibration data here
                logger.warning("Static quantization requires calibration data")
                
                quantized_model = torch.quantization.convert(self.model, inplace=False)
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            quantized_size = self.get_model_size(quantized_model)
            # Dynamically quantized layers store packed weights that are not
            # exposed through ``parameters()``/``buffers()``, so the measured
            # size can be 0. Guard against dividing by zero in that case.
            compression_ratio = (
                original_size / quantized_size if quantized_size > 0 else float("inf")
            )
            
            optimization_info = {
                'type': 'quantization',
                'method': quantization_type,
                'original_size_mb': original_size,
                'optimized_size_mb': quantized_size,
                'compression_ratio': compression_ratio,
                'dtype': str(dtype)
            }
            
            self.optimization_history.append(optimization_info)
            
            logger.info(f"Model quantized: {original_size:.2f}MB -> {quantized_size:.2f}MB "
                       f"(compression ratio: {compression_ratio:.2f}x)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
    
    def prune_model(
        self,
        pruning_ratio: float = 0.2,
        structured: bool = False,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Apply pruning to the model.
        
        Args:
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
            importance_scores: Custom importance scores for parameters
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            original_params = self.count_parameters()
            
            # Apply pruning to linear and convolutional layers
            modules_to_prune = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    modules_to_prune.append((module, 'weight'))
            
            if structured:
                # Structured pruning (remove entire channels/filters)
                for module, param_name in modules_to_prune:
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(
                            module, param_name, amount=pruning_ratio, n=2, dim=0
                        )
                    elif isinstance(module, nn.Linear):
                        prune.ln_structured(
                            module, param_name, amount=pruning_ratio, n=2, dim=0
                        )
            else:
                # Unstructured pruning (remove individual weights)
                if importance_scores:
                    # Use custom importance scores
                    for module, param_name in modules_to_prune:
                        module_name = None
                        for name, mod in self.model.named_modules():
                            if mod is module:
                                module_name = name
                                break
                        
                        if module_name and module_name in importance_scores:
                            prune.global_unstructured(
                                [(module, param_name)],
                                pruning_method=prune.L1Unstructured,
                                amount=pruning_ratio,
                                importance_scores=importance_scores[module_name]
                            )
                        else:
                            prune.l1_unstructured(module, param_name, amount=pruning_ratio)
                else:
                    # Global magnitude-based pruning
                    prune.global_unstructured(
                        modules_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=pruning_ratio
                    )
            
            # Make pruning permanent
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
            
            pruned_params = self.count_parameters()
            actual_pruning_ratio = 1 - (pruned_params / original_params)
            
            optimization_info = {
                'type': 'pruning',
                'method': 'structured' if structured else 'unstructured',
                'target_ratio': pruning_ratio,
                'actual_ratio': actual_pruning_ratio,
                'original_params': original_params,
                'pruned_params': pruned_params
            }
            
            self.optimization_history.append(optimization_info)
            
            logger.info(f"Model pruned: {original_params:,} -> {pruned_params:,} parameters "
                       f"(pruning ratio: {actual_pruning_ratio:.2%})")
            
            return self.model
            
        except ImportError:
            logger.error("Pruning requires PyTorch >= 1.4.0")
            raise
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise
    
    def optimize_for_inference(self) -> nn.Module:
        """Optimize model for inference.
        
        Returns:
            Optimized model
        """
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Fuse operations where possible
            if hasattr(torch.quantization, 'fuse_modules'):
                # Try to fuse conv-bn-relu patterns
                try:
                    fused_model = torch.quantization.fuse_modules(
                        self.model,
                        [['conv', 'bn', 'relu']] if hasattr(self.model, 'conv') else []
                    )
                    logger.info("Model operations fused for inference")
                    return fused_model
                except Exception as e:
                    logger.warning(f"Operation fusion failed: {e}")
            
            optimization_info = {
                'type': 'inference_optimization',
                'method': 'eval_mode_no_grad',
                'gradient_disabled': True
            }
            
            self.optimization_history.append(optimization_info)
            
            logger.info("Model optimized for inference")
            return self.model
            
        except Exception as e:
            logger.error(f"Inference optimization failed: {e}")
            raise
    
    def get_model_size(self, model: Optional[nn.Module] = None) -> float:
        """Get model size in MB.
        
        Args:
            model: Model to measure (uses self.model if None)
            
        Returns:
            Model size in MB
        """
        if model is None:
            model = self.model
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def count_parameters(self, model: Optional[nn.Module] = None) -> int:
        """Count total number of parameters.
        
        Args:
            model: Model to count (uses self.model if None)
            
        Returns:
            Total parameter count
        """
        if model is None:
            model = self.model
        
        return sum(p.numel() for p in model.parameters())
    
    def count_trainable_parameters(self, model: Optional[nn.Module] = None) -> int:
        """Count trainable parameters.
        
        Args:
            model: Model to count (uses self.model if None)
            
        Returns:
            Trainable parameter count
        """
        if model is None:
            model = self.model
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def benchmark_model(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            device: Device to run benchmark on
            
        Returns:
            Benchmark results
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Benchmark runs
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        memory_usage = end_memory - start_memory
        
        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time,
            'memory_usage_mb': memory_usage,
            'model_size_mb': self.get_model_size(),
            'parameter_count': self.count_parameters()
        }
        
        logger.info(f"Benchmark results: {avg_time*1000:.2f}ms avg, "
                   f"{throughput:.2f} FPS, {memory_usage:.2f}MB memory")
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations applied.
        
        Returns:
            Optimization summary
        """
        return {
            'optimization_history': self.optimization_history,
            'current_model_size_mb': self.get_model_size(),
            'current_parameter_count': self.count_parameters(),
            'original_state_available': self.original_state is not None
        }


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU and system cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Memory cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage.
        
        Returns:
            Memory usage statistics
        """
        memory_info = {
            'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'system_memory_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_memory_percent': (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            })
        
        return memory_info
    
    @staticmethod
    def optimize_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        max_memory_mb: float = 8000,
        start_batch_size: int = 1
    ) -> int:
        """Find optimal batch size for given memory constraint.
        
        Args:
            model: Model to test
            input_shape: Input shape (without batch dimension)
            max_memory_mb: Maximum memory usage in MB
            start_batch_size: Starting batch size for search
            
        Returns:
            Optimal batch size
        """
        model.eval()
        device = next(model.parameters()).device
        
        optimal_batch_size = start_batch_size
        
        for batch_size in range(start_batch_size, 128):
            try:
                # Clear cache before test
                MemoryOptimizer.clear_cache()
                
                # Create test input
                test_input = torch.randn(batch_size, *input_shape, device=device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_usage = MemoryOptimizer.get_memory_usage()
                current_memory = memory_usage.get('gpu_memory_allocated_mb', 
                                                memory_usage['system_memory_mb'])
                
                if current_memory > max_memory_mb:
                    break
                
                optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                raise
        
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size


def create_model_optimizer(model: nn.Module) -> ModelOptimizer:
    """Create a model optimizer instance.
    
    Args:
        model: Model to optimize
        
    Returns:
        ModelOptimizer instance
    """
    optimizer = ModelOptimizer(model)
    optimizer.save_original_state()
    return optimizer


def quick_quantize(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
    """Quick model quantization.
    
    Args:
        model: Model to quantize
        quantization_type: Type of quantization
        
    Returns:
        Quantized model
    """
    optimizer = ModelOptimizer(model)
    return optimizer.quantize_model(quantization_type)


def quick_prune(model: nn.Module, pruning_ratio: float = 0.2) -> nn.Module:
    """Quick model pruning.
    
    Args:
        model: Model to prune
        pruning_ratio: Fraction of parameters to prune
        
    Returns:
        Pruned model
    """
    optimizer = ModelOptimizer(model)
    return optimizer.prune_model(pruning_ratio)