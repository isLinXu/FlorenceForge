"""FlorenceForge设备管理工具模块

提供设备检测、配置和管理功能
"""

import logging
import torch
import platform
import subprocess
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """设备信息数据类"""

    device_type: str  # cpu, cuda, mps
    device_id: Optional[int] = None
    name: str = ""
    memory_total: float = 0.0  # GB
    memory_available: float = 0.0  # GB
    compute_capability: Optional[str] = None
    is_available: bool = False

    def __str__(self) -> str:
        """返回设备信息的字符串表示

        Returns:
            str: 格式化的设备信息字符串，包含设备类型、名称和内存信息（如适用）
        """
        if self.device_type == "cpu":
            return f"CPU: {self.name}"
        elif self.device_type == "cuda":
            return f"CUDA {self.device_id}: {self.name} ({self.memory_available:.1f}GB/{self.memory_total:.1f}GB)"
        elif self.device_type == "mps":
            return f"MPS: {self.name}"
        else:
            return f"{self.device_type}: {self.name}"


def get_cpu_info() -> DeviceInfo:
    """获取CPU信息

    Returns:
        CPU设备信息
    """
    try:
        # 获取CPU名称
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            cpu_name = (
                result.stdout.strip() if result.returncode == 0 else "Unknown CPU"
            )
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
                else:
                    cpu_name = "Unknown CPU"
        else:
            cpu_name = platform.processor() or "Unknown CPU"

        # 获取内存信息
        import psutil

        memory = psutil.virtual_memory()

        return DeviceInfo(
            device_type="cpu",
            name=cpu_name,
            memory_total=memory.total / (1024**3),
            memory_available=memory.available / (1024**3),
            is_available=True,
        )

    except Exception as e:
        warnings.warn(f"获取CPU信息失败: {e}")
        return DeviceInfo(device_type="cpu", name="Unknown CPU", is_available=True)


def get_cuda_info() -> List[DeviceInfo]:
    """获取CUDA设备信息

    Returns:
        CUDA设备信息列表
    """
    devices = []

    if not torch.cuda.is_available():
        return devices

    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)

            # 获取内存信息
            total_memory = props.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            available_memory = total_memory - allocated_memory

            # 计算能力
            compute_capability = f"{props.major}.{props.minor}"

            devices.append(
                DeviceInfo(
                    device_type="cuda",
                    device_id=i,
                    name=props.name,
                    memory_total=total_memory,
                    memory_available=available_memory,
                    compute_capability=compute_capability,
                    is_available=True,
                )
            )

    except Exception as e:
        warnings.warn(f"获取CUDA信息失败: {e}")

    return devices


def get_mps_info() -> Optional[DeviceInfo]:
    """获取MPS设备信息（Apple Silicon）

    Returns:
        MPS设备信息
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None

    try:
        # 获取系统信息
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
            )

            chip_name = "Apple Silicon"
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Chip" in line:
                        chip_name = line.split(":")[1].strip()
                        break
        else:
            chip_name = "MPS Device"

        return DeviceInfo(device_type="mps", name=chip_name, is_available=True)

    except Exception as e:
        warnings.warn(f"获取MPS信息失败: {e}")
        return DeviceInfo(device_type="mps", name="MPS Device", is_available=True)


def get_device_info() -> Dict[str, Any]:
    """获取所有设备信息

    Returns:
        设备信息字典
    """
    info = {
        "cpu": get_cpu_info(),
        "cuda_devices": get_cuda_info(),
        "mps": get_mps_info(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
    }

    return info


def get_optimal_device(
    prefer_gpu: bool = True, min_memory_gb: float = 2.0, device_id: Optional[int] = None
) -> torch.device:
    """获取最优设备

    Args:
        prefer_gpu: 是否优先使用GPU
        min_memory_gb: 最小内存要求（GB）
        device_id: 指定设备ID

    Returns:
        最优设备
    """
    if device_id is not None:
        # 指定设备ID
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return torch.device(f"cuda:{device_id}")
        else:
            warnings.warn(f"指定的CUDA设备 {device_id} 不可用，使用CPU")
            return torch.device("cpu")

    if prefer_gpu:
        # 优先使用GPU
        if torch.cuda.is_available():
            # 选择内存最多的GPU
            cuda_devices = get_cuda_info()
            suitable_devices = [
                dev for dev in cuda_devices if dev.memory_available >= min_memory_gb
            ]

            if suitable_devices:
                best_device = max(suitable_devices, key=lambda x: x.memory_available)
                return torch.device(f"cuda:{best_device.device_id}")

        # 如果CUDA不可用，尝试MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

    # 使用CPU
    return torch.device("cpu")


def set_device(
    device: Union[str, torch.device, int], verbose: bool = True
) -> torch.device:
    """设置设备

    Args:
        device: 设备（字符串、torch.device或设备ID）
        verbose: 是否打印信息

    Returns:
        设置的设备
    """
    if isinstance(device, int):
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            device = torch.device(f"cuda:{device}")
        else:
            warnings.warn(f"CUDA设备 {device} 不可用，使用CPU")
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 验证设备可用性
    if device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA不可用，使用CPU")
            device = torch.device("cpu")
        elif device.index is not None and device.index >= torch.cuda.device_count():
            warnings.warn(f"CUDA设备 {device.index} 不存在，使用设备0")
            device = torch.device("cuda:0")
    elif device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            warnings.warn("MPS不可用，使用CPU")
            device = torch.device("cpu")

    if verbose:
        device_info = get_device_info()
        if device.type == "cuda":
            cuda_info = device_info["cuda_devices"][device.index or 0]
            logger.info(f"使用设备: {cuda_info}")
        elif device.type == "mps":
            mps_info = device_info["mps"]
            logger.info(f"使用设备: {mps_info}")
        else:
            cpu_info = device_info["cpu"]
            logger.info(f"使用设备: {cpu_info}")

    return device


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module, Dict, List],
    device: torch.device,
    non_blocking: bool = False,
) -> Union[torch.Tensor, torch.nn.Module, Dict, List]:
    """将对象移动到指定设备

    Args:
        obj: 要移动的对象
        device: 目标设备
        non_blocking: 是否非阻塞传输

    Returns:
        移动后的对象
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {
            key: move_to_device(value, device, non_blocking)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [move_to_device(item, device, non_blocking) for item in obj]
    elif isinstance(obj, tuple):
        moved_items = [move_to_device(item, device, non_blocking) for item in obj]
        if hasattr(obj, "_fields"):
            return type(obj)(*moved_items)
        return tuple(moved_items)
    else:
        return obj


def check_device_compatibility(
    model_name: str = "florence-2", required_memory_gb: float = 4.0
) -> Dict[str, Any]:
    """检查设备兼容性

    Args:
        model_name: 模型名称
        required_memory_gb: 所需内存（GB）

    Returns:
        兼容性检查结果
    """
    device_info = get_device_info()

    result = {
        "model_name": model_name,
        "required_memory_gb": required_memory_gb,
        "recommendations": [],
        "warnings": [],
        "compatible_devices": [],
    }

    # 检查CPU
    cpu_info = device_info["cpu"]
    if cpu_info.memory_available >= required_memory_gb:
        result["compatible_devices"].append("cpu")
        result["recommendations"].append("CPU可用，但训练速度较慢")
    else:
        result["warnings"].append(
            f"CPU内存不足: {cpu_info.memory_available:.1f}GB < {required_memory_gb}GB"
        )

    # 检查CUDA
    cuda_devices = device_info["cuda_devices"]
    for cuda_dev in cuda_devices:
        if cuda_dev.memory_available >= required_memory_gb:
            result["compatible_devices"].append(f"cuda:{cuda_dev.device_id}")

            # 检查计算能力
            if cuda_dev.compute_capability:
                major, minor = map(int, cuda_dev.compute_capability.split("."))
                if major >= 7:  # Volta架构及以上
                    result["recommendations"].append(
                        f"CUDA {cuda_dev.device_id} 推荐使用（支持Tensor Cores）"
                    )
                elif major >= 6:  # Pascal架构
                    result["recommendations"].append(
                        f"CUDA {cuda_dev.device_id} 可用（较老架构）"
                    )
                else:
                    result["warnings"].append(
                        f"CUDA {cuda_dev.device_id} 架构较老，可能性能不佳"
                    )
        else:
            result["warnings"].append(
                f"CUDA {cuda_dev.device_id} 内存不足: {cuda_dev.memory_available:.1f}GB < {required_memory_gb}GB"
            )

    # 检查MPS
    mps_info = device_info["mps"]
    if mps_info and mps_info.is_available:
        result["compatible_devices"].append("mps")
        result["recommendations"].append("MPS可用（Apple Silicon优化）")

    # 总体建议
    if not result["compatible_devices"]:
        result["warnings"].append("没有找到兼容的设备")
    elif len([d for d in result["compatible_devices"] if d.startswith("cuda")]) > 0:
        result["recommendations"].insert(0, "推荐使用CUDA设备以获得最佳性能")

    return result


def optimize_device_settings(
    device: torch.device, enable_amp: bool = True, enable_compile: bool = False
) -> Dict[str, Any]:
    """优化设备设置

    Args:
        device: 设备
        enable_amp: 是否启用自动混合精度
        enable_compile: 是否启用torch.compile

    Returns:
        优化设置信息
    """
    settings = {"device": str(device), "optimizations": []}

    if device.type == "cuda":
        # CUDA优化设置
        torch.backends.cudnn.benchmark = True
        settings["optimizations"].append("启用cuDNN benchmark")

        if enable_amp:
            # 检查是否支持AMP
            device_props = torch.cuda.get_device_properties(device)
            if device_props.major >= 7:  # Volta及以上
                settings["optimizations"].append("启用自动混合精度（AMP）")
                settings["amp_enabled"] = True
            else:
                settings["optimizations"].append("设备不支持Tensor Cores，禁用AMP")
                settings["amp_enabled"] = False

        # 设置内存分配策略
        torch.cuda.empty_cache()
        settings["optimizations"].append("清理CUDA缓存")

    elif device.type == "mps":
        # MPS优化设置
        if enable_amp:
            settings["optimizations"].append("MPS支持自动混合精度")
            settings["amp_enabled"] = True

    else:
        # CPU优化设置
        torch.set_num_threads(torch.get_num_threads())
        settings["optimizations"].append(f"使用 {torch.get_num_threads()} 个CPU线程")
        settings["amp_enabled"] = False

    # torch.compile优化（PyTorch 2.0+）
    if enable_compile and hasattr(torch, "compile"):
        settings["optimizations"].append("启用torch.compile优化")
        settings["compile_enabled"] = True
    else:
        settings["compile_enabled"] = False

    return settings


def setup_device(
    device: Optional[Union[str, torch.device, int]] = None,
    prefer_gpu: bool = True,
    min_memory_gb: float = 2.0,
    enable_amp: bool = True,
    verbose: bool = True,
) -> Tuple[torch.device, Dict[str, Any]]:
    """设置和优化设备

    Args:
        device: 指定设备（可选）
        prefer_gpu: 是否优先使用GPU
        min_memory_gb: 最小内存要求
        enable_amp: 是否启用自动混合精度
        verbose: 是否打印详细信息

    Returns:
        (设备, 设备设置信息)
    """
    if device is None:
        # 自动选择最优设备
        selected_device = get_optimal_device(prefer_gpu, min_memory_gb)
    else:
        # 使用指定设备
        selected_device = set_device(device, verbose=False)

    # 优化设备设置
    settings = optimize_device_settings(selected_device, enable_amp)

    if verbose:
        device_info = get_device_info()
        logger.info(f"设备设置完成: {selected_device}")
        if selected_device.type == "cuda":
            cuda_info = device_info["cuda_devices"][selected_device.index or 0]
            logger.info(f"GPU信息: {cuda_info}")
        elif selected_device.type == "mps":
            mps_info = device_info["mps"]
            logger.info(f"MPS信息: {mps_info}")
        else:
            cpu_info = device_info["cpu"]
            logger.info(f"CPU信息: {cpu_info}")

        if settings["optimizations"]:
            logger.info("优化设置:")
            for opt in settings["optimizations"]:
                logger.info(f"  - {opt}")

    return selected_device, settings


class DeviceManager:
    """设备管理器

    统一管理设备选择和优化
    """

    def __init__(
        self,
        auto_select: bool = True,
        prefer_gpu: bool = True,
        min_memory_gb: float = 2.0,
    ):
        """初始化设备管理器

        Args:
            auto_select: 是否自动选择设备
            prefer_gpu: 是否优先使用GPU
            min_memory_gb: 最小内存要求
        """
        self.prefer_gpu = prefer_gpu
        self.min_memory_gb = min_memory_gb

        if auto_select:
            self.device = get_optimal_device(prefer_gpu, min_memory_gb)
        else:
            self.device = torch.device("cpu")

        self.device_info = get_device_info()
        self.settings = optimize_device_settings(self.device)

    def set_device(self, device: Union[str, torch.device, int]) -> None:
        """设置设备

        Args:
            device: 设备
        """
        self.device = set_device(device)
        self.settings = optimize_device_settings(self.device)

    def move_to_device(self, obj, non_blocking: bool = False):
        """移动对象到当前设备

        Args:
            obj: 要移动的对象
            non_blocking: 是否非阻塞

        Returns:
            移动后的对象
        """
        return move_to_device(obj, self.device, non_blocking)

    def get_memory_info(self) -> Dict[str, float]:
        """获取当前设备内存信息

        Returns:
            内存信息
        """
        if self.device.type == "cuda":
            device_id = self.device.index or 0
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
            }
        else:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
            }

    def clear_cache(self) -> None:
        """清理设备缓存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_device_summary(self) -> str:
        """获取设备摘要信息

        Returns:
            设备摘要字符串
        """
        summary = [f"当前设备: {self.device}"]

        if self.device.type == "cuda":
            cuda_devices = self.device_info["cuda_devices"]
            if cuda_devices:
                device_id = self.device.index or 0
                if device_id < len(cuda_devices):
                    cuda_info = cuda_devices[device_id]
                    summary.append(f"设备名称: {cuda_info.name}")
                    summary.append(f"计算能力: {cuda_info.compute_capability}")
                    summary.append(
                        f"内存: {cuda_info.memory_available:.1f}GB/{cuda_info.memory_total:.1f}GB"
                    )

        elif self.device.type == "mps":
            mps_info = self.device_info["mps"]
            if mps_info:
                summary.append(f"设备名称: {mps_info.name}")

        else:
            cpu_info = self.device_info["cpu"]
            summary.append(f"处理器: {cpu_info.name}")
            summary.append(
                f"内存: {cpu_info.memory_available:.1f}GB/{cpu_info.memory_total:.1f}GB"
            )

        # 添加优化信息
        if self.settings["optimizations"]:
            summary.append("优化设置:")
            for opt in self.settings["optimizations"]:
                summary.append(f"  - {opt}")

        return "\n".join(summary)
