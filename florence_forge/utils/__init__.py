"""FlorenceForge工具模块

提供各种辅助功能和工具
"""

from .io import (
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    ensure_dir,
    copy_file
)
from .image import (
    load_image,
    resize_image,
    normalize_image,
    save_image,
    ImageProcessor
)
from .text import (
    clean_text,
    tokenize_text,
    extract_coordinates,
    format_detection_result,
    TextProcessor
)
from .visualization import (
    plot_training_curves,
    plot_task_distribution,
    visualize_detection_results,
    create_evaluation_dashboard,
    VisualizationManager
)
from .memory import (
    get_memory_usage,
    clear_cache,
    optimize_memory
)
from .device import (
    get_device_info,
    set_device,
    move_to_device,
    DeviceManager
)
from .tools import (
    Timer,
    timing_decorator,
    FileHasher,
    ConfigManager,
    ProgressTracker,
    suppress_warnings,
    retry_on_failure,
    ensure_list,
    flatten_dict,
    unflatten_dict
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    
    # IO
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'ensure_dir',
    'copy_file',
    
    # Image
    'load_image',
    'resize_image',
    'normalize_image',
    'save_image',
    'ImageProcessor',
    
    # Text
    'clean_text',
    'tokenize_text',
    'extract_coordinates',
    'format_detection_result',
    'TextProcessor',
    
    # Visualization
    'plot_training_curves',
    'plot_task_distribution',
    'visualize_detection_results',
    'create_evaluation_dashboard',
    'VisualizationManager',
    
    # Memory
    'get_memory_usage',
    'clear_cache',
    'optimize_memory',
    
    # Device
    'get_device_info',
    'set_device',
    'move_to_device',
    'DeviceManager',
    
    # Tools
    'Timer',
    'timing_decorator',
    'FileHasher',
    'ConfigManager',
    'ProgressTracker',
    'suppress_warnings',
    'retry_on_failure',
    'ensure_list',
    'flatten_dict',
    'unflatten_dict',
    
    # Optimization
    'ModelOptimizer',
    'MemoryOptimizer',
    'create_model_optimizer',
    'quick_quantize',
    'quick_prune'
]