"""FlorenceForge IO工具模块

提供文件和数据的输入输出功能
"""

import json
import logging
import pickle
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Union, List, Optional

logger = logging.getLogger(__name__)

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)

def load_json(file_path: Union[str, Path]) -> Any:
    """从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(
    data: Any,
    file_path: Union[str, Path],
    default_flow_style: bool = False
) -> None:
    """保存数据为YAML文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        default_flow_style: 是否使用流式风格
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)

def load_yaml(file_path: Union[str, Path]) -> Any:
    """从YAML文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """保存数据为Pickle文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: Union[str, Path], trusted: bool = False) -> Any:
    """从Pickle文件加载数据

    安全警告：``pickle.load`` 会在反序列化时执行任意代码，绝不要加载来源不可信的
    pickle 文件。仅当你确认文件来自可信来源时，传入 ``trusted=True`` 允许加载。

    Args:
        file_path: 文件路径
        trusted: 是否确认文件来源可信。默认拒绝加载，避免执行不可信 pickle。

    Returns:
        加载的数据
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pickle文件不存在: {file_path}")

    if not trusted:
        raise ValueError(
            "拒绝加载不可信 Pickle 文件。pickle.load 会在反序列化时执行任意代码；"
            "确认文件来自可信来源后，请传入 trusted=True。"
        )

    logger.warning(
        "正在通过 pickle.load 反序列化可信文件 %s；请确认该文件未被篡改。",
        file_path,
    )
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> None:
    """复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 是否覆盖已存在的文件
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {dst}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def copy_directory(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> None:
    """复制目录
    
    Args:
        src: 源目录路径
        dst: 目标目录路径
        overwrite: 是否覆盖已存在的目录
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"源目录不存在: {src}")
    
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"目标目录已存在: {dst}")
        shutil.rmtree(dst)
    
    shutil.copytree(src, dst)

def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    return file_path.stat().st_size

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式
        recursive: 是否递归搜索
        
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))

def create_backup(
    file_path: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    timestamp: bool = True
) -> Path:
    """创建文件备份
    
    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录（默认为文件所在目录）
        timestamp: 是否在备份文件名中添加时间戳
        
    Returns:
        备份文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        ensure_dir(backup_dir)
    
    # 构建备份文件名
    if timestamp:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp_str}{file_path.suffix}"
    else:
        backup_name = f"{file_path.stem}_backup{file_path.suffix}"
    
    backup_path = backup_dir / backup_name
    
    # 复制文件
    shutil.copy2(file_path, backup_path)
    
    return backup_path

def safe_write(
    data: str,
    file_path: Union[str, Path],
    backup: bool = True,
    encoding: str = 'utf-8'
) -> None:
    """安全写入文件（先备份再写入）
    
    Args:
        data: 要写入的数据
        file_path: 文件路径
        backup: 是否创建备份
        encoding: 文件编码
    """
    file_path = Path(file_path)
    
    # 如果文件存在且需要备份
    if file_path.exists() and backup:
        create_backup(file_path)
    
    # 确保目录存在
    ensure_dir(file_path.parent)
    
    # 写入文件
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(data)

def read_text_file(
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> str:
    """读取文本文件
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        
    Returns:
        文件内容
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def write_text_file(
    content: str,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    backup: bool = False
) -> None:
    """写入文本文件
    
    Args:
        content: 文件内容
        file_path: 文件路径
        encoding: 文件编码
        backup: 是否创建备份
    """
    if backup:
        safe_write(content, file_path, backup=True, encoding=encoding)
    else:
        file_path = Path(file_path)
        ensure_dir(file_path.parent)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)

def append_text_file(
    content: str,
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> None:
    """追加内容到文本文件
    
    Args:
        content: 要追加的内容
        file_path: 文件路径
        encoding: 文件编码
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'a', encoding=encoding) as f:
        f.write(content)

class FileManager:
    """文件管理器
    
    提供便捷的文件操作接口
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """初始化文件管理器
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)
    
    def get_path(self, *parts: str) -> Path:
        """获取相对于基础目录的路径
        
        Args:
            *parts: 路径组件
            
        Returns:
            完整路径
        """
        return self.base_dir.joinpath(*parts)
    
    def save_json(self, data: Any, *path_parts: str, **kwargs) -> Path:
        """保存JSON文件
        
        Args:
            data: 要保存的数据
            *path_parts: 路径组件
            **kwargs: 传递给save_json的额外参数
            
        Returns:
            文件路径
        """
        file_path = self.get_path(*path_parts)
        save_json(data, file_path, **kwargs)
        return file_path
    
    def load_json(self, *path_parts: str) -> Any:
        """加载JSON文件
        
        Args:
            *path_parts: 路径组件
            
        Returns:
            加载的数据
        """
        file_path = self.get_path(*path_parts)
        return load_json(file_path)
    
    def save_pickle(self, data: Any, *path_parts: str) -> Path:
        """保存Pickle文件
        
        Args:
            data: 要保存的数据
            *path_parts: 路径组件
            
        Returns:
            文件路径
        """
        file_path = self.get_path(*path_parts)
        save_pickle(data, file_path)
        return file_path
    
    def load_pickle(self, *path_parts: str, trusted: bool = False) -> Any:
        """加载Pickle文件
        
        Args:
            *path_parts: 路径组件
            
        Returns:
            加载的数据
        """
        file_path = self.get_path(*path_parts)
        return load_pickle(file_path, trusted=trusted)
    
    def exists(self, *path_parts: str) -> bool:
        """检查文件是否存在
        
        Args:
            *path_parts: 路径组件
            
        Returns:
            文件是否存在
        """
        return self.get_path(*path_parts).exists()
    
    def ensure_dir(self, *path_parts: str) -> Path:
        """确保目录存在
        
        Args:
            *path_parts: 路径组件
            
        Returns:
            目录路径
        """
        dir_path = self.get_path(*path_parts)
        return ensure_dir(dir_path)
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> List[Path]:
        """列出文件
        
        Args:
            pattern: 文件模式
            recursive: 是否递归搜索
            
        Returns:
            文件路径列表
        """
        return list_files(self.base_dir, pattern, recursive)
    
    def cleanup_old_files(
        self,
        pattern: str = "*",
        days: int = 7,
        dry_run: bool = False
    ) -> List[Path]:
        """清理旧文件
        
        Args:
            pattern: 文件模式
            days: 保留天数
            dry_run: 是否只是预览（不实际删除）
            
        Returns:
            被删除（或将被删除）的文件列表
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        files_to_delete = []
        
        for file_path in self.list_files(pattern, recursive=True):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    files_to_delete.append(file_path)
                    if not dry_run:
                        file_path.unlink()
        
        return files_to_delete
