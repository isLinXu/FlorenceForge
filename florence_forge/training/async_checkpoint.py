#!/usr/bin/env python3
"""异步检查点保存模块

职责：
- 后台线程保存检查点
- 不阻塞训练循环
- 支持压缩存储
- 自动清理旧检查点
"""

import gzip
import logging
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional


from ._checkpoint_io import atomic_torch_save, prune_checkpoints

logger = logging.getLogger(__name__)

__all__ = ["AsyncCheckpointSaver"]


class AsyncCheckpointSaver:
    """异步检查点保存器

    特性：
    - 后台线程保存，不阻塞训练
    - 支持压缩（gzip / lz4 / zstd）
    - 自动清理超出 max_checkpoints 的旧检查点
    - 异常安全（保存失败时清理临时文件）
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        compression: str = "none",
        async_save: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.async_save = async_save
        self._executor = ThreadPoolExecutor(max_workers=1) if async_save else None
        self._pending_future: Optional[Future] = None

    def save(self, state: Dict[str, Any], checkpoint_path: str) -> None:
        """保存检查点（默认异步，不阻塞训练循环）。"""
        if self.async_save and self._executor is not None:
            self._save_async(state, checkpoint_path)
        else:
            self._write_checkpoint(state, checkpoint_path)

    def wait_for_pending(self, timeout: Optional[float] = 300) -> None:
        """等待上一次异步保存完成。"""
        if self._pending_future is not None:
            self._pending_future.result(timeout=timeout)
            self._pending_future = None

    def _save_async(self, state: Dict[str, Any], path: str) -> None:
        self.wait_for_pending()
        if self._executor is not None:
            self._pending_future = self._executor.submit(self._write_checkpoint, state, path)
            logger.debug("检查点保存任务已提交（异步）：%s", path)

    def _write_checkpoint(self, state: Dict[str, Any], path: str) -> None:
        """实际写入检查点（原子写 + 可选压缩）。"""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        temp_path = checkpoint_dir / "checkpoint.pt.tmp"

        try:
            atomic_torch_save(state, temp_path)
            compressed_path = self._maybe_compress(temp_path, checkpoint_path)
            if compressed_path != checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
            if compressed_path != checkpoint_path:
                compressed_path.rename(checkpoint_path)
            elif temp_path.exists():
                temp_path.rename(checkpoint_path)
            self._cleanup_old_checkpoints()
            logger.info("检查点已保存：%s", checkpoint_dir)
        except Exception:
            for leftover in (temp_path, checkpoint_path):
                if leftover.exists():
                    leftover.unlink(missing_ok=True)
            raise

    def _maybe_compress(self, source: Path, target: Path) -> Path:
        """按配置压缩检查点文件。"""
        if self.compression in (None, "", "none"):
            return source

        if self.compression == "gzip":
            compressed = target.with_suffix(".pt.gz")
            with open(source, "rb") as src, gzip.open(compressed, "wb") as dst:
                shutil.copyfileobj(src, dst)
            source.unlink(missing_ok=True)
            return compressed

        if self.compression == "lz4":
            try:
                import lz4.frame
            except ImportError:
                logger.warning("lz4 未安装，跳过压缩")
                return source
            compressed = target.with_suffix(".pt.lz4")
            with open(source, "rb") as src, lz4.frame.open(compressed, "wb") as dst:
                dst.write(src.read())
            source.unlink(missing_ok=True)
            return compressed

        if self.compression == "zstd":
            try:
                import zstandard as zstd
            except ImportError:
                logger.warning("zstandard 未安装，跳过压缩")
                return source
            compressed = target.with_suffix(".pt.zst")
            with open(source, "rb") as src, open(compressed, "wb") as dst:
                dst.write(zstd.ZstdCompressor().compress(src.read()))
            source.unlink(missing_ok=True)
            return compressed

        logger.warning("未知压缩格式 '%s'，跳过压缩", self.compression)
        return source

    def _cleanup_old_checkpoints(self) -> None:
        """清理超出 max_checkpoints 的旧检查点目录。"""
        if self.max_checkpoints <= 0:
            return

        checkpoint_dirs: List[Path] = sorted(
            [d for d in self.checkpoint_dir.glob("checkpoint-epoch-*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )

        def _remove(d: Path) -> None:
            shutil.rmtree(d)
            logger.info("已删除旧检查点：%s", d)

        prune_checkpoints(
            checkpoint_dirs,
            self.max_checkpoints,
            sort_key=lambda d: d.stat().st_mtime,
            is_protected=lambda d: (d / "BEST_MODEL").exists(),
            remove=_remove,
        )

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池并等待所有保存任务完成。"""
        self.wait_for_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
