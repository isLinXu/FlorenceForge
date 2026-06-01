"""FlorenceForge 图像缓存层

从 ``dataset.py`` 抽出的按字节预算 LRU 图像 payload 缓存。

设计要点：

- 缓存的是 RGB ``bytes`` payload，而非可变的 PIL ``Image`` 对象，避免调用方的数据
  增强或多线程处理修改共享对象。
- 按总字节预算淘汰（LRU），避免固定条数缓存高分辨率图像导致 RSS 放大。
- 默认预算为 256 MiB，可通过环境变量 ``FLORENCE_FORGE_IMAGE_CACHE_MAX_BYTES`` 调整。

``_load_image_cached`` 暴露 ``cache_clear`` / ``cache_info`` / ``cache_bytes`` /
``set_cache_max_bytes`` 几个属性，保留测试与调试入口。``dataset.py`` 重新导出
``_load_image_cached`` 以保持历史导入路径与单测 patch 目标不变。
"""

import os
import threading
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Tuple

from PIL import Image

_ImagePayloadCacheInfo = namedtuple(
    "_ImagePayloadCacheInfo",
    ["hits", "misses", "maxsize", "currsize"],
)
_IMAGE_PAYLOAD_CACHE_DEFAULT_MAX_BYTES = int(
    os.environ.get("FLORENCE_FORGE_IMAGE_CACHE_MAX_BYTES", str(256 * 1024 * 1024))
)
_image_payload_cache: "OrderedDict[str, Tuple[Tuple[int, int], bytes]]" = OrderedDict()
_image_payload_cache_hits = 0
_image_payload_cache_misses = 0
_image_payload_cache_bytes = 0
_image_payload_cache_max_bytes = _IMAGE_PAYLOAD_CACHE_DEFAULT_MAX_BYTES
_image_payload_cache_lock = threading.RLock()


def _load_image_payload_cached(image_path: str) -> Tuple[Tuple[int, int], bytes]:
    """缓存图像 RGB payload，避免跨调用复用可变 PIL Image 对象

    使用按字节预算的 LRU 缓存策略，避免固定条数缓存高分辨率 RGB bytes
    导致 RSS 放大。默认预算可通过 FLORENCE_FORGE_IMAGE_CACHE_MAX_BYTES 调整。

    Args:
        image_path: 图像文件路径

    Returns:
        (图像尺寸, RGB 字节) 元组

    Raises:
        FileNotFoundError: 当图像文件不存在时
        IOError: 当图像文件无法解码时
    """
    global _image_payload_cache_hits
    global _image_payload_cache_misses
    global _image_payload_cache_bytes

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    cache_key = str(Path(image_path).resolve())
    with _image_payload_cache_lock:
        cached = _image_payload_cache.get(cache_key)
        if cached is not None:
            _image_payload_cache_hits += 1
            _image_payload_cache.move_to_end(cache_key)
            return cached
        _image_payload_cache_misses += 1

    try:
        with Image.open(image_path) as img:
            rgb = img.convert('RGB')
            payload = rgb.tobytes()
    except Exception as e:
        raise IOError(f"无法加载图像 {image_path}: {e}") from e

    payload_size = len(payload)
    result = (rgb.size, payload)
    with _image_payload_cache_lock:
        if _image_payload_cache_max_bytes <= 0 or payload_size > _image_payload_cache_max_bytes:
            return result

        existing = _image_payload_cache.get(cache_key)
        if existing is not None:
            _image_payload_cache.move_to_end(cache_key)
            return existing

        _image_payload_cache[cache_key] = result
        _image_payload_cache_bytes += payload_size
        while _image_payload_cache_bytes > _image_payload_cache_max_bytes and _image_payload_cache:
            _, (_, evicted_payload) = _image_payload_cache.popitem(last=False)
            _image_payload_cache_bytes -= len(evicted_payload)

    return result


def _image_payload_cache_clear() -> None:
    global _image_payload_cache_hits
    global _image_payload_cache_misses
    global _image_payload_cache_bytes

    with _image_payload_cache_lock:
        _image_payload_cache.clear()
        _image_payload_cache_hits = 0
        _image_payload_cache_misses = 0
        _image_payload_cache_bytes = 0


def _image_payload_cache_info() -> _ImagePayloadCacheInfo:
    with _image_payload_cache_lock:
        return _ImagePayloadCacheInfo(
            hits=_image_payload_cache_hits,
            misses=_image_payload_cache_misses,
            maxsize=_image_payload_cache_max_bytes,
            currsize=len(_image_payload_cache),
        )


def _image_payload_cache_current_bytes() -> int:
    with _image_payload_cache_lock:
        return _image_payload_cache_bytes


def _set_image_payload_cache_max_bytes(max_bytes: int) -> int:
    """设置图像 payload 缓存预算，返回旧预算，供测试/诊断使用。"""
    global _image_payload_cache_bytes
    global _image_payload_cache_max_bytes

    old_value = _image_payload_cache_max_bytes
    with _image_payload_cache_lock:
        _image_payload_cache_max_bytes = max(0, int(max_bytes))
        while _image_payload_cache_bytes > _image_payload_cache_max_bytes and _image_payload_cache:
            _, (_, evicted_payload) = _image_payload_cache.popitem(last=False)
            _image_payload_cache_bytes -= len(evicted_payload)
    return old_value


def _load_image_cached(image_path: str) -> Image.Image:
    """加载图像并复用缓存的 RGB payload。

    每次调用都返回新的 PIL Image，避免调用方的数据增强或多线程处理修改共享对象。
    cache_clear/cache_info 代理到底层 payload 缓存，保留测试和调试入口。
    """
    size, payload = _load_image_payload_cached(image_path)
    return Image.frombytes('RGB', size, payload)


_load_image_cached.cache_clear = _image_payload_cache_clear  # type: ignore[attr-defined]
_load_image_cached.cache_info = _image_payload_cache_info  # type: ignore[attr-defined]
_load_image_cached.cache_bytes = _image_payload_cache_current_bytes  # type: ignore[attr-defined]
_load_image_cached.set_cache_max_bytes = _set_image_payload_cache_max_bytes  # type: ignore[attr-defined]
