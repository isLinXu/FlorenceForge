"""CLI 服务命令处理。

处理 ``serve`` 子命令：启动 FastAPI 推理服务。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def run_serve_task(args: "Namespace") -> bool:
    """运行模型推理服务器"""
    try:
        from florence_forge.deployment.server import create_server
    except ImportError as e:
        logger.error(f"❌ 无法导入服务模块: {e}")
        logger.error("请确保已安装 FastAPI: pip install fastapi uvicorn")
        return False

    model_path = args.model
    host = args.host
    port = args.port

    logger.info("🚀 启动模型推理服务器")
    logger.info(f"   模型路径: {model_path}")
    logger.info(f"   监听地址: {host}:{port}")
    logger.info(f"   设备: {args.device}")
    logger.info(f"   后端: {getattr(args, 'backend', 'native')}")
    model_revision = getattr(args, "model_revision", None)
    if model_revision:
        logger.info(f"   模型 revision: {model_revision}")

    server = create_server(
        model_path=model_path,
        host=host,
        port=port,
        device=args.device,
        backend=getattr(args, "backend", "native"),
        batch_size=getattr(args, "batch_size", 1),
        use_amp=getattr(args, "use_amp", False),
        model_revision=model_revision,
    )
    server.run(host=host, port=port)
    return True
