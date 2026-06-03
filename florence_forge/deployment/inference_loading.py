"""推理引擎模型与设备加载（从 ``inference.py`` 抽出）。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils.torch_serialization import safe_torch_load

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            mps_backend = vars(torch.backends).get("mps")
            mps_available = (
                mps_backend is not None
                and callable(getattr(mps_backend, "is_available", None))
                and mps_backend.is_available()
            )
            device = "mps" if mps_available else "cpu"
    return torch.device(device)


class InferenceModelLoader:
    """封装推理模型的设备解析与安全加载。"""

    def __init__(
        self,
        *,
        device: torch.device,
        compile_model: bool = False,
        allow_unsafe_torch_load: bool = False,
        model_revision: Optional[str] = None,
    ) -> None:
        self.device = device
        self.compile_model = compile_model
        self.allow_unsafe_torch_load = allow_unsafe_torch_load
        self.model_revision = model_revision

    def build_model_config_kwargs(
        self,
        model_name: str,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        config_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "device": str(self.device),
            "use_lora": False,
        }
        effective_revision = self.model_revision or revision
        if effective_revision:
            config_kwargs["revision"] = effective_revision
        return config_kwargs

    def load_torch_file(self, model_identifier: str) -> nn.Module:
        allow_unsafe = self.allow_unsafe_torch_load or (
            os.environ.get("FLORENCE_FORGE_ALLOW_UNSAFE_TORCH_LOAD") == "1"
        )
        try:
            loaded = safe_torch_load(
                model_identifier,
                map_location=self.device,
                context="Inference model",
            )
        except Exception as safe_exc:
            if not allow_unsafe:
                raise ValueError(
                    "安全加载本地 Torch 文件失败。FlorenceForge 默认使用 "
                    "torch.load(weights_only=True)；如果该文件是可信来源的整模型 "
                    "pickle，请传入 allow_unsafe_torch_load=True 或设置 "
                    "FLORENCE_FORGE_ALLOW_UNSAFE_TORCH_LOAD=1。"
                ) from safe_exc
            logger.warning(
                "正在使用 weights_only=False 加载本地 Torch 文件。"
                "这会执行 pickle 反序列化，只应对可信文件启用。"
            )
            try:
                loaded = torch.load(
                    model_identifier,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                loaded = torch.load(model_identifier, map_location=self.device)

        if not isinstance(loaded, nn.Module):
            raise TypeError(
                f"本地 Torch 文件加载结果是 {type(loaded).__name__}，不是 nn.Module。"
                "如果这是 state_dict，请先构建模型结构并传入模型实例。"
            )
        return loaded

    def load(self, model: Union[nn.Module, str, Path]) -> nn.Module:
        if isinstance(model, nn.Module):
            loaded_model = model
        else:
            model_identifier = str(model)
            model_path = Path(model_identifier)

            if model_path.suffix in [".pt", ".pth"] and model_path.is_file():
                logger.info("尝试加载本地Torch模型文件: %s", model_identifier)
                try:
                    loaded_model = torch.jit.load(model_identifier, map_location=self.device)
                    logger.info("TorchScript模型加载成功")
                except Exception:
                    logger.info("TorchScript加载失败，尝试安全加载PyTorch模型文件")
                    loaded_model = self.load_torch_file(model_identifier)
                    logger.info("PyTorch模型文件加载成功")
            else:
                try:
                    from ..core.model import Florence2MultiTaskModel, ModelConfig

                    if model_path.is_dir() and (model_path / "adapter_config.json").exists():
                        logger.info("检测到本地LoRA模型: %s", model_identifier)
                        with open(model_path / "adapter_config.json", "r", encoding="utf-8") as handle:
                            adapter_config = json.load(handle)
                        base_model_name = adapter_config.get(
                            "base_model_name_or_path", "microsoft/Florence-2-base"
                        )
                        config = ModelConfig(
                            **self.build_model_config_kwargs(
                                base_model_name,
                                revision=adapter_config.get("revision"),
                            )
                        )
                        loaded_model = Florence2MultiTaskModel.load_pretrained(
                            model_identifier,
                            config=config,
                            is_peft_model=True,
                        )
                        logger.info("LoRA模型加载成功")
                    else:
                        logger.info("尝试加载Hugging Face模型: %s", model_identifier)
                        config = ModelConfig(**self.build_model_config_kwargs(model_identifier))
                        loaded_model = Florence2MultiTaskModel(config)
                        loaded_model.load()
                        logger.info("Hugging Face模型加载成功")
                except ImportError as exc:
                    logger.error("无法导入核心模型组件: %s", exc)
                    raise ValueError(
                        "加载Hugging Face模型需要 `florence_forge.core.model` 支持。"
                    ) from exc
                except Exception as exc:
                    logger.error("加载Hugging Face模型 '%s' 失败: %s", model_identifier, exc)
                    raise ValueError(
                        "无法加载模型。请检查路径或模型ID是否正确，以及是否需要网络连接。"
                    ) from exc

        if not hasattr(loaded_model, "eval"):
            raise TypeError(
                f"加载结果 {type(loaded_model).__name__} 不支持 eval()，无法用于推理"
            )

        if hasattr(loaded_model, "to") and "Florence2MultiTaskModel" in str(
            loaded_model.__class__
        ):
            loaded_model = loaded_model.to(self.device)
        elif hasattr(loaded_model, "to"):
            loaded_model = loaded_model.to(self.device)
        else:
            logger.warning("模型 %s 不支持.to()方法，跳过设备移动", type(loaded_model))

        if self.compile_model and hasattr(torch, "compile"):
            try:
                loaded_model = torch.compile(loaded_model)
                logger.info("模型编译完成")
            except Exception as exc:
                logger.warning("模型编译失败: %s", exc)

        return loaded_model
