"""FlorenceForge 数据批处理模块

提供专业的 CollateFn，支持动态 padding 与多任务批次组装。
"""

import torch
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Florence2Collator:
    """Florence-2 专用 Collator

    支持：
    - 对 input_ids / attention_mask / labels 进行动态 padding（pad 到批次内最大长度）
    - pixel_values 直接 stack（Florence-2 processor 已统一图像尺寸）
    - 保留 task_type、weight、metadata 等非张量字段为列表
    - 支持指令微调中的 labels masking（prompt 部分设为 -100）
    """

    def __init__(self, pad_token_id: int = 0, padding_side: str = "right"):
        """初始化 Collator

        Args:
            pad_token_id: 用于 padding 的 token id，默认 0
            padding_side: padding 方向，"right" 或 "left"
        """
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对一批样本进行批处理

        Args:
            batch: 样本字典列表，每个样本来自 Dataset.__getitem__

        Returns:
            批处理后的字典，所有张量字段增加 batch 维度
        """
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        if any(sample.get("_needs_encoding") for sample in batch):
            raise RuntimeError(
                "批次包含未编码样本：DataLoader worker 中的 processor/backend 不可用。"
                "请将 num_workers 设为 0，或先在主进程完成预编码缓存。"
            )

        # 分离张量字段与非张量字段
        non_tensor_keys = ["task_type", "weight", "metadata", "prompt", "answer"]

        collated: Dict[str, Any] = {}

        required_encoded_keys = ("input_ids", "pixel_values")
        for key in required_encoded_keys:
            present_count = sum(sample.get(key) is not None for sample in batch)
            if 0 < present_count < len(batch):
                raise ValueError(
                    f"批次字段 {key} 只出现在 {present_count}/{len(batch)} 个样本中，"
                    "无法安全组装批次。请检查缓存或数据编码流程。"
                )

        # ---- 1. 处理 input_ids（动态 padding） ----
        if all(sample.get("input_ids") is not None for sample in batch):
            input_ids_list = [sample["input_ids"] for sample in batch]
            collated["input_ids"] = self._pad_sequence(input_ids_list)

        # ---- 2. 处理 attention_mask（动态 padding） ----
        if all(sample.get("attention_mask") is not None for sample in batch):
            attention_mask_list = [sample["attention_mask"] for sample in batch]
            collated["attention_mask"] = self._pad_sequence(attention_mask_list, pad_value=0)

        if all(sample.get("prompt_input_ids") is not None for sample in batch):
            prompt_input_ids_list = [sample["prompt_input_ids"] for sample in batch]
            collated["prompt_input_ids"] = self._pad_sequence(prompt_input_ids_list)

        if all(sample.get("prompt_attention_mask") is not None for sample in batch):
            prompt_attention_mask_list = [sample["prompt_attention_mask"] for sample in batch]
            collated["prompt_attention_mask"] = self._pad_sequence(prompt_attention_mask_list, pad_value=0)

        # ---- 3. 处理 labels（动态 padding，pad_value=-100） ----
        if all(sample.get("labels") is not None for sample in batch):
            labels_list = [sample["labels"] for sample in batch]
            collated["labels"] = self._pad_sequence(labels_list, pad_value=-100)
        else:
            # 如果没有 labels，尝试用 input_ids 作为替代（推理场景）
            if "input_ids" in collated:
                collated["labels"] = collated["input_ids"].clone()

        # ---- 4. Handle loss_weights (phase-aware loss, pad_value=0) ----
        # If ANY sample has loss_weights, fill missing ones with uniform 1.0
        # so mixed agentic+native batches don't lose the signal.
        any_has_weights = any(sample.get("loss_weights") is not None for sample in batch)
        if any_has_weights:
            import torch as _torch
            weights_list = []
            for sample in batch:
                lw = sample.get("loss_weights")
                if lw is not None:
                    weights_list.append(lw)
                else:
                    # Fill with uniform weight for supervised tokens, 0 for padding
                    labels = sample.get("labels")
                    if labels is not None and hasattr(labels, "shape"):
                        seq_len = labels.shape[-1] if labels.dim() > 0 else len(labels)
                        # Set weight=1.0 for supervised tokens, 0.0 for ignored
                        ignore_idx = -100
                        lw_tensor = _torch.where(
                            labels != ignore_idx,
                            _torch.ones(seq_len, dtype=_torch.float32),
                            _torch.zeros(seq_len, dtype=_torch.float32),
                        )
                    else:
                        ids = sample.get("input_ids", [])
                        seq_len = len(ids) if hasattr(ids, "__len__") else 1
                        lw_tensor = _torch.ones(seq_len, dtype=_torch.float32)
                    weights_list.append(lw_tensor)
            collated["loss_weights"] = self._pad_sequence(weights_list, pad_value=0)

        if all(sample.get("reference_ids") is not None for sample in batch):
            reference_ids_list = [sample["reference_ids"] for sample in batch]
            collated["reference_ids"] = self._pad_sequence(reference_ids_list)

        # ---- 5. 处理可选序列张量（PaliGemma 等 decoder-only VLM 可能返回） ----
        for key in ("token_type_ids", "position_ids", "mm_token_type_ids"):
            if all(sample.get(key) is not None for sample in batch):
                tensor_list = [sample[key] for sample in batch]
                collated[key] = self._pad_sequence(tensor_list, pad_value=0)

        # ---- 6. pixel_values (direct stack) ----
        if all(sample.get("pixel_values") is not None for sample in batch):
            pixel_values_list = [sample["pixel_values"] for sample in batch]
            # Dataset __getitem__ 已确保 pixel_values 是张量，直接 stack
            collated["pixel_values"] = torch.stack(pixel_values_list, dim=0)

        # ---- 7. Collect non-tensor fields as lists ----
        for key in non_tensor_keys:
            values = [sample.get(key) for sample in batch]
            if any(v is not None for v in values):
                collated[key] = values

        # ---- 8. Task type shortcut ----
        if "task_type" in collated:
            task_types = collated["task_type"]
            collated["task_types"] = task_types
            if len(set(task_types)) == 1:
                collated["task_type"] = task_types[0]  # 单任务批次，用字符串
            else:
                collated["task_type"] = task_types[0]  # 兼容旧代码

        collated["is_empty"] = False
        return collated

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        pad_value: int = None
    ) -> torch.Tensor:
        """对一组序列进行动态 padding

        Args:
            sequences: 张量列表，每个张量形状为 (seq_len,)
            pad_value: padding 值，默认使用 self.pad_token_id

        Returns:
            padding 后的张量，形状为 (batch_size, max_seq_len)
        """
        if pad_value is None:
            pad_value = self.pad_token_id

        # 统一为 1D 张量
        tensors = []
        for seq in sequences:
            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq)
            if seq.dim() > 1:
                seq = seq.squeeze()
            tensors.append(seq)

        max_length = max(t.shape[0] for t in tensors)
        batch_size = len(tensors)

        # 创建 padding 后的张量；保留首个序列的 device，避免 MPS/CUDA 张量被拉回 CPU。
        target_device = tensors[0].device
        padded = torch.full(
            (batch_size, max_length),
            pad_value,
            dtype=tensors[0].dtype,
            device=target_device,
        )

        for i, seq in enumerate(tensors):
            if seq.device != target_device:
                seq = seq.to(target_device)
            length = seq.shape[0]
            if self.padding_side == "right":
                padded[i, :length] = seq
            else:
                padded[i, -length:] = seq

        return padded


# 向后兼容：保留旧的 collate_fn 函数签名
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """默认的 collate 函数入口

    Args:
        batch: 样本列表

    Returns:
        批处理结果
    """
    # 处理空批次
    if not batch:
        dummy = {
            "input_ids": torch.tensor([[0]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "labels": torch.full((1, 1), -100, dtype=torch.long),
            "task_type": "CAPTION",
            "is_empty": True,
        }
        return dummy

    collator = Florence2Collator()
    return collator(batch)
