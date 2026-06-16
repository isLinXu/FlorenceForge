"""On-Policy Distillation (OPD) trainer for Visual Primitive reasoning.

Distills expert models (e.g., ETwG and ETwP) into a single unified
student model. Uses forward KL divergence with temperature scaling
for stable offline distillation.

Loss: L_OPD = Σ w_i * D_KL(π_Ei || π_θ) + ce_coeff * L_CE

Adapted from Thinking-with-Visual-Primitives-pytorch and integrated
into the FlorenceForge training infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Default task-to-teacher routing
TASK_TO_TEACHER: Dict[str, int] = {
    "counting": 0,
    "spatial": 0,
    "grounding": 0,
    "od_vp": 0,
    "phrase_grounding_vp": 0,
    "maze": 1,
    "path": 1,
}


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Forward KL: D_KL(teacher || student) with temperature scaling.

    Only computed on assistant tokens (where labels != -100).
    """
    s_logits = student_logits[:, :-1, :] / temperature
    t_logits = teacher_logits[:, :-1, :] / temperature

    teacher_probs = F.softmax(t_logits, dim=-1)
    student_log_probs = F.log_softmax(s_logits, dim=-1)

    per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)

    labels_shifted = labels[:, 1:]
    mask = (labels_shifted != -100).float()
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

    return loss * (temperature ** 2)


class OPDTrainer:
    """On-Policy Distillation trainer.

    Distills multiple expert teachers into a single student model.
    Teachers are routed by task type, with fallback to weighted
    averaging for unknown tasks.
    """

    def __init__(
        self,
        student: nn.Module,
        teachers: List[nn.Module],
        teacher_weights: Optional[List[float]] = None,
        task_routing: Optional[Dict[str, int]] = None,
        temperature: float = 2.0,
        ce_coeff: float = 0.3,
        lr: float = 5e-7,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.03,
        device: str = "cuda",
    ):
        self.student = student
        self.teachers = teachers
        self.teacher_weights = teacher_weights or [1.0 / len(teachers)] * len(teachers)
        self.task_routing = task_routing or TASK_TO_TEACHER
        self.temperature = temperature
        self.ce_coeff = ce_coeff
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device

        # Freeze all teachers
        for t in self.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad = False

        self.optimizer = AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Single OPD training step.

        Args:
            batch: Dictionary with model inputs including 'labels',
                   'metadata', and any extra tensors.

        Returns:
            Dictionary with loss, kl_loss, ce_loss statistics.
        """
        model_inputs = {
            k: v.to(self.device) for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        # Student forward (with gradients)
        student_outputs = self.student(**model_inputs)
        student_logits = student_outputs.logits
        ce_loss = student_outputs.loss

        # Determine which teacher to use based on task type
        metadata_list = batch.get("metadata", [{}])
        task_type = metadata_list[0].get("task_type", "") if metadata_list else ""
        teacher_idx = self.task_routing.get(task_type, None)

        teacher_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}

        if teacher_idx is not None and teacher_idx < len(self.teachers):
            # Route to the expert teacher for this task
            with torch.no_grad():
                teacher_outputs = self.teachers[teacher_idx](**teacher_inputs)
                teacher_logits = teacher_outputs.logits.detach()
            kl_loss = compute_distill_loss(
                student_logits, teacher_logits,
                model_inputs["labels"], self.temperature
            )
            del teacher_outputs, teacher_logits
        else:
            # Unknown task: average all teachers
            kl_losses = []
            for teacher in self.teachers:
                with torch.no_grad():
                    teacher_outputs = teacher(**teacher_inputs)
                    t_logits = teacher_outputs.logits.detach()
                kl = compute_distill_loss(
                    student_logits, t_logits,
                    model_inputs["labels"], self.temperature
                )
                kl_losses.append(kl)
                del teacher_outputs, t_logits
            kl_loss = sum(
                w * kl for w, kl in zip(self.teacher_weights, kl_losses)
            )

        loss = self.ce_coeff * ce_loss + kl_loss
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "kl_loss": kl_loss.item(),
            "ce_loss": ce_loss.item(),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        save_dir: Optional[Path] = None,
        save_every: int = 1,
    ) -> Dict[str, float]:
        """Run one epoch of OPD training."""

        self.student.train()
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"OPD Epoch {epoch}")
        for step, batch in enumerate(pbar):
            step_stats = self.train_step(batch)

            if (step + 1) % self.gradient_accumulation_steps == 0 or \
               (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += step_stats["loss"]
            total_kl += step_stats["kl_loss"]
            total_ce += step_stats["ce_loss"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{step_stats['loss']:.3f}",
                "kl": f"{step_stats['kl_loss']:.3f}",
                "ce": f"{step_stats['ce_loss']:.3f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        avg_kl = total_kl / max(num_batches, 1)
        avg_ce = total_ce / max(num_batches, 1)

        logger.info(
            f"OPD Epoch {epoch} avg loss: {avg_loss:.4f} "
            f"(kl={avg_kl:.4f}, ce={avg_ce:.4f})"
        )

        if save_dir and (epoch + 1) % save_every == 0:
            epoch_dir = save_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(self.student, 'save_pretrained'):
                self.student.save_pretrained(str(epoch_dir))
            else:
                torch.save(self.student.state_dict(), epoch_dir / "model.pt")

        return {"loss": avg_loss, "kl_loss": avg_kl, "ce_loss": avg_ce}
