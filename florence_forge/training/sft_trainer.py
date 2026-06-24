"""Supervised Fine-Tuning (SFT) trainer for Visual Primitive reasoning.

Stage 1 of the TVP three-stage training pipeline.
Teaches the base VLM to produce valid VP-formatted chain-of-thought
outputs via standard cross-entropy on teacher-generated demonstrations.

Supports:
  - LoRA / full fine-tuning
  - Gradient checkpointing
  - Mixed precision (fp16 / bf16)
  - Cosine LR schedule with warmup
  - Periodic checkpoint saving compatible with TVPPipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.torch_serialization import safe_torch_load

logger = logging.getLogger(__name__)

try:
    from transformers import get_cosine_schedule_with_warmup
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for SFT trainer.

    Mirrors the YAML keys expected by TVPPipeline._run_sft().
    """

    # Optimizer
    lr: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4

    # Precision
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "fp16" or "bf16"

    # Checkpointing
    save_dir: str = "outputs/tvp/sft"
    save_every_epochs: int = 1
    save_best: bool = True

    # Device
    device: str = "cuda"

    # Eval
    eval_every_steps: int = 500

    # Callbacks
    callbacks: List[Callable] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SFT Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """Supervised Fine-Tuning trainer for VP-formatted VLM outputs.

    Wraps a HuggingFace-compatible model and handles the full SFT loop
    with gradient accumulation, mixed precision, and cosine scheduling.

    Example::

        trainer = SFTTrainer(model, tokenizer, train_loader, config=SFTConfig())
        results = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[SFTConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_dataloader
        self.eval_loader = eval_dataloader
        self.config = config or SFTConfig()

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        # Scaler for AMP
        amp_dtype = (
            torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        )
        self._amp_dtype = amp_dtype
        self._use_amp = self.config.use_amp and torch.cuda.is_available()
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if self._use_amp and self.config.amp_dtype == "fp16"
            else None
        )

        # Scheduler (will be initialized in train())
        self.scheduler = None

        # Tracking
        self.global_step = 0
        self.best_eval_loss = float("inf")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SFTTrainer":
        """Build trainer from a raw config dict (as loaded from YAML).

        The config must include 'model', 'tokenizer', and 'dataloader'.
        All other keys map to :class:`SFTConfig` fields.
        """
        model = cfg["model"]
        tokenizer = cfg["tokenizer"]
        train_loader = cfg["dataloader"]
        eval_loader = cfg.get("eval_dataloader")

        sft_cfg = SFTConfig(
            lr=cfg.get("lr", 2e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
            num_epochs=cfg.get("epochs", cfg.get("num_epochs", 3)),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            use_amp=cfg.get("use_amp", True),
            amp_dtype=cfg.get("amp_dtype", "bf16"),
            save_dir=cfg.get("save_dir", "outputs/tvp/sft"),
            save_every_epochs=cfg.get("save_every_epochs", 1),
            save_best=cfg.get("save_best", True),
            device=cfg.get("device", "cuda"),
            eval_every_steps=cfg.get("eval_every_steps", 500),
        )

        return cls(model, tokenizer, train_loader, eval_loader, sft_cfg)

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full SFT training.

        Returns:
            Dictionary with final training statistics:
            ``{"train_loss", "eval_loss", "best_eval_loss", "total_steps"}``.
        """
        cfg = self.config
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_steps = (
            len(self.train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
        )

        # Build cosine schedule
        if _HAS_TRANSFORMERS:
            num_warmup = max(1, int(total_steps * cfg.warmup_ratio))
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=total_steps,
            )

        logger.info(
            "Starting SFT: epochs=%d, steps=%d, device=%s, amp=%s",
            cfg.num_epochs, total_steps, self.device, cfg.use_amp,
        )

        all_train_losses: List[float] = []
        final_eval_loss = float("inf")

        for epoch in range(cfg.num_epochs):
            train_loss = self._train_epoch(epoch)
            all_train_losses.append(train_loss)
            logger.info("SFT Epoch %d train_loss=%.4f", epoch, train_loss)

            if self.eval_loader is not None:
                eval_loss = self._eval_epoch(epoch)
                final_eval_loss = eval_loss
                logger.info("SFT Epoch %d eval_loss=%.4f", epoch, eval_loss)

                if cfg.save_best and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(save_dir / "best", epoch)

            if (epoch + 1) % cfg.save_every_epochs == 0:
                self._save_checkpoint(save_dir / f"epoch_{epoch}", epoch)

        # Always save final
        self._save_checkpoint(save_dir / "final", cfg.num_epochs - 1)

        return {
            "train_loss": sum(all_train_losses) / max(len(all_train_losses), 1),
            "eval_loss": final_eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Run one epoch, return average training loss."""
        self.model.train()
        total_loss = 0.0
        count = 0
        accum = self.config.gradient_accumulation_steps

        pbar = tqdm(
            self.train_loader,
            desc=f"SFT Epoch {epoch}",
            dynamic_ncols=True,
        )
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            loss = self._forward_step(batch)
            loss_scaled = loss / accum

            if self._scaler is not None:
                self._scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            total_loss += loss.item()
            count += 1

            # Gradient accumulation update
            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                if self._scaler is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            pbar.set_postfix({"loss": f"{total_loss / count:.4f}"})

            # Periodic evaluation
            if (
                self.eval_loader is not None
                and self.global_step > 0
                and self.global_step % self.config.eval_every_steps == 0
            ):
                eval_loss = self._eval_epoch(epoch)
                logger.info(
                    "Step %d eval_loss=%.4f", self.global_step, eval_loss
                )
                self.model.train()

        return total_loss / max(count, 1)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> float:
        """Run evaluation, return average loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch in tqdm(self.eval_loader, desc=f"SFT Eval {epoch}", leave=False):
            loss = self._forward_step(batch)
            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    def _forward_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Move batch to device, run forward, return loss."""
        model_inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        if self._use_amp:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
            ):
                outputs = self.model(**model_inputs)
        else:
            outputs = self.model(**model_inputs)

        # HuggingFace models return CausalLMOutputWithPast with .loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        # Fallback: compute CE from logits + labels
        logits = outputs.logits
        labels = model_inputs.get("labels")
        if labels is None:
            raise ValueError("Batch must contain 'labels' for SFT training")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        import torch.nn.functional as F
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, directory: Path, epoch: int) -> None:
        """Save model and tokenizer to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(directory))
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(str(directory))
        else:
            torch.save(self.model.state_dict(), directory / "model.pt")

        # Save optimizer state for potential resume
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_eval_loss": self.best_eval_loss,
            },
            directory / "trainer_state.pt",
        )
        logger.info("SFT checkpoint saved to %s", directory)

    def load_checkpoint(self, directory: str) -> None:
        """Resume training from *directory*."""
        state_path = Path(directory) / "trainer_state.pt"
        if state_path.exists():
            state = safe_torch_load(
                state_path,
                map_location=self.device,
                context="SFT trainer state",
            )
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.global_step = state.get("global_step", 0)
            self.best_eval_loss = state.get("best_eval_loss", float("inf"))
            logger.info(
                "Resumed from step %d, best_eval_loss=%.4f",
                self.global_step, self.best_eval_loss,
            )
        else:
            logger.warning("No trainer_state.pt found in %s", directory)
