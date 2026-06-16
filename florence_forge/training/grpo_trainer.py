"""GRPO (Group Relative Policy Optimization) trainer for Visual Primitive reasoning.

Reference: DeepSeek-V4 paper & DeepSeekMath GRPO.

Key ideas:
  - For each prompt, sample G rollouts from the current policy.
  - Compute reward for each rollout.
  - Advantage = (reward - mean(reward within group)) / std(reward within group).
  - Update policy to maximize log_prob * advantage.
  - Add KL penalty against a reference (SFT) model.

Adapted from Thinking-with-Visual-Primitives-pytorch and integrated
into the FlorenceForge training infrastructure.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO trainer for visual primitive reasoning.

    Integrates with FlorenceForge's model interface and reward model
    system. Designed to work with any VLM backend that exposes a
    standard ``forward()`` returning logits and ``generate()`` for
    rollout sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        reward_fns: List[Callable[[str, Dict], float]],
        group_size: int = 4,
        kl_coeff: float = 0.04,
        clip_eps: float = 0.2,
        lr: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        reward_weights: Optional[List[float]] = None,
        normalize_rewards: bool = True,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.device = device
        self.normalize_rewards = normalize_rewards

        # Reward weights: default TVP paper ratio format:quality:accuracy = 0.1:0.2:0.7
        if reward_weights is not None:
            total = sum(reward_weights)
            self.reward_weights = [w / max(total, 1e-8) for w in reward_weights]
        elif len(reward_fns) == 3:
            self.reward_weights = [0.1, 0.2, 0.7]
        else:
            self.reward_weights = [1.0 / max(len(reward_fns), 1)] * len(reward_fns)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_rollouts(
        self,
        pixel_values: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[str]:
        """Generate group_size rollouts for the given prompts."""

        all_texts: List[str] = []
        was_training = self.model.training
        original_use_cache = getattr(self.model.config, 'use_cache', None) if hasattr(self.model, 'config') else None

        self.model.eval()
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True

        try:
            for _ in range(self.group_size):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                new_tokens = outputs[:, input_ids.shape[1]:]
                texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
                all_texts.extend(texts)
        finally:
            if was_training:
                self.model.train()
            if original_use_cache is not None and hasattr(self.model, 'config'):
                self.model.config.use_cache = original_use_cache

        return all_texts

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_rewards(
        self,
        texts: List[str],
        metadata_list: List[Dict],
    ) -> torch.Tensor:
        """Compute weighted rewards for each rollout.

        Uses self.reward_weights (default: TVP ratio 0.1/0.2/0.7 for
        format/quality/accuracy) instead of simple averaging.
        Optionally applies per-batch z-score normalization.
        """
        rewards = []
        for text, meta in zip(texts, metadata_list):
            r = sum(
                w * fn(text, meta)
                for w, fn in zip(self.reward_weights, self.reward_fns)
            )
            rewards.append(r)
        t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if self.normalize_rewards and len(t) > 1:
            mean, std = t.mean(), t.std()
            t = (t - mean) / (std + 1e-8)

        return t

    # ------------------------------------------------------------------
    # Log probability computation
    # ------------------------------------------------------------------

    def _compute_log_probs(
        self,
        model: nn.Module,
        pixel_values: Optional[torch.Tensor],
        full_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute per-sequence log probabilities for response tokens."""

        outputs = model(
            pixel_values=pixel_values,
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits[:, :-1, :]
        targets = full_input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        mask = attention_mask[:, 1:].float()
        if prompt_len > 0:
            mask[:, :prompt_len - 1] = 0.0
        token_log_probs = token_log_probs * mask
        seq_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return seq_log_probs

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        pixel_values: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata_list: List[Dict],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> Dict[str, float]:
        """Single GRPO training step."""

        batch_size = input_ids.shape[0]

        # 1. Generate rollouts
        rollouts = self._generate_rollouts(
            pixel_values, input_ids, attention_mask, max_new_tokens, **kwargs
        )

        # 2. Compute rewards
        metadata_repeated = []
        for meta in metadata_list:
            metadata_repeated.extend([meta] * self.group_size)
        rewards = self._compute_rewards(rollouts, metadata_repeated)

        # 3. Group-relative advantages
        rewards_grouped = rewards.view(batch_size, self.group_size)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1)

        # 4. Re-tokenize rollouts
        rollout_tokens = self.tokenizer(
            rollouts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens,
        )
        rollout_ids = rollout_tokens["input_ids"].to(self.device)
        rollout_mask = rollout_tokens["attention_mask"].to(self.device)

        prompt_ids_repeated = input_ids.repeat_interleave(self.group_size, dim=0)
        prompt_mask_repeated = attention_mask.repeat_interleave(self.group_size, dim=0)

        full_ids = torch.cat([prompt_ids_repeated, rollout_ids], dim=1)
        full_mask = torch.cat([prompt_mask_repeated, rollout_mask], dim=1)

        prompt_len = input_ids.shape[1]

        # 5. Compute log probs for current and reference policy
        seq_log_probs_list = []
        ref_seq_log_probs_list = []
        old_seq_log_probs_list = []

        for i in range(full_ids.shape[0]):
            lp = self._compute_log_probs(
                self.model, pixel_values, full_ids[i:i+1], full_mask[i:i+1],
                prompt_len=prompt_len, **kwargs
            )
            seq_log_probs_list.append(lp)
            with torch.no_grad():
                ref_lp = self._compute_log_probs(
                    self.ref_model, pixel_values, full_ids[i:i+1], full_mask[i:i+1],
                    prompt_len=prompt_len, **kwargs
                )
                ref_seq_log_probs_list.append(ref_lp)
                old_seq_log_probs_list.append(lp.detach())

        seq_log_probs = torch.cat(seq_log_probs_list)
        ref_seq_log_probs = torch.cat(ref_seq_log_probs_list)
        old_seq_log_probs = torch.cat(old_seq_log_probs_list)

        # 6. KL penalty
        kl_div = seq_log_probs - ref_seq_log_probs

        # 7. Clipped surrogate objective (PPO-style, per GRPO)
        log_ratio = seq_log_probs - old_seq_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        loss = -torch.min(surr1, surr2).mean() + self.kl_coeff * kl_div.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "kl_divergence": kl_div.mean().item(),
        }

    # ------------------------------------------------------------------
    # Epoch training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_new_tokens: int = 512,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Run one epoch of GRPO training."""

        self.model.train()
        stats: Dict[str, float] = {"loss": 0.0, "mean_reward": 0.0, "kl_divergence": 0.0}
        count = 0

        pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            metadata_list = batch.get("metadata", [{}] * input_ids.shape[0])

            extra_kwargs = {
                k: v.to(self.device) for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k not in
                {"input_ids", "attention_mask", "pixel_values", "labels", "metadata"}
            }

            step_stats = self.train_step(
                pixel_values, input_ids, attention_mask,
                metadata_list, max_new_tokens, **extra_kwargs,
            )
            for k in stats:
                if k in step_stats:
                    stats[k] += step_stats[k]
            count += 1
            pbar.set_postfix({k: f"{v / count:.4f}" for k, v in stats.items()})

        return {k: v / max(count, 1) for k, v in stats.items()}
