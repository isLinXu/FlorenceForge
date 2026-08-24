"""Unit tests for the GRPO trainer.

Focus on the batched rollout / log-prob paths introduced to replace the
serial (per-rollout, per-sample) implementation. A deterministic tiny mock
model/tokenizer is used so the tests run on CPU without any real VLM.

Key invariants verified:
* ``generate`` is called exactly ONCE per training step (batched), not
  ``group_size`` times.
* ``generate`` receives ``batch_size * group_size`` prompts with matching
  ``pixel_values`` batch dimension.
* ``forward`` (log-prob) is called on the full ``B*G`` batch, never on
  size-1 slices.
* Reward grouping / advantage math produces finite values and expected keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from florence_forge.training.grpo_trainer import GRPOTrainer

VOCAB = 40
PROMPT_LEN = 6


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def batch_decode(self, ids, skip_special_tokens=False):
        return [f"rollout-{i}" for i in range(ids.shape[0])]

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        n = len(texts)
        return {
            "input_ids": torch.randint(2, VOCAB, (n, 4)),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _MockModel(nn.Module):
    """Tiny LM that records the batch sizes it is called with."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 8)
        self.head = nn.Linear(8, VOCAB)

        class _Cfg:
            use_cache = False

        self.config = _Cfg()
        self.generate_calls = 0
        self.generate_batch_sizes = []
        self.forward_batch_sizes = []

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kw):
        self.forward_batch_sizes.append(input_ids.shape[0])
        if pixel_values is not None:
            assert pixel_values.shape[0] == input_ids.shape[0], (
                "pixel_values batch must match input_ids in batched log-prob"
            )
        return _Out(self.head(self.emb(input_ids)))

    @torch.no_grad()
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None,
                 max_new_tokens=4, **kw):
        self.generate_calls += 1
        b = input_ids.shape[0]
        self.generate_batch_sizes.append(b)
        if pixel_values is not None:
            assert pixel_values.shape[0] == b, (
                "generate must receive repeated pixel_values matching prompts"
            )
        new = torch.randint(2, VOCAB, (b, 4))
        return torch.cat([input_ids, new], dim=1)


def _make_trainer(group_size=3):
    model = _MockModel()
    ref = _MockModel()
    reward_fns = [lambda t, m: 0.5, lambda t, m: 0.2, lambda t, m: 0.9]
    trainer = GRPOTrainer(
        model, ref, _MockTokenizer(), reward_fns,
        group_size=group_size, device="cpu",
    )
    return trainer, model, ref


def test_rollouts_are_batched_single_generate_call():
    torch.manual_seed(0)
    group_size = 3
    trainer, model, _ = _make_trainer(group_size)
    batch_size = 2

    stats = trainer.train_step(
        pixel_values=torch.randn(batch_size, 3, 8, 8),
        input_ids=torch.randint(2, VOCAB, (batch_size, PROMPT_LEN)),
        attention_mask=torch.ones(batch_size, PROMPT_LEN, dtype=torch.long),
        metadata_list=[{"i": 0}, {"i": 1}],
        max_new_tokens=4,
    )

    # Exactly one batched generate call (NOT group_size serial calls).
    assert model.generate_calls == 1
    assert model.generate_batch_sizes == [batch_size * group_size]
    assert set(stats) == {"loss", "mean_reward", "max_reward", "min_reward", "kl_divergence"}
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_log_probs_computed_on_full_batch():
    torch.manual_seed(1)
    group_size = 4
    trainer, model, ref = _make_trainer(group_size)
    batch_size = 2
    expected = batch_size * group_size

    trainer.train_step(
        pixel_values=torch.randn(batch_size, 3, 8, 8),
        input_ids=torch.randint(2, VOCAB, (batch_size, PROMPT_LEN)),
        attention_mask=torch.ones(batch_size, PROMPT_LEN, dtype=torch.long),
        metadata_list=[{}, {}],
        max_new_tokens=4,
    )

    # Policy + reference each do a single batched forward over B*G rows,
    # never size-1 serial slices.
    assert model.forward_batch_sizes == [expected]
    assert ref.forward_batch_sizes == [expected]


def test_repeat_for_group_layout():
    # repeat_interleave keeps the G rollouts of prompt i contiguous.
    t = torch.tensor([[10], [20]])
    out = GRPOTrainer._repeat_for_group(t, group_size=3, batch_size=2)
    assert out.squeeze(-1).tolist() == [10, 10, 10, 20, 20, 20]

    # non-matching batch dim is passed through unchanged
    other = torch.zeros(5, 2)
    assert GRPOTrainer._repeat_for_group(other, 3, 2) is other
    # non-tensors pass through
    assert GRPOTrainer._repeat_for_group("x", 3, 2) == "x"


def test_reward_weight_normalization():
    _, model, ref = _make_trainer()
    trainer = GRPOTrainer(
        model, ref, _MockTokenizer(),
        [lambda t, m: 1.0, lambda t, m: 1.0],
        reward_weights=[2.0, 2.0], device="cpu",
    )
    assert abs(sum(trainer.reward_weights) - 1.0) < 1e-6
    assert trainer.reward_weights == [0.5, 0.5]


def test_unified_train_runs_multiple_epochs():
    torch.manual_seed(2)
    trainer, model, _ = _make_trainer(group_size=2)
    batch_size = 2

    class _OneBatchLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.randint(2, VOCAB, (batch_size, PROMPT_LEN)),
                "attention_mask": torch.ones(batch_size, PROMPT_LEN, dtype=torch.long),
                "pixel_values": torch.randn(batch_size, 3, 8, 8),
                "metadata": [{}, {}],
            }

        def __len__(self):
            return 1

    result = trainer.train(_OneBatchLoader(), num_epochs=2, max_new_tokens=4)
    assert result["total_epochs"] == 2
    assert set(result["epochs"]) == {"epoch_0", "epoch_1"}
    assert "train_loss" in result
    # 2 epochs x 1 batch = 2 batched generate calls
    assert model.generate_calls == 2


def test_default_tvp_reward_weights():
    _, model, ref = _make_trainer()
    trainer = GRPOTrainer(
        model, ref, _MockTokenizer(),
        [lambda t, m: 0.0, lambda t, m: 0.0, lambda t, m: 0.0],
        device="cpu",
    )
    # TVP paper ratio format:quality:accuracy = 0.1:0.2:0.7
    assert trainer.reward_weights == [0.1, 0.2, 0.7]
