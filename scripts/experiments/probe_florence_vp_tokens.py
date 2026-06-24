#!/usr/bin/env python3
"""Probe Florence-VP tokenizer, labels, and generation scores.

The probe is intentionally read-only: it loads a Florence checkpoint and an
existing VP manifest, inspects whether VP marker tokens are supervised in the
training labels, and checks whether generation logits give those markers any
meaningful probability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from florence_forge.core.config import DataConfig, ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.visual_primitives import get_visual_primitive_tokens
from florence_forge.data.dataset import MultiTaskDataset


DEFAULT_MODEL_PATH = Path.home() / "Downloads" / "Florence2_det_base_ovd-v3-1751283651704-model"
DEFAULT_MANIFEST = Path(".codex_reports/florence_vp_od_only_64step/training/vp_real_data_manifest.json")
LOC_TOKEN_PATTERN = re.compile(r"<loc_\d+>")


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(args: argparse.Namespace) -> Florence2MultiTaskModel:
    from peft import PeftModel

    config = ModelConfig(
        model_name=str(Path(args.model_path).expanduser()),
        backend_name="florence-2",
        device=_choose_device(args.device),
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        use_lora=False,
        attn_implementation="eager",
        enable_visual_primitives=True,
    )
    model = Florence2MultiTaskModel(config).load()
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir).expanduser()
        model.model = PeftModel.from_pretrained(model.model, str(adapter_dir), is_trainable=False)
        model.is_peft_model = True
        model._backend.is_peft_model = True
    _patch_missing_generation_mixin(model.model)
    model.eval()
    return model


def _patch_missing_generation_mixin(model: Any) -> None:
    try:
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
    except Exception:
        return

    candidates = []
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner = getattr(getattr(base_model, "model", None), "language_model", None)
        if inner is not None:
            candidates.append(inner)
    inner = getattr(model, "language_model", None)
    if inner is not None:
        candidates.append(inner)

    for candidate in candidates:
        if hasattr(candidate, "generate"):
            continue
        patched_cls = type(
            f"Patched{candidate.__class__.__name__}",
            (candidate.__class__, GenerationMixin),
            {},
        )
        try:
            candidate.__class__ = patched_cls
        except TypeError:
            candidate.generate = GenerationMixin.generate.__get__(candidate, candidate.__class__)
        if getattr(candidate, "generation_config", None) is None and getattr(candidate, "config", None) is not None:
            try:
                candidate.generation_config = GenerationConfig.from_model_config(candidate.config)
            except Exception:
                pass


def _resolve_data_path(
    manifest: Mapping[str, Any],
    split: str,
    data_key: Optional[str] = None,
) -> Path:
    if data_key:
        keys = [key.strip() for key in str(data_key).split(",") if key.strip()]
        for key in keys:
            data_path = manifest.get(key)
            if data_path:
                return Path(str(data_path)).expanduser()
        raise KeyError(f"Manifest does not contain any data key from {keys}")

    key = "val_od_path" if split == "val" else "train_od_path"
    data_path = manifest.get(key)
    if not data_path:
        raise KeyError(f"Manifest does not contain {key}")
    return Path(str(data_path)).expanduser()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize_ids(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids = encoded.get("input_ids", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _convert_token_to_id(tokenizer: Any, token: str) -> Optional[int]:
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    token_id = None
    if token in vocab:
        token_id = int(vocab[token])
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if token_id is None and callable(convert):
        converted = convert(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if converted is not None and converted != unk_id:
            token_id = int(converted)
    ids = _tokenize_ids(tokenizer, token)
    if len(ids) == 1 and (token_id is None or token_id == ids[0]):
        return int(ids[0])
    return token_id


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            return str(convert(int(token_id)))
    return str(token_id)


def build_token_inventory(
    tokenizer: Any,
    model: Any = None,
    marker_style: str = "special",
) -> Dict[str, Any]:
    """Summarize VP token ids, tokenization length, and embedding resize state."""

    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    additional_specials = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    marker_rows = []
    for token in tuple(get_visual_primitive_tokens(marker_style).values()):
        tokenized_ids = _tokenize_ids(tokenizer, token)
        token_id = _convert_token_to_id(tokenizer, token)
        marker_rows.append({
            "token": token,
            "token_id": token_id,
            "in_vocab": token in vocab,
            "tokenized_ids": tokenized_ids,
            "tokenized_len": len(tokenized_ids),
            "single_token": token_id is not None and len(tokenized_ids) == 1,
            "additional_special": token in additional_specials,
        })

    loc_probe_tokens = ["<loc_0>", "<loc_1>", "<loc_999>"]
    loc_rows = []
    for token in loc_probe_tokens:
        tokenized_ids = _tokenize_ids(tokenizer, token)
        loc_rows.append({
            "token": token,
            "token_id": _convert_token_to_id(tokenizer, token),
            "in_vocab": token in vocab,
            "tokenized_ids": tokenized_ids,
            "tokenized_len": len(tokenized_ids),
            "single_token": len(tokenized_ids) == 1,
        })

    model_vocab_size = None
    if model is not None:
        model_vocab_size = _resolve_model_vocab_size(model)

    return {
        "marker_style": marker_style,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "model_vocab_size": model_vocab_size,
        "embedding_resized": (
            bool(model_vocab_size == len(tokenizer))
            if model_vocab_size is not None else None
        ),
        "vp_tokens": marker_rows,
        "loc_tokens": loc_rows,
        "all_vp_tokens_single": all(row["single_token"] for row in marker_rows),
    }


def _resolve_model_vocab_size(model: Any) -> Optional[int]:
    candidates = [model]
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        candidates.append(inner_model)
    backend = getattr(model, "_backend", None)
    if backend is not None and getattr(backend, "_model", None) is not None:
        candidates.append(backend._model)

    for candidate in candidates:
        get_input_embeddings = getattr(candidate, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embeddings = get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None and getattr(weight, "shape", None) is not None:
                return int(weight.shape[0])
    return None


def summarize_label_tokens(
    labels: Any,
    tokenizer: Any,
    vp_token_ids: Mapping[str, Optional[int]],
    vp_tokenizations: Optional[Mapping[str, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Count supervised VP and loc tokens in a label tensor."""

    import torch

    if isinstance(labels, torch.Tensor):
        flat = labels.detach().cpu().view(-1).tolist()
    else:
        flat = list(labels)
    supervised = [int(token_id) for token_id in flat if int(token_id) != -100]
    counts = Counter(supervised)
    id_to_vp_token = {
        int(token_id): token
        for token, token_id in vp_token_ids.items()
        if token_id is not None
    }
    vp_counts = {
        token: int(counts.get(int(token_id), 0)) if token_id is not None else 0
        for token, token_id in vp_token_ids.items()
    }
    vp_sequence_counts = {
        token: _count_any_subsequence(supervised, _normalize_tokenization_patterns(token_ids))
        for token, token_ids in (vp_tokenizations or {}).items()
        if token_ids
    }

    loc_count = 0
    decoded_counts = Counter()
    for token_id, count in counts.items():
        token_text = _decode_token(tokenizer, token_id)
        decoded_counts[token_text] += int(count)
        if LOC_TOKEN_PATTERN.fullmatch(token_text):
            loc_count += int(count)

    first_tokens = [
        {
            "token_id": token_id,
            "token": _decode_token(tokenizer, token_id),
            "is_vp_marker": token_id in id_to_vp_token,
        }
        for token_id in supervised[:12]
    ]
    return {
        "supervised_token_count": len(supervised),
        "unique_supervised_token_count": len(counts),
        "vp_marker_token_count": int(sum(vp_counts.values())),
        "vp_marker_token_counts": vp_counts,
        "vp_marker_sequence_count": int(sum(vp_sequence_counts.values())),
        "vp_marker_sequence_counts": vp_sequence_counts,
        "loc_token_count": int(loc_count),
        "first_supervised_tokens": first_tokens,
        "top_supervised_tokens": [
            {"token": token, "count": int(count)}
            for token, count in decoded_counts.most_common(12)
        ],
    }


def _count_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not values or not pattern or len(pattern) > len(values):
        return 0
    count = 0
    width = len(pattern)
    for start in range(0, len(values) - width + 1):
        if list(values[start:start + width]) == list(pattern):
            count += 1
    return count


def _count_any_subsequence(values: Sequence[int], patterns: Sequence[Sequence[int]]) -> int:
    return sum(_count_subsequence(values, pattern) for pattern in patterns)


def _normalize_tokenization_patterns(value: Any) -> List[List[int]]:
    if not value:
        return []
    first = value[0]
    if isinstance(first, int):
        return [[int(item) for item in value]]
    return [[int(item) for item in pattern] for pattern in value if pattern]


def rank_tokens_from_scores(
    score: Any,
    tokenizer: Any,
    token_ids: Mapping[str, Optional[int]],
    *,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Rank selected token ids against a single-step generation score vector."""

    import torch

    if score.dim() == 2:
        score = score[0]
    score = score.detach().float().cpu()
    probs = torch.softmax(score, dim=-1)
    top_values, top_indices = torch.topk(score, k=min(top_k, int(score.numel())))
    selected = {}
    for token, token_id in token_ids.items():
        if token_id is None or int(token_id) >= int(score.numel()):
            selected[token] = {"token_id": token_id, "rank": None, "probability": None, "logit": None}
            continue
        token_id = int(token_id)
        logit = float(score[token_id].item())
        probability = float(probs[token_id].item())
        rank = int(torch.sum(score > score[token_id]).item()) + 1 if math.isfinite(logit) else None
        selected[token] = {
            "token_id": token_id,
            "rank": rank,
            "probability": probability,
            "logit": logit,
        }
    return {
        "top_tokens": [
            {
                "token_id": int(token_id),
                "token": _decode_token(tokenizer, int(token_id)),
                "logit": float(logit),
                "probability": float(probs[int(token_id)].item()),
            }
            for logit, token_id in zip(top_values.tolist(), top_indices.tolist())
        ],
        "selected_tokens": selected,
    }


def _build_probe_token_ids(marker_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    token_ids: Dict[str, Optional[int]] = {}
    for row in marker_rows:
        token = str(row.get("token"))
        token_id = row.get("token_id")
        tokenized_ids = [int(value) for value in row.get("tokenized_ids", [])]
        if token_id is not None and len(tokenized_ids) == 1:
            token_ids[token] = int(token_id)
        elif tokenized_ids:
            token_ids[f"{token}[0]"] = int(tokenized_ids[0])
        else:
            token_ids[token] = None
    return token_ids


def _probe_key_for_marker(marker_rows: Sequence[Mapping[str, Any]], marker: str) -> str:
    for row in marker_rows:
        if str(row.get("token")) != marker:
            continue
        tokenized_ids = row.get("tokenized_ids", [])
        if row.get("token_id") is not None and len(tokenized_ids) == 1:
            return marker
        if tokenized_ids:
            return f"{marker}[0]"
    return marker


def _build_dataset(
    model: Florence2MultiTaskModel,
    data_path: Path,
    task_type: str,
) -> MultiTaskDataset:
    return MultiTaskDataset(
        data_configs=[{"task_type": task_type, "data_path": str(data_path)}],
        image_base_path="",
        config=DataConfig(use_cache=False, num_workers=0, batch_size=1),
        processor=model.processor,
        backend=model._backend,
    )


def _probe_labels(
    *,
    model: Florence2MultiTaskModel,
    dataset: MultiTaskDataset,
    tokenizer: Any,
    vp_token_ids: Mapping[str, Optional[int]],
    vp_tokenizations: Mapping[str, Sequence[int]],
    max_samples: int,
) -> Dict[str, Any]:
    sample_summaries = []
    aggregate_counter = Counter()
    aggregate_sequence_counter = Counter()
    total_supervised = 0
    total_loc = 0
    for idx in range(min(max_samples, len(dataset))):
        item = dataset[idx]
        summary = summarize_label_tokens(
            item["labels"],
            tokenizer,
            vp_token_ids,
            vp_tokenizations=vp_tokenizations,
        )
        sample_summaries.append({
            "index": idx,
            "task_type": item.get("task_type"),
            "answer": item.get("answer"),
            **summary,
        })
        total_supervised += int(summary["supervised_token_count"])
        total_loc += int(summary["loc_token_count"])
        for token, count in summary["vp_marker_token_counts"].items():
            aggregate_counter[token] += int(count)
        for token, count in summary["vp_marker_sequence_counts"].items():
            aggregate_sequence_counter[token] += int(count)

    return {
        "num_samples": len(sample_summaries),
        "total_supervised_token_count": total_supervised,
        "total_vp_marker_token_count": int(sum(aggregate_counter.values())),
        "total_vp_marker_token_counts": dict(aggregate_counter),
        "total_vp_marker_sequence_count": int(sum(aggregate_sequence_counter.values())),
        "total_vp_marker_sequence_counts": dict(aggregate_sequence_counter),
        "total_loc_token_count": int(total_loc),
        "samples": sample_summaries,
    }


def _probe_generation(
    *,
    model: Florence2MultiTaskModel,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    vp_token_ids: Mapping[str, Optional[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import torch

    records = []
    for idx, row in enumerate(rows[: args.max_samples]):
        image = Image.open(Path(str(row["image"])).expanduser()).convert("RGB")
        task_prompt = str(row.get("prefix") or model._backend.get_task_prompt(args.task_type))
        raw_text_input = row.get("text_input", row.get("query_label"))
        text_input = None if raw_text_input is None else str(raw_text_input)
        if (
            text_input
            and (
                getattr(model._backend, "SUPPORTS_PROMPT_ANSWER_ENCODING", False)
                or getattr(model._backend, "supports_prompt_answer_encoding", False)
            )
        ):
            inputs = model._backend.encode_with_task(
                images=[image],
                task_name=args.task_type,
                prompt_text_input=text_input,
                return_tensors="pt",
            )
        else:
            prompt = f"{task_prompt}{text_input}" if text_input else task_prompt
            inputs = model._backend.encode(text=[prompt], images=[image], return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = getattr(generated, "sequences", generated)
        scores = list(getattr(generated, "scores", []) or [])
        decoded = model.decode(sequences, skip_special_tokens=False)
        offset = max(0, int(sequences.shape[-1]) - len(scores)) if scores else int(sequences.shape[-1])
        generated_token_ids = sequences[0, offset:].detach().cpu().tolist() if scores else []

        steps = []
        for step_idx, score in enumerate(scores):
            token_id = int(generated_token_ids[step_idx]) if step_idx < len(generated_token_ids) else None
            steps.append({
                "step": step_idx,
                "generated_token_id": token_id,
                "generated_token": _decode_token(tokenizer, token_id) if token_id is not None else None,
                **rank_tokens_from_scores(score, tokenizer, vp_token_ids, top_k=args.top_k),
            })
        records.append({
            "index": idx,
            "image": row.get("image"),
            "task_prompt": task_prompt,
            "task_type": args.task_type,
            "text_input": text_input,
            "target": row.get("suffix"),
            "decoded": decoded[0] if isinstance(decoded, list) and decoded else str(decoded),
            "num_score_steps": len(scores),
            "steps": steps,
        })

    return {
        "num_samples": len(records),
        "max_new_tokens": args.max_new_tokens,
        "records": records,
    }


def _classify_probe(summary: Mapping[str, Any]) -> str:
    inventory = summary.get("token_inventory", {})
    labels = summary.get("label_probe", {})
    generation = summary.get("generation_probe", {})
    marker_style = str(inventory.get("marker_style", "special"))
    if marker_style == "special" and not inventory.get("all_vp_tokens_single"):
        return "blocked_vp_tokens_not_single"
    marker_count = int(labels.get("total_vp_marker_sequence_count", 0) or 0)
    if marker_count <= 0:
        marker_count = int(labels.get("total_vp_marker_token_count", 0) or 0)
    if marker_count <= 0:
        return "blocked_vp_markers_not_supervised"
    first_content_ranks = []
    ref_probe_key = str(summary.get("ref_open_probe_key") or "<|ref|>")
    for record in generation.get("records", []):
        for step in record.get("steps", []):
            if step.get("generated_token") in {"<s>", "</s>", "<pad>"}:
                continue
            ref_stats = step.get("selected_tokens", {}).get(ref_probe_key, {})
            rank = ref_stats.get("rank")
            if rank is not None:
                first_content_ranks.append(int(rank))
            break
    if first_content_ranks and min(first_content_ranks) > 100:
        return "generation_prior_blocks_wrapper"
    return "vp_token_path_probe_ready"


def render_markdown(summary: Mapping[str, Any]) -> str:
    inventory = summary.get("token_inventory", {})
    labels = summary.get("label_probe", {})
    generation = summary.get("generation_probe", {})
    lines = [
        "# Florence-VP Token Probe",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Model: `{summary.get('model_path')}`",
        f"- Adapter: `{summary.get('adapter_dir') or 'none'}`",
        f"- Tokenizer vocab size: `{inventory.get('tokenizer_vocab_size')}`",
        f"- Model vocab size: `{inventory.get('model_vocab_size')}`",
        f"- Embedding resized: `{inventory.get('embedding_resized')}`",
        f"- All VP markers single-token: `{inventory.get('all_vp_tokens_single')}`",
        f"- Ref-open probe key: `{summary.get('ref_open_probe_key')}`",
        "",
        "## Label Probe",
        "",
        f"- Samples: `{labels.get('num_samples', 0)}`",
        f"- Supervised tokens: `{labels.get('total_supervised_token_count', 0)}`",
        f"- VP marker supervised tokens: `{labels.get('total_vp_marker_token_count', 0)}`",
        f"- VP marker supervised sequences: `{labels.get('total_vp_marker_sequence_count', 0)}`",
        f"- Loc tokens: `{labels.get('total_loc_token_count', 0)}`",
        "",
        "## Generation Probe",
        "",
    ]
    for record in generation.get("records", []):
        lines.append(f"- Sample `{record.get('index')}` decoded: `{record.get('decoded')}`")
        content_step = next(
            (
                step for step in record.get("steps", [])
                if step.get("generated_token") not in {"<s>", "</s>", "<pad>"}
            ),
            None,
        )
        if content_step:
            ref_key = str(summary.get("ref_open_probe_key") or "<|ref|>")
            box_key = str(summary.get("box_open_probe_key") or "<|box|>")
            ref_stats = content_step.get("selected_tokens", {}).get(ref_key, {})
            box_stats = content_step.get("selected_tokens", {}).get(box_key, {})
            lines.append(
                f"  - First content step `{content_step.get('generated_token')}` ranks: "
                f"`{ref_key}`={ref_stats.get('rank')}, `{box_key}`={box_stats.get('rank')}"
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest_path).expanduser()
    if args.data_path:
        data_path = Path(args.data_path).expanduser()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_path = _resolve_data_path(manifest, args.split, args.manifest_data_key)
    rows = _read_jsonl(data_path)
    model = _load_model(args)
    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer; cannot probe VP token ids")

    inventory = build_token_inventory(tokenizer, model, marker_style=args.marker_style)
    vp_token_ids = {
        str(row["token"]): row.get("token_id")
        for row in inventory["vp_tokens"]
    }
    vp_tokenizations = {}
    for row in inventory["vp_tokens"]:
        token = str(row["token"])
        variants = [[int(value) for value in row.get("tokenized_ids", [])]]
        spaced_ids = _tokenize_ids(tokenizer, f" {token}")
        if spaced_ids and spaced_ids not in variants:
            variants.append([int(value) for value in spaced_ids])
        vp_tokenizations[token] = variants
    probe_token_ids = _build_probe_token_ids(inventory["vp_tokens"])
    marker_tokens = get_visual_primitive_tokens(args.marker_style)
    ref_probe_key = _probe_key_for_marker(inventory["vp_tokens"], marker_tokens["ref_open"])
    box_probe_key = _probe_key_for_marker(inventory["vp_tokens"], marker_tokens["box_open"])
    dataset = _build_dataset(model, data_path, args.task_type)
    summary: Dict[str, Any] = {
        "model_path": str(Path(args.model_path).expanduser()),
        "adapter_dir": args.adapter_dir,
        "manifest_path": None if args.data_path else str(manifest_path),
        "manifest_data_key": args.manifest_data_key,
        "data_path": str(data_path),
        "task_type": args.task_type,
        "split": args.split,
        "marker_style": args.marker_style,
        "device": _choose_device(args.device),
        "torch_dtype": args.torch_dtype,
        "ref_open_probe_key": ref_probe_key,
        "box_open_probe_key": box_probe_key,
        "token_inventory": inventory,
        "label_probe": _probe_labels(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            vp_token_ids=vp_token_ids,
            vp_tokenizations=vp_tokenizations,
            max_samples=args.max_samples,
        ),
        "generation_probe": _probe_generation(
            model=model,
            rows=rows,
            tokenizer=tokenizer,
            vp_token_ids=probe_token_ids,
            args=args,
        ),
    }
    summary["status"] = _classify_probe(summary)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "florence_vp_token_probe.json"
    markdown_path = output_dir / "florence_vp_token_probe.md"
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--manifest-data-key",
        default=None,
        help="Manifest key to inspect; comma-separated keys are tried in order.",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--task-type", default="OD_VP")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--marker-style", default="special", choices=["special", "plain"])
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_token_probe")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
