#!/usr/bin/env python3
"""Generate and visualize Florence-VP predictions from a saved LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.core.visual_primitives import (
    COORDINATE_MAX,
    VISUAL_PRIMITIVE_MARKER_SETS,
    denormalize_bbox,
)
from florence_forge.evaluation.structured_vp_decoder import (
    StructuredVisualPrimitiveDecoder,
    normalize_structured_vp_task_prompt,
    resolve_structured_vp_filter_caps,
)
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


_STRUCTURED_VP_AUTO_TASKS = {
    "OD",
    "OD_VP",
    "COUNT_VP",
    "PHRASE_GROUNDING_VP",
    "OPEN_VOCABULARY_DETECTION",
    "CAPTION_TO_PHRASE_GROUNDING",
}

_LOC_TOKEN_RE = re.compile(r"<loc_\d+>")
_GENERATION_BOUNDARY_TOKENS = ("</s>", "<s>", "<pad>", "<bos>", "<eos>")
_BOX_CLOSE_TOKENS = ("</box>", "<|/box|>")


class VPBoxCountStoppingCriteria:
    """Stop generation after a target number of VP/Florence loc-token boxes.

    Florence boxes are four ``<loc_*>`` tokens.  The visualization script runs
    one image at a time, so stopping the whole decode when any generated
    sequence reaches the target keeps the common greedy path fast while still
    working with beam search diagnostics.
    """

    def __init__(self, loc_token_ids: Sequence[int], max_total_boxes: int):
        self.loc_token_ids = frozenset(int(token_id) for token_id in loc_token_ids)
        self.max_total_boxes = int(max_total_boxes)
        self.target_loc_tokens = self.max_total_boxes * 4
        self._loc_token_tensors: Dict[Tuple[str, str], Any] = {}
        self.triggered = False
        self.last_loc_token_count = 0
        if self.max_total_boxes < 1:
            raise ValueError("max_total_boxes must be >= 1")
        if not self.loc_token_ids:
            raise ValueError("loc_token_ids must not be empty")

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        if input_ids is None or self.target_loc_tokens <= 0:
            return False
        try:
            counts = self._count_loc_tokens(input_ids)
            max_count = int(counts.max().item()) if counts.numel() else 0
            self.last_loc_token_count = max_count
            should_stop = bool((counts >= self.target_loc_tokens).any().item())
            self.triggered = self.triggered or should_stop
            return should_stop
        except Exception:
            rows = input_ids.detach().cpu().tolist()
            if rows and isinstance(rows[0], int):
                rows = [rows]
            max_count = max(
                (
                    sum(1 for token_id in row if int(token_id) in self.loc_token_ids)
                    for row in rows
                ),
                default=0,
            )
            self.last_loc_token_count = max_count
            should_stop = max_count >= self.target_loc_tokens
            self.triggered = self.triggered or should_stop
            return should_stop

    def _count_loc_tokens(self, input_ids: Any) -> Any:
        import torch

        key = (str(input_ids.device), str(input_ids.dtype))
        loc_tensor = self._loc_token_tensors.get(key)
        if loc_tensor is None:
            loc_tensor = torch.tensor(
                sorted(self.loc_token_ids),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            self._loc_token_tensors[key] = loc_tensor
        return torch.isin(input_ids, loc_tensor).sum(dim=-1)


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_query_box_count(row: Dict[str, Any]) -> Optional[int]:
    for key in ("query_box_count", "gt_box_count", "box_count"):
        value = _coerce_optional_int(row.get(key))
        if value is not None:
            return value
    return None


def _filter_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        query_box_count = _row_query_box_count(row)
        if args.min_query_boxes is not None:
            if query_box_count is None or query_box_count < args.min_query_boxes:
                continue
        if args.max_query_boxes is not None:
            if query_box_count is None or query_box_count > args.max_query_boxes:
                continue
        filtered.append(row)
    return filtered


def _prediction_length_diagnostics(
    raw_prediction_text: str,
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> Dict[str, Any]:
    token_count: Optional[int] = None
    if tokenizer is not None:
        try:
            token_count = len(tokenizer.encode(raw_prediction_text, add_special_tokens=False))
        except Exception:
            token_count = None
    loc_token_count = len(_LOC_TOKEN_RE.findall(raw_prediction_text))
    near_threshold = max(1, int(max_new_tokens * 0.9))
    return {
        "raw_prediction_token_count": token_count,
        "raw_loc_token_count": loc_token_count,
        "raw_loc_box_count": loc_token_count // 4,
        "generation_budget_near_hit": (
            token_count is not None and token_count >= near_threshold
        ),
        "generation_budget_hit": (
            token_count is not None and token_count >= max_new_tokens
        ),
    }


def _raw_loc_box_count(text: str) -> int:
    return len(_LOC_TOKEN_RE.findall(str(text))) // 4


def _vp_continuation_box_count(
    text: str,
    *,
    parser: Optional[VisualPrimitiveParser] = None,
    structured_decoder: Optional[StructuredVisualPrimitiveDecoder] = None,
    structured_filter_caps: Optional[Dict[str, Any]] = None,
    allowed_label_match_mode: str = "strict",
    repair_malformed_tail: bool = False,
) -> Tuple[int, str]:
    """Count boxes using the same parseable output path as evaluation.

    Raw loc-token counts can be misleading when malformed tails appear outside
    a valid VP ``<box>...</box>`` span.  Continuation therefore keys off the
    parser/structured decoder result that downstream metrics can actually use.
    """

    text = str(text or "")
    if structured_decoder is not None:
        caps = structured_filter_caps or {}
        try:
            result = structured_decoder.decode(
                text,
                max_boxes_per_label=caps.get("max_boxes_per_label"),
                max_total_boxes=caps.get("max_total_boxes"),
                nms_iou_threshold=caps.get("nms_iou_threshold"),
                allowed_labels=caps.get("allowed_labels"),
                allowed_label_match_mode=allowed_label_match_mode,
                repair_malformed_tail=repair_malformed_tail,
            )
            return len(result.detections), f"structured_{result.source}"
        except Exception:
            pass

    parser = parser or VisualPrimitiveParser()
    try:
        return len(parser.parse_detections(text)), "visual_primitive_parser"
    except Exception:
        return 0, "visual_primitive_parser_error"


def _resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data_path:
        return Path(args.data_path).expanduser()

    manifest_path = Path(args.manifest_path).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = args.data_key or ("val_od_path" if args.split == "val" else "train_od_path")
    data_path = manifest.get(key)
    if not data_path:
        raise KeyError(f"{manifest_path} does not contain {key}")
    return Path(data_path)


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
    """Patch Florence remote-code language models on Transformers versions
    where PreTrainedModel no longer inherits GenerationMixin.
    """

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


def _to_pixel_box(vp_box: Sequence[int], image_size: Tuple[int, int]) -> List[float]:
    x1, y1, x2, y2 = denormalize_bbox(vp_box, image_size)
    width, height = image_size
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def _try_native_od_parse(model: Florence2MultiTaskModel, text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    processor = model.processor
    post_process = getattr(processor, "post_process_generation", None)
    if post_process is None:
        return []
    try:
        parsed = post_process(text, task="<OD>", image_size=image_size)
    except Exception:
        return []
    od = parsed.get("<OD>") if isinstance(parsed, dict) else None
    if not isinstance(od, dict):
        return []
    boxes = od.get("bboxes") or []
    labels = od.get("labels") or []
    detections = []
    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        label = labels[i] if i < len(labels) else "native"
        detections.append({
            "label": str(label),
            "bbox": [float(v) for v in box],
            "format": "native",
            "source": "processor_postprocess",
        })
    return detections


def _annotate_pred_detections(
    detections: List[Dict[str, Any]],
    *,
    det_format: str,
    source: str,
) -> List[Dict[str, Any]]:
    annotated = []
    for detection in detections:
        item = dict(detection)
        item.setdefault("format", det_format)
        item["source"] = source
        annotated.append(item)
    return annotated


def _format_decoder_prefix(template: Optional[str], gt_detections: List[Dict[str, Any]]) -> Optional[str]:
    if not template:
        return None
    label = str(gt_detections[0].get("label", "")) if gt_detections else ""
    return template.replace("{label}", label)


def _format_text_input(
    template: Optional[str],
    text_input: Any,
    gt_detections: List[Dict[str, Any]],
) -> Optional[str]:
    if text_input is None:
        return None
    value = str(text_input)
    if not template:
        return value
    first_label = str(gt_detections[0].get("label", "")) if gt_detections else value
    return (
        str(template)
        .replace("{text_input}", value)
        .replace("{label}", first_label)
        .replace("{query_box_count}", str(len(gt_detections)))
    )


def _generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.length_penalty is not None:
        kwargs["length_penalty"] = args.length_penalty
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.early_stopping:
        kwargs["early_stopping"] = True
    return kwargs


def _resolve_loc_token_ids(tokenizer: Any, *, coord_max: int = COORDINATE_MAX) -> List[int]:
    if tokenizer is None:
        return []
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return []
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    loc_token_ids: List[int] = []
    seen = set()
    for coord in range(coord_max + 1):
        token = f"<loc_{coord}>"
        token_id = convert(token)
        if token_id is None or isinstance(token_id, list):
            continue
        try:
            token_id_int = int(token_id)
        except (TypeError, ValueError):
            continue
        if unk_token_id is not None and token_id_int == int(unk_token_id) and token not in vocab:
            continue
        if token_id_int in seen:
            continue
        seen.add(token_id_int)
        loc_token_ids.append(token_id_int)
    return loc_token_ids


def _build_vp_count_stopping_criteria(
    *,
    tokenizer: Any,
    max_total_boxes: Optional[int],
) -> Tuple[Optional[Any], Dict[str, Any]]:
    target_boxes = _coerce_optional_int(max_total_boxes)
    info = {
        "vp_count_stopping_target_boxes": target_boxes,
        "vp_count_stopping_available": False,
        "vp_count_stopping_loc_token_id_count": 0,
    }
    if target_boxes is None or target_boxes < 1:
        return None, info

    loc_token_ids = _resolve_loc_token_ids(tokenizer)
    info["vp_count_stopping_loc_token_id_count"] = len(loc_token_ids)
    if not loc_token_ids:
        return None, info

    try:
        from transformers import StoppingCriteriaList
    except Exception:
        return None, info

    criteria = StoppingCriteriaList([
        VPBoxCountStoppingCriteria(loc_token_ids, target_boxes)
    ])
    info["vp_count_stopping_available"] = True
    return criteria, info


def _vp_count_stopping_runtime_info(stopping_criteria: Optional[Any]) -> Dict[str, Any]:
    if not stopping_criteria:
        return {
            "vp_count_stopping_triggered": False,
            "vp_count_stopping_last_loc_token_count": 0,
        }
    criteria_items = list(stopping_criteria)
    for criterion in criteria_items:
        if isinstance(criterion, VPBoxCountStoppingCriteria):
            return {
                "vp_count_stopping_triggered": bool(criterion.triggered),
                "vp_count_stopping_last_loc_token_count": int(criterion.last_loc_token_count),
            }
    return {
        "vp_count_stopping_triggered": False,
        "vp_count_stopping_last_loc_token_count": 0,
    }


def _strip_generation_boundary_tokens(text: str) -> str:
    text = str(text or "").strip()
    changed = True
    while changed:
        changed = False
        for token in _GENERATION_BOUNDARY_TOKENS:
            if text.startswith(token):
                text = text[len(token):].lstrip()
                changed = True
            if text.endswith(token):
                text = text[:-len(token)].rstrip()
                changed = True
    return text


def _strip_trailing_box_close(text: str) -> str:
    text = str(text or "").rstrip()
    for token in _BOX_CLOSE_TOKENS:
        if text.endswith(token):
            return text[:-len(token)].rstrip()
    return text


def _truncate_to_last_parseable_box_payload(text: str) -> str:
    text = str(text or "").rstrip()
    if not text:
        return ""

    parser = VisualPrimitiveParser()
    candidates: List[Tuple[int, str]] = []
    for markers in VISUAL_PRIMITIVE_MARKER_SETS.values():
        box_pattern = re.compile(
            rf"{re.escape(markers['box_open'])}(.*?){re.escape(markers['box_close'])}",
            flags=re.DOTALL,
        )
        for match in box_pattern.finditer(text):
            candidate_with_close = text[:match.end()]
            try:
                if not parser.parse_detections(candidate_with_close):
                    continue
            except Exception:
                continue
            payload = match.group(1)
            loc_matches = list(_LOC_TOKEN_RE.finditer(payload))
            payload_end = (
                match.start(1) + loc_matches[-1].end()
                if loc_matches else match.end(1)
            )
            candidates.append((match.end(), text[:payload_end].rstrip()))

    if not candidates:
        return ""
    _, prefix = max(candidates, key=lambda item: item[0])
    return prefix


def _build_continuation_decoder_prefix(
    raw_prediction_text: str,
    *,
    fallback_prefix: Optional[str],
) -> str:
    prefix = _strip_generation_boundary_tokens(raw_prediction_text)
    parseable_prefix = _truncate_to_last_parseable_box_payload(prefix)
    if parseable_prefix:
        return parseable_prefix
    prefix = _strip_trailing_box_close(prefix)
    if prefix:
        return prefix
    return str(fallback_prefix or "").strip()


def _normalize_task_prompt_for_vp(task_prompt: Any) -> str:
    return normalize_structured_vp_task_prompt(task_prompt)


def _structured_vp_decode_enabled(args: argparse.Namespace, task_prompt: Any) -> bool:
    if args.structured_vp_decode:
        return True
    mode = str(getattr(args, "structured_vp_mode", "off") or "off").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        return _normalize_task_prompt_for_vp(task_prompt) in _STRUCTURED_VP_AUTO_TASKS
    raise ValueError("structured_vp_mode must be one of: off, auto, on")


def _resolve_row_allowed_labels(args: argparse.Namespace, row: Dict[str, Any], gt: List[Dict[str, Any]]) -> Any:
    if args.structured_vp_allowed_labels:
        return args.structured_vp_allowed_labels
    field = getattr(args, "structured_vp_allowed_labels_field", None)
    if not field:
        return None
    if field in {"target_labels", "reference_labels", "gt_labels"}:
        return [
            str(detection.get("label", "")).strip()
            for detection in gt
            if str(detection.get("label", "")).strip()
        ]
    value: Any = row
    for part in str(field).split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _resolve_row_positive_int_field(
    row: Dict[str, Any],
    field_spec: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> Optional[int]:
    if field_spec:
        for field in str(field_spec).replace("|", ",").replace(";", ",").split(","):
            field = field.strip()
            if not field:
                continue
            value: Any = row
            for part in field.split("."):
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
            try:
                if value is not None and str(value).strip() != "":
                    parsed = int(value)
                    if parsed >= 1:
                        return parsed
            except (TypeError, ValueError):
                continue
    if fallback is None:
        return None
    parsed_fallback = int(fallback)
    return parsed_fallback if parsed_fallback >= 1 else None


def _generate_prediction_text(
    *,
    model: Florence2MultiTaskModel,
    image: Image.Image,
    task_prompt: str,
    text_input: Optional[str],
    gt_detections: List[Dict[str, Any]],
    args: argparse.Namespace,
    stopping_criteria: Optional[Any] = None,
    decoder_prefix_override: Optional[str] = None,
    max_new_tokens_override: Optional[int] = None,
) -> str:
    import torch

    decoder_prefix = (
        decoder_prefix_override
        if decoder_prefix_override is not None
        else _format_decoder_prefix(args.decoder_prefix, gt_detections)
    )
    max_new_tokens = (
        int(max_new_tokens_override)
        if max_new_tokens_override is not None
        else int(args.max_new_tokens)
    )
    generation_kwargs = _generation_kwargs(args)
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria
    if not decoder_prefix:
        prediction = model.generate(
            images=image,
            task_prompt=task_prompt,
            text_input=text_input,
            max_new_tokens=max_new_tokens,
            num_beams=args.num_beams,
            **generation_kwargs,
        )
        return str(prediction)

    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer; cannot build decoder prefix")

    prompt = f"{task_prompt}{text_input}" if text_input else task_prompt
    inputs = model._backend.encode(text=[prompt], images=[image], return_tensors="pt")
    decoder_input_ids = tokenizer(
        decoder_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask"),
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=args.num_beams,
            **generation_kwargs,
        )
    decoded = model.decode(generated_ids, skip_special_tokens=False)
    return decoded[0] if isinstance(decoded, list) and decoded else str(decoded)


def _maybe_continue_underfilled_prediction(
    *,
    model: Florence2MultiTaskModel,
    image: Image.Image,
    task_prompt: str,
    text_input: Optional[str],
    gt_detections: List[Dict[str, Any]],
    args: argparse.Namespace,
    raw_prediction_text: str,
    target_box_count: Optional[int],
    parser: Optional[VisualPrimitiveParser] = None,
    structured_decoder: Optional[StructuredVisualPrimitiveDecoder] = None,
    structured_filter_caps: Optional[Dict[str, Any]] = None,
    structured_vp_allowed_label_match_mode: str = "strict",
    structured_vp_repair_malformed_tail: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    initial_parseable_boxes, initial_count_source = _vp_continuation_box_count(
        raw_prediction_text,
        parser=parser,
        structured_decoder=structured_decoder,
        structured_filter_caps=structured_filter_caps,
        allowed_label_match_mode=structured_vp_allowed_label_match_mode,
        repair_malformed_tail=structured_vp_repair_malformed_tail,
    )
    info: Dict[str, Any] = {
        "vp_continuation_enabled": bool(args.continue_underfilled_vp_boxes),
        "vp_continuation_attempted": False,
        "vp_continuation_applied": False,
        "vp_continuation_rounds": 0,
        "vp_continuation_target_boxes": target_box_count,
        "vp_continuation_count_basis": (
            "structured_decoder" if structured_decoder is not None else "visual_primitive_parser"
        ),
        "vp_continuation_initial_count_source": initial_count_source,
        "vp_continuation_final_count_source": initial_count_source,
        "vp_continuation_initial_raw_loc_box_count": _raw_loc_box_count(raw_prediction_text),
        "vp_continuation_final_raw_loc_box_count": _raw_loc_box_count(raw_prediction_text),
        "vp_continuation_added_raw_loc_boxes": 0,
        "vp_continuation_initial_parseable_box_count": initial_parseable_boxes,
        "vp_continuation_final_parseable_box_count": initial_parseable_boxes,
        "vp_continuation_added_parseable_boxes": 0,
        "vp_continuation_reached_target": False,
        "vp_continuation_stop_triggered": False,
        "vp_continuation_prefix_token_count": None,
        "vp_continuation_last_candidate_raw_prediction": None,
        "vp_continuation_last_candidate_raw_loc_box_count": None,
        "vp_continuation_last_candidate_parseable_box_count": None,
        "vp_continuation_last_candidate_count_source": None,
    }
    if not args.continue_underfilled_vp_boxes:
        return raw_prediction_text, info
    target_boxes = _coerce_optional_int(target_box_count)
    if target_boxes is None or target_boxes < 1:
        return raw_prediction_text, info

    current_text = str(raw_prediction_text)
    current_boxes = initial_parseable_boxes
    current_count_source = initial_count_source
    info["vp_continuation_reached_target"] = current_boxes >= target_boxes
    if target_boxes - current_boxes < max(1, int(args.vp_continuation_min_missing_boxes)):
        return current_text, info

    tokenizer = getattr(model.processor, "tokenizer", None)
    fallback_prefix = _format_decoder_prefix(args.decoder_prefix, gt_detections)
    max_rounds = max(0, int(args.vp_continuation_max_rounds))
    max_new_tokens = max(1, int(args.vp_continuation_max_new_tokens))
    for _ in range(max_rounds):
        continuation_prefix = _build_continuation_decoder_prefix(
            current_text,
            fallback_prefix=fallback_prefix,
        )
        if not continuation_prefix:
            break
        info["vp_continuation_attempted"] = True
        if tokenizer is not None:
            try:
                info["vp_continuation_prefix_token_count"] = len(
                    tokenizer.encode(continuation_prefix, add_special_tokens=False)
                )
            except Exception:
                info["vp_continuation_prefix_token_count"] = None
        stopping_criteria, _ = _build_vp_count_stopping_criteria(
            tokenizer=tokenizer,
            max_total_boxes=target_boxes,
        )
        continued_text = _generate_prediction_text(
            model=model,
            image=image,
            task_prompt=task_prompt,
            text_input=text_input,
            gt_detections=gt_detections,
            args=args,
            stopping_criteria=stopping_criteria,
            decoder_prefix_override=continuation_prefix,
            max_new_tokens_override=max_new_tokens,
        )
        runtime_info = _vp_count_stopping_runtime_info(stopping_criteria)
        info["vp_continuation_stop_triggered"] = (
            bool(info["vp_continuation_stop_triggered"])
            or bool(runtime_info["vp_count_stopping_triggered"])
        )
        continued_boxes, continued_count_source = _vp_continuation_box_count(
            continued_text,
            parser=parser,
            structured_decoder=structured_decoder,
            structured_filter_caps=structured_filter_caps,
            allowed_label_match_mode=structured_vp_allowed_label_match_mode,
            repair_malformed_tail=structured_vp_repair_malformed_tail,
        )
        info["vp_continuation_last_candidate_raw_prediction"] = continued_text
        info["vp_continuation_last_candidate_raw_loc_box_count"] = _raw_loc_box_count(
            continued_text
        )
        info["vp_continuation_last_candidate_parseable_box_count"] = continued_boxes
        info["vp_continuation_last_candidate_count_source"] = continued_count_source
        info["vp_continuation_rounds"] += 1
        if continued_boxes <= current_boxes:
            break
        current_text = continued_text
        current_boxes = continued_boxes
        current_count_source = continued_count_source
        info["vp_continuation_applied"] = True
        if current_boxes >= target_boxes:
            break

    final_raw_boxes = _raw_loc_box_count(current_text)
    info["vp_continuation_final_count_source"] = current_count_source
    info["vp_continuation_final_raw_loc_box_count"] = final_raw_boxes
    info["vp_continuation_added_raw_loc_boxes"] = max(
        0,
        final_raw_boxes - int(info["vp_continuation_initial_raw_loc_box_count"]),
    )
    info["vp_continuation_final_parseable_box_count"] = current_boxes
    info["vp_continuation_added_parseable_boxes"] = max(
        0,
        current_boxes - int(info["vp_continuation_initial_parseable_box_count"]),
    )
    info["vp_continuation_reached_target"] = current_boxes >= target_boxes
    return current_text, info


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=color + (220,),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _wrap(text: str, width: int = 100, max_lines: int = 10) -> str:
    lines: List[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def _draw_visualization(
    *,
    image: Image.Image,
    gt_detections: List[Dict[str, Any]],
    pred_detections: List[Dict[str, Any]],
    raw_prediction_text: str,
    structured_prediction_text: str,
    prediction_source: str,
    target_text: str,
    save_path: Path,
) -> None:
    image = image.convert("RGB")
    width, height = image.size
    panel_height = 300
    canvas = Image.new("RGB", (width, height + panel_height), color=(18, 22, 29))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for det in gt_detections:
        box = _to_pixel_box(det["bbox"], image.size)
        draw.rectangle(box, outline=(80, 220, 120, 255), width=4)
        _draw_label(draw, (box[0] + 3, max(3, box[1] + 3)), f"GT {det['label']}", (39, 150, 76))

    for det in pred_detections:
        box = det["bbox"] if det.get("format") == "native" else _to_pixel_box(det["bbox"], image.size)
        draw.rectangle(box, outline=(245, 83, 83, 255), width=3)
        det_format = det.get("format")
        det_source = det.get("source", prediction_source)
        if det_format == "native":
            prefix = "PRED native"
        elif det_source == "florence_native":
            prefix = "PRED VP(native)"
        elif det_source == "visual_primitive":
            prefix = "PRED VP"
        else:
            prefix = "PRED"
        _draw_label(draw, (box[0] + 3, min(height - 18, max(3, box[1] + 22))), f"{prefix} {det['label']}", (190, 45, 45))

    font = ImageFont.load_default()
    y = height + 12
    draw.text((14, y), f"Structured VP ({prediction_source})", fill=(255, 255, 255), font=font)
    draw.text((14, y + 20), _wrap(structured_prediction_text, max_lines=6), fill=(225, 231, 240), font=font)
    draw.text((14, y + 106), "Raw prediction", fill=(255, 255, 255), font=font)
    draw.text((14, y + 126), _wrap(raw_prediction_text, max_lines=4), fill=(204, 214, 226), font=font)
    draw.text((14, y + 202), "Target", fill=(255, 255, 255), font=font)
    draw.text((14, y + 222), _wrap(target_text, max_lines=4), fill=(195, 245, 205), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    model = _load_model(args)
    parser = VisualPrimitiveParser()
    structured_decoder = StructuredVisualPrimitiveDecoder(
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
        repair_malformed_tail=args.structured_vp_repair_malformed_tail,
    )
    data_path = _resolve_data_path(args)
    all_rows = _read_jsonl(data_path)
    filtered_rows = _filter_rows(all_rows, args)
    rows = filtered_rows[: args.max_samples]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(rows):
        image_path = Path(row["image"]).expanduser()
        image = Image.open(image_path).convert("RGB")
        target_text = str(row.get("suffix", ""))
        gt = parser.parse_detections(target_text)
        text_input_for_generation = _format_text_input(
            args.text_input_template,
            row.get("text_input"),
            gt,
        )
        use_structured_decode = _structured_vp_decode_enabled(args, row.get("prefix", "<OD>")) or args.native_fallback
        dynamic_max_total_boxes = _resolve_row_positive_int_field(
            row,
            args.structured_vp_max_total_boxes_field,
            fallback=args.structured_vp_max_total_boxes,
        )
        resolved_filter_caps = resolve_structured_vp_filter_caps(
            policy=args.structured_vp_filter_policy,
            task_prompt=row.get("prefix", "<OD>"),
            max_boxes_per_label=args.structured_vp_max_boxes_per_label,
            max_total_boxes=dynamic_max_total_boxes,
            nms_iou_threshold=args.structured_vp_nms_iou_threshold,
            allowed_labels=_resolve_row_allowed_labels(args, row, gt),
        )
        stopping_criteria, stopping_info = _build_vp_count_stopping_criteria(
            tokenizer=getattr(model.processor, "tokenizer", None),
            max_total_boxes=(
                resolved_filter_caps["max_total_boxes"]
                if args.stop_after_vp_max_total_boxes else None
            ),
        )
        raw_prediction_text = _generate_prediction_text(
            model=model,
            image=image,
            task_prompt=row.get("prefix", "<OD>"),
            text_input=text_input_for_generation,
            gt_detections=gt,
            args=args,
            stopping_criteria=stopping_criteria,
        )
        stopping_runtime_info = _vp_count_stopping_runtime_info(stopping_criteria)
        initial_raw_prediction_text = raw_prediction_text
        raw_prediction_text, continuation_info = _maybe_continue_underfilled_prediction(
            model=model,
            image=image,
            task_prompt=row.get("prefix", "<OD>"),
            text_input=text_input_for_generation,
            gt_detections=gt,
            args=args,
            raw_prediction_text=raw_prediction_text,
            target_box_count=resolved_filter_caps["max_total_boxes"],
            parser=parser,
            structured_decoder=structured_decoder if use_structured_decode else None,
            structured_filter_caps=resolved_filter_caps if use_structured_decode else None,
            structured_vp_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
            structured_vp_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
        )
        length_diagnostics = _prediction_length_diagnostics(
            raw_prediction_text,
            tokenizer=getattr(model.processor, "tokenizer", None),
            max_new_tokens=args.max_new_tokens,
        )
        vp_pred = parser.parse_detections(raw_prediction_text)
        pred = _annotate_pred_detections(
            vp_pred,
            det_format="visual_primitive",
            source="visual_primitive",
        )
        structured_prediction_text = raw_prediction_text
        structured_source = "disabled"
        structured_pred: List[Dict[str, Any]] = []
        used_structured_decoder = False
        if use_structured_decode:
            structured_result = structured_decoder.decode(
                raw_prediction_text,
                max_boxes_per_label=resolved_filter_caps["max_boxes_per_label"],
                max_total_boxes=resolved_filter_caps["max_total_boxes"],
                nms_iou_threshold=resolved_filter_caps["nms_iou_threshold"],
                allowed_labels=resolved_filter_caps["allowed_labels"],
                allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                repair_malformed_tail=args.structured_vp_repair_malformed_tail,
            )
            structured_prediction_text = structured_result.text
            structured_source = structured_result.source
            used_structured_decoder = structured_result.used_structured_decoder
            det_format = "structured_vp" if structured_result.source == "florence_native" else "visual_primitive"
            structured_pred = _annotate_pred_detections(
                structured_result.detections,
                det_format=det_format,
                source=structured_result.source,
            )
            if structured_pred:
                pred = structured_pred

        native_pred = []
        used_native_fallback = False
        if not pred and args.native_fallback:
            native_pred = _try_native_od_parse(model, raw_prediction_text, image.size)
            pred = native_pred
            used_native_fallback = bool(native_pred)

        save_path = None
        if args.visualization_limit is None or idx < args.visualization_limit:
            save_path = output_dir / f"{idx:02d}_{image_path.stem}_vp.png"
            _draw_visualization(
                image=image,
                gt_detections=gt,
                pred_detections=pred,
                raw_prediction_text=raw_prediction_text,
                structured_prediction_text=structured_prediction_text,
                prediction_source=structured_source,
                target_text=target_text,
                save_path=save_path,
            )
        record = {
            "index": idx,
            "image": str(image_path),
            "prefix": row.get("prefix", "<OD>"),
            "prediction": raw_prediction_text,
            "raw_prediction": raw_prediction_text,
            "initial_raw_prediction": initial_raw_prediction_text,
            "structured_prediction": structured_prediction_text,
            "structured_source": structured_source,
            "target": target_text,
            "text_input_for_generation": text_input_for_generation,
            "pred_box_count": len(pred),
            "vp_pred_box_count": len(vp_pred),
            "structured_pred_box_count": len(structured_pred),
            "native_pred_box_count": len(native_pred),
            "vp_format_valid": bool(vp_pred),
            "structured_vp_format_valid": bool(structured_pred),
            "used_structured_vp_decoder": used_structured_decoder,
            "structured_vp_filter_policy": args.structured_vp_filter_policy,
            "structured_vp_resolved_max_boxes_per_label": resolved_filter_caps["max_boxes_per_label"],
            "structured_vp_resolved_max_total_boxes": resolved_filter_caps["max_total_boxes"],
            "structured_vp_resolved_nms_iou_threshold": resolved_filter_caps["nms_iou_threshold"],
            "structured_vp_resolved_allowed_labels": resolved_filter_caps["allowed_labels"],
            "structured_vp_allowed_label_match_mode": args.structured_vp_allowed_label_match_mode,
            "stop_after_vp_max_total_boxes": args.stop_after_vp_max_total_boxes,
            "structured_raw_detection_count": (
                structured_result.raw_detection_count if use_structured_decode else 0
            ),
            "structured_filtered_detection_count": (
                structured_result.filtered_detection_count if use_structured_decode else 0
            ),
            "structured_repaired_tail_detection_count": (
                structured_result.repaired_tail_detection_count if use_structured_decode else 0
            ),
            "used_native_fallback": used_native_fallback,
            "gt_box_count": len(gt),
            "visualization_path": str(save_path) if save_path else None,
        }
        record.update(length_diagnostics)
        record.update(stopping_info)
        record.update(stopping_runtime_info)
        record.update(continuation_info)
        for key, value in row.items():
            if key not in {"image", "prefix", "suffix"} and key not in record:
                record[key] = value
        records.append(record)

    vp_valid_count = sum(1 for record in records if record["vp_format_valid"])
    structured_valid_count = sum(1 for record in records if record["structured_vp_format_valid"])
    structured_decoder_count = sum(1 for record in records if record["used_structured_vp_decoder"])
    fallback_count = sum(1 for record in records if record["used_native_fallback"])
    filtered_detection_count = sum(
        int(record.get("structured_filtered_detection_count", 0) or 0)
        for record in records
    )
    repaired_tail_detection_count = sum(
        int(record.get("structured_repaired_tail_detection_count", 0) or 0)
        for record in records
    )
    repaired_tail_record_count = sum(
        1 for record in records
        if int(record.get("structured_repaired_tail_detection_count", 0) or 0) > 0
    )
    avg_pred_boxes = (
        sum(int(record["pred_box_count"]) for record in records) / len(records)
        if records else 0.0
    )
    avg_gt_boxes = (
        sum(int(record["gt_box_count"]) for record in records) / len(records)
        if records else 0.0
    )
    overgenerated_count = sum(
        1 for record in records
        if int(record["pred_box_count"]) > int(record["gt_box_count"])
    )
    structured_source_counts: Dict[str, int] = {}
    for record in records:
        source = str(record["structured_source"])
        structured_source_counts[source] = structured_source_counts.get(source, 0) + 1
    budget_hit_count = sum(1 for record in records if record.get("generation_budget_hit"))
    budget_near_hit_count = sum(1 for record in records if record.get("generation_budget_near_hit"))
    vp_count_stopping_available_count = sum(
        1 for record in records if record.get("vp_count_stopping_available")
    )
    vp_count_stopping_triggered_count = sum(
        1 for record in records if record.get("vp_count_stopping_triggered")
    )
    vp_count_stopping_targeted_records = [
        record for record in records
        if record.get("vp_count_stopping_target_boxes") is not None
    ]
    continuation_attempt_count = sum(
        1 for record in records if record.get("vp_continuation_attempted")
    )
    continuation_applied_count = sum(
        1 for record in records if record.get("vp_continuation_applied")
    )
    continuation_reached_count = sum(
        1 for record in records if record.get("vp_continuation_reached_target")
    )
    continuation_stop_triggered_count = sum(
        1 for record in records if record.get("vp_continuation_stop_triggered")
    )
    continuation_count_basis_counts: Dict[str, int] = {}
    continuation_initial_source_counts: Dict[str, int] = {}
    continuation_final_source_counts: Dict[str, int] = {}
    for record in records:
        basis = str(record.get("vp_continuation_count_basis") or "unknown")
        initial_source = str(record.get("vp_continuation_initial_count_source") or "unknown")
        final_source = str(record.get("vp_continuation_final_count_source") or "unknown")
        continuation_count_basis_counts[basis] = continuation_count_basis_counts.get(basis, 0) + 1
        continuation_initial_source_counts[initial_source] = continuation_initial_source_counts.get(initial_source, 0) + 1
        continuation_final_source_counts[final_source] = continuation_final_source_counts.get(final_source, 0) + 1
    dense_records = [
        record for record in records
        if int(record.get("query_box_count") or record.get("gt_box_count") or 0) >= 4
    ]

    def _mean_numeric(records_for_mean: List[Dict[str, Any]], key: str) -> float:
        values = [
            float(record[key])
            for record in records_for_mean
            if record.get(key) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    summary = {
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "input_row_count": len(all_rows),
        "filtered_row_count": len(filtered_rows),
        "min_query_boxes": args.min_query_boxes,
        "max_query_boxes": args.max_query_boxes,
        "text_input_template": args.text_input_template,
        "num_samples": len(records),
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": args.early_stopping,
        "decoder_prefix": args.decoder_prefix,
        "stop_after_vp_max_total_boxes": args.stop_after_vp_max_total_boxes,
        "continue_underfilled_vp_boxes": args.continue_underfilled_vp_boxes,
        "vp_continuation_max_rounds": args.vp_continuation_max_rounds,
        "vp_continuation_max_new_tokens": args.vp_continuation_max_new_tokens,
        "vp_continuation_min_missing_boxes": args.vp_continuation_min_missing_boxes,
        "structured_vp_decode": args.structured_vp_decode,
        "structured_vp_mode": args.structured_vp_mode,
        "structured_vp_box_format": args.structured_vp_box_format,
        "structured_vp_marker_style": args.structured_vp_marker_style,
        "structured_vp_filter_policy": args.structured_vp_filter_policy,
        "structured_vp_max_boxes_per_label": args.structured_vp_max_boxes_per_label,
        "structured_vp_max_total_boxes": args.structured_vp_max_total_boxes,
        "structured_vp_max_total_boxes_field": args.structured_vp_max_total_boxes_field,
        "structured_vp_nms_iou_threshold": args.structured_vp_nms_iou_threshold,
        "structured_vp_allowed_labels": args.structured_vp_allowed_labels,
        "structured_vp_allowed_labels_field": args.structured_vp_allowed_labels_field,
        "structured_vp_allowed_label_match_mode": args.structured_vp_allowed_label_match_mode,
        "structured_vp_repair_malformed_tail": args.structured_vp_repair_malformed_tail,
        "visualization_limit": args.visualization_limit,
        "vp_format_valid_ratio": (vp_valid_count / len(records)) if records else 0.0,
        "structured_vp_format_valid_ratio": (structured_valid_count / len(records)) if records else 0.0,
        "structured_vp_decoder_ratio": (structured_decoder_count / len(records)) if records else 0.0,
        "structured_source_counts": structured_source_counts,
        "native_fallback_ratio": (fallback_count / len(records)) if records else 0.0,
        "avg_pred_boxes": avg_pred_boxes,
        "avg_gt_boxes": avg_gt_boxes,
        "box_count_overgeneration_ratio": (
            overgenerated_count / len(records) if records else 0.0
        ),
        "generation_budget_hit_ratio": (
            budget_hit_count / len(records) if records else 0.0
        ),
        "generation_budget_near_hit_ratio": (
            budget_near_hit_count / len(records) if records else 0.0
        ),
        "vp_count_stopping_available_ratio": (
            vp_count_stopping_available_count / len(records) if records else 0.0
        ),
        "vp_count_stopping_targeted_ratio": (
            len(vp_count_stopping_targeted_records) / len(records) if records else 0.0
        ),
        "vp_count_stopping_triggered_ratio": (
            vp_count_stopping_triggered_count / len(records) if records else 0.0
        ),
        "avg_vp_count_stopping_target_boxes": _mean_numeric(
            vp_count_stopping_targeted_records,
            "vp_count_stopping_target_boxes",
        ),
        "vp_continuation_attempted_ratio": (
            continuation_attempt_count / len(records) if records else 0.0
        ),
        "vp_continuation_applied_ratio": (
            continuation_applied_count / len(records) if records else 0.0
        ),
        "vp_continuation_reached_target_ratio": (
            continuation_reached_count / len(records) if records else 0.0
        ),
        "vp_continuation_stop_triggered_ratio": (
            continuation_stop_triggered_count / len(records) if records else 0.0
        ),
        "vp_continuation_count_basis_counts": continuation_count_basis_counts,
        "vp_continuation_initial_count_source_counts": continuation_initial_source_counts,
        "vp_continuation_final_count_source_counts": continuation_final_source_counts,
        "avg_vp_continuation_added_raw_loc_boxes": _mean_numeric(
            records,
            "vp_continuation_added_raw_loc_boxes",
        ),
        "avg_vp_continuation_initial_parseable_box_count": _mean_numeric(
            records,
            "vp_continuation_initial_parseable_box_count",
        ),
        "avg_vp_continuation_final_parseable_box_count": _mean_numeric(
            records,
            "vp_continuation_final_parseable_box_count",
        ),
        "avg_vp_continuation_added_parseable_boxes": _mean_numeric(
            records,
            "vp_continuation_added_parseable_boxes",
        ),
        "avg_raw_prediction_token_count": _mean_numeric(records, "raw_prediction_token_count"),
        "avg_raw_loc_token_count": _mean_numeric(records, "raw_loc_token_count"),
        "max_raw_loc_token_count": max(
            (int(record.get("raw_loc_token_count") or 0) for record in records),
            default=0,
        ),
        "dense_num_samples": len(dense_records),
        "dense_generation_budget_hit_ratio": (
            sum(1 for record in dense_records if record.get("generation_budget_hit")) / len(dense_records)
            if dense_records else 0.0
        ),
        "dense_generation_budget_near_hit_ratio": (
            sum(1 for record in dense_records if record.get("generation_budget_near_hit")) / len(dense_records)
            if dense_records else 0.0
        ),
        "dense_avg_raw_prediction_token_count": _mean_numeric(dense_records, "raw_prediction_token_count"),
        "dense_avg_raw_loc_token_count": _mean_numeric(dense_records, "raw_loc_token_count"),
        "structured_filtered_detection_count": filtered_detection_count,
        "structured_repaired_tail_detection_count": repaired_tail_detection_count,
        "structured_repaired_tail_record_ratio": (
            repaired_tail_record_count / len(records) if records else 0.0
        ),
        "avg_structured_repaired_tail_detection_count": _mean_numeric(
            records,
            "structured_repaired_tail_detection_count",
        ),
        "records": records,
    }
    summary_path = output_dir / "vp_inference_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest-path", default=".codex_reports/florence_vp_saved_adapter_smoke/vp_real_data_manifest.json")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--data-key",
        default=None,
        help="Optional manifest key to use when --data-path is not set, e.g. val_grounding_path.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_visualizations")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument(
        "--min-query-boxes",
        type=int,
        default=None,
        help="Only run rows whose query_box_count/gt_box_count is at least this value.",
    )
    parser.add_argument(
        "--max-query-boxes",
        type=int,
        default=None,
        help="Only run rows whose query_box_count/gt_box_count is at most this value.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument(
        "--continue-underfilled-vp-boxes",
        action="store_true",
        help=(
            "If the parseable/structured VP output emits fewer boxes than the "
            "resolved max_total_boxes/query count, run a short continuation "
            "decode using a sanitized current output as decoder prefix."
        ),
    )
    parser.add_argument("--vp-continuation-max-rounds", type=int, default=1)
    parser.add_argument("--vp-continuation-max-new-tokens", type=int, default=48)
    parser.add_argument("--vp-continuation-min-missing-boxes", type=int, default=1)
    parser.add_argument(
        "--text-input-template",
        default=None,
        help=(
            "Optional generation-only template for row text_input, e.g. "
            "'all {text_input}' or 'all instances of {label}'. "
            "The original row fields are kept for evaluation."
        ),
    )
    parser.add_argument(
        "--visualization-limit",
        type=int,
        default=None,
        help="Maximum number of PNG visualizations to save. Omit to save all samples.",
    )
    parser.add_argument("--native-fallback", action="store_true")
    parser.add_argument(
        "--structured-vp-decode",
        action="store_true",
        help=(
            "Wrap Florence native label<loc_*> outputs into VP ref/box text before parsing. "
            "This is also attempted when --native-fallback is set."
        ),
    )
    parser.add_argument("--structured-vp-mode", default="off", choices=["off", "auto", "on"])
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="special", choices=["special", "plain"])
    parser.add_argument("--structured-vp-filter-policy", default="none", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--structured-vp-max-boxes-per-label", type=int, default=None)
    parser.add_argument("--structured-vp-max-total-boxes", type=int, default=None)
    parser.add_argument(
        "--structured-vp-max-total-boxes-field",
        default=None,
        help="Optional per-record field used as dynamic max_total_boxes, e.g. query_box_count.",
    )
    parser.add_argument(
        "--stop-after-vp-max-total-boxes",
        action="store_true",
        help=(
            "Stop generation once the raw decoder stream contains the resolved "
            "max_total_boxes worth of <loc_*> box coordinates. The resolved cap "
            "comes from --structured-vp-max-total-boxes or its per-record field."
        ),
    )
    parser.add_argument("--structured-vp-nms-iou-threshold", type=float, default=None)
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument("--structured-vp-allowed-labels-field", default=None)
    parser.add_argument(
        "--structured-vp-allowed-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
        help="Allowed-label filtering mode for structured VP decoding.",
    )
    parser.add_argument(
        "--structured-vp-repair-malformed-tail",
        action="store_true",
        help=(
            "Merge malformed native loc groups after a valid VP box back into "
            "the structured VP output before filtering."
        ),
    )
    parser.add_argument(
        "--decoder-prefix",
        default=None,
        help=(
            "Optional decoder prefix for constrained diagnostics. "
            "Use {label} to insert the first GT label, e.g. '<|ref|>{label}<|/ref|><|box|>'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
