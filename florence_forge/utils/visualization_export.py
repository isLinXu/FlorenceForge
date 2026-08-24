"""Web-ready visualization export utilities.

Convert PIL-based overlays into base64 PNG strings so they can be
embedded directly into JSON API responses or Gradio Image components.

All functions are pure-PIL (no matplotlib) for lightweight server-side
rendering.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

logger = logging.getLogger(__name__)


def _ensure_pil(image: Union[Any, Any]) -> Any:
    """Coerce *image* to a PIL RGB image."""
    if Image is None:
        raise ImportError("PIL is required for visualization export")

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
    except Exception:
        pass
    raise TypeError(f"Expected PIL Image or ndarray, got {type(image).__name__}")


def pil_to_base64(image: Any, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 data URI.

    Returns a string like ``"data:image/png;base64,iVBORw0..."``.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def draw_detections_to_base64(
    image: Union[Any, Any],
    detections: List[Dict[str, Any]],
    colors: Optional[List[str]] = None,
) -> str:
    """Draw bounding boxes + labels on *image* and return a base64 PNG.

    Args:
        image: Input image (PIL or ndarray).
        detections: List of dicts with keys ``bbox`` [x1, y1, x2, y2],
            ``label`` (str), optional ``confidence`` (float).
        colors: Optional list of color names; cycles if fewer than
            detections.

    Returns:
        Base64 data URI string.
    """
    from PIL import ImageDraw, ImageFont

    img = _ensure_pil(image).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    palette = colors or [
        "red", "blue", "green", "yellow", "purple", "orange",
        "pink", "brown", "cyan", "lime",
    ]

    font_size = max(12, img.height // 40)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default(size=font_size)
        except Exception:
            font = ImageFont.load_default()

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        label = det.get("label", "")
        conf = det.get("confidence")
        color = palette[i % len(palette)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.height // 300))
        text = f"{label} ({conf:.2f})" if conf is not None and conf < 1.0 else str(label)
        if text:
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            except Exception:
                tw, th = len(text) * font_size // 2, font_size
            draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
            draw.text((x1, y1 - th), text, fill="white", font=font)

    return pil_to_base64(img)


def draw_caption_to_base64(
    image: Union[Any, Any],
    caption: str,
) -> str:
    """Overlay a caption banner at the top of *image* and return base64 PNG."""
    from PIL import ImageDraw, ImageFont
    import textwrap

    img = _ensure_pil(image).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(24, img.height // 25)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default(size=font_size)
        except Exception:
            font = ImageFont.load_default()

    caption = caption.replace("</s>", "").replace("<s>", "").strip()
    if not caption:
        return pil_to_base64(img)

    try:
        avg_w = font.getlength("a")
    except Exception:
        avg_w = font_size * 0.6
    wrap_w = int(img.width / avg_w * 1.2) if avg_w > 0 else 60
    wrapped = textwrap.fill(caption, width=wrap_w)

    try:
        bbox = draw.textbbox((0, 0), wrapped, font=font)
        th = bbox[3] - bbox[1]
    except Exception:
        th = font_size * 2

    draw.rectangle([(0, 0), (img.width, th + 20)], fill=(255, 255, 0, 180))
    draw.text((10, 10), wrapped, fill=(0, 0, 0), font=font)
    return pil_to_base64(img)


def draw_ocr_to_base64(
    image: Union[Any, Any],
    ocr_results: List[Dict[str, Any]],
) -> str:
    """Draw OCR polygons + text on *image* and return base64 PNG.

    *ocr_results* items should have keys ``polygon`` (list of [x, y])
    and ``text`` (str).
    """
    from PIL import ImageDraw, ImageFont

    img = _ensure_pil(image).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(12, img.height // 50)
    current_dir = Path(__file__).parent
    font_path = current_dir.parent.parent / "assets" / "fonts" / "SourceHanSansSC-Regular.ttf"
    font: Any = None
    for fp in (font_path, "SimHei.ttf", "/System/Library/Fonts/STHeitiLight.ttc", "arial.ttf"):
        try:
            font = ImageFont.truetype(str(fp), font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for result in ocr_results:
        polygon = result.get("polygon", [])
        text = result.get("text", "")
        if len(polygon) >= 3:
            draw.polygon(polygon, outline="lime", fill=(0, 255, 0, 60))
        if polygon and text:
            pos = polygon[0]
            draw.text(pos, text, fill="red", font=font)

    return pil_to_base64(img)


def draw_segmentation_to_base64(
    image: Union[Any, Any],
    polygons: List[List[Tuple[int, int]]],
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> str:
    """Draw segmentation polygons on *image* and return base64 PNG."""
    from PIL import Image, ImageDraw

    base = _ensure_pil(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for poly in polygons:
        if len(poly) > 2:
            draw.polygon(poly, fill=color + (int(255 * alpha),), outline=color)
    blended = Image.alpha_composite(base, overlay)
    return pil_to_base64(blended.convert("RGB"))


def generate_step_visualization(
    image: Union[Any, Any],
    step_record: Dict[str, Any],
) -> Optional[str]:
    """Generate a base64 visualization for a single orchestrator step.

    Inspects the step's ``tool_call.task_name`` and ``parsed`` fields to
    pick the appropriate drawer.

    Returns:
        Base64 data URI or ``None`` if no visual overlay is applicable.
    """
    task_name = step_record.get("tool_call", {}).get("task_name", "")
    parsed = step_record.get("parsed", {})

    if task_name in ("OD", "OPEN_VOCABULARY_DETECTION", "CAPTION_TO_PHRASE_GROUNDING"):
        boxes = parsed.get("boxes", [])
        if boxes:
            if isinstance(boxes[0], dict):
                detections = [{"bbox": b, "label": b.get("label", "")} for b in boxes]
            else:
                detections = [{"bbox": b, "label": ""} for b in boxes]
            return draw_detections_to_base64(image, detections)

    if task_name in ("OCR", "OCR_WITH_REGION"):
        ocr_results = parsed.get("ocr_results", [])
        if not ocr_results and parsed.get("text"):
            # Fallback: create a single full-image text region
            img = _ensure_pil(image)
            ocr_results = [{"polygon": [(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)], "text": parsed["text"]}]
        if ocr_results:
            return draw_ocr_to_base64(image, ocr_results)

    if task_name in ("CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"):
        text = parsed.get("text", "")
        if text:
            return draw_caption_to_base64(image, text)

    if task_name in ("REGION_TO_SEGMENTATION", "REFERRING_EXPRESSION_SEGMENTATION"):
        polygons = parsed.get("polygons", [])
        if polygons:
            return draw_segmentation_to_base64(image, polygons)

    return None
