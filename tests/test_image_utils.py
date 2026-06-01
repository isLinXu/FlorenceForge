"""Image utility regression tests."""

import numpy as np
import pytest
import torch
from PIL import Image

from florence_forge.utils.image import (
    ImageProcessor,
    apply_augmentation,
    create_image_grid,
    crop_image,
    denormalize_image,
    draw_bounding_boxes,
    draw_segmentation_mask,
    load_image,
    normalize_image,
    pad_image,
    resize_image,
    save_image,
    tensor_to_pil,
)


def test_resize_image_clamps_scaled_dimension_to_one_pixel():
    image = Image.new("RGB", (1, 100))

    resized = resize_image(image, 2, maintain_aspect=True)

    assert resized.size == (1, 2)


def test_resize_image_rejects_non_positive_target_size():
    image = Image.new("RGB", (10, 10))

    with pytest.raises(ValueError, match="正数"):
        resize_image(image, 0)

    with pytest.raises(ValueError, match="正数"):
        resize_image(image, (0, 10), maintain_aspect=False)


def test_resize_image_tuple_aspect_ratio_keeps_at_least_one_pixel():
    image = Image.new("RGB", (100, 1))

    resized = resize_image(image, (2, 2), maintain_aspect=True)

    assert resized.size == (2, 1)


def test_denormalize_image_preserves_tensor_dtype():
    tensor = torch.zeros((3, 2, 2), dtype=torch.float64)

    restored = denormalize_image(
        tensor,
        mean=[0.1, 0.2, 0.3],
        std=[1.0, 1.0, 1.0],
    )

    assert restored.dtype == torch.float64
    assert restored[:, 0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_tensor_to_pil_accepts_requires_grad_tensor():
    tensor = torch.ones((3, 4, 5), requires_grad=True)

    image = tensor_to_pil(tensor)

    assert image.size == (5, 4)


def test_tensor_to_pil_rejects_multi_sample_batch():
    tensor = torch.zeros((2, 3, 4, 5))

    with pytest.raises(ValueError, match="单张图像"):
        tensor_to_pil(tensor)


# ---------------------------------------------------------------------------
# load_image / save_image
# ---------------------------------------------------------------------------


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="不存在"):
        load_image(tmp_path / "nope.png")


def test_load_image_converts_mode(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (8, 8), color=128).save(path)

    image = load_image(path, mode="RGB")

    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_load_image_wraps_decode_errors(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not really a png")

    with pytest.raises(ValueError, match="无法加载图像"):
        load_image(path)


@pytest.mark.parametrize(
    "name,expected_format",
    [("out.png", "PNG"), ("out.jpg", "JPEG"), ("out.webp", "WEBP"), ("out.xyz", "JPEG")],
)
def test_save_image_picks_format_from_extension(tmp_path, name, expected_format):
    image = Image.new("RGB", (16, 16), color=(10, 20, 30))
    target = tmp_path / "nested" / name

    save_image(image, target, quality=80)

    assert target.exists()
    with Image.open(target) as reloaded:
        assert reloaded.format == expected_format


# ---------------------------------------------------------------------------
# crop / pad
# ---------------------------------------------------------------------------


def test_crop_image_returns_region():
    image = Image.new("RGB", (10, 10))
    cropped = crop_image(image, (1, 2, 5, 8))
    assert cropped.size == (4, 6)


def test_pad_image_returns_original_when_already_large_enough():
    image = Image.new("RGB", (20, 20))
    assert pad_image(image, (10, 10)) is image


@pytest.mark.parametrize(
    "position,expected_offset",
    [
        ("center", (3, 3)),
        ("top-left", (0, 0)),
        ("top-right", (6, 0)),
        ("bottom-left", (0, 6)),
        ("bottom-right", (6, 6)),
        ("unknown", (0, 0)),
    ],
)
def test_pad_image_positions_paste_origin(position, expected_offset):
    image = Image.new("RGB", (4, 4), color=(255, 255, 255))
    padded = pad_image(image, (10, 10), fill_color=(0, 0, 0), position=position)

    assert padded.size == (10, 10)
    x, y = expected_offset
    assert padded.getpixel((x, y)) == (255, 255, 255)


# ---------------------------------------------------------------------------
# normalize / denormalize
# ---------------------------------------------------------------------------


def test_normalize_image_from_pil_uses_default_imagenet_stats():
    image = Image.new("RGB", (4, 4), color=(255, 255, 255))
    tensor = normalize_image(image)
    expected_r = (1.0 - 0.485) / 0.229
    assert tensor.shape == (3, 4, 4)
    assert tensor[0, 0, 0].item() == pytest.approx(expected_r, rel=1e-3)


def test_normalize_image_from_numpy_hwc():
    array = np.full((4, 4, 3), 255, dtype=np.uint8)
    tensor = normalize_image(array, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    assert tensor.shape == (3, 4, 4)
    assert tensor.max().item() == pytest.approx(1.0)


def test_normalize_image_from_tensor_passthrough():
    chw = torch.ones((3, 2, 2))
    tensor = normalize_image(chw, mean=[1.0, 1.0, 1.0], std=[1.0, 1.0, 1.0])
    assert torch.allclose(tensor, torch.zeros_like(chw))


def test_normalize_image_rejects_unsupported_type():
    with pytest.raises(ValueError, match="不支持的图像类型"):
        normalize_image("not-an-image")


def test_normalize_denormalize_roundtrip():
    image = Image.new("RGB", (4, 4), color=(120, 130, 140))
    normalized = normalize_image(image)
    restored = denormalize_image(normalized)
    restored_uint8 = (torch.clamp(restored, 0, 1) * 255).round()
    assert restored_uint8[0, 0, 0].item() == pytest.approx(120, abs=1)


# ---------------------------------------------------------------------------
# tensor_to_pil edge shapes
# ---------------------------------------------------------------------------


def test_tensor_to_pil_accepts_single_sample_4d():
    tensor = torch.ones((1, 3, 4, 5))
    image = tensor_to_pil(tensor)
    assert image.size == (5, 4)


def test_tensor_to_pil_accepts_2d_grayscale():
    tensor = torch.ones((4, 5))
    image = tensor_to_pil(tensor)
    assert image.size == (5, 4)


def test_tensor_to_pil_rejects_invalid_dim():
    with pytest.raises(ValueError, match="2D/3D"):
        tensor_to_pil(torch.ones((2, 3, 4, 5, 6)))


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def test_draw_bounding_boxes_does_not_mutate_input():
    image = Image.new("RGB", (40, 40), color=(0, 0, 0))
    drawn = draw_bounding_boxes(
        image,
        boxes=[[5, 5, 20, 20], [10, 10, 30, 30]],
        labels=["cat", "dog"],
        scores=[0.9, 0.5],
    )
    assert drawn is not image
    assert drawn.size == image.size


def test_draw_segmentation_mask_blends_color():
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    blended = draw_segmentation_mask(image, mask, alpha=0.5, color=(255, 0, 0))
    assert blended.getpixel((0, 0))[0] > 0


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------


def test_image_processor_process_and_roundtrip(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(100, 110, 120)).save(path)
    processor = ImageProcessor(target_size=(4, 4), normalize=True)

    tensor = processor.process_image(path)
    assert tensor.shape == (3, 4, 4)

    batched = processor.preprocess_for_model(path, add_batch_dim=True)
    assert batched.shape == (1, 3, 4, 4)

    restored = processor.postprocess_from_model(batched, remove_batch_dim=True)
    assert restored.size == (4, 4)


def test_image_processor_process_batch_stacks(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (8, 8), color=(i, i, i)).save(p)
        paths.append(p)
    processor = ImageProcessor(target_size=(4, 4), normalize=False)

    batch = processor.process_batch(paths)
    assert batch.shape == (3, 3, 4, 4)


# ---------------------------------------------------------------------------
# create_image_grid / apply_augmentation
# ---------------------------------------------------------------------------


def test_create_image_grid_auto_layout():
    images = [Image.new("RGB", (4, 4), color=(i * 20, 0, 0)) for i in range(3)]
    grid = create_image_grid(images, spacing=1)
    # auto layout: cols=ceil(sqrt(3))=2, rows=2 -> 2*4+1 x 2*4+1
    assert grid.size == (9, 9)


def test_create_image_grid_empty_raises():
    with pytest.raises(ValueError, match="不能为空"):
        create_image_grid([])


def test_create_image_grid_explicit_layout_and_resize():
    images = [Image.new("RGB", (6, 6)) for _ in range(2)]
    grid = create_image_grid(images, grid_size=(1, 2), image_size=(4, 4), spacing=0)
    assert grid.size == (8, 4)


@pytest.mark.parametrize(
    "aug_type,kwargs",
    [
        ("rotation", {"angle": 90}),
        ("flip", {"direction": "horizontal"}),
        ("flip", {"direction": "vertical"}),
        ("brightness", {"factor": 1.5}),
        ("contrast", {"factor": 0.8}),
        ("saturation", {"factor": 1.2}),
    ],
)
def test_apply_augmentation_returns_image(aug_type, kwargs):
    image = Image.new("RGB", (8, 8), color=(50, 60, 70))
    result = apply_augmentation(image, aug_type, **kwargs)
    assert isinstance(result, Image.Image)


def test_apply_augmentation_rejects_unknown_type():
    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="不支持的增强类型"):
        apply_augmentation(image, "warp")
