#!/usr/bin/env python3
"""Convert Pascal VOC detection splits into Florence OD JSONL files.

Default split strategy:
- train: VOC2007 trainval + VOC2012 trainval
- val:   VOC2007 test
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

_defusedxml = importlib.util.find_spec("defusedxml.ElementTree")
if _defusedxml is not None:  # pragma: no branch
    ET = importlib.import_module("defusedxml.ElementTree")
else:  # pragma: no cover - optional hardening dependency
    ET = importlib.import_module("xml.etree.ElementTree")

from florence_forge.core.visual_primitives import normalize_bbox
from florence_forge.data.converter_od import _format_florence_od_suffix


def _read_split_file(split_file: Path) -> List[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_xml_files(voc_dir: Path, split_names: Sequence[str]) -> Iterable[Path]:
    annotations_dir = voc_dir / "Annotations"
    for sample_id in split_names:
        xml_path = annotations_dir / f"{sample_id}.xml"
        if xml_path.exists():
            yield xml_path


def _convert_xml_file(xml_path: Path, image_dir: Path, task_type: str) -> dict | None:
    root = ET.parse(xml_path).getroot()
    filename_node = root.find("filename")
    size_node = root.find("size")
    if filename_node is None or size_node is None:
        return None

    filename = filename_node.text
    width = int(size_node.findtext("width", "0"))
    height = int(size_node.findtext("height", "0"))
    if not filename or width <= 0 or height <= 0:
        return None

    labels = []
    bboxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name")
        bbox = obj.find("bndbox")
        if not label or bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin", "0")))
        ymin = int(float(bbox.findtext("ymin", "0")))
        xmax = int(float(bbox.findtext("xmax", "0")))
        ymax = int(float(bbox.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue

        labels.append(label)
        bboxes.append(
            normalize_bbox(
                [xmin, ymin, xmax, ymax],
                (width, height),
                input_format="xyxy",
            )
        )

    if not labels:
        return None

    image_path = image_dir / filename
    return {
        "image": str(image_path.resolve()),
        "xml_file": str(xml_path.resolve()),
        "prefix": f"<{task_type}>",
        "suffix": _format_florence_od_suffix(labels, bboxes),
    }


def _write_split(
    voc_roots: Sequence[Path],
    split_files: Sequence[Path],
    output_path: Path,
    task_type: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for voc_root, split_file in zip(voc_roots, split_files):
            split_names = _read_split_file(split_file)
            image_dir = voc_root / "JPEGImages"
            for xml_path in _iter_xml_files(voc_root, split_names):
                sample = _convert_xml_file(xml_path, image_dir, task_type)
                if sample is None:
                    continue
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    return total


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-root",
        default="/home/linxu/Downloads/VOC/VOCdevkit",
        help="VOCdevkit root directory",
    )
    parser.add_argument(
        "--output-train",
        default="/home/linxu/PycharmProjects/FlorenceForge/data/voc_od_train.jsonl",
        help="Output JSONL for training split",
    )
    parser.add_argument(
        "--output-val",
        default="/home/linxu/PycharmProjects/FlorenceForge/data/voc_od_val.jsonl",
        help="Output JSONL for validation split",
    )
    parser.add_argument(
        "--task-type",
        default="OD",
        help="Florence task type",
    )
    parser.add_argument(
        "--train-scheme",
        choices=["full", "voc2007"],
        default="full",
        help="Training split scheme: full=VOC2007+VOC2012, voc2007=VOC2007 only",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    voc_root = Path(args.voc_root).expanduser().resolve()

    voc2007 = voc_root / "VOC2007"
    voc2012 = voc_root / "VOC2012"
    if args.train_scheme == "voc2007":
        train_voc_roots = [voc2007]
        train_split_files = [voc2007 / "ImageSets" / "Main" / "trainval.txt"]
    else:
        train_voc_roots = [voc2007, voc2012]
        train_split_files = [
            voc2007 / "ImageSets" / "Main" / "trainval.txt",
            voc2012 / "ImageSets" / "Main" / "trainval.txt",
        ]

    train_count = _write_split(
        voc_roots=train_voc_roots,
        split_files=train_split_files,
        output_path=Path(args.output_train).expanduser().resolve(),
        task_type=args.task_type,
    )
    val_count = _write_split(
        voc_roots=[voc2007],
        split_files=[voc2007 / "ImageSets" / "Main" / "test.txt"],
        output_path=Path(args.output_val).expanduser().resolve(),
        task_type=args.task_type,
    )

    print(
        json.dumps(
            {
                "train_samples": train_count,
                "val_samples": val_count,
                "train_scheme": args.train_scheme,
                "output_train": str(Path(args.output_train).expanduser().resolve()),
                "output_val": str(Path(args.output_val).expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
