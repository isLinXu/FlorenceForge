"""Environment diagnostics for FlorenceForge.

The checks here are intentionally lightweight: they do not load model weights
or import optional heavy modules unless a module spec/version lookup is enough.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_MODEL_ID = "microsoft/Florence-2-base"


@dataclass(frozen=True)
class DependencyCheck:
    module: str
    package: str
    required: bool = True
    purpose: str = ""


CORE_DEPENDENCIES = (
    DependencyCheck("torch", "torch", True, "model runtime"),
    DependencyCheck("transformers", "transformers", True, "model loading"),
    DependencyCheck("PIL", "Pillow", True, "image IO"),
    DependencyCheck("pydantic", "pydantic", True, "configuration"),
    DependencyCheck("yaml", "PyYAML", True, "YAML configuration"),
    DependencyCheck("peft", "peft", False, "LoRA fine-tuning"),
    DependencyCheck("accelerate", "accelerate", False, "distributed training"),
    DependencyCheck("fastapi", "fastapi", False, "serving"),
    DependencyCheck("uvicorn", "uvicorn", False, "serving"),
    DependencyCheck("pycocotools", "pycocotools", False, "COCO metrics"),
    DependencyCheck("rouge_score", "rouge-score", False, "caption metrics"),
    DependencyCheck("cv2", "opencv-python", False, "segmentation metrics"),
)


def module_status(check: DependencyCheck) -> Dict[str, Any]:
    """Return availability and installed version for a dependency."""
    available = importlib.util.find_spec(check.module) is not None
    version: Optional[str] = None
    if available:
        try:
            version = metadata.version(check.package)
        except metadata.PackageNotFoundError:
            version = None

    return {
        "module": check.module,
        "package": check.package,
        "available": available,
        "version": version,
        "required": check.required,
        "purpose": check.purpose,
    }


def find_local_hf_snapshot(
    model_id: str = DEFAULT_MODEL_ID,
    cache_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find a local Hugging Face snapshot for a model id."""
    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = root / ("models--" + model_id.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    snapshots = model_dir / "snapshots"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots / revision
        if snapshot.exists():
            return snapshot

    if snapshots.exists():
        candidates = sorted(
            [path for path in snapshots.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _choose_device(requested_device: str, torch_module: Any) -> str:
    if requested_device != "auto":
        return requested_device
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _device_available(device: str, torch_module: Any) -> bool:
    if device == "cpu":
        return True
    if device.startswith("cuda"):
        if not torch_module.cuda.is_available():
            return False
        if ":" in device:
            try:
                return int(device.split(":", 1)[1]) < torch_module.cuda.device_count()
            except ValueError:
                return False
        return True
    if device == "mps":
        return hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    return False


def _torch_diagnostics(requested_device: str) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "version": None,
            "selected_device": "cpu",
            "selected_device_available": True,
            "mps_available": False,
            "mps_built": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
        }

    mps_backend = getattr(torch.backends, "mps", None)
    selected_device = _choose_device(requested_device, torch)
    return {
        "available": True,
        "version": torch.__version__,
        "selected_device": selected_device,
        "selected_device_available": _device_available(selected_device, torch),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "mps_built": bool(mps_backend and mps_backend.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


def _model_snapshot_status(
    model_id: str,
    model_path: Optional[str],
    cache_root: Optional[Path],
) -> Dict[str, Any]:
    if model_path:
        path = Path(model_path).expanduser()
        return {
            "model_id": model_id,
            "model_path": str(path),
            "local_snapshot": str(path) if path.exists() else None,
            "local_snapshot_exists": path.exists(),
        }

    snapshot = find_local_hf_snapshot(model_id, cache_root=cache_root)
    return {
        "model_id": model_id,
        "model_path": None,
        "local_snapshot": str(snapshot) if snapshot else None,
        "local_snapshot_exists": snapshot is not None,
    }


def collect_environment_diagnostics(
    requested_device: str = "auto",
    model_id: str = DEFAULT_MODEL_ID,
    model_path: Optional[str] = None,
    require_model: bool = False,
    cache_root: Optional[Path] = None,
    checks: Iterable[DependencyCheck] = CORE_DEPENDENCIES,
) -> Dict[str, Any]:
    """Collect lightweight runtime diagnostics for local development."""
    dependencies = [module_status(check) for check in checks]
    missing_required = [
        dep["package"]
        for dep in dependencies
        if dep["required"] and not dep["available"]
    ]
    torch_info = _torch_diagnostics(requested_device)
    model_info = _model_snapshot_status(model_id, model_path, cache_root)

    warnings = []
    if not torch_info["available"]:
        warnings.append("PyTorch is not installed.")
    elif not torch_info["selected_device_available"]:
        warnings.append("Requested device %s is not available." % torch_info["selected_device"])

    if require_model and not model_info["local_snapshot_exists"]:
        warnings.append("Required local model snapshot is missing.")

    recommended_dtype = "float32" if torch_info["selected_device"] == "mps" else "auto"
    smoke_device = torch_info["selected_device"] if torch_info["selected_device_available"] else "cpu"
    smoke_command = (
        "python scripts/smoke/real_florence_mps_smoke.py "
        "--mode forward --device %s --max-new-tokens 8" % smoke_device
    )

    ok = not missing_required and torch_info["selected_device_available"]
    if require_model:
        ok = ok and model_info["local_snapshot_exists"]

    return {
        "ok": ok,
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": torch_info,
        "dependencies": dependencies,
        "missing_required": missing_required,
        "model": model_info,
        "recommended_torch_dtype": recommended_dtype,
        "suggested_smoke_command": smoke_command,
        "warnings": warnings,
    }
