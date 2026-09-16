from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from upscaler.config import user_models_dir


@dataclass(frozen=True)
class ModelInfo:
    name: str
    filename: str
    scale: int
    architecture: str
    size_mb: float
    url: str | None
    description: str


KNOWN_MODELS: dict[str, ModelInfo] = {
    "2xHFA2kSPAN": ModelInfo(
        name="2xHFA2kSPAN",
        filename="2xHFA2kSPAN.onnx",
        scale=2,
        architecture="SPAN",
        size_mb=1.6,
        url="https://github.com/Phhofm/models/releases/download/2xHFA2kSPAN/2xHFA2kSPAN_fp32_opset17.onnx",
        description="Fast 2x anime upscaler (SPAN)",
    ),
    "4xNomosUni_span_multijpg": ModelInfo(
        name="4xNomosUni_span_multijpg",
        filename="4xNomosUni_span_multijpg.onnx",
        scale=4,
        architecture="SPAN",
        size_mb=1.6,
        url=None,
        description="Fast 4x universal upscaler (SPAN)",
    ),
    "realesr-animevideov3": ModelInfo(
        name="realesr-animevideov3",
        filename="realesr-animevideov3.onnx",
        scale=4,
        architecture="SRVGGNetCompact",
        size_mb=2.4,
        url=None,
        description="Fast 4x upscaler for animated video",
    ),
    "realesrgan-x4plus": ModelInfo(
        name="realesrgan-x4plus",
        filename="RealESRGAN_x4plus.onnx",
        scale=4,
        architecture="RRDBNet",
        size_mb=64.0,
        url=None,
        description="General 4x upscaler — photos and mixed content",
    ),
}

FRIENDLY_NAMES: dict[str, str] = {
    "2xHFA2kSPAN": "SPAN 2x Anime",
    "4xNomosUni_span_multijpg": "SPAN 4x Universal",
    "realesr-animevideov3": "Compact 4x Animation",
    "realesrgan-x4plus": "ESRGAN 4x General",
}

DEFAULT_ONNX_2X = "2xHFA2kSPAN"
DEFAULT_ONNX_4X = "4xNomosUni_span_multijpg"
DEFAULT_ONNX_MODEL = DEFAULT_ONNX_4X


def default_for_scale(scale: int) -> str:
    if scale <= 2:
        return DEFAULT_ONNX_2X
    return DEFAULT_ONNX_4X


def find_model(name_or_path: str) -> Path | None:
    p = Path(name_or_path)
    if p.suffix == ".onnx" and p.exists():
        return p

    info = KNOWN_MODELS.get(name_or_path)
    if info is not None:
        path = user_models_dir() / info.filename
        if path.exists():
            return path
        return None

    candidate = user_models_dir() / name_or_path
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        candidate = candidate.with_suffix(".onnx")
        if candidate.exists():
            return candidate
    return None


def download_model(
    name: str,
    log: Callable[[str], None] = print,
) -> Path:
    info = KNOWN_MODELS.get(name)
    if info is None:
        raise ValueError(
            f"Unknown model: {name}. Known: {', '.join(KNOWN_MODELS)}"
        )
    if info.url is None:
        raise ValueError(
            f"No download URL for {name}. Convert from .pth using scripts/convert_to_onnx.py "
            f"and place the .onnx file in {user_models_dir()}"
        )

    dest = user_models_dir() / info.filename
    if dest.exists():
        log(f"Model already downloaded: {dest}")
        return dest

    log(f"Downloading {info.filename} ({info.size_mb:.1f} MB)...")
    tmp = dest.with_suffix(".onnx.tmp")
    try:
        def _progress(block: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                sys.stderr.write(f"\r  {pct}%")
                sys.stderr.flush()

        urllib.request.urlretrieve(info.url, tmp, reporthook=_progress)
        sys.stderr.write("\n")
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log(f"Saved to {dest}")
    return dest


def ensure_model(
    name_or_path: str,
    log: Callable[[str], None] = print,
) -> Path:
    found = find_model(name_or_path)
    if found is not None:
        return found
    if name_or_path in KNOWN_MODELS:
        return download_model(name_or_path, log)
    raise FileNotFoundError(
        f"ONNX model not found: {name_or_path}\n"
        f"Place an .onnx file in {user_models_dir()}\n"
        f"Or use a known model: {', '.join(KNOWN_MODELS)}"
    )


def list_installed() -> list[tuple[str, ModelInfo]]:
    result = []
    for name, info in KNOWN_MODELS.items():
        if (user_models_dir() / info.filename).exists():
            result.append((name, info))
    for p in user_models_dir().glob("*.onnx"):
        if not any(info.filename == p.name for info in KNOWN_MODELS.values()):
            custom = ModelInfo(
                name=p.stem,
                filename=p.name,
                scale=4,
                architecture="unknown",
                size_mb=p.stat().st_size / 1024 / 1024,
                url=None,
                description="Custom ONNX model",
            )
            result.append((p.stem, custom))
    return result
