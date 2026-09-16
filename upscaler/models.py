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
    "realesrgan-x4plus": ModelInfo(
        name="realesrgan-x4plus",
        filename="RealESRGAN_x4plus.onnx",
        scale=4,
        architecture="RRDBNet",
        size_mb=67.0,
        url="https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/01179a4da7bf5ac91faca650e6afbf282ac93933/Real-ESRGAN-x4plus.onnx",
        description="General 4x upscaler — photos and mixed content",
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
    "realesrgan-x4plus-anime": ModelInfo(
        name="realesrgan-x4plus-anime",
        filename="RealESRGAN_x4plus_anime_6B.onnx",
        scale=4,
        architecture="RRDBNet",
        size_mb=18.0,
        url=None,
        description="Anime-optimised 4x upscaler",
    ),
}

FRIENDLY_NAMES: dict[str, str] = {
    "realesrgan-x4plus": "General (RealESRGAN x4plus)",
    "realesr-animevideov3": "Animation (animevideov3)",
    "realesrgan-x4plus-anime": "Anime (RealESRGAN x4plus anime)",
}

DEFAULT_ONNX_MODEL = "realesrgan-x4plus"


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

    log(f"Downloading {info.filename} ({info.size_mb:.0f} MB)...")
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


def list_available() -> list[tuple[str, ModelInfo, bool]]:
    result = []
    for name, info in KNOWN_MODELS.items():
        installed = (user_models_dir() / info.filename).exists()
        result.append((name, info, installed))
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
            result.append((p.stem, custom, True))
    return result
