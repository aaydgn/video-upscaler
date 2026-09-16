from __future__ import annotations

from pathlib import Path
from typing import Callable

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
AI_2X_MODEL = "realesr-animevideov3"
INTERPOLATE_FPS = 60.0
INTERPOLATE_FPS_TOLERANCE = 0.5

ONNX_TILE_SIZE = 0
ONNX_TILE_PAD = 10

ProgressCallback = Callable[[str, int | None, int | None], None]


def user_models_dir() -> Path:
    d = Path.home() / ".video-upscaler" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
