from __future__ import annotations

from typing import Callable

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
AI_2X_MODEL = "realesr-animevideov3"
INTERPOLATE_FPS = 60.0
INTERPOLATE_FPS_TOLERANCE = 0.5

ProgressCallback = Callable[[str, int | None, int | None], None]
