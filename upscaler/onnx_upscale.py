from __future__ import annotations

import math
from typing import Callable

import numpy as np

from upscaler.config import ONNX_TILE_PAD, ONNX_TILE_SIZE


def _available_providers() -> list[str]:
    import onnxruntime as ort
    return ort.get_available_providers()


def select_provider() -> str:
    providers = _available_providers()
    for ep in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if ep in providers:
            return ep
    return "CPUExecutionProvider"


def create_session(
    model_path: str | object,
    log: Callable[[str], None] = print,
) -> object:
    import onnxruntime as ort

    provider = select_provider()
    log(f"ONNX Runtime: {ort.__version__}, provider: {provider}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = False

    provider_opts: list[dict] = [{}]
    if provider == "DmlExecutionProvider":
        provider_opts = [{"performance_preference": "high_performance"}]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=[(provider, provider_opts[0]), "CPUExecutionProvider"],
    )
    actual = session.get_providers()
    log(f"Active providers: {actual}")
    return session


def detect_scale(session: object, test_h: int = 64, test_w: int = 64) -> int:
    in_meta = session.get_inputs()[0]
    out_name = session.get_outputs()[0].name
    dummy = np.random.rand(1, 3, test_h, test_w).astype(np.float32)
    result = session.run([out_name], {in_meta.name: dummy})[0]
    return result.shape[2] // test_h


def _use_fp16(session: object) -> bool:
    in_type = session.get_inputs()[0].type
    return "float16" in in_type


def _infer(session: object, frame: np.ndarray) -> np.ndarray:
    in_meta = session.get_inputs()[0]
    out_name = session.get_outputs()[0].name

    tensor = (frame.astype(np.float32) / 255.0)
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    if _use_fp16(session):
        tensor = tensor.astype(np.float16)

    result = session.run([out_name], {in_meta.name: tensor})[0]
    return np.clip(
        result.squeeze(0).transpose(1, 2, 0) * 255, 0, 255
    ).astype(np.uint8)


def upscale_frame(
    session: object,
    frame: np.ndarray,
    scale: int,
    tile_size: int = ONNX_TILE_SIZE,
    tile_pad: int = ONNX_TILE_PAD,
) -> np.ndarray:
    if tile_size <= 0:
        return _infer(session, frame)
    return _tile_upscale(session, frame, scale, tile_size, tile_pad)


def _tile_upscale(
    session: object,
    img: np.ndarray,
    scale: int,
    tile_size: int,
    tile_pad: int,
) -> np.ndarray:
    h, w, c = img.shape
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.uint8)

    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            xp0 = max(x0 - tile_pad, 0)
            yp0 = max(y0 - tile_pad, 0)
            xp1 = min(x1 + tile_pad, w)
            yp1 = min(y1 + tile_pad, h)

            tile = img[yp0:yp1, xp0:xp1, :]
            tile_out = _infer(session, tile)

            crop_top = (y0 - yp0) * scale
            crop_left = (x0 - xp0) * scale
            crop_h = (y1 - y0) * scale
            crop_w = (x1 - x0) * scale

            output[
                y0 * scale : y0 * scale + crop_h,
                x0 * scale : x0 * scale + crop_w,
                :,
            ] = tile_out[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w, :]

    return output
