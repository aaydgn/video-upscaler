from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from upscaler.config import ONNX_TILE_SIZE, ProgressCallback
from upscaler.ffmpeg import encoder_args, probe_video, select_video_encoder
from upscaler.onnx_upscale import FrameUpscaler, create_session, detect_scale
from upscaler.process import active_subprocess, report_progress
from upscaler.tools import ToolInfo


def run_onnx_pipeline(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model_path: Path,
    scale: int,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    fps: float,
    target_width: int | None,
    target_height: int | None,
    tile_size: int,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    width, height, duration, _ = probe_video(tools.ffprobe, input_path)
    if not width or not height:
        log("Error: could not determine input dimensions.")
        return 1

    report_progress("Loading ONNX model", None, None, log, progress)
    session = create_session(model_path, log)

    model_scale = detect_scale(session)
    if model_scale != scale:
        log(
            f"Model produces {model_scale}x but {scale}x was requested. "
            f"Will upscale {model_scale}x then resize to {scale}x."
        )
        if target_width is None:
            target_width = width * scale
            target_height = height * scale

    out_w = width * model_scale
    out_h = height * model_scale
    frame_bytes_in = width * height * 3
    if target_width and target_height:
        log(f"Pipeline: {width}x{height} -> {out_w}x{out_h} ({model_scale}x) -> {target_width}x{target_height}")
    else:
        log(f"Pipeline: {width}x{height} -> {out_w}x{out_h} ({model_scale}x)")
    scale = model_scale

    batch_size = 4
    upscaler = FrameUpscaler(session, width, height, scale, batch_size=batch_size)
    total_frames = int(duration * fps) if duration and fps else None

    decode_cmd = _build_decode_cmd(tools, input_path, enhance)
    encode_cmd = _build_encode_cmd(
        tools, input_path, output_path,
        out_w, out_h, fps, codec, quality, overwrite,
        enhance, target_width, target_height,
    )

    log(subprocess.list2cmdline(decode_cmd))
    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    active_subprocess[0] = decoder

    log(subprocess.list2cmdline(encode_cmd))
    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    enc_lines: list[str] = []

    def _drain_encoder_stderr() -> None:
        assert encoder.stderr is not None
        for raw in encoder.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                enc_lines.append(line)

    drain_thread = threading.Thread(target=_drain_encoder_stderr, daemon=True)
    drain_thread.start()

    frame_count = 0
    try:
        assert decoder.stdout is not None
        assert encoder.stdin is not None
        while True:
            batch: list[np.ndarray] = []
            for _ in range(batch_size):
                raw = decoder.stdout.read(frame_bytes_in)
                if len(raw) < frame_bytes_in:
                    break
                batch.append(
                    np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                )
            if not batch:
                break

            if len(batch) == 1:
                results = [upscaler.upscale(batch[0])]
            else:
                results = upscaler.upscale_batch(batch)

            for upscaled in results:
                encoder.stdin.write(upscaled.tobytes())

            frame_count += len(batch)
            if frame_count % 10 < batch_size or frame_count == total_frames:
                report_progress(
                    "ONNX upscaling", frame_count, total_frames, log, progress,
                )
    except BrokenPipeError:
        log("Encoder pipe closed early.")
    finally:
        active_subprocess[0] = None
        if decoder.stdout:
            decoder.stdout.close()
        decoder.wait()
        if encoder.stdin:
            try:
                encoder.stdin.close()
            except OSError:
                pass
        encoder.wait()
        drain_thread.join(timeout=5)

    for line in enc_lines[-20:]:
        log(f"[encode] {line}")

    if encoder.returncode != 0:
        log(f"Encoder exited with code {encoder.returncode}")
        return encoder.returncode
    if decoder.returncode != 0:
        log(f"Decoder exited with code {decoder.returncode}")
        return decoder.returncode

    log(f"Processed {frame_count} frames via ONNX pipeline.")
    return 0


def _build_decode_cmd(
    tools: ToolInfo,
    input_path: Path,
    enhance: bool,
) -> list[str]:
    cmd = [
        tools.ffmpeg, "-hide_banner",
        "-hwaccel", "auto",
        "-i", str(input_path),
    ]
    pre_filters: list[str] = []
    if enhance:
        if tools.has_deblock:
            pre_filters.append("deblock=filter=weak:block=8")
        if tools.has_hqdn3d:
            pre_filters.append("hqdn3d=1.2:1.2:4:4")
    if pre_filters:
        cmd.extend(["-vf", ",".join(pre_filters)])
    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "pipe:1",
    ])
    return cmd


def _build_encode_cmd(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    out_w: int,
    out_h: int,
    fps: float,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    target_width: int | None,
    target_height: int | None,
) -> list[str]:
    video_encoder = select_video_encoder(tools, codec)
    cmd = [
        tools.ffmpeg, "-hide_banner", "-stats",
        "-y" if overwrite else "-n",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-i", str(input_path),
        "-map", "0:v",
        "-map", "1:a?",
        "-map", "1:s?",
    ]
    post_filters: list[str] = []
    if target_width and target_height:
        post_filters.append(
            f"scale={target_width}:{target_height}:flags=lanczos"
        )
    if post_filters:
        cmd.extend(["-vf", ",".join(post_filters)])
    cmd.extend(encoder_args(video_encoder, quality))
    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-c:s", "copy",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])
    return cmd
