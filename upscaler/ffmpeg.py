from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Callable

from upscaler.config import AI_2X_MODEL, INTERPOLATE_FPS, INTERPOLATE_FPS_TOLERANCE
from upscaler.tools import ToolInfo


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            if denominator_float == 0:
                return None
            return float(numerator) / denominator_float
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(ffprobe: str, input_path: Path) -> tuple[int | None, int | None, float | None, float | None]:
    output = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None, None, None, None
    streams = data.get("streams", [])
    if not streams:
        return None, None, None, None
    stream = streams[0]
    duration_raw = stream.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except ValueError:
        duration = None
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    return stream.get("width"), stream.get("height"), duration, fps


def build_video_filter(
    tools: ToolInfo,
    enhance: bool,
    target_width: int | None = None,
    target_height: int | None = None,
) -> str:
    filters = []
    if enhance and tools.has_deblock:
        filters.append("deblock=filter=weak:block=8")
    if enhance and tools.has_hqdn3d:
        filters.append("hqdn3d=1.2:1.2:4:4")
    if target_width and target_height:
        filters.append(f"scale={target_width}:{target_height}:flags=lanczos")
    if enhance and tools.has_cas:
        filters.append("cas=strength=0.45")
    elif enhance:
        filters.append("unsharp=5:5:0.45:3:3:0.15")
    return ",".join(filters)


def build_ffmpeg_command(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool = False,
    target_width: int | None = None,
    target_height: int | None = None,
) -> list[str]:
    video_encoder = select_video_encoder(tools, codec)

    hw_args: list[str] = []
    video_filter: str | None = None
    use_cuda_hw = video_encoder.endswith("_nvenc")

    if target_width and target_height and tools.has_cuda_scale and not enhance and use_cuda_hw:
        video_filter = f"scale_cuda=w={target_width}:h={target_height}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif target_width and target_height and tools.has_npp_scale and not enhance and use_cuda_hw:
        video_filter = f"scale_npp={target_width}:{target_height}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        vf = build_video_filter(tools, enhance, target_width, target_height)
        if vf:
            video_filter = vf

    cmd = [
        tools.ffmpeg,
        "-hide_banner",
        "-stats",
        "-y" if overwrite else "-n",
        *hw_args,
        "-i",
        str(input_path),
        "-map",
        "0",
    ]
    if video_filter:
        cmd.extend(["-vf", video_filter])
    cmd.extend(encoder_args(video_encoder, quality, "quality"))
    cmd.extend([
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ])
    return cmd


def build_stream_copy_command(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> list[str]:
    return [
        tools.ffmpeg,
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def select_video_encoder(tools: ToolInfo, codec: str) -> str:
    if codec == "hevc":
        if tools.has_hevc_nvenc:
            return "hevc_nvenc"
        if tools.has_hevc_amf:
            return "hevc_amf"
        return "libx265"
    if tools.has_h264_nvenc:
        return "h264_nvenc"
    if tools.has_h264_amf:
        return "h264_amf"
    if tools.has_hevc_nvenc:
        return "hevc_nvenc"
    if tools.has_hevc_amf:
        return "hevc_amf"
    return "libx264"


def encoder_args(encoder: str, quality: int, preset: str = "balanced") -> list[str]:
    if encoder.endswith("_nvenc"):
        return [
            "-c:v", encoder,
            "-preset", {"fast": "p4", "balanced": "p4", "quality": "p6"}[preset],
            "-cq", str(quality),
            "-b:v", "0",
        ]
    if encoder.endswith("_amf"):
        return [
            "-c:v", encoder,
            "-quality", {"fast": "speed", "balanced": "balanced", "quality": "quality"}[preset],
            "-rc", "cqp",
            "-qp_i", str(quality),
            "-qp_p", str(quality),
            "-qp_b", str(quality),
        ]
    return [
        "-c:v", encoder,
        "-preset", {"fast": "fast", "balanced": "medium", "quality": "slow"}[preset],
        "-crf", str(quality),
    ]


def make_work_video_path(output_path: Path, label: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{label}_{uuid.uuid4().hex[:8]}.mp4")


def should_interpolate_to_60(
    requested: bool,
    fps: float | None,
    log: Callable[[str], None],
) -> bool:
    if not requested:
        return False
    if fps is None:
        log("Warning: could not detect frame rate; assuming interpolation to 60 fps is needed.")
        return True
    if fps > INTERPOLATE_FPS + INTERPOLATE_FPS_TOLERANCE:
        raise RuntimeError(f"Interpolation targets exact 60 fps. Input is already {fps:.3f} fps.")
    if abs(fps - INTERPOLATE_FPS) <= INTERPOLATE_FPS_TOLERANCE:
        log("Input is already 60 fps. Skipping interpolation.")
        return False
    return True


def select_two_x_model(model: str, log: Callable[[str], None]) -> str:
    if model != AI_2X_MODEL:
        log(
            f"Using {AI_2X_MODEL} for 2x AI. "
            f"{model} is an x4 model on this Real-ESRGAN build and breaks with -s 2."
        )
    return AI_2X_MODEL
