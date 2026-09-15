from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolInfo:
    ffmpeg: str
    ffprobe: str
    realesrgan: str | None
    rife: str | None
    has_cuda_scale: bool
    has_npp_scale: bool
    has_cas: bool
    has_deblock: bool
    has_hqdn3d: bool
    has_h264_nvenc: bool
    has_hevc_nvenc: bool


def find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found

    search_roots = [
        Path.cwd(),
        Path.home() / "Downloads",
        Path("C:/ffmpeg"),
        Path("C:/Tools/ffmpeg/bin"),
        Path("C:/Tools/uv/bin"),
        Path("C:/Tools/rife-ncnn-vulkan"),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]
    executable = f"{name}.exe" if os.name == "nt" else name
    for root in search_roots:
        if not root.exists():
            continue
        try:
            match = next(root.rglob(executable))
        except (StopIteration, PermissionError, OSError):
            continue
        return str(match)
    return None


def find_realesrgan() -> str | None:
    found = shutil.which("realesrgan-ncnn-vulkan")
    if found:
        return found

    candidates = [
        Path("C:/Tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe"),
        Path.cwd() / "tools" / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def find_rife() -> str | None:
    found = shutil.which("rife-ncnn-vulkan")
    if found:
        return found

    candidates = [
        Path("C:/Tools/rife-ncnn-vulkan/rife-ncnn-vulkan.exe"),
        Path.cwd() / "tools" / "rife-ncnn-vulkan" / "rife-ncnn-vulkan.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _run_capture(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def inspect_tools(ffmpeg_path: str | None = None, ffprobe_path: str | None = None) -> ToolInfo:
    ffmpeg = ffmpeg_path or find_executable("ffmpeg")
    ffprobe = ffprobe_path or find_executable("ffprobe")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and add its bin folder to PATH.")
    if not ffprobe:
        ffprobe_candidate = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if ffprobe_candidate.exists():
            ffprobe = str(ffprobe_candidate)
        else:
            raise RuntimeError("ffprobe was not found next to FFmpeg or on PATH.")

    filters = _run_capture([ffmpeg, "-hide_banner", "-filters"])
    encoders = _run_capture([ffmpeg, "-hide_banner", "-encoders"])

    return ToolInfo(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        realesrgan=find_realesrgan(),
        rife=find_rife(),
        has_cuda_scale="scale_cuda" in filters,
        has_npp_scale="scale_npp" in filters,
        has_cas=" cas " in filters,
        has_deblock=" deblock " in filters,
        has_hqdn3d=" hqdn3d " in filters,
        has_h264_nvenc="h264_nvenc" in encoders,
        has_hevc_nvenc="hevc_nvenc" in encoders,
    )
