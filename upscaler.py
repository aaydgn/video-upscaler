"""Tkinter/CLI wrapper for NVIDIA-assisted 720p -> 1080p video upscaling.

The script shells out to FFmpeg because NVDEC/NVENC and CUDA scalers are most
reliable through FFmpeg on Windows. It prefers GPU decode, GPU scaling, and GPU
encode when the local FFmpeg build exposes those features, then falls back to
CPU scaling only if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FOUR_K_WIDTH = 3840
FOUR_K_HEIGHT = 2160
REDETAIL_WIDTH = 960
REDETAIL_HEIGHT = 540
AI_2X_MODEL = "realesr-animevideov3"
INTERPOLATE_FPS = 60.0
INTERPOLATE_FPS_TOLERANCE = 0.1
ProgressCallback = Callable[[str, int | None, int | None], None]


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


def run_capture(args: list[str]) -> str:
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

    filters = run_capture([ffmpeg, "-hide_banner", "-filters"])
    encoders = run_capture([ffmpeg, "-hide_banner", "-encoders"])

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
    output = run_capture(
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
        ]
    )
    data = json.loads(output)
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


def stream_command(args: list[str], log: Callable[[str], None], cwd: str | None = None) -> int:
    log(subprocess.list2cmdline(args))
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    return process.wait()


def count_image_files(directory: Path) -> int:
    try:
        return sum(
            1
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
    except OSError:
        return 0


def report_progress(
    phase: str,
    current: int | None,
    total: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> None:
    if progress:
        progress(phase, current, total)
    if current is not None and total:
        percent = min(100.0, (current / total) * 100)
        log(f"{phase}: {current} / {total} frames ({percent:.1f}%)")
    else:
        log(f"{phase}...")


def derive_2x_output_path(output_path: Path) -> Path:
    if output_path.stem.endswith("_1080p_60fps"):
        stem = output_path.stem[: -len("_1080p_60fps")] + "_2x"
    elif output_path.stem.endswith("_1080p"):
        stem = output_path.stem[: -len("_1080p")] + "_2x"
    else:
        stem = output_path.stem + "_2x"
    two_x_path = output_path.with_name(stem + output_path.suffix)
    if two_x_path == output_path:
        return output_path.with_name(output_path.stem + "_2x" + output_path.suffix)
    return two_x_path


def stream_command_with_frame_progress(
    args: list[str],
    log: Callable[[str], None],
    output_dir: Path,
    total_frames: int,
    progress: ProgressCallback | None,
    phase: str = "AI upscaling frames",
    log_prefix: str = "Real-ESRGAN",
    cwd: str | None = None,
) -> int:
    log(subprocess.list2cmdline(args))
    stop_event = threading.Event()
    lock = threading.Lock()
    last_reported = {"count": -1, "time": 0.0}

    def emit(force: bool = False) -> None:
        current = count_image_files(output_dir)
        now = time.monotonic()
        with lock:
            changed = current != last_reported["count"]
            due = now - last_reported["time"] >= 2.0
            if force and not changed and last_reported["count"] != -1:
                return
            if not force and not (changed and (due or current == total_frames)):
                return
            last_reported["count"] = current
            last_reported["time"] = now
        report_progress(phase, current, total_frames, log, progress)

    def monitor() -> None:
        emit(force=True)
        while not stop_event.wait(1.0):
            emit()
        emit(force=True)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            log(f"[{log_prefix}] {text}")
    return_code = process.wait()
    stop_event.set()
    monitor_thread.join(timeout=5)
    return return_code


def build_ffmpeg_command(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool = False,
    target_width: int | None = TARGET_WIDTH,
    target_height: int | None = TARGET_HEIGHT,
) -> list[str]:
    if codec == "hevc" and tools.has_hevc_nvenc:
        video_encoder = "hevc_nvenc"
    elif tools.has_h264_nvenc:
        video_encoder = "h264_nvenc"
    elif tools.has_hevc_nvenc:
        video_encoder = "hevc_nvenc"
    else:
        raise RuntimeError("This FFmpeg build does not expose h264_nvenc or hevc_nvenc.")

    if target_width and target_height and tools.has_cuda_scale and not enhance:
        scale_filter = f"scale_cuda=w={target_width}:h={target_height}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif target_width and target_height and tools.has_npp_scale and not enhance:
        scale_filter = f"scale_npp={target_width}:{target_height}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        scale_filter = build_video_filter(tools, enhance, target_width, target_height)
        hw_args = []

    preset = "p6"
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
        "-vf",
        scale_filter,
        "-c:v",
        video_encoder,
        "-preset",
        preset,
        "-rc",
        "vbr",
        "-cq",
        str(quality),
        "-b:v",
        "0",
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return cmd


def build_video_filter(
    tools: ToolInfo,
    enhance: bool,
    target_width: int | None = TARGET_WIDTH,
    target_height: int | None = TARGET_HEIGHT,
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


def build_redetail_extract_filter(tools: ToolInfo) -> str:
    filters = []
    if tools.has_deblock:
        filters.append("deblock=filter=weak:block=8")
    if tools.has_hqdn3d:
        filters.append("hqdn3d=1.2:1.2:4:4")
    filters.append(f"scale={REDETAIL_WIDTH}:{REDETAIL_HEIGHT}:flags=lanczos")
    return ",".join(filters)


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
    if codec == "hevc" and tools.has_hevc_nvenc:
        return "hevc_nvenc"
    if tools.has_h264_nvenc:
        return "h264_nvenc"
    if tools.has_hevc_nvenc:
        return "hevc_nvenc"
    return "libx264"


def make_work_video_path(output_path: Path, label: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{label}_{uuid.uuid4().hex[:8]}.mp4")


def run_rife_interpolation(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    codec: str,
    quality: int,
    overwrite: bool,
    fps: float | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    if not tools.rife:
        raise RuntimeError(
            "RIFE was not found. Install rife-ncnn-vulkan to C:\\Tools\\rife-ncnn-vulkan "
            "or add rife-ncnn-vulkan.exe to PATH."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if fps is None:
        _width, _height, _duration, fps = probe_video(tools.ffprobe, input_path)
    if fps is None:
        fps = 30.0
        log("Warning: could not detect frame rate for RIFE; using 30 fps for frame-count math.")

    temp_path = output_path.parent / f"{output_path.stem}_rife_work_{uuid.uuid4().hex[:8]}"
    success = False
    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        frames_dir = temp_path / "frames"
        rife_dir = temp_path / "rife_frames"
        frames_dir.mkdir()
        rife_dir.mkdir()

        report_progress("Extracting frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            str(frames_dir / "frame_%08d.png"),
        ]
        code = stream_command(extract_cmd, log)
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code

        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            log(f"No extracted frames found. Work folder preserved for inspection: {temp_path}")
            return 1
        target_frames = max(total_frames, round(total_frames * INTERPOLATE_FPS / fps))
        log(f"RIFE target: {total_frames} frames at {fps:.3f} fps -> {target_frames} frames at 60 fps.")

        width, height, _duration, _probed_fps = probe_video(tools.ffprobe, input_path)
        rife_cmd = [
            tools.rife,
            "-i",
            str(frames_dir),
            "-o",
            str(rife_dir),
            "-m",
            "rife-v4",
            "-n",
            str(target_frames),
            "-g",
            "0",
            "-j",
            "2:2:2",
            "-f",
            "frame_%08d.png",
        ]
        if width and height and (width >= FOUR_K_WIDTH or height >= FOUR_K_HEIGHT):
            rife_cmd.append("-u")

        code = stream_command_with_frame_progress(
            rife_cmd,
            log,
            rife_dir,
            target_frames,
            progress,
            phase="Interpolating frames with RIFE",
            log_prefix="RIFE",
            cwd=str(Path(tools.rife).parent),
        )
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code

        processed_frames = count_image_files(rife_dir)
        if processed_frames < target_frames:
            log(f"Expected {target_frames} RIFE frames, found {processed_frames}.")
            log(f"Work folder preserved for inspection: {temp_path}")
            return 1

        video_encoder = select_video_encoder(tools, codec)
        report_progress("Reassembling 60 fps video", None, None, log, progress)
        assemble_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-stats",
            "-y" if overwrite else "-n",
            "-framerate",
            f"{INTERPOLATE_FPS:.6f}",
            "-start_number",
            "0",
            "-i",
            str(rife_dir / "frame_%08d.png"),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            "-c:v",
            video_encoder,
            "-preset",
            "p6" if video_encoder.endswith("_nvenc") else "slow",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        code = stream_command(assemble_cmd, log)
        success = code == 0
        if not success:
            log(f"Work folder preserved for inspection: {temp_path}")
        return code
    finally:
        if success:
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            except OSError as exc:
                log(f"Warning: could not remove work folder {temp_path}: {exc}")


def upscale(
    input_file: str,
    output_file: str | None = None,
    engine: str = "enhance",
    model: str = AI_2X_MODEL,
    codec: str = "h264",
    quality: int = 19,
    overwrite: bool = False,
    enhance: bool = False,
    interp60: bool = False,
    log: Callable[[str], None] = print,
    progress: ProgressCallback | None = None,
) -> int:
    apply_enhance = enhance or engine == "enhance"
    run_engine = "ffmpeg" if engine == "enhance" else engine
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_path = (
        Path(output_file).expanduser().resolve()
        if output_file
        else input_path.with_name(
            f"{input_path.stem}_"
            + (
                "60fps"
                if run_engine == "interp60"
                else "4k"
                if run_engine == "ai4k"
                else "redetail_1080p"
                if run_engine == "redetail"
                else "enhanced_1080p"
                if apply_enhance and run_engine != "ai"
                else "1080p"
            )
            + ("_60fps" if interp60 and run_engine != "interp60" else "")
            + ".mp4"
        )
    )
    tools = inspect_tools()
    width, height, duration, fps = probe_video(tools.ffprobe, input_path)
    apply_interp60 = should_interpolate_to_60(interp60 or run_engine == "interp60", fps, log)
    spatial_output_path = make_work_video_path(output_path, "pre60") if apply_interp60 and run_engine != "interp60" else output_path
    if width and height:
        log(f"Input: {width}x{height}")
        if run_engine == "ai" and (width != 1280 or height != 720):
            log("Warning: input is not exactly 1280x720. AI mode requires the original 720p file.")
        if run_engine == "redetail" and (width != TARGET_WIDTH or height != TARGET_HEIGHT):
            log("Warning: AI re-detail mode expects a 1920x1080 source.")
        if run_engine == "ai4k" and (width != TARGET_WIDTH or height != TARGET_HEIGHT):
            log("Warning: AI 4K mode expects a 1920x1080 source.")
    if duration:
        log(f"Duration: {duration:.1f}s")

    log(f"FFmpeg: {tools.ffmpeg}")
    if tools.realesrgan:
        log(f"Real-ESRGAN: {tools.realesrgan}")
    if tools.rife:
        log(f"RIFE: {tools.rife}")
    log(
        "Acceleration: "
        + (
            "CUDA scale"
            if tools.has_cuda_scale
            else "NPP scale"
            if tools.has_npp_scale
            else "CPU scale with NVENC encode"
        )
    )
    if interp60 or run_engine == "interp60":
        if not tools.rife:
            raise RuntimeError(
                "RIFE was not found. Install rife-ncnn-vulkan to C:\\Tools\\rife-ncnn-vulkan "
                "or add rife-ncnn-vulkan.exe to PATH."
            )
    if run_engine == "ai":
        if width and height and (width != 1280 or height != 720):
            raise RuntimeError(
                "AI mode expects the original 1280x720 input. "
                "Choose the original 720p file, not a previous 1080p/2x output."
            )
        if not tools.realesrgan:
            raise RuntimeError("Real-ESRGAN was not found. Install realesrgan-ncnn-vulkan or use --engine ffmpeg.")
        if not fps:
            fps = 30.0
            log("Warning: could not detect frame rate; using 30 fps.")
        return_code = upscale_with_realesrgan(
            tools,
            input_path,
            spatial_output_path,
            model,
            codec,
            quality,
            overwrite if spatial_output_path == output_path else True,
            apply_enhance,
            False,
            fps,
            derive_2x_output_path(output_path),
            log,
            progress,
        )
    elif run_engine == "redetail":
        if width and height and (width != TARGET_WIDTH or height != TARGET_HEIGHT):
            raise RuntimeError("AI re-detail mode expects the original 1920x1080 source.")
        if not tools.realesrgan:
            raise RuntimeError("Real-ESRGAN was not found. Install realesrgan-ncnn-vulkan or use --engine enhance.")
        if not fps:
            fps = 30.0
            log("Warning: could not detect frame rate; using 30 fps.")
        return_code = redetail_1080p_with_realesrgan(
            tools,
            input_path,
            spatial_output_path,
            model,
            codec,
            quality,
            overwrite if spatial_output_path == output_path else True,
            False,
            fps,
            log,
            progress,
        )
    elif run_engine == "ai4k":
        if width and height and (width != TARGET_WIDTH or height != TARGET_HEIGHT):
            raise RuntimeError("AI 4K mode expects the original 1920x1080 source.")
        if not tools.realesrgan:
            raise RuntimeError("Real-ESRGAN was not found. Install realesrgan-ncnn-vulkan or use --engine enhance.")
        if not fps:
            fps = 30.0
            log("Warning: could not detect frame rate; using 30 fps.")
        return_code = upscale_1080p_to_4k_with_realesrgan(
            tools,
            input_path,
            spatial_output_path,
            model,
            codec,
            quality,
            overwrite if spatial_output_path == output_path else True,
            False,
            fps,
            log,
            progress,
        )
    elif run_engine == "interp60":
        if apply_interp60:
            return_code = run_rife_interpolation(
                tools,
                input_path,
                output_path,
                codec,
                quality,
                overwrite,
                fps,
                log,
                progress,
            )
        else:
            log("Standalone 60 fps workflow requested, but input is already 60 fps. Copying streams.")
            report_progress("Copying existing 60 fps video", None, None, log, progress)
            return_code = stream_command(build_stream_copy_command(tools, input_path, output_path, overwrite), log)
    else:
        if apply_enhance:
            cmd = build_ffmpeg_command(
                tools,
                input_path,
                spatial_output_path,
                codec,
                quality,
                overwrite if spatial_output_path == output_path else True,
                enhance=True,
            )
            log("Running FFmpeg enhance pass:")
        else:
            cmd = build_ffmpeg_command(
                tools,
                input_path,
                spatial_output_path,
                codec,
                quality,
                overwrite if spatial_output_path == output_path else True,
            )
            log("Running FFmpeg scaler:")
        report_progress("Reassembling video", None, None, log, progress)
        return_code = stream_command(cmd, log)
    if return_code == 0 and apply_interp60 and run_engine != "interp60":
        return_code = run_rife_interpolation(
            tools,
            spatial_output_path,
            output_path,
            codec,
            quality,
            overwrite,
            fps,
            log,
            progress,
        )
        try:
            if spatial_output_path.exists():
                spatial_output_path.unlink()
        except OSError as exc:
            log(f"Warning: could not remove intermediate video {spatial_output_path}: {exc}")
    if return_code == 0:
        report_progress("Done", None, None, log, progress)
        log(f"Done: {output_path}")
    else:
        log(f"Upscale failed with exit code {return_code}")
    return return_code


def redetail_1080p_with_realesrgan(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model: str,
    codec: str,
    quality: int,
    overwrite: bool,
    interp60: bool,
    fps: float,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    assert tools.realesrgan is not None
    exe = Path(tools.realesrgan)
    model = select_two_x_model(model, log)
    temp_path = output_path.parent / f"{output_path.stem}_work_{uuid.uuid4().hex[:8]}"
    success = False
    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        frames_dir = temp_path / "frames_540p"
        ai_dir = temp_path / "ai_frames_1080p"
        frames_dir.mkdir()
        ai_dir.mkdir()

        log(
            f"AI re-detail pipeline: {TARGET_WIDTH}x{TARGET_HEIGHT} source -> "
            f"{REDETAIL_WIDTH}x{REDETAIL_HEIGHT} denoised frames -> "
            f"{TARGET_WIDTH}x{TARGET_HEIGHT} AI output"
        )
        report_progress("Denoising and downscaling frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            build_redetail_extract_filter(tools),
            str(frames_dir / "frame_%08d.png"),
        ]
        code = stream_command(extract_cmd, log)
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            log(f"No extracted frames found. Work folder preserved for inspection: {temp_path}")
            return 1
        log(f"Extracted {total_frames} denoised/downscaled frames.")

        log("Running Real-ESRGAN AI re-detailing at 2x.")
        ai_cmd = [
            str(exe),
            "-i",
            str(frames_dir),
            "-o",
            str(ai_dir),
            "-n",
            model,
            "-s",
            "2",
            "-g",
            "0",
            "-t",
            "512",
            "-j",
            "1:1:1",
            "-f",
            "png",
        ]
        code = stream_command_with_frame_progress(ai_cmd, log, ai_dir, total_frames, progress, cwd=str(exe.parent))
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        processed_frames = count_image_files(ai_dir)
        if processed_frames < total_frames:
            log(f"Expected {total_frames} AI frames, found {processed_frames}.")
            log(f"Work folder preserved for inspection: {temp_path}")
            return 1

        if codec == "hevc" and tools.has_hevc_nvenc:
            video_encoder = "hevc_nvenc"
        elif tools.has_h264_nvenc:
            video_encoder = "h264_nvenc"
        else:
            video_encoder = "libx264"

        report_progress("Interpolating to 60 fps" if interp60 else "Reassembling video", None, None, log, progress)
        final_filter = build_video_filter(tools, False, None, None)
        assemble_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-stats",
            "-y" if overwrite else "-n",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(ai_dir / "frame_%08d.png"),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            *(["-vf", final_filter] if final_filter else []),
            "-c:v",
            video_encoder,
            "-preset",
            "p6" if video_encoder.endswith("_nvenc") else "slow",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        code = stream_command(assemble_cmd, log)
        success = code == 0
        if not success:
            log(f"Work folder preserved for inspection: {temp_path}")
        return code
    finally:
        if success:
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            except OSError as exc:
                log(f"Warning: could not remove work folder {temp_path}: {exc}")


def upscale_1080p_to_4k_with_realesrgan(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model: str,
    codec: str,
    quality: int,
    overwrite: bool,
    interp60: bool,
    fps: float,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    assert tools.realesrgan is not None
    exe = Path(tools.realesrgan)
    model = select_two_x_model(model, log)
    temp_path = output_path.parent / f"{output_path.stem}_work_{uuid.uuid4().hex[:8]}"
    success = False
    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        frames_dir = temp_path / "frames_1080p"
        ai_dir = temp_path / "ai_frames_4k"
        frames_dir.mkdir()
        ai_dir.mkdir()

        log(f"AI 4K pipeline: {TARGET_WIDTH}x{TARGET_HEIGHT} source -> {FOUR_K_WIDTH}x{FOUR_K_HEIGHT} AI output")
        report_progress("Extracting frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            str(frames_dir / "frame_%08d.png"),
        ]
        code = stream_command(extract_cmd, log)
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            log(f"No extracted frames found. Work folder preserved for inspection: {temp_path}")
            return 1
        log(f"Extracted {total_frames} frames.")

        log("Running Real-ESRGAN AI 2x to 4K.")
        ai_cmd = [
            str(exe),
            "-i",
            str(frames_dir),
            "-o",
            str(ai_dir),
            "-n",
            model,
            "-s",
            "2",
            "-g",
            "0",
            "-t",
            "512",
            "-j",
            "1:1:1",
            "-f",
            "png",
        ]
        code = stream_command_with_frame_progress(ai_cmd, log, ai_dir, total_frames, progress, cwd=str(exe.parent))
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        processed_frames = count_image_files(ai_dir)
        if processed_frames < total_frames:
            log(f"Expected {total_frames} AI frames, found {processed_frames}.")
            log(f"Work folder preserved for inspection: {temp_path}")
            return 1

        if codec == "hevc" and tools.has_hevc_nvenc:
            video_encoder = "hevc_nvenc"
        elif tools.has_h264_nvenc:
            video_encoder = "h264_nvenc"
        else:
            video_encoder = "libx264"

        report_progress("Interpolating to 60 fps" if interp60 else "Saving 4K AI video", None, None, log, progress)
        final_filter = build_video_filter(tools, False, None, None)
        assemble_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-stats",
            "-y" if overwrite else "-n",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(ai_dir / "frame_%08d.png"),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            *(["-vf", final_filter] if final_filter else []),
            "-c:v",
            video_encoder,
            "-preset",
            "p6" if video_encoder.endswith("_nvenc") else "slow",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        code = stream_command(assemble_cmd, log)
        success = code == 0
        if not success:
            log(f"Work folder preserved for inspection: {temp_path}")
        return code
    finally:
        if success:
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            except OSError as exc:
                log(f"Warning: could not remove work folder {temp_path}: {exc}")


def upscale_with_realesrgan(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model: str,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    interp60: bool,
    fps: float,
    two_x_output_path: Path | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    assert tools.realesrgan is not None
    exe = Path(tools.realesrgan)
    model = select_two_x_model(model, log)
    temp_path = output_path.parent / f"{output_path.stem}_work_{uuid.uuid4().hex[:8]}"
    success = False
    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        frames_dir = temp_path / "frames"
        ai_dir = temp_path / "ai_frames"
        frames_dir.mkdir()
        ai_dir.mkdir()

        report_progress("Extracting frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            str(frames_dir / "frame_%08d.png"),
        ]
        code = stream_command(extract_cmd, log)
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            log(f"No extracted frames found. Work folder preserved for inspection: {temp_path}")
            return 1
        log(f"Extracted {total_frames} frames.")

        log("Running Real-ESRGAN AI upscaling. This should take much longer than a few seconds.")
        ai_cmd = [
            str(exe),
            "-i",
            str(frames_dir),
            "-o",
            str(ai_dir),
            "-n",
            model,
            "-s",
            "2",
            "-g",
            "0",
            "-t",
            "512",
            "-j",
            "1:1:1",
            "-f",
            "png",
        ]
        code = stream_command_with_frame_progress(ai_cmd, log, ai_dir, total_frames, progress, cwd=str(exe.parent))
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code
        processed_frames = count_image_files(ai_dir)
        if processed_frames < total_frames:
            log(f"Expected {total_frames} upscaled frames, found {processed_frames}.")
            log(f"Work folder preserved for inspection: {temp_path}")
            return 1

        if codec == "hevc" and tools.has_hevc_nvenc:
            video_encoder = "hevc_nvenc"
        elif tools.has_h264_nvenc:
            video_encoder = "h264_nvenc"
        else:
            video_encoder = "libx264"

        two_x_output_path = two_x_output_path or derive_2x_output_path(output_path)
        log(f"2x AI master output: {two_x_output_path}")
        report_progress("Saving 2x AI video", None, None, log, progress)
        assemble_2x_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-stats",
            "-y" if overwrite else "-n",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(ai_dir / "frame_%08d.png"),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            "-c:v",
            video_encoder,
            "-preset",
            "p6" if video_encoder.endswith("_nvenc") else "slow",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(two_x_output_path),
        ]
        code = stream_command(assemble_2x_cmd, log)
        if code != 0:
            log(f"Work folder preserved for inspection: {temp_path}")
            return code

        report_progress(
            "Downscaling to 1080p and interpolating to 60 fps" if interp60 else "Downscaling to 1080p",
            None,
            None,
            log,
            progress,
        )
        final_filter = build_video_filter(tools, enhance)
        downscale_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-stats",
            "-y" if overwrite else "-n",
            "-i",
            str(two_x_output_path),
            "-map",
            "0",
            "-vf",
            final_filter,
            "-c:v",
            video_encoder,
            "-preset",
            "p6" if video_encoder.endswith("_nvenc") else "slow",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        code = stream_command(downscale_cmd, log)
        success = code == 0
        if not success:
            log(f"Work folder preserved for inspection: {temp_path}")
        return code
    finally:
        if success:
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            except OSError as exc:
                log(f"Warning: could not remove work folder {temp_path}: {exc}")


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("NVIDIA Video Upscaler")
    root.geometry("760x520")
    root.minsize(680, 460)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    workflow_var = tk.StringVar(value="Fast enhance 1080p")
    model_var = tk.StringVar(value=AI_2X_MODEL)
    codec_var = tk.StringVar(value="h264")
    quality_var = tk.IntVar(value=19)
    interp60_var = tk.BooleanVar(value=False)
    overwrite_var = tk.BooleanVar(value=False)
    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value="Idle")
    messages = queue.Queue()
    workflows = {
        "Fast enhance 1080p": "enhance",
        "AI re-detail 1080p": "redetail",
        "AI 4K upscale": "ai4k",
        "720p AI upscale": "ai",
        "Fast scale": "ffmpeg",
        "Interpolate existing video to 60 fps": "interp60",
    }
    ai_workflows = {"AI re-detail 1080p", "AI 4K upscale", "720p AI upscale"}

    class Tooltip:
        def __init__(self, widget: tk.Widget, text: str) -> None:
            self.widget = widget
            self.text = text
            self.tip: tk.Toplevel | None = None
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

        def show(self, _event: tk.Event) -> None:
            if self.tip:
                return
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self.tip,
                text=self.text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=5,
                wraplength=340,
            )
            label.pack()

        def hide(self, _event: tk.Event) -> None:
            if self.tip:
                self.tip.destroy()
                self.tip = None

    def default_output_for(path: Path) -> Path:
        workflow = workflow_var.get()
        interpolate = interp60_var.get() and workflow != "Interpolate existing video to 60 fps"
        if workflow == "Fast enhance 1080p":
            suffix = "enhanced_1080p"
        elif workflow == "AI re-detail 1080p":
            suffix = "redetail_1080p"
        elif workflow == "AI 4K upscale":
            suffix = "4k"
        elif workflow == "Interpolate existing video to 60 fps":
            suffix = "60fps"
        else:
            suffix = "1080p"
        if interpolate:
            suffix += "_60fps"
        return path.with_name(f"{path.stem}_{suffix}.mp4")

    def update_default_output(_event: tk.Event | None = None) -> None:
        if input_var.get():
            output_var.set(str(default_output_for(Path(input_var.get()))))

    def update_workflow_controls(_event: tk.Event | None = None) -> None:
        workflow = workflow_var.get()
        if workflow == "Interpolate existing video to 60 fps":
            interp60_check.state(["disabled"])
        else:
            interp60_check.state(["!disabled"])
        model_combo.configure(state="readonly" if workflow in ai_workflows else "disabled")
        start_button.configure(text="Interpolate to 60 fps" if workflow == "Interpolate existing video to 60 fps" else "Process video")
        update_default_output()

    def choose_input() -> None:
        filename = filedialog.askopenfilename(
            title="Choose source video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            input_var.set(filename)
            path = Path(filename)
            output_var.set(str(default_output_for(path)))

    def choose_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if filename:
            output_var.set(filename)

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def queue_progress(phase: str, current: int | None, total: int | None) -> None:
        messages.put(("progress", phase, current, total))

    def apply_progress(phase: str, current: int | None, total: int | None) -> None:
        if current is not None and total:
            percent = min(100.0, (current / total) * 100)
            progress_var.set(percent)
            progress_text_var.set(f"{phase}: {current} / {total} frames ({percent:.1f}%)")
        elif phase == "Done":
            progress_var.set(100.0)
            progress_text_var.set("Done")
        else:
            progress_text_var.set(phase)

    def drain_messages() -> None:
        try:
            while True:
                item = messages.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "progress":
                    _, phase, current, total = item
                    apply_progress(phase, current, total)
                else:
                    append_log(str(item))
        except queue.Empty:
            pass
        root.after(100, drain_messages)

    def start() -> None:
        if not input_var.get():
            messagebox.showerror("Missing input", "Choose a source video first.")
            return
        selected_workflow = workflow_var.get()
        selected_engine = workflows[selected_workflow]
        start_button.configure(state="disabled")
        progress_var.set(0.0)
        progress_text_var.set("Starting")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")

        def worker() -> None:
            try:
                code = upscale(
                    input_var.get(),
                    output_var.get() or None,
                    selected_engine,
                    model_var.get(),
                    codec_var.get(),
                    quality_var.get(),
                    overwrite_var.get(),
                    False,
                    interp60_var.get() and selected_workflow != "Interpolate existing video to 60 fps",
                    messages.put,
                    queue_progress,
                )
                if code != 0:
                    messages.put("Upscale failed. Check the FFmpeg log above.")
            except Exception as exc:  # GUI boundary: show unexpected failures in log.
                messages.put(f"Error: {exc}")
            finally:
                root.after(0, lambda: start_button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(11, weight=1)

    ttk.Label(frame, text="Input video").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(frame, textvariable=input_var).grid(row=0, column=1, sticky="ew", padx=8)
    ttk.Button(frame, text="Browse", command=choose_input).grid(row=0, column=2)

    ttk.Label(frame, text="Output video").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(frame, textvariable=output_var).grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(frame, text="Browse", command=choose_output).grid(row=1, column=2)

    ttk.Label(frame, text="Codec").grid(row=2, column=0, sticky="w", pady=4)
    codec_frame = ttk.Frame(frame)
    codec_frame.grid(row=2, column=1, sticky="w", padx=8)
    ttk.Radiobutton(codec_frame, text="H.264 NVENC", value="h264", variable=codec_var).pack(side="left")
    ttk.Radiobutton(codec_frame, text="HEVC NVENC", value="hevc", variable=codec_var).pack(side="left", padx=16)

    ttk.Label(frame, text="Workflow").grid(row=3, column=0, sticky="w", pady=4)
    workflow_combo = ttk.Combobox(
        frame,
        textvariable=workflow_var,
        values=tuple(workflows.keys()),
        state="readonly",
    )
    workflow_combo.grid(row=3, column=1, sticky="ew", padx=8)
    workflow_combo.bind("<<ComboboxSelected>>", update_workflow_controls)
    Tooltip(
        workflow_combo,
        "Fast enhance: 1080p cleanup only.\n"
        "AI re-detail: 1080p -> denoised 540p frames -> Real-ESRGAN 2x -> 1080p.\n"
        "AI 4K upscale: 1080p -> Real-ESRGAN 2x -> 4K.\n"
        "720p AI upscale: original 720p -> 1440p AI master -> 1080p.\n"
        "Fast scale: resize only.\n"
        "Interpolate existing video to 60 fps: preserve resolution and change frame rate only.",
    )

    ttk.Label(frame, text="AI model").grid(row=4, column=0, sticky="w", pady=4)
    model_combo = ttk.Combobox(
        frame,
        textvariable=model_var,
        values=("realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime", "realesrnet-x4plus"),
        state="readonly",
    )
    model_combo.grid(row=4, column=1, sticky="ew", padx=8)

    ttk.Label(frame, text="Quality").grid(row=5, column=0, sticky="w", pady=4)
    ttk.Scale(frame, from_=14, to=28, variable=quality_var, orient="horizontal").grid(
        row=5, column=1, sticky="ew", padx=8
    )
    ttk.Label(frame, textvariable=quality_var, width=4).grid(row=5, column=2, sticky="w")

    interp60_check = ttk.Checkbutton(frame, text="Interpolate to 60 fps", variable=interp60_var, command=update_default_output)
    interp60_check.grid(row=6, column=1, sticky="w", padx=8, pady=4)
    Tooltip(
        interp60_check,
        "Uses RIFE on the GPU to produce exact 60 fps. "
        "The app extracts frames, runs rife-ncnn-vulkan, then reassembles with NVENC.",
    )

    ttk.Checkbutton(frame, text="Overwrite output if it exists", variable=overwrite_var).grid(
        row=7, column=1, sticky="w", padx=8, pady=4
    )

    start_button = ttk.Button(frame, text="Process video", command=start)
    start_button.grid(row=8, column=1, sticky="w", padx=8, pady=10)

    progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, mode="determinate")
    progress_bar.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 2))
    ttk.Label(frame, textvariable=progress_text_var).grid(row=10, column=0, columnspan=3, sticky="w")

    log_box = tk.Text(frame, height=14, state="disabled", wrap="word")
    log_box.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

    update_workflow_controls()
    drain_messages()
    root.mainloop()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale, enhance, or interpolate a video using FFmpeg and NVIDIA encoding.")
    parser.add_argument("input", nargs="?", help="Input video path. Omit to launch the Tkinter UI.")
    parser.add_argument("-o", "--output", help="Output video path. Defaults to a workflow-specific filename.")
    parser.add_argument(
        "--engine",
        choices=["ai", "redetail", "ai4k", "enhance", "ffmpeg", "interp60"],
        default="enhance",
        help="Use 720p AI, 1080p AI re-detail, 1080p to 4K AI, fast FFmpeg enhance, fast FFmpeg scale, or standalone 60 fps interpolation.",
    )
    parser.add_argument(
        "--model",
        choices=["realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime", "realesrnet-x4plus"],
        default=AI_2X_MODEL,
        help="Real-ESRGAN model. 2x workflows use realesr-animevideov3 to avoid x4 model cropping.",
    )
    parser.add_argument("--codec", choices=["h264", "hevc"], default="h264", help="NVENC codec to use.")
    parser.add_argument("--quality", type=int, default=19, help="NVENC CQ value, lower is larger/better. Default: 19.")
    parser.add_argument("--enhance", action="store_true", help="Add the fast deblock/denoise/CAS enhancement pass.")
    parser.add_argument("--interp60", action="store_true", help="Interpolate the final output to exact 60 fps.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    parser.add_argument("--check", action="store_true", help="Check FFmpeg/NVIDIA capabilities and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.check:
        try:
            tools = inspect_tools()
        except Exception as exc:
            print(f"Setup check failed: {exc}", file=sys.stderr)
            return 1
        print(f"FFmpeg: {tools.ffmpeg}")
        print(f"ffprobe: {tools.ffprobe}")
        print(f"scale_cuda: {'yes' if tools.has_cuda_scale else 'no'}")
        print(f"scale_npp: {'yes' if tools.has_npp_scale else 'no'}")
        print(f"cas: {'yes' if tools.has_cas else 'no'}")
        print(f"deblock: {'yes' if tools.has_deblock else 'no'}")
        print(f"hqdn3d: {'yes' if tools.has_hqdn3d else 'no'}")
        print(f"h264_nvenc: {'yes' if tools.has_h264_nvenc else 'no'}")
        print(f"hevc_nvenc: {'yes' if tools.has_hevc_nvenc else 'no'}")
        print(f"realesrgan: {tools.realesrgan or 'not found'}")
        print(f"rife: {tools.rife or 'not found'}")
        return 0
    if not args.input:
        launch_gui()
        return 0
    try:
        return upscale(
            args.input,
            args.output,
            args.engine,
            args.model,
            args.codec,
            args.quality,
            args.overwrite,
            args.enhance,
            args.interp60,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
