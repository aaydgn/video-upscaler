from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable

from upscaler.config import AI_2X_MODEL, INTERPOLATE_FPS, ProgressCallback
from upscaler.ffmpeg import (
    build_ffmpeg_command,
    build_stream_copy_command,
    make_work_video_path,
    probe_video,
    select_two_x_model,
    select_video_encoder,
    should_interpolate_to_60,
)
from upscaler.process import (
    count_image_files,
    report_progress,
    stream_command,
    stream_command_with_frame_progress,
)
from upscaler.tools import ToolInfo, inspect_tools

CHUNK_SIZE = 500


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
    width, height, _duration, probed_fps = probe_video(tools.ffprobe, input_path)
    if fps is None:
        fps = probed_fps
    if fps is None:
        fps = 30.0
        log("Warning: could not detect frame rate for RIFE; using 30 fps for frame-count math.")

    temp_path = Path(tempfile.mkdtemp(prefix="upscaler_rife_"))
    try:
        frames_dir = temp_path / "frames"
        rife_dir = temp_path / "rife_frames"
        frames_dir.mkdir()
        rife_dir.mkdir()

        report_progress("Step 1/3 – Extracting frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-hwaccel", "auto",
            "-y",
            "-i",
            str(input_path),
            "-vsync", "0",
            "-qscale:v", "2",
            str(frames_dir / "frame_%08d.jpg"),
        ]
        code = stream_command(extract_cmd, log)
        if code != 0:
            return code

        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            return 1
        target_frames = max(total_frames, round(total_frames * INTERPOLATE_FPS / fps))
        log(f"RIFE target: {total_frames} frames at {fps:.3f} fps -> {target_frames} frames at 60 fps.")

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
            "4:2:4",
            "-f",
            "frame_%08d.jpg",
        ]
        if width and height and (width >= 3840 or height >= 2160):
            rife_cmd.append("-u")

        code = stream_command_with_frame_progress(
            rife_cmd,
            log,
            rife_dir,
            target_frames,
            progress,
            phase="Step 2/3 – Interpolating with RIFE",
            log_prefix="RIFE",
            cwd=str(Path(tools.rife).parent),
        )
        if code != 0:
            return code

        processed_frames = count_image_files(rife_dir)
        if processed_frames < target_frames:
            log(f"Expected {target_frames} RIFE frames, found {processed_frames}.")
            return 1

        video_encoder = select_video_encoder(tools, codec)
        report_progress("Step 3/3 – Reassembling 60 fps video", None, None, log, progress)
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
            str(rife_dir / "frame_%08d.jpg"),
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
            "p4" if video_encoder.endswith("_nvenc") else "medium",
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
        return code
    finally:
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
        except OSError:
            pass


def _extract_chunk(
    tools: ToolInfo,
    input_path: Path,
    output_dir: Path,
    start_frame: int,
    count: int,
    enhance: bool,
    log: Callable[[str], None],
) -> int:
    cmd = [
        tools.ffmpeg,
        "-hide_banner",
        "-hwaccel", "auto",
        "-y",
        "-i",
        str(input_path),
        "-vsync", "0",
    ]
    vf_parts = []
    if enhance:
        if tools.has_deblock:
            vf_parts.append("deblock=filter=weak:block=8")
        if tools.has_hqdn3d:
            vf_parts.append("hqdn3d=1.2:1.2:4:4")
    select_expr = f"between(n\\,{start_frame}\\,{start_frame + count - 1})"
    vf_parts.append(f"select={select_expr}")
    cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend(["-start_number", str(start_frame + 1)])
    cmd.extend(["-qscale:v", "2"])
    cmd.append(str(output_dir / "frame_%08d.jpg"))
    return stream_command(cmd, log)


def run_ai_upscale(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model: str,
    scale: int,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    fps: float,
    target_width: int | None,
    target_height: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    assert tools.realesrgan is not None
    exe = Path(tools.realesrgan)
    if scale <= 3:
        model = select_two_x_model(model, log)

    width, height, duration, _fps = probe_video(tools.ffprobe, input_path)
    total_frames = _estimate_frame_count(duration, fps)

    if total_frames and total_frames <= CHUNK_SIZE * 2:
        return _run_ai_upscale_simple(
            tools, exe, input_path, output_path, model, scale,
            codec, quality, overwrite, enhance, fps,
            target_width, target_height, log, progress,
        )

    return _run_ai_upscale_chunked(
        tools, exe, input_path, output_path, model, scale,
        codec, quality, overwrite, enhance, fps,
        target_width, target_height, total_frames,
        log, progress,
    )


def _estimate_frame_count(duration: float | None, fps: float) -> int | None:
    if duration is None:
        return None
    return max(1, round(duration * fps))


def _run_ai_upscale_simple(
    tools: ToolInfo,
    exe: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    scale: int,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    fps: float,
    target_width: int | None,
    target_height: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    temp_path = Path(tempfile.mkdtemp(prefix="upscaler_ai_"))
    try:
        frames_dir = temp_path / "frames"
        ai_dir = temp_path / "ai_frames"
        frames_dir.mkdir()
        ai_dir.mkdir()

        report_progress("Step 1/3 – Extracting frames", None, None, log, progress)
        extract_cmd = [
            tools.ffmpeg,
            "-hide_banner",
            "-hwaccel", "auto",
            "-y",
            "-i",
            str(input_path),
            "-vsync", "0",
        ]
        if enhance:
            pre_filters = []
            if tools.has_deblock:
                pre_filters.append("deblock=filter=weak:block=8")
            if tools.has_hqdn3d:
                pre_filters.append("hqdn3d=1.2:1.2:4:4")
            if pre_filters:
                extract_cmd.extend(["-vf", ",".join(pre_filters)])
        extract_cmd.extend(["-qscale:v", "2"])
        extract_cmd.append(str(frames_dir / "frame_%08d.jpg"))
        code = stream_command(extract_cmd, log)
        if code != 0:
            return code

        total_frames = count_image_files(frames_dir)
        if total_frames <= 0:
            return 1
        log(f"Extracted {total_frames} frames.")

        log(f"Running Real-ESRGAN AI upscaling at {scale}x.")
        ai_cmd = _build_esrgan_cmd(exe, frames_dir, ai_dir, model, scale)
        code = stream_command_with_frame_progress(
            ai_cmd, log, ai_dir, total_frames, progress,
            phase="Step 2/3 – AI upscaling",
            cwd=str(exe.parent),
        )
        if code != 0:
            return code

        processed_frames = count_image_files(ai_dir)
        if processed_frames < total_frames:
            log(f"Expected {total_frames} AI frames, found {processed_frames}.")
            return 1

        return _assemble_video(
            tools, ai_dir, input_path, output_path,
            codec, quality, overwrite, fps,
            target_width, target_height, log, progress,
            "Step 3/3 – Reassembling video",
        )
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def _run_ai_upscale_chunked(
    tools: ToolInfo,
    exe: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    scale: int,
    codec: str,
    quality: int,
    overwrite: bool,
    enhance: bool,
    fps: float,
    target_width: int | None,
    target_height: int | None,
    estimated_frames: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    temp_path = Path(tempfile.mkdtemp(prefix="upscaler_ai_"))
    try:
        ai_dir = temp_path / "ai_frames"
        ai_dir.mkdir()

        total_frames = estimated_frames or 0
        processed = 0
        chunk_idx = 0
        error_code = 0

        log(f"Running chunked AI {scale}x pipeline (batch size {CHUNK_SIZE}).")

        while True:
            chunk_dir = temp_path / f"chunk_{chunk_idx}"
            chunk_dir.mkdir()
            start_frame = chunk_idx * CHUNK_SIZE

            report_progress(
                "Extracting + upscaling",
                processed, total_frames if total_frames else None,
                log, progress,
            )

            code = _extract_chunk(
                tools, input_path, chunk_dir,
                start_frame, CHUNK_SIZE, enhance, log,
            )
            if code != 0:
                error_code = code
                break

            chunk_frames = count_image_files(chunk_dir)
            if chunk_frames == 0:
                break

            chunk_ai_dir = temp_path / f"chunk_{chunk_idx}_ai"
            chunk_ai_dir.mkdir()

            ai_cmd = _build_esrgan_cmd(exe, chunk_dir, chunk_ai_dir, model, scale)
            code = stream_command_with_frame_progress(
                ai_cmd, log, chunk_ai_dir, chunk_frames, progress,
                phase=f"AI upscaling batch {chunk_idx + 1}",
                cwd=str(exe.parent),
            )

            shutil.rmtree(chunk_dir, ignore_errors=True)

            if code != 0:
                error_code = code
                break

            ai_count = count_image_files(chunk_ai_dir)
            if ai_count < chunk_frames:
                log(f"Batch {chunk_idx + 1}: expected {chunk_frames} AI frames, got {ai_count}.")
                error_code = 1
                break

            for f in sorted(chunk_ai_dir.iterdir()):
                if f.is_file():
                    processed += 1
                    f.rename(ai_dir / f"frame_{processed:08d}.jpg")
            shutil.rmtree(chunk_ai_dir, ignore_errors=True)

            if chunk_frames < CHUNK_SIZE:
                break
            chunk_idx += 1

        if not total_frames:
            total_frames = processed

        if error_code != 0:
            return error_code

        if processed == 0:
            log("No frames were extracted.")
            return 1

        log(f"Processed {processed} frames across {chunk_idx + 1} batches.")

        return _assemble_video(
            tools, ai_dir, input_path, output_path,
            codec, quality, overwrite, fps,
            target_width, target_height, log, progress,
            "Reassembling video",
        )
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def _build_esrgan_cmd(
    exe: Path,
    input_dir: Path,
    output_dir: Path,
    model: str,
    scale: int,
) -> list[str]:
    return [
        str(exe),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-n", model,
        "-s", str(scale),
        "-g", "0",
        "-t", "0",
        "-j", "4:2:4",
        "-f", "jpg",
    ]


def _assemble_video(
    tools: ToolInfo,
    frames_dir: Path,
    audio_source: Path,
    output_path: Path,
    codec: str,
    quality: int,
    overwrite: bool,
    fps: float,
    target_width: int | None,
    target_height: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
    phase: str,
) -> int:
    video_encoder = select_video_encoder(tools, codec)
    report_progress(phase, None, None, log, progress)
    cmd = [
        tools.ffmpeg,
        "-hide_banner",
        "-stats",
        "-y" if overwrite else "-n",
        "-framerate", f"{fps:.6f}",
        "-i", str(frames_dir / "frame_%08d.jpg"),
        "-i", str(audio_source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
    ]
    if target_width and target_height:
        cmd.extend(["-vf", f"scale={target_width}:{target_height}:flags=lanczos"])
    cmd.extend([
        "-c:v", video_encoder,
        "-preset", "p4" if video_encoder.endswith("_nvenc") else "medium",
        "-cq", str(quality),
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-c:s", "copy",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])
    return stream_command(cmd, log)


def upscale(
    input_file: str,
    output_file: str | None = None,
    engine: str = "ffmpeg",
    model: str = AI_2X_MODEL,
    scale: int = 2,
    codec: str = "h264",
    quality: int = 19,
    overwrite: bool = False,
    enhance: bool = True,
    interp60: bool = False,
    target: str | None = None,
    log: Callable[[str], None] = print,
    progress: ProgressCallback | None = None,
) -> int:
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    target_width: int | None = None
    target_height: int | None = None
    if target:
        parts = target.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid target resolution: {target}. Use WxH format, e.g. 1920x1080.")
        try:
            target_width, target_height = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid target resolution: {target}. Use WxH format, e.g. 1920x1080.")

    if engine == "interp60":
        suffix = "60fps"
    elif engine == "ai":
        suffix = f"ai{scale}x"
        if target_width and target_height:
            suffix += f"_{target_height}p"
    elif enhance:
        suffix = "enhanced"
    else:
        suffix = "encoded"
    if interp60 and engine != "interp60":
        suffix += "_60fps"

    output_path = (
        Path(output_file).expanduser().resolve()
        if output_file
        else input_path.with_name(f"{input_path.stem}_{suffix}.mp4")
    )
    tools = inspect_tools()
    width, height, duration, fps = probe_video(tools.ffprobe, input_path)
    apply_interp60 = should_interpolate_to_60(interp60 or engine == "interp60", fps, log)
    spatial_output_path = make_work_video_path(output_path, "pre60") if apply_interp60 and engine != "interp60" else output_path

    if width and height:
        log(f"Input: {width}x{height}")
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

    if interp60 or engine == "interp60":
        if not tools.rife:
            raise RuntimeError(
                "RIFE was not found. Install rife-ncnn-vulkan to C:\\Tools\\rife-ncnn-vulkan "
                "or add rife-ncnn-vulkan.exe to PATH."
            )

    if engine == "ai":
        if not tools.realesrgan:
            raise RuntimeError("Real-ESRGAN was not found. Install realesrgan-ncnn-vulkan.")
        if not fps:
            fps = 30.0
            log("Warning: could not detect frame rate; using 30 fps.")
        return_code = run_ai_upscale(
            tools,
            input_path,
            spatial_output_path,
            model,
            scale,
            codec,
            quality,
            overwrite if spatial_output_path == output_path else True,
            enhance,
            fps,
            target_width,
            target_height,
            log,
            progress,
        )
    elif engine == "interp60":
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
            log("Input is already 60 fps. Copying streams.")
            report_progress("Copying existing 60 fps video", None, None, log, progress)
            return_code = stream_command(build_stream_copy_command(tools, input_path, output_path, overwrite), log)
    else:
        cmd = build_ffmpeg_command(
            tools,
            input_path,
            spatial_output_path,
            codec,
            quality,
            overwrite if spatial_output_path == output_path else True,
            enhance=enhance,
            target_width=target_width,
            target_height=target_height,
        )
        log("Running FFmpeg" + (" enhance pass:" if enhance else ":"))
        report_progress("Processing video", None, None, log, progress)
        return_code = stream_command(cmd, log)

    if apply_interp60 and engine != "interp60":
        try:
            if return_code == 0:
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
        finally:
            try:
                if spatial_output_path != output_path and spatial_output_path.exists():
                    spatial_output_path.unlink()
            except OSError as exc:
                log(f"Warning: could not remove intermediate video {spatial_output_path}: {exc}")

    if return_code == 0:
        report_progress("Done", None, None, log, progress)
        log(f"Done: {output_path}")
    else:
        log(f"Upscale failed with exit code {return_code}")
    return return_code
