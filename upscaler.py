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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


@dataclass(frozen=True)
class ToolInfo:
    ffmpeg: str
    ffprobe: str
    realesrgan: str | None
    has_cuda_scale: bool
    has_npp_scale: bool
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
        has_cuda_scale="scale_cuda" in filters,
        has_npp_scale="scale_npp" in filters,
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


def build_ffmpeg_command(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    codec: str,
    quality: int,
    overwrite: bool,
) -> list[str]:
    if codec == "hevc" and tools.has_hevc_nvenc:
        video_encoder = "hevc_nvenc"
    elif tools.has_h264_nvenc:
        video_encoder = "h264_nvenc"
    elif tools.has_hevc_nvenc:
        video_encoder = "hevc_nvenc"
    else:
        raise RuntimeError("This FFmpeg build does not expose h264_nvenc or hevc_nvenc.")

    if tools.has_cuda_scale:
        scale_filter = f"scale_cuda=w={TARGET_WIDTH}:h={TARGET_HEIGHT}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif tools.has_npp_scale:
        scale_filter = f"scale_npp={TARGET_WIDTH}:{TARGET_HEIGHT}:interp_algo=lanczos"
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        scale_filter = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos"
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


def upscale(
    input_file: str,
    output_file: str | None = None,
    engine: str = "ai",
    model: str = "realesrgan-x4plus",
    codec: str = "h264",
    quality: int = 19,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_path = (
        Path(output_file).expanduser().resolve()
        if output_file
        else input_path.with_name(f"{input_path.stem}_1080p.mp4")
    )
    tools = inspect_tools()
    width, height, duration, fps = probe_video(tools.ffprobe, input_path)
    if width and height:
        log(f"Input: {width}x{height}")
        if width != 1280 or height != 720:
            log("Warning: input is not exactly 1280x720; it will still be scaled to 1920x1080.")
    if duration:
        log(f"Duration: {duration:.1f}s")

    log(f"FFmpeg: {tools.ffmpeg}")
    if tools.realesrgan:
        log(f"Real-ESRGAN: {tools.realesrgan}")
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
    if engine == "ai":
        if not tools.realesrgan:
            raise RuntimeError("Real-ESRGAN was not found. Install realesrgan-ncnn-vulkan or use --engine ffmpeg.")
        if not fps:
            fps = 30.0
            log("Warning: could not detect frame rate; using 30 fps.")
        return_code = upscale_with_realesrgan(tools, input_path, output_path, model, codec, quality, overwrite, fps, log)
    else:
        cmd = build_ffmpeg_command(tools, input_path, output_path, codec, quality, overwrite)
        log("Running FFmpeg scaler:")
        return_code = stream_command(cmd, log)
    if return_code == 0:
        log(f"Done: {output_path}")
    else:
        log(f"FFmpeg failed with exit code {return_code}")
    return return_code


def upscale_with_realesrgan(
    tools: ToolInfo,
    input_path: Path,
    output_path: Path,
    model: str,
    codec: str,
    quality: int,
    overwrite: bool,
    fps: float,
    log: Callable[[str], None],
) -> int:
    assert tools.realesrgan is not None
    exe = Path(tools.realesrgan)
    temp_path = output_path.parent / f"{output_path.stem}_work_{uuid.uuid4().hex[:8]}"
    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        frames_dir = temp_path / "frames"
        ai_dir = temp_path / "ai_frames"
        frames_dir.mkdir()
        ai_dir.mkdir()

        log("Extracting frames...")
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
            return code

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
            "-f",
            "png",
        ]
        code = stream_command(ai_cmd, log, cwd=str(exe.parent))
        if code != 0:
            return code

        if codec == "hevc" and tools.has_hevc_nvenc:
            video_encoder = "hevc_nvenc"
        elif tools.has_h264_nvenc:
            video_encoder = "h264_nvenc"
        else:
            video_encoder = "libx264"

        log("Reassembling video and copying audio...")
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
            "-vf",
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos",
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
        return stream_command(assemble_cmd, log)
    finally:
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path)
        except OSError as exc:
            log(f"Warning: could not remove work folder {temp_path}: {exc}")


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("NVIDIA 720p to 1080p Upscaler")
    root.geometry("760x520")
    root.minsize(680, 460)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    engine_var = tk.StringVar(value="ai")
    model_var = tk.StringVar(value="realesrgan-x4plus")
    codec_var = tk.StringVar(value="h264")
    quality_var = tk.IntVar(value=19)
    overwrite_var = tk.BooleanVar(value=False)
    messages: queue.Queue[str] = queue.Queue()

    def choose_input() -> None:
        filename = filedialog.askopenfilename(
            title="Choose 720p video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            input_var.set(filename)
            path = Path(filename)
            output_var.set(str(path.with_name(f"{path.stem}_1080p.mp4")))

    def choose_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save 1080p video as",
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

    def drain_messages() -> None:
        try:
            while True:
                append_log(messages.get_nowait())
        except queue.Empty:
            pass
        root.after(100, drain_messages)

    def start() -> None:
        if not input_var.get():
            messagebox.showerror("Missing input", "Choose a 720p video first.")
            return
        start_button.configure(state="disabled")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")

        def worker() -> None:
            try:
                code = upscale(
                    input_var.get(),
                    output_var.get() or None,
                    engine_var.get(),
                    model_var.get(),
                    codec_var.get(),
                    quality_var.get(),
                    overwrite_var.get(),
                    messages.put,
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
    frame.rowconfigure(8, weight=1)

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

    ttk.Label(frame, text="Engine").grid(row=3, column=0, sticky="w", pady=4)
    engine_frame = ttk.Frame(frame)
    engine_frame.grid(row=3, column=1, sticky="w", padx=8)
    ttk.Radiobutton(engine_frame, text="AI Real-ESRGAN", value="ai", variable=engine_var).pack(side="left")
    ttk.Radiobutton(engine_frame, text="Fast FFmpeg scale", value="ffmpeg", variable=engine_var).pack(side="left", padx=16)

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

    ttk.Checkbutton(frame, text="Overwrite output if it exists", variable=overwrite_var).grid(
        row=6, column=1, sticky="w", padx=8, pady=4
    )

    start_button = ttk.Button(frame, text="Upscale to 1080p", command=start)
    start_button.grid(row=7, column=1, sticky="w", padx=8, pady=10)

    log_box = tk.Text(frame, height=14, state="disabled", wrap="word")
    log_box.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

    drain_messages()
    root.mainloop()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale a video to 1080p using NVIDIA FFmpeg acceleration.")
    parser.add_argument("input", nargs="?", help="Input video path. Omit to launch the Tkinter UI.")
    parser.add_argument("-o", "--output", help="Output video path. Defaults to <input>_1080p.mp4.")
    parser.add_argument("--engine", choices=["ai", "ffmpeg"], default="ai", help="Use AI Real-ESRGAN or fast FFmpeg scale.")
    parser.add_argument(
        "--model",
        choices=["realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime", "realesrnet-x4plus"],
        default="realesrgan-x4plus",
        help="Real-ESRGAN model to use with --engine ai.",
    )
    parser.add_argument("--codec", choices=["h264", "hevc"], default="h264", help="NVENC codec to use.")
    parser.add_argument("--quality", type=int, default=19, help="NVENC CQ value, lower is larger/better. Default: 19.")
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
        print(f"h264_nvenc: {'yes' if tools.has_h264_nvenc else 'no'}")
        print(f"hevc_nvenc: {'yes' if tools.has_hevc_nvenc else 'no'}")
        print(f"realesrgan: {tools.realesrgan or 'not found'}")
        return 0
    if not args.input:
        launch_gui()
        return 0
    return upscale(args.input, args.output, args.engine, args.model, args.codec, args.quality, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
