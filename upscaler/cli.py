from __future__ import annotations

import argparse
import sys

from upscaler.config import AI_2X_MODEL
from upscaler.engines import upscale
from upscaler.models import DEFAULT_ONNX_MODEL
from upscaler.tools import inspect_tools


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale, enhance, or interpolate video with FFmpeg and NVIDIA encoding.")
    parser.add_argument("input", nargs="?", help="Input video path. Omit to launch the GUI.")
    parser.add_argument("-o", "--output", help="Output video path.")
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=1, help="AI upscale factor. 1 = off, 2x or 4x upscale.")
    parser.add_argument(
        "--model",
        choices=["realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime", "realesrnet-x4plus"],
        default=AI_2X_MODEL,
        help="Real-ESRGAN model (default auto-selected per scale).",
    )
    parser.add_argument("--codec", choices=["h264", "hevc"], default="h264", help="NVENC codec.")
    parser.add_argument("--quality", type=int, default=19, help="NVENC CQ value, lower is larger/better.")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=True, help="Apply deblock/denoise/sharpen filters (default: on).")
    parser.add_argument("--interp60", action="store_true", help="Interpolate the final output to exact 60 fps.")
    parser.add_argument("--target", help="Target output resolution, e.g. 1920x1080. Downscales after processing.")
    parser.add_argument("--backend", choices=["auto", "onnx", "ncnn"], default="auto", help="AI upscale backend: onnx (in-process, no disk I/O), ncnn (realesrgan-ncnn-vulkan), auto (onnx if available, else ncnn).")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="ONNX model name or path to .onnx file.")
    parser.add_argument("--tile-size", type=int, default=0, help="ONNX tile size (0 = whole frame, e.g. 512 for low VRAM).")
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
        try:
            import onnxruntime as ort
            print(f"onnxruntime: {ort.__version__}")
            print(f"onnx_providers: {', '.join(ort.get_available_providers())}")
        except ImportError:
            print("onnxruntime: not installed")
        return 0
    if not args.input:
        from upscaler.gui import launch_gui

        launch_gui()
        return 0

    engine = "ai" if args.scale > 1 else "ffmpeg"
    try:
        return upscale(
            args.input,
            args.output,
            engine,
            args.model,
            args.scale,
            args.codec,
            args.quality,
            args.overwrite,
            args.enhance,
            args.interp60,
            args.target,
            backend=args.backend,
            onnx_model=args.onnx_model,
            tile_size=args.tile_size,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
