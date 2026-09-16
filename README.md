# Video Upscaler

AI video upscaler with a Tkinter GUI. Supports two backends for
GPU-accelerated upscaling, FFmpeg enhancement filters, and RIFE
frame interpolation to 60 fps.

## Backends

| Backend | Speed | How |
|---|---|---|
| **ncnn** (default) | 28-41 fps | Shells out to `realesrgan-ncnn-vulkan`, Vulkan compute |
| **ONNX** (fallback) | 10-20 fps | In-process via ONNX Runtime + DirectML, zero disk I/O |

Auto mode uses ncnn when installed, falls back to ONNX.

## ONNX Models

Installed to `~/.video-upscaler/models/`. Convert any `.pth` or
`.safetensors` model to ONNX with `python scripts/convert_to_onnx.py`.

| Model | Scale | Size | Architecture |
|---|---|---|---|
| 2xHFA2kSPAN | 2x | 1.6 MB | SPAN |
| 4xNomosUni_span_multijpg | 4x | 1.6 MB | SPAN |
| realesr-animevideov3 | 4x | 2.4 MB | SRVGGNetCompact |
| RealESRGAN_x4plus | 4x | 64 MB | RRDBNet |

## Requirements

- NVIDIA GPU with current driver
- [uv](https://docs.astral.sh/uv/)
- FFmpeg with NVENC support on PATH
- Real-ESRGAN ncnn Vulkan (optional, for ncnn backend)
- RIFE ncnn Vulkan (optional, for 60 fps interpolation)

## Install

```powershell
uv sync
```

Check tools:

```powershell
uv run python -m upscaler --check
```

## GUI

```powershell
.\run-ui.ps1
```

Or from Command Prompt:

```bat
run-ui.bat
```

Controls:
- **Backend** — Auto / ONNX / ncnn
- **AI upscale** — Off / 2x / 4x
- **AI model** — per-backend model selection
- **Enhance** — deblock, denoise pre-filters
- **60 fps** — RIFE GPU interpolation
- **Codec** — H.264 or HEVC (NVENC)
- **Quality** — CQ value (lower = better quality, larger file)

Settings persist across sessions.

## CLI

```powershell
# AI upscale 4x with ncnn (fastest)
uv run python -m upscaler input.mp4 --scale 4

# AI upscale 2x with ONNX backend
uv run python -m upscaler input.mp4 --scale 2 --backend onnx

# Enhance only (deblock + denoise + sharpen, no upscaling)
uv run python -m upscaler input.mp4

# Skip enhance filters
uv run python -m upscaler input.mp4 --scale 4 --no-enhance

# Interpolate to 60 fps
uv run python -m upscaler input.mp4 --interp60

# Combine: upscale + interpolate
uv run python -m upscaler input.mp4 --scale 4 --interp60

# HEVC codec, quality 18
uv run python -m upscaler input.mp4 --scale 4 --codec hevc --quality 18

# Use a specific ONNX model
uv run python -m upscaler input.mp4 --scale 4 --backend onnx --onnx-model path/to/model.onnx

# Tile-based processing for low VRAM
uv run python -m upscaler input.mp4 --scale 4 --backend onnx --tile-size 512
```

## Project Structure

```
upscaler/
  __init__.py        # exports upscale()
  __main__.py        # entry point
  cli.py             # argparse CLI
  gui.py             # Tkinter UI
  config.py          # constants and type aliases
  engines.py         # upscale dispatch, ncnn pipeline
  pipeline.py        # ONNX + FFmpeg pipe pipeline
  onnx_upscale.py    # ONNX Runtime session and inference
  models.py          # ONNX model registry and download
  ffmpeg.py          # FFmpeg command building
  process.py         # subprocess helpers
  tools.py           # tool discovery (FFmpeg, ESRGAN, RIFE)
scripts/
  convert_to_onnx.py # convert .pth/.safetensors to ONNX
```
