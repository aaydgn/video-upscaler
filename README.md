# NVIDIA Video Upscaler

Small Python/Tkinter wrapper around FFmpeg for upscaling a 720p video to 1080p
with an RTX GPU. It uses CUDA/NPP scaling when your FFmpeg build supports it and
NVENC for the final encode.

## Requirements

- NVIDIA RTX GPU and current NVIDIA driver
- `uv`
- FFmpeg with NVENC support on `PATH`

This machine has been configured with:

- `uv`: `C:\Users\user\.local\bin`
- FFmpeg: `C:\Tools\ffmpeg\bin`
- GPU: NVIDIA GeForce RTX 5060 Ti

Both tool folders were added to the persistent user PATH.

## Install

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If this is your current PowerShell session immediately after installing `uv`,
run:

```powershell
$env:Path = "C:\Users\user\.local\bin;$env:Path"
```

Install FFmpeg with NVIDIA encoder support and add its `bin` directory to PATH
if you are setting this up on another machine. Recommended Windows builds are
from:

```text
https://www.gyan.dev/ffmpeg/builds/
```

After installation, restart PowerShell and check:

```powershell
uv run python upscaler.py --check
```

On this machine the check reports `scale_cuda: yes`, `h264_nvenc: yes`, and
`hevc_nvenc: yes`. `scale_cuda` or `scale_npp` is ideal, but the script can
still use NVENC encoding if only CPU scaling is available.

## UI

```powershell
.\run-ui.ps1
```

Choose the 720p input video, choose the output path, and click
`Upscale to 1080p`.

## CLI

```powershell
uv run python upscaler.py "C:\path\to\input_720p.mp4" --output "C:\path\to\output_1080p.mp4"
```

If Windows still has not picked up the PATH changes, run:

```powershell
.\setup-path.ps1
```

Useful options:

```powershell
uv run python upscaler.py input.mp4 --codec hevc --quality 18 --overwrite
```

Lower `--quality` values produce larger, higher-quality files. The default is
`19`, which is a good starting point for H.264 NVENC.

## Notes

- The target output is fixed at `1920x1080`.
- Audio and subtitles are copied when possible.
- If the input is not exactly `1280x720`, the script warns and still scales to
  1080p.
