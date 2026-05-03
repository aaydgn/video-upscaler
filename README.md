# NVIDIA Video Upscaler

Small Python/Tkinter wrapper for upscaling a 720p video to 1080p with an RTX GPU.
By default it uses Real-ESRGAN AI frame upscaling, then FFmpeg/NVENC to rebuild
the video. A fast FFmpeg-only scaler is also available.

## Requirements

- NVIDIA RTX GPU and current NVIDIA driver
- `uv`
- FFmpeg with NVENC support on `PATH`
- Real-ESRGAN ncnn Vulkan, optional but required for AI upscaling

## Install

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If this is your current PowerShell session immediately after installing `uv`,
run:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
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

You should see `h264_nvenc: yes`. `scale_cuda` or `scale_npp` is ideal, but the
script can still use NVENC encoding if only CPU scaling is available.
For AI upscaling, `realesrgan` should show a path instead of `not found`.

## UI

```powershell
.\run-ui.ps1
```

From Command Prompt, or by double-clicking:

```bat
run-ui.bat
```

Choose the 720p input video, choose the output path, and click
`Upscale to 1080p`.

## CLI

```powershell
uv run python upscaler.py "C:\path\to\input_720p.mp4" --output "C:\path\to\output_1080p.mp4"
```

That uses the AI pipeline. It extracts frames, upscales them with Real-ESRGAN,
then reassembles the video with copied audio. It should take meaningfully longer
than a few seconds for anything except very short clips.
The UI and CLI report AI progress using completed frame counts, so the main
progress indicator will not reset when Real-ESRGAN's own percentage output does.

The batch launcher also forwards CLI arguments:

```bat
run-ui.bat "C:\path\to\input_720p.mp4" --output "C:\path\to\output_1080p.mp4"
```

If Windows still has not picked up the PATH changes, run:

```powershell
.\setup-path.ps1
```

To add `uv` and FFmpeg to the system PATH for all users, open PowerShell as
Administrator and run:

```powershell
.\setup-system-path.ps1
```

Useful options:

```powershell
uv run python upscaler.py input.mp4 --codec hevc --quality 18 --overwrite
```

Use the old fast scaler when you only want a resize:

```powershell
uv run python upscaler.py input.mp4 --engine ffmpeg
```

For animation/anime content, try:

```powershell
uv run python upscaler.py input.mp4 --model realesr-animevideov3
```

Lower `--quality` values produce larger, higher-quality files. The default is
`19`, which is a good starting point for H.264 NVENC.

## Notes

- The target output is fixed at `1920x1080`.
- AI mode upscales frames 2x first, then downsamples to 1080p.
- Audio and subtitles are copied when possible.
- If the input is not exactly `1280x720`, the script warns and still scales to
  1080p.
